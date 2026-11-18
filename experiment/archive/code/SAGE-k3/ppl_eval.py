"""Qwen3-0.6B（或任意 HF 因果 LM）WikiText-2 PPL：BF16/FP32 基线 vs SAGE fake-quant。

量化逻辑复用 `model_quant`，本脚本只负责：基线 PPL → SAGE 量化 → 量化后 PPL。
用 --splits 指定单切分可跑固定位宽（如 64,2 → 2.0bpw）；默认跑 ABS 三切分。

运行示例：
  ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu --splits 64,2 --epochs 20
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from sage import BlockwiseConfig, SAGEConfig
from sage.utils import log, resolve_device
from model_quant import build_calibration_batches, parse_splits, quantize_model, target_linear_names

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@torch.no_grad()
def compute_ppl(model, tokenizer, text, device, max_length: int = 1024, stride: int = 512):
    """固定长度滑窗 NLL（HF 标准实现）。"""
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    if seq_len < 2:
        return float("nan"), 0
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss)
        prev_end = end
        if end == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl, len(nlls)


def main() -> None:
    ap = argparse.ArgumentParser(description="SAGE fake-quant PPL 评测")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--splits", default="64,4;128,3;256,2")
    ap.add_argument("--no-esg", action="store_true")
    ap.add_argument("--no-adaptive-bits", action="store_true")
    ap.add_argument("--kmeans-iters", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-batch", type=int, default=4)
    ap.add_argument("--only", default="all", choices=["all", "mlp", "attn"])
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("加载 WikiText-2 (test) ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype)
    )
    model.to(device).eval()

    log("[基线] 计算 PPL ...")
    t0 = time.time()
    base_ppl, n_win = compute_ppl(model, tokenizer, text, device, args.max_length, args.stride)
    log(f"基线 PPL = {base_ppl:.4f}（{n_win} windows, {time.time()-t0:.1f}s）")

    names = target_linear_names(model, args.only)
    if args.limit_layers > 0:
        names = names[: args.limit_layers]
    batches, _ = build_calibration_batches(
        tokenizer, args.calib_seqs, args.calib_len, args.calib_batch, device
    )
    model._sage_batches = batches  # noqa: SLF001

    config = SAGEConfig(
        v=8, group_size=128, splits=parse_splits(args.splits), esg=not args.no_esg,
        adaptive_bits=not args.no_adaptive_bits, kmeans_iters=args.kmeans_iters, seed=args.seed,
    )
    blockwise = BlockwiseConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                seed=args.seed) if args.epochs > 0 else None

    log(f"\n[SAGE] 量化 {len(names)} 层（splits={args.splits}, esg={config.esg}, "
        f"abs={config.adaptive_bits}, epochs={args.epochs}, sequential={args.sequential}）...")
    quantize_model(model, config, blockwise, names, device, args.sequential)

    log("\n[SAGE] 计算量化后 PPL ...")
    t0 = time.time()
    quant_ppl, _ = compute_ppl(model, tokenizer, text, device, args.max_length, args.stride)
    log(f"量化 PPL = {quant_ppl:.4f}（{time.time()-t0:.1f}s）")

    log("\n" + "=" * 60)
    log(f"基线 PPL : {base_ppl:.4f}")
    log(f"SAGE PPL : {quant_ppl:.4f}")
    log(f"相对增加 : {(quant_ppl - base_ppl) / base_ppl * 100:.2f}%")
    log("=" * 60)


if __name__ == "__main__":
    main()
