"""Qwen3-0.6B WikiText-2 PPL 评测：BF16 基线 vs PCDVQ fake-quant。

单脚本对比：加载原始模型算基线 PPL → 逐层 PCDVQ 量化 → 再算量化 PPL。
（避免 checkpoint 里 remote-code 依赖问题，量化逻辑直接复用 model_quant。）

运行示例：
  ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --a 14 --device npu
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch_npu  # noqa: F401

from model_quant import build_codebooks, load_codebooks_from_cache, quantize_model

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def log(*a):
    print(*a, flush=True)


def compute_ppl(model, tokenizer, text, device, max_length=1024, stride=512):
    """固定长度滑窗 NLL，返回 PPL（HF 文档标准实现）。"""
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    if seq_len < 2:
        return float("nan")
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss)
        prev_end = end
        if end == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl, len(nlls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--a", type=int, default=14)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="npu")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--cache", default="/opt/zh/q38quant/codebooks/pcdvq_k8_a14_b2.pt")
    ap.add_argument("--only", default="all", choices=["all", "mlp", "attn"],
                    help="只量化指定层组（all/mlp/attn），用于消融")
    args = ap.parse_args()

    device = "npu:0" if args.device == "npu" else "cpu"
    if args.device == "npu":
        torch.npu.set_device(0)

    # 数据集
    from datasets import load_dataset
    log("加载 WikiText-2 (test)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    log(f"评测文本: {len(text)} 字符")

    # 模型 + tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.to(device)
    model.eval()

    # 基线 PPL
    log("\n[基线] 计算 FP32 PPL ...")
    t0 = time.time()
    base_ppl, n_win = compute_ppl(model, tokenizer, text, device, args.max_length, args.stride)
    log(f"基线 PPL = {base_ppl:.4f}  ({n_win} windows, {time.time()-t0:.1f}s)")

    # 量化
    log(f"\n[量化] PCDVQ a={args.a} b={args.b} (BPW={(args.a+args.b)/args.k}) ...")
    if args.a == 14 and args.b == 2 and args.k == 8 and device == "npu:0":
        qz = load_codebooks_from_cache(args.cache, device)
    else:
        qz = build_codebooks(args.k, args.a, args.b, device)
    t0 = time.time()
    only_map = {"mlp": ("mlp",), "attn": ("self_attn",), "all": None}
    stats, dt, n_skip = quantize_model(model, qz, chunk=args.chunk, only_kw=only_map[args.only])
    log(f"量化完成: {len(stats)} 层, 跳过 {n_skip} 层, 耗时 {dt:.1f}s (only={args.only})")
    if stats:
        mean_rel = sum(r for _, _, r in stats) / len(stats)
        log(f"平均相对 Frobenius 误差: {mean_rel:.4f}")

    # 量化 PPL
    log("\n[量化] 计算 PPL ...")
    t0 = time.time()
    quant_ppl, n_win2 = compute_ppl(model, tokenizer, text, device, args.max_length, args.stride)
    log(f"量化 PPL = {quant_ppl:.4f}  ({n_win2} windows, {time.time()-t0:.1f}s)")

    log("\n" + "=" * 60)
    log(f"基线 PPL     : {base_ppl:.4f}")
    log(f"PCDVQ a={args.a} PPL: {quant_ppl:.4f}")
    log(f"相对增加     : {(quant_ppl - base_ppl) / base_ppl * 100:.2f}%")
    log("=" * 60)


if __name__ == "__main__":
    main()