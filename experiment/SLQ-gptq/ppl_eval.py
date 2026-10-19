# -*- coding: UTF-8 -*-
"""WikiText-2 PPL：BF16/FP32 基线 vs GPTQ fake-quant（统一位宽或逐层位宽表）。"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from model_quant import (build_calibration_batches, capture_inputs, load_model, log,
                         quantize_model, resolve_device, target_linear_names)

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@torch.no_grad()
def compute_ppl(model, tokenizer, text, device, max_length=1024, stride=512):
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
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
    return float(torch.exp(torch.stack(nlls).mean()).item()), len(nlls)


def main():
    ap = argparse.ArgumentParser(description="GPTQ fake-quant PPL 评测")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--bit-map", default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--only", default="all", choices=["all", "mlp", "attn"])
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-batch", type=int, default=4)
    ap.add_argument("--calib-text", default=None)
    ap.add_argument("--calib-dataset", default="wikitext", choices=["wikitext", "fineweb"])
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--device-map-split", type=int, default=2)
    ap.add_argument("--max-memory", default=None)
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    log("加载 WikiText-2 (test) ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    t0 = time.time()
    model, tokenizer, device = load_model(args)
    log(f"模型加载完成 {time.time()-t0:.1f}s")

    if args.skip_baseline:
        base_ppl, n_win = float("nan"), 0
        log("[基线] 跳过")
    else:
        log("[基线] 计算 PPL ...")
        t0 = time.time()
        base_ppl, n_win = compute_ppl(model, tokenizer, text, device, args.max_length, args.stride)
        log(f"基线 PPL = {base_ppl:.4f}（{n_win} windows, {time.time()-t0:.1f}s）")

    names = target_linear_names(model, args.only)
    if args.limit_layers > 0:
        names = names[: args.limit_layers]

    if args.bit_map:
        with open(args.bit_map, "r", encoding="utf-8") as fh:
            bits_map = {k: int(v) for k, v in json.load(fh).items()}
        avg = sum(bits_map.get(n, bits_map.get("__default__", args.bits)) for n in names) / max(1, len(names))
    else:
        bits_map = {"__default__": args.bits}
        avg = float(args.bits)
    log(f"\n[GPTQ] 量化 {len(names)} 层（bits={args.bits}, avg={avg:.3f}, "
        f"group_size={args.group_size}, only={args.only}）...")

    batches = build_calibration_batches(
        tokenizer, args.calib_seqs, args.calib_len, args.calib_batch, device,
        args.calib_text, args.calib_dataset
    )
    inputs = capture_inputs(model, names, batches)
    quantize_model(model, names, inputs, bits_map, args.group_size, device, asym=args.asym)

    log("\n[GPTQ] 计算量化后 PPL ...")
    t0 = time.time()
    quant_ppl, _ = compute_ppl(model, tokenizer, text, device, args.max_length, args.stride)
    log(f"量化 PPL = {quant_ppl:.4f}（{time.time()-t0:.1f}s）")

    log("\n" + "=" * 60)
    if base_ppl != base_ppl:  # NaN → 跳过基线
        log(f"GPTQ PPL : {quant_ppl:.4f}  (平均 {avg:.3f} bpw)")
    else:
        log(f"基线 PPL : {base_ppl:.4f}")
        log(f"GPTQ PPL : {quant_ppl:.4f}  (平均 {avg:.3f} bpw)")
        log(f"相对增加 : {(quant_ppl - base_ppl) / base_ppl * 100:.2f}%")
    log("=" * 60)


if __name__ == "__main__":
    main()
