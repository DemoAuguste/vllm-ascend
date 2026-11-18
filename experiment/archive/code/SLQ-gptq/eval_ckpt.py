# -*- coding: UTF-8 -*-
"""对（fake-quant）checkpoint 或基础模型直接计算 WikiText-2 PPL（不再量化）。

用法：
  python eval_ckpt.py --model /opt/zh/SLQ-gptq/ckpt_4b_uniform4_g128_asym --device npu:0
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import load_dataset  # noqa: E402

from model_quant import load_model, log  # noqa: E402
from ppl_eval import compute_ppl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--device-map-split", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    t0 = time.time()
    model, tokenizer, device = load_model(args)
    log(f"模型加载完成 {time.time()-t0:.1f}s")
    t0 = time.time()
    ppl, n = compute_ppl(model, tokenizer, text, device, args.max_length, args.stride)
    log(f"PPL = {ppl:.4f}（{n} windows, {time.time()-t0:.1f}s）  model={args.model}")


if __name__ == "__main__":
    main()
