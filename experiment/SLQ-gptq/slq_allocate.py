# -*- coding: UTF-8 -*-
"""SLQ 混合精度分配：GPTQ 逐层多档位量化 → KL 敏感度 → ILP 位宽分配 → bit_map。

用法：
  # 测量敏感度并分配
  python slq_allocate.py --device npu --model Qwen/Qwen3-0.6B --bits 3,4,8 \
      --target-avg 3.5 --group-size 128 --out bits.json --sens-out sens.json

  # 复用已测敏感度（或从日志解析）快速重分配（无需模型/NPU）
  python slq_allocate.py --sens-json sens.json --bits 3,4,8 --target-avg 3.25 --out bits2.json
  python slq_allocate.py --sens-log slq_alloc.log --bits 3,4,8 --target-avg 3.5 --out bits.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from slq.ilp import lagrangian_allocate
from slq.sensitivity import LayerSensitivity, parse_sens_log

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_sens(args):
    if args.sens_json:
        raw = json.load(open(args.sens_json, "r", encoding="utf-8"))
    elif args.sens_log:
        raw = parse_sens_log(args.sens_log)
    else:
        return None
    out = []
    for name, vals in raw.items():
        v = {int(k): float(x) for k, x in vals.items()}
        out.append(LayerSensitivity(name=name, metric="kl", value_by_bits=v,
                                    alpha_by_bits=dict(v)))
    return out


def measure_sens(args, bits_list, device):
    from gptq import GPTQ
    from model_quant import (build_calibration_batches, capture_inputs, log,
                             target_linear_names)
    from slq.sensitivity import make_eval_batches, mean_kl, model_logprobs

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("加载 WikiText-2 ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype))
    model.to(device).eval()

    names = target_linear_names(model, args.only)
    if args.limit_layers > 0:
        names = names[: args.limit_layers]

    batches = build_calibration_batches(
        tokenizer, args.calib_seqs, args.calib_len, args.calib_batch, device)
    inputs = capture_inputs(model, names, batches)

    log(f"[KL] 评估窗口 {args.sens_windows} × {args.max_length}，计算 FP 参考分布 ...")
    eval_batches = make_eval_batches(tokenizer, text, device, args.sens_windows,
                                      args.max_length, args.stride)
    ref_lp = model_logprobs(model, eval_batches)

    sens, t_all = [], time.time()
    for i, name in enumerate(names):
        module = model.get_submodule(name)
        W = module.weight.data.float()
        X = inputs[name].to(device)
        g = GPTQ(tuple(W.shape), bits=bits_list[0], group_size=args.group_size)
        g.observe(X)
        vals = {}
        for b in bits_list:
            module.weight.data = g.quantize(W, bits=b).to(module.weight.dtype)
            vals[b] = mean_kl(model, eval_batches, ref_lp)
        module.weight.data = W.to(module.weight.dtype)
        sens.append(LayerSensitivity(name=name, metric="kl", value_by_bits=vals,
                                     alpha_by_bits=dict(vals)))
        log(f"  [{i+1}/{len(names)}] {name:<50} " +
            " ".join(f"b{b}:{vals[b]:.3e}" for b in bits_list))
        del X, g
        if device.type == "npu":
            torch.npu.empty_cache()
    log(f"敏感度测量完成，耗时 {time.time()-t_all:.1f}s")
    return names, sens


def main():
    ap = argparse.ArgumentParser(description="SLQ 混合精度位宽分配（KL 敏感度）")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--bits", default="3,4,8")
    ap.add_argument("--target-avg", type=float, default=3.5)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--only", default="all", choices=["all", "mlp", "attn"])
    ap.add_argument("--calib-seqs", type=int, default=8)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-batch", type=int, default=4)
    ap.add_argument("--sens-windows", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--out", default="bits.json")
    ap.add_argument("--sens-json", default=None, help="复用已测敏感度 JSON，跳过测量")
    ap.add_argument("--sens-log", default=None, help="从运行日志解析敏感度，跳过测量")
    ap.add_argument("--sens-out", default=None, help="将本次測得的敏感度保存为 JSON")
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()

    bits_list = sorted(int(b) for b in args.bits.split(","))

    sens = load_sens(args)
    if sens is None:
        from model_quant import log, resolve_device
        device = resolve_device(args.device)
        names, sens = measure_sens(args, bits_list, device)
        if args.sens_out:
            json.dump({s.name: s.value_by_bits for s in sens},
                      open(args.sens_out, "w", encoding="utf-8"), indent=1)
    else:
        from model_quant import log
        names = [s.name for s in sens]
        log(f"复用敏感度：{len(sens)} 层")

    if args.limit_layers > 0:
        names = names[: args.limit_layers]
        sens = [s for s in sens if s.name in set(names)]

    assign, avg, total = lagrangian_allocate(sens, bits_list, args.target_avg)
    dist = {b: sum(1 for v in assign.values() if v == b) for b in bits_list}
    log(f"层数 {len(assign)}，候选 {bits_list}，目标 {args.target_avg} bpw")
    log(f"ILP 分配完成：平均 {avg:.3f} bpw，总 KL 成本 {total:.4e}")
    log(f"位宽分布：{dist}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(assign, fh, indent=1, ensure_ascii=False)
    log(f"bit_map 已写入 {args.out}")


if __name__ == "__main__":
    main()
