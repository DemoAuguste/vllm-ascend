# -*- coding: UTF-8 -*-
"""Qwen3-4B 的 SLQ 流水线（WikiText-2 校准，组级敏感度）：

  1) 逐层 GPTQ（非对称 INT g128，单 Hessian 多档位）→ 权重量 store
  2) 以 uniform-min-bit 为基准，测**组级**（qkv / o / gate+up / down）切换到各位宽的 top-10 KL 边际变化
  3) 参数量加权 ILP 分配
  4) 应用分配并保存 fake-quant checkpoint

用法（2 卡）：
  ASCEND_RT_VISIBLE_DEVICES=5,6 python run_slq_4b.py --model Qwen/Qwen3-4B \
      --device npu:0 --device-map split --device-map-split 2 --bits 4,6 \
      --target-avg 4.5 --asym --group-size 128 \
      --reuse-ckpt4 /opt/zh/SLQ-gptq/ckpt_4b_uniform4_g128_asym/model.safetensors \
      --save-dir /opt/zh/SLQ-gptq/ckpt_4b_slq45 --out /opt/zh/SLQ-gptq/slq_4b_45.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from gptq import GPTQ
from model_quant import (build_calibration_batches, capture_inputs, load_model, log,
                         target_linear_names)
from slq.ilp import lagrangian_allocate
from slq.sensitivity import LayerSensitivity, make_eval_batches

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BPP_OVERHEAD = 0.156   # g128 scale/zero 开销


@torch.no_grad()
def topk_ref(model, batches, k=10):
    out = []
    for b in batches:
        logits = model(b).logits.float()
        v, idx = logits.topk(k, dim=-1)
        out.append((idx.cpu(), torch.softmax(v, dim=-1).cpu()))
    return out


@torch.no_grad()
def topk_kl(model, batches, ref):
    tot, n = 0.0, 0
    for b, (idx, p_ref) in zip(batches, ref):
        logits = model(b).logits.float()
        q = logits.gather(-1, idx.to(logits.device))
        q_logp = torch.log_softmax(q, dim=-1)
        p = p_ref.to(logits.device)
        kl = (p * (torch.log(p + 1e-12) - q_logp)).sum(dim=-1).mean()
        tot += float(kl)
        n += 1
    return tot / max(1, n)


def group_of(name: str) -> str:
    parts = name.split(".")
    layer = parts[2]
    proj = parts[-1]
    if "self_attn" in name or "linear_attn" in name:
        grp = "qkv" if proj in ("q_proj", "k_proj", "v_proj", "in_proj_qkv") else "o"
    else:
        grp = "upgate" if proj in ("gate_proj", "up_proj") else "down"
    return f"{layer}.{grp}"


def save_fake_quant(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    from accelerate import Accelerator

    acc = Accelerator()
    unwrapped = acc.unwrap_model(model)
    sd = acc.get_state_dict(model)
    unwrapped.save_pretrained(save_dir, state_dict=sd, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    log(f"checkpoint 已保存：{save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="split")
    ap.add_argument("--device-map-split", type=int, default=2)
    ap.add_argument("--bits", default="4,6")
    ap.add_argument("--target-avg", type=float, default=4.5)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--asym", action="store_true", default=True)
    ap.add_argument("--only", default="all", choices=["all", "mlp", "attn"])
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-batch", type=int, default=4)
    ap.add_argument("--calib-dataset", default="wikitext", choices=["wikitext", "fineweb"])
    ap.add_argument("--sens-windows", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--reuse-ckpt4", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--out", default="slq_4b.json")
    ap.add_argument("--sens-out", default=None)
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()

    bits = sorted(int(b) for b in args.bits.split(","))
    low_bit = bits[0]

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    model, tokenizer, device = load_model(args)
    names = target_linear_names(model, args.only)
    if args.limit_layers > 0:
        names = names[: args.limit_layers]
        if args.save_dir:
            log("[warn] limit-layers 模式下不保存 checkpoint")
            args.save_dir = None
    params = {n: int(model.get_submodule(n).weight.numel()) for n in names}
    log(f"层数 {len(names)}，候选位宽 {bits}，目标平均 {args.target_avg} bit")

    batches = build_calibration_batches(
        tokenizer, args.calib_seqs, args.calib_len, args.calib_batch, device,
        None, args.calib_dataset)
    inputs = capture_inputs(model, names, batches)

    eval_batches = make_eval_batches(tokenizer, text, device, args.sens_windows,
                                      args.max_length, args.stride)
    log(f"[KL] {args.sens_windows} 窗口 × {args.max_length}，top-{args.topk}")
    ref = topk_ref(model, eval_batches, args.topk)

    reuse = None
    reuse_keys = set()
    if args.reuse_ckpt4 and 4 in bits and 4 == low_bit:
        from safetensors import safe_open
        reuse = safe_open(args.reuse_ckpt4, framework="pt")
        reuse_keys = set(reuse.keys())
        log(f"复用 b=4 权重：{args.reuse_ckpt4}（{len(reuse_keys)} keys）")

    # ---------- 1) 权重量 store ----------
    store, orig = {}, {}
    t_all = time.time()
    for i, name in enumerate(names):
        module = model.get_submodule(name)
        W = module.weight.data
        Wf = W.float()
        dev = W.device
        orig[name] = W
        store[name] = {}
        need = [b for b in bits if not (b == 4 and reuse is not None and 4 == low_bit)]
        g = None
        if need:
            X = inputs[name].to(dev)
            g = GPTQ(tuple(Wf.shape), bits=need[0], group_size=args.group_size,
                     asym=args.asym)
            g.observe(X)
            del X
        for b in bits:
            if b == 4 and reuse is not None and 4 == low_bit:
                key = name if name in reuse_keys else f"{name}.weight"
                Wq = reuse.get_tensor(key).to(torch.bfloat16)
            else:
                Wq = g.quantize(Wf, bits=b).to(torch.bfloat16)
            store[name][b] = Wq.cpu()
        del g
        if dev.type == "npu":
            torch.npu.empty_cache()
        if (i + 1) % 20 == 0 or i + 1 == len(names):
            log(f"  [store {i+1}/{len(names)}] {name}")

    # ---------- 2) 组级敏感度（以 low_bit 全模型为基准） ----------
    groups = {}
    for n in names:
        groups.setdefault(group_of(n), []).append(n)
    log(f"组数 {len(groups)}（每组 {{qkv,o,gate+up,down}}）")

    def set_group(gname, b):
        for n in groups[gname]:
            mod = model.get_submodule(n)
            mod.weight.data = store[n][b].to(mod.weight.data.device)

    for n in names:                      # 基准：全部 low_bit
        mod = model.get_submodule(n)
        mod.weight.data = store[n][low_bit].to(mod.weight.data.device)
    kl_base = topk_kl(model, eval_batches, ref)
    log(f"[KL] uniform-{low_bit} 基准 KL = {kl_base:.4e}")

    sens = []
    t_sens = time.time()
    for gi, (gname, members) in enumerate(groups.items()):
        vals = {low_bit: kl_base}
        for b in bits[1:]:
            set_group(gname, b)
            vals[b] = topk_kl(model, eval_batches, ref)
            set_group(gname, low_bit)
        sens.append(LayerSensitivity(name=gname, metric="kl", value_by_bits=vals,
                                     alpha_by_bits=dict(vals)))
        log(f"  [kl {gi+1}/{len(groups)}] {gname:<16} " +
            " ".join(f"b{b}:{vals[b]:.4e}" for b in bits))
    log(f"组级敏感度完成，耗时 {time.time()-t_sens:.1f}s")

    # ---------- 3) 参数量加权 ILP ----------
    gweights = [sum(params[n] for n in groups[g.name]) for g in sens]
    assign, avg, total = lagrangian_allocate(sens, bits, args.target_avg, weights=gweights)
    dist = {b: sum(1 for v in assign.values() if v == b) for b in bits}
    total_params = sum(gweights)
    weighted = sum(w * assign[g.name] for g, w in zip(sens, gweights)) / total_params
    log(f"\nILP：加权平均 {weighted:.3f} bit（目标 {args.target_avg}），"
        f"含开销 {weighted + BPP_OVERHEAD:.3f} bpp；组位宽分布 {dist}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(assign, fh, indent=1, ensure_ascii=False)
    if args.sens_out:
        json.dump({s.name: s.value_by_bits for s in sens},
                  open(args.sens_out, "w", encoding="utf-8"), indent=1)

    # ---------- 4) 应用 + 保存 ----------
    for gname, b in assign.items():
        set_group(gname, b)
    if args.save_dir:
        save_fake_quant(model, tokenizer, args.save_dir)

    log(f"完成，耗时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
