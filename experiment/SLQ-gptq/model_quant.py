# -*- coding: UTF-8 -*-
"""全模型 GPTQ fake-quant（CPU/NPU 通用）。

流程：
  1. 加载 HF 因果 LM，选取目标 Linear 层（默认除 lm_head/embed 外的所有投影）。
  2. 一次校准前向：forward hook 缓存每层输入激活（CPU）。
  3. 逐层 GPTQ：由该层激活累积 Hessian → 列式二阶误差补偿量化权重 → 回写。
  4. 支持统一位宽 `--bits` 或逐层位宽表 `--bit-map`（JSON，SLQ 分配结果）。

运行示例：
  python model_quant.py --device cpu --model Qwen/Qwen3-0.6B --bits 4
  python model_quant.py --device npu --model Qwen/Qwen3-0.6B --bits 4 --group-size 128
  python model_quant.py --device npu --model Qwen/Qwen3-0.6B --bit-map bits.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from gptq import GPTQ

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ATTN_KEYWORDS = ("self_attn", "linear_attn", "attention")
SKIP_KEYWORDS = ("lm_head", "embed_tokens", "embed")


def log(*args):
    print(*args, flush=True)


def resolve_device(spec: str) -> torch.device:
    spec = str(spec)
    if spec.startswith("npu"):
        import torch_npu  # noqa: F401
        return torch.device(spec if ":" in spec else "npu:0")
    return torch.device(spec)


def build_split_device_map(model_id: str, n: int) -> dict:
    """按层把模型均分到 n 张卡，返回显式 device_map（确定性分片）。"""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    sub = getattr(cfg, "text_config", cfg)
    L = getattr(sub, "num_hidden_layers", None)
    if L is None:
        raise ValueError("无法从 config 获取 num_hidden_layers，无法自动分片")
    per = (L + n - 1) // n
    dmap = {"model.embed_tokens": 0}
    for i in range(L):
        dmap[f"model.layers.{i}"] = min(i // per, n - 1)
    dmap["model.norm"] = n - 1
    dmap["lm_head"] = 0        # 与 embed_tokens 同卡，避免 tie_weights 跨设备比较
    log(f"显式分片：{n} 卡 × 约 {per} 层（共 {L} 层）")
    return dmap


def _resolve_model_class(model_path: str):
    """按 config.json 的 architectures 选择模型类（VLM checkpoint 不能用 AutoModelForCausalLM，
    否则参数前缀不匹配会静默加载随机权重）。"""
    import transformers

    archs = []
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            archs = json.load(f).get("architectures") or []
    for name in archs:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    return transformers.AutoModelForCausalLM


def load_model(args):
    """按单卡或多卡（device_map）加载模型，返回 (model, tokenizer, input_device)。"""
    from transformers import AutoTokenizer

    model_cls = _resolve_model_class(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kw = dict(trust_remote_code=True, dtype=getattr(torch, args.dtype))
    dm = getattr(args, "device_map", None)
    if dm:
        n = int(getattr(args, "device_map_split", 2) or 2)
        if dm == "split":
            dmap = build_split_device_map(args.model, n)
        else:
            dmap = dm
        max_memory = {i: "30GiB" for i in range(n)}
        max_memory["cpu"] = "200GiB"
        model = model_cls.from_pretrained(
            args.model, device_map=dmap, max_memory=max_memory, **kw)
        device = resolve_device(args.device)          # 输入落到首个卡（embed 所在）
        from collections import Counter
        dist = Counter(str(v) for v in model.hf_device_map.values())
        log(f"多卡加载完成 device_map={dm} 分布={dict(dist)}，input_device={device}")
    else:
        device = resolve_device(args.device)
        model = model_cls.from_pretrained(args.model, **kw)
        model.to(device)
    model.eval()
    return model, tokenizer, device


def target_linear_names(model, only: str = "all") -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(k in name for k in SKIP_KEYWORDS):
            continue
        if only == "mlp" and "mlp" not in name:
            continue
        if only == "attn" and not any(k in name for k in ATTN_KEYWORDS):
            continue
        names.append(name)
    return names


def build_calibration_batches(tokenizer, n_seqs, seq_len, batch, device, text_path=None,
                              dataset="wikitext"):
    if text_path:
        with open(text_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        from datasets import load_dataset

        if dataset == "fineweb":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            chunks, total = [], 0
            for row in ds:
                t = (row.get("text") or "").strip()
                if t:
                    chunks.append(t)
                    total += len(t)
                if total > n_seqs * seq_len * 6:
                    break
            text = "\n\n".join(chunks)
        else:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    need = n_seqs * seq_len
    if ids.numel() < need:
        raise ValueError(f"校准文本过短：需要 {need} token，仅 {ids.numel()}")
    ids = ids[:need].reshape(n_seqs, seq_len)
    batches = [{"input_ids": ids[i:i + batch].to(device)} for i in range(0, n_seqs, batch)]
    return batches


@torch.no_grad()
def capture_inputs(model, names, batches):
    store = {n: [] for n in names}
    hooks = []
    for n in names:
        module = model.get_submodule(n)

        def pre_hook(mod, args, n=n):
            store[n].append(args[0].detach().to("cpu", copy=True))

        hooks.append(module.register_forward_pre_hook(pre_hook))
    for b in batches:
        model(**b)
    for h in hooks:
        h.remove()
    return {n: torch.cat(store[n], dim=0) for n in names}


def quantize_model(model, names, inputs, bits_map, group_size, device, log_fn=log,
                   asym=False):
    stats = []
    t_all = time.time()
    for i, name in enumerate(names):
        module = model.get_submodule(name)
        W = module.weight.data.float()
        if W.dim() != 2:
            continue
        layer_dev = W.device                     # 多卡：每层在其所属卡上量化
        X = inputs[name].to(layer_dev)
        bits = bits_map.get(name, bits_map.get("__default__", 4))
        g = GPTQ(tuple(W.shape), bits=bits, group_size=group_size, asym=asym)
        g.observe(X)
        Wq = g.quantize(W).to(module.weight.dtype)
        rel = float((W - Wq.float()).norm() / W.norm())
        module.weight.data = Wq
        stats.append({"name": name, "shape": tuple(W.shape), "bits": bits, "rel": rel})
        log_fn(f"  [{i+1}/{len(names)}] {name:<52} {str(tuple(W.shape)):<16} "
               f"bits={bits} rel={rel:.4f} dev={layer_dev}")
        del X
        if layer_dev.type == "npu":
            torch.npu.empty_cache()
    log_fn(f"量化完成：{len(stats)} 层，耗时 {time.time()-t_all:.1f}s，"
           f"平均 rel={sum(s['rel'] for s in stats)/max(1,len(stats)):.4f}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="全模型 GPTQ fake-quant")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--bits", type=int, default=4, help="统一位宽")
    ap.add_argument("--bit-map", default=None, help="逐层位宽 JSON（覆盖 --bits）")
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--only", default="all", choices=["all", "mlp", "attn"])
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-batch", type=int, default=4)
    ap.add_argument("--calib-text", default=None)
    ap.add_argument("--calib-dataset", default="wikitext", choices=["wikitext", "fineweb"])
    ap.add_argument("--device-map", default=None,
                    help="多卡：'split'（按层均分）或 'auto'/'balanced'")
    ap.add_argument("--device-map-split", type=int, default=2, help="'split' 时的卡数")
    ap.add_argument("--max-memory", default=None, help="（保留参数）多卡显存上限")
    ap.add_argument("--asym", action="store_true", help="非对称 INT（zero-point，SLQ 论文默认）")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    log(f"加载模型 {args.model} (dtype={args.dtype}, device={args.device}, "
        f"device_map={args.device_map}) ...")
    t0 = time.time()
    model, tokenizer, device = load_model(args)
    log(f"模型加载完成 {time.time()-t0:.1f}s")

    names = target_linear_names(model, args.only)
    log(f"目标 Linear 层数: {len(names)} (only={args.only})")

    if args.bit_map:
        with open(args.bit_map, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        bits_map = {k: int(v) for k, v in raw.items()}
        log(f"逐层位宽表：{len(bits_map)} 项")
    else:
        bits_map = {"__default__": args.bits}

    batches = build_calibration_batches(
        tokenizer, args.calib_seqs, args.calib_len, args.calib_batch, device,
        args.calib_text, args.calib_dataset
    )
    t0 = time.time()
    inputs = capture_inputs(model, names, batches)
    log(f"校准前向完成（{len(names)} 层），耗时 {time.time()-t0:.1f}s")

    stats = quantize_model(model, names, inputs, bits_map, args.group_size, device,
                           asym=args.asym)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        saved = False
        try:
            from accelerate import Accelerator

            acc = Accelerator()
            unwrapped = acc.unwrap_model(model)
            state_dict = acc.get_state_dict(model)
            unwrapped.save_pretrained(args.save_dir, state_dict=state_dict,
                                      safe_serialization=True)
            saved = True
        except Exception as exc:  # noqa: BLE001
            log(f"[warn] accelerate 保存失败（{type(exc).__name__}: {exc}），回退到 CPU 保存")
        if not saved:
            model.to("cpu")
            model.save_pretrained(args.save_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.save_dir)
        log(f"fake-quant checkpoint 已保存至 {args.save_dir}")
    log("model_quant 完成")


if __name__ == "__main__":
    main()
