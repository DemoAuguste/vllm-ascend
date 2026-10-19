# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# SLQ 敏感度估计（多 bitwidth）。
#
# 默认用 **KL 散度**：把某层权重临时替换为 b-bit GPTQ fake-quant 权重，测
# 量化前后输出分布的平均 KL（比 NLL 更敏感、更稳健）。也提供 NLL 版本。
# 结果 α_l(b) 供 ilp.py 做全局位宽分配。
# ----------------------------------------------------------------------------
from __future__ import annotations

import re
from dataclasses import dataclass

import torch


def parse_sens_log(path: str) -> dict[str, dict[int, float]]:
    """从 slq_allocate 日志解析逐层逐位敏感度：{name: {bits: value}}。

    行格式：`  [12/196] model.layers.1.mlp.up_proj    b3:5.418e-03 b4:1.631e-03 b8:9.428e-04`
    """
    pat = re.compile(r"\]\s+(\S+)\s+((?:b\d+:[0-9.eE+-]+\s*)+)")
    out: dict[str, dict[int, float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            name, rest = m.group(1), m.group(2)
            vals = {}
            for bm in re.finditer(r"b(\d+):([0-9.eE+-]+)", rest):
                vals[int(bm.group(1))] = float(bm.group(2))
            out[name] = vals
    return out


@dataclass
class LayerSensitivity:
    name: str
    metric: str                          # 'kl' | 'nll'
    value_by_bits: dict[int, float]      # b -> 度量值（KL 或 NLL）
    alpha_by_bits: dict[int, float]      # b -> 相对 FP 的增量（越小越不敏感）


def make_eval_batches(tokenizer, text, device, n_windows=6, max_length=512, stride=512):
    enc = tokenizer(text, return_tensors="pt").input_ids
    seq_len = enc.size(1)
    batches = []
    for begin in range(0, seq_len, stride):
        if len(batches) >= n_windows:
            break
        end = min(begin + max_length, seq_len)
        batches.append(enc[:, begin:end].to(device))
        if end == seq_len:
            break
    return batches


@torch.no_grad()
def model_logprobs(model, batches):
    """返回每批的 log-prob（float16, CPU），作为 KL 参考分布。"""
    out = []
    for b in batches:
        logits = model(b).logits.float()
        out.append(torch.log_softmax(logits, dim=-1).to(torch.float16).cpu())
    return out


@torch.no_grad()
def mean_kl(model, batches, ref_logprobs):
    """量化模型的输出分布与 FP 参考分布的平均 KL。"""
    tot, n = 0.0, 0
    for b, ref_lp in zip(batches, ref_logprobs):
        lp = torch.log_softmax(model(b).logits.float(), dim=-1)
        ref = ref_lp.to(lp.device).float()
        kl = (ref.exp() * (ref - lp)).sum(dim=-1).mean()
        tot += float(kl)
        n += 1
    return tot / max(1, n)


def layer_kl_sensitivity(model, batches, ref_logprobs, target_names,
                         quantize_fn, bits_list, log=print):
    """逐层 KL 敏感度。

    Args:
        quantize_fn: callable(name, bits) -> (Wq, restore_fn)，负责产出该层量化权重。
    Returns:
        list[LayerSensitivity]（metric='kl'）
    """
    out = []
    for name in target_names:
        module = model.get_submodule(name)
        orig = module.weight.data
        vals = {}
        for b in bits_list:
            Wq = quantize_fn(name, b)
            module.weight.data = Wq.to(orig.dtype)
            vals[b] = mean_kl(model, batches, ref_logprobs)
        module.weight.data = orig
        base = vals[min(bits_list)]
        out.append(LayerSensitivity(
            name=name, metric="kl", value_by_bits=vals,
            alpha_by_bits={b: vals[b] - base for b in bits_list}))
        log(f"  [kl] {name:<50} " + " ".join(f"b{b}:{vals[b]:.3e}" for b in bits_list))
    return out
