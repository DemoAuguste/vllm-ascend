# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# INT 权重伪量化：per-channel / per-group 对称 & 非对称量化-反量化，任意位宽 {2..8}。
#
# 语义对齐昇腾 AMCT（Apache-2.0）`quant_dequant_tensor` 的 INT 分支，并补充
# **非对称（zero-point）** 分支 —— SLQ 论文方差律指出：对称量化噪声方差比非对称
# 放大 γ²，分布无损（DL）必须用非对称。
# ----------------------------------------------------------------------------
from __future__ import annotations

import torch


def _reshape_groups(W: torch.Tensor, group_size: int):
    rows, cols = W.shape
    ng = (cols + group_size - 1) // group_size
    pad = ng * group_size - cols
    Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
    return Wp.reshape(rows, ng, group_size), ng, pad


def int_qparams(W: torch.Tensor, bits: int, group_size: int | None = None,
                sym: bool = True, eps: float = 1e-9):
    """计算量化参数。返回 (scale, zero)。

    - 对称：qmax=2^(bits-1)-1, qmin=-qmax, zero=0, scale=absmax/qmax
    - 非对称：qmax=2^bits-1, qmin=0, zero=min, scale=(max-min)/qmax
    group_size=None 表示 per-output-channel（整行共享），否则沿输入维分组。
    """
    Wf = W.float()
    if group_size:
        Wg, ng, _ = _reshape_groups(Wf, group_size)
        if sym:
            qmax = float(2 ** (bits - 1) - 1)
            scale = Wg.abs().amax(dim=-1).clamp_min(eps) / qmax
            zero = torch.zeros_like(scale)
        else:
            qmax = float(2 ** bits - 1)
            mx = Wg.amax(dim=-1)
            mn = Wg.amin(dim=-1)
            scale = ((mx - mn) / qmax).clamp_min(eps)
            zero = mn
        return scale, zero

    if sym:
        qmax = float(2 ** (bits - 1) - 1)
        scale = Wf.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
        zero = torch.zeros_like(scale)
    else:
        qmax = float(2 ** bits - 1)
        mx = Wf.amax(dim=-1, keepdim=True)
        mn = Wf.amin(dim=-1, keepdim=True)
        scale = ((mx - mn) / qmax).clamp_min(eps)
        zero = mn
    return scale, zero


def quant_dequant_int(W: torch.Tensor, bits: int, group_size: int | None = None,
                      sym: bool = True, scale: torch.Tensor | None = None,
                      zero: torch.Tensor | None = None) -> torch.Tensor:
    """对称/非对称量化-反量化（fake-quant）。"""
    if scale is None:
        scale, zero = int_qparams(W, bits, group_size, sym)
    if zero is None:
        zero = torch.zeros_like(scale)
    qmax = float(2 ** (bits - 1) - 1) if sym else float(2 ** bits - 1)
    qmin = -qmax if sym else 0.0

    Wf = W.float()
    if group_size and scale.dim() == 2 and scale.shape[1] > 1:
        Wg, ng, pad = _reshape_groups(Wf, group_size)
        q = ((Wg - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(qmin, qmax)
        wq = (q * scale.unsqueeze(-1) + zero.unsqueeze(-1)).reshape(Wg.shape[0], -1)
        if pad:
            wq = wq[:, : W.shape[-1]]
        return wq.to(W.dtype)

    q = ((Wf - zero) / scale).round().clamp(qmin, qmax)
    return (q * scale + zero).to(W.dtype)


def quant_dequant_weight(W: torch.Tensor, wts_type: str) -> torch.Tensor:
    bits = int(wts_type.replace("int", ""))
    return quant_dequant_int(W, bits)
