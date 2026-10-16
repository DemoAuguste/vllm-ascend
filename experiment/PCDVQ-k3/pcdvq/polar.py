"""极坐标解耦 PCD —— 把 k 维笛卡尔向量拆成"方向角 + 幅值"。

论文 PCDVQ 3.2.2 节 Eq.(6)：

    phi_i = atan2( sqrt( sum_{j=i+1..k} v_j^2 ), v_i ),  i = 1..k-1
    r     = ||v||

论文声称 phi_{k-1} ∈ [0, 2π]、其余 ∈ [0, π]，但按字面公式 (即对最后一个角也用
sqrt(v_k^2) = |v_k|) 得到的 phi_{k-1} 实际落在 [0, π]，会丢失最后一个坐标的符号，
使变换非双射（负的 v_k 无法被重建）。参考 notebook 忠实复现了该字面公式，因此同样
存在此缺陷。

本实现做如下修正（标准超球坐标，保证双射）：
    i < k-1 : phi_i = atan2( sqrt( sum_{j>i} v_j^2 ), v_i )     ∈ [0, π]
    i = k-1 : phi_{k-1} = atan2( v_k, v_{k-1} )                 ∈ (-π, π]
重建时 v_k = r * prod(sin) * sin(phi_{k-1})，符号得以保留。

两组角度可分别用不同码本独立量化：方向码本 C_phi ∈ R^{2^a×(k-1)}，幅值码本
C_r ∈ R^{2^b}，总索引位 a+b 对应传统 VQ 的向量索引位。
"""

from __future__ import annotations

import torch


def to_polar(v: torch.Tensor):
    """笛卡尔 -> 极坐标。v: (N, k) -> (phis (N, k-1), r (N,))。"""
    n, k = v.shape
    r = v.norm(dim=1)
    sq = v * v
    tail = sq.flip(1).cumsum(1).flip(1)  # tail[:, i] = sum_{j>=i} v_j^2
    phis = v.new_zeros(n, k - 1)
    for i in range(k - 2):
        phis[:, i] = torch.atan2(tail[:, i + 1].clamp_min(0).sqrt(), v[:, i])
    # 最后一个角：使用带符号的 v_k（修正论文/notebook 的 |v_k| 缺陷）
    phis[:, k - 2] = torch.atan2(v[:, k - 1], v[:, k - 2])
    return phis, r


def from_polar(phis: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """极坐标 -> 笛卡尔。phis: (N, k-1), r: (N,) -> (N, k)。"""
    n, k1 = phis.shape
    k = k1 + 1
    out = phis.new_zeros(n, k)
    prod = r
    for i in range(k - 1):
        out[:, i] = prod * torch.cos(phis[:, i])
        prod = prod * torch.sin(phis[:, i])
    out[:, k - 1] = prod
    return out


def angles_to_unit(phis: torch.Tensor) -> torch.Tensor:
    """角度 -> 单位方向向量（r=1）。"""
    return from_polar(phis, phis.new_ones(phis.shape[0]))
