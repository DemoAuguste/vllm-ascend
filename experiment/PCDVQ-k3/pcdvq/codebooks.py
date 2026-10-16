"""分布对齐码本构造 DACC。

包含两个分布对齐的码本（论文 3.2.3 节）：

1. 方向码本（空间均匀分布）
   高斯向量方向在球面上均匀分布。E8 格是 8 维最密球堆积、方向高度均匀对称，
   因此在 E8 格点方向池上用贪心最远点采样 (FPS) 得到 2^a 个方向。
   * 参考 notebook 只用了 E8 的"最小壳层"(norm^2=2，共 240 个点)，无法支撑 a>=8；
     本实现枚举 norm^2<=max_norm_sq 的全部壳层。
   * 参考 notebook 未把 E8 点单位化；E8 各壳层范数不同（2、4、...），方向比较与
     重建都必须用单位向量，这里统一归一化。

2. 幅值码本（卡方根分布）
   r^2 ~ chi^2(k)，其 PDF 为 f(r)=2^{1-k/2}/Gamma(k/2) r^{k-1} e^{-r^2/2}。
   对其做最优一维标量量化（Lloyd-Max）得到 2^b 个码字。
   该部分参考 notebook 的实现是正确的（见 tests 中的金标准校验）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc

from .rht import hadamard_matrix  # noqa: F401  (便于外部统一引用)


def e8_unit_directions(max_norm_sq: int = 4) -> torch.Tensor:
    """枚举 E8 格中 norm^2 <= max_norm_sq 的全部格点并归一化为单位方向。

    E8 = D8  ∪  (D8 + (1/2,...,1/2))
      D8  : 所有分量均为整数的点
      D8' : 所有分量均为半整数 (x+1/2) 的点
    两者的公共条件：各分量之和为偶数。
    """
    pts = []
    s = int(math.isqrt(max_norm_sq))
    for vals in (
        torch.arange(-s, s + 1, dtype=torch.float64),
        torch.arange(-(s + 1), s + 1, dtype=torch.float64) + 0.5,
    ):
        grid = torch.meshgrid(*([vals] * 8), indexing="ij")
        p = torch.stack([g.reshape(-1) for g in grid], dim=1)  # (M, 8)
        n2 = (p * p).sum(dim=1)
        ssum = p.sum(dim=1)
        even = (torch.round(ssum / 2) * 2 - ssum).abs() < 1e-6
        nonzero = n2 > 1e-9  # 排除原点（原点不是方向）
        keep = (n2 <= max_norm_sq + 1e-9) & even & nonzero
        pts.append(p[keep])
    P = torch.unique(torch.cat(pts, dim=0), dim=0)
    P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return P.float()


def greedy_direction_codebook(pool: torch.Tensor, bits: int, seed: int = 0) -> torch.Tensor:
    """从方向池中贪心采样 2^bits 个彼此尽量远离的单位方向（论文 Algorithm 1）。"""
    n_centers = 2 ** bits
    m = pool.shape[0]
    if n_centers > m:
        raise ValueError(
            f"方向池只有 {m} 个候选，无法构造 {n_centers} (=2^{bits}) 个码字，"
            f"请增大 max_norm_sq"
        )
    gen = torch.Generator().manual_seed(seed)
    sel = [int(torch.randint(0, m, (1,), generator=gen).item())]
    # max_sim[j] = 候选 j 与已选集合的最大余弦相似度
    max_sim = pool @ pool[sel[0]]
    max_sim[sel[0]] = float("inf")
    for _ in range(n_centers - 1):
        j = int(torch.argmin(max_sim).item())
        sel.append(j)
        sim = pool @ pool[j]
        max_sim = torch.maximum(max_sim, sim)
        # 仅把刚选中的点置 inf（其余已选点此前已置 inf），避免每步 O(len(sel)) 重建
        max_sim[j] = float("inf")
    return pool[torch.tensor(sel)].contiguous()


def find_max_r(k: int, tau: float, iters: int = 64) -> float:
    """求幅值上界 max_r 使得 F(max_r) = chi2.cdf(max_r^2; k) = tau。"""
    lo = 0.0
    hi = max(2.0 * math.sqrt(k), 1.0)
    while stats.chi2.cdf(hi * hi, df=k) < tau:
        hi *= 1.5
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if stats.chi2.cdf(mid * mid, df=k) < tau:
            lo = mid
        else:
            hi = mid
    return hi


def lloyd_max_magnitude_codebook(
    bits: int, k: int, tau: float = 0.999, tol: float = 1e-8, max_iters: int = 500
) -> torch.Tensor:
    """对卡方根分布 r~chi(k) 做 Lloyd-Max 最优标量量化，返回 2^bits 个码字。"""
    n = 2 ** bits
    max_r = find_max_r(k, tau)
    edges = np.linspace(0.0, max_r, n + 1)
    cb = 0.5 * (edges[:-1] + edges[1:])
    const = np.sqrt(2.0) * gamma_fn((k + 1) / 2.0) / gamma_fn(k / 2.0)
    for _ in range(max_iters):
        u = np.empty(n + 1)
        u[0] = 0.0
        u[-1] = max_r
        u[1:-1] = 0.5 * (cb[:-1] + cb[1:])
        new_cb = np.empty_like(cb)
        for i in range(n):
            lo2, hi2 = u[i] ** 2 / 2.0, u[i + 1] ** 2 / 2.0
            num = gammainc((k + 1) / 2.0, hi2) - gammainc((k + 1) / 2.0, lo2)
            den = gammainc(k / 2.0, hi2) - gammainc(k / 2.0, lo2)
            new_cb[i] = const * num / (den + 1e-12)
        shift = np.abs(new_cb - cb).max()
        cb = new_cb
        if shift < tol:
            break
    return torch.tensor(cb, dtype=torch.float32)
