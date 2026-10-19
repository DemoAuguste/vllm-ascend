"""方向码本拟合：激活加权球面 k-means（非负正交象限）。

对应论文 §3.2「Weighted Quantizer Fitting」与 Eq.6。

在置换后的通道顺序上，方向位于非负正交象限（因为模长 p=|w|≥0）。
给定一组子向量的模长 p (N,v) 与权重 ω (N,v)，码字 c_k 皆非负单位向量，
拟合目标为加权重构误差  L = Σ_i ω_i (p_i − r c_{k,i})²。

对连续半径取闭式最优  r*_k = (Σ_i ω_i p_i c_{k,i}) / (Σ_i ω_i c_{k,i}²)
后代回目标，方向赋值等价于
    z = argmax_k  (Σ_i ω_i p_i c_{k,i})² / (Σ_i ω_i c_{k,i}²)
均匀权重时退化为余弦相似度 argmax（论文 §3.1）。

码本更新采用标准球面 k-means：c_k ∝ Σ_{n∈k} ω_n p_n（加权均值方向）后归一化。
"""

from __future__ import annotations

import torch

from .utils import chunked_matmul_argmax  # noqa: F401  (保留统一的分块实现入口)


def weighted_assignment(
    mag: torch.Tensor,
    omega: torch.Tensor,
    codewords: torch.Tensor,
    chunk: int = 8192,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """加权方向赋值 + 连续半径闭式解（Eq.6）。

    Args:
        mag: (N, v) 非负模长 p。
        omega: (N, v) 通道权重 ω（子向量内单位均值）。
        codewords: (K, v) 非负单位码字。
        chunk: 按 N 分块大小（控制 (chunk, K) 中间矩阵内存）。
        eps: 分母保护。
    Returns:
        z: (N,) 方向索引；r_cont: (N,) 连续半径 r̃。
    """
    n = mag.shape[0]
    wm = omega * mag                       # (N, v)
    c2 = codewords * codewords             # (K, v)
    z = torch.empty(n, dtype=torch.long, device=mag.device)
    r_cont = torch.empty(n, dtype=mag.dtype, device=mag.device)
    step = max(1, chunk)
    for start in range(0, n, step):
        stop = min(start + step, n)
        num = wm[start:stop] @ codewords.t()      # (b, K) = Σ ω p c
        den = omega[start:stop] @ c2.t()          # (b, K) = Σ ω c²
        score = (num * num) / den.clamp_min(eps)
        idx = score.argmax(dim=1)
        z[start:stop] = idx
        rows = torch.arange(stop - start, device=mag.device)
        r_cont[start:stop] = num[rows, idx] / den[rows, idx].clamp_min(eps)
    return z, r_cont


def weighted_spherical_kmeans(
    mag: torch.Tensor,
    omega: torch.Tensor,
    k: int,
    iters: int = 20,
    seed: int = 0,
    sample_size: int = 65536,
    chunk: int = 8192,
    eps: float = 1e-12,
) -> torch.Tensor:
    """加权球面 k-means 拟合 K 个非负单位码字。

    Args:
        mag: (N, v) 非负模长。
        omega: (N, v) 通道权重。
        k: 码本大小 K。
        iters: 迭代次数。
        seed: 随机种子（初始化与采样）。
        sample_size: 拟合时最多采样的子向量数（论文 65536）。
        chunk: 赋值分块大小。
        eps: 数值保护。
    Returns:
        (K, v) 非负单位码字（按第一个非零分量约定符号为正，天然非负）。
    """
    n, v = mag.shape
    if k <= 0:
        raise ValueError("k 必须为正")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # 采样用于拟合的子集（NPU 上仍按采样索引 gather，保持张量化）
    if n > sample_size:
        sel = torch.randperm(n, generator=gen)[:sample_size]
        sel = sel.to(mag.device)
        fit_mag = mag[sel]
        fit_omega = omega[sel]
    else:
        fit_mag, fit_omega = mag, omega
    nf = fit_mag.shape[0]

    # 初始化：随机选取 k 个样本方向（已在非负象限），单位化
    init = torch.randperm(nf, generator=gen)[:k].to(mag.device)
    C = fit_mag[init]
    C = C / C.norm(dim=1, keepdim=True).clamp_min(eps)
    C = C.clone()

    for _ in range(max(1, iters)):
        z, _ = weighted_assignment(fit_mag, fit_omega, C, chunk=chunk, eps=eps)
        # 加权均值方向更新：c_k ∝ Σ_{n∈k} ω_n p_n
        weighted = fit_omega * fit_mag                       # (nf, v)
        newC = torch.zeros_like(C)
        newC.index_add_(0, z, weighted)
        counts = torch.zeros(k, device=mag.device, dtype=C.dtype)
        counts.index_add_(0, z, torch.ones(nf, device=mag.device, dtype=C.dtype))
        norm = newC.norm(dim=1, keepdim=True)
        # 空簇保留原码字；非空簇归一化
        nonempty = (counts > 0) & (norm.squeeze(1) > eps)
        newC[~nonempty] = C[~nonempty]
        newC[nonempty] = newC[nonempty] / norm[nonempty]
        shift = (newC - C).norm()
        C = newC
        if shift < 1e-6:
            break
    return C
