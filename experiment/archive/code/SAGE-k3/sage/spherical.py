"""球坐标解耦（Spherical VQ）与半径网格量化。

对应论文 §3.1 与 Appendix A.6：
  子向量 w ∈ R^v 分解为
    符号 s = sign(w)              —— 显式存储（v bit）
    模长 |w| = r·d,  r = ‖w‖₂     —— 半径标量量化（V bit）
    非负单位方向 d = |w|/r        —— 码本查表（log2K bit）
  重建：ŵ = r̂ · (s ⊙ c_z)

半径共享网格（Eq.2/14）：每 G 个权重（G/v 个子向量）共享一组
  µ_g = min r̃,  δ_g = (max r̃ − µ_g)/(2^V − 1)
  r̂ = µ_g + q δ_g,  q = clip(round((r̃ − µ_g)/δ_g), 0, 2^V − 1)

码本角参数化（Eq.12-13）：保证码字非负且单位长度
  c₁ = cos θ₁;  c_j = (Π_{i<j} sin θ_i) cos θ_j (2≤j<v);  c_v = Π_{i=1}^{v-1} sin θ_i
  θ_i = ε + (π/2 − 2ε) σ(α_i),  ε = 1e-4
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-4  # 角度边界保护（论文 ε=1e-4）


def decompose(W: torch.Tensor, v: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把权重矩阵沿输入维切成 v 维子向量并分解。

    Args:
        W: (dout, din)，din 必须能被 v 整除。
        v: 子向量维度。
    Returns:
        neg: (N, v) bool，True 表示该分量为负；
        mag: (N, v) 非负模长 |w|；
        radii: (N,) 子向量模长 r=‖w‖₂。
    """
    din = W.shape[1]
    if din % v != 0:
        raise ValueError(f"输入维 {din} 必须能被 v={v} 整除")
    sub = W.reshape(-1, v)
    mag = sub.abs()
    neg = sub < 0
    radii = mag.norm(dim=1)
    return neg, mag, radii


def angles_to_codewords(alpha: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """未约束角 logits α (K, v-1) → 非负单位码字 (K, v)（Eq.12-13）。"""
    theta = eps + (math.pi / 2.0 - 2.0 * eps) * torch.sigmoid(alpha)
    K, v1 = theta.shape
    v = v1 + 1
    sin_t = theta.sin()
    cos_t = theta.cos()
    c = torch.empty((K, v), dtype=theta.dtype, device=theta.device)
    c[:, 0] = cos_t[:, 0]
    for j in range(1, v1):  # 0-based：c_j = (Π_{i<j} sin θ_i)·cos θ_j
        c[:, j] = sin_t[:, :j].prod(dim=1) * cos_t[:, j]
    c[:, v - 1] = sin_t.prod(dim=1)
    return c


def codewords_to_angles(C: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """非负单位码字 (K, v) → 角 logits (K, v-1)（angles_to_codewords 的逆）。"""
    K, v = C.shape
    theta = torch.empty((K, v - 1), dtype=C.dtype, device=C.device)
    rem = torch.ones(K, dtype=C.dtype, device=C.device)  # r_0 = 1
    for i in range(v - 1):
        cos_t = (C[:, i] / rem.clamp_min(1e-12)).clamp(0.0, 1.0)
        theta[:, i] = torch.acos(cos_t)
        rem = rem * theta[:, i].sin()
    t = ((theta - eps) / (math.pi / 2.0 - 2.0 * eps)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(t / (1.0 - t))


def radius_grid_quantize(
    r: torch.Tensor,
    radius_bits: int,
    subs_per_group: int,
    ste: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """对连续半径 r 做"每组共享均匀网格"的标量量化（Eq.2/14）。

    Args:
        r: (N,) 连续半径 r̃（blockwise 优化时为当前 softplus 半径）。
        radius_bits: V。
        subs_per_group: 每组子向量数 G/v。
        ste: 为 True 时返回直通估计值 r + (r̂ − r).detach()，用于反向传播。
    Returns:
        r_hat: (N,) 量化半径（或 STE 值）；
        minimum: (ng,) 每组网格下界 µ_g；
        step: (ng,) 每组网格步长 δ_g；
        codes: (N,) 组内整数码 q ∈ [0, 2^V−1]（step=0 的组记为 0）。
    """
    n = r.shape[0]
    if n % subs_per_group != 0:
        raise ValueError(f"子向量数 {n} 必须能被 subs_per_group={subs_per_group} 整除")
    ng = n // subs_per_group
    levels = (1 << radius_bits) - 1
    rg = r.reshape(ng, subs_per_group)
    minimum = rg.min(dim=1).values
    maximum = rg.max(dim=1).values
    step = (maximum - minimum) / levels
    min_e = minimum.repeat_interleave(subs_per_group)
    step_e = step.repeat_interleave(subs_per_group)
    positive = step_e > 0
    step_safe = torch.where(positive, step_e, torch.ones_like(step_e))
    codes = ((r - min_e) / step_safe).round().clamp(0.0, float(levels))
    codes = torch.where(positive, codes, torch.zeros_like(codes))
    r_hat = min_e + codes * step_e
    if ste:
        r_hat = r + (r_hat - r).detach()
    return r_hat, minimum, step, codes.to(torch.long)


def reconstruct_vectors(
    neg: torch.Tensor,
    codewords: torch.Tensor,
    direction_index: torch.Tensor,
    r_hat: torch.Tensor,
) -> torch.Tensor:
    """由符号/码字/方向索引/量化半径重建子向量（Eq.1）：ŵ = r̂·(s ⊙ c_z)。"""
    c = codewords[direction_index]                       # (N, v) 非负单位
    sign = torch.where(neg, -torch.ones_like(c), torch.ones_like(c))
    return sign * r_hat.unsqueeze(1) * c


def expand_radius(
    codes: torch.Tensor, minimum: torch.Tensor, step: torch.Tensor, subs_per_group: int
) -> torch.Tensor:
    """由组内码 q 与网格参数 (µ_g, δ_g) 还原量化半径 r̂（用于反量化）。"""
    min_e = minimum.repeat_interleave(subs_per_group)
    step_e = step.repeat_interleave(subs_per_group)
    return min_e + codes.to(step_e.dtype) * step_e


def unpermute_columns(W_perm: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """把"按置换后输入维"排布的权重还原到原始通道顺序（W[:, π] = W_perm）。"""
    W = torch.empty_like(W_perm)
    W[:, permutation] = W_perm
    return W
