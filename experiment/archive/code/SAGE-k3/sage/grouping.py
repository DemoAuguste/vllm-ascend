"""ESG：能量分层分组（Energy-Stratified Grouping）。

对应论文 §3.2 与 Eq.5。

动机：加权重建（ω∝e^γ）只有在"同一个子向量内通道能量差异大"时才有意义；
若共享一个子向量的通道能量相近，则 ω≈1，退化为普通欧氏重建。ESG 通过
把能量排序后的通道**跨层交错**分布到各子向量，使每个子向量都横跨多个能量层，
从而让加权拟合在每个联合编码向量内都能优先保护高能（敏感）通道。

置换构造（Eq.5，1-based 语义）：
  设 σ 为按能量降序排列的输入通道索引；把排序后的列表切成 v 个 tier，
  每 tier m = din/v 个通道。则
      π((j−1)v + t) = σ((t−1)m + j),  j=1..m,  t=1..v
  即：子向量 j 的第 t 个位置取自 tier t 中排名第 j 的通道。
  位置 t=1 对应最高能量 tier，位置 t=v 对应最低能量 tier。
"""

from __future__ import annotations

import torch


def channel_energy(activations: torch.Tensor) -> torch.Tensor:
    """由校准激活估计逐输入通道能量 e_c = E[x_c²]。

    Args:
        activations: (..., din) 该层输入激活（可为 (tokens, din) 或 (batches, tokens, din)）。
    Returns:
        (din,) 非负通道能量（float32）。
    """
    x = activations.reshape(-1, activations.shape[-1]).float()
    return (x * x).mean(dim=0)


def esg_permutation(energy: torch.Tensor, v: int = 8) -> torch.Tensor:
    """由通道能量构造 ESG 交错置换 π（Eq.5）。

    Args:
        energy: (din,) 通道能量。
        v: 子向量维度。
    Returns:
        (din,) 的置换索引；用法 `W_perm = W[:, pi]`（沿输入维重排）。
    """
    din = energy.numel()
    if din % v != 0:
        raise ValueError(f"输入维 {din} 必须能被 v={v} 整除")
    m = din // v
    sigma = torch.argsort(energy, descending=True)  # (din,) 降序
    # idx[t, j] = sigma[(t-1)m + j]（t=0..v-1, j=0..m-1），即"tier 优先"排布
    idx = sigma.reshape(v, m)
    # π[(j-1)v + t] = idx[t, j] → 转置后按行优先展平
    pi = idx.t().reshape(-1)
    return pi


def energy_weights(energy_perm: torch.Tensor, v: int, gamma: float, eps: float = 1e-8) -> torch.Tensor:
    """计算加权拟合的通道权重 ω（论文 Eq.4：ω ∝ e^γ，子向量内归一化到单位均值）。

    Args:
        energy_perm: (din,) 已按置换 π 重排的通道能量。
        v: 子向量维度。
        gamma: 能量指数 γ。
        eps: 能量下限。
    Returns:
        (N, v) 权重，每个子向量内均值为 1（sum = v）。
    """
    n = energy_perm.numel() // v
    w = energy_perm.clamp_min(eps).pow(gamma).reshape(n, v)
    mean = w.mean(dim=1, keepdim=True).clamp_min(eps)
    return w / mean
