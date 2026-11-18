"""随机化 Hadamard 变换 (RHT) —— 标准高斯正则化。

论文 PCDVQ (arXiv:2506.05432) 3.2.1 节：对权重逐列做随机化 Hadamard 变换，
再除以尺度因子，使变换后的权重近似服从标准高斯分布 N(0, 1)，从而让"方向码本"
与"幅值码本"可以对齐统一的解析分布。

本实现修正了参考 notebook 的关键 bug：
  * 参考实现构造了 Hadamard 矩阵 H 却从未使用（只乘了一个随机对角符号阵），
    因此完全没有做坐标混合，权重分布并未高斯化。这里补回 H 的混合作用。
  * 参考实现里 H D x 的尺度推导有误。正确做法（可逆、方差为 1）为：

        y = S (D x) / s,     s = ||x|| * sqrt(b / q)

    其中 S = I_{q/b} ⊗ H_b 为分块 Hadamard（块大小 b 为整除 q 的最大 2 的幂，
    H_b 为 b×b 的 ±1 Hadamard 矩阵，满足 H_b H_b^T = b I），D 为 ±1 随机符号对角阵。
    此时每个元素方差为 1，即 y ≈ N(0, 1)。逆变换：

        x = D · S^T (s · y) / b

  * 通过分块 Hadamard 支持任意维度 q（只要 b>=2），不再要求 q 是 2 的幂。
"""

from __future__ import annotations

import torch


def hadamard_matrix(n: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """构造 n×n 的 Sylvester Hadamard 矩阵（n 必须为 2 的幂）。"""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard 阶数必须为 2 的幂，收到 {n}")
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h


def largest_pow2_factor(dim: int) -> int:
    """返回整除 dim 的最大 2 的幂（块大小 b）。"""
    b = 1
    while dim % (b * 2) == 0:
        b *= 2
    return b


class RHTRegularizer:
    """沿最后一维的随机化分块 Hadamard 变换 + 逐向量尺度归一化。"""

    def __init__(self, dim: int, device="cpu", dtype=torch.float32, seed: int | None = 0):
        self.dim = dim
        self.block = largest_pow2_factor(dim)
        if self.block < 2:
            raise ValueError(
                f"dim={dim} 不存在 >=2 的 2 的幂因子，无法做 Hadamard 混合"
            )
        self.num_blocks = dim // self.block
        self.H = hadamard_matrix(self.block, device=device, dtype=dtype)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        sign = torch.randint(0, 2, (dim,), generator=gen).to(dtype) * 2 - 1
        self.D = sign.to(device)
        self.device = device
        self.dtype = dtype

    def _apply_S(self, x: torch.Tensor) -> torch.Tensor:
        """S x，其中 S = I_{q/b} ⊗ H_b，作用于最后一维。"""
        lead = x.shape[:-1]
        xr = x.reshape(*lead, self.num_blocks, self.block)
        yr = xr @ self.H.t()
        return yr.reshape(*lead, self.dim)

    def _apply_ST(self, y: torch.Tensor) -> torch.Tensor:
        """S^T y，作用于最后一维。"""
        lead = y.shape[:-1]
        yr = y.reshape(*lead, self.num_blocks, self.block)
        xr = yr @ self.H
        return xr.reshape(*lead, self.dim)

    def forward(self, x: torch.Tensor):
        """变换到标准高斯空间，返回 (y, scale)，scale 形状 (..., 1)。"""
        xd = x * self.D
        y = self._apply_S(xd)
        scale = x.norm(dim=-1, keepdim=True) * (self.block / self.dim) ** 0.5
        scale = scale.clamp_min(1e-12)
        return y / scale, scale

    def inverse(self, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """从标准高斯空间还原（用于反量化后的逆变换）。"""
        x = self._apply_ST(y * scale) * self.D / self.block
        return x
