"""SAGE 配置（超参 + ABS 候选切分）。

对齐论文与官方 `weight_vq/sage/config.py` 的语义：
  - 子向量维度 v=8，半径共享组 G=128（即每 16 个子向量共享一套半径网格）
  - 编码预算 B = v + log2K + V（每 8 个权重的 bit 数）
  - ABS 候选：{(64,4),(128,3),(256,2)}，三者 log2K+V=10 恒定，故总预算不变
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Split:
    """一个"方向码本大小 K / 半径 bit 数 V"的组合（ABS 候选）。"""

    codebook_size: int  # K，方向码本条数（2^log2K）
    radius_bits: int    # V，每组半径的 bit 数

    @property
    def direction_bits(self) -> int:
        """方向索引位数 log2K。"""
        return self.codebook_size.bit_length() - 1

    def payload_bits(self, v: int = 8) -> int:
        """每个子向量的总 bit 数 B = v + log2K + V。"""
        return v + self.direction_bits + self.radius_bits

    def bpw(self, v: int = 8) -> float:
        """仅编码载荷的 bit/权重 = B/v。"""
        return self.payload_bits(v) / v


# 论文/官方默认 ABS 候选：三者均为 18 bit / 8 权重 = 2.25 bpw
DEFAULT_SPLITS: tuple[Split, ...] = (
    Split(64, 4),
    Split(128, 3),
    Split(256, 2),
)

# 低位档扩展候选（供混合精度/消融使用，默认不启用）
EXTENDED_SPLITS: tuple[Split, ...] = (
    Split(32, 3),   # 8 + 5 + 3 = 16 -> 2.000 bpw
    Split(64, 2),   # 8 + 6 + 2 = 16 -> 2.000 bpw
    Split(64, 3),   # 8 + 6 + 3 = 17 -> 2.125 bpw
    Split(64, 4),
    Split(128, 3),
    Split(256, 2),
)


@dataclass
class SAGEConfig:
    """SAGE 量化器配置。

    Args:
        v: 子向量维度（沿输入维切分），论文固定 8。
        group_size: 半径共享组的权重数 G，论文 128（=16 个子向量）。
        energy_power: ESG 拟合的能量指数 γ，论文 0.75（单位均值归一化）。
        splits: ABS 候选切分；仅 1 个时退化为固定切分（复现 SPHQuant 64/4）。
        esg: 是否启用 ESG 能量分层分组与加权拟合。
        adaptive_bits: 是否启用 ABS 逐矩阵选切分。
        kmeans_iters: 球面 k-means 迭代次数（官方 20）。
        sample_size: 拟合码本时采样的子向量数（官方 65536）。
        seed: 随机种子（可复现，含 k-means 初始化）。
        chunk: 方向赋值分块大小（每块子向量数），防大矩阵中间矩阵 OOM。
        energy_eps: 通道能量下限，避免除零/能量为零的通道。
    """

    v: int = 8
    group_size: int = 128
    energy_power: float = 0.75
    splits: tuple[Split, ...] = field(default_factory=lambda: DEFAULT_SPLITS)
    esg: bool = True
    adaptive_bits: bool = True
    kmeans_iters: int = 20
    sample_size: int = 65536
    seed: int = 0
    chunk: int = 8192
    energy_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.v <= 1:
            raise ValueError("v 必须 >= 2")
        if self.group_size % self.v != 0:
            raise ValueError(f"group_size({self.group_size}) 必须能被 v({self.v}) 整除")
        if self.splits is None or len(self.splits) == 0:
            raise ValueError("splits 不能为空")

    @property
    def subs_per_group(self) -> int:
        """每个半径共享组包含的子向量数 G/v。"""
        return self.group_size // self.v
