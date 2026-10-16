"""PCDVQ 最小可验证实现（修正版）。

模块划分：
  rht        —— 随机化分块 Hadamard 标准高斯正则化（含逆变换）
  polar      —— 极坐标解耦 PCD（符号修正，双射）
  codebooks  —— 方向码本（E8 贪心）+ 幅值码本（chi Lloyd-Max）
  quantizer  —— PCDVQ 编解码器
"""

from .codebooks import (
    e8_unit_directions,
    find_max_r,
    greedy_direction_codebook,
    lloyd_max_magnitude_codebook,
)
from .polar import angles_to_unit, from_polar, to_polar
from .quantizer import PCDVQ, PCDVQState
from .rht import RHTRegularizer, hadamard_matrix, largest_pow2_factor

__all__ = [
    "RHTRegularizer",
    "hadamard_matrix",
    "largest_pow2_factor",
    "to_polar",
    "from_polar",
    "angles_to_unit",
    "e8_unit_directions",
    "greedy_direction_codebook",
    "find_max_r",
    "lloyd_max_magnitude_codebook",
    "PCDVQ",
    "PCDVQState",
]
