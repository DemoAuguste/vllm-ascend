"""SAGE 参考实现（NPU 友好）。

SAGE: Split Adaptation and Grouping by Energy —— 极低位权重向量量化（VQ）。
论文：arXiv:2610.14745v1；官方实现：https://github.com/Zincwolf/sage

模块划分：
  config     —— SAGEConfig（超参与 ABS 候选切分）
  utils      —— device 解析、分块方向赋值、日志等 NPU 友好辅助
  spherical  —— 球坐标分解/重建、非负单位码本角参数化、共享半径网格量化
  grouping   —— 激活通道能量估计 + ESG 能量分层交错置换
  codebook   —— 加权球面 k-means（非负正交象限）
  quantizer  —— SAGEQuantizer：拟合 / ABS 选择 / 编解码 / bpp
  blockwise  —— 逐层 blockwise 误差补偿优化（STE，更新半径与码本角）
"""

from .config import SAGEConfig, Split
from .quantizer import SAGEQuantizer, SAGEState
from .grouping import channel_energy, esg_permutation
from .spherical import (
    angles_to_codewords,
    codewords_to_angles,
    decompose,
    radius_grid_quantize,
    reconstruct_vectors,
)
from .codebook import weighted_spherical_kmeans
from .blockwise import BlockwiseConfig, blockwise_optimize

__all__ = [
    "SAGEConfig",
    "Split",
    "SAGEQuantizer",
    "SAGEState",
    "channel_energy",
    "esg_permutation",
    "angles_to_codewords",
    "codewords_to_angles",
    "decompose",
    "radius_grid_quantize",
    "reconstruct_vectors",
    "weighted_spherical_kmeans",
    "BlockwiseConfig",
    "blockwise_optimize",
]
