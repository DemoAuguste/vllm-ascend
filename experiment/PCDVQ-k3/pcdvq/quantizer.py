"""PCDVQ 编/解码器（最小实现）。

流程（对单个权重矩阵 W ∈ R^{p×q}）：
  1. 标准高斯正则化：沿输入维（最后一维）做 RHT，得到 Y ≈ N(0,1)、逐行尺度 scale。
  2. 分组：把每一行按 k 切块，得到 N = p*q/k 个 k 维向量。
  3. 极坐标解耦：v -> (phi (k-1), r)。
  4. 独立量化：方向 argmax cos(phi_dir, C_phi)，幅值 argmin |r - C_r|。
  5. 重建：v_hat = C_r[j] * unit(C_phi[i])，还原形状后做 RHT 逆变换。

位宽：BPW = (a + b) / k。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .codebooks import (
    e8_unit_directions,
    greedy_direction_codebook,
    lloyd_max_magnitude_codebook,
)
from .rht import RHTRegularizer


@dataclass
class PCDVQState:
    """单层量化后的存储对象（索引 + 逆变换所需信息）。"""

    iphi: torch.Tensor  # (N,) 方向索引
    ir: torch.Tensor    # (N,) 幅值索引
    scale: torch.Tensor  # (p, 1) RHT 逐行尺度
    shape: tuple         # (p, q)
    reg: RHTRegularizer  # 逆变换所需的 D / H 配置


class PCDVQ:
    def __init__(
        self,
        k: int = 8,
        a: int = 10,
        b: int = 2,
        tau: float = 0.999,
        max_norm_sq: int = 4,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.k = k
        self.a = a
        self.b = b
        self.tau = tau
        self.max_norm_sq = max_norm_sq
        self.seed = seed
        self.device = device
        self.C_phi = None  # (2^a, k) 单位方向
        self.C_r = None    # (2^b,) 幅值
        self.pool_size = 0

    @property
    def bpw(self) -> float:
        return (self.a + self.b) / self.k

    def build_codebooks(self) -> None:
        pool = e8_unit_directions(self.max_norm_sq).to(self.device)
        self.pool_size = pool.shape[0]
        self.C_phi = greedy_direction_codebook(pool, self.a, seed=self.seed).to(self.device)
        self.C_r = lloyd_max_magnitude_codebook(self.b, self.k, self.tau).to(self.device)

    # ---- 编码 ----
    def quantize(self, W: torch.Tensor, chunk: int | None = None) -> PCDVQState:
        """量化单个权重矩阵 W ∈ R^{p×q}。

        chunk 控制方向匹配的分块大小（每块向量数）：方向码本 C_phi 有 2^a 码字，
        全量 argmax 的中间矩阵为 (N, 2^a)，对大层（如 27B 的 MLP down_proj，N>10M）
        会 OOM；分块后每块峰值内存为 chunk×2^a，可安全处理任意规模。
        """
        assert self.C_phi is not None and self.C_r is not None, "请先 build_codebooks()"
        p, q = W.shape
        assert q % self.k == 0, f"输入维 {q} 必须能被 k={self.k} 整除"
        reg = RHTRegularizer(q, device=W.device, dtype=W.dtype, seed=self.seed)
        Y, scale = reg.forward(W)                      # (p, q), (p, 1)
        groups = Y.reshape(-1, self.k)                 # (N, k)
        r = groups.norm(dim=1)                         # 幅值（无需计算角度）
        dirs = groups / r.clamp_min(1e-12).unsqueeze(1)
        n = dirs.shape[0]
        C = self.C_phi
        if chunk is None or n <= chunk:
            iphi = (dirs @ C.t()).argmax(dim=1)        # 余弦相似度最大
        else:
            iphi = torch.empty(n, dtype=torch.long, device=dirs.device)
            for s in range(0, n, chunk):
                iphi[s : s + chunk] = (dirs[s : s + chunk] @ C.t()).argmax(dim=1)
        ir = (r.unsqueeze(1) - self.C_r.unsqueeze(0)).abs().argmin(dim=1)
        return PCDVQState(iphi=iphi, ir=ir, scale=scale, shape=(p, q), reg=reg)

    # ---- 解码 ----
    def dequantize(self, state: PCDVQState) -> torch.Tensor:
        p, q = state.shape
        d_hat = self.C_phi[state.iphi]                 # (N, k) 单位方向
        r_hat = self.C_r[state.ir].unsqueeze(1)        # (N, 1)
        Y_hat = (r_hat * d_hat).reshape(p, q)
        return state.reg.inverse(Y_hat, state.scale)

    # ---- 便于可视化/分析：分组向量空间中的方向/幅值 ----
    def group_vectors(self, W: torch.Tensor):
        reg = RHTRegularizer(W.shape[1], device=W.device, dtype=W.dtype, seed=self.seed)
        Y, scale = reg.forward(W)
        return Y.reshape(-1, self.k), scale
