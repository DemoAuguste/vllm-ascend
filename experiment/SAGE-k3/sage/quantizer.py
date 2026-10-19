"""SAGE 量化器：拟合 / ABS 选择 / 编解码 / bpp 计算。

对应论文 §3.1–§3.3 与 Algorithm 1。
单个权重矩阵 W ∈ R^{dout×din} 的流程：
  1. 通道能量 e_c 估计（外部传入）→ ESG 置换 π（Eq.5）；
  2. 按 π 重排输入维，切 v 维子向量，分解为符号 s / 模长 p / 半径 r；
  3. 加权球面 k-means 拟合方向码本（Eq.6）；
  4. 方向赋值 + 连续半径闭式解 → 半径网格量化（Eq.2）；
  5. 重建并还原到原始通道顺序，按激活加权误差评分；
  6. ABS：在候选切分中选激活加权误差最小者（Eq.7-9）。

bit 预算（Eq.3/15）：BPW = (v + log2K + V)/v + 网格/码本元数据开销。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .codebook import weighted_assignment, weighted_spherical_kmeans
from .config import SAGEConfig, Split
from .grouping import energy_weights, esg_permutation
from .spherical import (
    decompose,
    expand_radius,
    radius_grid_quantize,
    reconstruct_vectors,
    unpermute_columns,
)

_GRID_PARAM_BITS = 16  # 网格参数按 FP16 存储（论文 Appendix A.6）
_CODEBOOK_PARAM_BITS = 16


@dataclass
class SAGEState:
    """单个权重矩阵的 SAGE 量化状态（索引 + 码本 + 网格参数）。"""

    shape: tuple[int, int]
    v: int
    split: Split
    permutation: torch.Tensor      # (din,) 输入维置换 π（W_perm = W[:, π]）
    neg: torch.Tensor              # (N, v) bool 符号
    direction_index: torch.Tensor  # (N,) 方向索引
    codewords: torch.Tensor        # (K, v) 非负单位码字
    radius_codes: torch.Tensor     # (N,) 组内整数码 q
    grid_min: torch.Tensor         # (ng,) µ_g
    grid_step: torch.Tensor        # (ng,) δ_g
    score: float = float("nan")    # 激活加权重建误差（ABS 依据）
    r_cont: torch.Tensor | None = field(default=None, repr=False)  # 连续半径（blockwise 初值）

    @property
    def num_subvectors(self) -> int:
        return self.neg.shape[0]


class SAGEQuantizer:
    """SAGE 量化器（设备无关，NPU/CPU 通用）。"""

    def __init__(self, config: SAGEConfig | None = None) -> None:
        self.config = config or SAGEConfig()

    # ------------------------------------------------------------------ 拟合
    def _fit_split(
        self, W: torch.Tensor, energy: torch.Tensor | None, split: Split
    ) -> SAGEState:
        """按指定切分 (K,V) 拟合一次（内部用）。"""
        cfg = self.config
        dout, din = W.shape
        device = W.device
        v = cfg.v
        if (dout * din) % cfg.group_size != 0:
            raise ValueError(
                f"权重元素数 {dout*din} 必须是 group_size={cfg.group_size} 的整数倍"
                "（G=128 时要求 dout*din 能被 128 整除）"
            )

        # 1) ESG 置换
        if cfg.esg and energy is not None:
            e = energy.to(device)
            permutation = esg_permutation(e, v)
            e_perm = e[permutation]
        else:
            permutation = torch.arange(din, device=device)
            e_perm = torch.ones(din, device=device, dtype=W.dtype)

        W_perm = W[:, permutation]

        # 2) 分解
        neg, mag, _ = decompose(W_perm, v)

        # 3) 通道权重（ESG 拟合用 e^γ，否则均匀）
        # 每个输入通道的权重对所有输出行相同，故按行铺展到 (N, v)。
        if cfg.esg and energy is not None:
            w_col = energy_weights(e_perm.to(W.dtype), v, cfg.energy_power, cfg.energy_eps)  # (din/v, v)
            omega = w_col.unsqueeze(0).expand(dout, -1, -1).reshape(-1, v).contiguous()
        else:
            omega = torch.ones_like(mag)

        # 4) 加权球面 k-means
        C = weighted_spherical_kmeans(
            mag, omega, split.codebook_size,
            iters=cfg.kmeans_iters, seed=cfg.seed, sample_size=cfg.sample_size, chunk=cfg.chunk,
        )

        # 5) 赋值 + 连续半径 + 半径网格量化
        z, r_cont = weighted_assignment(mag, omega, C, chunk=cfg.chunk)
        r_hat, grid_min, grid_step, codes = radius_grid_quantize(
            r_cont, split.radius_bits, cfg.subs_per_group
        )

        state = SAGEState(
            shape=(dout, din), v=v, split=split, permutation=permutation,
            neg=neg, direction_index=z, codewords=C,
            radius_codes=codes, grid_min=grid_min, grid_step=grid_step,
            r_cont=r_cont,
        )
        state.score = self.activation_weighted_error(W, energy, state)
        return state

    # -------------------------------------------------------------- ABS 选择
    def prepare(self, W: torch.Tensor, energy: torch.Tensor | None = None) -> SAGEState:
        """拟合权重矩阵：ESG 分组 +（可选）ABS 逐矩阵选切分。

        Args:
            W: (dout, din) 权重。
            energy: (din,) 输入通道能量；为 None 时退化为均匀权重、恒等置换。
        Returns:
            最优 SAGEState（`state.score` 为其激活加权误差，越小越好）。
        """
        cfg = self.config
        splits = cfg.splits
        if not cfg.adaptive_bits or len(splits) == 1:
            return self._fit_split(W, energy, splits[0])
        best: SAGEState | None = None
        for split in splits:
            cand = self._fit_split(W, energy, split)
            if best is None or cand.score < best.score:
                best = cand
        assert best is not None
        return best

    # ------------------------------------------------------------------ 解码
    def dequantize(self, state: SAGEState) -> torch.Tensor:
        """由状态重建权重矩阵（原始通道顺序）。"""
        r_hat = expand_radius(
            state.radius_codes, state.grid_min, state.grid_step, self.config.subs_per_group
        )
        vec = reconstruct_vectors(state.neg, state.codewords, state.direction_index, r_hat)
        W_perm = vec.reshape(state.shape)
        return unpermute_columns(W_perm, state.permutation)

    # ------------------------------------------------------------------ 评分
    def activation_weighted_error(
        self, W: torch.Tensor, energy: torch.Tensor | None, state: SAGEState
    ) -> float:
        """激活加权重建误差（Eq.8）：Σ_c e_c‖W_c−Ŵ_c‖² / Σ_c e_c‖W_c‖²。

        energy 为 None 时退化为普通 Frobenius 相对误差。
        """
        W_hat = self.dequantize(state)
        diff = (W - W_hat).float()
        if energy is None:
            num = (diff * diff).sum()
            den = (W.float() * W.float()).sum().clamp_min(1e-12)
        else:
            e = energy.to(W.device).float().unsqueeze(0)   # (1, din) 原始能量
            num = (e * diff * diff).sum()
            den = (e * W.float() * W.float()).sum().clamp_min(1e-12)
        return float((num / den).item())

    # ------------------------------------------------------------------ bpp
    def bpp(self, state: SAGEState) -> dict[str, float]:
        """编码位宽统计（Eq.15）。

        返回：
            payload: 仅 符号+方向索引+半径码 的 bit/权重；
            grid:    半径网格参数开销（2×16bit/group_size）；
            codebook: 方向码本开销（K×v×16bit / 权重数）；
            total:   合计。
        """
        cfg = self.config
        dout, din = state.shape
        payload = state.split.bpw(cfg.v)
        grid = 2.0 * _GRID_PARAM_BITS / cfg.group_size
        codebook = state.split.codebook_size * cfg.v * _CODEBOOK_PARAM_BITS / (dout * din)
        return {"payload": payload, "grid": grid, "codebook": codebook,
                "total": payload + grid + codebook}
