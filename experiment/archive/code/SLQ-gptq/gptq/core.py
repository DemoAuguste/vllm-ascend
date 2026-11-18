# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# GPTQ: Hessian 二阶误差补偿的权重量化算子（CPU/NPU 通用，支持任意 int 位宽）。
#
# 移植并泛化自昇腾 AMCT（Ascend Model Compression Toolkit, Apache-2.0）
# `amct_pytorch/algorithms/quant/gptq.py`（Huawei Technologies, 2026）：
#   - observe_input 在线累积输入 Hessian（含 sqrt(2/nsamples) 缩放）
#   - quantize 列式分块（block_size=128）二阶误差补偿；Hessian 求逆走 CPU
#     （NPU 不支持 torch.linalg）后再回迁；dead 列置零
# 差异：位宽由 {4,8} 泛化到任意 int；位宽/分组/阻尼等为显式参数。
# ----------------------------------------------------------------------------
from __future__ import annotations

import math

import torch


class GPTQ:
    """单个权重矩阵的 GPTQ 量化器（先 observe 校准激活，再 quantize）。"""

    def __init__(self, weight_shape: tuple[int, int], bits: int = 4,
                 group_size: int | None = None, damp: float = 0.01,
                 block_size: int = 128, asym: bool = False):
        if len(weight_shape) != 2:
            raise ValueError(f"GPTQ 期望 2D 权重，得到 {weight_shape}")
        if bits < 2 or bits > 8:
            raise ValueError(f"位宽需在 [2, 8]，得到 {bits}")
        self.rows, self.columns = int(weight_shape[0]), int(weight_shape[1])
        self.bits = int(bits)
        self.group_size = group_size
        self.perc_damp = damp
        self.block_size = block_size
        self.asym = bool(asym)
        self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32)
        self.nsamples = 0

    # ------------------------------------------------------------------ 观测
    @torch.no_grad()
    def observe(self, X: torch.Tensor) -> None:
        """累积输入激活 Hessian。X: (T, din) 或 (B, S, din)。"""
        if X.ndim == 2:
            X = X.unsqueeze(0)
        batch = X.shape[0]
        Xi = X.reshape(-1, X.shape[-1]).t().float()          # (din, T)
        self.H = self.H.to(Xi.device)                        # 对齐设备（NPU/CPU）
        total = self.nsamples + batch
        if total <= 0:
            return
        if self.nsamples > 0:
            self.H.mul_(self.nsamples / total)
        self.nsamples = total
        Xi = math.sqrt(2.0 / total) * Xi
        self.H.add_(Xi @ Xi.t())
        if not torch.isfinite(self.H).all():
            raise RuntimeError("GPTQ Hessian 含 inf/nan")

    # -------------------------------------------------------------- 求逆辅助
    @staticmethod
    def _hessian_inverse(H: torch.Tensor, perc_damp: float) -> torch.Tensor:
        n = H.shape[0]
        H = H.clone()
        damp = perc_damp * torch.mean(torch.diag(H))
        idx = torch.arange(n, device=H.device)
        H[idx, idx] += damp
        device = H.device
        H = H.cpu()  # NPU 不支持 torch.linalg
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
        return Hinv.to(device)

    def _column_qparams(self, W: torch.Tensor, perm: torch.Tensor,
                        bits: int | None = None):
        """按置换后的列序，给出每列每行的 (scale, zero)：均为 (rows, columns)。

        对称时 zero=0；非对称时 zero=min（SLQ 论文 DL 必需）。
        """
        bits = self.bits if bits is None else bits
        Wf = W.float()
        if self.group_size:
            ng = (self.columns + self.group_size - 1) // self.group_size
            pad = ng * self.group_size - self.columns
            Wp = torch.nn.functional.pad(Wf, (0, pad)) if pad else Wf
            g = Wp.reshape(self.rows, ng, self.group_size)
            if self.asym:
                qmax = float(2 ** bits - 1)
                mx = g.amax(dim=-1)
                mn = g.amin(dim=-1)
                scale = ((mx - mn) / qmax).clamp_min(1e-9)
                zero = mn
            else:
                qmax = float(2 ** (bits - 1) - 1)
                scale = g.abs().amax(dim=-1).clamp_min(1e-9) / qmax
                zero = torch.zeros_like(scale)
            grp = perm // self.group_size
            return scale[:, grp], zero[:, grp]
        if self.asym:
            qmax = float(2 ** bits - 1)
            mx = Wf.amax(dim=-1, keepdim=True)
            mn = Wf.amin(dim=-1, keepdim=True)
            scale = ((mx - mn) / qmax).clamp_min(1e-9)
            zero = mn
        else:
            qmax = float(2 ** (bits - 1) - 1)
            scale = Wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / qmax
            zero = torch.zeros_like(scale)
        return scale.expand(self.rows, self.columns), zero.expand(self.rows, self.columns)

    # -------------------------------------------------------------- 量化主流程
    def _prepare(self):
        """由 Hessian 计算并缓存 (dead, perm, invperm, Hinv)——与位宽无关，可复用。"""
        if getattr(self, "_prepared", None) is not None:
            return self._prepared
        H = self.H.float()
        dead = torch.diag(H) == 0
        H = H.clone()
        H[dead, dead] = 1
        perm = torch.argsort(torch.diag(H), descending=True)
        invperm = torch.argsort(perm)
        Hp = H[perm][:, perm]
        Hinv = self._hessian_inverse(Hp, self.perc_damp)
        self._prepared = (dead, perm, invperm, Hinv)
        return self._prepared

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, bits: int | None = None) -> torch.Tensor:
        """由已观测 Hessian 做 GPTQ 量化-反量化，返回量化后的浮点权重（原列序）。

        `bits` 可覆盖实例位宽（复用同一 Hessian/置换/求逆，多档位高效）。
        """
        if self.nsamples == 0:
            raise RuntimeError("GPTQ 需先 observe 校准激活")
        bits = self.bits if bits is None else int(bits)
        dead, perm, invperm, Hinv = self._prepare()
        W = W.clone().float()
        W[:, dead.to(W.device)] = 0

        scale_all, zero_all = self._column_qparams(W, perm, bits)
        Wp = W[:, perm]
        qmax = float(2 ** bits - 1) if self.asym else float(2 ** (bits - 1) - 1)
        qmin = 0.0 if self.asym else -qmax
        Hinv = Hinv.to(W.device)
        for i1 in range(0, self.columns, self.block_size):
            i2 = min(i1 + self.block_size, self.columns)
            wb = Wp[:, i1:i2]
            errb = torch.zeros_like(wb)
            hb = Hinv[i1:i2, i1:i2]
            for j in range(i2 - i1):
                wcol = wb[:, j]
                d = hb[j, j]
                s = scale_all[:, i1 + j]
                z = zero_all[:, i1 + j]
                wq = ((wcol - z) / s).round().clamp(qmin, qmax) * s + z
                err = (wcol - wq) / d
                wb[:, j:] -= err.unsqueeze(1) @ hb[j, j:].unsqueeze(0)
                errb[:, j] = err
            Wp[:, i2:] -= errb @ Hinv[i1:i2, i2:]

        return Wp[:, invperm]

    # ------------------------------------------------------------------ 便捷
    @classmethod
    def run(cls, W: torch.Tensor, X: torch.Tensor, bits: int = 4,
            group_size: int | None = None, damp: float = 0.01,
            block_size: int = 128, asym: bool = False) -> torch.Tensor:
        """一次性：observe(X) → quantize(W)。"""
        g = cls(tuple(W.shape), bits=bits, group_size=group_size, damp=damp,
                block_size=block_size, asym=asym)
        g.observe(X)
        return g.quantize(W)
