"""GPTQ (OBQ-style) quantization of a single linear layer at several bitwidths from one shared Hessian."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from src.quant import dist_utils
from src.quant.quant_utils import dequantize, find_params, quantize


try:
    torch.backends.cuda.matmul.allow_tf32 = False
except Exception:
    pass  # NPU 无该后端
try:
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass


class FastOBQ:

    def __init__(
        self,
        layer: nn.Linear,
        bitwidth_options: List[int],
        group_size: int = 128,
        sym: bool = False,
        perchannel: bool = True,
        rel_damp: float = 1e-2,
    ):
        assert isinstance(layer, nn.Linear), "FastOBQ supports only nn.Linear layers."
        self.layer = layer
        self.bitwidth_options = bitwidth_options
        self.d_row, self.d_col = layer.weight.shape
        assert self.d_col % group_size == 0, f"in_features ({self.d_col}) must be divisible by group_size ({group_size})"
        self.group_size = group_size
        self.sym = sym
        self.perchannel = perchannel
        self.rel_damp = rel_damp
        self.H = None
        self.num_samples = 0

    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """Update the running Hessian estimate 2 X^T X / n with a batch of layer inputs."""
        batch_size = input.shape[0]
        input = input.reshape(-1, input.shape[-1]).float()
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        n = self.num_samples + batch_size
        self.H.addmm_(input.T, input, beta=self.num_samples / n, alpha=2.0 / n)
        self.num_samples = n

    @torch.no_grad()
    def _prepare(self) -> Tuple[Tensor, Tensor]:
        """Dampened Hessian -> upper Cholesky factor of its inverse, plus the float32 weight."""
        H, W = self.H, self.layer.weight.float().clone()
        # input channels that never fire get unit curvature and zero weight
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        H.diagonal().add_(self.rel_damp * torch.diag(H).mean())
        W[:, dead] = 0
        # all-zero weight columns are decoupled from the rest of the Hessian
        zero_cols = (W == 0).all(0)
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # NPU 不支持 torch.linalg；按 AMCT 经验在 CPU 求逆后回迁
        dev = H.device
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H.cpu()))
        H_cho = torch.linalg.cholesky(H_inv, upper=True).to(dev)
        return W, H_cho

    @torch.no_grad()
    def quantize(self) -> Dict[int, Tuple[Tensor, Tensor, Tensor]]:
        """Return {bits: (qweight uint8 [d_row, d_col], scales, zeros [d_row, n_groups])}, identical on all ranks."""
        assert self.H is not None, "No calibration samples were processed."
        if dist_utils.is_dist_available_and_initialized():
            dist.all_reduce(self.H, op=dist.ReduceOp.AVG)
        d_row, d_col, gs = self.d_row, self.d_col, self.group_size
        device, dtype = self.layer.weight.device, self.layer.weight.dtype
        out = {
            bits: (
                torch.empty(d_row, d_col, device=device, dtype=torch.uint8),
                torch.empty(d_row, d_col // gs, device=device, dtype=dtype),
                torch.empty(d_row, d_col // gs, device=device, dtype=dtype),
            )
            for bits in self.bitwidth_options
        }
        if dist_utils.is_main():
            W, H_inv_cho = self._prepare()
            for bits, (qweight, scales, zeros) in out.items():
                maxq = 2**bits - 1
                w = W.clone()
                # one GPTQ block per group, so group params see all earlier error feedback
                for c1 in range(0, d_col, gs):
                    c2 = c1 + gs
                    w_blk = w[:, c1:c2]
                    scale, zero = find_params(w_blk, bits, self.sym, self.perchannel)
                    # quantize with the stored-precision scale so saved weights match the error feedback
                    scale = scale.to(dtype).float()
                    scales[:, c1 // gs] = scale.flatten()
                    zeros[:, c1 // gs] = zero.flatten()
                    errs = torch.empty_like(w_blk)
                    H_blk = H_inv_cho[c1:c2, c1:c2]
                    for i in range(gs):
                        q = quantize(w_blk[:, i : i + 1], scale, zero, maxq)
                        qweight[:, c1 + i] = q.flatten()
                        err = (w_blk[:, i] - dequantize(q, scale, zero).flatten()) / H_blk[i, i]
                        w_blk[:, i:].addr_(err, H_blk[i, i:], alpha=-1)
                        errs[:, i] = err
                    w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)
        if dist_utils.is_dist_available_and_initialized():
            for tensors in out.values():
                for t in tensors:
                    dist.broadcast(t, src=0)
        self.H = None
        return out
