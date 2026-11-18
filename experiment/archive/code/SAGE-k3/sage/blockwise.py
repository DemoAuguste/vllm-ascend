"""逐层 blockwise 误差补偿优化（论文 §3.4）。

官方 ParoQuant 流程：按块序贯、以全精度块输出为目标，学生输入取"前序量化块"
的输出；只优化连续半径参数 ρ 与码本角 logits α（符号、方向索引、置换、切分固定），
半径取整通过 STE 直通。

本模块实现**单层**的 blockwise 优化：
    min_{α, ρ}  E_{(x, y)~D}[ ‖ F̂(x) − y ‖² ]
其中 F̂ 用当前 α/ρ 重建的量化权重做线性变换，y 为该层全精度输出（teacher）。
跨层序贯（学生输入传播）由上层 `model_quant.py` 负责组织。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .quantizer import SAGEState
from .spherical import (
    angles_to_codewords,
    codewords_to_angles,
    radius_grid_quantize,
    reconstruct_vectors,
    unpermute_columns,
)
from .utils import softplus_inverse


@dataclass
class BlockwiseConfig:
    """blockwise 优化超参（对齐论文/官方）。"""

    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 16
    seed: int = 0
    log_prefix: str = ""


def _reconstruct_weight(
    alpha: torch.Tensor,
    rho: torch.Tensor,
    state: SAGEState,
    subs_per_group: int,
    ste: bool,
) -> torch.Tensor:
    """由当前 α/ρ 重建权重矩阵（原始通道顺序）。"""
    codewords = angles_to_codewords(alpha)
    r_cont = torch.nn.functional.softplus(rho)
    r_hat, _, _, _ = radius_grid_quantize(r_cont, state.split.radius_bits, subs_per_group, ste=ste)
    vec = reconstruct_vectors(state.neg, codewords, state.direction_index, r_hat)
    W_perm = vec.reshape(state.shape)
    return unpermute_columns(W_perm, state.permutation)


@torch.enable_grad()
def blockwise_optimize(
    state: SAGEState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: BlockwiseConfig | None = None,
    subs_per_group: int = 16,
    log=None,
) -> SAGEState:
    """对单层做 blockwise 误差补偿，就地更新并返回 state。

    Args:
        state: 已拟合的 SAGEState（提供初始码本与连续半径）。
        inputs: (T, din) 该层校准输入激活（原始通道顺序）。
        targets: (T, dout) 该层全精度输出（teacher，含 bias 更佳）。
        config: 优化超参。
        subs_per_group: G/v。
        log: 可选的日志函数。
    Returns:
        更新了 codewords / 半径码 / 网格 / score 的 state（新对象）。
    """
    cfg = config or BlockwiseConfig()
    device = inputs.device
    inputs = inputs.to(device)
    targets = targets.to(device)
    t = inputs.shape[0]
    if t == 0:
        return state

    # 统一设备（state 可能来自 CPU 拟合，inputs 在 NPU 上）
    state.neg = state.neg.to(device)
    state.direction_index = state.direction_index.to(device)
    state.permutation = state.permutation.to(device)

    def discrete_mse(a: torch.Tensor, r: torch.Tensor) -> float:
        """用离散（非 STE）重建评估全量输出 MSE。"""
        with torch.no_grad():
            W_hat = _reconstruct_weight(a, r, state, subs_per_group, ste=False)
            total = 0.0
            nb = 0
            for start in range(0, t, cfg.batch_size):
                stop = min(start + cfg.batch_size, t)
                pred = inputs[start:stop] @ W_hat.t()
                total += float(torch.nn.functional.mse_loss(pred, targets[start:stop]))
                nb += 1
            return total / max(1, nb)

    alpha = codewords_to_angles(state.codewords.to(device)).requires_grad_(True)
    rho = softplus_inverse(state.r_cont.to(device)).requires_grad_(True)

    # 初始离散状态即作为候选，保证优化结果不劣于拟合初值（对齐官方 best-state 语义）
    best_alpha = alpha.detach().clone()
    best_rho = rho.detach().clone()
    best_loss = discrete_mse(best_alpha, best_rho)

    optimizer = torch.optim.Adam([alpha, rho], lr=cfg.lr)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for epoch in range(cfg.epochs):
        order = torch.randperm(t, generator=gen).to(device)
        for start in range(0, t, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            W_hat = _reconstruct_weight(alpha, rho, state, subs_per_group, ste=True)
            loss = torch.nn.functional.mse_loss(inputs[idx] @ W_hat.t(), targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        cur = discrete_mse(alpha.detach(), rho.detach())
        if cur < best_loss:
            best_loss = cur
            best_alpha = alpha.detach().clone()
            best_rho = rho.detach().clone()
        if log is not None and (epoch == 0 or epoch == cfg.epochs - 1):
            log(f"    {cfg.log_prefix}epoch {epoch+1}/{cfg.epochs} discrete_mse={cur:.6e} best={best_loss:.6e}")

    # 回写最优离散参数
    with torch.no_grad():
        state.codewords = angles_to_codewords(best_alpha)
        r_cont = torch.nn.functional.softplus(best_rho)
        _, grid_min, grid_step, codes = radius_grid_quantize(
            r_cont, state.split.radius_bits, subs_per_group
        )
        state.radius_codes = codes
        state.grid_min = grid_min
        state.grid_step = grid_step
        state.r_cont = r_cont
    return state
