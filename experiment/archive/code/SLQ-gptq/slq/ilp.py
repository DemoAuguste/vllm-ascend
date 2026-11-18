# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# SLQ 的 ILP 位宽分配（Eq. 全局混合精度分配）。
#
# 目标：min Σ_l α_l(b_l)   s.t.  (1/L) Σ_l b_l ≤ B_avg,  b_l ∈ bits_list
# 其中 α_l(b) 为层 l 在 b-bit 下的敏感度（NLL 增量，来自 sensitivity.py）。
#
# 提供两种求解：
#   - lagrangian_allocate：对偶（λ 二分），O(L·|bits|·iter)，大规模可用；
#   - milp_allocate：scipy.optimize.milp 精确解（层数少时校验用）。
# ----------------------------------------------------------------------------
from __future__ import annotations

import numpy as np


def _monotonic_fix(alpha: dict[int, float], bits_list) -> dict[int, float]:
    """α 随位宽单调不增（去噪）：从高位到低位取累计**最大**值。

    即保证 α(b_low) ≥ α(b_high)，使"多给 bit 不会更差"。
    """
    out = {}
    running = -float("inf")
    for b in sorted(bits_list, reverse=True):
        running = max(running, float(alpha[b]))
        out[b] = max(0.0, running)
    return out


def lagrangian_allocate(sens_list, bits_list, target_avg_bits, iters=200, weights=None):
    """对偶二分求解位宽分配。

    Args:
        sens_list: list[LayerSensitivity]（用 .alpha_by_bits）。
        bits_list: 候选位宽（升序）。
        target_avg_bits: 目标平均位宽（按参数量加权，若有 weights）。
        weights: 每层参数量权重（None = 等权）。
    Returns:
        (assignment: {name: bits}, avg_bits: float, total_alpha: float)
    """
    bits = sorted(bits_list)
    names = [s.name for s in sens_list]
    alpha = [_monotonic_fix(s.alpha_by_bits, bits) for s in sens_list]
    L = len(names)
    assert L > 0
    w = [1.0] * L if weights is None else [float(x) for x in weights]
    wsum = sum(w)

    def choose(lam):
        num, total_alpha, assign = 0.0, 0.0, {}
        for wi, nm, a in zip(w, names, alpha):
            costs = [(a[b] + lam * b, b) for b in bits]
            c, b = min(costs)
            assign[nm] = b
            num += wi * b
            total_alpha += a[b]
        return num / wsum, total_alpha, assign

    lo, hi = 0.0, 1.0
    avg_hi, _, _ = choose(hi)
    while avg_hi > target_avg_bits and hi < 1e12:
        hi *= 4
        avg_hi, _, _ = choose(hi)

    # avg(λ) 随 λ 单调不增；找**最小**的 λ 使 avg ≤ 目标（即在预算内保留最多 bit）
    best = choose(hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        avg, ta, assign = choose(mid)
        if avg <= target_avg_bits:
            best = (avg, ta, assign)
            hi = mid
        else:
            lo = mid
    avg, ta, assign = best
    return assign, float(avg), float(ta)


def milp_allocate(sens_list, bits_list, target_avg_bits):
    """scipy.milp 精确求解（层数较少时使用）。"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix

    bits = sorted(bits_list)
    names = [s.name for s in sens_list]
    L = len(names)
    nb = len(bits)

    # 变量 x[l, k] ∈ {0,1}，选第 k 个位宽
    nvar = L * nb
    c = np.zeros(nvar)
    for l, s in enumerate(sens_list):
        a = _monotonic_fix(s.alpha_by_bits, bits)
        for k, b in enumerate(bits):
            c[l * nb + k] = a[b]

    # 约束1：Σ_k x[l,k] = 1（每层选一个位宽）
    A_eq = lil_matrix((L, nvar))
    for l in range(L):
        A_eq[l, l * nb:(l + 1) * nb] = 1

    # 约束2：Σ b_l ≤ target*L
    a_bits = np.zeros((1, nvar))
    for l in range(L):
        for k, b in enumerate(bits):
            a_bits[0, l * nb + k] = b

    constraints = [
        LinearConstraint(A_eq.tocsr(), 1, 1),
        LinearConstraint(a_bits, -np.inf, target_avg_bits * L),
    ]
    res = milp(c=c, constraints=constraints,
               integrality=np.ones(nvar), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"milp 求解失败: {res.message}")
    assign = {}
    for l, nm in enumerate(names):
        k = int(np.argmax(res.x[l * nb:(l + 1) * nb] > 0.5))
        assign[nm] = bits[k]
    avg = sum(assign.values()) / L
    total_alpha = sum(_monotonic_fix(s.alpha_by_bits, bits)[assign[s.name]] for s in sens_list)
    return assign, float(avg), float(total_alpha)
