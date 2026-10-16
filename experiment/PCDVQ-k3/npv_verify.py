"""PCDVQ 在 Ascend NPU 环境上的验证脚本。

验证目标（demo.py 已覆盖纯 CPU 数值正确性，本脚本聚焦 NPU 部署/可用性）：
  1. NPU 可见性与 device_count
  2. 码本构造（E8 枚举 + 贪心 FPS + chi Lloyd-Max）耗时 —— 构造保持 CPU，
     避免 NPU 上 2^a 次小 kernel 与 .item() 同步灾难
  3. CPU vs NPU 量化/反量化结果一致性（相同 seed 应极小误差一致）
  4. 分块方向量化：避免大矩阵 (N, 2^a) OOM，并验证分块与全量结果一致
  5. Qwen3-0.6B 真实规模（down_proj 2816x1024）的分块量化误差/耗时/吞吐

选择 a=10（BPW=1.5）作为默认以快速打通链路；a=14 档位（BPW=2.0）的码本构造
实测在 CPU 高于 5 分钟（E8 meshgrid 枚举 + FPS 的 O(n^2) 是瓶颈），属离线一次性
成本，后续按需优化，不在此脚本重复构造。

运行：ASCEND_RT_VISIBLE_DEVICES=<card> python npv_verify.py
"""

from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401

from pcdvq import (
    PCDVQ,
    RHTRegularizer,
    e8_unit_directions,
    greedy_direction_codebook,
    lloyd_max_magnitude_codebook,
)

DEV = os.environ.get("NPU_DEVICE", "npu:0")
K, A, B = 8, 10, 2


def log(*a):
    print(*a, flush=True)


def vq_errors(v, v_hat):
    mag = v.norm(dim=1)
    mag_hat = v_hat.norm(dim=1)
    cos = ((v * v_hat).sum(1) / (mag * mag_hat).clamp_min(1e-12)).clamp(-1.0, 1.0)
    dir_mse = float((2 * mag ** 2 * (1 - cos)).mean())
    mag_mse = float(((mag - mag_hat) ** 2).mean())
    total = float(((v - v_hat) ** 2).mean())
    return dir_mse, mag_mse, total


def quantize_direction_blocked(dirs, C_phi, chunk=8192):
    """分块计算方向索引，避免 (N, 2^a) 全量矩阵过大。"""
    n = dirs.shape[0]
    iphi = torch.empty(n, dtype=torch.long, device=dirs.device)
    for s in range(0, n, chunk):
        iphi[s : s + chunk] = (dirs[s : s + chunk] @ C_phi.t()).argmax(dim=1)
    return iphi


def main():
    log("=" * 70)
    log(f"[1] NPU 环境 (k={K}, a={A}, b={B}, BPW={(A+B)/K})")
    log("=" * 70)
    log("ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(all)"))
    log("torch_npu:", torch_npu.__version__)
    log("npu device_count:", torch.npu.device_count())
    if torch.npu.device_count() == 0:
        log("!! no NPU visible, abort")
        return
    torch.npu.set_device(0)
    log("current npu device:", torch.npu.current_device())

    log()
    log("=" * 70)
    log("[2] 码本构造 (CPU)")
    log("=" * 70)
    t0 = time.time()
    pool = e8_unit_directions(4)  # norm^2<=4 -> 2400 单位方向
    log(f"E8 方向池 norm^2<=4: {tuple(pool.shape)}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    cb_dir = greedy_direction_codebook(pool, A, seed=0)
    log(f"方向码本 2^{A}={2**A}: {tuple(cb_dir.shape)}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    cb_mag = lloyd_max_magnitude_codebook(B, K, tau=0.999)
    log(f"幅值码本 2^{B}={2**B}: {[round(float(x),4) for x in cb_mag]}  ({time.time()-t0:.1f}s)")

    def make_qtz(dev: str) -> PCDVQ:
        qz = PCDVQ(k=K, a=A, b=B, max_norm_sq=4, seed=0, device=dev)
        qz.C_phi = cb_dir.to(dev)
        qz.C_r = cb_mag.to(dev)
        qz.pool_size = cb_dir.shape[0]
        return qz

    log()
    log("=" * 70)
    log("[3] CPU vs NPU 量化/反量化一致性")
    log("=" * 70)
    torch.manual_seed(0)
    W = torch.randn(512, 1024) * 0.02
    qz_cpu = make_qtz("cpu")
    qz_npu = make_qtz(DEV)
    Wh_cpu = qz_cpu.dequantize(qz_cpu.quantize(W))
    t0 = time.time()
    Wh_npu = qz_npu.dequantize(qz_npu.quantize(W.to(DEV)))
    npu_dt = time.time() - t0
    Wh_npu_cpu = Wh_npu.cpu()
    diff = (Wh_cpu - Wh_npu_cpu).abs().max().item()
    rel = float((W - Wh_npu_cpu).norm() / W.norm())
    log(f"NPU 量化+反量化耗时: {npu_dt:.3f}s  (W {tuple(W.shape)}, N={W.numel()//K} 向量)")
    log(f"CPU vs NPU 重建最大绝对差: {diff:.3e}")
    log(f"相对 Frobenius 误差: {rel:.6f}")
    log("一致性:", "PASS" if diff < 1e-3 else "FAIL")

    log()
    log("=" * 70)
    log("[4] 分块方向量化：一致性 + Qwen3-0.6B down_proj 规模")
    log("=" * 70)
    # 小矩阵：分块 vs 全量索引一致（先 reshape 成分组向量）
    Y, _ = RHTRegularizer(1024, seed=0).forward(W)
    g_small = Y.reshape(-1, K)  # (N, k)
    dirs = (g_small / g_small.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(DEV)
    iphi_full = (dirs @ cb_dir.to(DEV).t()).argmax(dim=1)
    iphi_blk = quantize_direction_blocked(dirs, cb_dir.to(DEV), chunk=8192)
    log(f"小矩阵 N={dirs.shape[0]}: 全量 vs 分块索引一致率 {(iphi_full == iphi_blk).float().mean().item():.4f}")

    # Qwen3-0.6B down_proj 规模 (2816 x 1024)
    torch.manual_seed(1)
    Wb = torch.randn(2816, 1024) * 0.02
    Nb = Wb.numel() // K
    t0 = time.time()
    reg_b = RHTRegularizer(1024, device=DEV, seed=0)
    Yb, scb = reg_b.forward(Wb.to(DEV))  # (2816, 1024)
    Gb = Yb.reshape(-1, K)               # (360448, 8)
    rb = Gb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dirsb = Gb / rb
    iphi_b = quantize_direction_blocked(dirsb, cb_dir.to(DEV), chunk=8192)
    ir_b = (rb - cb_mag.to(DEV).unsqueeze(0)).abs().argmin(dim=1)
    dt_blk = time.time() - t0
    log(f"down_proj 规模 N={Nb} 向量: 分块量化耗时 {dt_blk:.2f}s (吞吐 {Nb/dt_blk/1e6:.2f}M 向量/s)")

    # 反量化回权重空间，算相对误差
    d_hat = cb_dir.to(DEV)[iphi_b]
    r_hat = cb_mag.to(DEV)[ir_b].unsqueeze(1)
    Y_hat = (r_hat * d_hat).reshape(2816, 1024)
    Wb_hat = reg_b.inverse(Y_hat, scb).cpu()
    rel_b = float((Wb - Wb_hat).norm() / Wb.norm())
    log(f"down_proj 相对 Frobenius 误差: {rel_b:.6f}")

    log()
    log("=" * 70)
    log("[5] 误差分解（RHT 分组空间，down_proj 规模）")
    log("=" * 70)
    G = Yb.reshape(-1, K).cpu()
    Gh = Y_hat.reshape(-1, K).cpu()
    d_mse, m_mse, t_mse = vq_errors(G, Gh)
    log(f"方向 MSE: {d_mse:.6f}  幅值 MSE: {m_mse:.6f}  总 MSE: {t_mse:.6f}")

    log()
    log("NPU 验证完成")


if __name__ == "__main__":
    main()