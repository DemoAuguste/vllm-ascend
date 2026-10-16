"""码本构造性能 benchmark + 序列化缓存验证。

对 a=14（BPW=2.0，feasibility 关键档）分别计时 E8 枚举 / 贪心 FPS / 幅值码本，
并把构造结果序列化到磁盘复用，验证加载正确性。

运行：python benchmark_codebooks.py
"""

from __future__ import annotations

import os
import time

import torch

from pcdvq import (
    PCDVQ,
    e8_unit_directions,
    greedy_direction_codebook,
    lloyd_max_magnitude_codebook,
)

CACHE_DIR = os.environ.get("CODEBOOK_CACHE", "/opt/zh/q38quant/codebooks")


def log(*a):
    print(*a, flush=True)


def bench_e8(max_norm_sq):
    t0 = time.time()
    pool = e8_unit_directions(max_norm_sq)
    dt = time.time() - t0
    log(f"E8 枚举 norm^2<={max_norm_sq}: {tuple(pool.shape)}  耗时 {dt:.1f}s")
    return pool


def bench_fps(pool, bits, seed=0):
    t0 = time.time()
    cb = greedy_direction_codebook(pool, bits, seed=seed)
    dt = time.time() - t0
    log(f"贪心 FPS 2^{bits}={2**bits}: {tuple(cb.shape)}  耗时 {dt:.1f}s")
    return cb


def main():
    log("=" * 70)
    log("码本构造 benchmark（a=14 / b=2 / k=8）")
    log("=" * 70)

    # E8 方向池 norm^2<=8 -> 26640
    pool = bench_e8(8)
    # FPS 方向码本 a=14 -> 16384
    cb_dir = bench_fps(pool, 14, seed=0)
    # 幅值码本 b=2
    t0 = time.time()
    cb_mag = lloyd_max_magnitude_codebook(2, 8, tau=0.999)
    log(f"Lloyd-Max 幅值码本 2^2=4: {[round(float(x),4) for x in cb_mag]}  耗时 {time.time()-t0:.1f}s")

    # 正确性抽查：方向码本最小两两余弦（应接近 0.7，表明均匀）
    sim = cb_dir @ cb_dir.t()
    mask = ~torch.eye(cb_dir.shape[0], dtype=torch.bool)
    log(f"方向码本最小两两余弦: {sim[mask].max().item():.4f}（越小越均匀，论文 ≈0.71）")

    log()
    log("=" * 70)
    log("序列化缓存 + 加载往返")
    log("=" * 70)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "pcdvq_k8_a14_b2.pt")
    state = {
        "k": 8, "a": 14, "b": 2,
        "C_phi": cb_dir, "C_r": cb_mag, "bpw": 16 / 8,
    }
    t0 = time.time()
    torch.save(state, path)
    dt_save = time.time() - t0
    sz = os.path.getsize(path) / 1024 / 1024
    log(f"保存: {path}  {sz:.1f} MB  耗时 {dt_save:.2f}s")

    t0 = time.time()
    loaded = torch.load(path, weights_only=False)
    dt_load = time.time() - t0
    log(f"加载: 耗时 {dt_load:.2f}s")
    assert torch.equal(loaded["C_phi"], cb_dir) and torch.equal(loaded["C_r"], cb_mag)
    log("加载一致性: PASS")

    log()
    log("=" * 70)
    log("用缓存码本直接构造 PCDVQ（跳过重复构造）")
    log("=" * 70)
    qz = PCDVQ(k=8, a=14, b=2, max_norm_sq=8, seed=0, device="cpu")
    qz.C_phi = loaded["C_phi"]
    qz.C_r = loaded["C_r"]
    qz.pool_size = loaded["C_phi"].shape[0]
    W = torch.randn(128, 1024) * 0.02
    t0 = time.time()
    W_hat = qz.dequantize(qz.quantize(W))
    rel = float((W - W_hat).norm() / W.norm())
    log(f"a=14 量化+反量化: {time.time()-t0:.2f}s  相对 Frobenius {rel:.4f}")
    log("benchmark 完成")


if __name__ == "__main__":
    main()