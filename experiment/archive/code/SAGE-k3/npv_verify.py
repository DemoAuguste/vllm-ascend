"""SAGE 在 Ascend NPU 上的验证脚本（数值一致性 + 性能）。

demo.py / pytest 已覆盖纯 CPU 数值正确性；本脚本聚焦 NPU 可用性与性能：
  1. NPU 可见性与 device_count
  2. CPU vs NPU 编解码一致性（相同 seed）
  3. 编码吞吐：Qwen3-0.6B down_proj 规模（1024×3072）与 27B MLP 规模（5120×17408）
  4. ABS 选择与 bpp（NPU 上拟合 + 评分）
  5. 单层 blockwise 优化耗时/收敛

运行：ASCEND_RT_VISIBLE_DEVICES=<card> python npv_verify.py
"""

from __future__ import annotations

import os
import time

import torch
import torch_npu  # noqa: F401  注册 npu 后端

from sage import BlockwiseConfig, SAGEConfig, SAGEQuantizer, Split, blockwise_optimize, channel_energy
from sage.utils import log

DEV = os.environ.get("NPU_DEVICE", "npu:0")
SPLITS = (Split(64, 4), Split(128, 3), Split(256, 2))


def make_weight(dout, din, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dout, din, generator=g) * 0.02


def cfg(splits=SPLITS, adaptive=True, seed=0, iters=20, chunk=8192):
    return SAGEConfig(v=8, group_size=128, splits=splits, adaptive_bits=adaptive,
                      kmeans_iters=iters, seed=seed, chunk=chunk)


def main():
    log("=" * 70)
    log("[1] NPU 环境")
    log("=" * 70)
    log("ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(all)"))
    log("torch_npu:", torch_npu.__version__)
    log("npu device_count:", torch.npu.device_count())
    if torch.npu.device_count() == 0:
        log("!! 无可见 NPU，退出")
        return
    torch.npu.set_device(0)
    log("current npu device:", torch.npu.current_device())

    # ---------------------------------------------------------------- [2]
    log("\n" + "=" * 70)
    log("[2] CPU vs NPU 编解码一致性")
    log("=" * 70)
    torch.manual_seed(0)
    W = make_weight(512, 1024, seed=0)
    energy = channel_energy(torch.randn(2048, 1024, generator=torch.Generator().manual_seed(1)))
    qz_cpu = SAGEQuantizer(cfg(splits=(Split(64, 2),), adaptive=False))
    st_cpu = qz_cpu.prepare(W, energy)
    Wh_cpu = qz_cpu.dequantize(st_cpu)

    # NPU 侧：能量与权重搬过去，同 seed 拟合
    energy_dev = energy.to(DEV)
    qz_npu = SAGEQuantizer(cfg(splits=(Split(64, 2),), adaptive=False, seed=0))
    t0 = time.time()
    st_npu = qz_npu.prepare(W.to(DEV), energy_dev)
    dt = time.time() - t0
    Wh_npu = qz_npu.dequantize(st_npu).cpu()
    diff = (Wh_cpu - Wh_npu).abs().max().item()
    rel = float((W - Wh_npu).norm() / W.norm())
    log(f"NPU 拟合+编解码耗时 {dt:.3f}s（W {tuple(W.shape)}，N={W.numel()//8} 子向量）")
    log(f"CPU vs NPU 重建最大绝对差 {diff:.3e}")
    log(f"相对 Frobenius 误差 {rel:.6f}")
    log("一致性:", "PASS" if diff < 1e-3 else "FAIL")

    # ---------------------------------------------------------------- [3]
    log("\n" + "=" * 70)
    log("[3] 编码吞吐（单切分 64/4）")
    log("=" * 70)
    for tag, (dout, din) in {
        "Qwen3-0.6B down_proj": (1024, 3072),
        "27B MLP down_proj": (5120, 17408),
    }.items():
        Wt = make_weight(dout, din, seed=2)
        e = channel_energy(torch.randn(1024, din, generator=torch.Generator().manual_seed(3)))
        q = SAGEQuantizer(cfg(splits=(Split(64, 4),), adaptive=False, iters=20))
        t0 = time.time()
        st = q.prepare(Wt.to(DEV), e.to(DEV))
        Wh = q.dequantize(st)
        dt = time.time() - t0
        n_sub = Wt.numel() // 8
        # 触发设备同步以得到真实耗时
        torch.npu.synchronize()
        log(f"  {tag:<24} {str((dout,din)):<16} N={n_sub:>10d}  "
            f"耗时 {dt:.2f}s  吞吐 {n_sub/dt/1e6:.2f}M 子向量/s")

    # ---------------------------------------------------------------- [4]
    log("\n" + "=" * 70)
    log("[4] ABS 选择与 bpp（NPU）")
    log("=" * 70)
    W2 = make_weight(1024, 3072, seed=4)
    e2 = channel_energy(torch.randn(1024, 3072, generator=torch.Generator().manual_seed(5)))
    q_abs = SAGEQuantizer(cfg(SPLITS, adaptive=True, iters=20))
    t0 = time.time()
    st = q_abs.prepare(W2.to(DEV), e2.to(DEV))
    torch.npu.synchronize()
    log(f"ABS 三切分拟合耗时 {time.time()-t0:.2f}s，选中 ({st.split.codebook_size},{st.split.radius_bits})")
    b = q_abs.bpp(st)
    log(f"payload {b['payload']:.3f} bpw  + 网格 {b['grid']:.3f}  + 码本 {b['codebook']:.4f} "
        f"= total {b['total']:.3f} bpw")

    # ---------------------------------------------------------------- [5]
    log("\n" + "=" * 70)
    log("[5] 单层 blockwise 优化（NPU）")
    log("=" * 70)
    d = 1024
    Wb = make_weight(d, d, seed=6)
    X = torch.randn(512, d, generator=torch.Generator().manual_seed(7)) * 0.5
    Y = X @ Wb.t()
    e_b = channel_energy(X)
    q = SAGEQuantizer(cfg(splits=(Split(32, 2),), adaptive=False, iters=10))
    st = q.prepare(Wb.to(DEV), e_b.to(DEV))
    before = float(torch.nn.functional.mse_loss(X.to(DEV) @ q.dequantize(st).t(), Y.to(DEV)))
    t0 = time.time()
    st = blockwise_optimize(st, X.to(DEV), Y.to(DEV),
                            BlockwiseConfig(epochs=20, batch_size=64, lr=1e-2, seed=0),
                            subs_per_group=16, log=log)
    torch.npu.synchronize()
    after = float(torch.nn.functional.mse_loss(X.to(DEV) @ q.dequantize(st).t(), Y.to(DEV)))
    log(f"blockwise 20 epoch 耗时 {time.time()-t0:.2f}s")
    log(f"输出 MSE：before {before:.6e} → after {after:.6e}（降低 {100*(1-after/before):.1f}%）")

    log("\nNPU 验证完成")


if __name__ == "__main__":
    main()
