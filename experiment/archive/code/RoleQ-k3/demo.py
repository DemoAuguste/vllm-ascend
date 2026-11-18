"""RoleQ 最小可验证 demo（CPU 即可运行）。

运行：python demo.py

验证内容：
  A. 精确 ternary 拟合 Eq.2 与穷举最优解一致，且不劣于 TWN 阈值规则
  B. r1 旋转保持"单元轴"尺度不变（reader 行范数、writer 列范数），但混合残差流坐标
  C. 单层 residual block 上比较 per-OC / per-IC / RoleQ 的加权重建与端到端误差
  D. RoleQ-T 消融：GPTQ vs RTN、reader H-度量精修 on/off
  E. resident bpw 复算（对齐论文 1.590 bpw）
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from roleq import (
    exact_ternary_fit,
    levels_per_code,
    quantize_gptq,
    quantize_rtn,
    random_hadamard,
    relative_weighted_error,
    resident_bpw,
    rotate_reader,
    rotate_writer,
)
from roleq.core import refit_row_scales


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def brute_force_ternary_error(v: torch.Tensor) -> float:
    """穷举所有 support 的 OPT 误差（只适合小向量）。"""
    n = v.shape[0]
    best = float("inf")
    for mask in itertools.product([0, 1], repeat=n):
        support = [i for i, m in enumerate(mask) if m]
        if not support:
            continue
        a = v[support].abs().mean()
        approx = torch.zeros_like(v)
        approx[support] = v[support].sign() * a
        best = min(best, float((v - approx).pow(2).sum()))
    return best


def twn_fit(v: torch.Tensor) -> torch.Tensor:
    """TWN 阈值规则：tau = 0.7 * mean|w|。"""
    tau = 0.7 * v.abs().mean()
    codes = torch.where(v.abs() > tau, v.sign(), torch.zeros_like(v))
    support = codes != 0
    if support.any():
        scale = v[support].abs().mean()
        return codes * scale
    return codes


# ---------------------------------------------------------------------------
# A. 精确拟合
# ---------------------------------------------------------------------------
def part_a() -> None:
    sep("A. 精确 ternary 拟合（论文 Eq. 2）")
    g = torch.Generator().manual_seed(0)
    worse_twn = 0
    max_gap = 0.0
    for t in range(200):
        n = int(torch.randint(3, 8, (1,), generator=g))
        v = torch.randn(n, generator=g) * torch.exp(0.8 * torch.randn(n, generator=g))
        codes, scale = exact_ternary_fit(v)
        e_opt = float((v - scale * codes).pow(2).sum())
        e_bf = brute_force_ternary_error(v)
        gap = max(0.0, e_opt - e_bf)
        max_gap = max(max_gap, gap)
        e_twn = float((v - twn_fit(v)).pow(2).sum())
        worse_twn += int(e_opt > e_twn * (1 + 1e-6) + 1e-9)
        assert torch.all((codes == -1) | (codes == 0) | (codes == 1)), "codes not ternary"
    print(f"200 个随机向量：与穷举最优的最大误差偏差 = {max_gap:.3e}")
    print(f"精确拟合劣于 TWN 阈值的次数 = {worse_twn} / 200")
    assert max_gap < 1e-5
    assert worse_twn == 0


# ---------------------------------------------------------------------------
# B. 旋转只混合流坐标，不动单元轴
# ---------------------------------------------------------------------------
def part_b(d: int = 256, seed: int = 0) -> None:
    sep("B. r1 旋转：单元轴尺度不变，残差流坐标被混合")
    g = torch.Generator().manual_seed(seed)
    R = random_hadamard(d, seed=seed + 1)
    W_reader = torch.randn(512, d, generator=g) * torch.exp(torch.randn(512, generator=g)).unsqueeze(1)
    W_writer = torch.randn(d, 512, generator=g) * torch.exp(torch.randn(512, generator=g)).unsqueeze(0)

    reader_row_before = W_reader.norm(dim=1)
    reader_row_after = rotate_reader(W_reader, R).norm(dim=1)
    writer_col_before = W_writer.norm(dim=0)
    writer_col_after = rotate_writer(W_writer, R).norm(dim=0)
    stream_row_before = W_writer.norm(dim=1)
    stream_row_after = rotate_writer(W_writer, R).norm(dim=1)

    print(f"reader 行范数最大相对变化     : {(reader_row_after - reader_row_before).abs().max():.3e}")
    print(f"writer 列范数最大相对变化     : {(writer_col_after - writer_col_before).abs().max():.3e}")
    print(f"writer 行范数（流坐标）最大变化: {(stream_row_after - stream_row_before).abs().max():.3e}")
    assert torch.allclose(reader_row_after, reader_row_before, atol=1e-4)
    assert torch.allclose(writer_col_after, writer_col_before, atol=1e-4)
    assert (stream_row_after - stream_row_before).abs().max() > 1e-2


# ---------------------------------------------------------------------------
# C. residual block 上的放置规则
# ---------------------------------------------------------------------------
def make_block(d: int = 256, d_ff: int = 512, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # reader：每个 unit 的 read vector（行）有重尾幅度
    G1 = torch.randn(d_ff, d, generator=g)
    G1 = G1 / G1.norm(dim=1, keepdim=True)
    row_mag = torch.exp(1.2 * torch.randn(d_ff, generator=g))
    W_in = G1 * row_mag.unsqueeze(1)

    # writer：每个 unit 的 write vector（列）有重尾幅度
    G2 = torch.randn(d, d_ff, generator=g)
    G2 = G2 / G2.norm(dim=0, keepdim=True)
    col_mag = torch.exp(1.2 * torch.randn(d_ff, generator=g))
    W_out = G2 * col_mag.unsqueeze(0)

    # 残差流：通道各向异性 + 离群坐标
    x = torch.randn(512, d, generator=g)
    x = x * torch.exp(0.8 * torch.randn(d, generator=g)).unsqueeze(0)
    x[:, :4] *= 8.0
    return W_in, W_out, x


def run_block(
    W_in,
    W_out,
    X,
    axis_reader,
    axis_writer,
    method,
    rotate=False,
    seed=0,
    bits=1,
    max_candidates=None,
):
    d = X.shape[1]
    R = random_hadamard(d, seed=seed) if rotate else None
    if rotate:
        W_in, W_out, X = rotate_reader(W_in, R), rotate_writer(W_out, R), X @ R

    H_in = X.T @ X / X.shape[0]
    H_out_raw = F.silu(X @ W_in.T)
    H_out = H_out_raw.T @ H_out_raw / X.shape[0]

    if method == "rtn":
        Wq_in, _, _ = quantize_rtn(W_in, axis_reader, bits=bits, max_candidates=max_candidates)
        Wq_out, _, _ = quantize_rtn(W_out, axis_writer, bits=bits, max_candidates=max_candidates)
    else:
        Wq_in, _, _ = quantize_gptq(
            W_in, H_in, axis_reader, bits=bits, max_candidates=max_candidates
        )
        Wq_out, _, _ = quantize_gptq(
            W_out, H_out, axis_writer, bits=bits, max_candidates=max_candidates
        )

    err_in = relative_weighted_error(W_in, Wq_in, H_in)
    err_out = relative_weighted_error(W_out, Wq_out, H_out)

    Y = X + F.silu(X @ W_in.T) @ W_out.T
    Yq = X + F.silu(X @ Wq_in.T) @ Wq_out.T
    if rotate:
        Y, Yq = Y @ R.T, Yq @ R.T
    e2e = float((Yq - Y).norm() / Y.norm())
    return err_in, err_out, e2e


def part_c() -> None:
    sep("C. 单层 residual block：放置规则（相对误差，越低越好）")
    W_in, W_out, X = make_block()
    configs = [
        ("uniform per-OC", "row", "row"),
        ("uniform per-IC", "col", "col"),
        ("RoleQ", "row", "col"),
    ]
    for rotate in (False, True):
        print(f"\n--- {'r1 旋转基' if rotate else '原始基'} ---")
        print(f"{'placement':<18}{'reader err':>12}{'writer err':>12}{'end-to-end':>12}")
        results = {}
        for name, ar, aw in configs:
            err_in, err_out, e2e = run_block(W_in, W_out, X, ar, aw, "gptq", rotate=rotate)
            results[name] = (err_in, err_out, e2e)
            print(f"{name:<18}{err_in:>12.4f}{err_out:>12.4f}{e2e:>12.4f}")
        best_uniform = min(results["uniform per-OC"][2], results["uniform per-IC"][2])
        gain = best_uniform / results["RoleQ"][2]
        print(f"RoleQ 相对最佳 uniform 的端到端误差改善: {gain:.2f}x")
        assert results["RoleQ"][2] < best_uniform


# ---------------------------------------------------------------------------
# D. 求解器消融
# ---------------------------------------------------------------------------
def part_d() -> None:
    sep("D. 求解器消融：GPTQ vs RTN，以及 reader 精修")
    W_in, W_out, X = make_block()
    H_in = X.T @ X / X.shape[0]
    H_out_raw = F.silu(X @ W_in.T)
    H_out = H_out_raw.T @ H_out_raw / X.shape[0]

    for label, W, H, axis in (
        ("reader (W_in)", W_in, H_in, "row"),
        ("writer (W_out)", W_out, H_out, "col"),
    ):
        rtn = relative_weighted_error(W, quantize_rtn(W, axis)[0], H)
        gptq = relative_weighted_error(W, quantize_gptq(W, H, axis)[0], H)
        print(f"{label:<16} RTN={rtn:.4f}  GPTQ={gptq:.4f}  ({(rtn - gptq) / rtn * 100:.1f}% lower)")
        assert gptq <= rtn + 1e-6

    # reader 精修：在 GPTQ 最终码上比较"初始行网格尺度"与"H-度量精修尺度"
    _, codes, _ = quantize_gptq(W_in, H_in, "row")
    init_scales = exact_ternary_fit(W_in)[1]
    refined = refit_row_scales(W_in, codes, H_in)
    W_init = codes * init_scales.unsqueeze(1)
    W_re = codes * refined.unsqueeze(1)
    print(
        f"reader 精修：H-误差 {relative_weighted_error(W_in, W_init, H_in):.4f} -> "
        f"{relative_weighted_error(W_in, W_re, H_in):.4f}；"
        f"欧氏误差 {float((W_init - W_in).pow(2).sum()):.2f} -> "
        f"{float((W_re - W_in).pow(2).sum()):.2f}"
    )
    assert relative_weighted_error(W_in, W_re, H_in) <= relative_weighted_error(W_in, W_init, H_in) + 1e-9


# ---------------------------------------------------------------------------
# E. bit accounting
# ---------------------------------------------------------------------------
def part_e() -> None:
    sep("E. resident bpw 复算")
    d = 4096
    for bits in (1, 3, 4, 5, 6, 8):
        bpw = resident_bpw(d * 12288, 12288, bits=bits)
        levels = levels_per_code(bits)
        label = "ternary" if bits == 1 else f"{bits}-bit"
        print(f"{label:<15}: {bpw:.4f} bpw  (levels={levels}, code={math.log2(levels):.4f})")
    assert abs(resident_bpw(d * 12288, 12288, bits=1) - 1.590) < 1e-3
    assert abs(resident_bpw(d * 12288, 12288, bits=3) - 2.811) < 1e-3


def part_f() -> None:
    sep("F. bit sweep：各 bit 下 RoleQ 均取两侧最优（合成重尾权重）")
    W_in, W_out, X = make_block()
    d = X.shape[1]

    configs = [
        ("per-OC", "row", "row"),
        ("per-IC", "col", "col"),
        ("RoleQ", "row", "col"),
    ]
    print(f"{'bits':>4}{'bpw@8B':>9}{'placement':>14}{'reader':>10}{'writer':>10}{'e2e':>10}")
    gains = []
    for bits in (1, 3, 4, 5, 6):
        # 统一按 Qwen3-8B 的 d=4096 记账，便于与论文 1.590/2.810/3.910 对照
        bpw = resident_bpw(4096 * 12288, 12288, bits=bits)
        results = {}
        for name, ar, aw in configs:
            err_in, err_out, e2e = run_block(
                W_in, W_out, X, ar, aw, "gptq", bits=bits, max_candidates=32
            )
            results[name] = e2e
            print(f"{bits:>4}{bpw:>9.3f}{name:>14}{err_in:>10.4f}{err_out:>10.4f}{e2e:>10.4f}")
        best_uniform = min(results["per-OC"], results["per-IC"])
        gain = best_uniform / results["RoleQ"]
        gains.append(gain)
        print(f"{'':>4}{'':>9}{'RoleQ gain':>14}{'':>10}{'':>10}{gain:>9.2f}x")

    print("\ngain by bits:", ", ".join(f"{b}b={g:.2f}x" for b, g in zip((1, 3, 4, 5, 6), gains)))
    print("注：合成权重的 unit-axis 重尾很强，RoleQ 收益不会随 bit 单调衰减；")
    print("    论文 Table 3 的真实模型上 INT3/INT4 的放置收益明显小于 ternary。")
    # 结构性断言：RoleQ 在所有已支持的 bit 下不劣于最佳 uniform
    assert all(g > 1.0 for g in gains)


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()
    print("\n全部验证通过。")


if __name__ == "__main__":
    main()
