"""PCDVQ 最小实现单元测试。运行: pytest -q"""

import math

import torch

from pcdvq import (
    PCDVQ,
    RHTRegularizer,
    e8_unit_directions,
    from_polar,
    greedy_direction_codebook,
    hadamard_matrix,
    lloyd_max_magnitude_codebook,
    to_polar,
)


def test_hadamard_orthogonal():
    H = hadamard_matrix(16).double()
    assert torch.allclose(H @ H.t(), 16 * torch.eye(16, dtype=torch.float64), atol=1e-8)


def test_rht_roundtrip_and_gaussian():
    torch.manual_seed(0)
    x = torch.randn(512, 128) * 0.05
    reg = RHTRegularizer(128, seed=1)
    y, s = reg.forward(x)
    x_back = reg.inverse(y, s)
    assert (x - x_back).abs().max().item() < 1e-4
    # 变换后逐元素近似标准正态
    assert abs(y.mean().item()) < 0.05
    assert abs(y.std().item() - 1.0) < 0.05


def test_rht_non_pow2_dim():
    """非 2 的幂维度：分块 Hadamard（q=384=128*3）。"""
    torch.manual_seed(0)
    x = torch.randn(64, 384)
    reg = RHTRegularizer(384, seed=0)
    assert reg.block == 128
    y, s = reg.forward(x)
    assert (x - reg.inverse(y, s)).abs().max().item() < 1e-4
    assert abs(y.std().item() - 1.0) < 0.05


def test_pcd_roundtrip():
    torch.manual_seed(0)
    v = torch.randn(2048, 8) * torch.rand(2048, 1)
    phis, r = to_polar(v)
    assert (v - from_polar(phis, r)).abs().max().item() < 1e-5


def test_pcd_recovers_last_sign():
    """最后一维为负也必须可重建（论文/参考实现字面公式会丢失符号）。"""
    v = torch.randn(64, 8)
    v[:, -1] = -v[:, -1].abs() - 0.5
    phis, r = to_polar(v)
    assert (v - from_polar(phis, r)).abs().max().item() < 1e-5


def test_e8_lattice_counts():
    assert e8_unit_directions(2).shape[0] == 240
    assert e8_unit_directions(4).shape[0] == 2400
    # 全部单位化
    assert torch.allclose(e8_unit_directions(4).norm(dim=1), torch.ones(2400), atol=1e-5)


def test_direction_codebook_is_uniform():
    pool = e8_unit_directions(4)
    cb = greedy_direction_codebook(pool, bits=6, seed=0)
    assert cb.shape == (64, 8)
    sim = cb @ cb.t()
    mask = ~torch.eye(64, dtype=torch.bool)
    assert sim[mask].max().item() < 0.75  # 贪心采样后彼此远离


def test_magnitude_codebook_golden():
    """复现参考 notebook 的数值：用其超参 (tol=1e-3, 100 迭代) 得 [1.4241, ...]。

    注：notebook 输出实为未完全收敛的中间解；完全收敛 (tol=1e-8) 的固定点约为
    [1.4396, 1.8956, 2.2510, ...]。两者都验证了实现与参考一致。
    """
    cb_ref = lloyd_max_magnitude_codebook(bits=3, k=8, tau=0.99, tol=1e-3, max_iters=100)
    golden = torch.tensor([1.4241, 1.8740, 2.2255, 2.5455, 2.8623, 3.1978, 3.5805, 4.0698])
    assert torch.allclose(cb_ref, golden, atol=2e-3)

    cb_conv = lloyd_max_magnitude_codebook(bits=3, k=8, tau=0.99, tol=1e-8, max_iters=500)
    golden_conv = torch.tensor([1.4396, 1.8956, 2.2510, 2.5730, 2.8899, 3.2234, 3.6016, 4.0822])
    assert torch.allclose(cb_conv, golden_conv, atol=2e-3)
    assert torch.all(cb_conv[1:] > cb_conv[:-1])  # 单调递增


def test_pcdvq_end_to_end_runs():
    """端到端编解码可运行、位宽正确、误差有限。"""
    torch.manual_seed(0)
    W = torch.randn(128, 128) * 0.02
    qtz = PCDVQ(k=8, a=8, b=2, max_norm_sq=4, seed=0)
    qtz.build_codebooks()
    assert math.isclose(qtz.bpw, 1.25)
    W_hat = qtz.dequantize(qtz.quantize(W))
    rel = float((W - W_hat).norm() / W.norm())
    assert torch.isfinite(W_hat).all()
    assert 0.0 < rel < 1.0  # 低 bit 下误差有限（绝对量级见 demo 报告）
