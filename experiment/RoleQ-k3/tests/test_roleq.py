"""RoleQ-T 核心单测（CPU）。运行：pytest -q tests"""

from __future__ import annotations

import itertools

import pytest
import torch

from roleq import (
    exact_grid_fit,
    exact_ternary_fit,
    infer_role,
    levels_per_code,
    quantize_gptq,
    quantize_rtn,
    random_hadamard,
    refit_row_scales,
    relative_weighted_error,
    resident_bpw,
    role_axis,
    rotate_reader,
    rotate_writer,
)


def brute_force(v: torch.Tensor) -> float:
    n = v.shape[0]
    best = float("inf")
    for mask in itertools.product([0, 1], repeat=n):
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            continue
        a = v[idx].abs().mean()
        approx = torch.zeros_like(v)
        approx[idx] = v[idx].sign() * a
        best = min(best, float((v - approx).pow(2).sum()))
    return best


def test_exact_fit_matches_bruteforce():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        v = torch.randn(7, generator=g) * torch.exp(0.7 * torch.randn(7, generator=g))
        codes, scale = exact_ternary_fit(v)
        err = float((v - scale * codes).pow(2).sum())
        assert err <= brute_force(v) + 1e-5


def test_exact_fit_ternary_and_zero():
    v = torch.tensor([0.0, 0.0, 0.0])
    codes, scale = exact_ternary_fit(v)
    assert torch.all(codes == 0) and float(scale) == 0.0
    v = torch.tensor([1.0, -2.0, 0.3, 0.0])
    codes, scale = exact_ternary_fit(v)
    assert torch.all((codes == -1) | (codes == 0) | (codes == 1))
    assert float(scale) > 0


def test_role_axis_mapping():
    assert role_axis("reader") == "row"
    assert role_axis("writer") == "col"
    assert infer_role("model.layers.0.self_attn.q_proj") == "reader"
    assert infer_role("model.layers.0.self_attn.o_proj") == "writer"
    assert infer_role("model.layers.0.mlp.down_proj") == "writer"
    assert infer_role("model.layers.0.mlp.up_proj") == "reader"
    assert infer_role("model.embed_tokens") is None


def test_scale_counts_match_unit_axis():
    W = torch.randn(32, 16)
    _, _, row_scales = quantize_rtn(W, "row")
    _, _, col_scales = quantize_rtn(W, "col")
    assert row_scales.shape == (32,)
    assert col_scales.shape == (16,)


def test_writer_first_step_is_exact_vector_fit():
    """H=I 时第一列无补偿，writer 步骤应等于精确向量拟合。"""
    g = torch.Generator().manual_seed(1)
    W = torch.randn(16, 8, generator=g)
    Wq, codes, _ = quantize_gptq(W, torch.eye(8), "col")
    c, a = exact_ternary_fit(W[:, 0])
    assert torch.allclose(Wq[:, 0], a * c, atol=1e-6)
    assert torch.allclose(codes[:, 0], c, atol=1e-6)


def test_reader_identity_metric_matches_exact_fit():
    """H=I 时 reader 的行网格与精修都退化为逐行精确拟合。"""
    g = torch.Generator().manual_seed(2)
    W = torch.randn(8, 16, generator=g)
    Wq, _, _ = quantize_gptq(W, torch.eye(16), "row")
    c, a = exact_ternary_fit(W)
    assert torch.allclose(Wq, c * a.unsqueeze(1), atol=1e-5)


@pytest.mark.parametrize("axis", ["row", "col"])
def test_gptq_not_worse_than_rtn(axis):
    g = torch.Generator().manual_seed(3)
    W = torch.randn(48, 32, generator=g)
    X = torch.randn(256, 32, generator=g)
    H = X.T @ X / X.shape[0]
    rtn = relative_weighted_error(W, quantize_rtn(W, axis)[0], H)
    gptq = relative_weighted_error(W, quantize_gptq(W, H, axis)[0], H)
    assert gptq <= rtn + 1e-6


def test_reader_refit_never_increases_h_error():
    g = torch.Generator().manual_seed(4)
    W = torch.randn(32, 64, generator=g)
    X = torch.randn(512, 64, generator=g)
    H = X.T @ X / X.shape[0]
    _, codes, _ = quantize_gptq(W, H, "row")
    init_scales = exact_ternary_fit(W)[1]
    refined = refit_row_scales(W, codes, H)
    e_init = relative_weighted_error(W, codes * init_scales.unsqueeze(1), H)
    e_ref = relative_weighted_error(W, codes * refined.unsqueeze(1), H)
    assert e_ref <= e_init + 1e-9


def test_hadamard_orthogonal_and_unit_axis_invariant():
    R = random_hadamard(64, seed=7)
    assert torch.allclose(R @ R.T, torch.eye(64), atol=1e-5)

    g = torch.Generator().manual_seed(8)
    W_reader = torch.randn(32, 64, generator=g)
    W_writer = torch.randn(64, 32, generator=g)
    assert torch.allclose(
        rotate_reader(W_reader, R).norm(dim=1), W_reader.norm(dim=1), atol=1e-5
    )
    assert torch.allclose(
        rotate_writer(W_writer, R).norm(dim=0), W_writer.norm(dim=0), atol=1e-5
    )


def test_resident_bpw_matches_paper():
    d = 4096
    bpw = resident_bpw(4096 * 12288, 12288)
    assert abs(bpw - 1.590) < 1e-3


def test_act_order_path_runs():
    g = torch.Generator().manual_seed(9)
    W = torch.randn(24, 32, generator=g)
    X = torch.randn(128, 32, generator=g)
    H = X.T @ X / X.shape[0]
    for axis in ("row", "col"):
        Wq, codes, scales = quantize_gptq(W, H, axis, act_order=True)
        assert Wq.shape == W.shape and codes.shape == W.shape
        assert scales.numel() == (W.shape[0] if axis == "row" else W.shape[1])


# ---------------------------------------------------------------------------
# 任意 bit（对称整数网格）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [3, 4, 5])
def test_grid_fit_close_to_dense_scale_search(bits):
    g = torch.Generator().manual_seed(10 + bits)
    v = torch.randn(48, generator=g) * torch.exp(0.8 * torch.randn(48, generator=g))
    codes, scale = exact_grid_fit(v, bits=bits)
    err = float((v - scale * codes).pow(2).sum())

    qmax = 2 ** (bits - 1) - 1
    max_abs = float(v.abs().max())
    best = float("inf")
    for s in torch.linspace(max_abs / qmax * 0.05, max_abs / qmax, 4000):
        q = torch.clamp(torch.round(v / s), -qmax, qmax)
        best = min(best, float((v - s * q).pow(2).sum()))
    assert err <= best * 1.02 + 1e-6


@pytest.mark.parametrize("bits", [3, 4, 6])
def test_grid_codes_in_range(bits):
    g = torch.Generator().manual_seed(20 + bits)
    W = torch.randn(16, 24, generator=g)
    codes, scales = exact_grid_fit(W, bits=bits)
    qmax = 2 ** (bits - 1) - 1
    assert float(codes.abs().max()) <= qmax
    assert torch.all(scales >= 0)


def test_grid_fit_zero_vector():
    codes, scales = exact_grid_fit(torch.zeros(4, 8), bits=3)
    assert torch.all(codes == 0) and torch.all(scales == 0)


def test_reader_identity_h_matches_grid_fit_bits3():
    g = torch.Generator().manual_seed(30)
    W = torch.randn(8, 16, generator=g)
    Wq, codes, _ = quantize_gptq(W, torch.eye(16), "row", bits=3)
    ref_codes, ref_scales = exact_grid_fit(W, bits=3)
    assert torch.allclose(codes, ref_codes, atol=1e-6)
    # H=I 时精修不高于初始网格的逐行误差
    ref_err = relative_weighted_error(W, ref_scales.unsqueeze(1) * ref_codes, torch.eye(16))
    assert relative_weighted_error(W, Wq, torch.eye(16)) <= ref_err + 1e-6


def test_writer_first_step_is_grid_fit_bits3():
    g = torch.Generator().manual_seed(31)
    W = torch.randn(16, 8, generator=g)
    Wq, codes, _ = quantize_gptq(W, torch.eye(8), "col", bits=3)
    c, a = exact_grid_fit(W[:, 0], bits=3)
    assert torch.allclose(Wq[:, 0], a * c, atol=1e-5)
    assert torch.allclose(codes[:, 0], c, atol=1e-6)


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("axis", ["row", "col"])
def test_gptq_not_worse_than_rtn_bits(axis, bits):
    g = torch.Generator().manual_seed(40 + bits)
    W = torch.randn(48, 32, generator=g)
    X = torch.randn(256, 32, generator=g)
    H = X.T @ X / X.shape[0]
    rtn = relative_weighted_error(W, quantize_rtn(W, axis, bits=bits)[0], H)
    gptq = relative_weighted_error(W, quantize_gptq(W, H, axis, bits=bits)[0], H)
    assert gptq <= rtn + 1e-6


def test_resident_bpw_bits():
    d = 4096
    assert abs(resident_bpw(d * 12288, 12288, bits=1) - 1.590) < 1e-3
    assert abs(resident_bpw(d * 12288, 12288, bits=3) - 2.811) < 1e-3
    assert abs(resident_bpw(d * 12288, 12288, bits=4) - 3.911) < 1e-3
    assert abs(resident_bpw(d * 12288, 12288, bits=8) - 7.999) < 1e-3
    assert levels_per_code(1) == 3 and levels_per_code(3) == 7 and levels_per_code(4) == 15


def test_two_bit_asymmetric_not_supported():
    with pytest.raises(NotImplementedError):
        levels_per_code(2)
