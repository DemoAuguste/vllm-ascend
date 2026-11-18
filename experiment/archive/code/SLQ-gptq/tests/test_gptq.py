# -*- coding: UTF-8 -*-
"""GPTQ 算子单测（CPU）：注册语义、位宽泛化、误差补偿有效性。"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptq import GPTQ, int_qparams, quant_dequant_int  # noqa: E402


def _rtn(W, bits):
    return quant_dequant_int(W, bits)


def test_bits_validation():
    with pytest.raises(ValueError):
        GPTQ((4, 8), bits=1)
    with pytest.raises(ValueError):
        GPTQ((4, 8), bits=9)
    with pytest.raises(ValueError):
        GPTQ((4,), bits=4)


def test_requires_observe():
    g = GPTQ((4, 8), bits=4)
    with pytest.raises(RuntimeError):
        g.quantize(torch.randn(4, 8))


def test_all_bits_supported_and_quantize_to_grid():
    torch.manual_seed(0)
    W = torch.randn(8, 32) * 0.02
    X = torch.randn(64, 32) * 0.5
    for bits in (2, 3, 4, 6, 8):
        Wq = GPTQ.run(W, X, bits=bits)
        assert Wq.shape == W.shape and torch.isfinite(Wq).all()
        # 每个输出行的量化值应落在 scale 的整数倍网格上
        scale = W.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)
        codes = (Wq / scale)
        assert torch.allclose(codes, codes.round(), atol=1e-4)


def test_gptq_reduces_output_error_vs_rtn():
    torch.manual_seed(0)
    W = torch.randn(256, 512) * 0.02
    X = torch.randn(1024, 512) * 0.5

    def out_mse(Wq):
        return float(torch.nn.functional.mse_loss(X @ Wq.t(), X @ W.t()))

    for bits in (3, 4):
        rtn = _rtn(W, bits)
        gptq = GPTQ.run(W, X, bits=bits)
        assert out_mse(gptq) < out_mse(rtn)


def test_dead_hessian_column_is_zeroed():
    torch.manual_seed(0)
    W = torch.randn(6, 16)
    X = torch.randn(200, 16)
    X[:, 5] = 0  # 零方差列 → Hessian 对角为 0
    Wq = GPTQ.run(W, X, bits=4)
    assert torch.equal(Wq[:, 5], torch.zeros(6))


def test_group_size_shapes():
    torch.manual_seed(0)
    W = torch.randn(8, 96) * 0.02
    X = torch.randn(128, 96) * 0.5
    Wq = GPTQ.run(W, X, bits=4, group_size=32)
    assert Wq.shape == W.shape and torch.isfinite(Wq).all()


def test_asym_grid_and_zero_point():
    """非对称：量化值落在 (min, min+step, ...) 网格上，且最小值对齐 min。"""
    torch.manual_seed(0)
    W = torch.randn(8, 64) * 0.02 + 0.05           # 非零均值 → 适合非对称
    X = torch.randn(128, 64) * 0.5
    bits = 4
    Wq = GPTQ.run(W, X, bits=bits, group_size=32, asym=True)
    scale, zero = int_qparams(W, bits, 32, sym=False)
    import torch as _t
    for r in range(W.shape[0]):
        gq = Wq[r].reshape(-1, 32)
        for gi in range(gq.shape[0]):
            codes = (gq[gi] - zero[r, gi]) / scale[r, gi]
            assert _t.allclose(codes, codes.round(), atol=1e-3)


def test_asym_beats_sym_on_skewed_weights():
    """偏斜分布上非对称输出误差应低于对称（验证方差律方向）。"""
    torch.manual_seed(0)
    W = torch.randn(256, 512) * 0.02 + 0.06
    X = torch.randn(1024, 512) * 0.5

    def out_mse(Wq):
        return float(torch.nn.functional.mse_loss(X @ Wq.t(), X @ W.t()))

    for bits in (3, 4):
        ws = GPTQ.run(W, X, bits=bits, asym=False)
        wa = GPTQ.run(W, X, bits=bits, asym=True)
        assert out_mse(wa) < out_mse(ws)
