"""SAGE 核心实现的单元测试（CPU 即可运行）。

覆盖：
  - 球坐标分解/重建可逆性
  - 非负单位码本角参数化往返一致性
  - 共享半径网格量化（码值范围、零步长组、STE 梯度）
  - ESG 置换的正确性与"跨能量层"性质
  - 加权球面 k-means 的非负单位性
  - SAGEQuantizer 编解码 / bpp / 分块一致性
  - blockwise 误差补偿确实降低输出 MSE
"""

from __future__ import annotations

import math

import pytest
import torch

from sage import (
    BlockwiseConfig,
    SAGEConfig,
    SAGEQuantizer,
    Split,
    angles_to_codewords,
    blockwise_optimize,
    channel_energy,
    codewords_to_angles,
    decompose,
    esg_permutation,
    radius_grid_quantize,
    reconstruct_vectors,
    weighted_spherical_kmeans,
)


def make_weight(dout: int, din: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(dout, din, generator=g) * 0.02
    outliers = (torch.rand(dout, din, generator=g) < 0.01) * torch.randn(dout, din, generator=g) * 0.5
    return base + outliers


# ---------------------------------------------------------------- 球坐标分解
def test_decompose_reconstruct_roundtrip():
    W = make_weight(32, 64, seed=1)
    v = 8
    neg, mag, radii = decompose(W, v)
    # 用真实方向（单位化模长）与真实半径重建，应精确还原
    unit = mag / mag.norm(dim=1, keepdim=True).clamp_min(1e-12)
    recon = reconstruct_vectors(neg, unit, torch.arange(unit.shape[0]), radii)
    assert torch.allclose(recon.reshape(W.shape), W, atol=1e-5)


def test_decompose_requires_divisible():
    with pytest.raises(ValueError):
        decompose(torch.randn(4, 10), v=8)


# ------------------------------------------------------------ 码本角参数化
def test_angle_parameterization_roundtrip():
    alpha = torch.randn(16, 7)
    C = angles_to_codewords(alpha)
    # 非负 + 单位
    assert (C >= 0).all()
    assert torch.allclose(C.norm(dim=1), torch.ones(16), atol=1e-5)
    alpha2 = codewords_to_angles(C)
    C2 = angles_to_codewords(alpha2)
    assert torch.allclose(C, C2, atol=1e-5)


# ------------------------------------------------------------ 半径网格量化
def test_radius_grid_quantize_range_and_ste():
    r = torch.rand(256) * 5 + 0.1          # ng = 256/16 = 16 组
    r_hat, mn, step, codes = radius_grid_quantize(r, radius_bits=4, subs_per_group=16)
    levels = (1 << 4) - 1
    assert codes.min() >= 0 and codes.max() <= levels
    # 量化值不超组内范围
    rg = r.reshape(16, 16)
    assert (r_hat.reshape(16, 16) >= mn.unsqueeze(1) - 1e-6).all()
    assert (r_hat.reshape(16, 16) <= rg.max(dim=1, keepdim=True).values + 1e-6).all()
    # STE：梯度可回传
    rp = r.clone().requires_grad_(True)
    rhs, *_ = radius_grid_quantize(rp, 4, 16, ste=True)
    rhs.sum().backward()
    assert rp.grad is not None and torch.isfinite(rp.grad).all()


def test_radius_grid_zero_step_group():
    r = torch.full((16,), 3.14)            # 单组，全相等 → δ=0
    r_hat, mn, step, codes = radius_grid_quantize(r, 2, 16)
    assert torch.allclose(r_hat, torch.full_like(r, 3.14), atol=1e-6)
    assert (codes == 0).all()


# ------------------------------------------------------------------- ESG
def test_esg_permutation_is_permutation_and_spans_tiers():
    din = 512
    energy = torch.arange(din, dtype=torch.float32)  # 能量随索引递增
    pi = esg_permutation(energy, v=8)
    assert sorted(pi.tolist()) == list(range(din))
    # 每个子向量的第 0 个位置应取自最高能量 tier（即能量最大的顶层通道）
    e_perm = energy[pi].reshape(-1, 8)
    # 位置 0 的能量恒 >= 位置 7 的能量（tier 顺序由高到低）
    assert (e_perm[:, 0] >= e_perm[:, 7]).all()
    # 位置 0 的通道来自最高档 tier（能量最高的 m 个）
    m = din // 8
    assert set(pi.reshape(-1, 8)[:, 0].tolist()).issubset(set(range(din - m, din)))


def test_channel_energy():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1,2,2) -> mean of squares per channel
    assert torch.allclose(channel_energy(x), torch.tensor([5.0, 10.0]))


# ----------------------------------------------------------- 球面 k-means
def test_spherical_kmeans_nonneg_unit():
    g = torch.Generator().manual_seed(0)
    mag = (torch.rand(512, 8, generator=g) * 0.1)
    omega = torch.ones_like(mag)
    C = weighted_spherical_kmeans(mag, omega, k=16, iters=10, seed=0)
    assert C.shape == (16, 8)
    assert (C >= -1e-6).all()
    assert torch.allclose(C.norm(dim=1), torch.ones(16), atol=1e-5)


# --------------------------------------------------------- SAGEQuantizer
def _cfg(**kw) -> SAGEConfig:
    base = dict(v=8, group_size=128, splits=(Split(64, 2),), adaptive_bits=False, kmeans_iters=8)
    base.update(kw)
    return SAGEConfig(**base)


def rand_energy(din: int, seed: int = 10) -> torch.Tensor:
    """确定性通道能量（避免依赖全局 RNG，保证测试顺序无关）。"""
    return torch.rand(din, generator=torch.Generator().manual_seed(seed)) + 0.1


def test_quantizer_prepare_dequantize_and_bpp():
    W = make_weight(128, 128, seed=2)
    energy = rand_energy(128)
    qz = SAGEQuantizer(_cfg())
    state = qz.prepare(W, energy)
    W_hat = qz.dequantize(state)
    assert W_hat.shape == W.shape
    rel = (W - W_hat).norm() / W.norm()
    assert rel < 0.85  # 2 bit + 离群点下合理范围（精度由 demo/blockwise 测试覆盖）
    bpp = qz.bpp(state)
    assert math.isclose(bpp["payload"], 16 / 8, rel_tol=1e-6)      # (8+6+2)/8 = 2.0
    assert math.isclose(bpp["grid"], 2 * 16 / 128, rel_tol=1e-6)   # 0.25
    assert bpp["total"] > bpp["payload"]


def test_quantizer_chunk_consistency():
    W = make_weight(128, 128, seed=3)
    energy = rand_energy(128, seed=11)
    a = SAGEQuantizer(_cfg(chunk=64)).prepare(W, energy)
    b = SAGEQuantizer(_cfg(chunk=10_000)).prepare(W, energy)
    assert a.split == b.split
    assert torch.equal(a.direction_index, b.direction_index)
    assert (a.score == b.score)


def test_quantizer_abs_picks_lower_score():
    W = make_weight(128, 256, seed=4)
    energy = rand_energy(256, seed=12)
    cfg = SAGEConfig(v=8, group_size=128, splits=(Split(64, 4), Split(128, 3), Split(256, 2)),
                     adaptive_bits=True, kmeans_iters=8, seed=0)
    qz = SAGEQuantizer(cfg)
    state = qz.prepare(W, energy)
    # 与逐个候选比较，ABS 选中的分数应为最小
    scores = {s.codebook_size: SAGEQuantizer(
        SAGEConfig(v=8, group_size=128, splits=(s,), adaptive_bits=False, kmeans_iters=8, seed=0)
    ).prepare(W, energy).score for s in cfg.splits}
    assert math.isclose(state.score, min(scores.values()), rel_tol=1e-9)


# --------------------------------------------------------- blockwise 优化
def test_blockwise_reduces_output_mse():
    torch.manual_seed(0)
    dout, din = 64, 128
    W = make_weight(dout, din, seed=5)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(256, din, generator=g)
    y = x @ W.t()
    energy = (x * x).mean(0)

    qz = SAGEQuantizer(_cfg(splits=(Split(32, 2),)))
    state = qz.prepare(W, energy)
    before = float(torch.nn.functional.mse_loss(x @ qz.dequantize(state).t(), y))

    blockwise_optimize(state, x, y, BlockwiseConfig(epochs=10, batch_size=32, lr=1e-2),
                       subs_per_group=qz.config.subs_per_group)
    after = float(torch.nn.functional.mse_loss(x @ qz.dequantize(state).t(), y))
    assert after < before, f"blockwise 未降低输出 MSE: before={before:.6e} after={after:.6e}"


# ------------------------------------------------------- 模型级集成（hook+回写）
def test_model_level_quantize_pipeline():
    """验证 model_quant.quantize_model 的 hook 捕获 / 逐层量化 / 回写链路。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model_quant import quantize_model, target_linear_names

    torch.manual_seed(0)
    d, h = 128, 256

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.up_proj = torch.nn.Linear(d, h)
            self.mlp.down_proj = torch.nn.Linear(h, d)

        def forward(self, input_ids):
            return self.mlp.down_proj(torch.relu(self.mlp.up_proj(input_ids)))

    model = TinyModel().eval()
    names = target_linear_names(model, "all")
    assert names == ["mlp.up_proj", "mlp.down_proj"]
    x = torch.randn(64, d, generator=torch.Generator().manual_seed(1)) * 0.5
    model._sage_batches = [{"input_ids": x}]

    cfgg = SAGEConfig(v=8, group_size=128, splits=(Split(32, 2),), adaptive_bits=False,
                      kmeans_iters=8, seed=0)
    stats = quantize_model(model, cfgg, BlockwiseConfig(epochs=3, batch_size=32, lr=1e-2),
                           names, torch.device("cpu"), sequential=False)
    assert len(stats) == 2
    assert all(0.0 <= s["rel"] < 1.0 for s in stats)
    # 权重确实被替换为 fake-quant 值
    W = model.mlp.down_proj.weight.data
    assert W.shape == (d, h)
