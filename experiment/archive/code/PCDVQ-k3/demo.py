"""PCDVQ 最小可验证 demo。

运行：python demo.py

验证内容：
  A. RHT 标准高斯正则化：变换前后分布统计 + 可逆性
  B. PCD 极坐标解耦：可逆性（含最后一维符号不丢失的回归检查）
  C. DACC 码本：E8 方向码本均匀性、chi 幅值码本（与参考 notebook 数值比对）
  D. 方向/幅值敏感度：只量化方向 vs 只量化幅值（复现论文动机 Fig.1）
  E. 端到端：PCDVQ vs 等 bit 耦合 k-means（多档位），报告相对误差与误差分解
  F. DACC 消融：E8 贪心 vs 随机高斯 vs 球面 k-means 方向码本（复现论文 Table 4）
"""

from __future__ import annotations

import math

import torch

from pcdvq import (
    PCDVQ,
    RHTRegularizer,
    e8_unit_directions,
    from_polar,
    greedy_direction_codebook,
    lloyd_max_magnitude_codebook,
    to_polar,
)


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def make_weight(p: int, q: int, seed: int = 42) -> torch.Tensor:
    """带离群点的合成权重（模仿真实 LLM 权重 + outliers）。"""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(p, q, generator=g) * 0.02
    outliers = (torch.rand(p, q, generator=g) < 0.01) * torch.randn(p, q, generator=g) * 0.5
    return base + outliers


def kurtosis(x: torch.Tensor) -> float:
    x = x.flatten()
    m, s = x.mean(), x.std()
    return float(((x - m) ** 4).mean() / (s ** 4 + 1e-12))


def kmeans(x: torch.Tensor, n_clusters: int, iters: int = 20, seed: int = 0, mem_budget: int = 4_000_000):
    """内存安全 k-means（分块计算 cdist）。"""
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    C = x[torch.randperm(n, generator=g)[:n_clusters]].clone()
    chunk = max(128, mem_budget // n_clusters)
    lab = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        for s in range(0, n, chunk):
            lab[s : s + chunk] = torch.cdist(x[s : s + chunk], C).argmin(dim=1)
        newC = torch.zeros_like(C)
        counts = torch.zeros(n_clusters)
        newC.index_add_(0, lab, x)
        counts.index_add_(0, lab, torch.ones(n))
        empty = counts == 0
        newC[~empty] /= counts[~empty].unsqueeze(1)
        newC[empty] = C[empty]
        shift = float((newC - C).norm() / (C.norm() + 1e-12))
        C = newC
        if shift < 1e-4:
            break
    for s in range(0, n, chunk):
        lab[s : s + chunk] = torch.cdist(x[s : s + chunk], C).argmin(dim=1)
    return C, lab


def vq_errors(v: torch.Tensor, v_hat: torch.Tensor):
    """VQ 阶段误差分解（论文 Fig.1(b)）：方向 MSE 与幅值 MSE。"""
    mag = v.norm(dim=1)
    mag_hat = v_hat.norm(dim=1)
    cos = ((v * v_hat).sum(1) / (mag * mag_hat).clamp_min(1e-12)).clamp(-1.0, 1.0)
    dir_mse = float((2 * mag ** 2 * (1 - cos)).mean())
    mag_mse = float(((mag - mag_hat) ** 2).mean())
    return dir_mse, mag_mse, float(((v - v_hat) ** 2).mean())


def dir_mse(v: torch.Tensor, v_hat: torch.Tensor) -> float:
    return vq_errors(v, v_hat)[0]


def main() -> None:
    torch.manual_seed(0)
    k = 8

    # ---------- A. RHT 高斯化 ----------
    sep("A. RHT 标准高斯正则化")
    W0 = make_weight(256, 256)
    reg = RHTRegularizer(256, seed=0)
    Y, scale = reg.forward(W0)
    W_back = reg.inverse(Y, scale)
    print(f"权重形状: {tuple(W0.shape)}, 输入维 q=256, Hadamard 块大小 b={reg.block}")
    print(f"原始权重   : mean={W0.mean():+.4f}  std={W0.std():.4f}  kurtosis={kurtosis(W0):.3f}")
    print(f"RHT 变换后 : mean={Y.mean():+.4f}  std={Y.std():.4f}  kurtosis={kurtosis(Y):.3f}  (目标 0,1,3)")
    print(f"RHT 往返误差: {(W0 - W_back).abs().max().item():.3e}")

    # ---------- B. PCD 可逆性 ----------
    sep("B. PCD 极坐标解耦可逆性")
    V = torch.randn(4096, k) * torch.rand(4096, 1)
    phis, r = to_polar(V)
    print(f"PCD 往返最大误差: {(V - from_polar(phis, r)).abs().max().item():.3e}")
    Vneg = torch.randn(8, k)
    Vneg[:, -1] = -Vneg[:, -1].abs() - 0.1
    pn, rn = to_polar(Vneg)
    print(f"含负末维向量重建误差: {(Vneg - from_polar(pn, rn)).abs().max().item():.3e}"
          f"  (字面公式会丢失 v_k 符号，此实现修正)")
    print(f"最后一个角取值范围: [{phis[:, -1].min():+.3f}, {phis[:, -1].max():+.3f}] (-pi,pi]")

    # ---------- C. DACC 码本 ----------
    sep("C. DACC 码本构造")
    pool2 = e8_unit_directions(2)
    pool4 = e8_unit_directions(4)
    print(f"E8 单位方向数: norm^2<=2 -> {pool2.shape[0]} (期望 240); "
          f"norm^2<=4 -> {pool4.shape[0]} (期望 2400)")
    cb_ref = lloyd_max_magnitude_codebook(bits=3, k=8, tau=0.99, tol=1e-3, max_iters=100)
    cb_conv = lloyd_max_magnitude_codebook(bits=3, k=8, tau=0.99, tol=1e-8, max_iters=500)
    print(f"幅值码本 k=8,b=3,tau=0.99 (参考 notebook 超参): {[round(float(x),4) for x in cb_ref]}")
    print(f"  参考 notebook 输出                        : [1.4241, 1.874, 2.2255, 2.5455, 2.8623, 3.1978, 3.5805, 4.0698]")
    print(f"  完全收敛固定点 (tol=1e-8)                 : {[round(float(x),4) for x in cb_conv]}")
    C_phi = greedy_direction_codebook(pool4, bits=8, seed=0)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(256, 8, generator=g)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    mask = ~torch.eye(256, dtype=torch.bool)
    print(f"方向码本(a=8)最小两两余弦: {(C_phi @ C_phi.t())[mask].max():.4f}  "
          f"(随机方向 {min(1.0, float((rnd @ rnd.t())[mask].max())):.4f})  -> E8 更均匀")

    # ---------- D. 方向/幅值敏感度 ----------
    sep("D. 方向 vs 幅值敏感度（复现论文动机 Fig.1）")
    Wq = make_weight(128, 1024)
    regq = RHTRegularizer(1024, seed=0)
    Yq, _ = regq.forward(Wq)
    G = Yq.reshape(-1, k)
    _, r_g = to_polar(G)
    unit_g = G / r_g.clamp_min(1e-12).unsqueeze(1)
    pool8 = e8_unit_directions(8)
    print("在相同码本规模下，单独量化方向 与 单独量化幅值 所引入的误差：")
    print(f"  {'码本中心数':<12}{'只量化方向→方向MSE':<24}{'只量化幅值→幅值MSE':<24}")
    for c in (2, 4, 6):
        cb_dir = greedy_direction_codebook(pool8, bits=c, seed=0)
        cb_mag = lloyd_max_magnitude_codebook(c, k, tau=0.999)
        v_dir = r_g.unsqueeze(1) * cb_dir[(unit_g @ cb_dir.t()).argmax(1)]
        v_mag = cb_mag[(r_g.unsqueeze(1) - cb_mag.unsqueeze(0)).abs().argmin(1)].unsqueeze(1) * unit_g
        print(f"  {2**c:<12}{dir_mse(G, v_dir):<24.5f}{vq_errors(G, v_mag)[1]:<24.5f}")
    print("结论: 相同码本规模下，量化方向引入的误差远大于量化幅值 -> 方向更敏感，")
    print("      应把更多 bit 分配给方向（论文 Fig.1(a)(b) 的核心发现）。")

    # ---------- E. 端到端 vs 等 bit 耦合 VQ ----------
    sep("E. 端到端：PCDVQ vs 等 bit 耦合 k-means")
    print(f"{'配置':<26}{'方法':<10}{'相对Frobenius':<15}{'方向MSE':<12}{'幅值MSE':<12}")
    for (a, n_groups) in [(8, 16384), (10, 32768)]:
        p, q = (128, 1024) if n_groups == 16384 else (256, 1024)
        W = make_weight(p, q)
        qz = PCDVQ(k=k, a=a, b=2, tau=0.999, max_norm_sq=8, seed=0)
        qz.build_codebooks()
        Wp = qz.dequantize(qz.quantize(W))
        rel_p = float((W - Wp).norm() / W.norm())
        Gg, _ = qz.group_vectors(W)
        Yp, _ = qz.group_vectors(Wp)
        d_p, m_p, _ = vq_errors(Gg, Yp)

        nbits = a + 2
        C, lab = kmeans(Gg, n_clusters=2 ** nbits, iters=20, seed=0)
        d_c, m_c, _ = vq_errors(Gg, C[lab])
        reg2 = RHTRegularizer(q, seed=0)
        Yc, sc = reg2.forward(W)
        Wc = reg2.inverse(C[lab].reshape(p, q), sc)
        rel_c = float((W - Wc).norm() / W.norm())

        tag = f"a={a},b=2 BPW={qz.bpw:.3f}"
        print(f"{tag:<26}{'耦合VQ':<10}{rel_c:<15.6f}{d_c:<12.6f}{m_c:<12.6f}")
        print(f"{'':<26}{'PCDVQ':<10}{rel_p:<15.6f}{d_p:<12.6f}{m_p:<12.6f}")
    print("说明: (1) PCDVQ 的幅值 MSE 始终更低——chi-Lloyd-Max 是幅值分布的最优标量量化；")
    print("      (2) 本 demo 的方向码本分辨率(2^8~2^10)远低于论文(2^14~2^16)，且 E8 贪心 FPS")
    print("          优化的是最坏情况覆盖而非 MSE，故该低 bit 档位下自由训练的 k-means 更优；")
    print("      (3) 论文在 a=14~16、BPW~2.0 的档位才取得优势，需真实 LLM+微调流水线复现。")

    # ---------- F. DACC 消融 ----------
    sep("F. DACC 消融：方向码本构造方法对比（复现论文 Table 4）")
    a_abl = 9
    Wf = make_weight(128, 1024)
    regf = RHTRegularizer(1024, seed=0)
    Yf, _ = regf.forward(Wf)
    Gf = Yf.reshape(-1, k)
    _, rf = to_polar(Gf)
    uf = Gf / rf.clamp_min(1e-12).unsqueeze(1)
    pool_abl = e8_unit_directions(8)
    cb_e8 = greedy_direction_codebook(pool_abl, bits=a_abl, seed=0)
    gen = torch.Generator().manual_seed(0)
    cb_rand = torch.randn(2 ** a_abl, k, generator=gen)
    cb_rand = cb_rand / cb_rand.norm(dim=1, keepdim=True)
    cb_km, _ = kmeans(uf, n_clusters=2 ** a_abl, iters=20, seed=0)
    cb_km = cb_km / cb_km.norm(dim=1, keepdim=True)

    def max_pair_cos(cb):
        m = ~torch.eye(cb.shape[0], dtype=torch.bool)
        return float((cb @ cb.t())[m].max())

    def quant_dir_mse(cb):
        idx = (uf @ cb.t()).argmax(dim=1)
        return dir_mse(Gf, rf.unsqueeze(1) * cb[idx])

    print(f"方向码本大小 2^{a_abl}={2**a_abl}")
    print(f"  {'方法':<14}{'最大两两余弦↓(均匀性)':<24}{'方向MSE↓(失真)':<16}")
    for name, cb in [("随机高斯", cb_rand), ("球面k-means", cb_km), ("E8贪心(本文)", cb_e8)]:
        print(f"  {name:<14}{max_pair_cos(cb):<24.4f}{quant_dir_mse(cb):<16.5f}")
    print("结论: E8 贪心码本的方向最均匀（最大两两余弦最低，符合论文设计目标——覆盖最坏情况）；")
    print("      球面 k-means 直接在球面上最小化 MSE 目标，故方向 MSE 更低；随机方向最差。")
    print("      论文 Table4 中 Greedy E8 端到端 PPL 最优，需完整 LLM+微调流水线方能复现该排序。")


if __name__ == "__main__":
    main()
