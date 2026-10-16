"""
PCDVQ 最小可验证 demo
复现论文: "PCDVQ: Enhancing Vector Quantization for Large Language Models
             via Polar Coordinate Decoupling"  (arXiv:2506.05432)

本脚本实现并验证论文的核心方法，并修正了非官方参考实现
(https://github.com/danlee65071/pcdvq) 中的问题：

  1. 标准高斯正则化 (RHM)：原 notebook 构造了 Hadamard 矩阵却未使用，
     实际只做了随机符号翻转（无法高斯化）。本脚本使用正确的随机 Hadamard 变换 S = H_norm @ D。
  2. 极坐标解耦 (PCD)：把向量拆成方向 (单位向量) 与幅值 (范数)，分别量化。
  3. 分布对齐码本 (DACC)：
     - 方向码本: E8 格点最小向量方向 + 贪心最远点采样 (空间均匀)
     - 幅值码本: 根卡方分布 (chi 分布) 上的 Lloyd-Max 标量量化 (最优)
  4. 端到端量化/反量化 + 重建误差验证。

依赖: torch (cpu), numpy, scipy
运行: python pcdvq_demo.py
"""

import numpy as np
import torch
import scipy.special as sp
from scipy import stats
from itertools import combinations, product


# ---------------------------------------------------------------------------
# 1. 标准高斯正则化 (随机 Hadamard 变换, RHM)
# ---------------------------------------------------------------------------
def construct_hadamard(order: int) -> torch.Tensor:
    """Sylvester 构造 Hadamard 矩阵 (order 必须是 2 的幂), H H^T = order * I。"""
    h = torch.tensor([[1.0]])
    while h.shape[0] < order:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def gaussian_regularize(W: torch.Tensor):
    """
    对权重矩阵 W (p x q) 逐列做随机 Hadamard 变换并缩放为标准高斯 N(0,1)。

    返回 (W_reg, scales, signs, H_norm)，其中:
      x_reg = H_norm @ (signs * x) / scale,  scale = ||x|| / sqrt(p)
    """
    p, q = W.shape
    H_norm = construct_hadamard(p) / np.sqrt(p)          # 正交 Hadamard
    signs = torch.randint(0, 2, (p,)) * 2 - 1             # (p,) 随机对角符号
    signs = signs.float()
    W_reg = torch.empty_like(W)
    scales = torch.empty(q)
    for j in range(q):
        x = W[:, j]
        scales[j] = torch.norm(x) / np.sqrt(p)
        W_reg[:, j] = H_norm @ (signs * x) / scales[j]
    return W_reg, scales, signs, H_norm


def inverse_regularize(W_reg: torch.Tensor, scales, signs, H_norm) -> torch.Tensor:
    """gaussian_regularize 的逆变换: x = scale * signs * (H_norm @ x_reg)。"""
    p, q = W_reg.shape
    W = torch.empty_like(W_reg)
    for j in range(q):
        W[:, j] = scales[j] * (signs * (H_norm @ W_reg[:, j]))
    return W


# ---------------------------------------------------------------------------
# 2. 极坐标解耦 (PCD)
# ---------------------------------------------------------------------------
def pcd(V: torch.Tensor):
    """
    把 k 维向量解耦为方向与幅值。
      v = d * r,  r = ||v||,  d = v / r (单位方向)

    另按论文式(6)计算极坐标角 phi (k-1 维)，用于验证方向分量。
    """
    k = V.shape[1]
    r = V.norm(dim=1)                      # 幅值
    d = V / r.clamp(min=1e-12).unsqueeze(1)  # 单位方向
    phi = torch.zeros(V.shape[0], k - 1)
    for i in range(k - 1):
        denom = torch.sqrt((V[:, i + 1:] ** 2).sum(dim=1))
        phi[:, i] = torch.atan2(denom, V[:, i])
    return phi, r, d


# ---------------------------------------------------------------------------
# 3. 方向码本: E8 格点 + 贪心最远点采样
# ---------------------------------------------------------------------------
def e8_minimal_directions(dtype=torch.float32) -> torch.Tensor:
    """E8 格的 240 个最小向量 (norm^2=2) 的方向 (单位化)。

    E8 最小向量两类:
      A: 两个分量为 ±1, 其余为 0 (C(8,2) * 2^2 = 112)
      B: 8 个分量均为 ±1/2, 负号个数为偶数 (2^8 / 2 = 128)
    """
    vecs = []
    for i, j in combinations(range(8), 2):
        for s1, s2 in product([-1.0, 1.0], repeat=2):
            v = torch.zeros(8, dtype=dtype)
            v[i] = s1
            v[j] = s2
            vecs.append(v)
    for signs in product([-0.5, 0.5], repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            vecs.append(torch.tensor(signs, dtype=dtype))
    V = torch.stack(vecs)                       # (240, 8)
    return V / V.norm(dim=1, keepdim=True)      # 单位方向


def enumerate_e8_directions(max_sq_norm: int) -> torch.Tensor:
    """(扩展) 枚举 E8 格在半径边界内的所有点方向，支持更大的方向码本。

    E8 = { x in Z^8 : sum even } 并 { x in (Z+1/2)^8 : sum even }。
    论文使用 a=14/16 (方向码本 2^14/2^16 个)，而 240 个最小向量只能支持 a<=7。
    此函数用于复现论文的真实位宽配置。
    """
    m = int(np.floor(np.sqrt(max_sq_norm)))
    r = np.arange(-m, m + 1)
    g = np.meshgrid(*([r] * 8), indexing="ij")
    grid = np.stack(g, axis=-1).reshape(-1, 8)

    # 整数点, 坐标和为偶
    int_pts = grid[grid.sum(1) % 2 == 0]
    int_pts = int_pts[(int_pts ** 2).sum(1) <= max_sq_norm]
    # 半整数点 x = z + 0.5, z 坐标和为偶
    half = grid[grid.sum(1) % 2 == 0] + 0.5
    half = half[(half ** 2).sum(1) <= max_sq_norm]

    pts = np.concatenate([int_pts, half], axis=0)
    norms = np.linalg.norm(pts, axis=1)
    pts = pts[norms > 1e-8]
    norms = norms[norms > 1e-8]
    return torch.tensor(pts / norms[:, None], dtype=torch.float32)


def construct_direction_codebook(directions: torch.Tensor, a: int) -> torch.Tensor:
    """论文 Algorithm 1: 贪心最远点采样构造空间均匀方向码本 (2^a 个)。"""
    num_centers = 2 ** a
    M = directions.shape[0]
    assert num_centers <= M, f"2^{a} > 候选方向数 {M}"
    idx = [torch.randint(0, M, (1,)).item()]
    for _ in range(1, num_centers):
        sims = directions @ directions[idx].T          # 与已选方向的余弦相似度
        max_sim = sims.max(dim=1).values
        max_sim[idx] = 1e9                             # 排除已选
        idx.append(torch.argmin(max_sim).item())       # 选最不相似(最均匀)的
    return directions[idx]


# ---------------------------------------------------------------------------
# 4. 幅值码本: 根卡方分布上的 Lloyd-Max
# ---------------------------------------------------------------------------
def find_max_r(k: int, tau: float) -> float:
    """二分求 max_r 使 F(max_r)=tau, F 为 chi 分布 (根卡方, k 自由度) 的 CDF。"""
    lo, hi = 0.0, 2 * np.sqrt(k)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sp.gammainc(k / 2, mid * mid / 2) < tau:
            lo = mid
        else:
            hi = mid
    return hi


def construct_magnitude_codebook(k: int, b: int, tau=0.99, tol=1e-4, max_iters=200):
    """论文 Algorithm 2: 根卡方分布上的 Lloyd-Max 标量量化 (2^b 个中心)。

    区间 [a,c] 内条件均值 E[r|a<=r<=c] =
      sqrt(2) * Gamma((k+1)/2)/Gamma(k/2)
        * [gamma((k+1)/2, c^2/2) - gamma((k+1)/2, a^2/2)]
        / [gamma(k/2, c^2/2) - gamma(k/2, a^2/2)]
    """
    num = 2 ** b
    max_r = find_max_r(k, tau)
    edges = np.linspace(0, max_r, num + 1)
    codebook = 0.5 * (edges[:-1] + edges[1:])
    ratio = np.exp(sp.gammaln((k + 1) / 2) - sp.gammaln(k / 2))
    for _ in range(max_iters):
        u = np.empty(num + 1)
        u[0] = 0.0
        u[1:-1] = 0.5 * (codebook[:-1] + codebook[1:])
        u[-1] = max_r
        new = np.empty(num)
        for i in range(num):
            a_, c_ = u[i], u[i + 1]
            num_g = sp.gammainc((k + 1) / 2, c_ * c_ / 2) - sp.gammainc((k + 1) / 2, a_ * a_ / 2)
            den_g = sp.gammainc(k / 2, c_ * c_ / 2) - sp.gammainc(k / 2, a_ * a_ / 2)
            new[i] = np.sqrt(2) * ratio * num_g / (den_g + 1e-12)
        codebook = new
        if np.abs(new - codebook).max() < tol:
            break
    return torch.tensor(codebook, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 5. 量化 / 反量化
# ---------------------------------------------------------------------------
def quantize_vectors(V, dir_codebook, mag_codebook):
    """方向用余弦相似度 (argmax), 幅值用最近邻 (argmin |r-c|)。"""
    r = V.norm(dim=1)
    d = V / r.clamp(min=1e-12).unsqueeze(1)
    dir_idx = (d @ dir_codebook.T).argmax(dim=1)
    mag_idx = (r.unsqueeze(1) - mag_codebook.unsqueeze(0)).abs().argmin(dim=1)
    return dir_idx, mag_idx, r, d


def dequantize_vectors(dir_idx, mag_idx, dir_codebook, mag_codebook):
    """v_hat = r_hat * d_hat。"""
    d_hat = dir_codebook[dir_idx]
    r_hat = mag_codebook[mag_idx]
    return r_hat.unsqueeze(1) * d_hat


# ---------------------------------------------------------------------------
# 6. 端到端
# ---------------------------------------------------------------------------
def pcdvq_roundtrip(W, dir_codebook, mag_codebook, k):
    p, q = W.shape
    assert p * q % k == 0
    W_reg, scales, signs, H_norm = gaussian_regularize(W)
    V = W_reg.reshape(-1).reshape(-1, k)
    dir_idx, mag_idx, r, d = quantize_vectors(V, dir_codebook, mag_codebook)
    V_hat = dequantize_vectors(dir_idx, mag_idx, dir_codebook, mag_codebook)
    W_hat = inverse_regularize(V_hat.reshape(p, q), scales, signs, H_norm)

    # 分解误差(统一到 MSE 量纲, 与论文 Fig.1(b) 一致):
    #   方向误差贡献 = 2||v||^2 (1-cos),  幅值误差贡献 = (r-r_hat)^2
    cos_sim = (d * dir_codebook[dir_idx]).sum(dim=1)
    dir_err = (2 * r ** 2 * (1 - cos_sim)).mean().item()
    mag_err = ((r - mag_codebook[mag_idx]) ** 2).mean().item()
    rel_mse = (torch.norm(W - W_hat) / torch.norm(W)).item()
    return W_hat, {"dir_err": dir_err, "mag_err": mag_err, "rel_norm_err": rel_mse}


# ---------------------------------------------------------------------------
# 7. 演示与验证
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # 构造带离群值的合成权重 (模拟 LLM 权重的尖峰分布)
    p, q, k = 512, 256, 8
    W = torch.randn(p, q)
    outlier_mask = torch.rand(p, q) < 0.01
    W = W + outlier_mask * (torch.randn(p, q) * 8.0 + 15.0)

    # --- 验证 1: 高斯正则化 ---
    print("=" * 72)
    print("[1] 标准高斯正则化 (RHM, 修正参考实现 bug)")
    W_reg, scales, signs, H_norm = gaussian_regularize(W)
    wa, wr = W.numpy().ravel(), W_reg.numpy().ravel()
    print(f"    {'':16} {'mean':>8} {'std':>8} {'偏度':>8} {'峰度':>8}")
    print(f"    {'原始权重':16} {wa.mean():>8.3f} {wa.std():>8.3f} "
          f"{stats.skew(wa):>8.3f} {stats.kurtosis(wa):>8.3f}")
    print(f"    {'正则化后':16} {wr.mean():>8.4f} {wr.std():>8.4f} "
          f"{stats.skew(wr):>8.4f} {stats.kurtosis(wr):>8.4f}")
    print("    (标准正态: 偏度=0, 超额峰度=0; 离群值被 Hadamard 摊平后偏度/峰度大幅下降)")
    z = inverse_regularize(W_reg, scales, signs, H_norm)
    print(f"    逆变换误差 : {(W - z).abs().max().item():.2e} (应≈0)")

    # --- 验证 2: PCD 的幅值分布 = 根卡方 (chi) ---
    print("=" * 72)
    print("[2] PCD: 幅值 r 应服从根卡方分布 (chi, dof=k)")
    V = W_reg.reshape(-1).reshape(-1, k)
    phi, r, d = pcd(V)
    r2 = (r ** 2).numpy()
    ks_r2 = stats.kstest(r2, "chi2", args=(k,))
    print(f"    r^2 经验 mean={r2.mean():.3f} (理论 {k}), var={r2.var():.3f} "
          f"(理论 {2*k}), KS-D={ks_r2.statistic:.4f}")
    print(f"    单位方向 norm = {d.norm(dim=1).mean().item():.6f} (应=1)")
    print(f"    单位方向坐标 mean={d.mean():+.4f} std={d.std():.4f} (应≈0 / {1/np.sqrt(k):.4f})")

    # --- 验证 3: 码本构造 ---
    print("=" * 72)
    print("[3] 分布对齐码本 (DACC)")
    e8 = e8_minimal_directions()
    print(f"    E8 最小向量方向数 = {e8.shape[0]} (单位化)")
    C_dir = construct_direction_codebook(e8, a=6)
    C_mag = construct_magnitude_codebook(k, b=2)
    print(f"    方向码本 shape = {tuple(C_dir.shape)}, 幅值码本 = {C_mag.tolist()}")

    # --- 验证 4: 位分配扫描 (方向 bits 更重要) ---
    print("=" * 72)
    print("[4] 位分配扫描 (k=8): 方向 bits a 与幅值 bits b 对误差的影响")
    print(f"    {'a':>3} {'b':>3} {'总bits':>6} {'BPW':>6} {'方向err':>10} {'幅值err':>10} {'相对L2err':>10}")
    for a in (4, 5, 6, 7):
        cd = construct_direction_codebook(e8, a)
        for b in (1, 2, 3):
            cm = construct_magnitude_codebook(k, b)
            _, m = pcdvq_roundtrip(W, cd, cm, k)
            bpw = (a + b) / k
            print(f"    {a:>3} {b:>3} {a+b:>6} {bpw:>6.3f} "
                  f"{m['dir_err']:>10.4f} {m['mag_err']:>10.4f} {m['rel_norm_err']:>10.5f}")
    print("    结论: 方向误差远大于幅值误差(方向对量化更敏感); 增加 a 比增加 b 更能降低方向误差。")

    # --- 验证 5: 与耦合 KMeans VQ 对比 (相同总 bits, 分解方向/幅值误差) ---
    print("=" * 72)
    print("[5] 对比: PCDVQ vs 耦合 KMeans VQ (相同总 bits, 分解方向/幅值误差)")
    from scipy.cluster.vq import kmeans2
    Vn = V.numpy()
    print(f"    {'总bits':>6} {'方法':>8} {'方向err':>10} {'幅值err':>10} {'相对L2':>10}")
    for a, b in ((6, 2), (7, 3)):
        cd = construct_direction_codebook(e8, a)
        cm = construct_magnitude_codebook(k, b)
        _, m_pcd = pcdvq_roundtrip(W, cd, cm, k)
        centroids, labels = kmeans2(Vn.astype(np.float64), 2 ** (a + b), minit="++", seed=0)
        cc = torch.tensor(centroids, dtype=torch.float32)
        lab = ((V.unsqueeze(1) - cc.unsqueeze(0)) ** 2).sum(2).argmin(1)
        rk = cc.norm(dim=1)
        dk = cc / rk.clamp(min=1e-12).unsqueeze(1)
        dkm = dk[lab]
        dir_err_km = (2 * r ** 2 * (1 - (d * dkm).sum(1))).mean().item()
        mag_err_km = ((r - rk[lab]) ** 2).mean().item()
        rel_km = (torch.norm(V - cc[lab]) / torch.norm(V)).item()
        print(f"    {a+b:>6} {'PCDVQ':>8} {m_pcd['dir_err']:>10.4f} {m_pcd['mag_err']:>10.4f} "
              f"{m_pcd['rel_norm_err']:>10.5f}")
        print(f"    {'':>6} {'KMeans':>8} {dir_err_km:>10.4f} {mag_err_km:>10.4f} {rel_km:>10.5f}")
    print("    说明: 在合成各向同性数据上, KMeans(耦合+欧氏距离)直接最小化 MSE, 故总/方向误差更低。")
    print("          PCDVQ 的收益来自解耦: 在真实 LLM 权重(方向各向异性 + 幅值重尾)上, 欧氏距离")
    print("          对幅值误差过度加权、挤占方向分辨率; 该结论需用真实模型+下游任务验证, 超出本 demo 范围。")

    # --- 验证 6: 扩展 E8 枚举支持大 a (论文真实位宽 a=14/16) ---
    print("=" * 72)
    print("[6] 扩展: 枚举 E8 方向以支持论文真实位宽 (最小向量仅 240 个, a<=7)")
    big = enumerate_e8_directions(max_sq_norm=8)
    print(f"    E8 格半径内候选方向数 = {big.shape[0]}")
    cd10 = construct_direction_codebook(big, a=10)
    print(f"    a=10 方向码本 shape = {tuple(cd10.shape)} (2^10=1024, 需候选>=1024)")


if __name__ == "__main__":
    main()