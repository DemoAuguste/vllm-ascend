"""RoleQ-T 核心算法（论文 arXiv:2610.21808）。

只实现可确定性验证的部分：
  1. 精确 ternary 向量拟合（Eq. 2）
  2. 角色对齐的 scale 轴（reader=行，writer=列）
  3. RoleQ-T GPTQ 扫描：writer 列向量精确拟合、reader 固定网格 + H-度量精修
  4. r1 随机 Hadamard 旋转与 resident bpw 统计

不包含（见 README「范围与差异」）：scale distillation、Fisher secondary planes、
学习式 s1/s2 旋转、r4 运行时 Hadamard、模型级 fake-quant 内核。
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch

Role = Literal["reader", "writer"]
Axis = Literal["row", "col"]

TERNARY_BITS = math.log2(3.0)

READER_KINDS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "fc1", "w1", "w3")
WRITER_KINDS = ("o_proj", "down_proj", "fc2", "w2")


# ---------------------------------------------------------------------------
# 1. 精确 ternary 拟合（论文 Eq. 2）
# ---------------------------------------------------------------------------
def exact_ternary_fit(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """对最后一维的每个向量求 min_{a,t} ||v - a t||^2, t in {-1,0,+1}^n。

    论文 Eq. 2：按 |w| 降序取前缀 S，最大化 (sum_{i in S} |w_i|)^2 / |S|，
    尺度 a = mean_{i in S} |w_i|，码 t = sign(w) * 1[|w| > tau]。

    返回 codes（float32，取值 -1/0/1）与 scale（前导维形状）。
    """
    if v.ndim == 0:
        v = v.reshape(1)
    n = v.shape[-1]
    abs_v = v.abs()
    sorted_abs, order = torch.sort(abs_v, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_abs, dim=-1)
    k = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    # 最小化 ||v||^2 - cumsum^2/k，等价于最大化 cumsum^2/k
    best_k = torch.argmax(cumsum * cumsum / k, dim=-1) + 1
    scale = cumsum.gather(-1, (best_k - 1).unsqueeze(-1)).squeeze(-1) / best_k.to(v.dtype)

    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(n, device=v.device).expand_as(order))
    support = rank < best_k.unsqueeze(-1)
    codes = torch.sign(v) * support.to(v.dtype)
    return codes, scale


def levels_per_code(bits: int) -> int:
    """每码电平数：ternary 为 3；b>=3 的对称网格为 2^b - 1。"""
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    if bits == 2:
        raise NotImplementedError(
            "2-bit 需要 4 电平非对称网格（带 zero point），当前实现只支持对称网格"
        )
    return 3 if bits == 1 else 2**bits - 1


def _qmax(bits: int) -> int:
    """对称网格的最大绝对值码：ternary 为 1；b>=3 为 2^(b-1)-1。"""
    if bits == 2:
        raise NotImplementedError(
            "2-bit 需要 4 电平非对称网格（带 zero point），当前实现只支持对称网格"
        )
    return 1 if bits == 1 else 2 ** (bits - 1) - 1


def exact_grid_fit(
    v: torch.Tensor,
    bits: int = 1,
    max_candidates: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称 b-bit 均匀网格的逐向量精确拟合（bits=1 时即 ternary Eq. 2）。

    网格为 s * {-qmax, ..., 0, ..., qmax}，qmax = 2^(bits-1)-1。
    候选 scale 取舍入断点 |w|/(k+1/2) 与 max|w|/qmax，逐候选计算舍入 MSE，
    再用固定码的闭式最小二乘 s = <w,q>/||q||^2 精修（ternary 为全局精确）。

    max_candidates 限制候选数（对数间隔采样），用于大向量/模型级量化时的加速；
    None 表示使用全部舍入断点候选（demo 的单测使用该模式验证质量）。
    """
    if v.ndim == 0:
        v = v.reshape(1)
    if bits == 1:
        return exact_ternary_fit(v)

    qmax = _qmax(bits)
    orig_shape = v.shape
    n = orig_shape[-1]
    flat = v.reshape(-1, n).to(torch.float32)
    codes = torch.zeros_like(flat)
    scales = torch.zeros(flat.shape[0], dtype=torch.float32, device=v.device)

    for r in range(flat.shape[0]):
        w = flat[r]
        if float(w.abs().max()) == 0.0:
            continue
        abs_sorted, _ = torch.sort(w.abs(), descending=True)
        if max_candidates is None:
            # 舍入断点 |w| / (k + 1/2)，外加 max|w| / qmax
            bps = torch.cat(
                [
                    torch.arange(0.5, float(qmax), 1.0, device=v.device),
                    torch.tensor([float(qmax)], device=v.device),
                ]
            )
            cand_scales = (abs_sorted.unsqueeze(0) / bps.unsqueeze(1)).flatten()
        else:
            steps = max(2, max_candidates)
            raw = torch.logspace(
                0.0,
                math.log10(max(n - 1, 1)),
                steps=steps,
                device=v.device,
            )
            cand_idx = torch.unique(raw.round().long())
            cand_scales = abs_sorted[cand_idx] / qmax

        cand_scales = torch.unique(cand_scales[cand_scales > 0])
        q = torch.clamp(
            torch.round(w.unsqueeze(0) / cand_scales.unsqueeze(1)), -qmax, qmax
        )
        err = ((w.unsqueeze(0) - cand_scales.unsqueeze(1) * q) ** 2).sum(dim=-1)
        best = torch.argmin(err)
        best_scale = cand_scales[best]
        best_codes = q[best]
        best_err = err[best]

        # 固定码的最小二乘精修；只接受误差下降的更新
        for _ in range(4):
            denom = (best_codes * best_codes).sum()
            if float(denom) == 0.0:
                break
            s_new = (w * best_codes).sum() / denom
            if float(s_new) <= 0.0:
                break
            q_new = torch.clamp(torch.round(w / s_new), -qmax, qmax)
            e_new = ((w - s_new * q_new) ** 2).sum()
            if float(e_new) < float(best_err):
                best_scale, best_codes, best_err = s_new, q_new, e_new
            else:
                break

        scales[r] = best_scale
        codes[r] = best_codes

    return codes.reshape(orig_shape), scales.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# 2. 角色对齐的 scale 轴
# ---------------------------------------------------------------------------
def role_axis(role: Role) -> Axis:
    """RoleQ 放置规则：reader 每个输出单元一个 scale（行），writer 每个输入单元一个 scale（列）。"""
    if role == "reader":
        return "row"
    if role == "writer":
        return "col"
    raise ValueError(f"unknown role: {role}")


def infer_role(module_name: str) -> Role | None:
    """按 HF/Llama/Qwen 线性层命名推断 reader / writer 角色。"""
    name = module_name.lower()
    if name.endswith(READER_KINDS):
        return "reader"
    if name.endswith(WRITER_KINDS):
        return "writer"
    return None


# ---------------------------------------------------------------------------
# 3. GPTQ 基础设施
# ---------------------------------------------------------------------------
def _prepare_upper_cholesky(H: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """返回上三角 U，使 H^{-1} = U^T U（GPTQ 参考实现的 damping / dead-column 处理）。"""
    n = H.shape[0]
    H = H.to(torch.float32).clone()
    diag = torch.diag(H)
    dead = diag <= 0
    if dead.any():
        H[dead, dead] = 1.0
    mean_diag = torch.diag(H).mean()
    H[torch.arange(n, device=H.device), torch.arange(n, device=H.device)] += damp * mean_diag
    Hinv = torch.linalg.inv(H)
    return torch.linalg.cholesky(Hinv, upper=True)


def weighted_error(W: torch.Tensor, Wq: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """tr[(W-Wq) H (W-Wq)^T]，即 GPTQ 的 layer-local 重建目标。"""
    E = (W - Wq).to(torch.float32)
    return torch.sum((E @ H.to(torch.float32)) * E)


def relative_weighted_error(W: torch.Tensor, Wq: torch.Tensor, H: torch.Tensor) -> float:
    denom = weighted_error(W, torch.zeros_like(W), H)
    return float((weighted_error(W, Wq, H) / denom).item())


def _row_grid_quantize(
    w: torch.Tensor, row_scales: torch.Tensor, qmax: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把每个标量就近舍入到固定网格 s_i * {-qmax..qmax}（qmax=1 即 ternary）。"""
    safe = row_scales.clamp_min(1e-12)
    codes = torch.clamp(torch.round(w / safe), -qmax, qmax)
    codes = torch.where(row_scales > 0, codes, torch.zeros_like(codes))
    return codes, codes * row_scales


def refit_row_scales(W: torch.Tensor, codes: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """reader 的 H-度量闭式精修：a_i = t_i^T H w_i / (t_i^T H t_i)。"""
    Wf = W.to(torch.float32)
    T = codes.to(torch.float32)
    TH = T @ H.to(torch.float32)
    num = torch.sum(TH * Wf, dim=1)
    den = torch.sum(TH * T, dim=1).clamp_min(1e-12)
    return num / den


# ---------------------------------------------------------------------------
# 4. RoleQ-T GPTQ
# ---------------------------------------------------------------------------
def quantize_rtn(
    W: torch.Tensor, axis: Axis, bits: int = 1, max_candidates: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """无误差补偿的逐向量 / 逐通道 RTN，作为 GPTQ 的对照。bits=1 为 ternary。"""
    Wf = W.to(torch.float32)
    if axis == "col":
        codes, scales = exact_grid_fit(Wf.T, bits=bits, max_candidates=max_candidates)
        codes = codes.T
        return codes * scales.unsqueeze(0), codes, scales
    codes, scales = exact_grid_fit(Wf, bits=bits, max_candidates=max_candidates)
    return codes * scales.unsqueeze(1), codes, scales


def quantize_gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    axis: Axis,
    damp: float = 0.01,
    act_order: bool = False,
    bits: int = 1,
    max_candidates: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在给定 scale 轴上做 b-bit GPTQ（bits=1 为 RoleQ-T）。

    axis="col": writer / per-IC。扫描列，每列对补偿后的列向量做精确网格拟合，
                尺度落在扫描轴上（论文 §3.3 writer step）。
    axis="row": reader / per-OC。行网格先由 Eq.2 / 网格搜索初始化并在扫描中固定，
                扫描结束后按 H-度量闭式精修（论文 §3.3 reader refit）。

    返回 (Wq, codes, scales)；row 轴时 scales 长度为 out，col 轴时为 in。
    形参顺序与论文一致：W 为 [out, in]，H 为 [in, in]。
    """
    qmax = _qmax(bits)
    Wf = W.to(torch.float32).clone()
    out, inp = Wf.shape
    Hf = H.to(torch.float32)

    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(Hf), descending=True)
        Wf = Wf[:, perm]
        Hf = Hf[perm][:, perm]

    U = _prepare_upper_cholesky(Hf, damp=damp)
    W_ref = Wf.clone()  # GPTQ 补偿会改写 Wf；reader 精修必须用原始权重
    codes = torch.zeros(out, inp, dtype=torch.float32, device=Wf.device)
    col_scales = torch.zeros(inp, dtype=torch.float32, device=Wf.device) if axis == "col" else None
    row_scales = (
        exact_grid_fit(Wf, bits=bits, max_candidates=max_candidates)[1] if axis == "row" else None
    )

    for j in range(inp):
        w = Wf[:, j]
        if axis == "col":
            c, a = exact_grid_fit(w, bits=bits, max_candidates=max_candidates)
            q = a * c
            codes[:, j] = c
            col_scales[j] = a
        else:
            c, q = _row_grid_quantize(w, row_scales, qmax)
            codes[:, j] = c

        err = (w - q) / U[j, j].clamp_min(1e-12)
        if j + 1 < inp:
            Wf[:, j + 1 :] -= err[:, None] * U[j, j + 1 :][None, :]

    if axis == "col":
        Wq = codes * col_scales.unsqueeze(0)
        scales = col_scales
    else:
        scales = refit_row_scales(W_ref, codes, Hf)
        Wq = codes * scales.unsqueeze(1)

    if perm is not None:
        inv = torch.argsort(perm)
        Wq = Wq[:, inv]
        codes = codes[:, inv]
        if axis == "col":
            scales = scales[inv]

    return Wq, codes, scales


# ---------------------------------------------------------------------------
# 5. r1 随机 Hadamard 旋转
# ---------------------------------------------------------------------------
def random_hadamard(d: int, seed: int = 0, device: str | torch.device = "cpu") -> torch.Tensor:
    """随机符号 + 行置换的归一化 Hadamard 矩阵，形状 [d, d]，正交。"""
    if d & (d - 1) != 0:
        raise ValueError(f"d must be a power of two, got {d}")
    H = torch.ones(1, 1, dtype=torch.float32, device=device)
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    H = H / math.sqrt(d)
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=g).to(torch.float32) * 2 - 1
    perm = torch.randperm(d, generator=g)
    return (H[perm.to(device)] * signs.to(device).unsqueeze(0)).contiguous()


def rotate_reader(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """reader 的输入坐标在残差流中，W <- W R。"""
    return W.to(torch.float32) @ R.to(W.device)


def rotate_writer(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """writer 的输出坐标在残差流中，W <- R^T W。"""
    return R.to(W.device).T @ W.to(torch.float32)


# ---------------------------------------------------------------------------
# 6. resident bits per weight
# ---------------------------------------------------------------------------
def resident_bpw(
    num_weights: int, num_scales: int, norm_gain_bits: float = 0.0007, bits: int = 1
) -> float:
    """b-bit 对称网格信息位 + fp16 per-unit scale + 重新引入的 norm gain。"""
    code_bits = math.log2(levels_per_code(bits))
    return code_bits + 16.0 * num_scales / num_weights + norm_gain_bits
