# PCDVQ 复现调研与最小可验证 Demo

- 论文：*PCDVQ: Enhancing Vector Quantization for LLMs via Polar Coordinate Decoupling*（arXiv:2506.05432v2, Houmo AI / 哈工大深圳）
- 非官方参考实现：`danlee65071/pcdvq/notebooks/pcdvq.ipynb`
- 本目录：`experiment/PCDVQ-k3`（`pcdvq/` 实现 + `demo.py` + `tests/`）

---

## 1. 论文方法要点

PCDVQ 是**权重量化**框架，核心论点是"向量的**方向**比**幅值**对量化敏感得多"，因此应
**解耦量化并给方向分配更多 bit**。三个模块：

1. **Standard Gaussian Regularization（3.2.1）**
   对权重（逐列）做随机化 Hadamard 变换（RHT）再除尺度，使权重分布统一为 `N(0,1)`，
   从而方向 ≈ 球面均匀分布、幅值 ≈ 卡方根分布，便于用解析分布构造码本。

2. **Polar Coordinate Decoupling, PCD（3.2.2）**
   把 k 维向量变换为极坐标 `(phi_1..phi_{k-1}, r)`：
   `phi_i = atan2(sqrt(Σ_{j>i} v_j²), v_i)`，`r = ||v||`。
   方向与幅值分别用两个码本 `C_phi ∈ R^{2^a×(k-1)}`、`C_r ∈ R^{2^b}` 独立量化：
   方向用余弦相似度 `argmax cos`，幅值用欧氏 `argmin`。总索引位 `a+b`。

3. **Distribution Aligned Codebook Construction, DACC（3.2.3）**
   - 方向码本：E8 格方向（球面最均匀）上贪心最远点采样（FPS）取 `2^a` 个方向。
   - 幅值码本：对 `r² ~ chi²(k)` 的根分布 PDF 做 Lloyd-Max 最优一维标量量化取 `2^b` 个码字。
   - 码本**零训练、模型无关、全模型共享**（只离线构造一次）。

论文配置：`k=8`，幅值固定 `b=2`；`a=14`（称 2-bit）、`a=16`（称 2.125-bit）。

---

## 2. 非官方参考 notebook 逐条分析

**重要结论：该 notebook 是"半成品"，且含实质错误，不能照抄。**

- **内容不完整**：全 24 个 cell，最后一个 cell 为空。只实现了 `Hadamard 构造 / 高斯正则化 /
  PCD / E8 最小壳层 / 方向码本贪心 / 幅值码本 Lloyd-Max`，**没有**真正对一个权重/模型做
  编码-解码-重建或 PPL 评测。所谓参考实现并未复现论文的实验。

- **缺陷 1（严重）：高斯正则化从未做 Hadamard 混合。**
  `gaussian_regularization` 构造了 `hadamard_matrix` 却**完全没用**，实际只乘了一个随机
  对角符号阵 `diag(±1)`。对角阵不混合坐标，权重分布**没有**被高斯化。此外逐列重采样 `s`，
  且 `scaling_factors` 建成长度 `p` 却按列索引 `.shape[1]`，语义错乱。
  → 本实现补回 Hadamard 混合并给出可逆、方差为 1 的正确尺度（见 §3）。

- **缺陷 2：`construct_hadamard_matrix` 只支持 2 的幂。**
  `int(np.log2(order))` 对非 2 幂维度会静默截断（如 q=100 → 64×64），随后矩阵乘形状不匹配。
  → 本实现用**分块 Hadamard** `S = I_{q/b} ⊗ H_b`（`b` 为整除 q 的最大 2 的幂）支持任意维度。

- **缺陷 3：PCD 丢失最后一维符号（沿袭论文 Eq.6 的字面错误）。**
  论文/notebook 对最后一个角也用 `sqrt(v_k²)=|v_k|`，得到 `phi_{k-1} ∈ [0,π]`，重建时
  `v_k = r·Πsin·sin(phi_{k-1}) ≥ 0`，**永远是正的**。变换非双射，一半方向塌缩。
  → 本实现改用标准超球坐标 `phi_{k-1} = atan2(v_k, v_{k-1}) ∈ (-π,π]`，双射（demo B/单测验证）。

- **缺陷 4：E8 方向池只有最小壳层且未单位化。**
  `e8_minimal_directions` 仅返回 norm²=2 的 **240** 个点，无法支撑 `a≥8`（需 2^a 个码字）；
  且 E8 点范数为 `√2`（非单位），若直接与单位方向比余弦/用于重建会引入尺度错误。
  → 本实现枚举 `norm²≤max_norm_sq` 的全部壳层（norm²≤4→2400，≤8→26640），排除原点并**单位化**。

- **缺陷 5：论文自身位算术矛盾。** `(a+b)/k`：`a=16,b=2,k=8` 应为 **2.25**，论文正文却称 2.125
  （A.3 亦然）。真正 2.125 需 `a=15`。复现时应以 `(a+b)/k` 公式为准。

**正确的部分（经数值核对）：**

- `find_max_r_bisection`：二分求 `F(max_r)=chi2.cdf(max_r²;k)=tau`，正确（与 `chi2.ppf` 一致）。
- `construct_magnitude_codebook`（Lloyd-Max）：**实现正确**。用 `gammainc` 表达条件均值
  `E[r|a<r<b] = √2·Γ((k+1)/2)/Γ(k/2)·Δγ((k+1)/2)/Δγ(k/2)`，本实现逐位复现。
  - **注意**：notebook 输出的 `[1.4241, 1.874, …]` 是 `tol=1e-3` 下的**未收敛中间解**；
    用相同超参本实现**逐位一致**；完全收敛（`tol=1e-8`）的固定点为 `[1.4396, 1.8956, …]`。
    两者都验证了实现正确（单测 `test_magnitude_codebook_golden`）。
- E8 贪心 FPS：notebook 实现的是**正确的**最远点采样（论文 Algorithm 1 的伪代码反而有
  `if current_max` 的笔误，notebook 用 `argmin(max_sim)` 是对的）。

---

## 3. 本实现的修正要点

| 模块 | 修正 |
|---|---|
| RHT 高斯化 | 补回 Hadamard 混合 `y = S(Dx)/s`，`s = ‖x‖·√(b/q)`，逆变换 `x = D·Sᵀ(s·ŷ)/b`，可逆且元素方差为 1 |
| Hadamard | 分块 `I⊗H_b`，支持任意维度（如 q=5120=5×1024、17408=17×1024） |
| PCD | 最后一角 `atan2(v_k, v_{k-1})`，双射、保留符号 |
| E8 池 | 枚举多壳层 + 排除原点 + 单位化 |
| 尺度/索引 | 逐向量尺度语义一致；BPW 严格用 `(a+b)/k` |

验证（`demo.py` 实测）：

```
A. RHT: kurtosis 230.6 → 2.27, std → 1.0000, 往返误差 6e-7
B. PCD: 往返误差 5e-7；含负末维向量同样可精确重建；末角范围 (-π,π]
C. E8: norm²≤2→240, norm²≤4→2400；幅值码本与 notebook 用相同超参逐位一致
   方向码本(a=8) 最小两两余弦 0.7071 vs 随机 0.9696（更均匀）
D. 相同码本规模下，只量化方向的误差远大于只量化幅值（4中心: 9.46 vs 0.052）
   → 复现论文"方向更敏感"的核心动机
E. 端到端：PCDVQ 幅值 MSE 始终更低；但低 bit(<2bit) 档位自由 k-means 更强
F. DACC 消融：E8 贪心最均匀（最大两两余弦 0.75）；球面 k-means 因直接最小化 MSE 目标而 MSE 更低
```

---

## 4. 如何运行

```bash
# 依赖: python>=3.9, torch(CPU即可), numpy, scipy, pytest
cd experiment/PCDVQ-k3
python demo.py          # 打印 6 组验证
python -m pytest -q     # 9 个单元测试
```

代码结构：

```
pcdvq/
  rht.py         RHT 标准高斯正则化（含逆变换）
  polar.py       PCD 极坐标解耦（符号修正）
  codebooks.py   E8 方向码本 + chi Lloyd-Max 幅值码本
  quantizer.py   PCDVQ 编解码器
demo.py          端到端最小可验证 demo
tests/           单元测试
```

---

## 5. 结论与局限

- **方法可行性验证通过**：PCDVQ 的三大模块（RHT 高斯化、PCD 解耦、DACC 码本）均已在最小规模上
  正确实现并数值验证；与参考 notebook 在正确部分逐位一致，并修正了其 5 处缺陷 + 1 处论文缺陷。
- **未能复现论文的端到端精度优势**：本 demo 在 BPW≈1.25~1.5 的档位、以自由训练的 k-means 为基线时，
  PCDVQ 的权重重建/方向 MSE 反而更高。原因是：(1) 该档位的方向码本分辨率（2^8~2^10）远低于论文
  （a=14~16）；(2) E8 贪心 FPS 优化的是"最坏情况覆盖"而非 MSE；(3) 论文的端到端优势依赖真实 LLM
  的分布 + QuIP# 式 block-wise/e2e 微调，非最小 demo 所能覆盖。
- **对主任务的启示（衔接 `feasibility.md`）**：PCDVQ 单方法在极低 bit 有价值，但要达到"精度损失<0.5% +
  更低平均 bit"的目标，应按 `feasibility.md §2` 的方案——**PCDVQ 只承担低敏感度组（MLP），
  高敏感度组用 GPTQ-INT，再用 SLQ 的 Shapley/线性敏感度做混合精度分配**。本 demo 已验证 PCDVQ
  的编码器/码本可作为该框架的"低位宽算子"正确接入（fake-quant 权重 → 逆 RHT 回权重空间）。
