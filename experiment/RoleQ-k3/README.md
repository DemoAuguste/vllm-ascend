# RoleQ 复现（experiment/RoleQ-k3）

- 论文：*RoleQ: Role-Aligned Scales for Converting Language Models to Ternary*（arXiv:[2610.21808v1](https://arxiv.org/abs/2610.21808)，Harvard / Georgia Tech，2026-10-15）
- 官方代码仓：**未找到**（核查见 §1）
- 定位：论文核心方法的独立参考实现 + CPU 可验证 demo；不含完整 LLM 转换流水线
- bit 扩展可行性与 0.5% 目标分析：[`BIT_EXTENSION.md`](BIT_EXTENSION.md)

---

## 1. 官方代码仓核查结论

截至 2026-10-19，**论文未提供官方代码仓**：

| 核查项 | 结果 |
|---|---|
| arXiv 摘要页 / PDF / HTML | 无 code、GitHub、project page 链接 |
| arXiv TeX 源（`/src/2610.21808`） | 全文无任何 URL，无 "code available" 声明 |
| GitHub repo 搜索 `RoleQ ternary quantization` | 0 条 |
| GitHub repo 搜索 `role-aligned quantization ternary` | 0 条 |
| GitHub repo 搜索 `RoleQ-T` | 0 条 |
| 同名 `RoleQ` 仓库 | `Ankithac2004/RoleQ` 为 TypeScript 项目，与论文无关 |

因此本目录按论文公式（Eq. 2、§3.3 推导）与附录 D 的求解器设置从零实现，未参考任何官方源码。

---

## 2. 方法拆解

RoleQ 的核心不是新的量化网格，而是 **scale 放在哪条轴上**。

### 2.1 放置规则（§3.1）

| 角色 | 矩阵 | scale 轴 |
|---|---|---|
| reader（读残差流） | `W_Q, W_K, W_V, W_gate, W_in` | 每个**输出单元**一个 scale（行，per-OC） |
| writer（写残差流） | `W_O, W_out` | 每个**输入单元**一个 scale（列，per-IC） |

重建：reader 为 `diag(a)·T`，writer 为 `T·diag(a)`（Eq. 3）。规则是结构性的，无搜索、无校准、无 per-layer flag。

### 2.2 精确 ternary 向量拟合（Eq. 2）

对向量 `w`、尺度 `a`、码 `t ∈ {-1,0,+1}`：

```
t = sign(w) ⊙ [|w| > τ]
a*(S) = mean_{i∈S} |w_i|
S* = argmax_S (Σ_{i∈S}|w_i|)² / |S|
```

`S` 取 `|w|` 降序排列的非空前缀；一次排序 + 前缀扫描即得全局最优（`O(n log n)`）。

### 2.2b 任意 b-bit 对称网格

`bits=1` 即 ternary；`bits>=3` 使用 `qmax = 2^(bits-1)-1` 的对称均匀网格：

```
w_hat = s * clamp(round(w / s), -qmax, qmax)
```

- 候选 scale 取舍入断点 `|w|/(k+1/2)` 与 `max|w|/qmax`，再对固定码做闭式最小二乘精修；
- reader/writer 的 scale 轴规则不变，reader 精修公式对任意网格成立；
- `bits=2` 需要 4 电平非对称网格（带 zero point），当前实现显式报错。

### 2.3 RoleQ-T 求解器（§3.3）

GPTQ 在 `H = E[xxᵀ]` 上逐列扫描并补偿误差，两种角色的拟合时机不同：

- **writer（scale 在扫描轴上）**：每列对已补偿的列向量直接做 Eq. 2 的精确向量拟合，再传播误差。
- **reader（scale 在正交轴上）**：行网格 `{-a_i, 0, a_i}` 由 Eq. 2 初始化后在扫描中固定；扫描结束后按 H-度量闭式精修  
  `a_i = t_iᵀ H w_i / (t_iᵀ H t_i)`，不增加重建误差。

附录 D 设置：act-order、damping `0.01 × mean(diag H)`、lazy block 128；reader 用 LN 输入度量，`W_O` 用 `E[zzᵀ]`。

### 2.4 轻量转换流水线（§3.2–3.5）

1. **旋转**：`r1`（残差基）、`r2`（V→O，每 KV head）、`r4`（MLP 中间维，block 1024）。配方为 `s1(l4) + s2(opt) + r4`；`s1/s2` 仅用权重学习，无校准。
2. **RoleQ-T GPTQ**：上一节。
3. **Scale distillation（+SD）**：冻结码，只训练 `log a` 与重新引入的 norm gain（8B 约 0.024% 参数）；KL 对齐缓存 teacher 残差，400 步 × batch 8，lr 3e-3，RedPajama 3200 个 512-token 窗口。
4. **Fisher 稀疏第二平面（可选）**：`S_u = Σ_{i∈u} F_i δ_i²` 给 unit 排序，top-K 加第二 ternary 平面；8B 约 +0.57 bpw。

---

## 3. 需求分析

### 3.1 论文级完整复现需要什么

| 维度 | 论文设置 |
|---|---|
| 模型 | Qwen3-0.6B/1.7B/8B/14B-Base、Qwen3-1.7B/8B post-trained、Llama-3.1-8B-Base/Instruct |
| 校准 | C4 `512 × 512`（post-trained：272 条 chat 序列 × ≤1536 token） |
| 评测 | WikiText2 PPL、7 任务 zero-shot（lm-eval 0.4.13）、5-shot MMLU |
| 算力 | 8B 单平面 + SD 约 3–4 A100 小时；加 planes 约 10 小时；SD 阶段 8B 约 2 h/A100 |
| 显存 | Fisher probe：1.7B 需 17 GB host；8B+ 需 H200 |
| 依赖 | PyTorch + HF 模型 + 校准语料；无官方 code 可直接复用 |

### 3.2 最小可验证 demo 的范围

目标不是复现论文数值，而是**可判定地验证方法中确定性的部分**：

1. Eq. 2 的精确拟合是否真的是全局最优；
2. “reader 行 / writer 列”规则是否让同一模型同时获得两侧的最优轴；
3. GPTQ 适配（writer 精确列步、reader 固定网格 + H-度量精修）是否按论文语义工作；
4. `1.590 bpw` 的 resident 记账是否复算一致；
5. b-bit 网格（3/4/5/6-bit）下 RoleQ 是否仍同时取到两侧最优。

据此裁剪为：CPU 单层 residual block + 合成重尾权重 + 随机 Hadamard `r1`。  
**不包含**：SD、Fisher planes、学习式 `s1/s2`、`r4` 运行时 hook、2-bit 非对称网格、g128、真实模型 PPL/MMLU。

### 3.3 运行环境

本目录只用 `torch`（CPU 即可）。本机默认 `python` 无 torch，使用：

```bash
py -3.10 demo.py
py -3.10 -m pytest -q tests
```

实测环境：Python 3.10.1 + torch 2.9.1+cpu。

---

## 4. 论文 → 代码映射

| 论文要素 | 公式/章节 | 实现 |
|---|---|---|
| 精确 ternary 向量拟合 | Eq. 2 | [`roleq/core.py`](roleq/core.py) `exact_ternary_fit` |
| b-bit 对称网格拟合 | App. A / Table 3 | `exact_grid_fit(..., bits=b)` |
| reader/writer 角色 → scale 轴 | §3.1 | `role_axis`、`infer_role` |
| 重建格式 | Eq. 3 | `quantize_rtn` / `quantize_gptq` 的返回 |
| writer 精确列步 | §3.3 | `quantize_gptq(..., axis="col")` |
| reader 固定网格 | §3.3 | `_row_grid_quantize` |
| reader H-度量精修 | §3.3, App. D | `refit_row_scales` |
| GPTQ damping / act-order | App. D | `_prepare_upper_cholesky`、`act_order` |
| r1 随机 Hadamard | §3.2, App. F | `random_hadamard`、`rotate_reader/writer` |
| resident bpw | App. A | `resident_bpw` |
| RTN 对照 | §3 | `quantize_rtn`（支持 `bits`） |
| 单层端到端验证 | — | [`demo.py`](demo.py) `part_c` |
| bit sweep | Table 3 | [`demo.py`](demo.py) `part_f` |
| 单测 | — | [`tests/test_roleq.py`](tests/test_roleq.py) |

---

## 5. 验证结果（本机实测）

### A. 精确拟合 = 穷举最优

- 200 个随机重尾向量（n=3..7）：与穷举所有 support 的最大误差偏差 `1.19e-7`（浮点级）。
- 精确拟合劣于 TWN 阈值规则的次数：`0 / 200`。

### B. r1 只混合流坐标，不动单元轴

- reader 行范数最大变化：`3.05e-5`
- writer 列范数最大变化：`4.58e-5`
- writer 行范数（残差流坐标）最大变化：`32.16`

即：RoleQ 依赖的 unit-axis 统计在旋转下不变，而 per-OC 依赖的 stream-axis 统计被彻底混合。

### C. 单层 residual block（合成，相对误差，越低越好）

原始基：

| placement | reader err | writer err | end-to-end |
|---|---:|---:|---:|
| uniform per-OC | 0.1450 | 0.1229 | 0.5295 |
| uniform per-IC | 0.3171 | 0.0397 | 0.5863 |
| **RoleQ** | **0.1450** | **0.0397** | **0.4729** |

r1 旋转基：

| placement | reader err | writer err | end-to-end |
|---|---:|---:|---:|
| uniform per-OC | 0.0468 | 0.1176 | 0.3926 |
| uniform per-IC | 0.1582 | 0.0411 | 0.4862 |
| **RoleQ** | **0.0468** | **0.0411** | **0.2893** |

RoleQ 相对最佳 uniform 放置的端到端误差改善：原始基 `1.12×`，旋转基 `1.36×`。  
与论文一致的结构性结论：**任何单一轴都会在一侧输给另一侧，RoleQ 同时取到两侧的最小值**。

### D. 求解器消融

- reader：RTN `0.1948` → GPTQ `0.1450`（降低 25.5%）
- writer：RTN `0.1791` → GPTQ `0.0397`（降低 77.9%）
- reader 精修：H-误差 `0.1541 → 0.1450`；欧氏误差 `4272 → 4444`（H-度量精修不保证欧氏误差下降，符合推导）

### E. bit accounting

- `log2(3) = 1.584963`
- fp16 per-vector scale：`16/4096 = 0.003906`
- + norm gains `0.0007` → 合计 `1.5896 bpw`，与论文 `1.590` 一致。
- 3/4/5/6/8-bit 分别为 `2.812 / 3.912 / 4.959 / 5.982 / 7.999 bpw`。

### F. bit sweep（合成单层 residual block，GPTQ）

| bits | bpw | per-OC e2e | per-IC e2e | **RoleQ e2e** | RoleQ gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.590 | 0.5295 | 0.5863 | **0.4729** | 1.12× |
| 3 | 2.812 | 0.3362 | 0.3956 | **0.2816** | 1.19× |
| 4 | 3.912 | 0.1701 | 0.2560 | **0.1175** | 1.45× |
| 5 | 4.959 | 0.0937 | 0.1670 | **0.0614** | 1.53× |
| 6 | 5.982 | 0.0528 | 0.1199 | **0.0347** | 1.52× |

RoleQ 在所有已支持 bit 下都不劣于最佳 uniform 轴。  
注意：该合成 block 的 unit-axis 重尾很强，gain 不随 bit 衰减；论文真实模型上 INT3/INT4 的放置收益明显小于 ternary（见 `BIT_EXTENSION.md` §8.3）。

---

## 6. 与论文的差异（明确未覆盖）

| 项 | 状态 | 说明 |
|---|---|---|
| Scale distillation | 未实现 | 需要真实模型、teacher 缓存与训练循环 |
| Fisher secondary planes | 未实现 | 需要真实数据上的梯度/Fisher probe |
| 学习式 `s1(l4)/s2(opt)` | 未实现 | 仅提供随机 Hadamard `r1` |
| `r4` 运行时 Hadamard | 未实现 | 需要模型级 hook 与 kernel 配合 |
| 模型级 fake-quant / PPL / MMLU | 未实现 | demo 只做矩阵与单 block 级验证 |
| 3/4/5/6-bit 对称网格 | 已实现 | `bits` 参数；2-bit 4 电平非对称未实现 |
| 分组尺度 g128 / MXFP4 | 未实现 | 论文结论中 g128 会削弱 RoleQ 的边际收益 |
| GPTQ lazy block 128 | 未实现 | 顺序扫描，结果等价，仅速度不同 |
| 数值对比 | 不可比 | demo 用合成权重，数值不能与论文 PPL/MMLU 对照 |

另外，论文正文与附录在若干细节上未给出完整伪代码（如 `s1/s2` 的优化器与步数、SD 的 norm gain 注入位置、Fisher probe 的分层批处理），本实现只覆盖有明确公式的部分。

---

## 7. 建议的下一步

1. **模型级 PTQ**：在 Qwen3-0.6B 上按模块名映射角色，逐层捕获 C4 `512×512` 的 `H`，执行 `quantize_gptq`，用 fake-quant 评估 WikiText2 PPL，先验证放置规则的端到端收益。
2. **bit sweep**：在 Qwen3-0.6B/1.7B 上跑 bits ∈ {1,3,4,5,6}，对照论文 INT3sym/INT4sym，确认真实权重上放置收益随 bit 衰减的趋势。
3. **加入 SD**：冻结 codes，训练 `log a` + norm gains，KL 对齐缓存 teacher 残差；这是论文 1.590 bpw 数值的主要来源。
4. **加入 Fisher planes**：在真实模型上累积对角 Fisher，按 unit 排序加第二平面，复现 `+0.57 bpw` 的知识恢复。
5. **旋转与 kernel**：`r1/r2` 折入权重，`r4` 用运行时 Hadamard；Ascend 侧可先做 lookup-table / add-only 的 ternary GEMV 验证。
