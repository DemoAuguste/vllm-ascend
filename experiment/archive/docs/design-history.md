# 方案演进与决策记录（归档）

> 本文件是 [task.md](../../task.md) 早期版本中被移除的「方案修订 / 论文调研 / 需求分析 /
> 历史结论」部分的原文合并存档，仅作追溯用。**现行方案以 [slq-official/](../../slq-official/)
> 与 [summary-report.md](../../summary-report.md) 为准。**
>
> 逐日进展日志见 [progress-2026-10-16_10-19.md](../logs/progress-2026-10-16_10-19.md)。

---

## 1. 方案修订说明（2026-10-16）

原方案为“PCDVQ（低位档）+ GPTQ-INT（高位档）+ SLQ（Shapley/ILP 分配）”。阶段 4 实测发现
**PCDVQ 纯 fake-quant 无误差补偿时精度灾难性崩溃**（Qwen3-0.6B WikiText-2 PPL
20.04 → 360702），低位档不可用。

经调研（见 [SAGE-feasibility.md](SAGE-feasibility.md)），**改用 SAGE 作为低位档算子**。
SAGE 与 PCDVQ 同属“球坐标解耦 VQ”谱系，但：

- **内建 blockwise 误差补偿**（20 epoch STE，学生输入用前序量化块），直接对应 PCDVQ 崩溃根因；
- 激活感知（ESG 能量分层分组 + 加权球面 k-means）、逐矩阵 ABS 自适应位切分；
- 可学习码本，规避 E8 枚举/单位化/位算术缺陷；官方实现且已在 hybrid（Qwen3.5）/MoE/VLM 上验证。

**结论**：

1. SAGE 可替换 PCDVQ 作为低位档，且更优；但 SAGE 本质是 **≤3bit 极低位方法**
   （工作点约 2.0–2.5 bpw），**不能充当高位档**；
2. 与 SLQ 接口正交、可组合（SAGE 官方导出即密集 QDQ 权重，逐矩阵量化比 SLQ 组更细，
   敏感度黑盒兼容）；
3. 组合方案 = 低位档 SAGE{2.0, 2.125, 2.25} + 高位档 GPTQ-INT{3,4,6,8} + SLQ Shapley/ILP
   分配 + SLQ-TL 单点标定。

> 后续（阶段 5/6）实测 SAGE 在 2.25 bpw 下仍有 +178% PPL 损失，且缺少二阶误差补偿，
> 主线最终放弃所有 VQ，统一改用 GPTQ 全档位 + Shapley + ILP。

---

## 2. SLQ 论文实验结果调研（2026-10-16）

官方 [IST-DASLab/SLQ](https://github.com/IST-DASLab/SLQ)（COLM 2026）两种“无损”定义与实测区间：

| 目标 | 定义 | 达成 bpp（论文） |
|------|------|-----------------|
| **Task-Lossless (TL)** | 7 个 zero-shot 基准平均恢复 **≥99%** BF16（在采样方差内） | **3.3–4.7 bpp** |
| **Distribution-Lossless (DL)** | 下一 token 分布与 BF16 不可分：**EAR ≥ 0.99** | **5.0–6.6 bpp** |

- 论文实证：**zero-shot 准确率在 KL≲0.01 区间保持平坦**，超出后骤降 → **PPL/KL 不等于任务精度**，
  主指标应看下游恢复率。
- **方差律**：对称量化噪声方差比非对称放大约 γ² → **DL 必须非对称**（TL 不必须）。
- 官方流水线：① 逐层 GPTQ，**每个位宽 ∈{2..8}**、**非对称 INT、g128、每组一个 GPTQ block、
  单 Hessian 一次产出全部位宽**，`cal_bits=4` 序贯传播；② 多 bitwidth Shapley 敏感度
  （512×2048 FineWeb-Edu，top-K=10）；③ ILP 多选背包 + 二分搜索最小 bpp；
  ④ 拼装 HF checkpoint（fake / humming_online）。
- **bpp 计费**：g128 有 **+0.156** 开销（2→2.156 … 8→8.156）。
- 评测：`run_lmeval`（zero-shot×3 seeds）、`run_ear`、`run_kl`、`run_ppl`。

> **与本任务的关系**：任务要求“损失 <0.5%” → 恢复率 **≥99.5%**，比论文 TL（99%）更严、
> 指向 DL 的 bpp 区间；这解释了用户结论“**0.6B 需 ~6bit 才精度无损**”
> （对应 DL 5.0–6.6 bpp）。**主指标修正为下游 zero-shot 平均恢复率**，PPL 仅作参考。

---

## 3. 阶段 8+ 需求分析与规划（2026-10-16）

**需求**：① 在 **多卡** 上完成量化；② 用 **Qwen3-4B** 按 **SLQ 论文方法** 量化；
③ 模型在 **两卡** 部署。

**需求分解与方案**：

1. **多卡量化**（新增）
   - 目标：校准前向跨 ≥2 卡（27B BF16≈54GB，单卡 64GB 勉强；4B 单卡可行，但需先打通多卡路径）。
   - 方案：HF **`device_map` 层间分片**（2 卡流水）跑校准前向；forward hook 把各层输入收集到
     CPU；**逐层 GPTQ 在其所属卡上执行**（单层权重+Hessian 很小，无需 TP）。
   - 交付：`model_quant` / `ppl_eval` 支持 `--device-map`（多卡）；**单卡 vs 双卡量化结果一致性**验证。
2. **SLQ 论文方法对齐**（对当时自研实现的差距）
   - 补齐 **非对称 INT**（零偏）——当前仅对称；DL 必需。
   - 位宽集合 {2..8}、g128、**单 Hessian 全位宽**（已具备）、每组一个 GPTQ block。
   - 校准数据换成 **FineWeb-Edu 512×2048**（已缓存）；KL 取 **top-K=10**。
   - bpp 计费 +0.156；ILP 用**参数量加权**预算 + 二分搜索。
3. **敏感度**：多 bitwidth **Shapley**（Alg.1）+ 线性 α 双路；0.6B/4B 上做 Shapley vs 线性一致性。
4. **两卡部署**：fake-quant BF16 checkpoint（BF16 落在整数格点）→ **vllm-ascend TP=2**
   （沿用阶段 3 路径）serve + 生成验证 + 吞吐记录。
5. **评测口径**：主 = zero-shot 平均恢复率；辅 = EAR/KL（top-10）、WikiText-2 PPL；
   对照 = uniform-4（4.156 bpp）。

**验证计划**：

| 验证项 | 方法 | 通过标准 |
|-------|------|---------|
| 非对称量化正确性 | 单测（对称/非对称网格、零偏） | 数值对齐参考公式 |
| 多位宽一致性 | 同一 Hessian 产出 {2..8} 档 | 位宽↑→误差↓单调；与逐位宽单独求解一致 |
| 多卡正确性 | 单卡 vs 2 卡跑同 seed 量化 | 权重量化结果一致（逐层 rel 相同）；PPL 一致 |
| 敏感度一致性 | Shapley vs 线性 α 相关性 | 排序相关（Spearman）≥0.7 |
| 4B 精度 | DL/TL 配置 + lm-eval | TL 恢复 ≥99%；朝向 DL（≥99.5%）区间 |
| 双卡部署 | vllm-ascend TP=2 serve + 生成 | 文本通顺、TP2 正常、吞吐记录 |

**风险**：① 多卡 device_map 前向在 NPU 上的兼容性；② 非对称 INT 在 NPU GPTQ 列的数值稳定性；
③ lm-eval 在 NPU 上的耗时（可离线批量）；④ 4 卡资源竞争（当时仅少数卡 OK）。

---

## 4. 阶段 5/6 结论（历史记录）

**阶段 5 关键结论**：

1. SAGE 参考实现核心算法正确（rel 达理论值：合成高斯/真实权重 (64,4) rel≈0.31、方向余弦≈0.95）。
2. **纯 fake-quant 下 Qwen3-0.6B 在 2.0–2.75bpw 全部崩溃**（PPL 135–1.2e6）→ 0.6B 过于敏感，
   不能作为 2.25bpw 验收模型。
3. **ESG 在本环境有害**（rel 0.31→0.63），且非校准量不足；主线先 `--no-esg`。
4. **Qwen3-4B（论文验证规模）no-esg 2.25bpw：PPL 13.20→36.66（+178%）** —— 功能可用但明显损失
   → SAGE 2.25bpw 单档不能满足 <0.5%，必须靠混合精度。

**阶段 6（AMCT GPTQ 审查 + 可用性验证）**：

1. GPTQ 实现于 `amct_pytorch/algorithms/quant/gptq.py`，`targets=("weight",)`，Hessian 列式二阶补偿，
   per-channel 对称 int4/int8 + HiFloat4，BlockwiseSolver 驱动，`BitPolicy` 支持逐层位宽。
2. AMCT 仅支持 int4/int8（`_ALLOWED_BITS=(4,8,16)`），且依赖版本（torch 2.7.1 / torch_npu 2.7.1 /
   transformers 5.12.1）与服务器环境（2.10.0 / 5.14.1）存在偏差。
3. 该审查结论促成阶段 7 自研“任意 int、可跨位宽复用 Hessian”的 GPTQ 算子。

**阶段 7 规划（SLQ + GPTQ 混合精度）**：

1. **GPTQ 算子**：移植/复用 AMCT `gptq.py` 核心（Apache-2.0，标注出处）为独立可调用模块，
   扩展支持 `{3,4,6,8}`bit，CPU/NPU 通用，含单测。
2. **SLQ 敏感度**：逐层 Shapley（或线性 α 近似）——每个 (层, 候选位宽) 产出 fake-quant 权重并测 NLL/KL。
3. **ILP 混合精度分配**：给定平均 bit 预算，最小化总敏感度。
4. **SLQ-TL**：单点标定近无损补偿。
5. **评测**：0.6B 做流程验证、**Qwen3-4B 做精度验收**。

> 阶段 7 的自研实现在 4B 上验证了流程与口径，但因线性敏感度信噪比不足，
> 最终被阶段 10 起的官方 SLQ 移植取代（代码见 [../code/SLQ-gptq/](../code/SLQ-gptq/)）。
