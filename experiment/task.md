# Qwen3.8-27B模型量化任务

## 背景

实现Qwen3.8-27B的量化，在精度损失小于0.5%的情况下，实现更低的bit量化。

## 任务描述

1. 基于vllm-ascend实现Qwen3.8-27B多卡（至少两卡）部署，验证npu环境可用；
2. 分析参考论文，设计合适的模型量化方法，要求在精度损失小于0.5%的情况下实现更低的bit量化；
3. 先用Qwen3-0.6B模型进行小规模冒烟验证（单卡，多卡），验证通过后再进行Qwen3.8-27B全量实验。

## 量化方法设计论文参考

论文实现参考：
   - SLQ：https://arxiv.org/pdf/2605.02404，官方代码实现：https://github.com/IST-DASLab/SLQ
   - GPTQ（NPU 实现参考）：`c:\code\amct`（昇腾 AMCT / Ascend Model Compression Toolkit，Apache-2.0），核心见 `amct_pytorch/algorithms/quant/gptq.py`
   - SAGE（暂缓，低位档算子备选）：https://arxiv.org/pdf/2610.14745，官方代码实现：https://github.com/Zincwolf/sage
   - PCDVQ（已弃用，仅作历史参考）：https://arxiv.org/pdf/2506.05432，非官方参考实现：https://github.com/danlee65071/pcdvq/blob/main/notebooks/pcdvq.ipynb

> **主线调整（2026-10-16）**：阶段 5 证实纯 fake-quant 的 VQ（PCDVQ/SAGE）在低位档崩溃、缺少误差补偿。故后续主线改为 **SLQ + GPTQ**：以 **GPTQ（Hessian 误差补偿）** 作为逐层位宽量化算子，**SLQ（Shapley 敏感度 + ILP）** 做混合精度分配。**SAGE 暂缓**，待 SLQ+GPTQ 出结果后再评估。

### 方案修订说明（2026-10-16）

原方案为"PCDVQ（低位档）+ GPTQ-INT（高位档）+ SLQ（Shapley/ILP 分配）"。阶段 4 实测发现 **PCDVQ 纯 fake-quant 无误差补偿时精度灾难性崩溃**（Qwen3-0.6B WikiText-2 PPL 20.04 → 360702），低位档不可用。

经调研（见 [SAGE-feasibility.md](SAGE-feasibility.md)），**改用 SAGE 作为低位档算子**。SAGE 与 PCDVQ 同属"球坐标解耦 VQ"谱系，但：
- **内建 blockwise 误差补偿**（20 epoch STE，学生输入用前序量化块），直接对应 PCDVQ 崩溃根因；
- 激活感知（ESG 能量分层分组 + 加权球面 k-means）、逐矩阵 ABS 自适应位切分；
- 可学习码本，规避 E8 枚举/单位化/位算术缺陷；官方实现且已在 hybrid（Qwen3.5）/MoE/VLM 上验证。

**结论**：
1. SAGE 可替换 PCDVQ 作为低位档，且更优；但 SAGE 本质是 **≤3bit 极低位方法**（工作点约 2.0–2.5 bpw），**不能充当高位档**；
2. 与 SLQ 接口正交、可组合（SAGE 官方导出即密集 QDQ 权重，逐矩阵量化比 SLQ 组更细，敏感度黑盒兼容）；
3. 组合方案 = **低位档 SAGE{2.0, 2.125, 2.25} + 高位档 GPTQ-INT{3,4,6,8} + SLQ Shapley/ILP 分配 + SLQ-TL 单点标定**。

> PCDVQ-k3 实现（[PCDVQ-k3/](PCDVQ-k3/)）保留为代码资产，不再作为主线低位档算子。

## 阶段规划与交付件

| 阶段 | 内容 | 交付件 | 状态 |
|------|------|--------|------|
| 0 | 任务规划 | progress.md | ✅ 完成 |
| 1 | 调研：SLQ / SAGE（替代 PCDVQ）论文与代码 | [feasibility.md](feasibility.md) + [SAGE-feasibility.md](SAGE-feasibility.md) | ✅ 完成 |
| 2 | NPU 环境验证：Qwen3-0.6B 单卡+多卡冒烟部署 | 冒烟日志与环境结论（记录于 progress.md） | ✅ 完成 |
| 3 | Qwen3.8-27B 多卡部署验证（≥2卡，vllm-ascend） | 部署命令、推理验证结果 | ✅ 完成 |
| 4 | 低位档算子实现：SAGE 参考实现（替换 PCDVQ） | [SAGE-k3/](SAGE-k3/) 代码 + CPU 单测（13 passed） | ✅ 完成 |
| 5 | SAGE NPU 冒烟（0.6B/1.7B/4B）+ 根因分析（ESG/blockwise/位宽消融） | PPL 曲线与结论（progress.md 阶段 5） | ✅ 完成 |
| 6 | AMCT GPTQ 代码审查 + 可用性验证（单元测试 + 数值对比） | 审查结论 + 单测 19 passed + 误差补偿验证 | ✅ 完成 |
| 7 | GPTQ 算子 {2..8}bit + SLQ 敏感度/ILP 混合精度分配（0.6B 验证） | [SLQ-gptq/](SLQ-gptq/) 代码 + PPL 结果 | 🔶 首轮完成（见 progress.md） |
| 8 | **SLQ 论文方法对齐**：非对称 INT g128 + 单位宽集 {2..8} + 校准语料 | 对齐后的 GPTQ 权重量库 + 单测 | 🔶 非对称/位宽已对齐；FineWeb 校准待补（暂用 WikiText） |
| 9 | **多卡量化框架**：device_map 分片校准前向 + 逐层 GPTQ（4B，2 卡打通） | 多卡量化脚本 + 单卡/双卡一致性验证 | ✅ 完成（4B：126 层卡0 / 30 层卡1） |
| 10 | **官方 SLQ 仓获取 + NPU 移植**：clone IST-DASLab/SLQ；nccl→hccl、cuda→npu、Cholesky 走 CPU、离线 env | `/opt/zh/slq-official` + `patch_npu*.py` | ✅ 完成 |
| 11 | **论文校准数据**：FineWeb-Edu（hf-mirror parquet 2.1G）→ 官方 `.pt` | `calib_qwen3_8m_8192.pt`（1024×8192）、`calib_qwen3_1m_2048.pt`（512×2048） | ✅ 完成 |
| 12 | **官方 GPTQ 全位宽 {2..8}（0.6B 验证）**：cal=4、g128、非对称 | 196 层 × 7 档位 `.pth`（8.6GB，2244s） | ✅ 完成 |
| 13 | **官方敏感度数据库**：线性 α（Shapley 需 33,900 次全量前向，本机降采样） | `linear_*.json` + `error_db.json` + `baseline_*.pt` | ✅ 完成（112 组，17:51） |
| 14 | **官方 DL 搜索（EAR≥0.99）** → 平均位宽 <4bit 配置 | `config_dl.txt` | ✅ 完成（0.6B 即使 8bit EAR 仅 0.9810，DL 无解；验收转 27B） |
| 15 | **官方 checkpoint 拼装 + 评测**：build_checkpoint（fake）→ EAR/KL/PPL/zero-shot | 0.6B 量化前后对比 | ✅ 完成（流程打通；0.6B 不适合作验收模型） |
| 16 | **Qwen3.8-27B 全流程**（64 层 hybrid，linear_attn + full_attn，目标 <4bit）+ 双卡部署 | 27B 量化前后对比 | ✅ **3.600 bpp / 恢复 100.0%**（保守 3.800 bpp / 101.6%）；TP=2 部署 PASS |

## SLQ 论文实验结果（2026-10-16 补充调研）

官方 [IST-DASLab/SLQ](https://github.com/IST-DASLab/SLQ)（COLM 2026）两种"无损"定义与实测区间：

| 目标 | 定义 | 达成 bpp（论文） |
|------|------|-----------------|
| **Task-Lossless (TL)** | 7 个 zero-shot 基准平均恢复 **≥99%** BF16（在采样方差内） | **3.3–4.7 bpp** |
| **Distribution-Lossless (DL)** | 下一 token 分布与 BF16 不可分：**EAR ≥ 0.99** | **5.0–6.6 bpp** |

- 论文实证：**zero-shot 准确率在 KL≲0.01 区间保持平坦**，超出后骤降 → **PPL/KL 不等于任务精度**，主指标应看下游恢复率。
- **方差律**：对称量化噪声方差比非对称放大约 γ² → **DL 必须非对称**（TL 不必须）。
- 官方流水线：① 逐层 GPTQ，**每个位宽 ∈{2..8}**，**非对称 INT、g128、每组一个 GPTQ block、单 Hessian 一次产出全部位宽**，`cal_bits=4` 序贯传播；② 多 bitwidth Shapley 敏感度（512×2048 FineWeb-Edu，top-K=10）；③ ILP 多选背包 + 二分搜索最小 bpp；④ 拼装 HF checkpoint（fake / humming_online）。
- **bpp 计费**：g128 有 **+0.156** 开销（2→2.156 … 8→8.156）。
- 评测：`run_lmeval`（zero-shot×3 seeds）、`run_ear`、`run_kl`、`run_ppl`。

> **与本任务的关系**：任务要求"损失 <0.5%" → 恢复率 **≥99.5%**，比论文 TL（99%）更严、指向 DL 的 bpp 区间；这解释了用户结论"**0.6B 需 ~6bit 才精度无损**"（对应 DL 5.0–6.6 bpp）。**主指标修正为下游 zero-shot 平均恢复率**，PPL 仅作参考。

## 阶段 8+ 需求分析与规划（2026-10-16）

**需求**：① 在 **多卡** 上完成量化；② 用 **Qwen3-4B** 按 **SLQ 论文方法** 量化；③ 模型在 **两卡** 部署。

**需求分解与方案**：

1. **多卡量化**（新增）
   - 目标：校准前向跨 ≥2 卡（27B BF16≈54GB，单卡 64GB 勉强；4B 单卡可行，但需先打通多卡路径）。
   - 方案：HF **`device_map` 层间分片**（2 卡流水）跑校准前向；forward hook 把各层输入收集到 CPU；**逐层 GPTQ 在其所属卡上执行**（单层权重+Hessian 很小，无需 TP）。
   - 交付：`model_quant`/`ppl_eval` 支持 `--device-map`（多卡）；**单卡 vs 双卡量化结果一致性**验证。
2. **SLQ 论文方法对齐**（对现有 [SLQ-gptq/](SLQ-gptq/) 的差距）
   - 补齐 **非对称 INT**（零偏）——当前仅对称；DL 必需。
   - 位宽集合 {2..8}、g128、**单 Hessian 全位宽**（已具备）、每组一个 GPTQ block。
   - 校准数据换成 **FineWeb-Edu 512×2048**（已缓存）；KL 取 **top-K=10**。
   - bpp 计费 +0.156；ILP 用**参数量加权**预算 + 二分搜索。
3. **敏感度**：多 bitwidth **Shapley**（Alg.1）+ 线性 α 双路；0.6B/4B 上做 Shapley vs 线性一致性。
4. **两卡部署**：fake-quant BF16 checkpoint（BF16 落在整数格点）→ **vllm-ascend TP=2**（沿用阶段 3 路径）serve + 生成验证 + 吞吐记录。
5. **评测口径**：主 = zero-shot 平均恢复率（arc_easy/arc_challenge/hellaswag/piqa/winogrande/openbookqa，`lm_eval` 已装、数据集已缓存）；辅 = EAR/KL（top-10）、WikiText-2 PPL；对照 = uniform-4（4.156 bpp）。

**验证计划**：

| 验证项 | 方法 | 通过标准 |
|-------|------|---------|
| 非对称量化正确性 | 单测（对称/非对称网格、零偏） | 数值对齐参考公式 |
| 多位宽一致性 | 同一 Hessian 产出 {2..8} 档 | 位宽↑→误差↓单调；与逐位宽单独求解一致 |
| 多卡正确性 | 单卡 vs 2 卡跑同 seed 量化 | 权重量化结果一致（逐层 rel 相同）；PPL 一致 |
| 敏感度一致性 | Shapley vs 线性 α 相关性 | 排序相关（Spearman）≥0.7 |
| 4B 精度 | DL/TL 配置 + lm-eval | TL 恢复 ≥99%；朝向 DL（≥99.5%）区间 |
| 双卡部署 | vllm-ascend TP=2 serve + 生成 | 文本通顺、TP2 正常、吞吐记录 |

**风险**：① 多卡 device_map 前向在 NPU 上的兼容性；② 非对称 INT 在 NPU GPTQ 列的数值稳定性；③ lm-eval 在 NPU 上的耗时（可离线批量）；④ 4 卡资源竞争（当前仅少数卡 OK）。

## 阶段 5/6 结论（历史记录）

**阶段 5 关键结论**（详见 [progress.md](progress.md)）：

1. SAGE 参考实现核心算法正确（rel 达理论值：合成高斯/真实权重 (64,4) rel≈0.31、方向余弦≈0.95）。
2. **纯 fake-quant 下 Qwen3-0.6B 在 2.0–2.75bpw 全部崩溃**（PPL 135–1.2e6）→ 0.6B 过于敏感，不能作为 2.25bpw 验收模型。
3. **ESG 在本环境有害**（rel 0.31→0.63），且非校准量不足；主线先 `--no-esg`。
4. **Qwen3-4B（论文验证规模）no-esg 2.25bpw：PPL 13.20→36.66（+178%）** —— 功能可用但明显损失 → SAGE 2.25bpw 单档不能满足 <0.5%，必须靠混合精度。

**阶段 6 详细任务（AMCT GPTQ 审查 + 可用性验证）**：

1. **代码审查**：已完成（见 [progress.md](progress.md) 2026-10-16 最新条目）——GPTQ 实现于 `amct_pytorch/algorithms/quant/gptq.py`，`targets=("weight",)`，Hessian 列式二阶补偿，per-channel 对称 int4/int8 + HiFloat4，BlockwiseSolver 驱动，`BitPolicy` 支持逐层位宽。
2. **可用性验证（受阻，待终端恢复）**：
   - 暴露 `amct_pytorch`（纯 Python 包，`PYTHONPATH` 或 pip）
   - `pytest tests/unit_test/algorithms/test_gptq.py`（CPU）
   - Qwen3-0.6B 冒烟：`extract_ptq_data → ptq --algos gptq → eval`，核对 PPL
3. **注意**：AMCT 仅支持 int4/int8（`_ALLOWED_BITS=(4,8,16)`），且依赖版本（torch 2.7.1/torch_npu 2.7.1/transformers 5.12.1）与服务器环境（2.10.0/5.14.1）存在偏差。

**阶段 7 详细任务（SLQ + GPTQ 混合精度）**：

1. **GPTQ 算子**：移植/复用 AMCT `gptq.py` 核心（Apache-2.0，标注出处）为独立可调用模块（`experiment/` 下），扩展支持 `{3,4,6,8}`bit，CPU/NPU 通用，含单测。
2. **SLQ 敏感度**：逐层 Shapley（或线性 α 近似）——每个 (层, 候选位宽) 产出 fake-quant 权重并测 NLL/KL。
3. **ILP 混合精度分配**：给定平均 bit 预算，最小化总敏感度。
4. **SLQ-TL**：单点标定近无损补偿。
5. **评测**：0.6B 做流程验证、**Qwen3-4B 做精度验收**（目标 <0.5% 损失，平均 bit 更低）。

> **SAGE 定位**：暂缓，作为低位档备选；待 SLQ+GPTQ 出结果后决定是否重启（含 ESG 修复与自采样校准）。

## npu服务器环境信息

- npu环境说明：env\env.md
- docker启动命令说明：env\docker.md

## 模型

Qwen3.8-27B模型：https://huggingface.co/Qwen/Qwen3.8-27B
vllm-ascend部署Qwen3.8-27B模型说明：https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/Qwen3.8-27B.html

## 其他说明

1. 先进行任务规划，将任务分解成多个阶段，每个阶段要有可验收的交付件；
2. 及时任务进展更新到experiment\progress.md；
3. 使用https://hf-mirror.com/下载必要的模型。npu环境上已有一些模型文件，你可以尝试复用；
4. npu环境上已有一些量化工作，不建议参考。你可以复用npu环境上的模型文件，但是当前没有Qwen3.8-27B，只有Qwen3.5-27B（不要混淆！！！）；
5. 如果存在多次尝试无法解决的任务，及时向用户报告，用户会帮你解决。
