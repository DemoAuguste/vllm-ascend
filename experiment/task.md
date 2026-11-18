# Qwen3.8-27B 模型量化任务

## 背景

实现 Qwen3.8-27B 的量化：在精度损失小于 0.5% 的前提下，实现更低的 bit 量化。

## 任务描述

1. 基于 vllm-ascend 实现 Qwen3.8-27B 多卡（至少两卡）部署，验证 NPU 环境可用；
2. 分析参考论文，设计合适的模型量化方法，要求在精度损失小于 0.5% 的情况下实现更低的 bit 量化；
3. 先用 Qwen3-0.6B 模型进行小规模冒烟验证（单卡、多卡），验证通过后再进行 Qwen3.8-27B 全量实验。

## 验收口径

| 项 | 标准 | 实测（b3.6） |
|----|------|--------------|
| 平均位宽 | < 4 bit | **3.600 bpp**（参数加权，含 g128 开销） |
| 精度损失 | < 0.5% | **< 0.01%**（6 任务 × 500 样本，恢复率 100.0%） |
| 多卡部署 | ≥ 2 卡 | TP=2 PASS（vllm-ascend，卡 5,7） |

主指标是 6 个 zero-shot 任务（arc_easy / arc_challenge / piqa / hellaswag / winogrande /
openbookqa）的平均恢复率，PPL/KL 只作辅指标；500 样本的均值标准误约 0.6%。
详细判定依据见 [summary-report.md](summary-report.md#12-验收口径重要)。

## 阶段与交付状态

| 阶段 | 内容 | 交付件 | 状态 |
|------|------|--------|------|
| 0–2 | 任务规划、论文调研、0.6B 单卡/多卡 NPU 冒烟 | 历史记录（见归档日志） | ✅ |
| 3 | Qwen3.8-27B 多卡部署验证 | 部署命令与推理结果 | ✅ |
| 4–6 | 低位档算子探索（PCDVQ / SAGE）与 AMCT GPTQ 审查 | [archive/code/](archive/code/) | ✅ 结论：主线路弃 |
| 7–9 | 自研 GPTQ + SLQ 混合精度（0.6B / 4B / 多卡） | [archive/code/SLQ-gptq/](archive/code/SLQ-gptq/) | ✅ 被官方移植取代 |
| 10–13 | 官方 SLQ 仓 NPU 移植、校准数据、全位宽层库、敏感度库 | [slq-official/](slq-official/) | ✅ |
| 14–15 | DL 搜索（0.6B 无解）与 checkpoint 拼装/评测流程 | [archive/slq-official-legacy/](archive/slq-official-legacy/) | ✅ 流程验证 |
| 16 | **27B 全流程 + 双卡部署** | [summary-report.md](summary-report.md) | ✅ **3.600 bpp / 恢复 100.0%**，TP=2 PASS |

## 代码位置

| 路径 | 内容 | 状态 |
|------|------|------|
| [slq-official/](slq-official/) | **现行实现**：官方 IST-DASLab/SLQ 的 NPU 适配 + 27B 配置与流水线脚本 | 维护中 |
| [reproduction-guide.md](reproduction-guide.md) | 从零复现手册（环境 → 数据 → 量化 → 评测 → 部署） | 维护中 |
| [summary-report.md](summary-report.md) | 最终验收报告与全部结论数字 | 维护中 |
| [quantization-analysis.md](quantization-analysis.md) | 位宽分布 / 存储 / 部署收益专项分析 | 维护中 |
| [progress.md](progress.md) | 当前阶段状态（历史日志已冻结） | 维护中 |
| [archive/](archive/) | 历史方案代码、过期分析文档、逐日进展日志 | 只读归档 |

## 参考论文

- **SLQ**（主方案）：https://arxiv.org/pdf/2605.02404 ，官方实现 https://github.com/IST-DASLab/SLQ
- **GPTQ**（NPU 实现参考）：昇腾 AMCT，`amct_pytorch/algorithms/quant/gptq.py`
- **SAGE**（已暂缓，低位档备选）：https://arxiv.org/pdf/2610.14745
- **PCDVQ**（已弃用，仅作历史参考）：https://arxiv.org/pdf/2506.05432

方案从 PCDVQ → SAGE → 自研 SLQ+GPTQ → 官方 SLQ 移植的演进过程、各阶段结论与
决策依据，见归档文档 [archive/docs/design-history.md](archive/docs/design-history.md)。

## 环境与模型

- NPU 服务器环境说明：`env/env.md`，docker 启动说明：`env/docker.md`
- Qwen3.8-27B：https://huggingface.co/Qwen/Qwen3.8-27B
- vllm-ascend 部署 Qwen3.8-27B：https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/Qwen3.8-27B.html
- 复现所需镜像与软件版本见 [reproduction-guide.md §1](reproduction-guide.md#1-环境准备)

## 其他说明

1. 先进行任务规划，将任务分解成多个阶段，每个阶段要有可验收的交付件；
2. 及时把任务进展更新到 [progress.md](progress.md)；
3. 使用 https://hf-mirror.com/ 下载必要的模型；NPU 环境上已有一些模型文件，可以复用；
4. NPU 环境上已有的量化工作不建议参考；环境里只有 Qwen3.5-27B，没有 Qwen3.8-27B（不要混淆）；
5. 如果存在多次尝试无法解决的任务，及时向用户报告。
