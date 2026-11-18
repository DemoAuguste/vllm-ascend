# Qwen3.8-27B 混合精度量化（实验区）

用 SLQ 官方流水线（IST-DASLab/SLQ）+ GPTQ 逐层误差补偿，在 **Qwen3.8-27B** 上做到
**3.600 bpp** 且 6 任务 zero-shot 平均恢复率 **100.0%**，并以 vllm-ascend **TP=2** 完成部署。

| 指标 | BF16 | b3.6（本方案） |
|------|------|----------------|
| 平均位宽 | 16 bpp | **3.600 bpp**（参数加权，含 g128 开销） |
| 模型存储（INT 打包） | 54.71 GB | **17.26 GB（3.17×）** |
| 下游精度（6 任务 × 500） | 0.6237 | **0.6237（100.0%）** |
| 双卡部署 | — | TP=2 PASS，KV 23.99 GiB / 326K tokens |

## 入口

| 我想… | 看这里 |
|-------|--------|
| 从零复现（环境 → 数据 → 量化 → 评测 → 部署） | [reproduction-guide.md](reproduction-guide.md) |
| 看最终结论、全部数字与工程坑清单 | [summary-report.md](summary-report.md) |
| 看位宽分布 / 存储 / 显存收益的推算 | [quantization-analysis.md](quantization-analysis.md) |
| 看现行代码 | [slq-official/](slq-official/) |
| 看任务定义与验收口径 | [task.md](task.md) |
| 看当前进度 | [progress.md](progress.md) |
| 找历史方案（PCDVQ / SAGE / RoleQ / 自研 SLQ-GPTQ） | [archive/](archive/) |

## 目录结构

```
experiment/
├── README.md                  # 本文件
├── task.md                    # 任务定义、验收口径、阶段状态
├── summary-report.md          # 最终验收报告（权威数字）
├── reproduction-guide.md      # 复现手册（唯一操作文档）
├── quantization-analysis.md   # 位宽/存储专项分析
├── progress.md                # 当前进度（历史日志已归档）
├── slq-official/              # 现行实现
│   ├── README.md              #   ← 代码怎么装、每个脚本干什么
│   ├── slq_quantize.py        #   ① 多次 GPTQ（一次 Hessian 出 {2,4,6,8}）
│   ├── slq_build_database.py  #   ② Shapley / 线性敏感度库
│   ├── slq_search_dl.py       #   ③ ILP 位宽分配（TL / DL 搜索）
│   ├── src/                   #   官方仓的 NPU 适配覆盖文件（4 个）
│   ├── eval/zeroshot_eval.py  #   评测：6 任务 log-likelihood
│   ├── configs/               #   4 个预算的位宽配置（b3.4/3.6/3.8/4.0）
│   ├── scripts/               #   流水线入口（common.sh + run_*.sh）
│   └── patches/               #   单个 NPU 适配 diff（对官方 eb9f1f2）
└── archive/                   # 只读归档：历史方案、过期文档、逐日日志
```

## 能跑的验证（CPU，无需 NPU）

历史方案的算子单测随代码一起归档，仍可运行：

```bash
cd experiment/archive/code/PCDVQ-k3  && python -m pytest tests -q   # 9 passed（需 scipy）
cd experiment/archive/code/SAGE-k3   && python -m pytest tests -q   # 13 passed
cd experiment/archive/code/RoleQ-k3  && python -m pytest tests -q   # 27 passed
cd experiment/archive/code/SLQ-gptq  && python -m pytest tests -q   # 8 passed
```

现行 SLQ 流水线依赖 NPU 与 275 GiB 层库，只能按 [reproduction-guide.md](reproduction-guide.md)
在 Ascend 910B 上执行；其中 `configs/` 的位宽配置可在 CPU 上做一致性校验。
