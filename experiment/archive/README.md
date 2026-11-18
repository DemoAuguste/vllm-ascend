# 归档区（只读）

这里保存的是**已经不在主线上**的方案代码、一次性脚本、过期分析文档和逐日日志。
保留它们是为了可追溯与代码资产复用，**不再维护**；现行实现见 [../slq-official/](../slq-official/)，
结论见 [../summary-report.md](../summary-report.md)。

## 目录

| 路径 | 内容 | 归档原因 | 还能怎么用 |
|------|------|----------|-----------|
| [code/PCDVQ-ds/](code/PCDVQ-ds/) | 单文件 PCDVQ 最小 demo（335 行，6 段验证） | 被 `PCDVQ-k3` 完全取代（后者多了 A–F 消融与单测），全仓零引用 | 只作阅读 |
| [code/PCDVQ-k3/](code/PCDVQ-k3/) | PCDVQ（极坐标解耦 VQ）参考实现 + 全模型脚本 + 9 项单测 | 纯 fake-quant 无误差补偿，0.6B 在 2–4 bit 崩溃（PPL 20→360702），主线弃用；按当时决定**保留为代码资产** | `python -m pytest tests -q`（需 scipy） |
| [code/SAGE-k3/](code/SAGE-k3/) | SAGE（blockwise 误差补偿 VQ）参考实现 + 13 项单测 | 2.25 bpw 下 4B 仍有 +178% PPL，不能单独满足 <0.5% 损失；主线暂缓 | `python -m pytest tests -q` |
| [code/RoleQ-k3/](code/RoleQ-k3/) | RoleQ（ternary 角色对齐 scale）参考实现 + 27 项单测 + bit 扩展分析 | 独立论文研究，收益随 bit 增大衰减（3-bit 仅 0.2%–0.3%），从未接入流水线 | `python -m pytest tests -q` |
| [code/SLQ-gptq/](code/SLQ-gptq/) | 自研 GPTQ 算子 + 线性敏感度 + ILP（阶段 7–9，4B 上打通） | 阶段 10 起改用官方 SLQ 移植；线性敏感度信噪比不足 | `python -m pytest tests -q`；`zeroshot_eval.py` 已移入现行目录 |
| [slq-official-legacy/](slq-official-legacy/) | 官方流水线的一次性脚本与产物：TL 搜索脚本、TL 配置（4.509 bpp）、层库统计、耗时基准 | TL 搜索结论为“不采用”（锚点外推过保守）；统计脚本是一次性分析、结果已进报告 | 复核报告数字时参考 |
| [docs/](docs/) | 方案可行性论证、SAGE 可行性、方案演进史、对外提案稿 | 决策已完成，内容与最终方案不一致 | 追溯决策依据 |
| [logs/](logs/) | 2026-10-16 → 10-19 逐日进展日志（冻结原文） | 时序记录，不再更新 | 追溯实验过程 |

## 注意

- `code/SLQ-gptq/` 里的 `model_quant.py`、`slq_allocate.py` 等是 4B 阶段产物：层名前缀按
  `model.layers.N` 硬编码、含 `.cuda()` 硬编码与 `--trust-remote` 之类的无效开关，
  **不能直接用于 27B（VLM）**；27B 路径请用 `slq-official/` 的官方流水线。
- 各目录的 README 保留了当时的写法，其中指向 `progress.md` / `feasibility.md` 的链接
  已改写为本归档区内的实际位置。
- `docs/llm-extreme-quantization.md` 是当时的对外提案稿（NQ-PG），数字全部派生自
  [../summary-report.md](../summary-report.md)，其中把 b3.6 的 fake-quant 每卡权重
  25.61 GiB 写成“BF16 基线”并不准确，引用时以 summary-report 为准。
