# Qwen3.8-27B 量化任务进展

> 逐日实验日志（2026-10-16 → 10-19）已冻结到
> [archive/logs/progress-2026-10-16_10-19.md](archive/logs/progress-2026-10-16_10-19.md)。
> 本文件只维护**当前状态**与最近的结构性变更。

## 当前状态

- **阶段 0–16 全部完成**：3.600 bpp、6 任务平均恢复率 100.0%（500 样本）、vllm-ascend TP=2 部署 PASS。
- 现行实现：[slq-official/](slq-official/)（官方 SLQ + NPU 适配）。
- 权威数字与结论：[summary-report.md](summary-report.md)；复现步骤：[reproduction-guide.md](reproduction-guide.md)。
- 位宽与存储推算：[quantization-analysis.md](quantization-analysis.md)。

## 2026-11-18 代码整理（本次变更）

解决的问题：实验目录经过 5 轮以上方案迭代，现行实现只占约 1/4 代码量，脚本层有
15 份逐字重复的环境样板、28 个脚本里 5 个纯残留、9 个补丁脚本互相耦合且其中一个
重跑会写坏文件。

| 变更 | 内容 |
|------|------|
| 目录归档 | PCDVQ-ds / PCDVQ-k3 / SAGE-k3 / RoleQ-k3 / SLQ-gptq → `archive/code/`；过期文档与逐日日志 → `archive/docs/`、`archive/logs/` |
| 脚本合并 | 28 → 15：`run_slq27b_v2`→`run_slq27b`；`run_db27b_base`+`run_shapley27b`→`run_db27b.sh [base|shapley]`；`run_b34`+`run_eval_budgets`→`run_build_eval.sh <limit> <budget>...`；`run_zs27b`+`run_zs500`+`run_zs500_b38`→`run_zs.sh <limit> <spec>...`；`run_tl6`→`run_tl06b.sh <cal_bits> <recovery>`；`mk_calib27`→`run_calib27b` |
| 公共环境 | 新增 `scripts/common.sh`：路径/快照/任务列表/离线与 conda 环境集中一处，全部可用环境变量覆盖 |
| 补丁合并 | 9 个 `patch_npu*.py` → 单个 `patches/npu-adaptation.patch`（幂等，apply 后与覆盖文件逐字节一致） |
| 死代码 | 删 `args_slq.sh`、`dl_fineweb2.py`、`inspect_qparams.py`、`inspect_shapley.py`、`run_search06b.sh`；覆盖文件删未用 import、ortools 死依赖、补丁留下的悬空括号 |
| Bug 修复 | `run_deploy27b.sh` 调用了不存在的 `/work/scripts/smoke_llm.py`（部署必然失败）→ 改为挂载本目录 `smoke_llm.py`；断点续跑分支漏传 `bias`；多卡 resume 分支不一致可能挂死；评测脚本不再依赖 `SLQ-gptq/model_quant.py` |

验证：归档后的 4 套算子单测仍全部通过（9 + 27 + 13 + 8 = 57）；27B 四档配置的层数/位宽/KL
单调性校验通过；补丁在官方 `eb9f1f2` 上 apply 后与 7 个覆盖文件 hash 一致；文档相对链接全通。

## 2026-11-18/19 重新量化 + HTTP 部署验证

服务器上原有的层库与 checkpoint 已被清理，本次按 SLQ 方法**从零重跑**并完成上线验证
（完整记录见 [slq-official/DEPLOYMENT-VALIDATION.md](slq-official/DEPLOYMENT-VALIDATION.md)）。

| 项 | 结果 |
|----|------|
| 量化 | 496/496 层，层库 275 GB；中途一次 NPU `vector core timeout`，自愈管道换卡续跑 |
| checkpoint | b3.6，51 GB（fake-quant，2 分片）+ VLM 预处理文件 |
| 指纹校验 | 每 128 组取值数 4/16/54/101，上限 4/16/64/256 → PASS |
| 服务 | vLLM-Ascend `v0.30.0`，TP=2（卡 0、2），端口 8098，`/health` 200 |
| HTTP | `/v1/models`、chat、completions、流式 SSE 全部可用；预热后 128 tokens ≈ 4.3 s |
| 新增脚本 | `deploy/{pipeline_27b,finalize_steps,serve_27b,serve_b36,http_validate}.sh`、`scripts/{build_ckpt,verify_fakequant}.sh` |

偏差：本次校准用 65536 tokens（文档口径 262144）以缩短重跑时间；权重仍是 fake-quant，
显存/存储收益需 INT 打包后才会体现。

## 下一步（可选）

1. fake-quant → 真 INT 打包（官方 `humming_online` 或自研加载），把 3.17× 存储收益与
   每卡 8.1 GiB 显存收益落地；
2. 补 1000 样本复核 openbookqa（b3.6 −4.6pp，约 2σ）；
3. Shapley 采样从 8k tokens 提到 32k，尝试把无损下界压到 3.4–3.5 bpp；
4. 若仍需复现论文 TL 流程，补 3-bit GPTQ 档位作为近无损锚点。
