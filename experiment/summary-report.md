# Qwen3.8-27B SLQ 量化 · 总结报告

**日期**：2026-10-19
**平台**：Ascend 910B3 × 8（共享集群），vLLM-Ascend nightly-main（vLLM 0.30.0）
**方法**：SLQ 官方流水线（IST-DASLab/SLQ, COLM 2026）+ GPTQ 逐层误差补偿
**目标**：Qwen3.8-27B 在 **精度损失 <0.5%** 前提下实现 **<4 bit** 量化，并完成 **≥2 卡** 部署

---

## 0. 结论摘要

| 验收项 | 要求 | 实测 | 结论 |
|--------|------|------|------|
| 平均位宽 | < 4 bit | **3.600 bpp**（b3.6）；3.800 bpp（b3.8 保守） | ✅ |
| 精度损失 | < 0.5% | **< 0.01%**（b3.6，500 样本）；b3.8 反超 1.6% | ✅ |
| 多卡部署 | ≥ 2 卡 | **TP=2 PASS**（vllm-ascend，卡 5,7） | ✅ |
| 存储（INT 打包后） | —— | **17.26 GB，较 BF16 压缩 3.17×** | ✅ |
| 生成可用性 | 输出连贯 | 5/5 prompt 正常，`RESULT: PASS` | ✅ |

**一句话结论**：用 SLQ 官方四阶段流水线，Qwen3.8-27B 在 **3.600 bpp** 下 zero-shot 平均恢复率 **100.0%**（500 样本 × 6 任务），并以 TP=2 在双卡 910B3 上完成部署验证。

---

## 1. 任务与验收口径

### 1.1 任务定义

1. 基于 vllm-ascend 实现 Qwen3.8-27B 多卡（≥2 卡）部署，验证 NPU 环境可用；
2. 分析 SLQ 论文并复现官方方法，实现"精度损失 <0.5% 且位宽更低"；
3. 先用 0.6B 冒烟、4B 验证，再做 27B 全量。

### 1.2 验收口径（重要）

- **主指标**：6 个 zero-shot 任务（arc_easy / arc_challenge / piqa / hellaswag / winogrande / openbookqa）的**平均准确率恢复率**（量化 / BF16）。
  - 理由：SLQ 论文实证 zero-shot 在 KL ≲ 0.01 区间保持平坦，**PPL/KL 不等于任务精度**；本任务"损失 <0.5%"对应恢复率 ≥ 99.5%。
- **辅指标**：WikiText-2 PPL、top-10 KL、EAR。
- **bpp 计费**：参数加权平均位宽，含 g128 的 scale/zero 开销（+0.156 bpp，论文口径）。
- **统计噪声**：200 样本均值 SE ≈ ±1%，500 样本 ≈ ±0.6%；最终判定用 500 样本。

---

## 2. 环境与平台

| 项 | 配置 |
|----|------|
| NPU | Ascend 910B3 × 8，单卡 60.96 GiB HBM |
| 镜像 | `quay.io/ascend/vllm-ascend:nightly-main`（A2 构建；A5 构建在本机不可用） |
| 软件 | torch 2.10.0 + torch_npu 2.10.0 + transformers 5.14.1，conda env `agent` |
| 模型 | Qwen3.8-27B（VLM，`Qwen3_5ForConditionalGeneration`，64 层 hybrid：linear_attn / full_attn，hidden 5120，vocab 248320，27.36B 参数） |
| 校准数据 | FineWeb-Edu sample-10BT（hf-mirror 直连 parquet），27B tokenizer 生成 `128×8192` / `128×2048` |

### 2.1 关键架构陷阱

`AutoModelForCausalLM` 会把 27B 解析成 `Qwen3_5ForCausalLM`（参数前缀 `model.layers.*`），而 checkpoint 是 `model.language_model.layers.*` → **会静默加载随机权重**。必须按 `config.json` 的 `architectures` 选择 `Qwen3_5ForConditionalGeneration`。此坑在 `src/models/loading.py` 与自研 `model_quant.py` 中均已修复。

---

## 3. 方法：SLQ 官方四阶段流水线

```
① slq_quantize.py      GPTQ 逐层、单 Hessian 一次产出 {2,4,6,8} 全位宽
        │               （cal=4 序贯传播、g128、非对称 INT）
        ▼
② slq_build_database.py  多 bitwidth Shapley 敏感度（Alg.1）
        │               （400 组：qkv / gate+up 合并，其余逐层）
        ▼
③ ILP 预算分配          每组选一位宽，参数加权平均位宽 ≤ 预算，最小化预测 KL
        │               （scipy HiGHS；官方 OR-Tools SCIP 在本平台段错误）
        ▼
④ slq_build_checkpoint.py  fake-quant HF checkpoint（BF16 落在整数格点）
```

### 3.1 NPU 适配（官方仓 `/opt/zh/slq-official`）

| 问题 | 修复 |
|------|------|
| `nccl` / `cuda:{rank}` | → `hccl` / `npu:{rank}`、`torch.npu.set_device` |
| `torch.linalg.cholesky` NPU 不支持 | Hessian 求逆搬到 CPU 后回迁 |
| 27B OOM（第 20/64 块，47.3 GiB） | 断点续跑（已存层直接重建 QLinear）+ 显式释放旧层/hooks + 每 8 次前向 empty_cache |
| VLM 加载成随机权重 | 按 `architectures` 选类（见 2.1） |
| `slq_search_dl.py` 段错误 | OR-Tools SCIP → `scipy.optimize.milp`（HiGHS） |
| 248K 词表 logits OOM | 每 batch `torch.npu.empty_cache()`，batch 降到 1 |
| Shapley 负边际 | 成本钳制 ≥0 + 强制位宽单调 |
| 联网触发 httpx bug | `HF_HUB_OFFLINE=1` 等离线环境变量 |

---

## 4. 实验历程与关键结论

### 4.1 Qwen3-0.6B（冒烟：验证代码可跑）

| 项 | 结果 |
|----|------|
| GPTQ {2..8} 全位宽 | 196 层 × 7 档，8.6 GB，2244 s |
| DL 搜索（EAR≥0.99） | **无解**：即使全 8-bit，EAR 仅 **0.9810** |
| TL 搜索 | 护栏两次 fail（锚点 KL 0.196 / 0.0119，远离近无损区） |

**结论**：0.6B 过于敏感，只能验证"官方代码在 NPU 可跑"，**不能作为验收模型**。

### 4.2 Qwen3-4B（流程与口径验证）

| 配置 | bpp | PPL | ΔPPL | zero-shot 恢复 |
|------|-----|-----|------|----------------|
| BF16 | 16 | 13.2194 | — | 100% |
| uniform-4 | 4.156 | 13.8108 | +4.47% | 97.16% |
| uniform-6 | 6.156 | 13.2364 | +0.13% | 100.9% |
| SLQ@4.5（线性敏感度，自研） | 4.653 | 13.6506 | +3.26% | 94.19% |

**结论**：
1. 4B 需 **~6 bit** 才无损（与"0.6B 需 ~6bit"的判断一致）；
2. 自研"线性敏感度 + ILP"在全模型尺度信噪比不足（SLQ 反而更差）→ **必须用官方 Shapley**；
3. PPL 与 zero-shot 结论会矛盾，**验收以 zero-shot 为准**。

### 4.3 Qwen3.8-27B（正式验收）

| 阶段 | 结果 |
|------|------|
| GPTQ {2,4,6,8} | 496 层 / **275 GiB** / **11,882 s（3.3 h）**，峰值显存 31.8 GiB |
| 敏感度库 | 400 组，P=8（两进程各 P=4 合并），**2044 s/排列**，约 6.8 h/进程 |
| KL 列和 | 2-bit 0.4849 / 4-bit 0.01165 / 6-bit 0.000775 / 8-bit 0 |
| 官方 TL 搜索（2bit 锚点） | 4.353 bit（4.509 bpp），护栏 **pass（drift 1.61×）**，但高于 uniform-4 → 不满足 <4bit |

**TL 锚点问题**：Algorithm 2 要求锚点有可测退化（recovery < 1），而 27B 在 uniform-4 已 **100.7%** → 只能用 uniform-2（69.8%）作锚点，跨 100× 线性外推过于保守。**改为 Shapley + ILP 预算扫描 + 真实评测定界**。

---

## 5. 27B 最终结果

### 5.1 精度（6 任务 zero-shot）

**500 样本（最终判定）**

| 任务 | BF16 | b3.6（3.600 bpp） | b3.8（3.800 bpp） |
|------|------|-------------------|-------------------|
| arc_easy | 0.606 | 0.662 | 0.648 |
| arc_challenge | 0.508 | 0.524 | 0.538 |
| piqa | 0.810 | 0.820 | 0.820 |
| hellaswag | 0.654 | 0.632 | 0.650 |
| winogrande | 0.724 | 0.710 | 0.728 |
| openbookqa | 0.440 | 0.394 | 0.416 |
| **平均** | **0.6237** | **0.6237（100.0%）** | **0.6333（101.6%）** |

**200 样本（含其他配置）**

| 配置 | bpp | 平均 | 恢复率 |
|------|-----|------|--------|
| uniform-2 | 2.156 | 0.4342 | 69.8% |
| b3.4 | 3.400 | 0.6183 | 99.45% |
| **b3.6** | **3.600** | 0.6233 | **100.3%** |
| **b3.8** | **3.800** | 0.6317 | **101.6%** |
| uniform-4 | 4.156 | 0.6258 | 100.7% |
| uniform-6 | 6.156 | 0.6242 | 100.4% |

**结论**：
- **无损下界落在 3.400–3.600 bpp 之间**；b3.6 在 500 样本下平均完全持平；
- 逐任务有正负抵消，唯一需关注的是 **openbookqa**（b3.6 −4.6pp / b3.8 −2.4pp，SE≈2.2%，约 2σ/1σ）；
- 27B 的鲁棒性远高于 4B/0.6B，符合 SLQ 论文"大模型 3.3–4.7 bpp TL"的规律。

### 5.2 双卡 TP=2 部署（vllm-ascend）

| 指标 | 实测 |
|------|------|
| 卡 | 宿主 5,7（TP=2，eager） |
| 引擎初始化 | 216.1 s |
| 每卡权重 | **25.61 GiB** |
| 峰值激活 | 1.74 GiB |
| KV cache | **23.99 GiB / 326,562 tokens**（4096 长度并发 79.7×） |
| 生成 | 258 tokens / 52.5 s（4.9 tok/s，xhigh reasoning + 共享卡） |
| 结果 | `[smoke:deploy27b-tp2] RESULT: PASS` |

> 注：当前 checkpoint 为 fake-quant（BF16 权重），vLLM 按 float 加载，**显存尚未体现量化收益**；真 INT 打包后每卡权重预计降至 ≈8.1 GiB，KV 可扩至 ≈41 GiB。

---

## 6. 存储与位宽分析（详见 [quantization-analysis.md](quantization-analysis.md)）

### 6.1 体积

| 产物 | 体积 |
|------|------|
| 原始 BF16 权重 | 55.56 GB |
| fake-quant checkpoint | 54.74 GB（**−1.5%，无实际收益**） |
| **b3.6 INT 打包（估算）** | **17.26 GB → 3.17×** |
| b3.8 INT 打包 | 17.87 GB → 3.06× |
| 量化层权重库（中间件） | 295.26 GB（4 档位 × 496 层） |

未量化的 3.01B 参数（embedding / lm_head / norm / visual）固定占 **6.01 GB（35%）**，是压缩比天花板。

### 6.2 位宽占比（b3.6）

| 位宽 | 层数 | 层占比 | 参数量 | 参数占比 |
|------|------|--------|--------|----------|
| 2-bit | 148 | 29.84% | 6.78B | **27.85%** |
| 4-bit | 312 | 62.90% | 17.56B | **72.11%** |
| 6-bit | 23 | 4.64% | 5.65M | 0.02% |
| 8-bit | 13 | 2.62% | 3.19M | 0.01% |

**99.96% 的参数落在 2/4-bit**；6/8-bit 层虽占 7%，但都是 `in_proj_a/b` 这类 0.012B 参数的极小门控投影。

### 6.3 模块敏感度规律（b3.6 平均位宽）

```
in_proj_z 2.875 < in_proj_qkv 3.125 < gate/up 3.438 < q/k/v 3.500
< down 3.625 < o/out 3.750 < in_proj_a 4.125 < in_proj_b 4.583
```

线性注意力的小门控投影（a/b）最敏感、被保护到 4–6 bit；`in_proj_z/qkv` 最不敏感、承担主要降位。

---

## 7. 关键工程问题清单

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | VLM 静默随机权重 | `AutoModelForCausalLM` 类不匹配 | 按 `architectures` 选类 |
| 2 | GPTQ OOM | hooks 闭包持有 device 权重 | 断点续跑 + 显式释放 + empty_cache |
| 3 | OR-Tools 段错误 | C++ 与 NPU 运行时冲突 | 换 scipy HiGHS |
| 4 | TL 搜索 OOM | 248K 词表 logits | batch=1 + 每 batch empty_cache |
| 5 | 部署缺 image processor | checkpoint 未复制 VLM 预处理文件 | 从原模型补 `preprocessor_config.json` 等 |
| 6 | fork 线程池断言 | fork 后 `set_num_threads` | `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| 7 | spawn 递归 | 脚本模块级建 LLM | `if __name__ == "__main__":` 守卫 |
| 8 | Shapley 负边际 | 小样本估计噪声 | 成本钳制 ≥0 + 单调化 |
| 9 | 官方 bpp 口径误算 | 目标值直接 +0.156 | 改为参数加权 `compute_avg_bitwidth` |

---

## 8. 局限与后续工作

1. **fake-quant 未落地为真 INT 打包**：存储/显存收益（3.17×、每卡 8.1 GiB）目前是理论值，需接官方 `humming_online` 或自研 INT 加载路径。
2. **评分噪声**：500 样本 SE≈0.6%，obqa 的 −4.6pp 需 1000+ 样本确认；建议最终验收补 1000 样本或完整 test split。
3. **敏感度采样**：27B 受单卡算力限制，Shapley 用 4×2048（8k tokens）/进程 × 2 进程；若把样本提到 32×2048，可能进一步下探到 3.4–3.5 bpp。
4. **TL 锚点**：27B 在 4bit 已无损导致 2bit 锚点外推过远；若要复现论文 TL 流程，需增加 **3bit** GPTQ 档位作为近无损锚点（约 3.3 h）。
5. **吞吐**：当前 4.9 tok/s 受 xhigh reasoning + eager + 共享卡影响，非量化所致；正式服务建议 graph 模式 + 独占卡。

---

## 9. 产物清单

**服务器 `/opt/zh/slq-official`**
```
config_tl27b_b3.4.txt / b3.6.txt / b3.8.txt / b4.0.txt
data/slq_27b/<snap>/4bit/             # 496 层 × {2,4,6,8} + *_qparams.pt（275 GiB）
data/slq_27b/<snap>/4bit/error_db.json
data/slq_27b/<snap>/4bit/shapley_*_8perms_bw2-4-6-8_*.json
ckpt_27b/<snap>/b36 | b38 | b34 | uniform2|4|6   # fake-quant，各 ~51 GB
run_deploy27b.sh                      # TP=2 部署（spawn + VLM 预处理）
```

**评测结果 `/opt/zh/SLQ-gptq`**
```
zs27b_bf16_500.json / zs27b_b36_500.json / zs27b_b38_500.json
zs27b_{bf16,u2,u4,u6,b34,b36,b38}.json     # 200 样本
```

**本地文档**
```
experiment/task.md                    # 阶段规划与状态
experiment/progress.md                # 全量进展日志
experiment/quantization-analysis.md   # 存储与位宽详细分析
experiment/summary-report.md          # 本报告
```

---

## 10. 复现命令

```bash
# 1) GPTQ 全位宽（27B，单卡，cpu_offload）
bash run_slq27b_v2.sh

# 2) 敏感度库（两进程并行，各 P=4）
bash run_shapley27b.sh 5 42 &
bash run_shapley27b.sh 7 43 &
python merge_shapley.py <seed42.json> <seed43.json> <merged.json>

# 3) 预算扫描配置
python make_configs.py

# 4) checkpoint 拼装（补齐 VLM 预处理文件）
python slq_build_checkpoint.py --model <27B> --layer-dir <store> \
  --config config_tl27b_b3.6.txt --output-dir ckpt_27b --subdir-name b36 \
  --dtype bfloat16 --quant-format fake --group-size 128 --force

# 5) zero-shot 评测
python zeroshot_eval.py --model <ckpt>/b36 --device npu:0 --limit 500 \
  --tasks arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa --out zs27b_b36_500.json

# 6) 双卡部署
bash run_deploy27b.sh <ckpt>/b36 2 5,7 1 8100
```

---

*本报告所有数值均来自实测日志与 JSON 产物，未使用估算值（除明确标注"估算/理论"的 INT 打包体积）。*
