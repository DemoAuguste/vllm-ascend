# Qwen3.8-27B SLQ 量化 · 完整复现指南

> 目标：从零复现 **3.600 bpp / zero-shot 恢复 100.0% / 双卡 TP=2 部署** 的完整结果。
> 对应代码：`experiment/slq-official/`（官方仓 + NPU 适配），结果分析见 [summary-report.md](summary-report.md)、[quantization-analysis.md](quantization-analysis.md)。
> 所有命令在 NPU 服务器（Ascend 910B3）上执行，路径以 `/opt/zh/slq-official` 为例。

---

## 0. 总览

```
[0] 环境 + 代码 + 数据
      │
[A] 0.6B 冒烟（可选，验证代码可跑，~40 min）
      │
[B1] 27B GPTQ {2,4,6,8}          ~3.3 h   产出 275 GiB 层库
      │
[B2] error_db + baseline          ~3 min
      │
[B3] Shapley 敏感度（2 卡并行）    ~6.8 h   400 组 × P=8
      │
[B4] ILP 预算扫描                 ~30 s    4 个配置
      │
[B5] checkpoint 拼装               ~5 min/个
      │
[B6] zero-shot 评测（500 样本）     ~70 min/模型
      │
[B7] 双卡 TP=2 部署               ~6 min
```

**资源预算**：单卡 60.96 GiB；27B GPTQ 峰值 31.8 GiB；Shapley 每进程 51 GiB；磁盘至少 **600 GB**（模型 55 GB + 层库 295 GB + 若干 checkpoint 各 55 GB）。

---

## 1. 环境准备

### 1.1 硬件与软件基线

| 项 | 要求 |
|----|------|
| NPU | Ascend 910B3（A2），单卡 60.96 GiB，≥2 卡可用 |
| CANN | 9.2.0（`source /opt/zh/cann/ascend-toolkit/set_env.sh`） |
| Python 环境 | conda env `agent`：torch 2.10.0 + torch_npu 2.10.0 + transformers 5.14.1 |
| vLLM | 镜像 `quay.io/ascend/vllm-ascend:nightly-main`（vLLM 0.30.0，A2 构建） |
| 模型 | Qwen3.8-27B（VLM，`Qwen3_5ForConditionalGeneration`） |

### 1.2 选择可用卡

```bash
npu-smi info        # 选 Health=OK 且 HBM 空闲的卡；本实验用 5 和 7
```

> **坑**：被其它**运行中容器**映射过的卡，在新容器里可能枚举失败（`torch.npu.device_count()=0`）或初始化崩溃。优先用两个 OK 卡，部署固定用 5,7（实测可用组合）。

### 1.3 通用环境变量（每个脚本都已内置）

```bash
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=5          # 单卡
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

> `HF_*_OFFLINE=1` 是必需的：`huggingface_hub/httpx` 在本环境联网会报 `client has been closed`。

---

## 2. 代码准备

### 2.1 获取代码

```bash
cd /opt/zh
git clone https://github.com/IST-DASLab/SLQ.git slq-official
cd slq-official
git rev-parse HEAD          # 记录基线 commit，便于回溯
```

### 2.2 应用 NPU 适配（两种方式任选）

**方式 A（推荐）：直接使用已同步文件**

把本仓库 `experiment/slq-official/` 下的以下文件覆盖到 `/opt/zh/slq-official/`：

```
slq_quantize.py
slq_build_database.py
slq_search_dl.py
src/metrics/evaluation.py
src/models/loading.py
src/quant/fast_obq.py
src/quant/quantizer.py
```

**方式 B：运行增量补丁**（`patches/patch_npu*.py`，按编号顺序）

```bash
for p in patch_npu.py patch_npu2.py patch_npu3.py patch_npu4.py \
         patch_npu5.py patch_npu6.py patch_npu7.py patch_npu8.py \
         patch_eval_npu.py; do
  python patches/$p
done
```

### 2.3 修改清单（复现时必须生效）

| 文件 | 关键点 |
|------|--------|
| `slq_quantize.py` | `hccl`、`npu:{rank}`、`--model_class`（VLM 类） |
| `slq_build_database.py` | `--calibration-data` 支持 `.pt`；tag 用文件 stem |
| `slq_search_dl.py` | **scipy MILP 替代 OR-Tools**；Shapley 成本钳制 ≥0 + 单调 |
| `src/models/loading.py` | 按 `config.json architectures` 选类（必须，否则随机权重） |
| `src/quant/fast_obq.py` | Cholesky 走 CPU |
| `src/quant/quantizer.py` | 断点续跑 + 显存释放 |
| `src/metrics/evaluation.py` | 每 batch `empty_cache` |

### 2.4 校验 NPU 适配生效

```bash
python - <<'PY'
import torch, torch_npu
print(torch.npu.is_available(), torch.npu.device_count())
PY
# 期望：True 1
```

---

## 3. 数据准备

### 3.1 下载 FineWeb-Edu 分片（2.1 GB）

```bash
mkdir -p data
curl -L --retry 5 -o data/fineweb_10bt_000.parquet \
  https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet
ls -lh data/fineweb_10bt_000.parquet     # 约 2.1G
```

### 3.2 用 27B tokenizer 生成校准集

`scripts/build_calib.py`：从 parquet 抽文本、tokenize、切成 `(1, seq)` 张量列表。

```bash
python -u build_calib.py --parquet data/fineweb_10bt_000.parquet \
    --tokenizer "$M" --seq 8192 --num 128 --out data/calib_27b_1m_8192.pt

python -u build_calib.py --parquet data/fineweb_10bt_000.parquet \
    --tokenizer "$M" --seq 2048 --num 128 --out data/calib_27b_256k_2048.pt
```

其中：

```bash
M=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

**校验**：

```bash
python -c "import torch;d=torch.load('data/calib_27b_256k_2048.pt');print(len(d), d[0].shape)"
# 期望：128 (1, 2048)
```

> 0.6B 冒烟另需 `calib_qwen3_8m_8192.pt`（1024×8192）与 `calib_qwen3_1m_2048.pt`（512×2048），用 0.6B tokenizer 同样生成。

---

## 4. 阶段 A：0.6B 冒烟（可选，~40 min）

目的仅验证"代码能跑通"，**0.6B 不能作为精度验收模型**（全 8-bit 时 EAR 仍只有 0.9810）。

### A1 GPTQ 全位宽

```bash
bash scripts/run_slq06b.sh
```

期望：196 层 × 7 档位 `.pth`，约 8.6 GB，日志 `SLQ_QUANT_DONE`，耗时约 37 min。

### A2 敏感度库 + TL 搜索

```bash
bash scripts/run_db06b.sh       # 线性 α，约 18 min
bash scripts/run_tl06b.sh       # TL（需要 shapley 库时先跑 run_shapley06b.sh）
```

期望：TL 流程可执行并输出 config；**护栏可能 fail**（0.6B 远离近无损区），属预期。

---

## 5. 阶段 B：27B 主实验

### B1 GPTQ 全位宽 {2,4,6,8}（约 3.3 h）

```bash
bash scripts/run_slq27b_v2.sh
```

关键参数：

| 参数 | 值 | 说明 |
|------|----|------|
| `--quantizable_modules` | `.*layers.*((q\|k\|v\|o\|gate\|up\|down)_proj\|in_proj_qkv\|in_proj_a\|in_proj_b\|in_proj_z\|out_proj)$` | 覆盖 linear_attn + full_attn + MLP |
| `--bitwidth_options` | `2 4 6 8` | 单 Hessian 一次产出 4 档 |
| `--calibration_bitwidth` | `4` | 序贯传播锚点 |
| `--calibration_tokens` | `262144` | 32×8192 |
| `--group_size` | `128` | 非对称 INT |
| `--cpu_offload_modules/activations` | 开 | 控显存 |

**断点续跑**：每块完成后落盘 `<layer>/<cal_bits>.pth`；中断后重跑会自动跳过已完成的块（日志 `resumed block k`）。

**校验**：

```bash
LD=/opt/zh/slq-official/data/slq_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/4bit
ls "$LD" | wc -l          # 期望 496
du -sh /opt/zh/slq-official/data/slq_27b   # 期望约 275G
ls "$LD/model.language_model.layers.0.mlp.gate_proj"
# 期望：2.pth 2_qparams.pt 4.pth 4_qparams.pt 6.pth 6_qparams.pt 8.pth 8_qparams.pt
```

日志尾部期望：

```
block 63 done, npu alloc 31.8 GiB
Quantization took 11882 s.
SLQ27_DONE
```

### B2 error_db + baseline（约 3 min）

```bash
bash scripts/run_db27b_base.sh
```

期望日志：`496 layers in 400 groups`，产出：

- `<LD>/error_db.json`（含 `param_counts`）
- `<LD>/baseline_topk10_calib_27b_256k_2048_4x2048_seed42.pt`

### B3 Shapley 敏感度（约 6.8 h，2 卡并行）

```bash
# 两个独立进程，不同 seed，各 P=4
nohup bash scripts/run_shapley27b.sh 5 42 > shapley27b_s42.log 2>&1 &
nohup bash scripts/run_shapley27b.sh 7 43 > shapley27b_s43.log 2>&1 &
```

成本模型：`(4-1 个档位) × 4 排列 × 400 组 × 单次 metric(≈5.1 s)` ≈ 6.8 h/进程。实测 **2044 s/排列**。

完成后合并（等价 P=8）：

```bash
cd "$LD"
python /opt/zh/slq-official/merge_shapley.py \
  shapley_topk10_calib_27b_256k_2048_4x2048_4perms_bw2-4-6-8_grpb7761ae4_seed42.json \
  shapley_topk10_calib_27b_256k_2048_4x2048_4perms_bw2-4-6-8_grpb7761ae4_seed43.json \
  shapley_topk10_calib_27b_256k_2048_4x2048_8perms_bw2-4-6-8_grpb7761ae4_seed42.json
```

**校验**（数值应基本一致）：

| 位宽 | KL 列和（400 组） |
|------|------------------|
| 2-bit | ≈ 0.4849 |
| 4-bit | ≈ 0.01165 |
| 6-bit | ≈ 0.000775 |
| 8-bit | 0（定义） |

> 若出现少量负值属估计噪声，ILP 会做"钳制 ≥0 + 单调化"。

### B4 生成混合精度配置（约 30 s）

```bash
python make_configs.py
```

期望输出：

| 配置 | 平均位宽（含开销） | 预测 KL |
|------|-------------------|---------|
| `config_tl27b_b3.4.txt` | 3.400 | 0.0753 |
| `config_tl27b_b3.6.txt` | **3.600** | 0.0473 |
| `config_tl27b_b3.8.txt` | 3.800 | 0.0269 |
| `config_tl27b_b4.0.txt` | 4.000 | 0.0155 |

> bpp 必须用**参数加权**的 `compute_avg_bitwidth`，不要用"目标 nominal + 0.156"（会多算）。

### B5 checkpoint 拼装（约 5 min/个）

```bash
# b3.6 / b3.8 混合配置
python -u slq_build_checkpoint.py --model "$M" --layer-dir "$LD" \
  --config config_tl27b_b3.6.txt --output-dir /opt/zh/slq-official/ckpt_27b \
  --subdir-name b36 --dtype bfloat16 --quant-format fake --group-size 128 --force

python -u slq_build_checkpoint.py --model "$M" --layer-dir "$LD" \
  --config config_tl27b_b3.8.txt --output-dir /opt/zh/slq-official/ckpt_27b \
  --subdir-name b38 --dtype bfloat16 --quant-format fake --group-size 128 --force

# 对照锚点（uniform）
# bash scripts/run_ckpt27b.sh 0 4    # uniform-4
# bash scripts/run_ckpt27b.sh 0 6    # uniform-6
```

**关键：补齐 VLM 预处理文件**（否则 vLLM 报 `Can't load image processor`）

```bash
C=/opt/zh/slq-official/ckpt_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
for D in b36 b38; do
  for f in "$M"/*.json "$M"/*.jinja "$M"/*.txt "$M"/*.model; do
    [ -f "$f" ] && cp -n "$f" "$C/$D/"
  done
done
```

> `-n` 很重要：不可覆盖 checkpoint 自己的 `config.json` 与 `model.safetensors.index.json`。

**校验**：

```bash
grep -o 'model-0000[0-9]-of-000[0-9][0-9]' "$C/b36/model.safetensors.index.json" | sort -u
# 期望：model-00001-of-00002 / model-00002-of-00002
ls "$C/b36/preprocessor_config.json" "$C/b36/video_preprocessor_config.json"
du -sh "$C/b36"        # 期望约 55G（fake-quant，与 BF16 同尺寸）
```

### B6 zero-shot 评测（500 样本，约 70 min/模型）

评测脚本 `zeroshot_eval.py`（log-likelihood，NPU 原生；`lm_eval` 在本环境会超时）。

```bash
cd /opt/zh/SLQ-gptq
T=arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa

python -u zeroshot_eval.py --model "$M"        --device npu:0 --limit 500 --tasks $T --out zs27b_bf16_500.json
python -u zeroshot_eval.py --model "$C/b36"    --device npu:0 --limit 500 --tasks $T --out zs27b_b36_500.json
python -u zeroshot_eval.py --model "$C/b38"    --device npu:0 --limit 500 --tasks $T --out zs27b_b38_500.json
```

> 单卡运行；若 `--limit 200`，耗时约 17 min/模型（用于快速对比）。

**期望结果（必须对齐）**：

| 任务 | BF16 | b3.6 | b3.8 |
|------|------|------|------|
| arc_easy | 0.606 | 0.662 | 0.648 |
| arc_challenge | 0.508 | 0.524 | 0.538 |
| piqa | 0.810 | 0.820 | 0.820 |
| hellaswag | 0.654 | 0.632 | 0.650 |
| winogrande | 0.724 | 0.710 | 0.728 |
| openbookqa | 0.440 | 0.394 | 0.416 |
| **平均** | **0.6237** | **0.6237（100.0%）** | **0.6333（101.6%）** |

> 允许 ±0.02 的逐任务波动（500 样本 SE≈0.6%，任务级 SE≈2%）；**平均值应非常接近**。

### B7 双卡 TP=2 部署（约 6 min）

```bash
bash /opt/zh/SLQ-gptq/run_deploy27b.sh \
  "$C/b36" 2 5,7 1 8100
```

脚本要点（缺一不可）：

- `-e VLLM_WORKER_MULTIPROC_METHOD=spawn -e OMP_NUM_THREADS=1`（否则 fork 线程池断言崩溃）
- smoke 脚本必须有 `if __name__ == "__main__": main()` 守卫（否则 spawn 递归）
- 镜像 `nightly-main`，`--net=host --ipc=host --shm-size=64g`，映射 davinci5/7

**期望日志**：

```
[smoke] model loaded in 216.1s
Available KV cache memory: 23.99 GiB
[s.smoke:deploy27b-tp2] RESULT: PASS
```

| 指标 | 期望值 |
|------|--------|
| 每卡权重 | 25.61 GiB |
| 峰值激活 | 1.74 GiB |
| KV cache | 23.99 GiB / 326,562 tokens |
| 生成 | ≈258 tokens / 52.5 s（4.9 tok/s） |

---

## 6. 结果校验清单

| # | 检查项 | 通过标准 |
|---|--------|---------|
| 1 | 层库完整性 | 496 个层目录，每个 8 个文件 |
| 2 | 层库体积 | ≈275 GiB |
| 3 | 分组数 | `496 layers in 400 groups` |
| 4 | Shapley KL 单调 | 2bit 0.485 > 4bit 0.0117 > 6bit 0.00078 > 8bit 0 |
| 5 | 配置 bpp | b3.4/3.6/3.8/4.0 = 3.400/3.600/3.800/4.000 |
| 6 | checkpoint | 2 分片、index 指向 02、含 VLM 预处理文件 |
| 7 | 精度 | b3.6 平均 0.6237（100.0%） |
| 8 | 部署 | `RESULT: PASS`、每卡 25.61 GiB |
| 9 | 存储换算 | b3.6 打包后 17.26 GB（3.17×） |

---

## 7. 故障排查

| 症状 | 根因 | 处理 |
|------|------|------|
| 27B 加载后输出乱码/随机 | `AutoModelForCausalLM` 解析成 `Qwen3_5ForCausalLM` | 用 `--model_class Qwen3_5ForConditionalGeneration` |
| 第 20 块左右 OOM（47 GiB） | 旧层 + hooks 闭包持有权重 | 断点续跑 + `layer.cpu()/H=None/handles.clear()` + 每 8 次前向 `empty_cache` |
| `cholesky` 报不支持 | NPU 无 `torch.linalg` | CPU 求逆后回迁 |
| Shapley 负边际 | 小样本估计噪声 | 成本钳制 ≥0 + 单调化 |
| `slq_search_dl.py` 段错误 | OR-Tools SCIP 与 NPU 运行时冲突 | 换 `scipy.optimize.milp`（HiGHS） |
| `compute_baseline_topk` OOM | 248K 词表 logits | `--batch-size 1` + 每 batch `empty_cache` |
| `Can't load image processor` | checkpoint 缺 `preprocessor_config.json` | 从原模型 `cp -n` 补齐 |
| `pool INTERNAL ASSERT FAILED` | fork 后 `set_num_threads` | `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| spawn 后进程无限递归 | 脚本模块级构造 `LLM(...)` | 加 `if __name__ == "__main__":` |
| `huggingface_hub: client has been closed` | httpx 版本 bug | 全部 `HF_*_OFFLINE=1`，数据用 curl |
| 容器内 `npu-smi` 不存在 | 镜像未带工具 | 忽略（仅影响 CPU 绑核提示） |
| 卡枚举为 0 | 卡被其它容器锁定 | 换 OK 卡（5,7 已验证） |

---

## 8. 时间与空间预算

| 步骤 | 时间 | 磁盘增量 |
|------|------|---------|
| 环境 + 数据 | ~20 min | 2.1 GB |
| 0.6B 冒烟（可选） | ~1 h | ~10 GB |
| B1 GPTQ 27B | **3.3 h** | **275 GiB** |
| B2 error_db/baseline | ~3 min | <1 GB |
| B3 Shapley（2 卡） | **6.8 h** | <10 MB |
| B4 配置 | ~30 s | <1 MB |
| B5 checkpoint（每个） | ~5 min | 55 GB |
| B6 评测（每个模型） | ~70 min | <1 MB |
| B7 部署 | ~6 min | — |

**总计（从零到部署）**：约 **12–13 h**（含两个 Shapley 进程并行）。

---

## 9. 产物目录

```
/opt/zh/slq-official/
├── data/
│   ├── fineweb_10bt_000.parquet
│   ├── calib_27b_1m_8192.pt
│   ├── calib_27b_256k_2048.pt
│   └── slq_27b/<snap>/4bit/          # 496 层 × {2,4,6,8}.pth + *_qparams.pt
│       ├── error_db.json
│       ├── baseline_topk10_*.pt
│       └── shapley_*_8perms_bw2-4-6-8_*.json
├── ckpt_27b/<snap>/
│   ├── b36/  b38/  b34/              # 混合精度 fake checkpoint
│   └── uniform2/ uniform4/ uniform6/ # 对照锚点
├── config_tl27b_b{3.4,3.6,3.8,4.0}.txt
└── scripts/                          # 本仓库 experiment/slq-official/scripts

/opt/zh/SLQ-gptq/
├── zeroshot_eval.py
├── zs27b_bf16_500.json / zs27b_b36_500.json / zs27b_b38_500.json
└── run_deploy27b.sh
```

---

## 10. 口径与已知偏差

1. **bpp**：参数加权、含 g128 开销（+0.156），必须用 `compute_avg_bitwidth` 计算；本报告 b3.6 = **3.600**。
2. **fake-quant**：checkpoint 权重是 BF16 落在整数格点，**体积与 BF16 相同**；17.26 GB 是 INT 打包后的理论值。
3. **评测口径**：6 任务 × 500 样本 log-likelihood，提示格式为 `Question: … Answer:` + 选项原文，绝对准确率偏低但相对恢复率自洽。
4. **Shapley 采样**：27B 受单卡算力限制使用 4×2048（8k tokens）× 2 进程；提高采样可能进一步下探到 3.4–3.5 bpp。
5. **TL 搜索**：官方 Algorithm 2 要求锚点有退化，27B 在 4bit 已无损，故本实验最终用"Shapley + ILP 预算扫描 + 真实评测评测"定界，TL 结果（4.509 bpp，护栏 pass 1.61×）仅作流程验证。
