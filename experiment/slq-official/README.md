# slq-official：官方 SLQ 流水线的 NPU 适配（现行实现）

本目录不是一份完整代码仓，而是 **官方 [IST-DASLab/SLQ](https://github.com/IST-DASLab/SLQ)
（基线 commit `eb9f1f2`）的 NPU 适配层 + 27B 产物**。官方仓只有少数文件需要改，它们以
“覆盖文件”的形式保存在这里；其余官方文件保持原样。

## 1. 目录内容

| 路径 | 作用 |
|------|------|
| `slq_quantize.py` | ① 逐层多次 GPTQ：一次 Hessian 产出 `{2,4,6,8}` 全位宽，`cal_bits=4` 序贯传播 |
| `slq_build_database.py` | ② 敏感度库：线性 α 或 多 bitwidth Shapley（Alg.1） |
| `slq_search_dl.py` | ③ ILP 位宽分配：TL（Algorithm 2）与 DL（EAR）两种搜索 |
| `src/models/loading.py` | 按 `config.json architectures` 选模型类（VLM 必需） |
| `src/quant/quantizer.py` | 断点续跑、OOM 释放、多卡一致性 |
| `src/quant/fast_obq.py` | Cholesky 走 CPU（NPU 无 `torch.linalg`） |
| `src/metrics/evaluation.py` | 大词表 logits 的每 batch `empty_cache` |
| `eval/zeroshot_eval.py` | 评测：6 任务 log-likelihood（替代超时的 lm_eval） |
| `configs/` | 27B 的 4 个预算配置（b3.4 / b3.6 / b3.8 / b4.0） |
| `scripts/` | 流水线入口脚本（校准 → GPTQ → 敏感度 → 配置 → 拼装 → 校验），见 §4 |
| `deploy/` | 部署与 HTTP 验证：`pipeline_27b.sh`（自愈总控）、`serve_*.sh`、`http_validate.sh`，见 §5 |
| `DEPLOYMENT-VALIDATION.md` | 一次完整的「重建 → 拼装 → 上线 → HTTP 验证」记录 |
| `patches/npu-adaptation.patch` | 上面 7 个 python 文件的**全部** NPU 改动，单个 diff |

## 2. 把 NPU 适配装进官方仓

```bash
# 1) 取官方代码
git clone https://github.com/IST-DASLab/SLQ.git /opt/zh/slq-official
cd /opt/zh/slq-official && git checkout eb9f1f2

# 2) 方式 A（推荐）：直接覆盖文件
cp -v <本目录>/slq_quantize.py <本目录>/slq_build_database.py <本目录>/slq_search_dl.py /opt/zh/slq-official/
cp -rv <本目录>/src/. /opt/zh/slq-official/src/
mkdir -p /opt/zh/slq-official/eval && cp -v <本目录>/eval/zeroshot_eval.py /opt/zh/slq-official/eval/

# 3) 方式 B（等价，便于审查）：打 diff
git -C /opt/zh/slq-official apply <本目录>/patches/npu-adaptation.patch

# 4) 校验：两种方式结果必须与覆盖文件逐字节一致
```

> 早期版本这里是 9 个 `patch_npu*.py` 脚本，存在两个问题：`patch_npu.py` 重跑会把
> `fast_obq.py` 写成语法错误，且没有任何补丁产生 `loading.py` 的 VLM 修复，
> 因此“打补丁”这条路会静默加载随机权重。现在统一为 **一个幂等的 diff**，
> 并已在官方 `eb9f1f2` 上验证 apply 后与覆盖文件 hash 完全一致。

## 3. NPU 适配清单

| 问题 | 修复位置 |
|------|----------|
| `nccl` / `cuda:{rank}` | `hccl` / `npu:{rank}`、`torch.npu.set_device` |
| `torch.linalg.cholesky` NPU 不支持 | Hessian 求逆落 CPU 后回迁（`fast_obq.py`） |
| 27B OOM（第 20/64 块） | 断点续跑 + 显式释放旧层/hooks + 每 8 次前向 `empty_cache`（`quantizer.py`） |
| 多卡断点续跑分支不一致导致挂死 | resume 标志跨 rank 取 MIN（`quantizer.py`） |
| VLM 加载成随机权重 | 按 `architectures` 选类（`loading.py`） |
| OR-Tools SCIP 段错误 | 换 `scipy.optimize.milp`（HiGHS，`slq_search_dl.py`） |
| Shapley 负边际 | 成本钳制 ≥0 + 强制位宽单调（`slq_search_dl.py`） |
| 248K 词表 logits OOM | 每 batch `empty_cache`，batch 降到 1（`evaluation.py`） |
| 校准集用本地 `.pt` | `--calibration-data` 支持 `.pt`，tag 取文件名 stem（`slq_build_database.py`） |

## 4. 脚本索引

所有脚本都 `source scripts/common.sh`，共享路径与离线/conda 环境；路径可用环境变量覆盖
（`SLQ_ROOT`、`SLQ_MODEL`、`SLQ_LAYERS`、`SLQ_CKPT`、`SLQ_RESULTS`、`ASCEND_RT_VISIBLE_DEVICES`）。

| 脚本 | 阶段 | 说明 |
|------|------|------|
| `run_calib27b.sh` | 准备 | 生成 `128×8192` 与 `128×2048` 两套校准集 |
| `run_slq27b.sh` | ① | GPTQ 全位宽层库（496 层 × 4 档，约 3.3 h / 275 GiB） |
| `run_db27b.sh base` | ② | `error_db.json` + baseline（线性 α，3 min） |
| `run_db27b.sh shapley <seed>` | ② | Shapley 库（4 排列/进程，约 6.8 h；两卡不同 seed 后合并） |
| `merge_shapley.py` | ② | 合并两个 seed 的库 → 等效 8 排列 |
| `make_configs.py` | ③ | ILP 预算扫描 → `configs/config_tl27b_b*.txt` |
| `run_ckpt27b.sh <bits>` | ④ | uniform 锚点 checkpoint（对照基线） |
| `build_ckpt.sh` | ④ | 按配置拼装 fake-quant checkpoint，并补齐 VLM 预处理文件 |
| `verify_fakequant.sh` | ④ 校验 | 指纹校验：每 128 组取值数 ≤ 2^bits，且与 BF16 有量化差异 |
| `run_build_eval.sh [limit] <budget>...` | ④+评测 | 按配置拼 checkpoint 并评测（默认 200 样本） |
| `run_zs.sh [limit] <spec>...` | 评测 | 对已有模型评测（`bf16` / `b36` / `uniform4` / 绝对路径） |
| `run_deploy27b.sh <ckpt> [TP] [devices]` | 部署 | vllm-ascend TP=2 容器 smoke |
| `smoke_llm.py` | 部署 | 容器内执行的 smoke（由 `run_deploy27b.sh` 挂载） |
| `run_slq06b.sh` / `run_db06b.sh` / `run_shapley06b.sh` / `run_tl06b.sh` | 冒烟 | 0.6B 全流程（只验证能跑通，不是验收模型） |

评测结果统一写入 `$SLQ_RESULTS`（默认 `$SLQ_ROOT/results`），文件名形如
`zs27b_b36_500.json`。

## 5. 部署与 HTTP 验证

`deploy/` 下的脚本把「拼装 → 校验 → 上线 → 接口验证」串起来，路径同样可用 `SLQ_ROOT` 等覆盖。

| 脚本 | 说明 |
|------|------|
| `pipeline_27b.sh` | 无人值守总控：SLQ 量化（断点续跑 + 失败换卡重试）→ 拼装 → 校验 → 上线 → HTTP 验证 |
| `finalize_steps.sh` | 后四步：拼装 → 校验 → 上线 → HTTP 验证（层库已就绪时单独用） |
| `serve_27b.sh` | 通用 vLLM-Ascend OpenAI 服务启动器（TP、卡、端口、HCCL 网卡可配） |
| `serve_b36.sh` | b3.6 checkpoint 专用：起服务并等到 `/health` 变绿 |
| `http_validate.sh` | 五连验证：`/health`、`/v1/models`、chat、completions、流式 SSE |

```bash
# 一键（后台，自愈）
nohup bash deploy/pipeline_27b.sh > /dev/null 2>&1 &

# 或分步
bash scripts/build_ckpt.sh && bash scripts/verify_fakequant.sh
bash deploy/serve_b36.sh                      # 默认 TP=2、卡 0,2、端口 8098
bash deploy/http_validate.sh 8098 qwen3.8-27b-slq-b36
```

该主机上部署有两个硬性条件（详见 [DEPLOYMENT-VALIDATION.md](DEPLOYMENT-VALIDATION.md)）：

- 必须用 `quay.io/ascend/vllm-ascend:v0.30.0` 镜像（nightly main 的 vLLM 0.33 加载不了该架构）；
- 卡必须**非连续**映射（如 `--device /dev/davinci0 --device /dev/davinci2`），TP=2 还要显式
  设置 `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`。

冷启动后头几个不同形状的请求会触发图编译（实测 194–300 s），上线后先 warmup 再压测。

## 6. bpp 口径与配置

- 位宽计费含 g128 的 scale/zero 开销：`2→2.156, 4→4.156, 6→6.156, 8→8.156`；
- 平均位宽必须用 `compute_avg_bitwidth`（参数加权）计算，不要用“目标 nominal + 0.156”；
- 四档预算与预测 KL：3.400/0.0753、**3.600/0.0473**、3.800/0.0269、4.000/0.0155；
- b3.6 实测 100.0% 恢复（500 样本），b3.8 为 101.6%。

## 7. 已知限制

- checkpoint 是 **fake-quant**（BF16 落在整数格点），体积与 BF16 相同；17.26 GB / 每卡 8.1 GiB
  是 INT 打包后的理论值，尚未接 humming/自研 INT 加载。
- Shapley 采样只有 8k tokens/进程（4×2048），提高采样可能进一步下探到 3.4–3.5 bpp。
- 官方 TL 搜索在 27B 上不可用：2-bit 锚点跨 100× 外推过保守（4.509 bpp），最终改用
  “预算扫描 + 真实评测定界”。
