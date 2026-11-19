# 部署验证记录：Qwen3.8-27B SLQ b3.6（2026-11-18 → 11-19）

> 本文记录一次从零重建量化产物 → 拼装 → vLLM-Ascend TP=2 上线 → HTTP 接口验证的完整过程。
> 复现入口：`deploy/pipeline_27b.sh`（无人值守，自愈）；手工步骤见
> [reproduction-guide.md](../reproduction-guide.md) 与 [README.md](README.md)。

## 1. 结论

| 验收项 | 结果 |
|--------|------|
| SLQ 量化 | 496/496 层，层库 **275 GB**，b3.6 配置（2/4/6/8 bit） |
| checkpoint | **51 GB**（fake-quant BF16，2 分片）+ VLM 预处理文件 |
| fake-quant 指纹 | 每 128 组取值数 4/16/54/101，对应上限 4/16/64/256 → **PASS** |
| 服务 | vLLM-Ascend `v0.30.0` 镜像，TP=2，端口 8098，`/health` 200 |
| HTTP 接口 | `/v1/models`、`/v1/chat/completions`、`/v1/completions`、流式 SSE 全部可用 |
| 生成质量 | 中英文回答连贯，推理题（9.11 vs 9.9）结论正确 |
| 性能 | 预热后 128 tokens ≈ **4.3 s**（约 28–30 tok/s 解码） |

## 2. 时间线

| 时间（CST） | 事件 |
|---|---|
| 11-18 13:12 | 阶段① GPTQ 启动（65536 校准 tokens，断点续跑，5 号卡） |
| 11-18 15:05 | ⚠️ NPU `Vector core execution timed out`（错误码 507034），已完成 188 层落盘 |
| 11-18 15:07 | 自愈管道重启（换卡 + 断点续跑），此后未再中断 |
| 11-18 18:10 | 阶段①完成：496/496 层，本段耗时 10983 s（3 h 03 m） |
| 11-18 18:11 | checkpoint 拼装完成（51 GB） |
| 11-18 18:15 | fake-quant 指纹校验 PASS |
| 11-18 18:26 | 服务就绪 + HTTP 五连验证通过（`ALL_DONE`） |

> 背景：服务器上原有的层库与 checkpoint 已被清理，本次为**从零重跑**。

## 3. 环境要点（A2 主机，Ascend 910B3）

| 项 | 值 / 说明 |
|----|-----------|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0`（vLLM 0.30.0 / transformers 5.14.1） |
| 为什么不用 nightly | nightly main 已升到 vLLM 0.33 + transformers 5.19，无法加载 `Qwen3_5ForConditionalGeneration` |
| 设备映射 | 该主机只允许**非连续**映射，`--device /dev/davinci0 --device /dev/davinci2`；映射被其他 namespace 占用的卡会让容器看到 0 张卡（日志 `uda_occupy_dev_by_ns Conflict`，表层报错是 triton `get_arch`） |
| TP=2 | 必须显式 `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`，否则 `hcclGetRootInfo error 19` |
| 量化性能 | torch 默认 192 线程会让每层 90–140 s（小矩阵求逆线程超订）；设 `OMP/MKL_NUM_THREADS=16` 后约 33–37 s/层 |

## 4. 产物

服务器 `/opt/zh/slq-official`：

```
data/slq_27b/4bit/                        496 层 × {2,4,6,8}.pth + *_qparams.pt（275 GB）
ckpt_27b/<snap>/b36/                      fake-quant checkpoint（51 GB，2 分片）
  ├── model-00001-of-00002.safetensors (47 GB)
  ├── model-00002-of-00002.safetensors (4.6 GB)
  ├── config.json / preprocessor_config.json / video_preprocessor_config.json
  └── quantization_config.txt
finalize_state.txt / build_b36.log / verify_b36.log / http_validation.log
```

## 5. 服务配置与指标

```
容器  slq-serve-27b-tp2-8098    镜像 vllm-ascend:v0.30.0
卡    0、2（TP=2）              端口 0.0.0.0:8098
模型  qwen3.8-27b-slq-b36       max_model_len 4096
```

| 指标 | 实测 |
|------|------|
| 每卡权重 | 25.6118 GB（fake-quant，与 BF16 同尺寸） |
| KV cache | 24.11 GiB / 320,605 tokens（4096 长度并发 78×） |
| 引擎初始化 | 179.3 s（含 69 s 图编译） |
| 预热后解码 | 约 28–30 tok/s |

> ⚠️ 冷启动后**头几个不同形状的请求会触发图编译，实测 194–300 s**；上线后先做 warmup 请求再压测。

## 6. HTTP 验证结果

| 端点 | 结果 |
|------|------|
| `GET /health` | `200` |
| `GET /v1/models` | `id=qwen3.8-27b-slq-b36`，`root=/work/b36`，`max_model_len=4096` |
| `POST /v1/chat/completions` | 128 tokens ≈ 4.3 s，内容连贯 |
| `POST /v1/completions` | `The capital of China is` → *"Beijing, which is also the largest city..."* |
| 流式 SSE | 正常逐 token 下发 |

调用示例：

```bash
# 在能直连 NPU 主机的机器上（或先 ssh -L 8098:127.0.0.1:8098）
curl http://127.0.0.1:8098/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-27b-slq-b36",
  "messages":[{"role":"user","content":"你好，介绍一下你自己"}],
  "max_tokens":128}'
```

## 7. 与 BF16 的对比（同一模型、同样 TP=2）

| 项 | BF16 | SLQ b3.6 |
|---|---|---|
| 每卡权重 | 25.61 GiB | 25.61 GiB（fake-quant，尺寸未变） |
| KV cache | 23.56 GiB | 24.11 GiB |
| 同 prompt 续写 | `The capital of China is` → 重复循环 | → *"Beijing, which is also the largest city..."*（更稳） |
| 预热后解码 | ~13 tok/s（短请求） | ~28–30 tok/s |

> 两者都连贯可用；`SLQ b3.6` 的显存/存储收益尚未体现，因为当前是 fake-quant（BF16 落在整数格点）。

## 8. 偏差与限制

1. **校准 token 数**：本次用 **65536**（文档口径为 262144），以缩短重跑时间；方法一致、超参略降。
2. **fake-quant**：权重仍是 BF16，17.26 GB / 每卡 8.1 GiB 是 INT 打包后的理论值，需要 humming 或自研 INT 加载路径。
3. **MTP 头**：checkpoint 未导出 15 个 `mtp.*` 张量（多 token 预测头），常规推理不受影响，但不能直接用 MTP 投机解码。
4. **首个请求慢**：见 §5 的图编译说明。

## 9. 复现

```bash
# 一键（自愈，断点续跑，完成后自动拼装/校验/上线/HTTP 验证）
nohup bash /opt/zh/slq-official/deploy/pipeline_27b.sh > /dev/null 2>&1 &

# 分步
bash scripts/run_slq27b.sh                # ① GPTQ 全位宽
bash scripts/build_ckpt.sh                # ④ 拼装 b3.6 checkpoint
bash scripts/verify_fakequant.sh          # 指纹校验
bash deploy/serve_b36.sh                  # vLLM TP=2 上线（8098）
bash deploy/http_validate.sh 8098 qwen3.8-27b-slq-b36
```

`pipeline_27b.sh` 在失败时会丢弃半写的层目录、换卡并从已完成块继续，适合长时间无人值守运行。
