# Qwen3.8-27B（SLQ b3.6 量化）在 NPU 环境部署指南

> 目标模型：HuggingFace 仓库 [`DemoAuguste/Qwen3.8-27B-Quant`](https://huggingface.co/DemoAuguste/Qwen3.8-27B-Quant)
> 方法：SLQ b3.6 混合精度（GPTQ，fake-quant BF16 权重），vLLM-Ascend 提供 OpenAI 兼容 HTTP 服务
> 本文所有命令与预期输出均来自一次真实部署（Ascend 910B3 × 2，镜像 `quay.io/ascend/vllm-ascend:v0.30.0`）

## 0. 一页速览

```
① 前置检查（驱动/CANN/镜像/卡/端口）
② hf download 拉取 63 个分片 + 配置（≈51 GiB）→ 校验 sha256
③ docker run vLLM-Ascend（TP=2，端口 8098）→ 等 /health 200
④ curl 调 /v1/chat/completions 验证
```

| 关键事实 | 值 |
|---|---|
| 仓库文件数 | **81**（63 个 `model-*-of-00064.safetensors` + 18 个配置/分词器/文档/日志） |
| 模型体积 | 本地 ≈ **51 GiB**；HF `usedStorage` 54.84 GB（十进制） |
| 张量数 / 总字节 | 1184 个张量 / `total_size = 54,713,457,120` 字节（见 `model.safetensors.index.json`） |
| 分片命名 | `model-00002-of-00064` … `model-00064-of-00064`（**编号 1 不存在**，共 63 个；以 index 引用为准） |
| 权重类型 | fake-quant BF16（按整数格点量化后又反量化回 BF16，**体积与 BF16 相同**） |
| 已验证运行环境 | Ascend 910B3 × 2（TP=2）、CANN 9.2.1、镜像 vLLM-Ascend **v0.30.0** |
| 每卡权重 / KV cache | **25.6118 GB** / **23.56 GiB** |
| 引擎初始化 | 168 s（含 68 s 图编译） |

---

## 1. 前置条件

### 1.1 硬件与操作系统

| 项 | 要求 | 说明 |
|---|---|---|
| NPU | Ascend **910B**（A2）≥ 2 张 | TP=2 部署；单卡 HBM ≥ 64 GB（每卡权重 25.6 GB + 激活/KV） |
| 卡状态 | 至少 2 张 `Health=OK` 且空闲 | 共享集群需先确认没有别人占用 |
| 操作系统 | openEuler 24.03（已验证）或其他 CANN 9.2.x 支持的系统 | — |
| 磁盘 | `/opt` 或模型盘空闲 ≥ **120 GB** | 51 GiB 模型 + 镜像/临时空间 |
| 端口 | 一个空闲 TCP 端口（示例 8098） | 不要与 8080/8000 等已有服务冲突 |

> 其他型号（910C/A3、950/A5）需改用对应架构的 vLLM-Ascend 镜像并重新验证；本文参数不能直接照搬。

### 1.2 软件版本（镜像内置，无需宿主安装 Python 依赖）

部署使用容器镜像 **`quay.io/ascend/vllm-ascend:v0.30.0`**，镜像内已包含：

| 组件 | 版本 |
|---|---|
| CANN | 9.2.1（`/usr/local/Ascend/cann-9.2.1`） |
| vLLM | 0.30.0 |
| transformers | 5.14.1 |
| torch / torch_npu | 2.10.0 (+cpu) / 2.10.0.post7 |

宿主侧只需要：**驱动 + 固件与 CANN 9.2.1 兼容**、`docker` 可用、`/usr/local/Ascend/driver` 等设备文件存在。

> ⚠️ 不要用 `nightly-main`（vLLM 0.33 + transformers 5.19）：在已验证环境里它无法加载本模型的
> `Qwen3_5ForConditionalGeneration` 架构（模型注册阶段报错）。

### 1.3 检查命令（逐条执行，含预期输出）

```bash
# 1) NPU 与驱动：能看到 8 张 910B3，Health=OK
npu-smi info
# 预期（节选）：
# | 0     910B3               | OK            | ...
# | 0                         | 0000:C1:00.0  | 0           0    / 0          3765 / 65536

# 2) 驱动/CANN 安装信息
cat /etc/ascend_install.info
# 预期：包含 driver version 与 firmware version 两行

# 3) 镜像是否已就绪（没有则先 docker pull，见 1.4）
docker images | grep vllm-ascend
# 预期：quay.io/ascend/vllm-ascend   v0.30.0   ...   23.9GB

# 4) 宿主资源
free -g | head -2          # 建议 available ≥ 64 GB（加载权重时用内存做中转）
df -h /opt                 # 建议 ≥ 120 GB 可用

# 5) 端口占用（8098 应无输出）
ss -lntp | grep 8098 || echo "port 8098 free"

# 6) 到 HuggingFace 的网络（下载模型用；不通见 2.6）
curl -sI --max-time 15 https://huggingface.co | head -1
# 预期：HTTP/2 200
```

### 1.4 拉取镜像（如本地没有）

```bash
docker pull quay.io/ascend/vllm-ascend:v0.30.0
docker images | grep vllm-ascend        # 预期能看到 v0.30.0
```

国内网络慢可在能访问 HF 的机器上 `docker save` 后离线导入：

```bash
# 有网机器
docker save quay.io/ascend/vllm-ascend:v0.30.0 -o vllm-ascend-v0.30.0.tar
# 目标机器
docker load -i vllm-ascend-v0.30.0.tar
```

### 1.5 客户端工具

- 服务端：`curl`（几乎所有系统自带）
- 可选：Python `openai` SDK（`pip install openai`）

---

## 2. 下载模型文件

### 2.1 安装/升级 HF CLI

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf --version            # 新 CLI 命令为 hf；旧版仍可用 huggingface-cli
# 预期：hf version x.y.z
```

> 下文用新 CLI `hf`。若你的版本没有 `hf`，把 `hf` 换成 `huggingface-cli`，参数相同。

### 2.2 登录（仓库是公开的，可跳过；限流/私有镜像站建议登录）

```bash
hf auth login          # 浏览器设备码登录；或 hf auth login --token $HF_TOKEN
hf auth whoami         # 预期：Logged in / user: <你的账号或空>
```

公开仓库**不需要** token 也能下载；若遇到 401/403 或限流再登录。

### 2.3 下载（推荐：指定本地目录 + 断点续传 + 并行）

```bash
mkdir -p /opt/models
cd /opt/models

# 全量下载（63 个分片 + 配置），自动校验 LFS sha256，可重复执行续传
hf download DemoAuguste/Qwen3.8-27B-Quant \
  --local-dir /opt/models/Qwen3.8-27B-Quant
```

加速/续传说明：

```bash
# 并行下载（新版本 CLI 支持 --max-workers，默认 8；网络好可调大）
hf download DemoAuguste/Qwen3.8-27B-Quant \
  --local-dir /opt/models/Qwen3.8-27B-Quant --max-workers 16

# 中断后：原样重跑同一条命令即可续传（已完成文件自动跳过）
# 只缺部分文件时，也可只下载指定分片
hf download DemoAuguste/Qwen3.8-27B-Quant \
  model-00002-of-00064.safetensors --local-dir /opt/models/Qwen3.8-27B-Quant

# hf_transfer 高速传输后端（可选）
python3 -m pip install -U hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download DemoAuguste/Qwen3.8-27B-Quant \
  --local-dir /opt/models/Qwen3.8-27B-Quant
```

> `hf download` 会在下载时对每个 LFS 文件做 sha256 校验，校验失败的块会自动重试；仍失败时删除该文件重下即可（见 §6.1）。

### 2.4 下载后的目录结构（预期）

```
/opt/models/Qwen3.8-27B-Quant/
├── model-00002-of-00064.safetensors      # 47 GB 级分片，共 63 个
│   ...
├── model-00064-of-00064.safetensors
├── model.safetensors.index.json          # 1184 个张量 → 分片映射（权威清单）
├── config.json                           # architectures: Qwen3_5ForConditionalGeneration
├── generation_config.json
├── chat_template.jinja
├── tokenizer.json / tokenizer_config.json / vocab.json / merges.txt / crc32.txt
├── preprocessor_config.json              # VLM 图像预处理（缺失会导致加载失败）
├── video_preprocessor_config.json
├── quantization_config.txt               # SLQ 位宽分配记录
├── quantization_info.json                # 机器可读量化摘要
├── README.md                             # 模型卡
└── validation/                           # 上传时的校验日志
    ├── verify_b36.log
    ├── http_validation.log
    └── finalize_state.txt
```

核对文件数：

```bash
cd /opt/models/Qwen3.8-27B-Quant
ls model-*-of-00064.safetensors | wc -l
# 预期：63

du -sh .
# 预期：约 51G（十进制约 54.8 GB）
```

### 2.5 完整性校验（三层，必做）

**第 1 层：文件数量**

```bash
ls model-*-of-00064.safetensors | wc -l     # 预期 63
```

**第 2 层：index 与实际文件一一对应**（最可靠，别假设编号连续）

```bash
python3 - <<'PY'
import json, os
d = "."
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
missing = [f for f in files if not os.path.exists(os.path.join(d, f))]
print("index shards:", len(files))
print("tensors:", len(idx["weight_map"]))
print("total_size:", idx["metadata"]["total_size"])   # 预期 54713457120
print("missing on disk:", missing)                     # 预期 []
PY
# 预期输出：
# index shards: 63
# tensors: 1184
# total_size: 54713457120
# missing on disk: []
```

**第 3 层：sha256 与远端 LFS 对象逐一比对**（最强校验）

```bash
python3 - <<'PY'
import hashlib, json, os, urllib.request

REPO = "DemoAuguste/Qwen3.8-27B-Quant"
DIR = "."

def api(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)

# 分页拉取远端文件树（含 LFS oid = sha256）
entries, url = [], f"https://huggingface.co/api/models/{REPO}/tree/main?recursive=true&expand=true"
while url:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as r:
        entries.extend(json.load(r))
        link = r.headers.get("Link", "")
    nxt = [p.split(">")[0].strip("<") for p in link.split(",")
           if 'rel="next"' in p]
    url = nxt[0] if nxt else None

remote = {e["path"]: e["lfs"]["oid"] for e in entries
          if e.get("type") == "file" and "lfs" in e}
print("remote lfs files:", len(remote))

ok = bad = miss = 0
for name, oid in sorted(remote.items()):
    if not name.endswith(".safetensors"):
        continue
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        miss += 1
        print("MISSING", name)
        continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() == oid:
        ok += 1
    else:
        bad += 1
        print("MISMATCH", name)
print(f"RESULT matched={ok} mismatched={bad} missing={miss}")
PY
# 预期最后一行：RESULT matched=63 mismatched=0 missing=0
```

> 全部 51 GiB 计算 sha256 约需数分钟（受磁盘速度影响）。

### 2.6 网络受限环境（下载失败/极慢时）

| 情况 | 处理 |
|---|---|
| 无法访问 huggingface.co | 用镜像站：`HF_ENDPOINT=https://hf-mirror.com hf download DemoAuguste/Qwen3.8-27B-Quant --local-dir ...`。注意：镜像站对新上传仓库可能有缓存延迟，失败时回退直连或离线拷贝 |
| 完全离线 | 在能下载的机器上下载后打包：`tar -czf qwen27b-ckpt.tgz Qwen3.8-27B-Quant`，拷贝到目标机解包（保持目录结构） |
| 已有部署机（如已验证主机） | 直接 `scp -r`/`rsync` 整个目录到目标机，再做 §2.5 校验 |

`rsync` 拷贝可断点续传：

```bash
rsync -avP --partial root@<已部署主机>:/opt/models/Qwen3.8-27B-Quant/ \
  /opt/models/Qwen3.8-27B-Quant/
```

---

## 3. 启动部署

### 3.1 选择卡与设备映射

```bash
npu-smi info        # 选 2 张 Health=OK 且 HBM 占用低的卡，例如 0、1
```

容器需要映射 **对应 davinci 设备 + manager/hdc/devmm_svm**。先做一次“容器内可见卡数”自检：

```bash
DEV0=0; DEV1=1         # 换成你选的两张卡号
docker run --rm \
  --device /dev/davinci${DEV0} --device /dev/davinci${DEV1} \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
  --entrypoint python quay.io/ascend/vllm-ascend:v0.30.0 \
  -c 'import torch, torch_npu; print("devices:", torch.npu.device_count())'
# 预期：devices: 2
```

若输出 `devices: 0`：说明映射的卡被其他容器/命名空间占用，换卡重试；共享集群上有时需要**非连续映射**
（我们验证过的环境用 `--device /dev/davinci0 --device /dev/davinci2` 才可见 2 张卡）。判定方法：

```bash
dmesg -T | grep -i uda_occupy | tail -3
# 若出现 "Conflict open udevid=..."，说明该卡被别的 namespace 占用，换一张空闲卡
```

### 3.2 HCCL 网卡（TP>1 必须显式指定）

```bash
IFACE=$(ip route get 1.1.1.1 | awk '{print $5; exit}')
echo "IFACE=$IFACE"       # 例：enp189s0f0
```

不指定会报：`hcclGetRootInfo ... error code is 19 (call network api fail)`。

### 3.3 完整启动命令

```bash
MODEL_DIR=/opt/models/Qwen3.8-27B-Quant
CONTAINER=slq-serve-qwen27b
PORT=8098
DEV0=0; DEV1=1                     # 换成 §3.1 选中的两张卡
IFACE=$(ip route get 1.1.1.1 | awk '{print $5; exit}')
LOG=/var/log/slq-serve-qwen27b.log

docker run -d \
  --name "$CONTAINER" \
  --net=host --ipc=host --shm-size=64g \
  --device /dev/davinci${DEV0} --device /dev/davinci${DEV1} \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
  -v /etc/hixlep/:/etc/hixlep/ \
  -v "$MODEL_DIR":/work/model \
  -w /work \
  -e OMP_NUM_THREADS=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e HCCL_SOCKET_IFNAME="$IFACE" \
  -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e HCCL_IF_BASE_PORT=63000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  quay.io/ascend/vllm-ascend:v0.30.0 \
  vllm serve /work/model \
    --served-model-name qwen3.8-27b-slq-b36 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code

# 日志跟到文件（-d 模式下 vLLM 日志在 docker logs 里）
nohup docker logs -f "$CONTAINER" > "$LOG" 2>&1 &
```

参数说明：

| 参数 | 必填 | 说明 |
|---|---|---|
| `--name` | 建议 | 固定容器名，便于重启/停止 |
| `--net=host` | 建议 | 端口直接暴露在宿主；也可改成 `-p 8098:8098` |
| `--device davinciN` | 必填 | 选中的卡；TP=2 就映射 2 张 |
| `--device davinci_manager/hisi_hdc/devmm_svm` | 必填 | Ascend 运行时必需 |
| `-v /usr/local/Ascend/driver ...` | 必填 | 驱动/设备信息透传 |
| `-v MODEL_DIR:/work/model` | 必填 | 模型以只读方式挂进容器（本命令未加 `:ro`，可自行加） |
| `OMP_NUM_THREADS=1` | 必填 | 避免容器内线程超订 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 必填 | 避免 fork 线程池断言崩溃 |
| `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | TP>1 必填 | 用 §3.2 探测到的网卡 |
| `HCCL_IF_BASE_PORT` | 建议 | 多机/多容器时避免端口冲突 |
| `HF_HUB_OFFLINE=1` | 建议 | 模型已在本地，禁止联网检查 |
| `--served-model-name` | 建议 | 请求体里的 `model` 字段必须与它一致 |
| `--tensor-parallel-size 2` | 必填 | 2 卡 TP |
| `--max-model-len 4096` | 建议 | 上下文长度；越大 KV 显存占用越高 |
| `--gpu-memory-utilization 0.85` | 建议 | 显存利用率；OOM 时降到 0.8 |
| `--trust-remote-code` | 必填 | 模型使用自定义实现 |

### 3.4 后台常驻与日志

```bash
docker ps --filter name=slq-serve-qwen27b --format '{{.Names}} {{.Status}}'
# 预期：slq-serve-qwen27b Up ...

docker logs --tail 50 slq-serve-qwen27b      # 实时：docker logs -f
tail -f /var/log/slq-serve-qwen27b.log       # 若按 §3.3 做了日志重定向
```

### 3.5 启动成功判定

按顺序出现以下日志（时间仅供参考，取决于机器）：

```text
(Worker_TP0 ...) Loading model weights took 25.6118 GB
(Worker_TP0 ...) Available KV cache memory: 23.56 GiB
(EngineCore ...) init engine (profile, create kv cache, warmup model) took 168.00 s (compilation: 67.89 s)
(APIServer ...) Application startup complete.
```

然后健康检查返回 200：

```bash
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8098/health)
  echo "[$i] health=$code"; [ "$code" = "200" ] && break; sleep 10
done
# 预期最终：[n] health=200
```

> 冷启动约 3 分钟（加载 51 GiB + 图编译）；期间 `/health` 可能是 000/503，属正常。

### 3.6 冷启动后的“首请求偏慢”是正常现象

NPU 上 vLLM 会为不同请求形状做图编译。实测：**服务启动后的头 1–3 个不同形状的请求可能耗时 194–300 s**，
预热后同样请求约 4 s（128 tokens）。建议上线后先发一次 warmup 请求：

```bash
curl -s http://127.0.0.1:8098/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-slq-b36","messages":[{"role":"user","content":"ping"}],"max_tokens":16}' \
  -o /dev/null -w 'warmup=%{http_code} time=%{time_total}s\n'
```

---

## 4. 接口调用

### 4.1 协议：**是 OpenAI 兼容接口**（OpenAI-compatible HTTP API）

- Base URL：`http://<NPU主机IP>:8098/v1`
- 模型名（`model` 字段）：**`qwen3.8-27b-slq-b36`**（即启动参数 `--served-model-name`）
- 鉴权：本次未设置 `--api-key`，因此**无需 Authorization**；如需鉴权，启动时加 `--api-key <key>`，请求加 `-H 'Authorization: Bearer <key>'`
- 传输：普通 JSON；流式用 `"stream": true`（SSE，`data: {...}` 逐块）

### 4.2 端点一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 健康检查（就绪返回 200） |
| GET | `/v1/models` | 列出已加载模型 |
| POST | `/v1/chat/completions` | 对话（含 system/user 消息，支持流式） |
| POST | `/v1/completions` | 文本补全（纯 prompt，支持流式） |
| GET | `/metrics` | Prometheus 指标（如启用） |
| POST | `/v1/responses` | 新版 Responses API（vLLM 0.30 已提供，可选） |

### 4.3 curl 示例（可直接复制）

**列出模型**

```bash
curl -s http://127.0.0.1:8098/v1/models | python3 -m json.tool
# 预期（节选）：
# "id": "qwen3.8-27b-slq-b36",
# "max_model_len": 4096,
# "root": "/work/model"
```

**对话（非流式）**

```bash
curl -s http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-slq-b36",
    "messages": [
      {"role": "system", "content": "你是一个简洁的中文助手。"},
      {"role": "user", "content": "用一句话说明什么是混合精度量化。"}
    ],
    "temperature": 0,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | python3 -m json.tool
```

响应体结构（节选）：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "qwen3.8-27b-slq-b36",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "混合精度量化是……"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 20, "completion_tokens": 42, "total_tokens": 62}
}
```

**文本补全**

```bash
curl -s http://127.0.0.1:8098/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-slq-b36","prompt":"The capital of China is","temperature":0,"max_tokens":24}' \
  | python3 -m json.tool
# 预期 choices[0].text 形如：" Beijing, which is also the largest city in the country. ..."
```

**流式（SSE）**

```bash
curl -sN http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-slq-b36","messages":[{"role":"user","content":"列出三个中国省份，用顿号分隔。"}],
       "temperature":0,"max_tokens":32,"stream":true,
       "chat_template_kwargs":{"enable_thinking":false}}'
# 预期逐行输出：
# data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"delta":{"role":"assistant","content":""}}
# data: {...,"delta":{"content":"北京"}}
# ...
# data: [DONE]
```

### 4.4 Python（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8098/v1", api_key="EMPTY")  # 未设 --api-key 时 api_key 可任意

resp = client.chat.completions.create(
    model="qwen3.8-27b-slq-b36",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
    temperature=0,
    max_tokens=128,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

### 4.5 常用请求参数

| 参数 | 说明 |
|---|---|
| `model` | 必须为 `qwen3.8-27b-slq-b36` |
| `messages` | `role` ∈ system/user/assistant；内容为字符串 |
| `temperature` / `top_p` | 采样参数；确定性验证用 `temperature: 0` |
| `max_tokens` | 最大生成 token 数 |
| `stream` | `true` 时 SSE 逐块返回 |
| `chat_template_kwargs.enable_thinking` | Qwen3.8 默认开启"思考"；设为 `false` 可只输出最终答案（更快、更短） |

---

## 5. 验证

### 5.1 健康检查

```bash
curl -s -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8098/health
# 预期：health=200
```

### 5.2 推理验证（判定服务真正可用）

```bash
curl -s http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-slq-b36","messages":[{"role":"user","content":"用一句话说明什么是混合精度量化。"}],
       "temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

判定标准：

1. HTTP 200，且响应 JSON 有 `choices[0].message.content`；
2. 内容语义正确、中文通顺（不应是乱码/重复循环/空字符串）；
3. `usage.completion_tokens > 0`。参考：128 tokens 预热后约 **4.3 s**；
4. 换一个英文 prompt 再验一次（确认不是碰巧）：

```bash
curl -s http://127.0.0.1:8098/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-slq-b36","prompt":"The capital of China is","temperature":0,"max_tokens":24}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
# 预期包含 "Beijing"
```

### 5.3 确认服务加载的是本量化模型

```bash
# 1) 模型 id 与 root 路径
curl -s http://127.0.0.1:8098/v1/models | python3 -m json.tool | grep -E '"id"|"root"|"max_model_len"'

# 2) 日志里每卡权重应约 25.6118 GB（fake-quant BF16）
docker logs slq-serve-qwen27b 2>&1 | grep -E 'Loading model weights took|Available KV cache'

# 3) 可选：本地 checkpoint 的 sha256 已在 §2.5 与 HF 仓库比对通过
```

### 5.4 性能基线（预热后，TP=2 / 910B3）

| 指标 | 实测 |
|---|---|
| 128 tokens 对话 | ≈ 4.3 s（约 28–30 tok/s 解码） |
| 117 tokens 生成 | ≈ 3.9 s |
| 每卡权重 / KV | 25.6118 GB / 23.56 GiB（320,605 tokens，4096 长度并发约 78×） |
| 引擎初始化 | 168 s |

> 注意：本 checkpoint 是 fake-quant（BF16 权重），**显存与磁盘不会比 BF16 小**；
> 3.17× 压缩是 INT 打包后的理论值，需要额外的 INT 加载/打包实现。

---

## 6. 常见问题排查

### 6.1 分片缺失 / 校验不通过

**现象**：`ls model-*-of-00064.safetensors | wc -l` ≠ 63；或 §2.5 的 index/ sha256 检查报 `missing` / `MISMATCH`。

**处理**：

```bash
cd /opt/models/Qwen3.8-27B-Quant
# 1) 找出缺失/损坏文件（以 index 为准）
python3 - <<'PY'
import json, os
idx = json.load(open("model.safetensors.index.json"))
for f in sorted(set(idx["weight_map"].values())):
    if not os.path.exists(f):
        print("MISSING:", f)
PY

# 2) 只补缺失文件（重复执行会续传/跳过已完成）
hf download DemoAuguste/Qwen3.8-27B-Quant \
  model-000NN-of-00064.safetensors --local-dir .

# 3) 若 sha256 不匹配：删除该文件后重下
rm -f model-000NN-of-00064.safetensors
hf download DemoAuguste/Qwen3.8-27B-Quant --local-dir .
```

注意：**不要**因为“编号 1 不存在”而报错——本仓库就是 63 个分片（`model-00002` 起），以 `model.safetensors.index.json` 的引用为准。

### 6.2 加载失败（启动后容器退出）

按日志关键字定位：

| 日志关键字 | 原因 | 处理 |
|---|---|---|
| `failed to be inspected` / `Unknown model architecture` | 镜像版本不对（如用了 nightly 的 vLLM 0.33） | 换 `quay.io/ascend/vllm-ascend:v0.30.0` |
| `Can't load image processor` / `preprocessor_config.json` | 缺少 VLM 预处理文件 | 确认目录里有 `preprocessor_config.json`、`video_preprocessor_config.json`（本仓库已含） |
| `Invalid repository ID or local directory specified` | `--model` 路径不对/未挂载 | 检查 `-v` 挂载与 `vllm serve /work/model` 的路径 |
| `KeyError` / `size mismatch` / `Missing keys` | 分片损坏或不完整 | 回到 §2.5 校验并重下 |
| `torch_npu` / `Failed to load the backend extension` | 容器没映射 NPU 设备 | 回到 §3.1 确认 `torch.npu.device_count()==2` |

查看完整报错：

```bash
docker logs --tail 100 slq-serve-qwen27b 2>&1 | grep -nE 'ERROR|Traceback|Error' | tail -20
```

### 6.3 显存不足（OOM / 初始化失败）

**现象**：日志 `NPU out of memory`、`Failed to allocate`，或进程被杀。

**排查**：

```bash
npu-smi info | grep -A2 '^| [0-9] '    # 看目标卡 HBM 占用；每卡需要约 28 GB 空闲
docker logs slq-serve-qwen27b 2>&1 | grep -iE 'oom|out of memory' | tail -5
```

**处理**（按优先级）：

1. 换两张更空闲的卡（§3.1）；
2. 降低 `--gpu-memory-utilization`：`0.85 → 0.8`；
3. 降低 `--max-model-len`：`4096 → 2048`（KV 显存随长度线性下降）；
4. 加 `--enforce-eager`（关闭图模式，省约 0.8 GB 图内存，性能略降）。

> 单卡放不下（每卡权重 25.6 GB + 激活/图 ~3 GB）：**必须 TP≥2**，单卡部署会 OOM。

### 6.4 端口占用

```bash
ss -lntp | grep 8098
# 若有输出，说明端口被占：
#   LISTEN 0 128 0.0.0.0:8098 ... users:(("python",pid=1234,...))
```

处理：换端口（`--port 8099` 并同步修改探活/请求地址），或停掉占用进程：

```bash
docker ps --format '{{.Names}} {{.Ports}}' | grep 8098
docker rm -f <占用容器名>
```

### 6.5 服务无响应 / 健康检查一直非 200

按顺序排查：

```bash
# 1) 容器是否还在跑
docker ps -a --filter name=slq-serve-qwen27b

# 2) 容器内 NPU 是否可见（进容器执行）
docker exec slq-serve-qwen27b python -c 'import torch, torch_npu; print(torch.npu.device_count())'
# 预期 2；若 0 → 见 6.6

# 3) 是不是还在加载/编译（冷启动约 3 分钟）
docker logs --tail 30 slq-serve-qwen27b 2>&1 | tail -10

# 4) 是否卡在 HCCL
docker logs slq-serve-qwen27b 2>&1 | grep -i hccl | tail -5
```

若日志停在 `Application startup complete` 但请求超时：多半是**首个请求在编译图**（§3.6），等 3–5 分钟或先发 warmup 请求。

### 6.6 容器看不到 NPU（device_count=0）

**现象**：`devices: 0`，或日志报 `get_arch` / `cann_get_soc_name failed`。

**排查**：

```bash
npu-smi info                                   # 宿主能看到卡
dmesg -T | grep -i 'uda_occupy\|conflict' | tail -3
# 若出现 "Conflict open udevid=N" → 该卡被其他容器/namespace 独占
```

**处理**：

1. 换两张空闲卡；
2. 共享集群上尝试**非连续映射**（如 `--device /dev/davinci0 --device /dev/davinci2`）；
3. 确认三个必需设备都映射了：`davinci_manager`、`hisi_hdc`、`devmm_svm`；
4. 重新执行 §3.1 的自检，直到 `devices: 2` 再启动服务。

### 6.7 TP>1 启动失败：`hcclGetRootInfo ... error code is 19`

**原因**：未指定/指定错 HCCL 网卡。

**处理**：

```bash
IFACE=$(ip route get 1.1.1.1 | awk '{print $5; exit}')   # 例 enp189s0f0
# 启动命令加：
#   -e HCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e HCCL_IF_BASE_PORT=63000
```

### 6.8 首个请求超时

**现象**：`/health` 200，但第一次 `/v1/chat/completions` 超过 60 s 甚至几分钟。

**原因**：NPU 图编译（按请求形状缓存）。**处理**：先发 §3.6 的 warmup 请求；正式压测/验收在预热后开始。
若多个不同 `max_tokens`/`stream` 组合都要首次编译，建议压测前用同一组参数各打一次 warmup。

### 6.9 下载慢 / 中断

- 重跑 `hf download` 会自动续传；
- 开启 `--max-workers 16` 或 `HF_HUB_ENABLE_HF_TRANSFER=1`；
- 网络受限用 `HF_ENDPOINT=https://hf-mirror.com`（§2.6）；
- 大文件校验失败：删除对应分片重下（§6.1）。

---

## 7. 附录

### 7.1 一键脚本（保存为 `serve_qwen27b.sh`）

```bash
#!/bin/bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-/opt/models/Qwen3.8-27B-Quant}"
PORT="${PORT:-8098}"
DEV0="${DEV0:-0}"; DEV1="${DEV1:-1}"
NAME="slq-serve-qwen27b"
IFACE=$(ip route get 1.1.1.1 | awk '{print $5; exit}')

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --net=host --ipc=host --shm-size=64g \
  --device /dev/davinci${DEV0} --device /dev/davinci${DEV1} \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
  -v /etc/hixlep/:/etc/hixlep/ \
  -v "$MODEL_DIR":/work/model -w /work \
  -e OMP_NUM_THREADS=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e HCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e HCCL_IF_BASE_PORT=63000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  quay.io/ascend/vllm-ascend:v0.30.0 \
  vllm serve /work/model \
    --served-model-name qwen3.8-27b-slq-b36 \
    --tensor-parallel-size 2 --host 0.0.0.0 --port "$PORT" \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --trust-remote-code

echo "waiting for /health ..."
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")
  echo "[$i] health=$code"
  [ "$code" = "200" ] && { echo "READY"; exit 0; }
  sleep 10
done
echo "not ready, check: docker logs $NAME"; exit 1
```

### 7.2 停止 / 重启

```bash
docker rm -f slq-serve-qwen27b          # 停止并删除容器
bash serve_qwen27b.sh                    # 重新启动
```

### 7.3 关于 fake-quant 的说明

本仓库权重是 **fake-quant**：量化后的整数格点在导出时被反量化回 BF16 保存，因此：

- transformers / vLLM 按普通 BF16 模型加载，**精度行为等价于量化模型**；
- **磁盘与显存占用与 BF16 相同**（本地 51 GiB，每卡 25.6 GB）；
- 3.6 bpp / 17.26 GB / 3.17× 压缩是 **INT 打包后**的理论值，需要 `humming_online` 或自研 INT 加载路径才能落地。

---

*文档基于 2026-11-19/20 在 Ascend 910B3 × 2 + vLLM-Ascend v0.30.0 上的实际部署验证编写；
文中数值（25.6118 GB/卡、23.56 GiB KV、168 s 初始化、约 4.3 s/128 tokens）均来自该次实测日志。*
