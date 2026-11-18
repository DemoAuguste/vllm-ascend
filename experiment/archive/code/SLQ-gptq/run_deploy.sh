#!/bin/bash
# Launch a vllm-ascend multi-card deployment smoke for a local (fake-quant) checkpoint.
# Runs ON the NPU server. Usage: run_deploy.sh <MODEL_PATH> <TP> <DEVICES> [EAGER] [PORT]
#   MODEL_PATH : host path to HF checkpoint dir (mounted read-write into /work)
#   TP         : tensor parallel size (default 2)
#   DEVICES    : comma-separated host davinci ids (default "5,6")
#   EAGER      : 1 = eager (default), 0 = graph
set -e
MODEL_PATH="${1:?model path required}"
TP="${2:-2}"
DEVS="${3:-5,6}"
EAGER="${4:-1}"
PORT="${5:-8100}"
IMAGE="${SMOKE_IMAGE:-quay.io/ascend/vllm-ascend:nightly-main}"
NAME=slq-deploy-tp${TP}
WORK=/opt/zh/SLQ-gptq
LOG=$WORK/deploy_tp${TP}.log
rm -rf "$WORK/logs"; mkdir -p "$WORK/logs"

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

DEV_ARGS=""
IFS=',' read -ra DARR <<< "$DEVS"
for d in "${DARR[@]}"; do DEV_ARGS="$DEV_ARGS --device /dev/davinci${d}"; done

# 把 checkpoint 所在目录整体挂到 /work（保持一致路径），模型用容器内路径
HOST_DIR="$(dirname "$MODEL_PATH")"
CONT_MODEL="/work/$(basename "$MODEL_PATH")"

nohup docker run --rm \
    --name "$NAME" \
    --net=host --ipc=host --shm-size=64g \
    $DEV_ARGS \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /etc/hixlep/:/etc/hixlep/ \
    -v /root/.cache:/root/.cache \
    -v "$HOST_DIR":/work -w /work \
    -e SMOKE_TP="$TP" -e SMOKE_TAG="deploy-tp${TP}" \
    -e SMOKE_MODEL="$CONT_MODEL" \
    -e SMOKE_EAGER="$EAGER" \
    -e SMOKE_CHAT=1 \
    -e SMOKE_GPU_MEM=0.85 \
    -e SMOKE_MAX_LEN=4096 \
    -e SMOKE_MAX_TOKENS=64 \
    -e SMOKE_PORT="$PORT" \
    -e HF_ENDPOINT=https://hf-mirror.com -e VLLM_USE_V1=1 \
    "$IMAGE" \
    python /work/scripts/smoke_llm.py \
    > "$LOG" 2>&1 &
echo "[run] launched $NAME (TP=$TP, host devices=$DEVS, model=$HOST_DIR -> $CONT_MODEL)"
echo "[run] log: $LOG"
