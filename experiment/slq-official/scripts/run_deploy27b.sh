#!/bin/bash
# Launch a vllm-ascend multi-card deployment smoke for a local (fake-quant) checkpoint.
# Usage: run_deploy27b.sh <MODEL_PATH> <TP> <DEVICES> [EAGER] [PORT]
set -e
MODEL_PATH="${1:?model path required}"
TP="${2:-2}"
DEVS="${3:-5,7}"
EAGER="${4:-1}"
PORT="${5:-8100}"
IMAGE="${SMOKE_IMAGE:-quay.io/ascend/vllm-ascend:nightly-main}"
NAME=slq-deploy27b-tp${TP}
WORK=/opt/zh/SLQ-gptq
LOG=$WORK/deploy27b_tp${TP}.log

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

DEV_ARGS=""
IFS=',' read -ra DARR <<< "$DEVS"
for d in "${DARR[@]}"; do DEV_ARGS="$DEV_ARGS --device /dev/davinci${d}"; done

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
    -e OMP_NUM_THREADS=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e SMOKE_TP="$TP" -e SMOKE_TAG="deploy27b-tp${TP}" \
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
echo "[run] launched $NAME (TP=$TP, devices=$DEVS, model=$CONT_MODEL)"
echo "[run] log: $LOG"
