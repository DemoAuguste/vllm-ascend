#!/bin/bash
# Serve an SLQ fake-quant Qwen3.8-27B checkpoint with the vllm-ascend
# OpenAI-compatible HTTP API (vLLM 0.30.0 image, TP=2).
#
#   MODEL_DIR=/opt/zh/slq-official/ckpt_27b/<snap>/b36 bash deploy/serve_27b.sh
#
# Environment:
#   MODEL_DIR            host path of the checkpoint (required)
#   MODEL_IN_CONTAINER   path the container loads (default /work/<basename>);
#                        needed for a symlink tree such as the HF cache
#   SERVED_NAME, TP, DEVS, PORT, MAX_LEN, GPU_MEM, LOG, IMAGE
#   HCCL_IFACE           HCCL/GLOO interface (default: detected from the route)
#   HCCL_BASE_PORT       HCCL base port (default 63000, avoid clashing with peers)
#
# Notes learned on the A2 host:
#   - Ascend allows only non-contiguous device mappings there, e.g.
#     DEVS=0,2 -> `--device /dev/davinci0 --device /dev/davinci2`; mapping a card
#     held by another namespace makes the container see zero devices.
#   - TP>1 needs HCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME explicitly.
#   - Use the v0.30.0 image: nightly main (vLLM 0.33) cannot load the
#     Qwen3_5ForConditionalGeneration architecture.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?path to the checkpoint dir on the host}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b-slq-b36}"
TP="${TP:-2}"
DEVS="${DEVS:-0,2}"
PORT="${PORT:-8098}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_MEM="${GPU_MEM:-0.85}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.30.0}"
LOG="${LOG:-$(pwd)/serve_27b_tp${TP}_${PORT}.log}"
NAME="slq-serve-27b-tp${TP}-${PORT}"
DETECTED_IFACE="$(ip route get 1.1.1.1 2>/dev/null | awk '{print $5; exit}')"

[ -d "$MODEL_DIR" ] || { echo "model dir not found: $MODEL_DIR" >&2; exit 1; }
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
HOST_DIR="$(dirname "$MODEL_DIR")"
CONT_MODEL="${MODEL_IN_CONTAINER:-/work/$(basename "$MODEL_DIR")}"

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

DEV_ARGS=""
IFS=',' read -ra DARR <<< "$DEVS"
for d in "${DARR[@]}"; do
    DEV_ARGS="$DEV_ARGS --device /dev/davinci${d}"
done

docker run -d \
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
    -v "$HOST_DIR":/work \
    -w /work \
    -e OMP_NUM_THREADS=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HCCL_SOCKET_IFNAME="${HCCL_IFACE:-$DETECTED_IFACE}" \
    -e GLOO_SOCKET_IFNAME="${HCCL_IFACE:-$DETECTED_IFACE}" \
    -e HCCL_IF_BASE_PORT="${HCCL_BASE_PORT:-63000}" \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_USE_V1=1 \
    "$IMAGE" \
    vllm serve "$CONT_MODEL" \
        --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size "$TP" \
        --host 0.0.0.0 --port "$PORT" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_MEM" \
        --trust-remote-code > /dev/null

: > "$LOG"
nohup docker logs -f "$NAME" >> "$LOG" 2>&1 &
echo "[serve] container $NAME started"
echo "[serve] host model:      $MODEL_DIR"
echo "[serve] container model: $CONT_MODEL"
echo "[serve] endpoint: http://0.0.0.0:$PORT/v1  (model name: $SERVED_NAME)"
echo "[serve] log: $LOG"
