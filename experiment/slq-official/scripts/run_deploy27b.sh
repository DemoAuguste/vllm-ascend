#!/bin/bash
# Stage ⑤: launch a vllm-ascend multi-card deployment smoke for a local (fake-quant)
# checkpoint. The generator runs inside the official image; the smoke script is
# mounted from this directory, the checkpoint directory is mounted as /work.
#
# Usage: run_deploy27b.sh <MODEL_PATH> [TP] [DEVICES] [EAGER] [LOG]
#        run_deploy27b.sh ckpt_27b/<snap>/b36 2 5,7 1
#
# Notes:
#   - VLLM_WORKER_MULTIPROC_METHOD=spawn and OMP_NUM_THREADS=1 avoid the fork
#     thread-pool assertion inside the container.
#   - A fake-quant checkpoint still loads as BF16, so the memory figures of this
#     smoke do not yet reflect the INT packing gain.
set -euo pipefail

MODEL_PATH="${1:?model path required}"
TP="${2:-2}"
DEVS="${3:-5,7}"
EAGER="${4:-1}"
LOG="${5:-$(pwd)/deploy27b_tp${TP}.log}"
IMAGE="${SMOKE_IMAGE:-quay.io/ascend/vllm-ascend:nightly-main}"
NAME="slq-deploy27b-tp${TP}"

HERE="$(cd "$(dirname "$0")" && pwd)"
SMOKE_SCRIPT="$HERE/smoke_llm.py"
[ -f "$SMOKE_SCRIPT" ] || { echo "missing smoke script: $SMOKE_SCRIPT" >&2; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "model directory not found: $MODEL_PATH" >&2; exit 1; }

MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
HOST_DIR="$(dirname "$MODEL_PATH")"
CONT_MODEL="/work/$(basename "$MODEL_PATH")"
mkdir -p "$(dirname "$LOG")"

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

DEV_ARGS=""
IFS=',' read -ra DARR <<< "$DEVS"
for d in "${DARR[@]}"; do
    DEV_ARGS="$DEV_ARGS --device /dev/davinci${d}"
done

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
    -v "$HOST_DIR":/work \
    -v "$SMOKE_SCRIPT":/smoke/smoke_llm.py:ro \
    -w /work \
    -e OMP_NUM_THREADS=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e SMOKE_TP="$TP" -e SMOKE_TAG="deploy27b-tp${TP}" \
    -e SMOKE_MODEL="$CONT_MODEL" \
    -e SMOKE_EAGER="$EAGER" \
    -e SMOKE_CHAT=1 \
    -e SMOKE_GPU_MEM=0.85 \
    -e SMOKE_MAX_LEN=4096 \
    -e SMOKE_MAX_TOKENS=64 \
    -e HF_ENDPOINT=https://hf-mirror.com -e VLLM_USE_V1=1 \
    "$IMAGE" \
    python /smoke/smoke_llm.py \
    > "$LOG" 2>&1 &
echo "[run] launched $NAME (TP=$TP, devices=$DEVS, model=$CONT_MODEL)"
echo "[run] log: $LOG"
echo "[run] follow with: docker logs -f $NAME"
