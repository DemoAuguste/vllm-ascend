#!/bin/bash
# Serve the SLQ b3.6 fake-quant checkpoint and wait until /health is green.
#
#   bash deploy/serve_b36.sh
#   DEVS=0,2 PORT=8098 bash deploy/serve_b36.sh
#
# The first requests after startup can take minutes: vLLM compiles/captures the
# graph for each new request shape on NPU. Warm the server up before benchmarking.
set -u

SLQ_ROOT="${SLQ_ROOT:-/opt/zh/slq-official}"
SNAP="${SLQ_SNAPSHOT:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ 2>/dev/null | head -1)}"
SNAP="${SNAP%/}"
CKPT="${CKPT:-$SLQ_ROOT/ckpt_27b/$(basename "$SNAP")/b36}"
PORT="${PORT:-8098}"
TP="${TP:-2}"

[ -d "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

MODEL_DIR="$CKPT" \
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b-slq-b36}" \
TP="$TP" DEVS="${DEVS:-0,2}" PORT="$PORT" \
MAX_LEN="${MAX_LEN:-4096}" GPU_MEM="${GPU_MEM:-0.85}" \
LOG="${LOG:-$SLQ_ROOT/serve_b36_${PORT}.log}" \
bash "$(dirname "$0")/serve_27b.sh"

echo "=== waiting for /health (model load + graph capture takes a few minutes) ==="
name="slq-serve-27b-tp${TP}-${PORT}"
for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")
    if [ "$code" = "200" ]; then echo "[$i] HEALTH 200"; exit 0; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]; then
        echo "[$i] container exited"; docker logs --tail 40 "$name" 2>&1; exit 1
    fi
    sleep 15
done
echo "health never became green"; exit 2
