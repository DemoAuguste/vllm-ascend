#!/bin/bash
# Zero-shot evaluation of an existing model (BF16 reference, uniform anchor or
# mixed-precision checkpoint).
#
# Usage: run_zs.sh [limit] <spec>...
#   spec: bf16 | b34 | b36 | b38 | uniform2 | uniform4 | uniform6 | /absolute/path
#   e.g.  run_zs.sh 500 bf16 b36 b38
set -euo pipefail
. "$(dirname "$0")/common.sh"

LIMIT="${1:-500}"
shift || true
if [ "$#" -eq 0 ]; then
    set -- bf16 b36
fi

for SPEC in "$@"; do
    case "$SPEC" in
        bf16)
            MODEL="$SLQ_MODEL"
            TAG="bf16"
            ;;
        /*)
            MODEL="$SPEC"
            TAG="$(basename "$SPEC")"
            ;;
        *)
            MODEL="$SLQ_CKPT/$SLQ_SNAPSHOT/$SPEC"
            TAG="$SPEC"
            ;;
    esac
    python -u eval/zeroshot_eval.py \
        --model "$MODEL" --device npu:0 --limit "$LIMIT" --tasks "$SLQ_TASKS" \
        --out "$SLQ_RESULTS/zs27b_${TAG}_${LIMIT}.json"
done
echo ZS_DONE
