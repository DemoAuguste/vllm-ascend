#!/bin/bash
# Stage ④ + evaluation: assemble the fake-quant checkpoint for each average-bitwidth
# budget and score it with the log-likelihood zero-shot evaluator.
#
# Usage: run_build_eval.sh [limit] <budget>...     e.g. run_build_eval.sh 500 3.6
#                                                   run_build_eval.sh 200 3.8 3.6
set -euo pipefail
. "$(dirname "$0")/common.sh"

LIMIT="${1:-200}"
shift || true
if [ "$#" -eq 0 ]; then
    set -- 3.8 3.6
fi

for BUDGET in "$@"; do
    SUB="b${BUDGET/./}"
    python -u slq_build_checkpoint.py \
        --model "$SLQ_MODEL" \
        --layer-dir "$SLQ_LAYERS" \
        --config "$(slq_config "config_tl27b_b${BUDGET}.txt")" \
        --output-dir "$SLQ_CKPT" \
        --subdir-name "$SUB" \
        --dtype bfloat16 --quant-format fake --group-size 128 --force
    python -u eval/zeroshot_eval.py \
        --model "$SLQ_CKPT/$SLQ_SNAPSHOT/$SUB" \
        --device npu:0 --limit "$LIMIT" --tasks "$SLQ_TASKS" \
        --out "$SLQ_RESULTS/zs27b_${SUB}_${LIMIT}.json"
done
echo BUILD_EVAL_DONE
