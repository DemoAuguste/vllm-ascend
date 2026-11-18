#!/bin/bash
# 0.6B smoke: task-lossless search. The 0.6B model is far outside the near-lossless
# regime, so the guard is expected to fail -- this only proves the search runs.
#
# Usage: run_tl06b.sh [calibration_bitwidth] [calibration_recovery]
#        run_tl06b.sh 4 0.90      # default, 4-bit anchor
#        run_tl06b.sh 6 0.96      # 6-bit anchor (the "tl6" variant)
set -euo pipefail
. "$(dirname "$0")/common.sh"

CAL="${1:-4}"
REC="${2:-0.90}"
LAYERS="$SLQ_ROOT/data/slq_qwen3_06b/Qwen3-0.6B/4bit"

python -u slq_search_dl.py \
    --model Qwen/Qwen3-0.6B \
    --layer-dir "$LAYERS" \
    --mode tl \
    --sensitivity-method shapley \
    --calibration-bitwidth "$CAL" \
    --calibration-recovery "$REC" \
    --min-recovery 0.99 \
    --bitwidths 2,4,6,8 \
    --bitwidth-tolerance 0.05 \
    --calibration-data data/calib_qwen3_1m_2048.pt \
    --calibration-samples 8 \
    --calibration-seq-length 2048 \
    --batch-size 4 \
    --top-k 10 \
    --output "$LAYERS/config_tl_cal${CAL}.txt"
echo TL06B_DONE
