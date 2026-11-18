#!/bin/bash
# Uniform-bitwidth anchor checkpoint, the baseline the mixed-precision budgets are
# compared against (uniform-4 = 4.156 bpp, uniform-6 = 6.156 bpp).
#
# Usage: run_ckpt27b.sh <bits>          e.g. run_ckpt27b.sh 4
set -euo pipefail
. "$(dirname "$0")/common.sh"

BITS="${1:?bitwidth required (2|4|6)}"

python -u slq_build_checkpoint.py \
    --model "$SLQ_MODEL" \
    --layer-dir "$SLQ_LAYERS" \
    --uniform "$BITS" \
    --output-dir "$SLQ_CKPT" \
    --dtype bfloat16 --quant-format fake --group-size 128 --force
echo "CKPT27B_U${BITS}_DONE"
