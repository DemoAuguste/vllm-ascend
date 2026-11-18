#!/bin/bash
# Stage ②: sensitivity database over the layer store.
#
#   base    -> error_db.json + baseline_topk (linear alpha, no Shapley)
#   shapley -> multi-bitwidth Shapley (4 permutations; run one process per card
#              with different seeds and merge with merge_shapley.py)
#
# Usage: run_db27b.sh [base|shapley] [seed]
set -euo pipefail
. "$(dirname "$0")/common.sh"

MODE="${1:-base}"
SEED="${2:-42}"

ARGS=(
    --model "$SLQ_MODEL"
    --layer-dir "$SLQ_LAYERS"
    --calibration-data "$SLQ_DATA/calib_27b_256k_2048.pt"
    --calibration-samples 4
    --calibration-seq-length 2048
    --batch-size 2
    --skip-linear
    --layer-groups 'mlp.gate_proj,mlp.up_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj'
    --gpu-cache off
    --seed "$SEED"
)

case "$MODE" in
    base)
        python -u slq_build_database.py "${ARGS[@]}" --skip-shapley
        ;;
    shapley)
        python -u slq_build_database.py "${ARGS[@]}" \
            --shapley-bitwidths 2,4,6,8 --num-permutations 4
        ;;
    *)
        echo "usage: $0 [base|shapley] [seed]" >&2
        exit 2
        ;;
esac
echo "DB27B_${MODE}_DONE"
