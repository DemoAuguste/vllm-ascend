#!/bin/bash
# 0.6B smoke: linear-alpha sensitivity database (no Shapley).
#
# Usage: run_db06b.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"

python -u slq_build_database.py \
    --model Qwen/Qwen3-0.6B \
    --layer-dir data/slq_qwen3_06b/Qwen3-0.6B/4bit \
    --calibration-data data/calib_qwen3_1m_2048.pt \
    --calibration-samples 32 \
    --calibration-seq-length 2048 \
    --batch-size 4 \
    --top-k 10 \
    --seed 42 \
    --num-noise-samples 8 \
    --skip-shapley \
    --layer-groups 'mlp.gate_proj,mlp.up_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj'
echo DB_DONE
