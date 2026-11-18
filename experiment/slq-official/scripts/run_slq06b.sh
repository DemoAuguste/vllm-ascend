#!/bin/bash
# 0.6B smoke: GPTQ the full {2..8} staircase. Only proves the pipeline runs; the
# 0.6B model is far from task-lossless and is not an acceptance model.
#
# Usage: run_slq06b.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"

python -u slq_quantize.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
    --pre_block_modules model.embed_tokens \
    --block_modules model.layers \
    --calibration_data data/calib_qwen3_8m_8192.pt \
    --calibration_tokens 1048576 \
    --calibration_sequence_length 8192 \
    --bitwidth_options 2 3 4 5 6 7 8 \
    --calibration_bitwidth 4 \
    --group_size 128 \
    --low_cpu_mem_usage \
    --cpu_offload_activations \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --save_dir "$SLQ_ROOT/data/slq_qwen3_06b"
echo SLQ_QUANT_DONE
