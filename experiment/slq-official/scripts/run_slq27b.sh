#!/bin/bash
# Stage ①: GPTQ every quantizable projection once, emitting the full {2,4,6,8}
# bitwidth layer store (single Hessian per layer, cal_bits=4 sequential propagation).
#
# Usage: run_slq27b.sh            # card from ASCEND_RT_VISIBLE_DEVICES (default 5)
set -euo pipefail
. "$(dirname "$0")/common.sh"

python -u slq_quantize.py \
    --model_name_or_path "$SLQ_MODEL" \
    --model_class Qwen3_5ForConditionalGeneration \
    --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)$' \
    --pre_block_modules model.language_model.embed_tokens \
    --block_modules model.language_model.layers \
    --calibration_data "$SLQ_DATA/calib_27b_1m_8192.pt" \
    --calibration_tokens 262144 \
    --calibration_sequence_length 8192 \
    --bitwidth_options 2 4 6 8 \
    --calibration_bitwidth 4 \
    --group_size 128 \
    --low_cpu_mem_usage \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --save_dir "$SLQ_DATA/slq_27b"
echo SLQ27_DONE
