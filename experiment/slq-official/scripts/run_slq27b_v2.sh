#!/bin/bash
set -e
cd /opt/zh/slq-official
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

M=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

python -u slq_quantize.py \
    --model_name_or_path "$M" \
    --model_class Qwen3_5ForConditionalGeneration \
    --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)$' \
    --pre_block_modules model.language_model.embed_tokens \
    --block_modules model.language_model.layers \
    --calibration_data data/calib_27b_1m_8192.pt \
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
    --save_dir /opt/zh/slq-official/data/slq_27b
echo SLQ27_DONE
