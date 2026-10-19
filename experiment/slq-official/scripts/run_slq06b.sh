#!/bin/bash
set -e
cd /opt/zh/slq-official
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
    --save_dir /opt/zh/slq-official/data/slq_qwen3_06b
echo SLQ_QUANT_DONE
