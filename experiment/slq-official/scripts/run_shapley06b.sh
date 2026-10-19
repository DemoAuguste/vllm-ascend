#!/bin/bash
set -e
cd /opt/zh/slq-official
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -u slq_build_database.py \
    --model Qwen/Qwen3-0.6B \
    --layer-dir data/slq_qwen3_06b/Qwen3-0.6B/4bit \
    --calibration-data data/calib_qwen3_1m_2048.pt \
    --calibration-samples 8 \
    --calibration-seq-length 2048 \
    --batch-size 4 \
    --top-k 10 \
    --seed 42 \
    --skip-linear \
    --shapley-bitwidths 2,4,6,8 \
    --num-permutations 4 \
    --layer-groups 'mlp.gate_proj,mlp.up_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj'
echo SHAPLEY_DONE
