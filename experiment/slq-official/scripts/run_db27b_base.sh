#!/bin/bash
set -e
cd /opt/zh/slq-official
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

M=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
LD=/opt/zh/slq-official/data/slq_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/4bit

python -u slq_build_database.py \
    --model "$M" \
    --layer-dir "$LD" \
    --calibration-data data/calib_27b_256k_2048.pt \
    --calibration-samples 4 \
    --calibration-seq-length 2048 \
    --batch-size 2 \
    --skip-linear \
    --skip-shapley \
    --layer-groups 'mlp.gate_proj,mlp.up_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj' \
    --gpu-cache off \
    --seed 42
echo DB27B_BASE_DONE
