#!/bin/bash
set -e
cd /opt/zh/slq-official
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

M=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
LD=/opt/zh/slq-official/data/slq_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/4bit
DB=$LD/shapley_topk10_calib_27b_256k_2048_4x2048_8perms_bw2-4-6-8_grpb7761ae4_seed42.json

python -u slq_search_dl.py \
    --model "$M" \
    --layer-dir "$LD" \
    --mode tl \
    --sensitivity-method shapley \
    --shapley-db "$DB" \
    --calibration-data data/calib_27b_256k_2048.pt \
    --calibration-samples 4 \
    --calibration-seq-length 2048 \
    --batch-size 1 \
    --calibration-bitwidth 2 \
    --calibration-recovery 0.6984 \
    --min-recovery 0.995 \
    --bitwidths 2,4,6,8 \
    --bitwidth-map 2:2.156,4:4.156,6:6.156,8:8.156 \
    --output config_tl27b.txt
echo TL27B_DONE
