#!/bin/bash
set -e
cd /opt/zh/slq-official
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export HF_HUB_OFFLINE=1

M=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
python -u build_calib.py --parquet data/fineweb_10bt_000.parquet \
    --tokenizer "$M" --seq 8192 --num 128 --out data/calib_27b_1m_8192.pt
python -u build_calib.py --parquet data/fineweb_10bt_000.parquet \
    --tokenizer "$M" --seq 2048 --num 128 --out data/calib_27b_256k_2048.pt
echo CALIB27_DONE
