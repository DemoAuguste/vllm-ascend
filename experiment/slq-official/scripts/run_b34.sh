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
OUT=/opt/zh/slq-official/ckpt_27b
T=arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa

python -u slq_build_checkpoint.py --model "$M" --layer-dir "$LD" \
    --config config_tl27b_b3.4.txt --output-dir "$OUT" --subdir-name b34 \
    --dtype bfloat16 --quant-format fake --group-size 128 --force

cd /opt/zh/SLQ-gptq
python -u zeroshot_eval.py --model "$OUT/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/b34" \
    --device npu:0 --limit 200 --tasks $T --out zs27b_b34.json
echo B34_DONE
