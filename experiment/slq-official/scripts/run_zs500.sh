#!/bin/bash
set -e
cd /opt/zh/SLQ-gptq
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

M=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
C=/opt/zh/slq-official/ckpt_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
T=arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa

python -u zeroshot_eval.py --model "$M" --device npu:0 --limit 500 --tasks $T --out zs27b_bf16_500.json
python -u zeroshot_eval.py --model "$C/b36" --device npu:0 --limit 500 --tasks $T --out zs27b_b36_500.json
echo ZS500_DONE
