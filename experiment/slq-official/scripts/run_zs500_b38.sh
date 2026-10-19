#!/bin/bash
set -e
cd /opt/zh/SLQ-gptq
source /opt/zh/cann/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agent
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

C=/opt/zh/slq-official/ckpt_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
T=arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa
python -u zeroshot_eval.py --model "$C/b38" --device npu:0 --limit 500 --tasks $T --out zs27b_b38_500.json
echo ZS500_B38_DONE
