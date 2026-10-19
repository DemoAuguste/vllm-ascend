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

for B in 3.8 3.6; do
  python -u slq_build_checkpoint.py --model "$M" --layer-dir "$LD" \
      --config "config_tl27b_b$B.txt" --output-dir "$OUT" --subdir-name "b${B/./}" \
      --dtype bfloat16 --quant-format fake --group-size 128 --force
done

cd /opt/zh/SLQ-gptq
for B in 3.8 3.6; do
  python -u zeroshot_eval.py --model "$OUT/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/b${B/./}" \
      --device npu:0 --limit 200 --tasks $T --out "zs27b_b${B/./}.json"
done
echo BUDGET_EVAL_DONE
