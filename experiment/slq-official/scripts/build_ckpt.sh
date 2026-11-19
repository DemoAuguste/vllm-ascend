#!/bin/bash
# Assemble an SLQ fake-quant checkpoint from the layer store, add the VLM
# preprocessing files and print a short summary.
#
#   CONFIG=$SLQ_ROOT/configs/config_tl27b_b3.6.txt SUBDIR=b36 bash scripts/build_ckpt.sh
#
# Environment:
#   SLQ_ROOT    official SLQ checkout carrying this overlay (default /opt/zh/slq-official)
#   SNAP        Qwen3.8-27B snapshot dir (default: from the HF cache)
#   LAYER_DIR   layer store (default $SLQ_ROOT/data/slq_27b/4bit)
#   CONFIG      per-layer bitwidth config (default configs/config_tl27b_b3.6.txt)
#   SUBDIR      output subdirectory name (default b36)
#   OUT         output root (default $SLQ_ROOT/ckpt_27b)
set -euo pipefail
source /opt/zh/cann9.2.1/ascend-toolkit/set_env.sh 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate agent 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SLQ_ROOT="${SLQ_ROOT:-/opt/zh/slq-official}"
SNAP="${SNAP:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ 2>/dev/null | head -1)}"
SNAP="${SNAP%/}"
LAYER_DIR="${LAYER_DIR:-$SLQ_ROOT/data/slq_27b/4bit}"
CONFIG="${CONFIG:-$SLQ_ROOT/configs/config_tl27b_b3.6.txt}"
SUBDIR="${SUBDIR:-b36}"
OUT="${OUT:-$SLQ_ROOT/ckpt_27b}"
cd "$SLQ_ROOT"

echo "[build] layers in store: $(ls "$LAYER_DIR" | wc -l)"
echo "[build] config: $CONFIG -> subdir $SUBDIR"
python -u slq_build_checkpoint.py \
    --model "$SNAP" \
    --layer-dir "$LAYER_DIR" \
    --config "$CONFIG" \
    --output-dir "$OUT" \
    --subdir-name "$SUBDIR" \
    --dtype bfloat16 \
    --quant-format fake \
    --group-size 128 \
    --force

C="$OUT/$(basename "$SNAP")/$SUBDIR"
echo "[build] checkpoint: $C ($(du -sh "$C" | cut -f1))"

echo "[build] adding VLM preprocessing files (never overwrite checkpoint files) ..."
for f in "$SNAP"/*.json "$SNAP"/*.jinja "$SNAP"/*.txt "$SNAP"/*.model; do
    [ -f "$f" ] && cp -n "$f" "$C/" || true
done

echo "[build] shards: $(grep -o 'model-0000[0-9]-of-0000[0-9]' "$C/model.safetensors.index.json" 2>/dev/null | sort -u | tr '\n' ' ')"
ls "$C"/preprocessor_config.json "$C"/video_preprocessor_config.json >/dev/null 2>&1 \
    && echo "[build] VLM preprocessor files present"
echo "[build] CKPT_DONE $C"
