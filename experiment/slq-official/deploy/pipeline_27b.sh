#!/bin/bash
# Unattended 27B pipeline: SLQ multi-bitwidth GPTQ (self-healing, resumable),
# then checkpoint assembly, fake-quant verification, vLLM TP=2 serving and HTTP
# validation.
#
#   nohup bash deploy/pipeline_27b.sh > /dev/null 2>&1 &
#
# Behaviour:
#   - resumes from the layer store (<layer>/<bits>.pth): finished blocks are skipped
#   - on failure (e.g. transient NPU "vector core execution timed out") it drops
#     partially written layer dirs, rotates to the next card and continues
#   - once all 496 layers exist it chains deploy/finalize_steps.sh
#
# Environment:
#   SLQ_ROOT      official SLQ checkout (default /opt/zh/slq-official)
#   CALIB_TOKENS  calibration tokens for stage 1 (default 65536; the paper recipe
#                 uses 262144, which is ~3x slower but slightly better)
#   QUANT_CARDS   cards to rotate through (default "5 3 0 2 7")
#   OMP/MKL threads default to 16: torch's 192-thread default oversubscribes and
#                 makes the per-layer CPU linear algebra 3-4x slower.
set -u
SLQ_ROOT="${SLQ_ROOT:-/opt/zh/slq-official}"
LD="${LD:-$SLQ_ROOT/data/slq_27b/4bit}"
LOG="${LOG:-$SLQ_ROOT/pipeline_27b.log}"
STATE="${STATE:-$SLQ_ROOT/finalize_state.txt}"
CALIB_TOKENS="${CALIB_TOKENS:-65536}"
QUANT_CARDS="${QUANT_CARDS:-5 3 0 2 7}"
: > "$LOG"
exec >> "$LOG" 2>&1
say() { echo "[$(date '+%H:%M:%S')] $*"; echo "$*" > "$STATE"; }

source /opt/zh/cann9.2.1/ascend-toolkit/set_env.sh 2>/dev/null || true
source /opt/zh/cann9.2.1/nnal/atb/set_env.sh 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate agent 2>/dev/null || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
cd "$SLQ_ROOT" || exit 1
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)

cleanup() {
    for d in "$LD"/*/; do
        [ -d "$d" ] || continue
        n=$(find "$d" -maxdepth 1 -type f | wc -l)
        [ "$n" -lt 8 ] && { echo "  drop incomplete $(basename "$d") ($n)"; rm -rf "$d"; }
    done
}

for attempt in 1 2 3 4 5; do
    for card in $QUANT_CARDS; do
        n=$(ls "$LD" 2>/dev/null | wc -l)
        [ "$n" -ge 496 ] && break 2
        say "QUANT attempt=$attempt card=$card layers=$n"
        cleanup
        export ASCEND_RT_VISIBLE_DEVICES="$card"
        python -u slq_quantize.py \
            --model_name_or_path "$SNAP" \
            --model_class Qwen3_5ForConditionalGeneration \
            --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)$' \
            --pre_block_modules model.language_model.embed_tokens \
            --block_modules model.language_model.layers \
            --calibration_data data/calib_27b_1m_8192.pt \
            --calibration_tokens "$CALIB_TOKENS" \
            --calibration_sequence_length 8192 \
            --bitwidth_options 2 4 6 8 \
            --calibration_bitwidth 4 \
            --group_size 128 \
            --low_cpu_mem_usage \
            --cpu_offload_modules \
            --cpu_offload_activations \
            --dtype bfloat16 \
            --attn_implementation sdpa \
            --save_dir "$SLQ_ROOT/data/slq_27b"
        rc=$?
        n=$(ls "$LD" 2>/dev/null | wc -l)
        say "QUANT_EXIT rc=$rc card=$card layers=$n"
        sleep 20
    done
done

n=$(ls "$LD" 2>/dev/null | wc -l)
if [ "$n" -lt 496 ]; then
    say "QUANT_INCOMPLETE layers=$n after retries"
    exit 2
fi
say "QUANT_COMPLETE layers=$n"
bash "$SLQ_ROOT/deploy/finalize_steps.sh"
