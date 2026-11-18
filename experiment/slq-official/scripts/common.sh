#!/bin/bash
# Shared environment for the SLQ Qwen3.8-27B pipeline.
#
# Source it from any run_*.sh script:
#     . "$(dirname "$0")/common.sh"
#
# Every value can be overridden from the environment, e.g.
#     SLQ_ROOT=/workspace/slq-official ASCEND_RT_VISIBLE_DEVICES=7 bash scripts/run_slq27b.sh
#
# The scripts assume the official IST-DASLab/SLQ checkout lives at $SLQ_ROOT and
# that the NPU overlay files of this repository were copied on top of it
# (see ../README.md, section "把 NPU 适配装进官方仓").

SLQ_ROOT="${SLQ_ROOT:-/opt/zh/slq-official}"
SLQ_CANN_ENV="${SLQ_CANN_ENV:-/opt/zh/cann/ascend-toolkit/set_env.sh}"
SLQ_CONDA_SH="${SLQ_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
SLQ_CONDA_ENV="${SLQ_CONDA_ENV:-agent}"

# Qwen3.8-27B snapshot and the artifacts derived from it.
SLQ_SNAPSHOT="${SLQ_SNAPSHOT:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
SLQ_MODEL="${SLQ_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/$SLQ_SNAPSHOT}"
SLQ_DATA="${SLQ_DATA:-$SLQ_ROOT/data}"
SLQ_LAYERS="${SLQ_LAYERS:-$SLQ_DATA/slq_27b/$SLQ_SNAPSHOT/4bit}"
SLQ_CKPT="${SLQ_CKPT:-$SLQ_ROOT/ckpt_27b}"
SLQ_RESULTS="${SLQ_RESULTS:-$SLQ_ROOT/results}"
SLQ_TASKS="${SLQ_TASKS:-arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa}"

# Card selection: honour an explicit ASCEND_RT_VISIBLE_DEVICES, else card 5.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-5}"

# The NPU host has no working network path to the HF hub (httpx client is closed).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CANN and conda are site specific; activate them when present, otherwise the
# caller is expected to have a working environment already.
if [ -f "$SLQ_CANN_ENV" ]; then
    # shellcheck disable=SC1090
    . "$SLQ_CANN_ENV"
fi
if [ -f "$SLQ_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    . "$SLQ_CONDA_SH"
    conda activate "$SLQ_CONDA_ENV" || true
fi

if [ ! -d "$SLQ_ROOT" ]; then
    echo "common.sh: SLQ_ROOT does not exist: $SLQ_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi

# All pipeline entry points are executed from the overlay root so that
# `from src...` imports resolve.
cd "$SLQ_ROOT" || return 1 2>/dev/null || exit 1

# Locate a generated config: the repository keeps them under configs/, while a
# checkout where they were produced by slq_search_dl.py may have them at the root.
slq_config() {
    local name="$1"
    if [ -f "$SLQ_ROOT/configs/$name" ]; then
        printf '%s\n' "$SLQ_ROOT/configs/$name"
    elif [ -f "$SLQ_ROOT/$name" ]; then
        printf '%s\n' "$SLQ_ROOT/$name"
    else
        printf '%s\n' "$SLQ_ROOT/configs/$name"
    fi
}

mkdir -p "$SLQ_RESULTS"
