#!/bin/bash
# Build the two 27B calibration sets:
#   calib_27b_1m_8192.pt   128 x 8192 tokens  (stage ① GPTQ, 262144 tokens used)
#   calib_27b_256k_2048.pt 128 x 2048 tokens  (stages ②/③, 4x2048 per Shapley run)
#
# Needs data/fineweb_10bt_000.parquet; see reproduction guide §3.1 for the download.
#
# Usage: run_calib27b.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"

PARQUET="${PARQUET:-$SLQ_DATA/fineweb_10bt_000.parquet}"
if [ ! -f "$PARQUET" ]; then
    echo "missing $PARQUET (see reproduction guide §3.1)" >&2
    exit 1
fi

python -u scripts/build_calib.py --parquet "$PARQUET" --tokenizer "$SLQ_MODEL" \
    --seq 8192 --num 128 --out "$SLQ_DATA/calib_27b_1m_8192.pt"
python -u scripts/build_calib.py --parquet "$PARQUET" --tokenizer "$SLQ_MODEL" \
    --seq 2048 --num 128 --out "$SLQ_DATA/calib_27b_256k_2048.pt"
echo CALIB27B_DONE
