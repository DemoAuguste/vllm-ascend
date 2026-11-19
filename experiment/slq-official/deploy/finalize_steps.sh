#!/bin/bash
# Build -> verify -> serve -> validate. Assumes the layer store is complete.
# Writes progress to $SLQ_ROOT/finalize_state.txt and logs next to the artifacts.
set -u
SLQ_ROOT="${SLQ_ROOT:-/opt/zh/slq-official}"
PORT="${PORT:-8098}"
DEVS="${DEVS:-0,2}"
STATE="$SLQ_ROOT/finalize_state.txt"
say() { echo "[$(date '+%H:%M:%S')] $*"; echo "$*" > "$STATE"; }

cd "$SLQ_ROOT" || exit 4
say "BUILD_CHECKPOINT start"
if bash "$SLQ_ROOT/scripts/build_ckpt.sh" > "$SLQ_ROOT/build_b36.log" 2>&1; then
    say "BUILD_OK $(ls -d "$SLQ_ROOT"/ckpt_27b/*/b36 2>/dev/null | head -1)"
else
    say "BUILD_FAILED"; tail -30 "$SLQ_ROOT/build_b36.log"; exit 5
fi

say "VERIFY_FAKEQUANT start"
if bash "$SLQ_ROOT/scripts/verify_fakequant.sh" > "$SLQ_ROOT/verify_b36.log" 2>&1; then
    say "VERIFY_PASS"
else
    say "VERIFY_FAILED"; tail -40 "$SLQ_ROOT/verify_b36.log"; exit 6
fi

say "SERVE start"
PORT="$PORT" DEVS="$DEVS" bash "$SLQ_ROOT/deploy/serve_b36.sh" > "$SLQ_ROOT/serve_b36_launch.log" 2>&1
health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health")
say "SERVE health=$health"
if [ "$health" != "200" ]; then
    say "SERVE_FAILED"; tail -40 "$SLQ_ROOT/serve_b36_${PORT}.log" 2>/dev/null; exit 7
fi

say "HTTP_VALIDATE start"
bash "$SLQ_ROOT/deploy/http_validate.sh" "$PORT" qwen3.8-27b-slq-b36 > "$SLQ_ROOT/http_validation.log" 2>&1
say "HTTP_VALIDATE done rc=$?"
cat "$SLQ_ROOT/http_validation.log"
say "ALL_DONE"
