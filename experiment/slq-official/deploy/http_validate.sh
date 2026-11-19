#!/bin/bash
# End-to-end HTTP validation of a served Qwen3.8-27B endpoint (OpenAI-compatible).
#
#   bash deploy/http_validate.sh 8098 qwen3.8-27b-slq-b36
#
# Covers: /health, /v1/models, /v1/chat/completions (non-stream),
#         /v1/completions, streaming SSE, and prints latency/token usage.
set -u
PORT="${1:-8098}"
MODEL="${2:-qwen3.8-27b-slq-b36}"
BASE="http://127.0.0.1:$PORT"

py() { python3 -c "$1"; }

echo "=== 1) /health ==="
echo "status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health")"

echo "=== 2) /v1/models ==="
curl -s --max-time 10 "$BASE/v1/models" | py \
 'import json,sys; d=json.load(sys.stdin); m=d["data"][0]
print("id:",m["id"],"| root:",m["root"],"| max_model_len:",m["max_model_len"])'

echo "=== 3) /v1/chat/completions (non-stream) ==="
body=$(cat <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"用一句话说明什么是混合精度量化。"}],
 "temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}
EOF
)
t0=$(date +%s.%N)
resp=$(curl -s --max-time 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
t1=$(date +%s.%N)
echo "$resp" | py '
import json,sys
d=json.load(sys.stdin); u=d["usage"]
print("content:", d["choices"][0]["message"]["content"].strip()[:200])
print("tokens: prompt=%d completion=%d" % (u["prompt_tokens"], u["completion_tokens"]))'
echo "wall_seconds=$(echo "$t1 - $t0" | bc)"

echo "=== 4) /v1/completions ==="
curl -s --max-time 300 "$BASE/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of China is\",\"temperature\":0,\"max_tokens\":24}" \
    | py 'import json,sys; d=json.load(sys.stdin)
print("text:", d["choices"][0]["text"].strip()[:160])'

echo "=== 5) streaming (SSE) ==="
timeout 300 curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"列出三个中国省份，用顿号分隔。\"}],\"temperature\":0,\"max_tokens\":32,\"stream\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | head -3 | cut -c1-160
echo "=== SSE_OK ==="
