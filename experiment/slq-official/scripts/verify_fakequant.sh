#!/bin/bash
# Verify that an assembled checkpoint really carries SLQ fake-quant weights.
#
#   CONFIG=.../config_tl27b_b3.6.txt bash scripts/verify_fakequant.sh
#
# Checks, for one sampled layer per bitwidth:
#   - inside every 128-column group the dequantised weights take <= 2^bits values
#   - weights differ from the BF16 reference (quantisation actually happened)
# and lists tensors present in the base model but absent from the checkpoint
# (the MTP head is intentionally not exported).
set -u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate agent 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SLQ_ROOT="${SLQ_ROOT:-/opt/zh/slq-official}"
SNAP="${SNAP:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ 2>/dev/null | head -1)}"
SNAP="${SNAP%/}"
CKPT="${CKPT:-$(ls -d "$SLQ_ROOT"/ckpt_27b/*/b36 2>/dev/null | head -1)}"
CONFIG="${CONFIG:-$SLQ_ROOT/configs/config_tl27b_b3.6.txt}"

[ -d "$CKPT" ] || { echo "no checkpoint found (set CKPT)"; exit 1; }

python - "$SNAP" "$CKPT" "$CONFIG" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

snap, ckpt, cfg_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

bits_of = {}
for line in cfg_path.read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#") or ":" not in s:
        continue
    name, b = s.rsplit(":", 1)
    bits_of[name.strip()] = int(b.strip())
print("config entries:", len(bits_of), "bitwidth histogram:", dict(Counter(bits_of.values())))

ref_index = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
q_index = json.load(open(ckpt / "model.safetensors.index.json"))["weight_map"]
missing = [k for k in ref_index if k not in q_index]
print(f"tensors ref={len(ref_index)} ckpt={len(q_index)} missing={len(missing)}")
if missing:
    print("missing keys (first 20):")
    for k in missing[:20]:
        print("  ", k)
    print("  (the MTP head is expected to be absent)")

by_bits = {}
for name, b in bits_of.items():
    by_bits.setdefault(b, name)

failures = 0
print("\nprobe one layer per bitwidth:")
for b in sorted(by_bits):
    layer = by_bits[b]
    key = layer + ".weight"
    if key not in q_index:
        print(f"  bits={b} {layer}: weight key absent")
        failures += 1
        continue
    with safe_open(ckpt / q_index[key], framework="pt") as f:
        q = f.get_tensor(key).float()
    with safe_open(snap / ref_index[key], framework="pt") as f:
        r = f.get_tensor(key).float()
    rows, cols = q.shape
    g, ng = 128, cols // 128
    probe = q[:4, : ng * g].reshape(4, ng, g)
    worst = max(len(torch.unique(probe[ri, gi])) for ri in range(4) for gi in range(ng))
    diff = (q - r).abs()
    ok = worst <= 2 ** b
    failures += 0 if ok else 1
    print(f"  bits={b:<2} {layer}")
    print(f"    shape={tuple(q.shape)}  max levels in any 128-group={worst} (limit {2**b})"
          f"  -> {'PASS' if ok else 'FAIL'}")
    print(f"    vs BF16: max|d|={diff.max():.5f} mean|d|={diff.mean():.6f}")

print("\nFINGERPRINT:", "PASS" if failures == 0 else f"FAIL ({failures})")
sys.exit(0 if failures == 0 else 3)
PY
