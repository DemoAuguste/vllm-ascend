"""Summarize SLQ 27B bit allocations and storage accounting."""
import collections
import json
import os
import sys

LD = '/opt/zh/slq-official/data/slq_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/4bit'
CK = '/opt/zh/slq-official/ckpt_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MODEL = '/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def parse_cfg(path):
    cfg = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith('#') and ':' in line:
            n, b = line.rsplit(':', 1)
            cfg[n.strip()] = int(b)
    return cfg


def size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def suffix(name):
    return name.rsplit('.', 1)[-1]


db = json.load(open(os.path.join(LD, 'error_db.json')))
print('error_db keys:', sorted(db.keys()))
pc = db['param_counts']
print('param_counts entries:', len(pc))
print('quantized params:', sum(pc.values()))

out = {'configs': {}}
for tag in ['b3.4', 'b3.6', 'b3.8', 'b4.0']:
    path = f'/opt/zh/slq-official/config_tl27b_{tag}.txt'
    cfg = parse_cfg(path)
    d_layers = collections.Counter(cfg.values())
    d_params = collections.Counter()
    for n, b in cfg.items():
        d_params[b] += pc.get(n, 0)
    total_p = sum(d_params.values())
    by_type = collections.defaultdict(lambda: [0, 0])
    for n, b in cfg.items():
        t = suffix(n)
        by_type[t][0] += pc.get(n, 0)
        by_type[t][1] += pc.get(n, 0) * b
    out['configs'][tag] = {
        'layers': {b: d_layers[b] for b in sorted(d_layers)},
        'layer_pct': {b: round(100 * d_layers[b] / len(cfg), 2) for b in sorted(d_layers)},
        'params': {b: d_params[b] for b in sorted(d_params)},
        'param_pct': {b: round(100 * d_params[b] / total_p, 2) for b in sorted(d_params)},
        'avg_bits_params': round(sum(d_params[b] * b for b in d_params) / total_p, 4),
        'avg_bits_layers': round(sum(b * c for b, c in d_layers.items()) / len(cfg), 4),
        'n_layers': len(cfg),
        'quant_params': total_p,
        'by_type': {t: {'params': by_type[t][0],
                        'avg_bits': round(by_type[t][1] / by_type[t][0], 3)}
                    for t in sorted(by_type)},
    }

# a qparams sample
sample = os.path.join(LD, 'model.language_model.layers.0.mlp.gate_proj', '_qparams.pt')
q = None
if os.path.exists(sample):
    import torch
    q = torch.load(sample, map_location='cpu')
    info = {}
    if isinstance(q, dict):
        for k, v in q.items():
            if hasattr(v, 'shape'):
                info[k] = {'shape': list(v.shape), 'dtype': str(v.dtype),
                           'nbytes': v.numel() * v.element_size()}
            else:
                info[k] = str(type(v))
    out['qparams_sample'] = info

out['sizes'] = {
    'orig_model_bytes': size(MODEL),
    'ckpt_b36_bytes': size(os.path.join(CK, 'b36')),
    'ckpt_b38_bytes': size(os.path.join(CK, 'b38')),
    'layer_store_bytes': size(LD),
}

print(json.dumps(out, indent=1, ensure_ascii=False))
