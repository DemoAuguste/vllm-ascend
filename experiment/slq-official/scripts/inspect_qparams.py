import sys

import torch

path = '/opt/zh/slq-official/data/slq_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/4bit/model.language_model.layers.0.mlp.gate_proj/4_qparams.pt'
q = torch.load(path, map_location='cpu')
print('type', type(q))
if isinstance(q, dict):
    for k, v in q.items():
        if hasattr(v, 'shape'):
            print(k, tuple(v.shape), v.dtype, v.numel() * v.element_size(), 'bytes')
        else:
            print(k, type(v), v)
else:
    print(q)
