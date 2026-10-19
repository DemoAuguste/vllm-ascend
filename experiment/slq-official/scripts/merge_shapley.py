"""Average two Shapley databases produced with different permutation seeds."""
import json
import sys

import numpy as np

p1, p2, out = sys.argv[1:4]
a = json.load(open(p1))
b = json.load(open(p2))
assert a['bitwidths'] == b['bitwidths']
assert a['layer_names'] == b['layer_names']

res = {}
for k in ('nll_shapley_values', 'kl_shapley_values', 'ear_shapley_values'):
    x, y = np.array(a[k], dtype=float), np.array(b[k], dtype=float)
    assert x.shape == y.shape
    res[k] = ((x + y) / 2.0).tolist()

for k in ('nll_shapley_per_level', 'kl_shapley_per_level', 'ear_shapley_per_level'):
    res[k] = {layer: {bit: (a[k][layer][bit] + b[k][layer][bit]) / 2.0
                      for bit in a[k][layer]}
              for layer in a[k]}

res['bitwidths'] = a['bitwidths']
res['num_permutations'] = a['num_permutations'] + b['num_permutations']
for k in ('top_k', 'calibration_dataset', 'calibration_samples', 'seed',
          'per_layer_normalized', 'layer_names', 'layer_groups'):
    if k in a:
        res[k] = a[k]

json.dump(res, open(out, 'w'), indent=2)

kl = np.array(res['kl_shapley_values'], dtype=float)
print('merged ->', out)
print('bitwidths', res['bitwidths'], 'perms', res['num_permutations'], 'groups', kl.shape[1])
for i, bw in enumerate(res['bitwidths']):
    print(f'  bw {bw}: total KL {kl[1, :, i].sum():.6f}, negative groups {int((kl[1, :, i] < 0).sum())}')
