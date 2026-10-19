import json
import sys

import numpy as np

path = sys.argv[1]
a = json.load(open(path))
print('keys', sorted(a.keys()))
for k in ('nll_shapley_values', 'kl_shapley_values', 'ear_shapley_values'):
    arr = np.array(a[k], dtype=float)
    print(k, arr.shape)
    if arr.ndim == 2:
        print('  column sums', np.round(arr.sum(axis=0), 6).tolist())
        print('  negative count', int((arr < 0).sum()))
print('per_level layers', len(a.get('kl_shapley_per_level', {})))
print('bitwidths', a.get('bitwidths'), 'perms', a.get('num_permutations'))
