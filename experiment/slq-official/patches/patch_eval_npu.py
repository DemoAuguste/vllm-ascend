"""Release cached NPU blocks between batches in the metrics module (large-vocab logits)."""

PATH = '/opt/zh/slq-official/src/metrics/evaluation.py'

HELPER = '''

def _empty_cache():
    """Release cached NPU/CUDA blocks between batches with large vocab logits."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
'''

src = open(PATH).read()
if '_empty_cache' not in src:
    src = src.replace('from tqdm import trange\n', 'from tqdm import trange\n' + HELPER, 1)
    src = src.replace('        del logits\n', '        del logits\n        _empty_cache()\n')
    open(PATH, 'w').write(src)
    print('patched')
else:
    print('already patched')
