import sys, time

sys.path.insert(0, '/opt/zh/slq-official')

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

from src.metrics.evaluation import compute_all_metrics, compute_baseline_topk
from src.models.loading import load_model
from src.utils.weight_store import WeightStore

M = '/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
LD = '/opt/zh/slq-official/data/slq_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/4bit'

t0 = time.time()
model = load_model(M)
torch.npu.synchronize()
print('model_load_s', round(time.time() - t0, 1), flush=True)
print('npu_alloc_GiB', round(torch.npu.memory_allocated() / 2 ** 30, 1), flush=True)

data = torch.load('/opt/zh/slq-official/data/calib_27b_256k_2048.pt')[:8]
x = torch.cat(data, dim=0)
print('data_shape', tuple(x.shape), flush=True)

bs = 2
t0 = time.time()
v, k = compute_baseline_topk(model, x, bs, 10)
torch.npu.synchronize()
print('baseline_8x2048_s', round(time.time() - t0, 1), flush=True)

t0 = time.time()
m = compute_all_metrics(model, x, v, k, bs)
torch.npu.synchronize()
print('metric_8x2048_s', round(time.time() - t0, 1), m, flush=True)

ws = WeightStore(LD).load()
name = 'model.language_model.layers.0.mlp.gate_proj'
t0 = time.time()
w = ws.get_layer_weight(name, 4)
print('load_tensor_s', round(time.time() - t0, 2), tuple(w.shape), flush=True)
t0 = time.time()
wd = w.to('npu:0', torch.bfloat16)
torch.npu.synchronize()
print('h2d_s', round(time.time() - t0, 2), flush=True)
print('BENCH_DONE', flush=True)
