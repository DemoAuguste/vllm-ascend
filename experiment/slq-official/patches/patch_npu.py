"""把官方 SLQ 仓库的 CUDA 硬编码替换为 NPU 兼容写法（最小改动）。"""
import pathlib
import re

ROOT = pathlib.Path("/opt/zh/slq-official")

ENTRIES = ["slq_quantize.py", "slq_build_database.py", "slq_search_dl.py",
           "slq_search_tl.py", "slq_build_checkpoint.py"]

EMPTY_OLD = "torch.cuda.empty_cache()"
EMPTY_NEW = '(torch.npu.empty_cache() if hasattr(torch, "npu") else torch.cuda.empty_cache())'

TF32_OLD = "torch.backends.cuda.matmul.allow_tf32 = False"
TF32_NEW = ('try:\n    torch.backends.cuda.matmul.allow_tf32 = False\n'
            'except Exception:\n    pass  # NPU 无该后端')


def patch_file(rel, fn):
    p = ROOT / rel
    if not p.exists():
        print("skip missing", rel)
        return
    s = p.read_text()
    s2 = fn(s)
    if s2 != s:
        p.write_text(s2)
        print("patched", rel)


def entry_patch(s: str) -> str:
    if "import torch_npu" not in s:
        s = s.replace("import torch\n", "import torch\nimport torch_npu  # noqa: F401  (NPU backend)\n", 1)
    s = s.replace('backend="nccl"', 'backend="hccl"')
    s = s.replace("device_id=torch.device(f\"cuda:{os.environ['LOCAL_RANK']}\")",
                  "device_id=torch.device(f\"npu:{os.environ['LOCAL_RANK']}\")")
    s = s.replace('device = f"cuda:{rank}"', 'device = f"npu:{rank}"')
    s = s.replace("torch.cuda.set_device(device)", "torch.npu.set_device(device)")
    s = s.replace(EMPTY_OLD, EMPTY_NEW)
    return s


def cache_patch(s: str) -> str:
    return s.replace(EMPTY_OLD, EMPTY_NEW)


def tf32_patch(s: str) -> str:
    return s.replace(TF32_OLD, TF32_NEW)


for e in ENTRIES:
    patch_file(e, entry_patch)
patch_file("src/quant/quantizer.py", cache_patch)
patch_file("src/quant/fast_obq.py", tf32_patch)

print("\n=== remaining cuda refs ===")
for p in sorted(ROOT.rglob("*.py")):
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if "cuda" in line and "cuda.empty_cache() if" not in line:
            print(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:110]}")
