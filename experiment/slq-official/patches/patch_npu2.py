"""补丁 2：Cholesky 走 CPU（NPU 不支持 torch.linalg）、cudnn tf32 保护。"""
import pathlib

ROOT = pathlib.Path("/opt/zh/slq-official")
p = ROOT / "src/quant/fast_obq.py"
s = p.read_text()

OLD = """        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        return W, torch.linalg.cholesky(H_inv, upper=True)"""
NEW = """        # NPU 不支持 torch.linalg；按 AMCT 经验在 CPU 求逆后回迁
        dev = H.device
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H.cpu()))
        H_cho = torch.linalg.cholesky(H_inv, upper=True).to(dev)
        return W, H_cho"""
assert OLD in s, "cholesky block not found"
s = s.replace(OLD, NEW)

OLD2 = "torch.backends.cudnn.allow_tf32 = False"
NEW2 = """try:
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass"""
if "torch.backends.cudnn.allow_tf32 = False" in s:
    s = s.replace("try:\n    torch.backends.cuda.matmul.allow_tf32 = False\nexcept Exception:\n    pass  # NPU 无该后端\ntorch.backends.cudnn.allow_tf32 = False",
                  "try:\n    torch.backends.cuda.matmul.allow_tf32 = False\nexcept Exception:\n    pass  # NPU 无该后端\ntry:\n    torch.backends.cudnn.allow_tf32 = False\nexcept Exception:\n    pass")
p.write_text(s)
print("patched fast_obq.py")

# 同时检查 quant_utils 的 find_params/quantize 是否有 CUDA 假设
q = ROOT / "src/quant/quant_utils.py"
print("quant_utils cuda refs:", [i for i, l in enumerate(q.read_text().splitlines(), 1) if "cuda" in l])
