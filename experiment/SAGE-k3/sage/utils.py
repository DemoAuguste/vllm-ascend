"""NPU 友好的通用辅助：device 解析、分块方向赋值、日志。

设计要点（对齐 AGENTS.md 的 NPU 性能要求）：
  - 惰性导入 torch_npu：CPU 环境不依赖，NPU 环境才注册后端。
  - 提供分块 argmax：方向赋值 `[N,v] @ [v,K]` 的中间矩阵按 N 分块，
    避免大权重矩阵（如 27B 的单层 MLP）一次性物化 OOM。
  - 避免热路径 `.item()`：本模块不返回 Python 标量；需要标量的地方
    由调用方显式承担一次同步（如每 epoch 的 loss 记录）。
"""

from __future__ import annotations

import sys

import torch


def log(*args: object) -> None:
    """带 flush 的打印（NPU 容器内日志实时可见）。"""
    print(*args, flush=True)


def resolve_device(spec: str | torch.device | None) -> torch.device:
    """把 'cpu' / 'npu' / 'npu:0' / 'cuda:0' / 显式 device 解析为 torch.device。

    'npu' 或 'npu:N' 会惰性导入 torch_npu 以注册后端；导入失败则给出清晰报错。
    """
    if isinstance(spec, torch.device):
        return spec
    spec = "cpu" if spec is None else str(spec)
    if spec.startswith("npu"):
        try:
            import torch_npu  # noqa: F401  注册 npu 后端
        except ImportError as exc:  # pragma: no cover - 仅 NPU 环境触发
            raise RuntimeError(
                "指定了 NPU device，但导入 torch_npu 失败；请在 Ascend 环境运行，"
                "或改用 --device cpu"
            ) from exc
        return torch.device(spec if ":" in spec else "npu:0")
    return torch.device(spec)


def chunked_matmul_argmax(
    a: torch.Tensor,
    b: torch.Tensor,
    chunk: int,
    out_dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """分块计算 `argmax(a @ b.T, dim=1)`，避免中间矩阵 (N, M) 过大。

    Args:
        a: (N, d)
        b: (M, d)
        chunk: 每块样本数（按 N 分块）
    Returns:
        (N,) 的索引张量
    """
    n = a.shape[0]
    if chunk <= 0 or n <= chunk:
        return (a @ b.t()).argmax(dim=1).to(out_dtype)
    out = torch.empty(n, dtype=out_dtype, device=a.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = (a[start:stop] @ b.t()).argmax(dim=1)
    return out


def softplus_inverse(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """softplus 的逆：由正的连续半径 r 反解未约束参数 ρ（r = softplus(ρ)）。

    使用数值稳定的 log(expm1(x)) 形式；x 需为正。
    """
    x = x.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


def format_bytes(num_bytes: float) -> str:
    """人类可读的字节数（用于日志/性能报告）。"""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f}PiB"


def on_npu() -> bool:  # pragma: no cover - 环境相关
    """当前是否运行在 Ascend NPU 后端。"""
    return "torch_npu" in sys.modules and hasattr(torch, "npu")
