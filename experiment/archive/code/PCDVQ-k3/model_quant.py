"""Qwen3-0.6B 全模型 PCDVQ fake-quant（分块方向匹配，NPU/CPU 可选）。

流程：
  1. 加载/构造 PCDVQ 码本（a/b/k）
  2. 用 transformers 加载模型（本地 HF 缓存），整体搬到 device
  3. 遍历所有 nn.Linear（跳过 embed/lm_head），逐层分块 PCDVQ 量化 + 反量化回权重空间
  4. 打印逐层相对 Frobenius 误差与汇总，可选保存 fake-quant checkpoint

运行示例：
  ASCEND_RT_VISIBLE_DEVICES=5 python model_quant.py --device npu --save-dir <dir>
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch_npu  # noqa: F401  (注册 npu backend)

from pcdvq import PCDVQ, e8_unit_directions, greedy_direction_codebook, lloyd_max_magnitude_codebook


def log(*a):
    print(*a, flush=True)


def build_codebooks(k: int, a: int, b: int, device: str, max_norm_sq: int = 8, seed: int = 0) -> PCDVQ:
    pool = e8_unit_directions(max_norm_sq)
    C_phi = greedy_direction_codebook(pool, a, seed=seed)
    C_r = lloyd_max_magnitude_codebook(b, k, tau=0.999)
    qz = PCDVQ(k=k, a=a, b=b, max_norm_sq=max_norm_sq, seed=seed, device=device)
    qz.C_phi = C_phi.to(device)
    qz.C_r = C_r.to(device)
    qz.pool_size = C_phi.shape[0]
    return qz


def load_codebooks_from_cache(path: str, device: str) -> PCDVQ:
    state = torch.load(path, weights_only=False, map_location="cpu")
    qz = PCDVQ(k=state["k"], a=state["a"], b=state["b"], max_norm_sq=8, seed=0, device=device)
    qz.C_phi = state["C_phi"].to(device)
    qz.C_r = state["C_r"].to(device)
    qz.pool_size = state["C_phi"].shape[0]
    return qz


def quantize_model(model, qz: PCDVQ, chunk: int = 8192, only_kw=None):
    """遍历模型 Linear 层做 PCDVQ fake-quant，返回 (逐层统计, 总耗时, 跳过层数)。

    only_kw: 若非 None，只量化名字包含任一关键字的层（用于"仅 MLP / 仅 attention"消融）；
             None 表示量化全部（除 skip_kw 外）。
    """
    skip_kw = ("lm_head", "embed_tokens", "embed")
    stats = []
    t0 = time.time()
    n_skip = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(k in name for k in skip_kw):
            continue
        if only_kw is not None and not any(k in name for k in only_kw):
            n_skip += 1
            continue
        W = module.weight.data
        if W.dim() != 2 or W.shape[1] % qz.k != 0:
            continue  # 非常规形状或输入维非 k 倍数，跳过
        state = qz.quantize(W, chunk=chunk)
        W_hat = qz.dequantize(state)
        rel = float((W - W_hat).norm() / W.norm())
        module.weight.data = W_hat
        stats.append((name, tuple(W.shape), rel))
    dt = time.time() - t0
    return stats, dt, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--a", type=int, default=14)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="npu")
    ap.add_argument("--cache", default="/opt/zh/q38quant/codebooks/pcdvq_k8_a14_b2.pt")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--no-trust-remote", action="store_true")
    args = ap.parse_args()

    device = "npu:0" if args.device == "npu" else "cpu"
    if args.device == "npu":
        log(f"npu device_count={torch.npu.device_count()}")
        torch.npu.set_device(0)

    # 1) 码本
    t0 = time.time()
    if os.path.exists(args.cache) and args.a == 14 and args.b == 2 and args.k == 8:
        qz = load_codebooks_from_cache(args.cache, device)
        log(f"码本从缓存加载: {args.cache}")
    else:
        qz = build_codebooks(args.k, args.a, args.b, device)
        log(f"码本构造 (k={args.k}, a={args.a}, b={args.b}) 耗时 {time.time()-t0:.1f}s")
    log(f"BPW = (a+b)/k = {(args.a + args.b) / args.k}")

    # 2) 模型
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=not args.no_trust_remote,
        torch_dtype=torch.float32,
    )
    model.to(device)
    log(f"模型加载 {args.model} 耗时 {time.time()-t0:.1f}s")
    cfg = model.config
    log(f"架构: hidden={getattr(cfg, 'hidden_size', '?')} "
        f"layers={getattr(cfg, 'num_hidden_layers', '?')} "
        f"intermediate={getattr(cfg, 'intermediate_size', '?')}")

    # 3) 量化
    stats, dt, n_skip = quantize_model(model, qz, chunk=args.chunk)
    log(f"\n量化完成: {len(stats)} 层 Linear 已量化, 跳过 {n_skip} 层, 总耗时 {dt:.1f}s")
    if stats:
        rels = [r for _, _, r in stats]
        # 按参数规模加权平均相对误差（Frobenius 是逐层范数比，这里用算术平均作参考）
        log(f"相对 Frobenius 误差: min={min(rels):.4f} max={max(rels):.4f} mean={sum(rels)/len(rels):.4f}")
        log("\n逐层误差（前 12 层 + 后 5 层）:")
        for name, shp, rel in stats[:12]:
            log(f"  {name:<48} {str(shp):<18} rel={rel:.4f}")
        if len(stats) > 17:
            log("  ...")
            for name, shp, rel in stats[-5:]:
                log(f"  {name:<48} {str(shp):<18} rel={rel:.4f}")

    # 4) 保存 fake-quant checkpoint
    if args.save_dir:
        t0 = time.time()
        model.to("cpu")
        model.save_pretrained(args.save_dir)
        log(f"\n已保存 fake-quant checkpoint 到 {args.save_dir} 耗时 {time.time()-t0:.1f}s")

    log("model_quant 完成")


if __name__ == "__main__":
    main()