"""SAGE 全模型 fake-quant（NPU/CPU 通用）。

流程（对齐论文 Algorithm 1 与官方 ParoQuant 管线）：
  1. 加载模型，选取目标 Linear 层（默认除 lm_head/embed 外的所有 *_proj）。
  2. 校准前向（一次）：用 forward hook 缓存每层的输入激活与全精度输出（teacher）。
  3. 逐层量化：
       - 由该层输入估计通道能量 e_c；
       - `SAGEQuantizer.prepare`：ESG 分组 +（ABS）选切分 + 拟合，得到 fake-quant 权重；
       - 若 epochs>0：`blockwise_optimize` 做误差补偿（输出 MSE 目标，STE 更新半径/码本角）；
       - 回写量化权重到模块。
  4. 可选保存 fake-quant checkpoint。

序贯模式（--sequential）：与论文一致，量化某层前用"已量化前缀"重新前向捕获该层
学生输入（x_q）；teacher 目标仍来自初始全精度前向。默认关闭（用全精度输入，速度快）。

运行示例：
  # 本地/CPU 小规模冒烟
  python model_quant.py --device cpu --model Qwen/Qwen3-0.6B --calib-seqs 4 --calib-len 256 \
      --kmeans-iters 8 --epochs 5
  # NPU
  ASCEND_RT_VISIBLE_DEVICES=5 python model_quant.py --device npu --model /root/.cache/.../Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from sage import BlockwiseConfig, SAGEConfig, SAGEQuantizer, Split, channel_energy, blockwise_optimize
from sage.utils import log, resolve_device

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ATTN_KEYWORDS = ("self_attn", "linear_attn", "attention")
SKIP_KEYWORDS = ("lm_head", "embed_tokens", "embed")


def parse_splits(spec: str) -> tuple[Split, ...]:
    """解析 '64,4;128,3;256,2' → (Split(64,4), Split(128,3), Split(256,2))。"""
    out = []
    for part in spec.split(";"):
        k, v = part.split(",")
        out.append(Split(int(k), int(v)))
    return tuple(out)


def target_linear_names(model, only: str = "all") -> list[str]:
    """选取待量化的 Linear 层名（跳过 lm_head/embed，可按 mlp/attn 过滤）。"""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(k in name for k in SKIP_KEYWORDS):
            continue
        if only == "mlp" and "mlp" not in name:
            continue
        if only == "attn" and not any(k in name for k in ATTN_KEYWORDS):
            continue
        names.append(name)
    return names


def build_calibration_batches(tokenizer, n_seqs: int, seq_len: int, batch: int, device, text_path: str | None = None):
    """构造定长校准序列批次。

    text_path 为 None 时使用 WikiText-2（HF 缓存离线）；否则读取本地文本文件。
    """
    if text_path:
        with open(text_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    need = n_seqs * seq_len
    if ids.numel() < need:
        raise ValueError(f"校准文本过短：需要 {need} token，仅 {ids.numel()}")
    ids = ids[:need].reshape(n_seqs, seq_len)
    batches = [{"input_ids": ids[i:i + batch].to(device)} for i in range(0, n_seqs, batch)]
    return batches, ids


@torch.no_grad()
def capture(model, names: list[str], batches, with_output: bool):
    """前向一次，缓存指定层的输入激活（与全精度输出）。返回 CPU 张量字典。"""
    in_store = {n: [] for n in names}
    out_store = {n: [] for n in names} if with_output else None
    hooks = []
    for n in names:
        module = model.get_submodule(n)

        def pre_hook(mod, args, n=n):
            in_store[n].append(args[0].detach().to("cpu", copy=True))

        hooks.append(module.register_forward_pre_hook(pre_hook))
        if with_output:
            def out_hook(mod, args, out, n=n):
                out_store[n].append(out.detach().to("cpu", copy=True) if torch.is_tensor(out)
                                    else out[0].detach().to("cpu", copy=True))

            hooks.append(module.register_forward_hook(out_hook))
    for b in batches:
        model(**b)
    for h in hooks:
        h.remove()
    inputs = {n: torch.cat(in_store[n], dim=0) for n in names}
    targets = ({n: torch.cat(out_store[n], dim=0) for n in names} if with_output else None)
    return inputs, targets


def quantize_model(
    model,
    config: SAGEConfig,
    blockwise: BlockwiseConfig | None,
    names: list[str],
    device: torch.device,
    sequential: bool = False,
    log_fn=log,
):
    """逐层 SAGE 量化 + 可选 blockwise；返回逐层统计。"""
    # 1) 初始全精度前向：缓存每层输入（FP 输入）与全精度输出（teacher 目标）
    batches = model._sage_batches  # 由调用方挂载，避免重复构造
    t0 = time.time()
    fp_inputs, targets = capture(model, names, batches, with_output=True)
    log_fn(f"校准前向完成（{len(names)} 层），耗时 {time.time()-t0:.1f}s")

    quantizer = SAGEQuantizer(config)
    stats = []
    t_all = time.time()
    for i, name in enumerate(names):
        module = model.get_submodule(name)
        W = module.weight.data
        if W.dim() != 2 or (W.shape[0] * W.shape[1]) % config.group_size != 0:
            log_fn(f"  [skip] {name} 形状 {tuple(W.shape)} 不满足 group_size 约束")
            continue
        din = W.shape[1]

        # 学生输入：默认全精度输入；序贯模式用"已量化前缀"重新前向捕获
        if sequential and i > 0:
            x_in, _ = capture(model, [name], batches, with_output=False)
            x_cpu = x_in[name]
        else:
            x_cpu = fp_inputs[name]

        x_dev = x_cpu.to(device)
        energy = channel_energy(x_dev)
        t1 = time.time()
        state = quantizer.prepare(W, energy)
        fit_dt = time.time() - t1

        bw_dt = 0.0
        if blockwise is not None and blockwise.epochs > 0:
            t2 = time.time()
            y_dev = targets[name].to(device)
            if y_dev.shape[1] != W.shape[0]:
                y_dev = y_dev[..., : W.shape[0]]
            state = blockwise_optimize(
                state, x_dev, y_dev, blockwise, subs_per_group=config.subs_per_group,
                log=lambda s: log_fn(s),
            )
            bw_dt = time.time() - t2
            # 补偿后重新评分
            state.score = quantizer.activation_weighted_error(W, energy, state)

        W_hat = quantizer.dequantize(state)
        rel = float((W - W_hat).norm() / W.norm())
        module.weight.data = W_hat.to(W.dtype)

        bpp = quantizer.bpp(state)
        stats.append({
            "name": name, "shape": tuple(W.shape), "split": state.split,
            "rel": rel, "score": state.score, "payload_bpw": bpp["payload"],
            "total_bpw": bpp["total"], "fit_s": fit_dt, "blockwise_s": bw_dt,
        })
        log_fn(f"  [{i+1}/{len(names)}] {name:<52} {str(tuple(W.shape)):<16} "
               f"split=({state.split.codebook_size},{state.split.radius_bits}) "
               f"rel={rel:.4f} bpp={bpp['payload']:.3f} fit={fit_dt:.1f}s bw={bw_dt:.1f}s")
        del x_dev, energy
        if device.type == "npu":
            torch.npu.empty_cache()

    log_fn(f"量化完成：{len(stats)} 层，总耗时 {time.time()-t_all:.1f}s")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="SAGE 全模型 fake-quant（NPU/CPU）")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    # SAGE 超参
    ap.add_argument("--splits", default="64,4;128,3;256,2", help="ABS 候选；单个即固定切分")
    ap.add_argument("--no-esg", action="store_true", help="关闭 ESG（均匀权重 + 恒等置换）")
    ap.add_argument("--no-adaptive-bits", action="store_true", help="关闭 ABS（用 --splits 第一个）")
    ap.add_argument("--energy-power", type=float, default=0.75)
    ap.add_argument("--kmeans-iters", type=int, default=20)
    ap.add_argument("--sample-size", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=8192)
    # blockwise
    ap.add_argument("--epochs", type=int, default=20, help="blockwise epoch；0 关闭误差补偿")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sequential", action="store_true", help="序贯学生输入（慢，更贴近论文）")
    # 校准
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-batch", type=int, default=4)
    ap.add_argument("--calib-text", default=None, help="本地校准文本文件；默认用 WikiText-2")
    # 其它
    ap.add_argument("--only", default="all", choices=["all", "mlp", "attn"])
    ap.add_argument("--limit-layers", type=int, default=0, help=">0 时只量化前 N 层（快速冒烟）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--trust-remote", action="store_true", default=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"加载模型 {args.model} (dtype={args.dtype}, device={device}) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote, dtype=getattr(torch, args.dtype),
    )
    model.to(device).eval()
    log(f"模型加载完成，耗时 {time.time()-t0:.1f}s")

    names = target_linear_names(model, args.only)
    if args.limit_layers > 0:
        names = names[: args.limit_layers]
    log(f"目标 Linear 层数: {len(names)} (only={args.only})")

    batches, _ = build_calibration_batches(
        tokenizer, args.calib_seqs, args.calib_len, args.calib_batch, device, args.calib_text
    )
    model._sage_batches = batches  # noqa: SLF001  传给 quantize_model

    config = SAGEConfig(
        v=8, group_size=128, energy_power=args.energy_power,
        splits=parse_splits(args.splits), esg=not args.no_esg,
        adaptive_bits=not args.no_adaptive_bits, kmeans_iters=args.kmeans_iters,
        sample_size=args.sample_size, seed=args.seed, chunk=args.chunk,
    )
    blockwise = BlockwiseConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                seed=args.seed) if args.epochs > 0 else None

    stats = quantize_model(model, config, blockwise, names, device, args.sequential)

    if stats:
        avg_rel = sum(s["rel"] for s in stats) / len(stats)
        avg_bpp = sum(s["payload_bpw"] for s in stats) / len(stats)
        log(f"\n汇总：{len(stats)} 层，平均相对 Frobenius 误差 {avg_rel:.4f}，"
            f"平均 payload {avg_bpp:.3f} bpw")

    if args.save_dir:
        t0 = time.time()
        model.to("cpu")
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        log(f"fake-quant checkpoint 已保存至 {args.save_dir}（{time.time()-t0:.1f}s）")

    log("model_quant 完成")


if __name__ == "__main__":
    main()
