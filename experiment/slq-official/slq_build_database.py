#!/usr/bin/env python3
"""Build the linear and multi-bitwidth Shapley sensitivity databases (Sec. 3.3, Algorithm 1).

Usage: see scripts/run_build_database.sh.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch_npu  # noqa: F401  (NPU backend)
from tqdm import tqdm

from src.data.loading import MASKED_DATASETS, get_data
from src.metrics.evaluation import compute_all_metrics, compute_baseline_topk
from src.models.grouping import build_layer_groups, parse_group_spec
from src.models.loading import get_tokenizer, load_model
from src.sensitivity.database import ErrorDatabase, LayerGroup, LinearSensitivityData, ShapleySensitivityData
from src.utils.common import compute_layer_error, fix_seed, get_layer_weights, set_layer_weights
from src.utils.gpu_cache import GPUWeightCache
from src.utils.weight_store import WeightStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_error_database(model, weight_store: WeightStore) -> ErrorDatabase:
    """Relative weight error of every stored quantized layer against the original model."""
    error_db = ErrorDatabase()
    for name in weight_store.get_layer_names():
        original = get_layer_weights(model, name)
        if original is None:
            continue
        error_db.param_counts[name] = original.numel()
        for bw in weight_store.get_available_bitwidths(name):
            error = compute_layer_error(original, weight_store.get_layer_weight(name, bw))
            error_db.add_layer_option(name, bw, error)
    return error_db


def estimate_linear_alphas(model, original_weights: Dict[str, torch.Tensor], groups: List[LayerGroup],
                           metric_fn, num_samples: int = 15):
    """Per-layer linear alphas: mean of metric degradation / t^2 over Gaussian noise levels t."""
    t_levels = np.linspace(0.01, 0.3, num_samples)
    nll_base, kl_base, ear_base = metric_fn()
    nll_alphas, kl_alphas, ear_alphas = {}, {}, {}

    for group in tqdm(groups, desc="Estimating alphas"):
        deltas = []
        for t in t_levels:
            for name in group.layer_names:
                orig = original_weights[name]
                set_layer_weights(model, name, orig + torch.randn_like(orig) * orig.std() * t)
            nll, kl, ear = metric_fn()
            deltas.append((nll - nll_base, kl - kl_base, ear_base - ear))
            for name in group.layer_names:
                set_layer_weights(model, name, original_weights[name])

        alphas = np.mean(np.array(deltas) / t_levels[:, None] ** 2, axis=0).clip(min=0.0)
        # Each layer stores alpha / |group| so summing per-layer costs over a group counts it once (Eqs. 5-6).
        alphas /= len(group.layer_names)
        for name in group.layer_names:
            nll_alphas[name], kl_alphas[name], ear_alphas[name] = map(float, alphas)

    return nll_alphas, kl_alphas, ear_alphas


def estimate_shapley_values(model, gpu_cache, groups: List[LayerGroup], bitwidths: List[int],
                            metric_fn, num_permutations: int = 50) -> np.ndarray:
    """Algorithm 1: [3 (NLL, KL, EAR), groups, bitwidths] Shapley values; the b_max column stays zero."""
    max_bw = max(bitwidths)
    shapley = np.zeros((3, len(groups), len(bitwidths)))
    sign = np.array([1.0, 1.0, -1.0])

    for level_idx, target_bw in enumerate(bitwidths):
        if target_bw == max_bw:
            continue
        for _ in tqdm(range(num_permutations), desc=f"Permutations ({max_bw}->{target_bw} bit)"):
            perm = np.random.permutation(len(groups))
            gpu_cache.swap_all_layers(model, {n: max_bw for g in groups for n in g.layer_names})
            prev = np.array(metric_fn())
            for g_idx in perm:
                for name in groups[g_idx].layer_names:
                    gpu_cache.swap_layer(model, name, target_bw)
                cur = np.array(metric_fn())
                shapley[:, g_idx, level_idx] += sign * (cur - prev) / num_permutations
                prev = cur
            (torch.npu.empty_cache() if hasattr(torch, "npu") else torch.cuda.empty_cache())

    return shapley


def per_layer_values(values: np.ndarray, groups: List[LayerGroup], bitwidths: List[int]) -> Dict[str, Dict[int, float]]:
    """Each layer stores phi_group / |group| so summing over a group's members counts it once (Eqs. 7-8)."""
    return {name: {bw: float(values[i, j]) / len(g.layer_names) for j, bw in enumerate(bitwidths)}
            for i, g in enumerate(groups) for name in g.layer_names}


def save_json(data, filepath: Path):
    d = asdict(data)
    with open(filepath, 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}, f, indent=2)
    logger.info(f"Saved {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Build sensitivity databases")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer-dir', type=str, required=True, help="Directory of quantized layer weights")
    parser.add_argument('--calibration-data', type=str, default='fineweb_edu',
                        )
    parser.add_argument('--calibration-samples', type=int, default=512)
    parser.add_argument('--calibration-seq-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-noise-samples', type=int, default=15, help="Noise levels for linear alphas")
    parser.add_argument('--skip-linear', action='store_true')
    parser.add_argument('--shapley-bitwidths', type=str, default='2,3,4,5,6,7,8')
    parser.add_argument('--num-permutations', type=int, default=50)
    parser.add_argument('--skip-shapley', action='store_true')
    parser.add_argument('--layer-groups', type=str,
                        default='mlp.gate_proj,mlp.up_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj',
                        help="Space-separated groups of comma-separated patterns sharing a bitwidth; '' for per-layer")
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument('--gpu-cache', type=str, default='off', choices=['off', 'lazy', 'eager'],
                        help="Keep swapped-in weights on GPU (lazy) or preload all of them (eager)")
    args = parser.parse_args()

    fix_seed(args.seed)
    layer_dir = Path(args.layer_dir)
    use_grouping = bool(args.layer_groups.strip())
    group_suffix = f"_grp{hashlib.md5(args.layer_groups.encode()).hexdigest()[:8]}" if use_grouping else ""
    tag = (f"topk{args.top_k}_{Path(args.calibration_data).stem}_"
           f"{args.calibration_samples}x{args.calibration_seq_length}")
    bitwidths = sorted(int(b) for b in args.shapley_bitwidths.split(','))
    error_db_path = layer_dir / "error_db.json"
    baseline_path = layer_dir / f"baseline_{tag}_seed{args.seed}.pt"
    linear_path = layer_dir / f"linear_{tag}_{args.num_noise_samples}noise{group_suffix}_seed{args.seed}.json"
    shapley_path = layer_dir / (f"shapley_{tag}_{args.num_permutations}perms_bw{'-'.join(map(str, bitwidths))}"
                                f"{group_suffix}_seed{args.seed}.json")

    weight_store = WeightStore(str(layer_dir)).load()
    model = load_model(args.model)
    tokenizer = get_tokenizer(args.model)

    data = get_data(args.calibration_data, args.calibration_samples * args.calibration_seq_length,
                    args.calibration_seq_length, tokenizer, train=True, seed=args.seed)
    masks = None
    if args.calibration_data in MASKED_DATASETS:
        data, masks = data
        masks = torch.cat(masks, dim=0)
    calibration_data = torch.cat(data, dim=0)

    if error_db_path.exists() and not args.force_recompute:
        error_db = ErrorDatabase.load(str(error_db_path))
    else:
        error_db = build_error_database(model, weight_store)
        error_db.save(str(error_db_path))

    if baseline_path.exists() and not args.force_recompute:
        baseline = torch.load(str(baseline_path))
        topk_vals, topk_idx = baseline['topk_vals'], baseline['topk_idx']
    else:
        topk_vals, topk_idx = compute_baseline_topk(model, calibration_data, args.batch_size, args.top_k)
        torch.save({'topk_vals': topk_vals, 'topk_idx': topk_idx}, str(baseline_path))

    def metric_fn():
        return compute_all_metrics(model, calibration_data, topk_vals, topk_idx, args.batch_size, masks=masks)

    group_patterns = parse_group_spec(args.layer_groups) if use_grouping else []
    groups = build_layer_groups(error_db.layer_names, group_patterns)
    logger.info(f"{len(error_db.layer_names)} layers in {len(groups)} groups")
    meta = dict(layer_names=error_db.layer_names, top_k=args.top_k, calibration_dataset=args.calibration_data,
                calibration_samples=args.calibration_samples, seed=args.seed, per_layer_normalized=True,
                layer_groups=[asdict(g) for g in groups] if use_grouping else None)

    if not args.skip_linear and (args.force_recompute or not linear_path.exists()):
        original_weights = {n: get_layer_weights(model, n).cpu().clone() for n in error_db.layer_names}
        nll_a, kl_a, ear_a = estimate_linear_alphas(model, original_weights, groups, metric_fn,
                                                    args.num_noise_samples)
        save_json(LinearSensitivityData(nll_alphas=nll_a, kl_alphas=kl_a, ear_alphas=ear_a,
                                        num_noise_samples=args.num_noise_samples, **meta), linear_path)

    if not args.skip_shapley and (args.force_recompute or not shapley_path.exists()):
        gpu_cache = GPUWeightCache(model, weight_store, bitwidths, cache=args.gpu_cache != 'off',
                                   preload=args.gpu_cache == 'eager')
        nll_s, kl_s, ear_s = estimate_shapley_values(model, gpu_cache, groups, bitwidths, metric_fn,
                                                     args.num_permutations)
        save_json(ShapleySensitivityData(
            nll_shapley_values=nll_s, kl_shapley_values=kl_s, ear_shapley_values=ear_s,
            nll_shapley_per_level=per_layer_values(nll_s, groups, bitwidths),
            kl_shapley_per_level=per_layer_values(kl_s, groups, bitwidths),
            ear_shapley_per_level=per_layer_values(ear_s, groups, bitwidths),
            bitwidths=bitwidths, num_permutations=args.num_permutations, **meta), shapley_path)


if __name__ == "__main__":
    main()
