#!/usr/bin/env python3
"""SLQ bitwidth allocation: ILP + binary search for DL (EAR) and TL (Algorithm 2) targets."""

import argparse
import dataclasses
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch_npu  # noqa: F401  (NPU backend)
from ortools.linear_solver import pywraplp

from src.data.loading import MASKED_DATASETS, get_data
from src.metrics.evaluation import compute_baseline_topk, compute_kl_and_ear
from src.models.loading import get_tokenizer, load_model
from src.sensitivity.database import ErrorDatabase, LayerGroup, LinearSensitivityData, ShapleySensitivityData
from src.utils.common import fix_seed, get_layer_param_counts, set_layer_weights
from src.utils.weight_store import WeightStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SOLVER_TIME_LIMIT_MS = 600 * 1000
GUARDRAIL_MAX_DRIFT = 2.0

Sensitivity = Union[LinearSensitivityData, ShapleySensitivityData]


def parse_bitwidth_map(s: str) -> Dict[int, float]:
    """Parse "4:4.156,8:8.156" into {nominal_bits: actual_bits}."""
    return {int(k): float(v) for k, v in (p.split(':') for p in s.split(',') if p.strip())}


def get_actual_bitwidth(nominal: int, bitwidth_map: Dict[int, float]) -> float:
    """Actual bits per parameter of a nominal bitwidth."""
    return bitwidth_map.get(nominal, float(nominal))


def load_sensitivity(filepath: str, cls):
    """Load a linear_*.json or shapley_*.json database as cls."""
    with open(filepath) as f:
        d = json.load(f)
    # Grouped values must be stored as per-layer shares (phi_group / |group|) to be summed per layer.
    if d.get('layer_groups') and not d.get('per_layer_normalized', False):
        raise ValueError(f"{filepath}: per_layer_normalized is false; rebuild it with slq_build_database.py")
    for k, v in d.items():
        if k.endswith('_per_level') and v:
            d[k] = {n: {int(b): c for b, c in bv.items()} for n, bv in v.items()}
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def find_database(layer_dir: Path, prefix: str, explicit: Optional[str]) -> str:
    """Return the explicit path, or the single <prefix>_*.json in layer_dir."""
    if explicit:
        return explicit
    candidates = sorted(layer_dir.glob(f"{prefix}_*.json"))
    if len(candidates) != 1:
        names = ", ".join(c.name for c in candidates) or "none"
        raise ValueError(f"Need exactly one {prefix}_*.json in {layer_dir} (found: {names}); pick one with --{prefix}-db")
    logger.info(f"Using {prefix} database: {candidates[0]}")
    return str(candidates[0])


def extract_layer_groups(sens: Sensitivity) -> Optional[List[LayerGroup]]:
    """Layer groups stored in the sensitivity database, if any."""
    return [LayerGroup(**g) for g in sens.layer_groups] if sens.layer_groups else None


def layer_costs(error_db: ErrorDatabase, sens: Sensitivity, metric: str) -> Dict[str, Dict[int, float]]:
    """Per-layer cost table c[name][bw] for metric 'kl' or 'ear' (EAR loss)."""
    if isinstance(sens, ShapleySensitivityData):
        table = getattr(sens, f'{metric}_shapley_per_level') or {}
        # 小校准集下 Shapley 边际可能为负（噪声）；成本物理上非负，且应随位宽单调不增。
        out = {}
        for _n, _costs in table.items():
            _bits = sorted(_costs)
            _vals = {b: max(0.0, float(_costs[b])) for b in _bits}
            for _i in range(len(_bits) - 2, -1, -1):
                _vals[_bits[_i]] = max(_vals[_bits[_i]], _vals[_bits[_i + 1]])
            out[_n] = _vals
        return out
    alphas = getattr(sens, f'{metric}_alphas')
    return {n: {b: e * alphas.get(n, 0.0) for b, e in errs.items()} for n, errs in error_db.layers.items()}


def solve_allocation(
    error_db: ErrorDatabase,
    sens: Sensitivity,
    target_bitwidth: float,
    available_bw: List[int],
    kl_weight: float = 1.0,
    ear_weight: float = 0.0,
    bitwidth_map: Optional[Dict[int, float]] = None,
    layer_groups: Optional[List[LayerGroup]] = None,
) -> Dict[str, int]:
    """Multiple-choice knapsack ILP: one bitwidth per group, average bits <= target."""
    bitwidth_map = bitwidth_map or {}
    groups = layer_groups or [LayerGroup(i, [n], n) for i, n in enumerate(error_db.layer_names)]
    tables = [(w, layer_costs(error_db, sens, m))
              for m, w in (('kl', kl_weight), ('ear', ear_weight)) if w]

    def cost(group, bw):
        return sum(w * t.get(n, {}).get(bw, 0.0) for w, t in tables for n in group.layer_names)

    params = [sum(error_db.param_counts.get(n, 1) for n in g.layer_names) for g in groups]
    n_g, n_b = len(groups), len(available_bw)
    n_var = n_g * n_b
    import numpy as _np
    from scipy.optimize import milp, LinearConstraint, Bounds

    c = _np.zeros(n_var)
    for i, g in enumerate(groups):
        for j, b in enumerate(available_bw):
            c[i * n_b + j] = cost(g, b)

    A_eq = _np.zeros((n_g, n_var))
    for i in range(n_g):
        A_eq[i, i * n_b:(i + 1) * n_b] = 1.0

    A_ub = _np.zeros((1, n_var))
    for i in range(n_g):
        for j, b in enumerate(available_bw):
            A_ub[0, i * n_b + j] = get_actual_bitwidth(b, bitwidth_map) * params[i]

    A = _np.vstack([A_eq, A_ub])
    lb = _np.concatenate([_np.ones(n_g), [-_np.inf]])
    ub = _np.concatenate([_np.ones(n_g), [target_bitwidth * sum(params)]])
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=_np.ones(n_var), bounds=Bounds(0, 1),
               options={'time_limit': SOLVER_TIME_LIMIT_MS / 1000.0})
    if res.x is None:
        raise RuntimeError(f"ILP found no solution at average bitwidth {target_bitwidth:.3f} ({res.message})")
    if res.status != 0:
        logger.warning(f"ILP status {res.status}: {res.message}; using best feasible solution")
    x_arr = res.x.reshape(n_g, n_b)
    return {n: available_bw[j] for i, g in enumerate(groups) for j in range(n_b)
            if x_arr[i, j] > 0.5 for n in g.layer_names}


def predict_metrics_from_config(config: Dict[str, int], error_db: ErrorDatabase,
                                sens: Sensitivity) -> Tuple[float, float]:
    """Predicted (KL, EAR) as sums of per-layer costs (App. A.2)."""
    kl, ear_loss = (layer_costs(error_db, sens, m) for m in ('kl', 'ear'))
    return (sum(kl.get(n, {}).get(b, 0.0) for n, b in config.items()),
            1.0 - sum(ear_loss.get(n, {}).get(b, 0.0) for n, b in config.items()))


def compute_avg_bitwidth(config: Dict[str, int], param_counts: Dict[str, int],
                         bitwidth_map: Optional[Dict[int, float]] = None) -> float:
    """Parameter-weighted average of actual bits per parameter."""
    bitwidth_map = bitwidth_map or {}
    total = sum(param_counts.get(n, 1) for n in config)
    bits = sum(get_actual_bitwidth(b, bitwidth_map) * param_counts.get(n, 1) for n, b in config.items())
    return bits / total if total else 0.0


def distribution_lines(config: Dict[str, int]) -> List[str]:
    """One 'b-bit: n layers (p%)' line per bitwidth, highest first."""
    dist = Counter(config.values())
    return [f"{b}-bit: {dist[b]} layers ({100.0 * dist[b] / len(config):.1f}%)" for b in sorted(dist, reverse=True)]


def write_config(path, header: Dict[str, object], config: Dict[str, int]):
    """Write '# Key: value' header lines, the commented distribution and the 'layer: bits' lines."""
    with open(path, 'w') as f:
        f.writelines(f"# {k}: {v}\n" for k, v in header.items())
        f.write("#\n")
        f.write("# Bitwidth distribution:\n")
        f.writelines(f"#  {line}\n" for line in distribution_lines(config))
        f.write("#\n")
        f.writelines(f"{n}: {b}\n" for n, b in config.items())
    logger.info(f"Config saved to: {path}")


def apply_config(model, config: Dict[str, int], weight_store: WeightStore):
    """Load each layer's quantized weights at its configured bitwidth into the model."""
    for name, bw in config.items():
        set_layer_weights(model, name, weight_store.get_layer_weight(name, bw))


def load_calibration(args, tokenizer) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Calibration sequences as one tensor; masks only for MASKED_DATASETS."""
    result = get_data(args.calibration_data, args.calibration_samples * args.calibration_seq_length,
                      args.calibration_seq_length, tokenizer, train=True, seed=args.seed)
    if args.calibration_data not in MASKED_DATASETS:
        return torch.cat(result, dim=0), None
    data, masks = result
    return torch.cat(data, dim=0), torch.cat(masks, dim=0)


def load_measure(args, error_db: ErrorDatabase, top_k: int) -> Callable[[Dict[str, int]], Tuple[float, float]]:
    """Load the model and calibration set; return config -> measured (KL, EAR), one forward pass each."""
    weight_store = WeightStore(args.layer_dir).load()
    model = load_model(args.model)
    error_db.param_counts = get_layer_param_counts(model, error_db.layer_names)
    data, masks = load_calibration(args, get_tokenizer(args.model))
    ref_vals, ref_idx = compute_baseline_topk(model, data, args.batch_size, top_k)

    def measure(config):
        apply_config(model, config, weight_store)
        return compute_kl_and_ear(model, ref_vals, ref_idx, data, args.batch_size, masks=masks)[:2]
    return measure


def bisect_budget(solve, evaluate, accept, low: float, high: float, tolerance: float):
    """Smallest ILP budget whose config passes accept; (config, metrics), falling back to budget high."""
    best = None
    while high - low > tolerance:
        mid = (low + high) / 2
        config = solve(mid)
        passed = accept(metrics := evaluate(config))
        logger.info(f"budget={mid:.3f}: {'pass' if passed else 'fail'}")
        if passed:
            best, high = (config, metrics), mid
        else:
            low = mid
    if best is None:
        config = solve(high)
        best = (config, evaluate(config))
    return best


def task_lossless_search(
    error_db: ErrorDatabase,
    shapley_data: ShapleySensitivityData,
    solve: Callable[[float], Dict[str, int]],
    bw_range: Tuple[float, float],
    calibration_bitwidth: int,
    calibration_recovery: float,
    calibration_actual_kl: float,
    min_recovery: float,
    tolerance: float,
) -> Tuple[Dict[str, int], float, float, float, float]:
    """Algorithm 2: returns (config, calibrated KL, rho, alpha, KL threshold)."""
    if not 0.0 < calibration_recovery < 1.0 or calibration_actual_kl <= 0:
        raise ValueError(f"Calibration anchor must show degradation (recovery={calibration_recovery}, "
                         f"KL={calibration_actual_kl}); use a lower --calibration-bitwidth")
    if not 0.0 < min_recovery < 1.0:
        raise ValueError(f"--min-recovery must be in (0, 1), got {min_recovery}")

    cal_config = {n: calibration_bitwidth for n in error_db.layer_names}
    pred_kl_cal = predict_metrics_from_config(cal_config, error_db, shapley_data)[0]
    if pred_kl_cal <= 0:
        raise ValueError(f"Predicted KL at uniform {calibration_bitwidth}-bit is {pred_kl_cal:.6g}; "
                         f"the Shapley database has no cost at the anchor, use a lower --calibration-bitwidth")
    rho = calibration_actual_kl / pred_kl_cal
    alpha = (1.0 - calibration_recovery) / calibration_actual_kl
    threshold = (1.0 - min_recovery) / alpha
    logger.info(f"rho={rho:.4f}, recovery = 1 - {alpha:.4f} * KL, KL threshold={threshold:.6f}")

    def evaluate(config):
        cal_kl = rho * predict_metrics_from_config(config, error_db, shapley_data)[0]
        logger.info(f"cal_KL={cal_kl:.6f}, pred_recovery={1.0 - alpha * cal_kl:.4f}")
        return cal_kl

    config, cal_kl = bisect_budget(solve, evaluate, lambda kl: kl <= threshold, *bw_range, tolerance)
    return config, cal_kl, rho, alpha, threshold


def recovery_from_results(baseline_json: str, anchor_json: str) -> float:
    """Mean per-benchmark score ratio between two eval/run_lmeval.py results.json files."""
    base, anchor = (json.load(open(p))['averages'] for p in (baseline_json, anchor_json))
    labels = [l for l in base if l in anchor and base[l] > 0]
    if set(labels) != set(base) | set(anchor):
        raise ValueError(f"{baseline_json} and {anchor_json} cover different benchmarks: {sorted(base)} vs {sorted(anchor)}")
    if not labels:
        raise ValueError(f"No common benchmarks in {baseline_json} and {anchor_json}")
    return sum(anchor[l] / base[l] for l in labels) / len(labels)


def run_task_lossless(args, error_db, shapley_data, solve, bw_range, header) -> Tuple[Dict[str, int], bool]:
    """--mode tl: calibrate, search (Algorithm 2) and run the 2x guardrail."""
    if not isinstance(shapley_data, ShapleySensitivityData):
        raise ValueError("--mode tl requires --sensitivity-method shapley")
    if args.calibration_results:
        args.calibration_recovery = recovery_from_results(*args.calibration_results)
        logger.info(f"Recovery of the uniform {args.calibration_bitwidth}-bit anchor: {args.calibration_recovery:.4f}")
    if args.calibration_recovery is None:
        raise ValueError("--mode tl needs --calibration-results or --calibration-recovery")

    cal_kl = args.calibration_kl
    measure = None
    if cal_kl is None or not args.skip_guardrail:
        # Use the top-K the Shapley database was built with so rho compares like with like.
        measure = load_measure(args, error_db, shapley_data.top_k)
    if cal_kl is None:
        cal_kl = measure({n: args.calibration_bitwidth for n in error_db.layer_names})[0]
        logger.info(f"Actual KL at uniform {args.calibration_bitwidth}-bit: {cal_kl:.6f}")

    config, final_kl, rho, alpha, threshold = task_lossless_search(
        error_db, shapley_data, solve, bw_range, args.calibration_bitwidth,
        args.calibration_recovery, cal_kl, args.min_recovery, args.bitwidth_tolerance)
    ok = final_kl <= threshold

    # Out-of-regime guardrail (Sec. 3.3): actual/predicted KL at the result must stay within 2x of rho.
    guardrail = "skipped"
    pred_kl = predict_metrics_from_config(config, error_db, shapley_data)[0]
    if measure is not None and pred_kl > 0:
        drift = measure(config)[0] / pred_kl / rho
        passed = 1.0 / GUARDRAIL_MAX_DRIFT <= drift <= GUARDRAIL_MAX_DRIFT
        guardrail = f"{'pass' if passed else 'fail'} (drift {drift:.2f}x)"
        ok = ok and passed
        if not passed:
            logger.error("Guardrail failed: the anchor lies outside the near-lossless regime; "
                         "re-anchor at a different --calibration-bitwidth (4 or 6 recommended)")

    header.update({
        'Min recovery': args.min_recovery,
        'Calibration bitwidth': args.calibration_bitwidth,
        'Calibration recovery': f"{args.calibration_recovery:.4f}",
        'Calibration actual KL': f"{cal_kl:.6f}",
        'Alpha': f"{alpha:.4f}",
        'KL threshold': f"{threshold:.6f}",
        'Calibrated KL': f"{final_kl:.6f}",
        'Predicted recovery': f"{1.0 - alpha * final_kl:.4f}",
        'Guardrail (2x drift check)': guardrail,
    })
    return config, ok


def run_distribution_lossless(args, error_db, sens, solve, bw_range, header) -> Tuple[Dict[str, int], bool]:
    """--mode dl: fixed --target-bitwidth, or binary search for EAR >= --min-ear (measured unless predicted)."""
    predicted = args.use_predicted_constraints or args.target_bitwidth is not None
    measure = (lambda c: predict_metrics_from_config(c, error_db, sens)) if predicted \
        else load_measure(args, error_db, args.top_k)

    def evaluate(config):
        kl, ear = measure(config)
        logger.info(f"KL={kl:.6f}, EAR={ear:.4f}")
        return kl, ear

    if args.target_bitwidth is not None:
        config = solve(args.target_bitwidth)
        (kl, ear), ok = evaluate(config), True
        header['Target bitwidth'] = args.target_bitwidth
    else:
        config, (kl, ear) = bisect_budget(solve, evaluate, lambda m: m[1] >= args.min_ear, *bw_range,
                                          args.bitwidth_tolerance)
        ok = ear >= args.min_ear
        header['Constraint min_ear'] = args.min_ear
    kind = "Predicted" if predicted else "Final"
    header.update({f'{kind} KL': f"{kl:.6f}", f'{kind} EAR': f"{ear:.6f}"})
    return config, ok


def main():
    p = argparse.ArgumentParser(description="SLQ bitwidth search (ILP + binary search)")
    p.add_argument('--model', required=True)
    p.add_argument('--layer-dir', required=True)
    p.add_argument('--mode', default='dl', choices=['dl', 'tl'], help="dl: EAR target; tl: Algorithm 2")
    p.add_argument('--sensitivity-method', default='shapley', choices=['linear', 'shapley'])
    p.add_argument('--linear-db', help="default: the single linear_*.json in --layer-dir")
    p.add_argument('--shapley-db', help="default: the single shapley_*.json in --layer-dir")
    p.add_argument('--kl-weight', type=float, help="ILP weight on KL costs (default: 1 in tl, 0 in dl)")
    p.add_argument('--ear-weight', type=float, help="ILP weight on EAR costs (default: 0 in tl, 1 in dl)")
    p.add_argument('--min-ear', type=float, help="dl: minimum EAR")
    p.add_argument('--target-bitwidth', type=float, help="dl: solve once at this average bitwidth")
    p.add_argument('--use-predicted-constraints', action='store_true',
                   help="dl: check EAR predicted from the database instead of measuring it")
    p.add_argument('--min-recovery', type=float, default=0.99, help="tl: target task recovery")
    p.add_argument('--calibration-bitwidth', type=int, default=4, help="tl: uniform anchor bitwidth")
    anchor = p.add_mutually_exclusive_group()
    anchor.add_argument('--calibration-results', nargs=2, metavar=('BF16_JSON', 'ANCHOR_JSON'),
                   help="tl: eval/run_lmeval.py results.json of BF16 and of the uniform anchor")
    anchor.add_argument('--calibration-recovery', type=float, help="tl: anchor recovery vs BF16, if already known")
    p.add_argument('--calibration-kl', type=float, help="tl: actual KL of the anchor (default: measured)")
    p.add_argument('--skip-guardrail', action='store_true', help="tl: skip the 2x drift check")
    p.add_argument('--bitwidth-tolerance', type=float, default=0.01)
    p.add_argument('--bitwidths', help="comma-separated subset, e.g. 4,8 (default: all in --layer-dir)")
    p.add_argument('--bitwidth-map', default='', help="nominal:actual bits, e.g. '4:4.156,8:8.156' (Table 7)")
    p.add_argument('--disable-grouping', action='store_true', help="ignore layer groups in the database")
    p.add_argument('--calibration-data', default='fineweb_edu')
    p.add_argument('--calibration-samples', type=int, default=512)
    p.add_argument('--calibration-seq-length', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output')
    args = p.parse_args()

    tl = args.mode == 'tl'
    if args.kl_weight is None:
        args.kl_weight = 1.0 if tl else 0.0
    if args.ear_weight is None:
        args.ear_weight = 0.0 if tl else 1.0
    if not tl and (args.min_ear is None) == (args.target_bitwidth is None):
        p.error("--mode dl needs exactly one of --min-ear or --target-bitwidth")

    fix_seed(args.seed)
    layer_dir = Path(args.layer_dir)
    bitwidth_map = parse_bitwidth_map(args.bitwidth_map)
    error_db = ErrorDatabase.load(str(layer_dir / "error_db.json"))
    method = args.sensitivity_method
    cls = ShapleySensitivityData if method == 'shapley' else LinearSensitivityData
    sens = load_sensitivity(find_database(layer_dir, method, getattr(args, f'{method}_db')), cls)

    available_bw = error_db.get_common_bitwidths()
    if args.bitwidths:
        available_bw = [b for b in available_bw if b in {int(s) for s in args.bitwidths.split(',')}]
    if isinstance(sens, ShapleySensitivityData):
        available_bw = [b for b in available_bw if b in sens.bitwidths]
    if not available_bw:
        p.error("no requested bitwidth is in both --layer-dir and the sensitivity database")
    bw_range = tuple(get_actual_bitwidth(f(available_bw), bitwidth_map) for f in (min, max))
    if args.target_bitwidth is not None and args.target_bitwidth <= bw_range[0]:
        p.error(f"--target-bitwidth must exceed the minimum available bitwidth {bw_range[0]}")
    logger.info(f"Available bitwidths: {available_bw}, bitwidth map: {bitwidth_map or 'none'}")

    groups = None if args.disable_grouping else extract_layer_groups(sens)

    def solve(budget):
        if not error_db.param_counts:
            raise ValueError("error_db.json has no param_counts and the model was not loaded")
        return solve_allocation(error_db, sens, budget, available_bw, args.kl_weight, args.ear_weight,
                                bitwidth_map, groups)

    header = {'Model': args.model, 'Mode': args.mode, 'Sensitivity method': method,
              'Available bitwidths': available_bw, 'Bitwidth map': bitwidth_map or 'none',
              'Layer groups': len(groups) if groups else 'none',
              'Weights': f"kl={args.kl_weight}, ear={args.ear_weight}"}
    run = run_task_lossless if tl else run_distribution_lossless
    config, ok = run(args, error_db, sens, solve, bw_range, header)

    avg_bw = compute_avg_bitwidth(config, error_db.param_counts, bitwidth_map)
    header.update({'Average bitwidth': f"{avg_bw:.4f}", 'Satisfies constraints': ok})
    logger.info("Bitwidth distribution: " + ", ".join(distribution_lines(config)))
    logger.info(f"Average bitwidth {avg_bw:.3f}, satisfies constraints: {ok}")

    if tl:
        tag = f"rec{args.min_recovery}_cal{args.calibration_bitwidth}bit"
    elif args.target_bitwidth is not None:
        tag = f"{method}_target{args.target_bitwidth:.2f}"
    else:
        tag = f"{method}_{'predicted' if args.use_predicted_constraints else 'measured'}_ear{args.min_ear}"
    bws = '-'.join(map(str, available_bw))
    write_config(args.output or layer_dir / f"config_{args.mode}_bws{bws}_{tag}_bw{avg_bw:.2f}"
                 f"{'_grouped' if groups else ''}_seed{args.seed}.txt", header, config)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
