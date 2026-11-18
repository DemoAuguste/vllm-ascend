"""Stage ③: ILP budget scan -- one mixed-precision config per average-bitwidth budget.

Each config assigns every layer group one bitwidth from {2,4,6,8} such that the
parameter-weighted average (including the g128 scale/zero overhead, see
BITWIDTH_MAP) equals the requested budget while the predicted KL is minimised.

Run from the overlay root, e.g.:

    python scripts/make_configs.py --budgets 3.4,3.6,3.8,4.0

Paths default to the 27B layout and can be overridden with SLQ_ROOT / SLQ_LAYERS
or the matching command line flags.
"""
import argparse
import os
import sys
from pathlib import Path

SLQ_ROOT = Path(os.environ.get("SLQ_ROOT", "/opt/zh/slq-official"))
SNAPSHOT = os.environ.get("SLQ_SNAPSHOT", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
DEFAULT_LAYER_DIR = Path(
    os.environ.get("SLQ_LAYERS", SLQ_ROOT / "data" / "slq_27b" / SNAPSHOT / "4bit")
)

sys.path.insert(0, str(SLQ_ROOT))

from slq_search_dl import (  # noqa: E402
    compute_avg_bitwidth,
    distribution_lines,
    extract_layer_groups,
    load_sensitivity,
    predict_metrics_from_config,
    solve_allocation,
)
from src.sensitivity.database import ErrorDatabase, ShapleySensitivityData  # noqa: E402

BITWIDTHS = [2, 4, 6, 8]
BITWIDTH_MAP = {2: 2.156, 4: 4.156, 6: 6.156, 8: 8.156}


def resolve_shapley_db(layer_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = sorted(layer_dir.glob("shapley_*8perms*bw2-4-6-8*.json"))
    if not candidates:
        raise SystemExit(
            f"no 8-permutation Shapley database under {layer_dir}; "
            "run run_db27b.sh shapley (twice, different seeds) and merge_shapley.py first"
        )
    return str(candidates[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer-dir", default=str(DEFAULT_LAYER_DIR))
    ap.add_argument("--shapley-db", default=None,
                    help="default: first shapley_*8perms*.json under --layer-dir")
    ap.add_argument("--budgets", default="3.4,3.6,3.8,4.0",
                    help="comma separated average-bitwidth budgets")
    ap.add_argument("--out-dir", default=str(SLQ_ROOT / "configs"))
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    error_db = ErrorDatabase.load(str(layer_dir / "error_db.json"))
    sens = load_sensitivity(resolve_shapley_db(layer_dir, args.shapley_db), ShapleySensitivityData)
    groups = extract_layer_groups(sens)

    for budget in (float(b) for b in args.budgets.split(",")):
        cfg = solve_allocation(error_db, sens, budget, BITWIDTHS, kl_weight=1.0, ear_weight=0.0,
                               bitwidth_map=BITWIDTH_MAP, layer_groups=groups)
        avg = compute_avg_bitwidth(cfg, error_db.param_counts, BITWIDTH_MAP)
        pred_kl = predict_metrics_from_config(cfg, error_db, sens)[0]
        out = out_dir / f"config_tl27b_b{budget:.1f}.txt"
        with open(out, "w") as f:
            f.write("# Mode: tl-budget\n")
            f.write("# Sensitivity method: shapley\n")
            f.write(f"# Available bitwidths: {BITWIDTHS}\n")
            f.write(f"# Bitwidth map: {BITWIDTH_MAP}\n")
            f.write(f"# Layer groups: {len(groups)}\n")
            f.write(f"# Target nominal bits: {budget:.2f}\n")
            f.write(f"# Average bitwidth: {avg:.4f}\n")
            f.write(f"# Predicted KL: {pred_kl:.6f}\n")
            f.write("# Bitwidth distribution:\n")
            for line in distribution_lines(cfg):
                f.write(f"#  {line}\n")
            f.write("#\n")
            f.writelines(f"{name}: {bits}\n" for name, bits in cfg.items())
        print(f"budget {budget}: avg {avg:.4f} bits, pred KL {pred_kl:.6f}, -> {out}", flush=True)
    print("CONFIGS_DONE")


if __name__ == "__main__":
    main()
