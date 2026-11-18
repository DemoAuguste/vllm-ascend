"""Average two Shapley databases produced with different permutation seeds.

Example (two 4-permutation runs -> one 8-permutation database):

    python scripts/merge_shapley.py seed42.json seed43.json merged.json

The averaging is what turns two single-card nights into the P=8 sampling used by
the 27B sensitivity library.
"""
import argparse
import json

import numpy as np

ARRAY_KEYS = ("nll_shapley_values", "kl_shapley_values", "ear_shapley_values")
LEVEL_KEYS = ("nll_shapley_per_level", "kl_shapley_per_level", "ear_shapley_per_level")
COPIED_KEYS = (
    "top_k", "calibration_dataset", "calibration_samples", "seed",
    "per_layer_normalized", "layer_names", "layer_groups",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("output")
    args = ap.parse_args()

    with open(args.first) as f:
        a = json.load(f)
    with open(args.second) as f:
        b = json.load(f)

    assert a["bitwidths"] == b["bitwidths"], "the two databases use different bitwidths"
    assert a["layer_names"] == b["layer_names"], "the two databases use different layers"

    res = {}
    for key in ARRAY_KEYS:
        x, y = np.array(a[key], dtype=float), np.array(b[key], dtype=float)
        assert x.shape == y.shape, f"{key}: shape mismatch {x.shape} vs {y.shape}"
        res[key] = ((x + y) / 2.0).tolist()

    for key in LEVEL_KEYS:
        res[key] = {
            layer: {bit: (a[key][layer][bit] + b[key][layer][bit]) / 2.0
                    for bit in a[key][layer]}
            for layer in a[key]
        }

    res["bitwidths"] = a["bitwidths"]
    res["num_permutations"] = a["num_permutations"] + b["num_permutations"]
    for key in COPIED_KEYS:
        if key in a:
            res[key] = a[key]

    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)

    kl = np.array(res["kl_shapley_values"], dtype=float)
    print("merged ->", args.output)
    print("bitwidths", res["bitwidths"], "perms", res["num_permutations"], "groups", kl.shape[1])
    for i, bw in enumerate(res["bitwidths"]):
        print(f"  bw {bw}: total KL {kl[1, :, i].sum():.6f}, negative groups {int((kl[1, :, i] < 0).sum())}")


if __name__ == "__main__":
    main()
