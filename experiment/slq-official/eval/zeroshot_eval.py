#!/usr/bin/env python
"""Lightweight zero-shot evaluation (log-likelihood) used for the SLQ "recovery" metric.

Tasks: arc_easy / arc_challenge / piqa / hellaswag / winogrande / openbookqa.

The script is deliberately independent of `lm_eval`, which times out on the NPU
host: each item is scored by the mean log-probability of its answer tokens.

Usage:
    python eval/zeroshot_eval.py --model <path> --device npu:0 --limit 500 \
        --tasks arc_easy,arc_challenge,piqa,hellaswag,winogrande,openbookqa \
        --out results/zs27b_b36_500.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.loading import _resolve_model_class  # noqa: E402

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def log(*args):
    print(*args, flush=True)


def load_model(args):
    """Load the architecture recorded in config.json; returns (model, tokenizer, device)."""
    from transformers import AutoTokenizer

    spec = args.device
    if spec.startswith("npu"):
        import torch_npu  # noqa: F401
        device = torch.device(spec if ":" in spec else "npu:0")
    else:
        device = torch.device(spec)

    model_cls = _resolve_model_class(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = dict(trust_remote_code=True, dtype=getattr(torch, args.dtype))
    if args.device_map:
        model = model_cls.from_pretrained(args.model, device_map=args.device_map, **kwargs)
    else:
        model = model_cls.from_pretrained(args.model, **kwargs).to(device)
    model.config.use_cache = False
    if hasattr(model, "language_model"):
        model.language_model.config.use_cache = False
    log(f"loaded {args.model} as {model_cls.__name__} on {device}")
    return model.eval(), tokenizer, device


@torch.no_grad()
def score_choices(model, tok, ctx, choices, device, pad_id):
    ctx_ids = tok(ctx, add_special_tokens=False).input_ids
    ch_ids = [tok(c, add_special_tokens=False).input_ids for c in choices]
    seqs = [ctx_ids + c for c in ch_ids]
    maxlen = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, maxlen - len(s):] = torch.tensor(s)
        attn[k, maxlen - len(s):] = 1
    ids, attn = ids.to(device), attn.to(device)
    logits = model(input_ids=ids, attention_mask=attn).logits.float()

    scores = []
    for k, s in enumerate(seqs):
        n_ch = len(ch_ids[k])
        lp = 0.0
        for i in range(n_ch):
            p = maxlen - n_ch + i - 1          # predicts token p+1
            logp = torch.log_softmax(logits[k, p], dim=-1)
            lp += float(logp[ids[k, p + 1]])
        scores.append(lp / max(1, n_ch))
    return scores


def load_tasks(tasks):
    from datasets import load_dataset

    out = {}
    for t in tasks:
        if t == "arc_easy":
            out[t] = ("arc", load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"))
        elif t == "arc_challenge":
            out[t] = ("arc", load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
        elif t == "piqa":
            out[t] = ("piqa", load_dataset("baber/piqa", split="validation"))
        elif t == "hellaswag":
            out[t] = ("hella", load_dataset("Rowan/hellaswag", split="validation"))
        elif t == "winogrande":
            out[t] = ("wino", load_dataset("allenai/winogrande", "winogrande_xl", split="validation"))
        elif t == "openbookqa":
            out[t] = ("obqa", load_dataset("allenai/openbookqa", "main", split="test"))
        else:
            raise ValueError(f"unknown task {t}")
    return out


def build_item(kind, row):
    """Return (context, choices, gold_index)."""
    if kind == "arc":
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        return ("Question: " + row["question"] + "\nAnswer:", texts, labels.index(row["answerKey"]))
    if kind == "obqa":
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        return ("Question: " + row["question_stem"] + "\nAnswer:", texts, labels.index(row["answerKey"]))
    if kind == "piqa":
        return ("Question: " + row["goal"] + "\nAnswer:", [row["sol1"], row["sol2"]], int(row["label"]))
    if kind == "hella":
        return (row["ctx"], list(row["endings"]), int(row["label"]))
    if kind == "wino":
        s = row["sentence"]
        opts = [s.replace("_", row["option1"]), s.replace("_", row["option2"])]
        return ("", opts, int(row["answer"]) - 1)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default=None,
                    help="pass through to transformers (e.g. auto); default: single --device")
    ap.add_argument("--tasks", default="arc_easy,piqa,hellaswag,winogrande,openbookqa")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok, device = load_model(args)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data = load_tasks(tasks)

    results = {}
    for t in tasks:
        kind, ds = data[t]
        n = min(args.limit, len(ds)) if args.limit > 0 else len(ds)
        correct = 0
        t0 = time.time()
        for i in range(n):
            ctx, choices, gold = build_item(kind, ds[i])
            sc = score_choices(model, tok, ctx, choices, device, pad)
            pred = int(torch.tensor(sc).argmax())
            correct += int(pred == gold)
            if (i + 1) % 50 == 0:
                log(f"    {t}: {i + 1}/{n} acc={correct / (i + 1):.4f}")
        acc = correct / max(1, n)
        results[t] = {"acc": acc, "n": n, "sec": round(time.time() - t0, 1)}
        log(f"  [{t}] acc={acc:.4f} (n={n}, {time.time() - t0:.1f}s)")

    mean = sum(v["acc"] for v in results.values()) / max(1, len(results))
    log(f"\nMEAN acc = {mean:.4f}  model={args.model}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "results": results, "mean": mean}, f, indent=1)


if __name__ == "__main__":
    main()
