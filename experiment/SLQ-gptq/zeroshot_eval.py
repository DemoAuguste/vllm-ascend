# -*- coding: UTF-8 -*-
"""轻量 zero-shot 评测（log-likelihood，NPU 原生），用于 SLQ 的"恢复率"口径。

任务：arc_easy / arc_challenge / piqa / hellaswag / winogrande / openbookqa
用法：
  python zeroshot_eval.py --model Qwen/Qwen3-4B --device npu:0 \
      --device-map split --device-map-split 2 --limit 200 --out bf16.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from model_quant import load_model, log

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


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
            p = maxlen - n_ch + i - 1          # 预测第 p+1 个 token
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
    return out


def build_item(kind, row):
    """返回 (context, choices, gold_index)。"""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--device-map-split", type=int, default=2)
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
                log(f"    {t}: {i+1}/{n} acc={correct/(i+1):.4f}")
        acc = correct / max(1, n)
        results[t] = {"acc": acc, "n": n, "sec": round(time.time() - t0, 1)}
        log(f"  [{t}] acc={acc:.4f} (n={n}, {time.time()-t0:.1f}s)")

    mean = sum(v["acc"] for v in results.values()) / max(1, len(results))
    log(f"\nMEAN acc = {mean:.4f}  model={args.model}")
    if args.out:
        json.dump({"model": args.model, "results": results, "mean": mean},
                  open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
