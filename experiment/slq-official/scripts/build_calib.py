# -*- coding: UTF-8 -*-
"""从 FineWeb-Edu parquet 构建官方 SLQ 所需的校准集（.pt：list[(1, seq_len)]）。

用法：
  python build_calib.py --parquet data/fineweb_10bt_000.parquet \
      --tokenizer Qwen/Qwen3-0.6B --seq 8192 --num 1024 --out data/calib_qwen3_8m_8192.pt
"""
import argparse
import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--num", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-col", default="text")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    need = args.seq * args.num
    print(f"need {need} tokens ({args.num} x {args.seq})")

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(args.parquet)
    print("rows:", pf.metadata.num_rows, "row_groups:", pf.num_row_groups)

    buf = torch.empty(0, dtype=torch.long)
    samples = []
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[args.text_col])
        for text in tbl.column(0).to_pylist():
            if not text or not text.strip():
                continue
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if ids.numel() == 0:
                continue
            buf = torch.cat([buf, ids])
            while buf.numel() >= args.seq and len(samples) < args.num:
                samples.append(buf[:args.seq].unsqueeze(0))
                buf = buf[args.seq:]
            if len(samples) >= args.num:
                break
        if len(samples) >= args.num:
            break
        print(f"  row_group {rg+1}/{pf.num_row_groups}: samples={len(samples)}")

    assert len(samples) >= args.num, f"only {len(samples)} samples"
    torch.save(samples, args.out)
    print("saved", args.out, "samples:", len(samples),
          "size MB:", round(os.path.getsize(args.out) / 1e6, 1))


if __name__ == "__main__":
    main()
