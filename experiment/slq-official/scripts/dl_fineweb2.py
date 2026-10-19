"""从 hf-mirror 拉取 FineWeb-Edu sample-10BT 的一个 parquet 分片并写成 .pt 校准集。"""
import os

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
EP = "https://hf-mirror.com"

api = HfApi(endpoint=EP)
files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
print("num parquet:", len(files))
print("first:", files[:3])

target = [f for f in files if "10BT" in f]
use = (target or files)[0]
print("downloading:", use)
path = hf_hub_download(REPO, use, repo_type="dataset", endpoint=EP)
print("local:", path, os.path.getsize(path) / 1e6, "MB")
