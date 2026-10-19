"""HuggingFace model and tokenizer loaders."""

import json
import os

import torch


def _resolve_model_class(model_path: str):
    """Pick the architecture recorded in config.json (VLM vs CausalLM) to avoid
    silently loading random weights when the class does not match the checkpoint."""
    import transformers

    archs = []
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            archs = json.load(f).get("architectures") or []
    for name in archs:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    return transformers.AutoModelForCausalLM


def load_model(model_path: str, dtype: str = "bfloat16", device_map="balanced"):
    """Causal LM in eval mode with the KV cache off, dispatched across visible GPUs by accelerate."""
    cls = _resolve_model_class(model_path)

    model = cls.from_pretrained(
        model_path, dtype=getattr(torch, dtype), device_map=device_map,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "language_model"):
        model.language_model.config.use_cache = False
    return model.eval()


def _fix_tokenizer_config(model_path: str):
    """Rewrite a list-valued extra_special_tokens as a dict in the (cached) tokenizer_config.json."""
    from huggingface_hub import hf_hub_download

    path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.isdir(model_path):
        path = hf_hub_download(model_path, "tokenizer_config.json")
    with open(path) as f:
        config = json.load(f)
    if isinstance(config.get("extra_special_tokens"), list):
        config["extra_special_tokens"] = {tok: "" for tok in config["extra_special_tokens"]}
        with open(path, "w") as f:
            json.dump(config, f, indent=2)


def get_tokenizer(model_path: str):
    """Load a tokenizer; pad_token falls back to eos_token."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except (AttributeError, TypeError) as e:
        if "extra_special_tokens" not in str(e) and "'list' object" not in str(e):
            raise
        _fix_tokenizer_config(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer
