"""Perplexity, top-K KL and EAR metrics."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


def _empty_cache():
    """Release cached NPU/CUDA blocks between batches with large vocab logits."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token cross-entropy, shape of labels."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='none').view(labels.shape)


@torch.no_grad()
def compute_perplexity(model, data: List[torch.Tensor], batch_size: int = 1) -> Tuple[float, float, List[float]]:
    """Mean perplexity (exp of mean per-sample NLL), variance of per-sample PPL, and per-sample PPL."""
    device = next(model.parameters()).device
    nll = []
    for i in trange(0, len(data), batch_size, desc="Computing perplexity", leave=False):
        inputs = torch.cat(data[i:i + batch_size]).to(device)
        nll.extend(_token_nll(model(inputs).logits[:, :-1], inputs[:, 1:]).mean(dim=1).tolist())
    ppl = np.exp(nll)
    return float(np.exp(np.mean(nll))), float(np.var(ppl)), ppl.tolist()


@torch.no_grad()
def compute_baseline_topk(
    model, input_ids: torch.Tensor, batch_size: int = 4, top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K baseline logits and indices at every position, shape [N, seq_len-1, top_k]."""
    model.eval()
    device = next(model.parameters()).device
    vals, idx = [], []
    for i in range(0, input_ids.shape[0], batch_size):
        logits = model(input_ids[i:i + batch_size].to(device)).logits[:, :-1].float()
        v, k = logits.topk(top_k, dim=-1)
        vals.append(v.cpu())
        idx.append(k.cpu())
        del logits
        _empty_cache()
    return torch.cat(vals), torch.cat(idx)


@torch.no_grad()
def _nll_kl_ear(model, data, topk_vals, topk_idx, batch_size, masks, with_nll):
    """Token-mean NLL, top-K KL(p||q) and EAR = sum_k min(p, q), with p, q renormalized over the top-K under p."""
    model.eval()
    device = next(model.parameters()).device
    nll = kl = ear = n = 0.0
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size].to(device)
        logits = model(batch).logits[:, :-1]
        w = masks[i:i + batch_size, 1:].to(device).float() if masks is not None else torch.ones_like(batch[:, 1:], dtype=torch.float)
        if with_nll:
            nll += (_token_nll(logits, batch[:, 1:]) * w).sum().item()
        log_p = F.log_softmax(topk_vals[i:i + batch_size].to(device), dim=-1)
        log_q = F.log_softmax(logits.gather(-1, topk_idx[i:i + batch_size].to(device)).float(), dim=-1)
        p = log_p.exp()
        kl += ((p * (log_p - log_q)).sum(-1) * w).sum().item()
        ear += (torch.minimum(p, log_q.exp()).sum(-1) * w).sum().item()
        n += w.sum().item()
        del logits
        _empty_cache()
    d = max(n, 1.0)
    return nll / d, kl / d, ear / d, int(n)


def compute_all_metrics(
    model, calibration_data: torch.Tensor, baseline_topk_vals: torch.Tensor, baseline_topk_idx: torch.Tensor,
    batch_size: int = 4, masks: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Mean NLL, top-K KL and EAR against the baseline top-K cache."""
    return _nll_kl_ear(model, calibration_data, baseline_topk_vals, baseline_topk_idx, batch_size, masks, True)[:3]


def compute_kl_and_ear(
    model, baseline_topk_vals: torch.Tensor, baseline_topk_idx: torch.Tensor, eval_data: torch.Tensor,
    batch_size: int = 4, masks: Optional[torch.Tensor] = None,
) -> Tuple[float, float, int]:
    """(mean top-K KL, mean EAR, number of scored tokens)."""
    return _nll_kl_ear(model, eval_data, baseline_topk_vals, baseline_topk_idx, batch_size, masks, False)[1:]
