"""Correctness helpers for the ELDER Stage 1 retrieval baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


QWEN2_VL_2B_CONTRACT = {
    "model_type": "qwen2_vl",
    "architecture": "Qwen2VLForConditionalGeneration",
    "hidden_size": 1536,
    "num_hidden_layers": 28,
}


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def validate_qwen2_vl_2b_config(config: Any) -> dict[str, Any]:
    """Fail fast unless ``config`` identifies the fixed ELDER 2B backbone."""

    architectures = list(_config_value(config, "architectures") or [])
    actual = {
        "model_type": _config_value(config, "model_type"),
        "architectures": architectures,
        "hidden_size": _config_value(config, "hidden_size"),
        "num_hidden_layers": _config_value(config, "num_hidden_layers"),
    }
    mismatches = []
    for key in ("model_type", "hidden_size", "num_hidden_layers"):
        expected = QWEN2_VL_2B_CONTRACT[key]
        if actual[key] != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, got {actual[key]!r}"
            )
    expected_architecture = QWEN2_VL_2B_CONTRACT["architecture"]
    if expected_architecture not in architectures:
        mismatches.append(
            "architectures: expected to contain "
            f"{expected_architecture!r}, got {architectures!r}"
        )
    if mismatches:
        raise ValueError(
            "ELDER is fixed to Qwen/Qwen2-VL-2B-Instruct; "
            + "; ".join(mismatches)
        )
    return {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        **QWEN2_VL_2B_CONTRACT,
    }


def _directional_metrics(
    similarities: torch.Tensor,
    positive_indices: torch.Tensor,
    ks: Sequence[int],
    temperature: float,
) -> dict[str, Any]:
    if similarities.ndim != 2:
        raise ValueError("similarities must have shape [num_query, num_candidate]")
    if positive_indices.shape != (similarities.shape[0],):
        raise ValueError("positive_indices must contain one index per query")

    ranked_indices = similarities.argsort(dim=1, descending=True)
    matches = ranked_indices.eq(positive_indices[:, None])
    if not torch.all(matches.any(dim=1)):
        raise RuntimeError("At least one positive index is absent from the ranking.")
    ranks = matches.to(torch.int64).argmax(dim=1) + 1

    positive_scores = similarities.gather(1, positive_indices[:, None]).squeeze(1)
    negative_mask = torch.ones_like(similarities, dtype=torch.bool)
    negative_mask.scatter_(1, positive_indices[:, None], False)
    if similarities.shape[1] > 1:
        hardest_negative = similarities.masked_fill(
            ~negative_mask, -torch.inf
        ).max(dim=1).values
        hardest_negative_mean = hardest_negative.mean().item()
        positive_margin_mean = (positive_scores - hardest_negative).mean().item()
    else:
        hardest_negative_mean = None
        positive_margin_mean = None

    recalls = {
        str(int(k)): ranks.le(min(int(k), similarities.shape[1]))
        .float()
        .mean()
        .item()
        for k in ks
    }
    return {
        "recall_at": recalls,
        "mrr": ranks.float().reciprocal().mean().item(),
        "mean_rank": ranks.float().mean().item(),
        "median_rank": ranks.float().median().item(),
        "infonce_loss": F.cross_entropy(
            similarities / temperature, positive_indices
        ).item(),
        "positive_similarity_mean": positive_scores.mean().item(),
        "hardest_negative_similarity_mean": hardest_negative_mean,
        "positive_margin_mean": positive_margin_mean,
    }


def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    ks: Sequence[int] = (1, 5, 10),
    temperature: float = 0.02,
) -> dict[str, Any]:
    """Compute paired query/candidate retrieval diagnostics in both directions."""

    query_embeddings = query_embeddings.detach().float().cpu()
    candidate_embeddings = candidate_embeddings.detach().float().cpu()
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2 tensors.")
    if query_embeddings.shape != candidate_embeddings.shape:
        raise ValueError(
            "Paired Stage 1 evaluation requires query/candidate tensors with "
            f"the same shape, got {tuple(query_embeddings.shape)} and "
            f"{tuple(candidate_embeddings.shape)}."
        )
    if query_embeddings.shape[0] == 0:
        raise ValueError("At least one retrieval pair is required.")
    if not torch.isfinite(query_embeddings).all() or not torch.isfinite(
        candidate_embeddings
    ).all():
        raise ValueError("Embeddings contain NaN or Inf.")

    similarities = query_embeddings @ candidate_embeddings.T
    targets = torch.arange(query_embeddings.shape[0], dtype=torch.long)
    query_norms = query_embeddings.norm(dim=-1)
    candidate_norms = candidate_embeddings.norm(dim=-1)
    q2c = _directional_metrics(similarities, targets, ks, temperature)
    c2q = _directional_metrics(similarities.T, targets, ks, temperature)

    return {
        "num_pairs": query_embeddings.shape[0],
        "embedding_dim": query_embeddings.shape[1],
        "query_norm": {
            "min": query_norms.min().item(),
            "mean": query_norms.mean().item(),
            "max": query_norms.max().item(),
        },
        "candidate_norm": {
            "min": candidate_norms.min().item(),
            "mean": candidate_norms.mean().item(),
            "max": candidate_norms.max().item(),
        },
        "query_to_candidate": q2c,
        "candidate_to_query": c2q,
        "mean_recall_at": {
            str(int(k)): (
                q2c["recall_at"][str(int(k))]
                + c2q["recall_at"][str(int(k))]
            )
            / 2.0
            for k in ks
        },
    }


def compare_stage1_reports(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    random_recall_multiplier: float = 3.0,
    norm_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Apply the small-set Stage 1 acceptance gate to two eval reports."""

    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    num_pairs = int(after_metrics["num_pairs"])
    if num_pairs != int(before_metrics["num_pairs"]):
        raise ValueError("Before/after reports used different numbers of pairs.")

    before_q2c = before_metrics["query_to_candidate"]
    after_q2c = after_metrics["query_to_candidate"]
    before_recall = float(before_q2c["recall_at"]["1"])
    after_recall = float(after_q2c["recall_at"]["1"])
    before_loss = float(before_q2c["infonce_loss"])
    after_loss = float(after_q2c["infonce_loss"])
    random_recall = 1.0 / num_pairs
    minimum_recall = min(1.0, random_recall * random_recall_multiplier)

    norm_values = [
        float(after_metrics[side]["mean"])
        for side in ("query_norm", "candidate_norm")
    ]
    checks = {
        "q2c_infonce_decreased": after_loss < before_loss,
        "q2c_recall_at_1_not_decreased": after_recall >= before_recall,
        "q2c_recall_at_1_above_random": after_recall >= minimum_recall,
        "embedding_norms_near_one": all(
            math.isfinite(value) and abs(value - 1.0) <= norm_tolerance
            for value in norm_values
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "thresholds": {
            "random_recall_at_1": random_recall,
            "minimum_recall_at_1": minimum_recall,
            "random_recall_multiplier": random_recall_multiplier,
            "norm_tolerance": norm_tolerance,
        },
        "changes": {
            "q2c_infonce_loss": after_loss - before_loss,
            "q2c_recall_at_1": after_recall - before_recall,
        },
        "before": {
            "q2c_infonce_loss": before_loss,
            "q2c_recall_at_1": before_recall,
        },
        "after": {
            "q2c_infonce_loss": after_loss,
            "q2c_recall_at_1": after_recall,
        },
    }
