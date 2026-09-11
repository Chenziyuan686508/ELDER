#!/usr/bin/env python3
"""Read-only DoRA component ablation for ELDER Stage 1 checkpoints.

It reuses the fixed MMEB-V2 query/candidate preparation from
``diagnose_stage1_checkpoint.py``.  Each profile enables only a selected
subset of the already-trained DoRA modules in memory; checkpoint files are
never changed.  It can also trace the pooled representation after every
decoder layer for base vs. full DoRA.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import diagnose_stage1_checkpoint as checkpoint_diag


ATTENTION_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
MLP_MODULES = frozenset({"gate_proj", "up_proj", "down_proj"})
SUPPORTED_PROFILES = (
    "base",
    "all",
    "attention",
    "mlp",
    "lower",
    "middle",
    "upper",
)
LAYER_RANGES = {
    "lower": range(0, 10),
    "middle": range(10, 19),
    "upper": range(19, 28),
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def describe_adapter_module(name: str) -> dict[str, Any] | None:
    """Extract the decoder-layer index and projection family from a PEFT name."""
    layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    module_match = re.search(
        r"(?:^|\.)(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)(?:\.|$)",
        name,
    )
    if layer_match is None or module_match is None:
        return None
    module_type = module_match.group(1)
    family = "attention" if module_type in ATTENTION_MODULES else "mlp"
    return {
        "name": name,
        "layer": int(layer_match.group(1)),
        "module_type": module_type,
        "family": family,
    }


def profile_matches(profile: str, descriptor: dict[str, Any]) -> bool:
    if profile == "all":
        return True
    if profile in {"attention", "mlp"}:
        return descriptor["family"] == profile
    if profile in LAYER_RANGES:
        return descriptor["layer"] in LAYER_RANGES[profile]
    if profile == "base":
        return False
    raise ValueError(f"Unsupported profile: {profile}")


def expected_global_labels(local_batch_size: int, world_size: int) -> list[int]:
    """Return the diagonal InfoNCE labels used after equal-size DDP gather."""
    if local_batch_size < 1 or world_size < 1:
        raise ValueError("local_batch_size and world_size must be positive")
    return list(range(local_batch_size * world_size))


def positive_margin_statistics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: list[str],
    ground_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = checkpoint_diag.normalize_rows(query_embeddings) @ checkpoint_diag.normalize_rows(
        candidate_embeddings
    ).T
    positions: dict[str, list[int]] = {}
    for index, candidate_id in enumerate(candidate_ids):
        positions.setdefault(str(candidate_id), []).append(index)

    positive_scores, hardest_negatives, margins = [], [], []
    for row, info in enumerate(ground_truth):
        indices = positions.get(str(info["label_name"]), [])
        if not indices:
            raise KeyError(f"Missing positive candidate: {info['label_name']!r}")
        positive = float(scores[row, indices].max())
        negative_mask = np.ones(scores.shape[1], dtype=bool)
        negative_mask[indices] = False
        negative = float(scores[row, negative_mask].max())
        positive_scores.append(positive)
        hardest_negatives.append(negative)
        margins.append(positive - negative)
    return {
        "positive_similarity_mean": float(np.mean(positive_scores)),
        "hardest_negative_similarity_mean": float(np.mean(hardest_negatives)),
        "positive_minus_hardest_negative_mean": float(np.mean(margins)),
        "positive_beats_hardest_negative_rate": float(np.mean(np.asarray(margins) > 0)),
    }


def top1_distribution(top1: list[str]) -> dict[str, Any]:
    counts = Counter(top1)
    total = len(top1)
    if total == 0:
        return {"unique_fraction": 0.0, "largest_fraction": 0.0, "top_candidates": []}
    return {
        "unique_fraction": len(counts) / total,
        "largest_fraction": max(counts.values()) / total,
        "top_candidates": [
            {"candidate": key, "count": value, "fraction": value / total}
            for key, value in counts.most_common(10)
        ],
    }


def attach_profiled_dora(model: Any, checkpoint: Path, profile: str) -> dict[str, Any]:
    """Attach the saved adapter and disable all modules outside ``profile``."""
    from peft import LoraConfig, PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer

    if not hasattr(model.encoder, "model"):
        raise RuntimeError("Qwen2-VL encoder has no nested language model")
    config = LoraConfig.from_pretrained(str(checkpoint))
    adapter = PeftModel.from_pretrained(
        model.encoder.model,
        str(checkpoint),
        config=config,
        is_trainable=False,
    )
    selected: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    unknown: list[str] = []
    for name, module in adapter.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        descriptor = describe_adapter_module(name)
        if descriptor is None:
            unknown.append(name)
            module.enable_adapters(False)
            continue
        if profile_matches(profile, descriptor):
            module.enable_adapters(True)
            selected.append(descriptor)
        else:
            module.enable_adapters(False)
            disabled.append(descriptor)
    if not selected:
        raise RuntimeError(f"Profile {profile!r} selected no DoRA modules")
    model.encoder.model = adapter
    return {
        "profile": profile,
        "adapter_scope": "encoder.model",
        "merged": False,
        "use_dora": bool(config.use_dora),
        "lora_r": int(config.r),
        "lora_alpha": int(config.lora_alpha),
        "selected_module_count": len(selected),
        "disabled_module_count": len(disabled),
        "unknown_module_count": len(unknown),
        "selected_by_family": dict(Counter(item["family"] for item in selected)),
        "selected_by_layer": {
            str(layer): count
            for layer, count in sorted(Counter(item["layer"] for item in selected).items())
        },
    }


def load_profile_model(base_model: Path, checkpoint: Path, profile: str):
    if profile == "base":
        model = checkpoint_diag.build_base_model(base_model)
        return model, {"profile": "base", "adapter_enabled": False}
    model = checkpoint_diag.build_base_model(base_model)
    details = attach_profiled_dora(model, checkpoint, profile)
    model.model_backbone = "qwen2_vl"
    return model, details


def release_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_profile(
    profile: str,
    base_model: Path,
    checkpoint: Path,
    prepared: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    import torch

    started = time.time()
    model, load_details = load_profile_model(base_model, checkpoint, profile)
    model = model.to(device, dtype=torch.bfloat16)
    query_embeddings = checkpoint_diag.encode_cached_batches(
        model, prepared["query_batches"], "qry", device
    )
    candidate_embeddings = checkpoint_diag.encode_cached_batches(
        model, prepared["candidate_batches"], "tgt", device
    )
    metrics, top1 = checkpoint_diag.rank_candidates(
        query_embeddings,
        candidate_embeddings,
        prepared["candidate_ids"],
        prepared["ground_truth"],
    )
    result = {
        "status": "success",
        "profile": profile,
        "duration_seconds": time.time() - started,
        "load_details": load_details,
        "retrieval_metrics": metrics,
        "positive_margin": positive_margin_statistics(
            query_embeddings,
            candidate_embeddings,
            prepared["candidate_ids"],
            prepared["ground_truth"],
        ),
        "top1_distribution": top1_distribution(top1),
        "query_embedding_statistics": checkpoint_diag.embedding_statistics(
            query_embeddings
        ),
        "candidate_embedding_statistics": checkpoint_diag.embedding_statistics(
            candidate_embeddings
        ),
    }
    release_model(model)
    return result


def find_decoder_layers(model: Any) -> dict[int, Any]:
    layers: dict[int, Any] = {}
    for name, module in model.encoder.model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
        if match is not None:
            layers[int(match.group(1))] = module
    if not layers:
        raise RuntimeError("Could not find decoder layers below model.encoder.model")
    return dict(sorted(layers.items()))


def collect_layer_embeddings(
    model: Any, batches: Iterable[dict[str, Any]], device: Any
) -> dict[int, np.ndarray]:
    """Pool every decoder-layer output at the same last-token position as eval."""
    import torch

    from src.utils.basic_utils import batch_to_device

    layer_outputs: dict[int, list[np.ndarray]] = {}
    hooks = []
    attention_mask: dict[str, Any] = {"value": None}

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pooled = model._pooling(hidden, attention_mask["value"])
            layer_outputs.setdefault(layer_index, []).append(
                pooled.detach().float().cpu().numpy()
            )

        return hook

    for layer_index, layer in find_decoder_layers(model).items():
        hooks.append(layer.register_forward_hook(make_hook(layer_index)))
    model.eval()
    try:
        with torch.inference_mode():
            for inputs in batches:
                device_inputs = batch_to_device(inputs, device)
                attention_mask["value"] = device_inputs["attention_mask"]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(qry=device_inputs)
                del device_inputs
    finally:
        for hook in hooks:
            hook.remove()
    return {
        layer: np.concatenate(values, axis=0)
        for layer, values in sorted(layer_outputs.items())
    }


def trace_hidden_states(
    base_model: Path,
    checkpoint: Path,
    prepared: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    import torch

    embeddings: dict[str, dict[int, np.ndarray]] = {}
    for profile in ("base", "all"):
        model, _details = load_profile_model(base_model, checkpoint, profile)
        model = model.to(device, dtype=torch.bfloat16)
        embeddings[profile] = collect_layer_embeddings(
            model, prepared["query_batches"], device
        )
        release_model(model)
    shared_layers = sorted(set(embeddings["base"]) & set(embeddings["all"]))
    trace = []
    for layer in shared_layers:
        base_values = embeddings["base"][layer]
        dora_values = embeddings["all"][layer]
        trace.append(
            {
                "layer": layer,
                "base": checkpoint_diag.embedding_statistics(base_values),
                "dora_all": checkpoint_diag.embedding_statistics(dora_values),
                "base_vs_dora": checkpoint_diag.compare_embedding_sets(
                    base_values, dora_values
                ),
            }
        )
    return {"layer_count": len(trace), "layers": trace}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("experiments/elder/mmeb_v2/image.yaml"),
    )
    parser.add_argument("--dataset", default="ImageNet-R")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--resize-max-pixels", type=int, default=200704)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profiles", default="base,all,attention,mlp,lower,middle,upper")
    parser.add_argument("--trace-hidden-states", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = parse_csv(args.profiles)
    invalid = sorted(set(profiles).difference(SUPPORTED_PROFILES))
    if invalid:
        raise ValueError(f"Unsupported profiles: {invalid}")
    if not profiles:
        raise ValueError("At least one profile is required")

    import torch
    from src.arguments import DataArguments

    for path, label in (
        (args.base_model, "base model"),
        (args.checkpoint, "checkpoint"),
        (args.data_root, "data root"),
        (args.dataset_config, "dataset config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("DoRA ablation requires a visible CUDA GPU")

    args.base_model = args.base_model.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.data_root = args.data_root.resolve()
    args.dataset_config = args.dataset_config.resolve()
    args.output = args.output.resolve()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    data_args = DataArguments(
        data_basedir=str(args.data_root),
        max_len=args.max_length,
        resize_use_processor=True,
        resize_max_pixels=args.resize_max_pixels,
    )
    processor_model_args = checkpoint_diag.make_model_args(
        args.base_model, checkpoint=None, lora=False
    )
    print(f"Preparing fixed {args.dataset} inputs...", flush=True)
    prepared = checkpoint_diag.prepare_fixed_batches(
        args, processor_model_args, data_args
    )
    report: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.base_model),
        "checkpoint": checkpoint_diag.checkpoint_summary(args.checkpoint),
        "dataset": {
            "name": args.dataset,
            "query_count": prepared["query_count"],
            "candidate_count": prepared["candidate_count"],
            "resize_max_pixels": args.resize_max_pixels,
        },
        "profiles": {},
        "hidden_state_trace": None,
    }
    partial = args.output.with_name(args.output.stem + ".partial" + args.output.suffix)
    checkpoint_diag.atomic_write_json(partial, report)
    for profile in profiles:
        print(f"[{profile}] loading and encoding...", flush=True)
        report["profiles"][profile] = run_profile(
            profile, args.base_model, args.checkpoint, prepared, device
        )
        stats = report["profiles"][profile]["query_embedding_statistics"]
        print(
            f"[{profile}] query_offdiag_cos="
            f"{stats['off_diagonal_cosine']['mean']:.6f}",
            flush=True,
        )
        checkpoint_diag.atomic_write_json(partial, report)
    if args.trace_hidden_states:
        print("[trace] collecting base vs. full-DoRA decoder-layer embeddings...", flush=True)
        report["hidden_state_trace"] = trace_hidden_states(
            args.base_model, args.checkpoint, prepared, device
        )
        checkpoint_diag.atomic_write_json(partial, report)
    report["status"] = "success"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint_diag.atomic_write_json(args.output, report)
    partial.unlink(missing_ok=True)
    print(f"Wrote report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
