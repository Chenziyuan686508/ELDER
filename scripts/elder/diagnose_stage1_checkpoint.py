#!/usr/bin/env python3
"""Diagnose Stage 1 hybrid checkpoints on a fixed MMEB-V2 image task.

The diagnostic is intentionally read-only.  It compares merged and unmerged
DoRA inference, isolates the trained visual tower and language-side adapter,
and can trace representation health across multiple checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUPPORTED_MODES = (
    "base",
    "full_unmerged",
    "full_merged",
    "visual_only",
    "dora_only",
)
DEFAULT_MODES = (
    "full_unmerged",
    "full_merged",
    "base",
    "visual_only",
    "dora_only",
)


@dataclass(frozen=True)
class DiagnosticThresholds:
    collapse_offdiag_cosine: float = 0.90
    merge_row_cosine: float = 0.995
    merge_top1_agreement: float = 0.99


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a rank-2 embedding matrix, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def embedding_statistics(embeddings: np.ndarray) -> dict[str, Any]:
    array = np.asarray(embeddings, dtype=np.float32)
    normalized = normalize_rows(array)
    norms = np.linalg.norm(array, axis=1)
    centered = normalized - normalized.mean(axis=0, keepdims=True)

    if len(normalized) > 1:
        similarities = normalized @ normalized.T
        off_diagonal = similarities[~np.eye(len(normalized), dtype=bool)]
        offdiag_stats = {
            "mean": _to_float(off_diagonal.mean()),
            "std": _to_float(off_diagonal.std()),
            "min": _to_float(off_diagonal.min()),
            "max": _to_float(off_diagonal.max()),
        }
    else:
        offdiag_stats = {"mean": None, "std": None, "min": None, "max": None}

    return {
        "shape": list(array.shape),
        "norm": {
            "mean": _to_float(norms.mean()),
            "std": _to_float(norms.std()),
            "min": _to_float(norms.min()),
            "max": _to_float(norms.max()),
        },
        "off_diagonal_cosine": offdiag_stats,
        "centered_rms": _to_float(np.sqrt(np.mean(centered * centered))),
        "mean_dimension_std": _to_float(normalized.std(axis=0).mean()),
    }


def rank_candidates(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: Sequence[str],
    ground_truth: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], list[str]]:
    queries = normalize_rows(query_embeddings)
    candidates = normalize_rows(candidate_embeddings)
    if len(candidate_ids) != len(candidates):
        raise ValueError("candidate_ids and candidate_embeddings have different lengths")
    if len(ground_truth) != len(queries):
        raise ValueError("ground_truth and query_embeddings have different lengths")

    candidate_to_index = {str(name): index for index, name in enumerate(candidate_ids)}
    scores = queries @ candidates.T
    ranks: list[int] = []
    top1_predictions: list[str] = []

    for row_index, info in enumerate(ground_truth):
        allowed_names = info.get("cand_names") or list(candidate_ids)
        allowed_indices = [
            candidate_to_index[str(name)]
            for name in allowed_names
            if str(name) in candidate_to_index
        ]
        if not allowed_indices:
            raise ValueError(f"Query {row_index} has no candidates present in the corpus")

        label_name = str(info.get("label_name", allowed_names[0]))
        if label_name not in candidate_to_index:
            raise ValueError(f"Ground-truth candidate is missing: {label_name}")

        ordered_local = np.argsort(-scores[row_index, allowed_indices], kind="stable")
        ordered_indices = [allowed_indices[int(index)] for index in ordered_local]
        top1_predictions.append(str(candidate_ids[ordered_indices[0]]))
        label_index = candidate_to_index[label_name]
        try:
            rank = ordered_indices.index(label_index) + 1
        except ValueError as exc:
            raise ValueError(
                f"Ground-truth candidate {label_name!r} is outside the local candidate set"
            ) from exc
        ranks.append(rank)

    rank_array = np.asarray(ranks, dtype=np.int64)
    metrics = {
        "hit@1": float(np.mean(rank_array <= 1)),
        "hit@5": float(np.mean(rank_array <= 5)),
        "hit@10": float(np.mean(rank_array <= 10)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(rank_array.mean()),
    }
    return metrics, top1_predictions


def compare_embedding_sets(
    left: np.ndarray,
    right: np.ndarray,
    left_top1: Sequence[str] | None = None,
    right_top1: Sequence[str] | None = None,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Cannot compare embedding matrices with shapes {left_array.shape} and {right_array.shape}"
        )

    left_normalized = normalize_rows(left_array)
    right_normalized = normalize_rows(right_array)
    row_cosines = np.sum(left_normalized * right_normalized, axis=1)
    report: dict[str, Any] = {
        "shape": list(left_array.shape),
        "row_cosine": {
            "mean": _to_float(row_cosines.mean()),
            "min": _to_float(row_cosines.min()),
            "max": _to_float(row_cosines.max()),
            "std": _to_float(row_cosines.std()),
        },
        "max_absolute_difference": _to_float(
            np.max(np.abs(left_array - right_array))
        ),
        "mean_l2_difference": _to_float(
            np.linalg.norm(left_array - right_array, axis=1).mean()
        ),
    }
    if left_top1 is not None and right_top1 is not None:
        if len(left_top1) != len(right_top1):
            raise ValueError("Top-1 prediction lists have different lengths")
        report["top1_agreement"] = float(
            np.mean(np.asarray(left_top1, dtype=object) == np.asarray(right_top1, dtype=object))
        )
    return report


def classify_mode_health(
    mode_result: dict[str, Any], thresholds: DiagnosticThresholds
) -> str:
    mean_cosine = (
        mode_result.get("query_embedding_statistics", {})
        .get("off_diagonal_cosine", {})
        .get("mean")
    )
    if mean_cosine is None:
        return "insufficient_samples"
    return "near_collapse" if mean_cosine >= thresholds.collapse_offdiag_cosine else "not_collapsed"


def merge_verdict(
    comparison: dict[str, Any], thresholds: DiagnosticThresholds
) -> str:
    row_cosine = comparison["row_cosine"]["mean"]
    top1_agreement = comparison.get("top1_agreement")
    if row_cosine is None:
        return "insufficient_samples"
    if row_cosine < thresholds.merge_row_cosine:
        return "merge_mismatch"
    if top1_agreement is not None and top1_agreement < thresholds.merge_top1_agreement:
        return "merge_mismatch"
    return "merge_equivalent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=REPO_ROOT / "experiments/elder/mmeb_v2/image.yaml",
    )
    parser.add_argument("--dataset", default="ImageNet-R")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--resize-max-pixels", type=int, default=200704)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help=f"Comma-separated subset of: {','.join(SUPPORTED_MODES)}",
    )
    parser.add_argument(
        "--trajectory-checkpoint",
        action="append",
        type=Path,
        default=[],
        help="Additional checkpoint to evaluate in full_merged mode; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collapse-threshold", type=float, default=0.90)
    parser.add_argument("--merge-cosine-threshold", type=float, default=0.995)
    parser.add_argument("--merge-top1-threshold", type=float, default=0.99)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[str]:
    modes = parse_csv(args.modes)
    invalid_modes = sorted(set(modes).difference(SUPPORTED_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {invalid_modes}")
    if not modes:
        raise ValueError("At least one diagnostic mode is required")
    if args.sample_count < 2:
        raise ValueError("--sample-count must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for path, description in (
        (args.base_model, "base model"),
        (args.checkpoint, "checkpoint"),
        (args.data_root, "MMEB-V2 data root"),
        (args.dataset_config, "dataset config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
    return modes


def checkpoint_summary(checkpoint: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(checkpoint)}
    for filename in (
        "adapter_config.json",
        "vlm2vec_hybrid_checkpoint.json",
        "trainer_state.json",
    ):
        path = checkpoint / filename
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if filename == "trainer_state.json":
                payload = {
                    "global_step": payload.get("global_step"),
                    "epoch": payload.get("epoch"),
                    "best_metric": payload.get("best_metric"),
                }
            summary[filename] = payload
    summary["files"] = {
        filename: (checkpoint / filename).stat().st_size
        for filename in (
            "model.safetensors",
            "adapter_model.safetensors",
            "adapter_config.json",
            "vlm2vec_hybrid_checkpoint.json",
        )
        if (checkpoint / filename).is_file()
    }
    return summary


def resolve_task_config(
    dataset_config: Path, dataset_name: str, data_root: Path
) -> dict[str, Any]:
    import yaml

    with dataset_config.open("r", encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)
    if dataset_name not in configs:
        raise KeyError(f"Dataset {dataset_name!r} is not in {dataset_config}")
    task = copy.deepcopy(configs[dataset_name])
    for key in (
        "image_root",
        "video_root",
        "frame_root",
        "clip_root",
        "data_path",
        "query_file",
        "candidate_file",
        "qrels_file",
    ):
        value = task.get(key)
        if value and not os.path.isabs(value):
            task[key] = str(data_root / value)
    return task


def prepare_fixed_batches(args: argparse.Namespace, model_args: Any, data_args: Any):
    from torch.utils.data import DataLoader

    from src.data.collator.eval_collator import MultimodalEvalDataCollator
    from src.data.eval_dataset.base_eval_dataset import (
        AutoEvalPairDataset,
        generate_cand_dataset,
    )
    from src.model.processor import load_processor

    task_config = resolve_task_config(args.dataset_config, args.dataset, args.data_root)
    task_config["num_sample_per_subset"] = args.sample_count
    query_dataset, corpus = AutoEvalPairDataset.instantiate(
        model_args=model_args,
        data_args=data_args,
        **task_config,
    )
    if len(query_dataset) > args.sample_count:
        query_dataset = query_dataset.select(range(args.sample_count))
    candidate_dataset = generate_cand_dataset(query_dataset, corpus)
    if len(query_dataset) < 2 or len(candidate_dataset) < 2:
        raise RuntimeError(
            f"Diagnostic dataset is too small: queries={len(query_dataset)}, candidates={len(candidate_dataset)}"
        )

    processor = load_processor(model_args, data_args)
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=MultimodalEvalDataCollator(
            processor, model_args, data_args, "qry"
        ),
    )
    candidate_loader = DataLoader(
        candidate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=MultimodalEvalDataCollator(
            processor, model_args, data_args, "cand"
        ),
    )

    query_batches = []
    ground_truth: list[dict[str, Any]] = []
    for inputs, infos in query_loader:
        query_batches.append(inputs)
        ground_truth.extend(infos)

    candidate_batches = []
    candidate_ids: list[str] = []
    for inputs, infos in candidate_loader:
        candidate_batches.append(inputs)
        candidate_ids.extend(str(info["cand_name"]) for info in infos)

    prompt_rows = []
    first_row = copy.deepcopy(query_dataset[0])
    original_text = str(first_row["query_text"][0]).rstrip()
    for suffix in (
        "",
        " Focus on the main object identity.",
        " Focus on visual appearance and style.",
        " Focus on fine-grained attributes.",
    ):
        row = copy.deepcopy(first_row)
        row["query_text"] = [original_text + suffix]
        prompt_rows.append(row)
    prompt_inputs, _ = MultimodalEvalDataCollator(
        processor, model_args, data_args, "qry"
    )(prompt_rows)

    return {
        "query_batches": query_batches,
        "candidate_batches": candidate_batches,
        "prompt_batch": prompt_inputs,
        "ground_truth": ground_truth,
        "candidate_ids": candidate_ids,
        "query_count": len(query_dataset),
        "candidate_count": len(candidate_dataset),
    }


def make_model_args(base_model: Path, checkpoint: Path | None, lora: bool):
    from src.arguments import ModelArguments

    model_args = ModelArguments(
        model_name=str(base_model),
        processor_name=str(base_model),
        checkpoint_path=str(checkpoint) if checkpoint else None,
        pooling="eos",
        normalize=True,
        temperature=0.02,
        lora=lora,
    )
    model_args.model_backbone = "qwen2_vl"
    return model_args


def build_base_model(base_model: Path):
    from src.model.model import MMEBModel

    model_args = make_model_args(base_model, checkpoint=None, lora=False)
    model = MMEBModel.build(model_args)
    model.model_backbone = "qwen2_vl"
    return model


def load_visual_state(model: Any, checkpoint: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    metadata_path = checkpoint / "vlm2vec_hybrid_checkpoint.json"
    full_state_name = "model.safetensors"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            full_state_name = json.load(handle).get("full_state_file", full_state_name)
    full_state_path = checkpoint / full_state_name
    if not full_state_path.is_file():
        raise FileNotFoundError(f"Missing full hybrid state: {full_state_path}")

    destination = model.encoder.state_dict()
    loaded = 0
    missing = []
    with safe_open(str(full_state_path), framework="pt", device="cpu") as source:
        visual_keys = [key for key in source.keys() if key.startswith("visual.")]
        for key in visual_keys:
            if key not in destination:
                missing.append(key)
                continue
            with torch.no_grad():
                destination[key].copy_(
                    source.get_tensor(key).to(destination[key].dtype)
                )
            loaded += 1
    if missing:
        raise RuntimeError(
            f"Visual-only load found unexpected checkpoint keys: {missing[:10]}"
        )
    if loaded == 0:
        raise RuntimeError("No visual.* tensors were found in the hybrid checkpoint")
    return {"loaded_visual_tensors": loaded, "full_state": str(full_state_path)}


def attach_dora_only(model: Any, checkpoint: Path) -> dict[str, Any]:
    from peft import LoraConfig, PeftModel

    if not hasattr(model.encoder, "model"):
        raise RuntimeError("Qwen2-VL encoder has no nested language model")
    config = LoraConfig.from_pretrained(str(checkpoint))
    adapter = PeftModel.from_pretrained(
        model.encoder.model,
        str(checkpoint),
        config=config,
        is_trainable=False,
    )
    model.encoder.model = adapter.merge_and_unload()
    return {
        "adapter_scope": "encoder.model",
        "merged": True,
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "use_dora": bool(config.use_dora),
    }


def load_diagnostic_model(
    mode: str, base_model: Path, checkpoint: Path
) -> tuple[Any, dict[str, Any]]:
    from src.model.model import MMEBModel

    details: dict[str, Any] = {}
    if mode == "base":
        model = build_base_model(base_model)
    elif mode == "full_unmerged":
        model_args = make_model_args(base_model, checkpoint=checkpoint, lora=True)
        model = MMEBModel.load(model_args, is_trainable=True)
        details["merged"] = False
    elif mode == "full_merged":
        model_args = make_model_args(base_model, checkpoint=checkpoint, lora=True)
        model = MMEBModel.load(model_args, is_trainable=False)
        details["merged"] = True
    elif mode == "visual_only":
        model = build_base_model(base_model)
        details.update(load_visual_state(model, checkpoint))
    elif mode == "dora_only":
        model = build_base_model(base_model)
        details.update(attach_dora_only(model, checkpoint))
    else:
        raise ValueError(f"Unsupported diagnostic mode: {mode}")
    model.model_backbone = "qwen2_vl"
    return model, details


def encode_cached_batches(
    model: Any,
    batches: Iterable[dict[str, Any]],
    side: str,
    device: Any,
) -> np.ndarray:
    import torch

    from src.utils.basic_utils import batch_to_device

    outputs = []
    output_key = "qry_reps" if side == "qry" else "tgt_reps"
    model.eval()
    with torch.inference_mode():
        for inputs in batches:
            device_inputs = batch_to_device(inputs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if side == "qry":
                    result = model(qry=device_inputs)
                else:
                    result = model(tgt=device_inputs)
            outputs.append(result[output_key].detach().float().cpu().numpy())
            del device_inputs, result
    return np.concatenate(outputs, axis=0)


def run_mode(
    mode: str,
    base_model: Path,
    checkpoint: Path,
    prepared: dict[str, Any],
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    started = time.time()
    model, load_details = load_diagnostic_model(mode, base_model, checkpoint)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    query_embeddings = encode_cached_batches(
        model, prepared["query_batches"], "qry", device
    )
    candidate_embeddings = encode_cached_batches(
        model, prepared["candidate_batches"], "tgt", device
    )
    prompt_embeddings = encode_cached_batches(
        model, [prepared["prompt_batch"]], "qry", device
    )
    metrics, top1 = rank_candidates(
        query_embeddings,
        candidate_embeddings,
        prepared["candidate_ids"],
        prepared["ground_truth"],
    )
    prompt_stats = embedding_statistics(prompt_embeddings)
    result = {
        "mode": mode,
        "checkpoint": str(checkpoint),
        "duration_seconds": time.time() - started,
        "load_details": load_details,
        "retrieval_metrics": metrics,
        "query_embedding_statistics": embedding_statistics(query_embeddings),
        "candidate_embedding_statistics": embedding_statistics(candidate_embeddings),
        "same_image_prompt_variation": prompt_stats,
    }
    arrays = {
        "query_embeddings": query_embeddings,
        "candidate_embeddings": candidate_embeddings,
        "top1": top1,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result, arrays


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def build_conclusion(report: dict[str, Any], thresholds: DiagnosticThresholds) -> dict[str, Any]:
    modes = report.get("modes", {})
    conclusion: dict[str, Any] = {
        "mode_health": {
            name: classify_mode_health(result, thresholds)
            for name, result in modes.items()
            if result.get("status") == "success"
        }
    }
    comparison = report.get("comparisons", {}).get("full_unmerged_vs_full_merged")
    if comparison:
        conclusion["merge_verdict"] = merge_verdict(comparison, thresholds)

    health = conclusion["mode_health"]
    if conclusion.get("merge_verdict") == "merge_mismatch":
        conclusion["likely_failure_domain"] = "hybrid_dora_merge_or_load"
    elif health.get("full_unmerged") == "near_collapse":
        if health.get("dora_only") == "near_collapse":
            conclusion["likely_failure_domain"] = "language_dora_or_visual_fusion"
        elif health.get("visual_only") == "near_collapse":
            conclusion["likely_failure_domain"] = "trained_visual_tower"
        else:
            conclusion["likely_failure_domain"] = "trained_hybrid_interaction"
    elif health.get("full_merged") == "near_collapse":
        conclusion["likely_failure_domain"] = "hybrid_dora_merge_or_load"
    else:
        conclusion["likely_failure_domain"] = "not_identified_by_component_test"
    return conclusion


def main() -> None:
    args = parse_args()
    modes = validate_args(args)

    import torch
    from src.arguments import DataArguments

    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint diagnostics require a visible CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    base_model = args.base_model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    args.data_root = data_root
    args.dataset_config = args.dataset_config.expanduser().resolve()

    thresholds = DiagnosticThresholds(
        collapse_offdiag_cosine=args.collapse_threshold,
        merge_row_cosine=args.merge_cosine_threshold,
        merge_top1_agreement=args.merge_top1_threshold,
    )
    report: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model),
        "primary_checkpoint": checkpoint_summary(checkpoint),
        "dataset": {
            "name": args.dataset,
            "root": str(data_root),
            "config": str(args.dataset_config),
            "requested_sample_count": args.sample_count,
            "resize_max_pixels": args.resize_max_pixels,
        },
        "thresholds": asdict(thresholds),
        "modes": {},
        "trajectory": {},
        "comparisons": {},
    }
    partial_path = output.with_name(output.stem + ".partial" + output.suffix)
    atomic_write_json(partial_path, report)

    data_args = DataArguments(
        data_basedir=str(data_root),
        max_len=args.max_length,
        resize_use_processor=True,
        resize_max_pixels=args.resize_max_pixels,
    )
    processor_model_args = make_model_args(base_model, checkpoint=None, lora=False)
    print(f"Preparing fixed {args.dataset} inputs...", flush=True)
    prepared = prepare_fixed_batches(args, processor_model_args, data_args)
    report["dataset"]["query_count"] = prepared["query_count"]
    report["dataset"]["candidate_count"] = prepared["candidate_count"]
    atomic_write_json(partial_path, report)

    arrays_by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        print(f"[{mode}] loading and encoding...", flush=True)
        try:
            result, arrays = run_mode(
                mode, base_model, checkpoint, prepared, device
            )
            result["status"] = "success"
            report["modes"][mode] = result
            arrays_by_mode[mode] = arrays
            print(
                f"[{mode}] hit@1={result['retrieval_metrics']['hit@1']:.4f}, "
                f"offdiag_cos={result['query_embedding_statistics']['off_diagonal_cosine']['mean']:.6f}",
                flush=True,
            )
        except Exception as exc:
            report["modes"][mode] = {
                "status": "failed",
                "mode": mode,
                "checkpoint": str(checkpoint),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            atomic_write_json(partial_path, report)
            raise
        atomic_write_json(partial_path, report)

    if "full_unmerged" in arrays_by_mode and "full_merged" in arrays_by_mode:
        left = arrays_by_mode["full_unmerged"]
        right = arrays_by_mode["full_merged"]
        report["comparisons"]["full_unmerged_vs_full_merged"] = {
            "query": compare_embedding_sets(
                left["query_embeddings"],
                right["query_embeddings"],
                left["top1"],
                right["top1"],
            ),
            "candidate": compare_embedding_sets(
                left["candidate_embeddings"], right["candidate_embeddings"]
            ),
        }
        # The query comparison includes both representation and ranking behavior.
        report["comparisons"]["full_unmerged_vs_full_merged"].update(
            report["comparisons"]["full_unmerged_vs_full_merged"]["query"]
        )

    primary_resolved = checkpoint.resolve()
    trajectory_paths = []
    for path in args.trajectory_checkpoint:
        resolved = path.expanduser().resolve()
        if resolved != primary_resolved and resolved not in trajectory_paths:
            trajectory_paths.append(resolved)
    if "full_merged" in arrays_by_mode:
        report["trajectory"][checkpoint.name] = report["modes"]["full_merged"]
    for trajectory_checkpoint in trajectory_paths:
        print(f"[trajectory:{trajectory_checkpoint.name}] loading and encoding...", flush=True)
        try:
            result, _ = run_mode(
                "full_merged",
                base_model,
                trajectory_checkpoint,
                prepared,
                device,
            )
            result["status"] = "success"
            report["trajectory"][trajectory_checkpoint.name] = result
            print(
                f"[trajectory:{trajectory_checkpoint.name}] "
                f"hit@1={result['retrieval_metrics']['hit@1']:.4f}, "
                f"offdiag_cos={result['query_embedding_statistics']['off_diagonal_cosine']['mean']:.6f}",
                flush=True,
            )
        except Exception as exc:
            report["trajectory"][trajectory_checkpoint.name] = {
                "status": "failed",
                "checkpoint": str(trajectory_checkpoint),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            atomic_write_json(partial_path, report)
            raise
        atomic_write_json(partial_path, report)

    report["conclusion"] = build_conclusion(report, thresholds)
    report["status"] = "success"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output, report)
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2))
    print(f"Diagnostic report: {output}")


if __name__ == "__main__":
    main()
