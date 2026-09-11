#!/usr/bin/env python3
"""Read-only integrity audit for real ELDER Stage 1 training batches."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_env_file(path: Path) -> list[str]:
    loaded = []
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values = shlex.split(value, comments=True, posix=True)
        os.environ.setdefault(key, os.path.expandvars(values[0] if values else ""))
        loaded.append(key)
    return loaded


def has_visual(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(bool(x) for x in payload.get("paths", [])) or any(
        x not in (None, b"") for x in payload.get("bytes", [])
    )


def has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(has_text(item) for item in value)
    return value is not None and bool(str(value).strip())


def direction(row: dict[str, Any]) -> str:
    query_visual = has_visual(row["query_image"])
    positive_visual = has_visual(row["pos_image"])
    if query_visual and positive_visual:
        return "visual_to_visual"
    if query_visual:
        return "visual_to_text"
    if positive_visual:
        return "text_to_visual"
    return "text_to_text"


def expected_global_labels(local_batch: int, world_size: int) -> list[int]:
    if local_batch < 1 or world_size < 1:
        raise ValueError("local_batch and world_size must be positive")
    return list(range(local_batch * world_size))


def flattened_metadata(
    inputs: dict[str, Any], side: str, chunk_size: int
) -> dict[str, list[Any]]:
    from src.data.collator.train_collator import split_and_process_vlm_inputs

    result = {"text": [], "global_dataset_name": []}
    for chunk in split_and_process_vlm_inputs({side: inputs}, chunk_size):
        for key in result:
            result[key].extend(list(chunk[side][key]))
    return result


def visual_mask(inputs: dict[str, Any]) -> list[bool]:
    size = int(inputs["input_ids"].shape[0])
    image = inputs.get("pixel_values", [None] * size)
    video = inputs.get("pixel_values_videos", [None] * size)
    image = image if isinstance(image, list) else [image] * size
    video = video if isinstance(video, list) else [video] * size
    return [image[i] is not None or video[i] is not None for i in range(size)]


def audit_processed_batch(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    positive: dict[str, Any],
    q_chunk: int,
    p_chunk: int,
    world_size: int,
) -> dict[str, Any]:
    errors = []
    for side, inputs, image_key, chunk in (
        ("qry", query, "query_image", q_chunk),
        ("tgt", positive, "pos_image", p_chunk),
    ):
        if int(inputs["input_ids"].shape[0]) != len(rows):
            errors.append(f"{side}: input_ids batch size mismatch")
        if int(inputs["attention_mask"].shape[0]) != len(rows):
            errors.append(f"{side}: attention_mask batch size mismatch")
        if len(inputs["text"]) != len(rows):
            errors.append(f"{side}: text metadata batch size mismatch")
        if len(inputs["global_dataset_name"]) != len(rows):
            errors.append(f"{side}: source metadata batch size mismatch")
        lost = [
            i
            for i, (raw, processed) in enumerate(
                zip(
                    (has_visual(row[image_key]) for row in rows),
                    visual_mask(inputs),
                )
            )
            if raw and not processed
        ]
        if lost:
            errors.append(f"{side}: visual payload lost at rows {lost[:10]}")
        flattened = flattened_metadata(inputs, side, chunk)
        if flattened["text"] != list(inputs["text"]):
            errors.append(f"{side}: GradCache chunking reordered text")
        if flattened["global_dataset_name"] != list(inputs["global_dataset_name"]):
            errors.append(f"{side}: GradCache chunking reordered sources")
    for i, row in enumerate(rows):
        if not (has_text(row["query_text"]) or has_visual(row["query_image"])):
            errors.append(f"row {i}: empty query")
        if not (has_text(row["pos_text"]) or has_visual(row["pos_image"])):
            errors.append(f"row {i}: empty positive")
    return {
        "batch_size": len(rows),
        "global_infonce_labels": expected_global_labels(len(rows), world_size),
        "source_counts": dict(Counter(row["global_dataset_name"] for row in rows)),
        "direction_counts": dict(Counter(direction(row) for row in rows)),
        "errors": errors,
        "passed": not errors,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("experiments/elder/stage1_full_local.yaml"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/root/autodl-tmp/datasets")
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env.elder"))
    parser.add_argument("--batch-size", type=int, default=171)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument("--interleave-global-chunk", type=int, default=66)
    parser.add_argument("--gc-q-chunk-size", type=int, default=3)
    parser.add_argument("--gc-p-chunk-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resize-max-pixels", type=int, default=1003520)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.base_model, "base model"),
        (args.dataset_config, "dataset config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    positive_values = (
        args.batch_size,
        args.batches,
        args.world_size,
        args.interleave_global_chunk,
        args.gc_q_chunk_size,
        args.gc_p_chunk_size,
    )
    if min(positive_values) < 1:
        raise ValueError("all batch and chunk sizes must be positive")
    args.base_model = args.base_model.resolve()
    args.dataset_config = args.dataset_config.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    env_keys = load_env_file(args.env_file.resolve())
    if args.dry_run:
        print("Stage 1 batch-audit configuration:")
        print(f"  config: {args.dataset_config}")
        print(f"  local batch/world: {args.batch_size}/{args.world_size}")
        print(f"  interleave global chunk: {args.interleave_global_chunk}")
        print(f"  GradCache q/p chunks: {args.gc_q_chunk_size}/{args.gc_p_chunk_size}")
        print(f"  loaded env keys: {len(env_keys)}")
        print("Dry run complete; no data was loaded.")
        return

    import yaml
    from torch.utils.data import DataLoader

    import src.data.dataset  # noqa: F401
    from src.arguments import DataArguments, ModelArguments
    from src.data.collator.train_collator import MultimodalDataCollator
    from src.data.loader.mixed_dataset import init_mixed_dataset
    from src.model.processor import load_processor
    from src.utils.config_utils import resolve_dataset_paths

    config = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))
    resolve_dataset_paths(config, data_basedir=str(args.data_root))
    model_args = ModelArguments(
        model_name=str(args.base_model),
        processor_name=str(args.base_model),
        lora=False,
    )
    model_args.model_backbone = "qwen2_vl"
    data_args = DataArguments(
        dataset_config=str(args.dataset_config),
        data_basedir=str(args.data_root),
        max_len=-1,
        resize_use_processor=True,
        resize_max_pixels=args.resize_max_pixels,
    )
    context = SimpleNamespace(
        per_device_train_batch_size=args.batch_size,
        homogeneous_batch_size_per_device=args.interleave_global_chunk,
        interleave_batch_size=0,
        seed=args.seed,
        interleave_stopping_strategy="all_exhausted",
        dataloader_num_workers=args.workers,
        model_backbone="qwen2_vl",
    )
    processor = load_processor(model_args, data_args)
    dataset = init_mixed_dataset(config, model_args, data_args, context)
    collator = MultimodalDataCollator(
        processor, model_args, data_args, context, batch_size=args.batch_size
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda rows: rows,
    )
    report: dict[str, Any] = {
        "status": "running",
        "dataset_config": str(args.dataset_config),
        "base_model": str(args.base_model),
        "loaded_env_keys": env_keys,
        "settings": {
            "local_batch_size": args.batch_size,
            "simulated_world_size": args.world_size,
            "interleave_global_chunk": args.interleave_global_chunk,
            "gc_q_chunk_size": args.gc_q_chunk_size,
            "gc_p_chunk_size": args.gc_p_chunk_size,
        },
        "batches": [],
    }
    partial = args.output.with_name(args.output.stem + ".partial" + args.output.suffix)
    write_json(partial, report)
    for index, rows in enumerate(loader):
        if index >= args.batches:
            break
        try:
            query, positive = collator(rows)
            result = audit_processed_batch(
                rows,
                query,
                positive,
                args.gc_q_chunk_size,
                args.gc_p_chunk_size,
                args.world_size,
            )
        except Exception as exc:
            result = {
                "batch_size": len(rows),
                "passed": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        result["batch_index"] = index
        report["batches"].append(result)
        write_json(partial, report)
        print(
            f"batch={index} passed={result['passed']} "
            f"errors={len(result['errors'])}",
            flush=True,
        )
    passed = sum(row["passed"] for row in report["batches"])
    report["summary"] = {
        "requested_batches": args.batches,
        "completed_batches": len(report["batches"]),
        "passed_batches": passed,
        "failed_batches": len(report["batches"]) - passed,
    }
    report["status"] = "success" if report["summary"]["failed_batches"] == 0 else "failed"
    write_json(args.output, report)
    partial.unlink(missing_ok=True)
    print(f"Wrote report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

