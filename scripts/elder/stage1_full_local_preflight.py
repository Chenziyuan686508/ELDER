#!/usr/bin/env python
"""Validate ELDER's full local Stage 1 data layout without loading the model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq
import yaml
from datasets import load_from_disk

from src.utils.config_utils import resolve_dataset_paths
from src.utils.vision_utils.vision_utils import process_video_frames


CANONICAL_VISRAG_SHARD = re.compile(r"train-\d{5}-of-\d{5}\.parquet")


def _read_first_row(path: Path, columns: list[str]) -> dict:
    parquet_file = pq.ParquetFile(path)
    batch = next(parquet_file.iter_batches(batch_size=1, columns=columns))
    return batch.to_pylist()[0]


def _canonical_visrag_shards(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.glob("*.parquet")
        if CANONICAL_VISRAG_SHARD.fullmatch(candidate.name)
    )


def _validate_mmeb(config: dict) -> dict:
    entries = [item for item in config.values() if item["dataset_parser"] == "mmeb"]
    missing = []
    selected_rows = 0
    for entry in entries:
        root = Path(entry["dataset_path"])
        subset = entry["subset_name"]
        split = entry.get("dataset_split", "original")
        metadata_files = sorted((root / subset).glob(f"{split}-*.parquet"))
        if not metadata_files:
            missing.append(f"metadata:{subset}")
            continue
        sample = _read_first_row(
            metadata_files[0],
            ["qry_image_path", "pos_image_path", "neg_image_path"],
        )
        for key, image_path in sample.items():
            if image_path and not (Path(entry["image_dir"]) / image_path).is_file():
                missing.append(f"image:{subset}:{key}:{image_path}")
        selected_rows += int(entry["num_sample_per_subset"])
    if missing:
        raise FileNotFoundError("Missing local MMEB assets: " + ", ".join(missing[:10]))
    return {"subsets": len(entries), "configured_samples": selected_rows}


def _validate_colpali(config: dict) -> dict:
    entries = [item for item in config.values() if item["dataset_parser"] == "vidore"]
    if len(entries) != 1:
        raise ValueError(f"Expected one local ViDoRe entry, found {len(entries)}")
    dataset = load_from_disk(entries[0]["dataset_path"])
    required_columns = {"image", "query", "answer"}
    missing_columns = sorted(required_columns.difference(dataset.column_names))
    if missing_columns:
        raise ValueError(f"ColPali is missing columns: {missing_columns}")
    if dataset.num_rows == 0:
        raise ValueError("ColPali dataset is empty")
    return {"rows": dataset.num_rows}


def _validate_visrag(config: dict) -> dict:
    entries = [item for item in config.values() if item["dataset_parser"] == "visrag"]
    if len(entries) != 1:
        raise ValueError(f"Expected one local VisRAG entry, found {len(entries)}")
    shard_dir = Path(entries[0]["dataset_path"])
    shards = _canonical_visrag_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(f"No canonical VisRAG shards found in {shard_dir}")
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
    return {"shards": len(shards), "rows": rows}


def _validate_video(config: dict) -> dict:
    entries = [item for item in config.values() if item["dataset_parser"].startswith("llavahound_")]
    by_annotation = {}
    for entry in entries:
        annotation_path = Path(entry["dataset_path"])
        frame_root = Path(entry["video_frame_basedir"])
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing LLaVA-Hound annotations: {annotation_path}")
        if not frame_root.is_dir():
            raise FileNotFoundError(f"Missing LLaVA-Hound frames: {frame_root}")
        by_annotation.setdefault(annotation_path, frame_root)

    reports = {}
    for annotation_path, frame_root in by_annotation.items():
        with annotation_path.open() as handle:
            first = json.loads(next(handle))
        video_id = first["video"]
        frames = process_video_frames(str(frame_root / video_id), num_frames=8)
        if len(frames) != 8 or not all(Path(frame).is_file() for frame in frames):
            raise FileNotFoundError(
                f"Unable to sample eight frames for LLaVA-Hound video {video_id!r}"
            )
        reports[annotation_path.name] = {"sample_video": video_id, "sampled_frames": len(frames)}
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("experiments/elder/stage1_full_local.yaml"),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid local model path: {args.model_path}")

    with args.dataset_config.open() as handle:
        config = yaml.safe_load(handle)
    resolve_dataset_paths(config)
    parser_counts = Counter(entry["dataset_parser"] for entry in config.values())
    expected_counts = {"mmeb": 20, "vidore": 1, "visrag": 1, "llavahound_caption": 2, "llavahound_qa": 1}
    if dict(parser_counts) != expected_counts:
        raise ValueError(f"Unexpected full-local parser mix: {dict(parser_counts)}")

    report = {
        "status": "ok",
        "model_path": str(args.model_path),
        "dataset_config": str(args.dataset_config),
        "mmeb": _validate_mmeb(config),
        "colpali": _validate_colpali(config),
        "visrag": _validate_visrag(config),
        "llavahound": _validate_video(config),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
