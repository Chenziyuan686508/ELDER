#!/usr/bin/env python3
"""Download only the metadata needed by the 78-task MMEB-V2 evaluation.

This script deliberately does not snapshot video repositories.  Video task
repositories can contain many gigabytes of raw media; instead, each requested
Hugging Face dataset split is loaded and materialized as the small local
JSON/JSONL file consumed by ELDER's existing evaluation loaders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from mmeb_v2_manifest import (
    IMAGE_TASKS,
    MVBench_SUBSETS,
    TASKS_BY_MODALITY,
    VIDEO_METADATA,
    VIDEO_TASKS,
)


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MMEB_V2_DATA_DIR",
        os.environ.get("MMEB_V3_DATA_DIR", "/root/autodl-tmp/datasets/MMEB-V2"),
    )
)
IMAGE_METADATA_REPO = "ziyjiang/MMEB_Test_Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MMEB-V2 query/label metadata without downloading raw "
            "image or video media. Existing valid metadata is skipped."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--modality",
        action="append",
        choices=("image", "video", "all"),
        help="May be repeated. Default: image and video.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name, or comma-separated task names. Default: all selected tasks.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact plan without importing download libraries or writing files.",
    )
    return parser.parse_args()


def selected_modalities(raw_modalities: list[str] | None) -> tuple[str, ...]:
    if not raw_modalities or "all" in raw_modalities:
        return ("image", "video")
    return tuple(dict.fromkeys(raw_modalities))


def selected_tasks(args: argparse.Namespace) -> dict[str, list[str]]:
    modalities = selected_modalities(args.modality)
    requested = []
    for item in args.task:
        requested.extend(name.strip() for name in item.split(",") if name.strip())

    allowed = {
        task_name
        for modality in modalities
        for task_name in TASKS_BY_MODALITY[modality]
    }
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise SystemExit(
            "Unknown task(s) for selected modalities: " + ", ".join(unknown)
        )

    result: dict[str, list[str]] = {}
    for modality in modalities:
        candidates = TASKS_BY_MODALITY[modality]
        result[modality] = (
            [name for name in candidates if name in requested]
            if requested
            else list(candidates)
        )
    return result


def nonempty_files(path: Path, pattern: str) -> list[Path]:
    if not path.is_dir():
        return []
    return [item for item in path.glob(pattern) if item.is_file() and item.stat().st_size]


def image_task_ready(data_root: Path, task_name: str) -> bool:
    return bool(nonempty_files(data_root / "image-query" / task_name, "**/*.parquet"))


def video_task_ready(data_root: Path, task_name: str) -> bool:
    destination = data_root / VIDEO_METADATA[task_name][3]
    if task_name == "MVBench":
        return all(
            (destination / f"{subset}.json").is_file()
            and (destination / f"{subset}.json").stat().st_size
            for subset in MVBench_SUBSETS
        )
    return destination.is_file() and destination.stat().st_size > 0


def print_plan(
    data_root: Path,
    tasks: dict[str, list[str]],
    force: bool,
) -> tuple[list[str], list[str]]:
    image_downloads = [
        name
        for name in tasks.get("image", [])
        if force or not image_task_ready(data_root, name)
    ]
    video_downloads = [
        name
        for name in tasks.get("video", [])
        if force or not video_task_ready(data_root, name)
    ]

    print(f"MMEB-V2 root: {data_root}")
    print("Media download: disabled (metadata only)")
    print(
        f"Image metadata: {len(image_downloads)} download, "
        f"{len(tasks.get('image', [])) - len(image_downloads)} skip"
    )
    for task_name in image_downloads:
        print(f"  [image] {task_name} <- {IMAGE_METADATA_REPO}/{task_name}/")
    print(
        f"Video metadata: {len(video_downloads)} download, "
        f"{len(tasks.get('video', [])) - len(video_downloads)} skip"
    )
    for task_name in video_downloads:
        repo, subset, split, destination, _ = VIDEO_METADATA[task_name]
        subset_text = subset if subset is not None else "<default>"
        if task_name == "MVBench":
            subset_text = f"{len(MVBench_SUBSETS)} task subsets"
        print(
            f"  [video] {task_name} <- {repo} "
            f"(config={subset_text}, split={split}) -> {destination}"
        )
    return image_downloads, video_downloads


def ensure_online_mode() -> None:
    blocked = [
        name
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
        if os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
    ]
    if blocked:
        raise RuntimeError(
            "Metadata download is disabled by "
            + ", ".join(blocked)
            + ". Unset these variables for this command."
        )


def download_image_metadata(
    data_root: Path,
    task_names: Iterable[str],
    revision: str,
    cache_dir: Path | None,
    max_workers: int,
) -> None:
    task_names = list(task_names)
    if not task_names:
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for image metadata") from exc

    destination = data_root / "image-query"
    destination.mkdir(parents=True, exist_ok=True)
    allow_patterns = ["README.md", ".gitattributes"]
    allow_patterns.extend(f"{task_name}/**" for task_name in task_names)
    snapshot_download(
        repo_id=IMAGE_METADATA_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )


def load_remote_dataset(repo: str, subset: str | None, split: str, cache_dir: Path | None):
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for video metadata") from exc

    kwargs = {
        "path": repo,
        "split": split,
        "cache_dir": str(cache_dir) if cache_dir else None,
    }
    if subset is not None:
        kwargs["name"] = subset
    dataset = load_dataset(**kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset from {repo}, got {type(dataset).__name__}")
    if len(dataset) == 0:
        raise ValueError(f"Remote dataset is empty: repo={repo}, subset={subset}, split={split}")
    return dataset


def validate_columns(task_name: str, columns: Iterable[str], required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            f"{task_name}: remote metadata is missing required columns: {missing}; "
            f"available={sorted(columns)}"
        )


def atomic_dataset_to_json(dataset, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        dataset.to_json(str(temp_path), force_ascii=False)
        if temp_path.stat().st_size == 0:
            raise ValueError(f"Refusing to install empty metadata: {destination}")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_json_array_from_dataset(dataset, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".jsonl.tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        jsonl_path = Path(handle.name)
    array_path = jsonl_path.with_suffix(".array.tmp")
    try:
        dataset.to_json(str(jsonl_path), force_ascii=False)
        records = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"Refusing to install empty metadata: {destination}")
        with array_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False)
        os.replace(array_path, destination)
    finally:
        jsonl_path.unlink(missing_ok=True)
        array_path.unlink(missing_ok=True)


def download_mvbench(
    data_root: Path,
    cache_dir: Path | None,
    force: bool,
) -> None:
    repo, _, split, relative_destination, required = VIDEO_METADATA["MVBench"]
    destination = data_root / relative_destination
    destination.mkdir(parents=True, exist_ok=True)
    for subset_name in MVBench_SUBSETS:
        subset_destination = destination / f"{subset_name}.json"
        if not force and subset_destination.is_file() and subset_destination.stat().st_size:
            print(f"[skip] MVBench/{subset_name}")
            continue
        print(f"[fetch] MVBench/{subset_name}")
        dataset = load_remote_dataset(repo, subset_name, split, cache_dir)
        validate_columns("MVBench", dataset.column_names, required)
        atomic_json_array_from_dataset(dataset, subset_destination)
        print(f"[ready] {subset_destination} ({len(dataset)} rows)")


def download_video_metadata(
    data_root: Path,
    task_names: Iterable[str],
    cache_dir: Path | None,
    force: bool,
) -> list[str]:
    failures = []
    for task_name in task_names:
        try:
            if task_name == "MVBench":
                download_mvbench(data_root, cache_dir, force)
                continue
            repo, subset, split, relative_destination, required = VIDEO_METADATA[task_name]
            destination = data_root / relative_destination
            print(f"[fetch] {task_name}: {repo}")
            dataset = load_remote_dataset(repo, subset, split, cache_dir)
            validate_columns(task_name, dataset.column_names, required)
            atomic_dataset_to_json(dataset, destination)
            print(f"[ready] {destination} ({len(dataset)} rows)")
        except Exception as exc:  # keep independent tasks progressing
            failures.append(task_name)
            print(f"[failed] {task_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return failures


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    tasks = selected_tasks(args)
    image_downloads, video_downloads = print_plan(args.data_root, tasks, args.force)
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0
    if not image_downloads and not video_downloads:
        print("All selected metadata is already present.")
        return 0

    ensure_online_mode()
    args.data_root.mkdir(parents=True, exist_ok=True)
    failures = []
    if image_downloads:
        try:
            download_image_metadata(
                args.data_root,
                image_downloads,
                args.revision,
                args.cache_dir,
                args.max_workers,
            )
            missing_after_download = [
                task_name
                for task_name in image_downloads
                if not image_task_ready(args.data_root, task_name)
            ]
            if missing_after_download:
                raise RuntimeError(
                    "Image snapshot completed but parquet is missing for: "
                    + ", ".join(missing_after_download)
                )
        except Exception as exc:
            failures.extend(image_downloads)
            print(f"[failed] image metadata: {type(exc).__name__}: {exc}", file=sys.stderr)

    failures.extend(
        download_video_metadata(
            args.data_root,
            video_downloads,
            args.cache_dir,
            args.force,
        )
    )
    if failures:
        print("Metadata preparation failed for: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("Selected MMEB-V2 metadata is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

