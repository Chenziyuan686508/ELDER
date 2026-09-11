#!/usr/bin/env python3
"""Validate the exact 78-task MMEB-V2 local evaluation layout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from mmeb_v2_manifest import (
    IMAGE_TASKS,
    MVBench_SUBSETS,
    TASKS_BY_MODALITY,
    VIDEO_FRAME_ROOTS,
    VIDEO_METADATA,
    VISDOC_LAYOUT,
)


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MMEB_V2_DATA_DIR",
        os.environ.get("MMEB_V3_DATA_DIR", "/root/autodl-tmp/datasets/MMEB-V2"),
    )
)
DEFAULT_CONFIG_DIR = Path("experiments/elder/mmeb_v2")
IMAGE_REQUIRED_COLUMNS = {
    "qry_text",
    "qry_img_path",
    "tgt_text",
    "tgt_img_path",
    "qry_inst",
    "tgt_inst",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check MMEB-V2 configs, metadata, and extracted media for 78 tasks."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--modality",
        action="append",
        choices=("image", "video", "visdoc", "all"),
        help="May be repeated. Default: all three modalities.",
    )
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Only validate configs and metadata; do not inspect extracted media roots.",
    )
    parser.add_argument(
        "--media-check-limit",
        type=int,
        default=100,
        help=(
            "Image metadata rows sampled per task for path validation. "
            "Use 0 to check every row. Default: 100."
        ),
    )
    parser.add_argument("--json-report", type=Path)
    return parser.parse_args()


def selected_modalities(raw: list[str] | None) -> tuple[str, ...]:
    if not raw or "all" in raw:
        return ("image", "video", "visdoc")
    return tuple(dict.fromkeys(raw))


def any_nonempty_file(root: Path) -> bool:
    if not root.is_dir():
        return False
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(directory) / filename
            try:
                if path.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def nonempty_parquet_files(root: Path, split: str | None = None) -> list[Path]:
    if not root.is_dir():
        return []
    patterns = [f"{split}*.parquet"] if split else []
    patterns.append("*.parquet")
    for pattern in patterns:
        paths = [
            path
            for path in sorted(root.glob(pattern))
            if path.is_file() and path.stat().st_size > 0
        ]
        if paths:
            return paths
    return []


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_strings(item)


def check_image_task(
    data_root: Path,
    task_name: str,
    skip_media: bool,
    row_limit: int,
) -> tuple[list[str], dict[str, int]]:
    issues = []
    stats = {"metadata_files": 0, "media_references_checked": 0}
    metadata_dir = data_root / "image-query" / task_name
    parquet_files = nonempty_parquet_files(metadata_dir, "test")
    stats["metadata_files"] = len(parquet_files)
    if not parquet_files:
        return [f"missing metadata parquet: {metadata_dir}"], stats

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ["pyarrow is required to inspect image metadata"], stats

    schema_names = set(pq.ParquetFile(parquet_files[0]).schema_arrow.names)
    missing_columns = sorted(IMAGE_REQUIRED_COLUMNS - schema_names)
    if missing_columns:
        issues.append(f"metadata columns missing: {missing_columns}")

    if skip_media:
        return issues, stats

    image_root = data_root / "image-tasks" / "MMEB"
    task_media_root = image_root / task_name
    if not any_nonempty_file(task_media_root):
        issues.append(f"missing or empty media directory: {task_media_root}")
        return issues, stats

    missing_paths = []
    rows_seen = 0
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        available_path_columns = [
            name for name in ("qry_img_path", "tgt_img_path") if name in schema_names
        ]
        for batch in parquet.iter_batches(batch_size=128, columns=available_path_columns):
            for row in batch.to_pylist():
                if row_limit and rows_seen >= row_limit:
                    break
                rows_seen += 1
                for column_name in available_path_columns:
                    for relative_path in flatten_strings(row.get(column_name)):
                        candidate = Path(relative_path)
                        if not candidate.is_absolute():
                            candidate = image_root / candidate
                        stats["media_references_checked"] += 1
                        if not candidate.is_file() and len(missing_paths) < 5:
                            missing_paths.append(str(candidate))
            if row_limit and rows_seen >= row_limit:
                break
        if row_limit and rows_seen >= row_limit:
            break
    if missing_paths:
        issues.append(
            f"missing sampled media references ({len(missing_paths)} shown): "
            + "; ".join(missing_paths)
        )
    return issues, stats


def first_json_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first_nonspace = ""
        while True:
            character = handle.read(1)
            if not character:
                raise ValueError("empty JSON file")
            if not character.isspace():
                first_nonspace = character
                break
        handle.seek(0)
        if first_nonspace == "[":
            payload = json.load(handle)
            if not isinstance(payload, list) or not payload:
                raise ValueError("expected a non-empty JSON array")
            record = payload[0]
        else:
            line = handle.readline()
            record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError(f"expected object record, got {type(record).__name__}")
    return record


def check_video_task(
    data_root: Path,
    task_name: str,
    skip_media: bool,
) -> tuple[list[str], dict[str, int]]:
    issues = []
    stats = {"metadata_files": 0, "media_references_checked": 0}
    _, _, _, relative_destination, required_columns = VIDEO_METADATA[task_name]
    destination = data_root / relative_destination

    metadata_paths: list[Path]
    if task_name == "MVBench":
        metadata_paths = [destination / f"{subset}.json" for subset in MVBench_SUBSETS]
    else:
        metadata_paths = [destination]
    missing = [path for path in metadata_paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        issues.append(
            f"missing metadata ({len(missing)}/{len(metadata_paths)}): "
            + ", ".join(str(path) for path in missing[:5])
        )
    existing = [path for path in metadata_paths if path not in missing]
    stats["metadata_files"] = len(existing)
    if existing:
        try:
            record = first_json_record(existing[0])
            missing_columns = sorted(set(required_columns) - set(record))
            if missing_columns:
                issues.append(f"metadata columns missing: {missing_columns}")
        except Exception as exc:
            issues.append(f"invalid metadata {existing[0]}: {type(exc).__name__}: {exc}")

    if not skip_media:
        frame_root = data_root / VIDEO_FRAME_ROOTS[task_name]
        if not any_nonempty_file(frame_root):
            issues.append(f"missing or empty frame directory: {frame_root}")
    return issues, stats


def parquet_has_embedded_images(parquet_files: Iterable[Path]) -> bool:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return False
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        if "image" not in parquet.schema_arrow.names:
            continue
        for batch in parquet.iter_batches(batch_size=8, columns=["image"]):
            for value in batch.column(0).to_pylist():
                if isinstance(value, dict):
                    if value.get("bytes") or value.get("path"):
                        return True
                elif value is not None:
                    return True
            break
    return False


def check_visdoc_task(
    data_root: Path,
    task_name: str,
    skip_media: bool,
) -> tuple[list[str], dict[str, int]]:
    issues = []
    stats = {
        "metadata_files": 0,
        "media_references_checked": 0,
        "embedded_media_fallback": 0,
    }
    data_dir_name, image_dir_name, split = VISDOC_LAYOUT[task_name]
    data_dir = data_root / "visdoc-tasks" / "data" / data_dir_name
    corpus_files: list[Path] = []
    for subset in ("queries", "corpus", "qrels"):
        files = nonempty_parquet_files(data_dir / subset, split)
        if subset == "corpus":
            corpus_files = files
        stats["metadata_files"] += len(files)
        if not files:
            issues.append(f"missing {subset} parquet: {data_dir / subset}")
    if not skip_media:
        image_root = data_root / "visdoc-tasks" / "images" / image_dir_name
        if not any_nonempty_file(image_root):
            if parquet_has_embedded_images(corpus_files):
                stats["embedded_media_fallback"] = 1
            else:
                issues.append(
                    f"missing image directory and corpus has no embedded image: {image_root}"
                )
    return issues, stats


def validate_config(config_path: Path, expected_tasks: tuple[str, ...]) -> list[str]:
    if not config_path.is_file():
        return [f"missing config: {config_path}"]
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid YAML {config_path}: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return [f"config is not a task mapping: {config_path}"]
    actual_tasks = tuple(payload)
    issues = []
    missing = [name for name in expected_tasks if name not in payload]
    extra = [name for name in actual_tasks if name not in expected_tasks]
    if missing:
        issues.append("config missing tasks: " + ", ".join(missing))
    if extra:
        issues.append("config has non-V2 tasks: " + ", ".join(extra))
    if len(actual_tasks) != len(expected_tasks):
        issues.append(
            f"config task count is {len(actual_tasks)}, expected {len(expected_tasks)}"
        )
    return issues


def main() -> int:
    args = parse_args()
    if args.media_check_limit < 0:
        raise SystemExit("--media-check-limit must be non-negative")
    data_root = args.data_root.expanduser().resolve()
    config_dir = args.config_dir.expanduser()
    if not config_dir.is_absolute():
        config_dir = (Path.cwd() / config_dir).resolve()
    modalities = selected_modalities(args.modality)

    report: dict[str, Any] = {
        "data_root": str(data_root),
        "config_dir": str(config_dir),
        "modalities": {},
        "config_issues": {},
    }
    total_counter = Counter()
    print(f"MMEB-V2 root: {data_root}")
    print(f"Config root: {config_dir}")
    print(f"Media validation: {'disabled' if args.skip_media else 'enabled'}")

    for modality in modalities:
        expected_tasks = TASKS_BY_MODALITY[modality]
        config_issues = validate_config(config_dir / f"{modality}.yaml", expected_tasks)
        report["config_issues"][modality] = config_issues
        if config_issues:
            for issue in config_issues:
                print(f"[CONFIG:{modality}] {issue}")

        task_report = {}
        ready_count = 0
        for task_name in expected_tasks:
            if modality == "image":
                issues, stats = check_image_task(
                    data_root,
                    task_name,
                    args.skip_media,
                    args.media_check_limit,
                )
            elif modality == "video":
                issues, stats = check_video_task(data_root, task_name, args.skip_media)
            else:
                issues, stats = check_visdoc_task(data_root, task_name, args.skip_media)
            status = "ready" if not issues else "missing"
            if status == "ready":
                ready_count += 1
            else:
                print(f"[{modality}:{task_name}] " + " | ".join(issues))
            task_report[task_name] = {"status": status, "issues": issues, **stats}

        if config_issues:
            ready_count = 0
        missing_count = len(expected_tasks) - ready_count
        report["modalities"][modality] = {
            "expected": len(expected_tasks),
            "ready": ready_count,
            "missing": missing_count,
            "tasks": task_report,
        }
        total_counter["expected"] += len(expected_tasks)
        total_counter["ready"] += ready_count
        total_counter["missing"] += missing_count
        print(
            f"{modality.capitalize()}: {ready_count}/{len(expected_tasks)} ready, "
            f"{missing_count} missing or broken"
        )

    report["summary"] = dict(total_counter)
    print(
        f"Total: {total_counter['ready']}/{total_counter['expected']} ready, "
        f"{total_counter['missing']} missing or broken"
    )
    if args.json_report:
        report_path = args.json_report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report: {report_path}")
    return 0 if total_counter["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

