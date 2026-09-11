#!/usr/bin/env python3
"""Validate, account for, and optionally merge ELDER CoT generation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.elder.cot import (  # noqa: E402
    COT_SCHEMA_VERSION,
    STAGE_KEYS,
    validate_cot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated ELDER CoT shards.")
    parser.add_argument(
        "--root", type=Path, default=Path("/root/autodl-tmp/datasets/ELDER-CoT")
    )
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--merge-output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-errors", type=int, default=100)
    return parser.parse_args()


def jsonl(path: Path) -> Iterable[tuple[int, dict, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line), line
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def expected_samples(manifest_dir: Path) -> tuple[dict[str, str], Counter]:
    with (manifest_dir / "manifest_index.json").open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    expected: dict[str, str] = {}
    task_counts = Counter()
    for task in index["tasks"]:
        path = manifest_dir / task["path"]
        for line_number, row, _ in jsonl(path):
            sample_id = row.get("sample_id")
            if not sample_id:
                raise ValueError(f"Missing sample_id at {path}:{line_number}")
            if sample_id in expected:
                raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
            expected[sample_id] = row["task_id"]
            task_counts[row["task_id"]] += 1
    return expected, task_counts


def validate_side(side: str, value: dict) -> list[str]:
    errors: list[str] = []
    enabled = value.get("cot_enabled")
    steps = value.get("cot_steps")
    mask = value.get("cot_step_mask")
    if not isinstance(enabled, bool):
        return [f"{side}:cot_enabled_not_bool"]
    if enabled:
        if not isinstance(steps, list) or len(steps) != 4:
            errors.append(f"{side}:enabled_steps_not_length_4")
        if mask != [1, 1, 1, 1]:
            errors.append(f"{side}:enabled_mask_invalid")
        if isinstance(steps, list) and len(steps) == 4:
            result = validate_cot(dict(zip(STAGE_KEYS, steps)), side=side)
            errors.extend(f"{side}:{error}" for error in result.errors)
    else:
        if steps != []:
            errors.append(f"{side}:disabled_steps_not_empty")
        if mask != [0, 0, 0, 0]:
            errors.append(f"{side}:disabled_mask_invalid")
    return errors


def atomic_report(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def merge_shards(paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as output:
        temp_path = Path(output.name)
        for path in paths:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        output.write(line if line.endswith("\n") else line + "\n")
    os.replace(temp_path, destination)


def main() -> int:
    args = parse_args()
    manifest_dir = args.manifest_dir or args.root / "manifests"
    generated_dir = args.generated_dir or args.root / "generated"
    report_path = args.report or args.root / "validation_report.json"
    expected, expected_by_task = expected_samples(manifest_dir)
    shard_paths = sorted(generated_dir.glob("cot-shard-*-of-*.jsonl"))
    failure_paths = sorted(generated_dir.glob("failures-shard-*-of-*.jsonl"))
    seen: set[str] = set()
    cot_fingerprints: dict[bytes, str] = {}
    completed_by_task = Counter()
    gate_counts = Counter()
    errors: list[dict] = []
    duplicate_count = template_duplicate_count = extra_count = invalid_count = 0

    def record_error(path: Path, line_number: int, sample_id: object, issues: list[str]) -> None:
        nonlocal invalid_count
        invalid_count += 1
        if len(errors) < args.max_errors:
            errors.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "sample_id": sample_id,
                    "errors": issues,
                }
            )

    for path in shard_paths:
        for line_number, row, _ in jsonl(path):
            sample = row.get("sample_id")
            row_errors: list[str] = []
            if row.get("schema_version") != COT_SCHEMA_VERSION:
                row_errors.append("schema_version_mismatch")
            if sample in seen:
                duplicate_count += 1
                row_errors.append("duplicate_generated_sample")
            else:
                seen.add(sample)
            expected_task = expected.get(sample)
            if expected_task is None:
                extra_count += 1
                row_errors.append("sample_not_in_manifest")
            elif row.get("task_id") != expected_task:
                row_errors.append("task_id_mismatch")
            for side in ("query", "candidate"):
                value = row.get(side)
                if not isinstance(value, dict):
                    row_errors.append(f"{side}:missing_or_not_object")
                    continue
                row_errors.extend(validate_side(side, value))
                if isinstance(value.get("cot_enabled"), bool):
                    gate_counts[f"{side}_{'enabled' if value['cot_enabled'] else 'disabled'}"] += 1
                if value.get("cot_enabled") and isinstance(value.get("cot_steps"), list) and len(value["cot_steps"]) == 4:
                    fingerprint = hashlib.sha256(
                        (side + "\0" + "\0".join(value["cot_steps"])).encode("utf-8")
                    ).digest()
                    previous_sample = cot_fingerprints.get(fingerprint)
                    if previous_sample is not None and previous_sample != sample:
                        template_duplicate_count += 1
                        row_errors.append(f"{side}:duplicate_cot_template:{previous_sample}")
                    else:
                        cot_fingerprints[fingerprint] = sample
            if row_errors:
                record_error(path, line_number, sample, row_errors)
            elif expected_task is not None:
                completed_by_task[expected_task] += 1

    failure_ids: set[str] = set()
    failure_events = 0
    for path in failure_paths:
        for _, row, _ in jsonl(path):
            failure_events += 1
            if row.get("sample_id"):
                failure_ids.add(row["sample_id"])
    unresolved_failure_ids = failure_ids - seen
    missing = set(expected) - seen
    incomplete_tasks = {
        task: expected_count - completed_by_task[task]
        for task, expected_count in sorted(expected_by_task.items())
        if completed_by_task[task] != expected_count
    }
    report = {
        "schema_version": COT_SCHEMA_VERSION,
        "manifest_dir": str(manifest_dir),
        "generated_dir": str(generated_dir),
        "shard_files": [str(path) for path in shard_paths],
        "expected_samples": len(expected),
        "generated_unique_samples": len(seen),
        "valid_completed_samples": sum(completed_by_task.values()),
        "missing_samples": len(missing),
        "extra_samples": extra_count,
        "duplicate_samples": duplicate_count,
        "duplicate_cot_templates": template_duplicate_count,
        "invalid_samples": invalid_count,
        "failure_events": failure_events,
        "unresolved_failure_samples": len(unresolved_failure_ids),
        "gate_counts": dict(sorted(gate_counts.items())),
        "expected_by_task": dict(sorted(expected_by_task.items())),
        "completed_by_task": dict(sorted(completed_by_task.items())),
        "incomplete_by_task": incomplete_tasks,
        "error_examples": errors,
    }
    atomic_report(report_path, report)
    complete = not (
        missing or extra_count or duplicate_count or invalid_count or unresolved_failure_ids
    )
    print(
        f"CoT validation: expected={len(expected):,}, generated={len(seen):,}, "
        f"valid={sum(completed_by_task.values()):,}, missing={len(missing):,}, "
        f"invalid={invalid_count:,}, duplicate={duplicate_count:,}, "
        f"unresolved_failures={len(unresolved_failure_ids):,}"
    )
    print(f"Report: {report_path}")
    if args.merge_output:
        if not complete:
            raise SystemExit("Refusing to merge an incomplete or invalid generation run.")
        merge_shards(shard_paths, args.merge_output)
        print(f"Merged training data: {args.merge_output}")
    if complete:
        return 0
    return 0 if args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
