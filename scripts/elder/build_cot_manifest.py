#!/usr/bin/env python3
"""Build deterministic, task-balanced ELDER CoT generation manifests.

No model is loaded here.  The script samples at most 50,000 valid pairs per
real task and records references to local media.  Embedded Arrow/Parquet
images remain references and are decoded only by the generation workers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.elder.cot import (  # noqa: E402
    COT_SCHEMA_VERSION,
    clean_input_text,
    safe_task_filename,
    sample_id,
    semantic_richness_gate,
    stable_int,
)


DEFAULT_CONFIG = REPO_ROOT / "experiments/elder/cot_generation_local.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Reservoir:
    limit: int
    rng: random.Random

    def __post_init__(self) -> None:
        self.seen = 0
        self.items: list[dict] = []

    def add(self, item: dict) -> None:
        self.seen += 1
        if len(self.items) < self.limit:
            self.items.append(item)
            return
        replacement = self.rng.randrange(self.seen)
        if replacement < self.limit:
            self.items[replacement] = item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible per-task manifests for offline GLM CoT generation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-per-task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task ID or comma-separated task IDs. Default: all configured tasks.",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        help="Debug only: inspect at most this many source rows per dataset task.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")

    def expand(value):
        if isinstance(value, str):
            result = os.path.expanduser(os.path.expandvars(value))
            if re.search(r"\$\{?[A-Za-z_]", result):
                raise ValueError(
                    f"Unresolved environment variable in {value!r}. "
                    "Source .env.elder or use the four-GPU launcher."
                )
            return result
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return expand(raw)


def requested_tasks(raw: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for item in raw:
        result.update(part.strip() for part in item.split(",") if part.strip())
    return result


def configured_task_ids(config: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    datasets = config["datasets"]
    for task in datasets["mmeb"]["tasks"]:
        result.append(f"mmeb__{task['name']}")
    for task in datasets["llavahound"]["tasks"]:
        result.append(f"llavahound__{task['name']}")
    # ColPali and VisRAG tasks are discovered from their source columns.
    result.extend(f"colpali__{name}" for name in datasets["colpali"]["expected_sources"])
    result.extend(f"visrag__{name}" for name in datasets["visrag"]["expected_sources"])
    return result


def selected(task_id: str, selection: set[str]) -> bool:
    return not selection or task_id in selection


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temp_path, path)
    return count


def make_reservoir(task_id: str, limit: int, seed: int) -> Reservoir:
    return Reservoir(limit=limit, rng=random.Random(stable_int(task_id, seed)))


def make_row(
    *,
    dataset: str,
    task: str,
    source_index: int,
    query_text: object,
    query_media: list[dict],
    candidate_text: object,
    candidate_media: list[dict],
    source_metadata: Mapping[str, object] | None = None,
) -> dict | None:
    query_text = clean_input_text(query_text)
    candidate_text = clean_input_text(candidate_text)
    if not (query_text or query_media) or not (candidate_text or candidate_media):
        return None
    query_gate = semantic_richness_gate(query_text, query_media, side="query")
    candidate_gate = semantic_richness_gate(
        candidate_text, candidate_media, side="candidate"
    )
    task_id = f"{dataset}__{task}"
    return {
        "schema_version": COT_SCHEMA_VERSION,
        "sample_id": sample_id(dataset, task, source_index),
        "dataset": dataset,
        "task": task,
        "task_id": task_id,
        "source_index": source_index,
        "query": {
            "text": query_text,
            "media": query_media,
            "cot_enabled": query_gate.enabled,
            "cot_gate_reason": query_gate.reason,
        },
        "candidate": {
            "text": candidate_text,
            "media": candidate_media,
            "cot_enabled": candidate_gate.enabled,
            "cot_gate_reason": candidate_gate.reason,
        },
        "source_metadata": dict(source_metadata or {}),
    }


def media_path(path: Path, *, temporal_index: int | None = None) -> dict:
    result: dict[str, object] = {"type": "image_path", "path": str(path)}
    if temporal_index is not None:
        result["temporal_index"] = temporal_index
        result["sequence_type"] = "video_frames"
    return result


def embedded_media(dataset_key: str, row_index: int, column: str = "image") -> dict:
    return {
        "type": "dataset_image",
        "dataset_key": dataset_key,
        "row_index": row_index,
        "column": column,
    }


def parquet_media(path: Path, row_index: int, column: str = "image") -> dict:
    return {
        "type": "parquet_image",
        "path": str(path),
        "row_index": row_index,
        "column": column,
    }


def uniform_frames(frame_dir: Path, count: int) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    names = [
        entry.name
        for entry in os.scandir(frame_dir)
        if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
    ]

    def natural_key(name: str):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

    names.sort(key=natural_key)
    if not names or count <= 0:
        return []
    if count == 1:
        indices = [0]
    else:
        indices = [(position * (len(names) - 1)) // (count - 1) for position in range(count)]
    return [frame_dir / names[index] for index in indices]


def frame_dir_has_media(frame_dir: Path) -> bool:
    if not frame_dir.is_dir():
        return False
    try:
        return any(
            entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
            for entry in os.scandir(frame_dir)
        )
    except OSError:
        return False


def limited_count(count: int, debug_limit: int | None) -> int:
    return min(count, debug_limit) if debug_limit is not None else count


def build_mmeb(
    config: dict,
    selection: set[str],
    limit: int,
    seed: int,
    debug_limit: int | None,
) -> Iterator[tuple[str, Reservoir]]:
    from datasets import load_dataset

    root = Path(config["root"])
    split = config.get("split", "original")
    for task_config in config["tasks"]:
        task = task_config["name"]
        task_id = f"mmeb__{task}"
        if not selected(task_id, selection):
            continue
        files = sorted((root / task).glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"{task_id}: no {split} parquet under {root / task}")
        dataset = load_dataset(
            "parquet", data_files=[str(path) for path in files], split="train"
        )
        source_rows = limited_count(
            min(len(dataset), int(task_config.get("max_source_rows", len(dataset)))),
            debug_limit,
        )
        reservoir = make_reservoir(task_id, limit, seed)
        for index in range(source_rows):
            item = dataset[index]
            query_media = []
            candidate_media = []
            if item.get("qry_image_path"):
                path = root / item["qry_image_path"]
                if not path.is_file():
                    continue
                query_media.append(media_path(path))
            if item.get("pos_image_path"):
                path = root / item["pos_image_path"]
                if not path.is_file():
                    continue
                candidate_media.append(media_path(path))
            row = make_row(
                dataset="mmeb",
                task=task,
                source_index=index,
                query_text=item.get("qry"),
                query_media=query_media,
                candidate_text=item.get("pos_text"),
                candidate_media=candidate_media,
                source_metadata={"split": split},
            )
            if row:
                reservoir.add(row)
        yield task_id, reservoir


def _conversation_pair(item: Mapping[str, object]) -> tuple[str, str] | None:
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None
    first, second = conversations[0], conversations[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    query = clean_input_text(first.get("value"))
    answer = clean_input_text(second.get("value"))
    return (query, answer) if query and answer else None


def build_llavahound(
    config: dict,
    selection: set[str],
    limit: int,
    seed: int,
    debug_limit: int | None,
) -> Iterator[tuple[str, Reservoir]]:
    root = Path(config["root"])
    for task_config in config["tasks"]:
        task = task_config["name"]
        task_id = f"llavahound__{task}"
        if not selected(task_id, selection):
            continue
        metadata_path = root / task_config["metadata"]
        frame_root = root / task_config["frame_root"]
        mode = task_config["mode"]
        frame_count = int(task_config.get("num_frames", 8))
        reservoir = make_reservoir(task_id, limit, seed)
        with metadata_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if debug_limit is not None and index >= debug_limit:
                    break
                item = json.loads(line)
                pair = _conversation_pair(item)
                video_id = item.get("video")
                if pair is None or not isinstance(video_id, str):
                    continue
                frame_dir = frame_root / video_id
                if not frame_dir_has_media(frame_dir):
                    continue
                instruction, caption_or_answer = pair
                if mode in {"caption_retrieval", "video_qa"}:
                    query_text = (
                        f"Answer a question based on the content of a video. {instruction}"
                        if mode == "video_qa"
                        else instruction
                    )
                    query_media_spec = {"frame_dir": str(frame_dir), "count": frame_count}
                    candidate_text = caption_or_answer
                    candidate_media_spec = None
                elif mode == "video_retrieval":
                    query_text = (
                        "Find a video that contains the following visual content: "
                        + caption_or_answer
                    )
                    query_media_spec = None
                    candidate_text = "Understand the content of the provided video."
                    candidate_media_spec = {"frame_dir": str(frame_dir), "count": frame_count}
                else:
                    raise ValueError(f"{task_id}: unsupported mode {mode!r}")
                row = make_row(
                    dataset="llavahound",
                    task=task,
                    source_index=index,
                    query_text=query_text,
                    query_media=[],
                    candidate_text=candidate_text,
                    candidate_media=[],
                    source_metadata={
                        "mode": mode,
                        "video_id": video_id,
                        "record_id": item.get("id"),
                        "query_video": query_media_spec,
                        "candidate_video": candidate_media_spec,
                    },
                )
                if row:
                    # Media lists are filled only for selected rows to avoid retaining
                    # 8 paths for every one of the 300k source records.
                    if query_media_spec:
                        row["query"]["_deferred_video"] = query_media_spec
                        row["query"]["cot_enabled"] = True
                        row["query"]["cot_gate_reason"] = "media_present"
                    if candidate_media_spec:
                        row["candidate"]["_deferred_video"] = candidate_media_spec
                        row["candidate"]["cot_enabled"] = True
                        row["candidate"]["cot_gate_reason"] = "media_present"
                    reservoir.add(row)

        for row in reservoir.items:
            for side in ("query", "candidate"):
                video_spec = row[side].pop("_deferred_video", None)
                if not video_spec:
                    continue
                frames = uniform_frames(Path(video_spec["frame_dir"]), video_spec["count"])
                row[side]["media"] = [
                    media_path(path, temporal_index=index)
                    for index, path in enumerate(frames)
                ]
        yield task_id, reservoir


def build_grouped_dataset(
    *,
    dataset_name: str,
    dataset,
    expected_sources: list[str],
    selection: set[str],
    limit: int,
    seed: int,
    debug_limit: int | None,
    row_factory: Callable[[int, Mapping[str, object]], dict | None],
) -> Iterator[tuple[str, Reservoir]]:
    reservoirs = {
        source: make_reservoir(f"{dataset_name}__{source}", limit, seed)
        for source in expected_sources
        if selected(f"{dataset_name}__{source}", selection)
    }
    columns = [name for name in dataset.column_names if name != "image"]
    metadata_dataset = dataset.select_columns(columns)
    source_rows = limited_count(len(metadata_dataset), debug_limit)
    for index in range(source_rows):
        source_item = metadata_dataset[index]
        item = {name: source_item[name] for name in columns}
        source = item.get("source")
        if source not in reservoirs:
            continue
        row = row_factory(index, item)
        if row:
            reservoirs[source].add(row)
    for source in expected_sources:
        if source in reservoirs:
            yield f"{dataset_name}__{source}", reservoirs[source]


def build_colpali(
    config: dict,
    selection: set[str],
    limit: int,
    seed: int,
    debug_limit: int | None,
) -> Iterator[tuple[str, Reservoir]]:
    if selection and not any(task.startswith("colpali__") for task in selection):
        return
    from datasets import load_from_disk

    dataset = load_from_disk(config["path"])

    def factory(index: int, item: Mapping[str, object]) -> dict | None:
        source = str(item["source"])
        return make_row(
            dataset="colpali",
            task=source,
            source_index=index,
            query_text=item.get("query"),
            query_media=[embedded_media("colpali", index)],
            candidate_text=item.get("answer"),
            candidate_media=[],
            source_metadata={"source": source, "answer_type": item.get("answer_type")},
        )

    yield from build_grouped_dataset(
        dataset_name="colpali",
        dataset=dataset,
        expected_sources=config["expected_sources"],
        selection=selection,
        limit=limit,
        seed=seed,
        debug_limit=debug_limit,
        row_factory=factory,
    )


def canonical_visrag_files(path: Path) -> list[Path]:
    pattern = re.compile(r"train-\d{5}-of-\d{5}\.parquet")
    files = sorted(item for item in path.glob("*.parquet") if pattern.fullmatch(item.name))
    if not files:
        raise FileNotFoundError(f"No canonical VisRAG shards found in {path}")
    return files


def build_visrag(
    config: dict,
    selection: set[str],
    limit: int,
    seed: int,
    debug_limit: int | None,
) -> Iterator[tuple[str, Reservoir]]:
    if selection and not any(task.startswith("visrag__") for task in selection):
        return
    import pyarrow.parquet as pq

    files = canonical_visrag_files(Path(config["path"]))
    expected_sources = config["expected_sources"]
    reservoirs = {
        source: make_reservoir(f"visrag__{source}", limit, seed)
        for source in expected_sources
        if selected(f"visrag__{source}", selection)
    }
    global_index = 0
    stop = False
    for path in files:
        parquet_file = pq.ParquetFile(path)
        file_index = 0
        for batch in parquet_file.iter_batches(
            batch_size=2048, columns=["query", "source"]
        ):
            values = batch.to_pydict()
            for query, source in zip(values["query"], values["source"]):
                if debug_limit is not None and global_index >= debug_limit:
                    stop = True
                    break
                if source in reservoirs:
                    row = make_row(
                        dataset="visrag",
                        task=str(source),
                        source_index=global_index,
                        query_text=query,
                        query_media=[],
                        candidate_text=(
                            f"A visual document candidate from the {source} collection."
                        ),
                        candidate_media=[parquet_media(path, file_index)],
                        source_metadata={
                            "source": source,
                            "parquet_file": path.name,
                            "file_row_index": file_index,
                        },
                    )
                    if row:
                        reservoirs[source].add(row)
                global_index += 1
                file_index += 1
            if stop:
                break
        if stop:
            break
    for source in expected_sources:
        if source in reservoirs:
            yield f"visrag__{source}", reservoirs[source]


def write_task(
    manifest_dir: Path,
    task_id: str,
    reservoir: Reservoir,
) -> dict:
    if reservoir.seen == 0:
        raise RuntimeError(f"{task_id}: no valid source rows were found")
    rows = sorted(reservoir.items, key=lambda item: item["source_index"])
    filename = safe_task_filename(task_id)
    output_path = manifest_dir / filename
    gate_counts = Counter()
    for rank, row in enumerate(rows):
        row["task_sample_rank"] = rank
        gate_counts[f"query_{'enabled' if row['query']['cot_enabled'] else 'disabled'}"] += 1
        gate_counts[
            f"candidate_{'enabled' if row['candidate']['cot_enabled'] else 'disabled'}"
        ] += 1
    written = atomic_jsonl(output_path, rows)
    return {
        "task_id": task_id,
        "path": filename,
        "valid_source_rows": reservoir.seen,
        "sampled_rows": written,
        "gate_counts": dict(sorted(gate_counts.items())),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    all_task_ids = configured_task_ids(config)
    if args.list_tasks:
        print("\n".join(all_task_ids))
        return 0

    selection = requested_tasks(args.task)
    unknown = sorted(selection - set(all_task_ids))
    if unknown:
        raise SystemExit("Unknown task(s): " + ", ".join(unknown))
    generation = config["generation"]
    limit = int(args.max_per_task or generation.get("max_per_task", 50000))
    seed = int(args.seed if args.seed is not None else generation.get("seed", 42))
    if limit <= 0:
        raise ValueError("max_per_task must be positive")
    output_root = Path(args.output_dir or generation["output_dir"])
    manifest_dir = output_root / "manifests"
    index_path = manifest_dir / "manifest_index.json"
    if index_path.exists() and not args.force:
        raise SystemExit(f"Manifest already exists: {index_path}; pass --force to replace it.")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    builders = (
        build_mmeb(config["datasets"]["mmeb"], selection, limit, seed, args.max_source_rows),
        build_llavahound(
            config["datasets"]["llavahound"], selection, limit, seed, args.max_source_rows
        ),
        build_colpali(
            config["datasets"]["colpali"], selection, limit, seed, args.max_source_rows
        ),
        build_visrag(
            config["datasets"]["visrag"], selection, limit, seed, args.max_source_rows
        ),
    )
    summaries: list[dict] = []
    for builder in builders:
        for task_id, reservoir in builder:
            summary = write_task(manifest_dir, task_id, reservoir)
            summaries.append(summary)
            print(
                f"[{task_id}] valid={summary['valid_source_rows']:,} "
                f"sampled={summary['sampled_rows']:,}"
            )

    built_ids = {item["task_id"] for item in summaries}
    expected_ids = set(selection or all_task_ids)
    missing = sorted(expected_ids - built_ids)
    if missing:
        raise RuntimeError("Configured tasks were not built: " + ", ".join(missing))
    index = {
        "schema_version": COT_SCHEMA_VERSION,
        "config": str(args.config.resolve()),
        "seed": seed,
        "max_per_task": limit,
        "debug_max_source_rows": args.max_source_rows,
        "task_count": len(summaries),
        "sample_count": sum(item["sampled_rows"] for item in summaries),
        "tasks": summaries,
    }
    atomic_json(index_path, index)
    print(
        f"Manifest ready: {index['sample_count']:,} pairs across "
        f"{index['task_count']} tasks -> {manifest_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
