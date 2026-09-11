#!/usr/bin/env python3
"""Generate validated four-stage ELDER CoT with a local GLM-4.1V model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import signal
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.elder.cot import (  # noqa: E402
    COT_SCHEMA_VERSION,
    PROMPT_VERSION,
    STAGE_KEYS,
    build_generation_prompt,
    cot_side_payload,
    metadata_payload,
    parse_cot_json,
    stable_int,
    validate_cot,
    GateDecision,
)


DEFAULT_CONFIG = REPO_ROOT / "experiments/elder/cot_generation_local.yaml"
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one resumable shard of local GLM-4.1V CoT generation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--attn-implementation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    def expand(value):
        if isinstance(value, str):
            result = os.path.expanduser(os.path.expandvars(value))
            if re.search(r"\$\{?[A-Za-z_]", result):
                raise ValueError(
                    f"Unresolved environment variable in {value!r}; source .env.elder first."
                )
            return result
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return expand(raw)


def task_selection(values: Iterable[str]) -> set[str]:
    result = set()
    for value in values:
        result.update(part.strip() for part in value.split(",") if part.strip())
    return result


def manifest_rows(manifest_dir: Path, tasks: set[str]) -> Iterator[dict]:
    index_path = manifest_dir / "manifest_index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    known = {item["task_id"] for item in index["tasks"]}
    unknown = sorted(tasks - known)
    if unknown:
        raise ValueError("Unknown manifest task(s): " + ", ".join(unknown))
    for task in index["tasks"]:
        if tasks and task["task_id"] not in tasks:
            continue
        path = manifest_dir / task["path"]
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                yield row


def completed_ids(path: Path) -> set[str]:
    result: set[str] = set()
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                result.add(item["sample_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Corrupt resume output at {path}:{line_number}") from exc
    return result


def canonical_visrag_files(path: Path) -> list[str]:
    pattern = re.compile(r"train-\d{5}-of-\d{5}\.parquet")
    files = sorted(item for item in path.glob("*.parquet") if pattern.fullmatch(item.name))
    if not files:
        raise FileNotFoundError(f"No canonical VisRAG shards found in {path}")
    return [str(item) for item in files]


class DatasetImageResolver:
    """Lazily open memory-mapped local datasets only when a sampled row needs one."""

    def __init__(self, config: Mapping[str, object]):
        self.config = config
        self.datasets: dict[str, object] = {}
        self.parquet_files: dict[str, object] = {}
        self.row_group_cache: OrderedDict[tuple[str, int, str], object] = OrderedDict()

    def _dataset(self, key: str):
        if key in self.datasets:
            return self.datasets[key]
        from datasets import load_from_disk

        dataset_config = self.config["datasets"]
        if key == "colpali":
            dataset = load_from_disk(dataset_config["colpali"]["path"])
        else:
            raise KeyError(f"Unknown dataset image reference: {key}")
        self.datasets[key] = dataset
        return dataset

    @staticmethod
    def _to_image(value: object) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes"):
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            if value.get("path"):
                return Image.open(value["path"]).convert("RGB")
        raise TypeError(f"Unsupported embedded image value: {type(value).__name__}")

    def _resolve_parquet(self, reference: Mapping[str, object]) -> Image.Image:
        import pyarrow.parquet as pq

        path = str(reference["path"])
        column = str(reference.get("column", "image"))
        row_index = int(reference["row_index"])
        parquet_file = self.parquet_files.get(path)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(path)
            self.parquet_files[path] = parquet_file
        offset = 0
        selected_group = None
        for group_index in range(parquet_file.num_row_groups):
            group_rows = parquet_file.metadata.row_group(group_index).num_rows
            if row_index < offset + group_rows:
                selected_group = group_index
                break
            offset += group_rows
        if selected_group is None:
            raise IndexError(f"Parquet row {row_index} is out of range for {path}")
        cache_key = (path, selected_group, column)
        table = self.row_group_cache.get(cache_key)
        if table is None:
            table = parquet_file.read_row_group(selected_group, columns=[column])
            self.row_group_cache[cache_key] = table
            self.row_group_cache.move_to_end(cache_key)
            while len(self.row_group_cache) > 4:
                self.row_group_cache.popitem(last=False)
        else:
            self.row_group_cache.move_to_end(cache_key)
        value = table.column(column)[row_index - offset].as_py()
        return self._to_image(value)

    def resolve(self, reference: Mapping[str, object]) -> Image.Image:
        if reference.get("type") == "parquet_image":
            return self._resolve_parquet(reference)
        dataset = self._dataset(str(reference["dataset_key"]))
        value = dataset[int(reference["row_index"])][str(reference.get("column", "image"))]
        return self._to_image(value)


class MediaMaterializer:
    def __init__(self, resolver: DatasetImageResolver):
        self.resolver = resolver
        self.temp_dir = tempfile.TemporaryDirectory(prefix="elder-cot-media-")

    def close(self) -> None:
        self.temp_dir.cleanup()

    def paths(self, references: Iterable[Mapping[str, object]], side: str) -> list[str]:
        result: list[str] = []
        for index, reference in enumerate(references):
            reference_type = reference.get("type")
            if reference_type == "image_path":
                path = Path(str(reference["path"]))
                if not path.is_file():
                    raise FileNotFoundError(path)
                result.append(str(path))
            elif reference_type in {"dataset_image", "parquet_image"}:
                image = self.resolver.resolve(reference)
                path = Path(self.temp_dir.name) / f"{side}_{index}.png"
                image.save(path, format="PNG")
                result.append(str(path))
            else:
                raise ValueError(f"Unsupported media reference type: {reference_type!r}")
        return result


class GlmGenerator:
    def __init__(
        self,
        *,
        model_path: Path,
        attn_implementation: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ):
        try:
            import torch
            import transformers
            from packaging.version import Version
            from transformers import AutoProcessor, Glm4vForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "GLM-4.1V requires the separate elder-cot environment. Run "
                "scripts/elder/create_cot_env.sh, then conda activate elder-cot."
            ) from exc
        if Version(transformers.__version__) < Version("4.57.1"):
            raise RuntimeError(
                f"transformers {transformers.__version__} is too old for GLM-4.1V; "
                "need >=4.57.1. Do not upgrade the Stage 1 elder environment in-place."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for GLM-4.1V generation.")
        self.torch = torch
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def token_count(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        *,
        prompt: str,
        image_paths: list[str],
        sample_key: str,
        attempt: int,
    ) -> str:
        content = [{"type": "image", "url": path} for path in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        local_seed = stable_int(f"{sample_key}:{attempt}", self.seed) % (2**31)
        self.torch.manual_seed(local_seed)
        self.torch.cuda.manual_seed_all(local_seed)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def process_side(
    *,
    generator: GlmGenerator,
    materializer: MediaMaterializer,
    row: Mapping[str, object],
    side: str,
    max_retries: int,
) -> tuple[dict, int, dict]:
    side_input = row[side]
    gate = GateDecision(
        bool(side_input["cot_enabled"]), str(side_input["cot_gate_reason"])
    )
    if not gate.enabled:
        return (
            cot_side_payload(
                text=side_input["text"], media=side_input["media"], gate=gate, stages=None
            ),
            0,
            {"valid": True, "skipped": True, "reason": gate.reason},
        )
    image_paths = materializer.paths(side_input["media"], side)
    prior_errors: list[str] = []
    last_hash = None
    for attempt in range(max_retries + 1):
        prompt = build_generation_prompt(
            side=side,
            text=str(side_input["text"]),
            media_count=len(image_paths),
            attempt=attempt,
            prior_errors=prior_errors,
        )
        raw = generator.generate(
            prompt=prompt,
            image_paths=image_paths,
            sample_key=f"{row['sample_id']}:{side}",
            attempt=attempt,
        )
        last_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        try:
            stages = parse_cot_json(raw)
        except ValueError as exc:
            prior_errors = [str(exc)]
            continue
        validation = validate_cot(
            stages,
            side=side,
            forbidden_terms=(str(row["sample_id"]),),
            token_counter=generator.token_count,
        )
        if validation.valid:
            return (
                cot_side_payload(
                    text=side_input["text"],
                    media=side_input["media"],
                    gate=gate,
                    stages=stages,
                ),
                attempt,
                validation.as_dict(),
            )
        prior_errors = validation.errors
    raise RuntimeError(
        json.dumps(
            {
                "side": side,
                "errors": prior_errors,
                "attempts": max_retries + 1,
                "last_generation_sha256": last_hash,
            },
            ensure_ascii=False,
        )
    )


def process_row(
    generator: GlmGenerator,
    materializer: MediaMaterializer,
    row: Mapping[str, object],
    model_path: Path,
    max_retries: int,
) -> dict:
    query, query_retries, query_validation = process_side(
        generator=generator,
        materializer=materializer,
        row=row,
        side="query",
        max_retries=max_retries,
    )
    candidate, candidate_retries, candidate_validation = process_side(
        generator=generator,
        materializer=materializer,
        row=row,
        side="candidate",
        max_retries=max_retries,
    )
    metadata = metadata_payload(
        model_name=str(model_path),
        retries={"query": query_retries, "candidate": candidate_retries},
    )
    metadata.update(
        {
            "candidate_disable_reason": (
                None if candidate["cot_enabled"] else candidate["cot_gate_reason"]
            ),
            "validation": {
                "query": query_validation,
                "candidate": candidate_validation,
            },
        }
    )
    return {
        "schema_version": COT_SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "task": row["task"],
        "task_id": row["task_id"],
        "source_index": row["source_index"],
        "query": query,
        "candidate": candidate,
        "source_metadata": row.get("source_metadata", {}),
        "metadata": metadata,
    }


def append_jsonl(handle, value: object, *, sync: bool = False) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    if sync:
        os.fsync(handle.fileno())


def handle_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; stopping after the current sample.", flush=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    generation = config["generation"]
    output_root = Path(args.output_dir or generation["output_dir"])
    manifest_dir = Path(args.manifest_dir or output_root / "manifests")
    model_path = Path(args.model_path or generation["model_path"])
    num_shards = int(args.num_shards or generation.get("num_shards", 4))
    shard_index = args.shard_index
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Require 0 <= shard_index < num_shards, got {shard_index}/{num_shards}")
    max_retries = int(
        args.max_retries if args.max_retries is not None else generation.get("max_retries", 3)
    )
    max_new_tokens = int(args.max_new_tokens or generation.get("max_new_tokens", 1536))
    temperature = float(
        args.temperature if args.temperature is not None else generation.get("temperature", 0.2)
    )
    top_p = float(args.top_p if args.top_p is not None else generation.get("top_p", 0.8))
    seed = int(args.seed if args.seed is not None else generation.get("seed", 42))
    attn_implementation = args.attn_implementation or generation.get(
        "attn_implementation", "sdpa"
    )
    tasks = task_selection(args.task)
    generated_dir = output_root / "generated"
    suffix = f"{shard_index:05d}-of-{num_shards:05d}"
    output_path = generated_dir / f"cot-shard-{suffix}.jsonl"
    failure_path = generated_dir / f"failures-shard-{suffix}.jsonl"
    done = completed_ids(output_path) if args.resume else set()
    if not args.resume and output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --resume or remove it.")

    print(
        json.dumps(
            {
                "event": "startup",
                "prompt_version": PROMPT_VERSION,
                "model": str(model_path),
                "manifest_dir": str(manifest_dir),
                "output": str(output_path),
                "shard": [shard_index, num_shards],
                "resume_completed": len(done),
                "tasks": sorted(tasks) if tasks else "all",
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        assigned = 0
        for row in manifest_rows(manifest_dir, tasks):
            if stable_int(row["sample_id"]) % num_shards == shard_index:
                assigned += 1
                if args.max_samples is not None and assigned >= args.max_samples:
                    break
        print(f"Dry run: shard {shard_index} has {assigned:,} selected rows (capped if requested).")
        return 0

    pending = 0
    for row in manifest_rows(manifest_dir, tasks):
        if stable_int(row["sample_id"]) % num_shards != shard_index:
            continue
        if row["sample_id"] in done:
            continue
        pending += 1
        if args.max_samples is not None and pending >= args.max_samples:
            break
    if pending == 0:
        print(f"Shard {shard_index} has no pending samples; model loading skipped.", flush=True)
        return 0

    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid local GLM model directory: {model_path}")
    generated_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    generator = GlmGenerator(
        model_path=model_path,
        attn_implementation=attn_implementation,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    materializer = MediaMaterializer(DatasetImageResolver(config))
    processed = succeeded = failed = skipped = 0
    started = time.monotonic()
    try:
        with output_path.open("a", encoding="utf-8") as output_handle, failure_path.open(
            "a", encoding="utf-8"
        ) as failure_handle:
            for row in manifest_rows(manifest_dir, tasks):
                if STOP_REQUESTED:
                    break
                if stable_int(row["sample_id"]) % num_shards != shard_index:
                    continue
                if row["sample_id"] in done:
                    skipped += 1
                    continue
                if args.max_samples is not None and processed >= args.max_samples:
                    break
                processed += 1
                try:
                    result = process_row(
                        generator, materializer, row, model_path, max_retries
                    )
                    append_jsonl(output_handle, result, sync=succeeded % 10 == 0)
                    done.add(row["sample_id"])
                    succeeded += 1
                except Exception as exc:  # keep a week-long offline job alive
                    failed += 1
                    append_jsonl(
                        failure_handle,
                        {
                            "schema_version": COT_SCHEMA_VERSION,
                            "sample_id": row.get("sample_id"),
                            "task_id": row.get("task_id"),
                            "source_index": row.get("source_index"),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:4000],
                            "timestamp": time.time(),
                        },
                        sync=True,
                    )
                    if hasattr(generator, "torch") and "out of memory" in str(exc).lower():
                        generator.torch.cuda.empty_cache()
                elapsed = max(time.monotonic() - started, 1e-6)
                if processed == 1 or processed % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "shard": shard_index,
                                "processed": processed,
                                "succeeded": succeeded,
                                "failed": failed,
                                "resume_skipped": skipped,
                                "samples_per_hour": processed / elapsed * 3600,
                                "last_task": row.get("task_id"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        materializer.close()
    print(
        json.dumps(
            {
                "event": "complete" if not STOP_REQUESTED else "interrupted",
                "shard": shard_index,
                "processed": processed,
                "succeeded": succeeded,
                "failed": failed,
                "resume_skipped": skipped,
                "elapsed_seconds": time.monotonic() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 130 if STOP_REQUESTED else (1 if failed and not succeeded else 0)


if __name__ == "__main__":
    raise SystemExit(main())
