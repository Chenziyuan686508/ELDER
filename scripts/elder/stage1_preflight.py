#!/usr/bin/env python3
"""Validate the local ELDER Stage 1 baseline without downloading anything."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import pyarrow.parquet as pq
from safetensors import safe_open
from transformers import AutoConfig

from src.arguments import DataArguments, ModelArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.dataset.vidore_dataset import data_prepare_v5
from src.loss import SimpleContrastiveLoss
from src.model.processor import (
    VLM_IMAGE_TOKENS,
    get_backbone_name,
    load_processor,
)


DEFAULT_MODEL = Path("/data/chenziyuan/models/Qwen2-VL-2B-Instruct")
DEFAULT_DATASET = Path(
    "/data/chenziyuan/datasets/colpali_train_set/data/"
    "train-00000-of-00082.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Only validate model artifacts and processor paths.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final report as compact JSON.",
    )
    return parser.parse_args()


def validate_weight_index(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing weight index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    missing = [name for name in shard_names if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing weight shards: {missing}")

    tensor_count = 0
    shard_bytes = 0
    for shard_name in shard_names:
        shard_path = model_path / shard_name
        shard_bytes += shard_path.stat().st_size
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            tensor_count += len(handle.keys())

    incomplete = list(model_path.rglob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"Incomplete downloads remain: {incomplete}")

    return {
        "weight_shards": shard_names,
        "weight_file_bytes": shard_bytes,
        "indexed_tensor_bytes": index.get("metadata", {}).get("total_size"),
        "tensor_count": tensor_count,
    }


def build_runtime_args(
    model_path: Path,
    model_backbone: str,
) -> tuple[ModelArguments, DataArguments, SimpleNamespace]:
    model_args = ModelArguments(
        model_name=str(model_path),
        pooling="last",
        normalize=True,
        temperature=0.02,
    )
    model_args.model_backbone = model_backbone

    data_args = DataArguments(
        max_len=None,
        resize_use_processor=True,
        resize_min_pixels=28 * 28 * 4,
        resize_max_pixels=28 * 28 * 256,
        image_resolution="low",
    )
    training_args = SimpleNamespace(model_backbone=model_backbone)
    return model_args, data_args, training_args


def tensor_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            shapes[key] = list(value.shape)
        elif isinstance(value, list):
            shapes[key] = f"list[{len(value)}]"
    return shapes


def validate_processor(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: SimpleNamespace,
    dataset_file: Path,
    skip_dataset: bool,
) -> dict[str, Any]:
    processor = load_processor(model_args, data_args)
    collator = MultimodalDataCollator(
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    empty_visual = {
        "paths": [""],
        "bytes": [b""],
        "resolutions": [[28, 28]],
    }
    text_examples = [
        {
            "query_text": "Represent this query for multimodal retrieval.",
            "query_image": empty_visual,
            "pos_text": "A matching retrieval candidate.",
            "pos_image": empty_visual,
            "neg_text": [],
            "neg_image": [],
            "global_dataset_name": "elder/preflight-text",
        },
        {
            "query_text": "Find the relevant visual document.",
            "query_image": empty_visual,
            "pos_text": "Relevant document.",
            "pos_image": empty_visual,
            "neg_text": [],
            "neg_image": [],
            "global_dataset_name": "elder/preflight-text",
        },
    ]
    text_qry, text_pos = collator(text_examples)
    result: dict[str, Any] = {
        "tokenizer_class": type(processor.tokenizer).__name__,
        "processor_class": type(processor).__name__,
        "text_query_shapes": tensor_shapes(text_qry),
        "text_target_shapes": tensor_shapes(text_pos),
    }

    if skip_dataset:
        return result
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Missing local dataset shard: {dataset_file}")

    parquet_file = pq.ParquetFile(dataset_file)
    if parquet_file.metadata.num_rows < 2:
        raise RuntimeError("The debug dataset must contain at least two rows.")

    raw_table = parquet_file.read_row_group(0).slice(0, 2)
    raw_batch = raw_table.to_pydict()
    prepared = data_prepare_v5(
        raw_batch,
        model_backbone=model_args.model_backbone,
        image_resolution=data_args.image_resolution,
        global_dataset_name="vidore/colpali_train_debug",
    )
    examples = [
        {key: value[row] for key, value in prepared.items()}
        for row in range(2)
    ]
    image_qry, image_pos = collator(examples)

    image_token = VLM_IMAGE_TOKENS[model_args.model_backbone]
    if not all(image_token in example["query_text"] for example in examples):
        raise AssertionError("Visual-document queries are missing the image token.")

    result.update(
        {
            "dataset_rows_in_shard": parquet_file.metadata.num_rows,
            "dataset_columns": raw_table.column_names,
            "image_query_shapes": tensor_shapes(image_qry),
            "image_target_shapes": tensor_shapes(image_pos),
            "debug_sources": raw_batch["source"],
        }
    )
    return result


def validate_contrastive_backward() -> dict[str, float]:
    query = (torch.eye(4) * 0.1).requires_grad_()
    positive = torch.eye(4)
    loss = SimpleContrastiveLoss(temperature=0.02)(query, positive)
    loss.backward()

    grad_norm = query.grad.norm().item()
    if not math.isfinite(loss.item()) or not math.isfinite(grad_norm):
        raise RuntimeError("Contrastive loss produced a non-finite value.")
    if grad_norm <= 0:
        raise RuntimeError("Contrastive loss did not backpropagate.")
    return {"loss": loss.item(), "query_grad_norm": grad_norm}


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    dataset_file = args.dataset_file.expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing local model directory: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_backbone = get_backbone_name(config)
    model_args, data_args, training_args = build_runtime_args(
        model_path,
        model_backbone,
    )

    report = {
        "status": "ok",
        "model_path": str(model_path),
        "dataset_file": None if args.skip_dataset else str(dataset_file),
        "model_type": config.model_type,
        "model_backbone": model_backbone,
        "hidden_size": config.hidden_size,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "weights": validate_weight_index(model_path),
        "processor": validate_processor(
            model_args,
            data_args,
            training_args,
            dataset_file,
            args.skip_dataset,
        ),
        "contrastive_backward": validate_contrastive_backward(),
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
