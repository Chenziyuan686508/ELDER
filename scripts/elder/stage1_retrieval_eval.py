#!/usr/bin/env python3
"""Evaluate paired Stage 1 retrieval on a fixed local ViDoRe subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq
import torch
from transformers import AutoConfig

from src.arguments import DataArguments, ModelArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.dataset.vidore_dataset import data_prepare_v5
from src.elder.stage1 import (
    compute_retrieval_metrics,
    validate_qwen2_vl_2b_config,
)
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.utils.basic_utils import batch_to_device


DEFAULT_MODEL = Path(
    os.environ.get(
        "ELDER_MODEL_PATH",
        "/data/chenziyuan/models/Qwen2-VL-2B-Instruct",
    )
)
DEFAULT_DATASET = Path(
    os.environ.get(
        "ELDER_STAGE1_EVAL_DATASET",
        "/data/chenziyuan/datasets/colpali_train_set/data/"
        "train-00000-of-00082.parquet",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resize-max-pixels", type=int, default=200704)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_examples(
    dataset_file: Path,
    *,
    offset: int,
    num_samples: int,
    model_backbone: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Missing local dataset file: {dataset_file}")
    parquet_file = pq.ParquetFile(dataset_file)
    if offset < 0 or num_samples <= 0:
        raise ValueError("offset must be >= 0 and num_samples must be > 0")
    if offset + num_samples > parquet_file.metadata.num_rows:
        raise ValueError(
            f"Requested rows [{offset}, {offset + num_samples}) from a "
            f"{parquet_file.metadata.num_rows}-row shard."
        )

    raw_table = pq.read_table(dataset_file).slice(offset, num_samples)
    raw_batch = raw_table.to_pydict()
    prepared = data_prepare_v5(
        raw_batch,
        model_backbone=model_backbone,
        image_resolution="low",
        global_dataset_name="vidore/colpali_train_stage1_acceptance",
    )
    examples = [
        {key: value[row] for key, value in prepared.items()}
        for row in range(num_samples)
    ]
    candidate_texts = prepared["pos_text"]
    return examples, {
        "dataset_file": str(dataset_file),
        "dataset_rows": parquet_file.metadata.num_rows,
        "offset": offset,
        "num_samples": num_samples,
        "unique_candidate_texts": len(set(candidate_texts)),
        "duplicate_candidate_rows": len(candidate_texts)
        - len(set(candidate_texts)),
    }


def encode_pairs(
    model: MMEBModel,
    collator: MultimodalDataCollator,
    examples: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_embeddings = []
    candidate_embeddings = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            query_inputs, candidate_inputs = collator(
                examples[start : start + batch_size]
            )
            query_embeddings.append(
                model.encode_input(batch_to_device(query_inputs, device))
                .float()
                .cpu()
            )
            candidate_embeddings.append(
                model.encode_input(batch_to_device(candidate_inputs, device))
                .float()
                .cpu()
            )
    return torch.cat(query_embeddings), torch.cat(candidate_embeddings)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 retrieval evaluation requires a visible GPU.")

    base_model = args.base_model.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve() if args.checkpoint else None
    )
    dataset_file = args.dataset_file.expanduser().resolve()
    if checkpoint is not None and not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint}")

    config = AutoConfig.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=True,
    )
    contract = validate_qwen2_vl_2b_config(config)
    model_backbone = get_backbone_name(config)
    model_args = ModelArguments(
        model_name=str(base_model),
        checkpoint_path=str(checkpoint) if checkpoint else None,
        pooling="last",
        normalize=True,
        temperature=args.temperature,
        lora=checkpoint is not None,
    )
    model_args.model_backbone = model_backbone
    data_args = DataArguments(
        max_len=None,
        image_resolution="low",
        resize_use_processor=True,
        resize_min_pixels=28 * 28 * 4,
        resize_max_pixels=args.resize_max_pixels,
    )
    training_args = SimpleNamespace(model_backbone=model_backbone)

    examples, dataset_report = load_examples(
        dataset_file,
        offset=args.offset,
        num_samples=args.num_samples,
        model_backbone=model_backbone,
    )
    processor = load_processor(model_args, data_args)
    collator = MultimodalDataCollator(
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.monotonic()
    model = MMEBModel.load(model_args, is_trainable=False)
    model.to(device)
    query_embeddings, candidate_embeddings = encode_pairs(
        model,
        collator,
        examples,
        batch_size=args.batch_size,
        device=device,
    )
    elapsed_seconds = time.monotonic() - started_at

    report = {
        "status": "ok",
        "mode": "checkpoint" if checkpoint else "base_model",
        "base_model": str(base_model),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model_contract": contract,
        "dataset": dataset_report,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "metrics": compute_retrieval_metrics(
            query_embeddings,
            candidate_embeddings,
            temperature=args.temperature,
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
