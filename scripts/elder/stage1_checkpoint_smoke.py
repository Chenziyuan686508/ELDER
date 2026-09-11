#!/usr/bin/env python3
"""Load a Stage 1 checkpoint twice and verify deterministic embeddings."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoConfig

from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import (
    get_backbone_name,
    load_processor,
    process_vlm_inputs_fns,
)
from src.utils.basic_utils import batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/data/chenziyuan/models/Qwen2-VL-2B-Instruct"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data/chenziyuan/checkpoints/elder/stage1_smoke_v2"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def make_model_args(
    base_model: Path,
    checkpoint: Path,
    model_backbone: str,
) -> ModelArguments:
    model_args = ModelArguments(
        model_name=str(base_model),
        checkpoint_path=str(checkpoint),
        pooling="last",
        normalize=True,
        temperature=0.02,
        lora=True,
    )
    model_args.model_backbone = model_backbone
    return model_args


def encode_once(
    model_args: ModelArguments,
    processed_inputs: dict,
    device: torch.device,
) -> torch.Tensor:
    model = MMEBModel.load(model_args, is_trainable=False)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        embedding = model.encode_input(
            batch_to_device(processed_inputs, device)
        ).float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embedding


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint smoke test requires a visible CUDA GPU.")

    base_model = args.base_model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = torch.device(args.device)

    config = AutoConfig.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=True,
    )
    model_backbone = get_backbone_name(config)
    model_args = make_model_args(base_model, checkpoint, model_backbone)
    data_args = DataArguments(max_len=128)
    processor = load_processor(model_args, data_args)
    process_fn = process_vlm_inputs_fns[model_backbone]
    processed_inputs = process_fn(
        {
            "text": [
                "Represent this query for multimodal retrieval.",
                "Find the matching visual document.",
            ],
            "images": [None, None],
        },
        processor=processor,
        max_length=128,
    )

    first = encode_once(model_args, processed_inputs, device)
    second = encode_once(model_args, processed_inputs, device)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    report = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "embedding_shape": list(first.shape),
        "embedding_norms": first.norm(dim=-1).tolist(),
        "max_reload_difference": (first - second).abs().max().item(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
