#!/usr/bin/env python3
"""Build Qwen2-VL Stage 1 DoRA and fail unless its trainable scope is exact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arguments import ModelArguments
from src.elder.stage1 import validate_stage1_v2_dora_configuration
from src.model.model import MMEBModel


ORIGINAL_TARGETS = (
    "qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; parent directory is created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing local model directory: {model_path}")

    model_args = ModelArguments(
        model_name=str(model_path),
        pooling="eos",
        normalize=True,
        temperature=0.02,
        lora=True,
        lora_r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        lora_target_modules=ORIGINAL_TARGETS,
        lora_adapter_scope="full_model",
        strict_stage1_dora=True,
    )
    model = MMEBModel.build(model_args)
    report = validate_stage1_v2_dora_configuration(model_args, model)
    report["model_path"] = str(model_path)

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    parameters = report["parameters"]
    compact = {
        "status": report["status"],
        "model_path": str(model_path),
        "trainable_parameters": parameters["trainable_parameters"],
        "expected_trainable_parameters": parameters[
            "expected_trainable_parameters"
        ],
        "observed_module_types": parameters["observed_module_types"],
        "visual_tower_frozen": parameters["checks"]["visual_tower_frozen"],
        "gate_and_up_proj_frozen": parameters["checks"][
            "gate_and_up_proj_frozen"
        ],
        "output": str(args.output.expanduser().resolve()) if args.output else None,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
