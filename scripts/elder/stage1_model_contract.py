#!/usr/bin/env python3
"""Validate that an ELDER run uses the fixed Qwen2-VL-2B backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoConfig

from src.elder.stage1 import validate_qwen2_vl_2b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/chenziyuan/models/Qwen2-VL-2B-Instruct"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing local model directory: {model_path}")
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    contract = validate_qwen2_vl_2b_config(config)
    print(
        json.dumps(
            {
                "status": "ok",
                "model_path": str(model_path),
                "contract": contract,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
