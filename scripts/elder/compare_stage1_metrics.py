#!/usr/bin/env python3
"""Compare Stage 1 before/after retrieval reports and enforce acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.elder.stage1 import compare_stage1_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-recall-multiplier", type=float, default=3.0)
    parser.add_argument("--norm-tolerance", type=float, default=0.05)
    return parser.parse_args()


def read_report(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    result = compare_stage1_reports(
        read_report(args.before),
        read_report(args.after),
        random_recall_multiplier=args.random_recall_multiplier,
        norm_tolerance=args.norm_tolerance,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
