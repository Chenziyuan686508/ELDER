import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "elder"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dora = load_module("dora_ablation", "diagnose_dora_ablation.py")
audit = load_module("batch_audit", "audit_stage1_batch_integrity.py")


def test_dora_profile_parser_and_matching():
    descriptor = dora.describe_adapter_module(
        "base_model.model.layers.17.self_attn.q_proj"
    )
    assert descriptor == {
        "name": "base_model.model.layers.17.self_attn.q_proj",
        "layer": 17,
        "module_type": "q_proj",
        "family": "attention",
    }
    assert dora.profile_matches("attention", descriptor)
    assert dora.profile_matches("middle", descriptor)
    assert not dora.profile_matches("mlp", descriptor)
    assert not dora.profile_matches("lower", descriptor)


def test_dora_margin_statistics_identifies_correct_positive():
    queries = np.eye(2, dtype=np.float32)
    candidates = np.eye(2, dtype=np.float32)
    stats = dora.positive_margin_statistics(
        queries,
        candidates,
        ["a", "b"],
        [
            {"label_name": "a"},
            {"label_name": "b"},
        ],
    )
    assert stats["positive_beats_hardest_negative_rate"] == 1.0
    assert stats["positive_minus_hardest_negative_mean"] == 1.0


def test_top1_distribution_reports_concentration():
    stats = dora.top1_distribution(["a", "a", "a", "b"])
    assert stats["unique_fraction"] == 0.5
    assert stats["largest_fraction"] == 0.75
    assert stats["top_candidates"][0]["candidate"] == "a"


def test_batch_audit_pure_helpers():
    assert audit.expected_global_labels(3, 2) == [0, 1, 2, 3, 4, 5]
    visual = {"paths": ["x.jpg"], "bytes": [None]}
    text = {"paths": [""], "bytes": [None]}
    assert audit.direction(
        {"query_image": visual, "pos_image": text}
    ) == "visual_to_text"
    assert audit.direction(
        {"query_image": text, "pos_image": visual}
    ) == "text_to_visual"

