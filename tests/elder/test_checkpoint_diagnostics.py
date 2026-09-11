import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "elder"
    / "diagnose_stage1_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("checkpoint_diagnostics", SCRIPT_PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = diagnostics
SPEC.loader.exec_module(diagnostics)


def test_embedding_statistics_detects_near_collapse():
    rng = np.random.default_rng(7)
    anchor = rng.normal(size=(1, 16)).astype(np.float32)
    collapsed = anchor + 1e-3 * rng.normal(size=(8, 16)).astype(np.float32)
    stats = diagnostics.embedding_statistics(collapsed)
    assert stats["off_diagonal_cosine"]["mean"] > 0.99


def test_rank_candidates_computes_local_classification_metrics():
    queries = np.eye(3, dtype=np.float32)
    candidates = np.eye(3, dtype=np.float32)
    candidate_ids = ["a", "b", "c"]
    ground_truth = [
        {"cand_names": candidate_ids, "label_name": name}
        for name in candidate_ids
    ]
    metrics, predictions = diagnostics.rank_candidates(
        queries, candidates, candidate_ids, ground_truth
    )
    assert metrics["hit@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert predictions == candidate_ids


def test_merge_comparison_reports_equivalence():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    comparison = diagnostics.compare_embedding_sets(
        embeddings,
        embeddings.copy(),
        ["a", "b"],
        ["a", "b"],
    )
    assert comparison["row_cosine"]["mean"] == 1.0
    assert comparison["max_absolute_difference"] == 0.0
    assert comparison["top1_agreement"] == 1.0
    assert diagnostics.merge_verdict(
        comparison, diagnostics.DiagnosticThresholds()
    ) == "merge_equivalent"


def test_component_conclusion_prioritizes_merge_mismatch():
    report = {
        "modes": {
            "full_unmerged": {
                "status": "success",
                "query_embedding_statistics": {
                    "off_diagonal_cosine": {"mean": 0.4}
                },
            },
            "full_merged": {
                "status": "success",
                "query_embedding_statistics": {
                    "off_diagonal_cosine": {"mean": 0.97}
                },
            },
        },
        "comparisons": {
            "full_unmerged_vs_full_merged": {
                "row_cosine": {"mean": 0.5},
                "top1_agreement": 0.1,
            }
        },
    }
    conclusion = diagnostics.build_conclusion(
        report, diagnostics.DiagnosticThresholds()
    )
    assert conclusion["merge_verdict"] == "merge_mismatch"
    assert conclusion["likely_failure_domain"] == "hybrid_dora_merge_or_load"

