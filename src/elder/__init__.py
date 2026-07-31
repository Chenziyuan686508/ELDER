"""ELDER-specific model contracts and training utilities."""

from .stage1 import (
    QWEN2_VL_2B_CONTRACT,
    compare_stage1_reports,
    compute_retrieval_metrics,
    validate_qwen2_vl_2b_config,
)

__all__ = [
    "QWEN2_VL_2B_CONTRACT",
    "compare_stage1_reports",
    "compute_retrieval_metrics",
    "validate_qwen2_vl_2b_config",
]
