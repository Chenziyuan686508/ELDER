"""ELDER-specific model contracts and training utilities."""

from .stage1 import (
    QWEN2_VL_2B_CONTRACT,
    STAGE1_V2_DORA_TARGET_MODULES,
    STAGE1_V2_EXPECTED_TRAINABLE_PARAMETERS,
    audit_stage1_v2_trainable_parameters,
    compare_stage1_reports,
    compute_retrieval_metrics,
    validate_qwen2_vl_2b_config,
    validate_stage1_v2_dora_configuration,
)

__all__ = [
    "QWEN2_VL_2B_CONTRACT",
    "STAGE1_V2_DORA_TARGET_MODULES",
    "STAGE1_V2_EXPECTED_TRAINABLE_PARAMETERS",
    "audit_stage1_v2_trainable_parameters",
    "compare_stage1_reports",
    "compute_retrieval_metrics",
    "validate_qwen2_vl_2b_config",
    "validate_stage1_v2_dora_configuration",
]
