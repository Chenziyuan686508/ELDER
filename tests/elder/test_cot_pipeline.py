import json

from src.elder.cot import (
    GateDecision,
    STAGE_KEYS,
    build_generation_prompt,
    cot_side_payload,
    parse_cot_json,
    semantic_richness_gate,
    stable_int,
    validate_cot,
)


GOOD_STAGES = {
    "stage_1": (
        "The input depicts a red passenger car on a paved urban street with "
        "buildings and parked vehicles in the background."
    ),
    "stage_2": (
        "The discriminative evidence is the red car body and passenger-car shape; "
        "background buildings and unrelated parked vehicles should be ignored."
    ),
    "stage_3": (
        "The candidate must bind the red color attribute to the central car object; "
        "there is no additional temporal or numeric constraint."
    ),
    "stage_4": (
        "Retrieve a candidate centered on a red passenger car that preserves the "
        "required object and color constraints."
    ),
}


def test_semantic_gate_routes_short_candidate_but_keeps_query():
    assert semantic_richness_gate("12", [], side="candidate").reason == "short_number"
    assert not semantic_richness_gate("dog", [], side="candidate").enabled
    assert semantic_richness_gate("find a dog", [], side="query").enabled
    assert semantic_richness_gate("", [{"type": "image_path"}], side="candidate").enabled


def test_prompt_is_side_local_and_has_exact_schema_instruction():
    prompt = build_generation_prompt(
        side="query", text="find the red car", media_count=1
    )
    assert "find the red car" in prompt
    assert "stage_1" in prompt and "stage_4" in prompt
    assert "paired item" in prompt


def test_parse_discards_thinking_and_extracts_json():
    raw = "<think>private analysis</think>\n```json\n" + json.dumps(GOOD_STAGES) + "\n```"
    assert parse_cot_json(raw) == GOOD_STAGES


def test_quality_validator_accepts_retrieval_oriented_stages():
    result = validate_cot(GOOD_STAGES, side="query")
    assert result.valid, result.errors
    assert result.total_tokens <= 384


def test_quality_validator_rejects_duplicate_and_answer_phrase():
    stages = dict(GOOD_STAGES)
    stages["stage_2"] = stages["stage_1"]
    stages["stage_4"] = "The correct answer is red car, which is the final answer for this task."
    result = validate_cot(stages, side="query")
    assert not result.valid
    assert any(error.startswith("adjacent_stage_duplication") for error in result.errors)
    assert "direct_answer_phrase" in result.errors


def test_disabled_side_has_empty_steps_and_zero_mask():
    payload = cot_side_payload(
        text="yes", media=[], gate=GateDecision(False, "short_label"), stages=None
    )
    assert payload["cot_steps"] == []
    assert payload["cot_step_mask"] == [0, 0, 0, 0]


def test_short_explicit_no_relation_stage_is_allowed():
    stages = dict(GOOD_STAGES)
    stages["stage_3"] = "No extra relation or constraint."
    result = validate_cot(stages, side="query")
    assert result.valid, result.errors


def test_page_number_leak_is_rejected():
    stages = dict(GOOD_STAGES)
    stages["stage_1"] += " The source is page 12."
    result = validate_cot(stages, side="query")
    assert "page_number_leak" in result.errors


def test_stable_sharding_hash():
    assert stable_int("sample-a", 42) == stable_int("sample-a", 42)
    assert 0 <= stable_int("sample-a") % 4 < 4
    assert tuple(GOOD_STAGES) == STAGE_KEYS
