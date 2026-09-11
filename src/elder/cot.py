"""Core schema, prompting, routing, and validation for ELDER CoT data.

The generated text is privileged supervision for Stage 2/3.  It is not a
task answer and is never part of the student's inference-time input.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


COT_SCHEMA_VERSION = "elder_cot_v2"
PROMPT_VERSION = "elder_cot_v2_glm4v"
STAGE_SCHEMA = "perception-evidence-relation-synthesis"
STAGE_KEYS = ("stage_1", "stage_2", "stage_3", "stage_4")
STAGE_TYPES = (
    "modality_perception",
    "evidence_selection",
    "relation_reasoning",
    "retrieval_synthesis",
)

_MEDIA_TOKENS = re.compile(
    r"<\|image(?:_\d+)?\|>|<image>|</image>|<video>|</video>", re.IGNORECASE
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u3400-\u9fff]")
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    reason: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_token_counts: list[int] = field(default_factory=list)
    total_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stage_token_counts": self.stage_token_counts,
            "total_tokens": self.total_tokens,
        }


def stable_int(value: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sample_id(dataset: str, task: str, source_index: int) -> str:
    digest = hashlib.sha256(
        f"{dataset}\0{task}\0{source_index}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{dataset}:{task}:{source_index}:{digest}"


def safe_task_filename(task_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")
    if not name:
        name = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    return f"{name}.jsonl"


def clean_input_text(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = _MEDIA_TOKENS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def approximate_token_count(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


def semantic_richness_gate(
    text: str,
    media: Sequence[Mapping[str, object]],
    *,
    side: str,
    min_text_tokens: int = 12,
    min_text_chars: int = 64,
) -> GateDecision:
    """Route rich inputs to four-stage CoT without dataset-name shortcuts."""

    if side not in {"query", "candidate"}:
        raise ValueError(f"side must be query or candidate, got {side!r}")
    text = clean_input_text(text)
    real_media = [item for item in media if item and item.get("type")]
    if real_media:
        return GateDecision(True, "media_present")
    if not text:
        return GateDecision(False, "empty_input")
    if side == "query":
        return GateDecision(True, "query_input")
    if approximate_token_count(text) >= min_text_tokens or len(text) >= min_text_chars:
        return GateDecision(True, "descriptive_text")
    if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?%?", text):
        return GateDecision(False, "short_number")
    if approximate_token_count(text) <= 4:
        return GateDecision(False, "short_label")
    return GateDecision(False, "short_answer")


def build_generation_prompt(
    *,
    side: str,
    text: str,
    media_count: int,
    attempt: int = 0,
    prior_errors: Sequence[str] = (),
) -> str:
    """Build a side-local prompt.  The paired side must never be passed here."""

    if side not in {"query", "candidate"}:
        raise ValueError(f"side must be query or candidate, got {side!r}")
    role_instruction = (
        "Stage 4 must state what kind of candidate should be retrieved and the "
        "core conditions that candidate must satisfy."
        if side == "query"
        else "Stage 4 must state what evidence this candidate contains and what "
        "kind of retrieval need that evidence can support."
    )
    media_instruction = (
        f"There are {media_count} attached images. If they are ordered video "
        "frames, reason about their temporal order."
        if media_count
        else "This input has no attached visual media."
    )
    correction = ""
    if attempt and prior_errors:
        correction = (
            "\nThe previous attempt was rejected for: "
            + "; ".join(prior_errors[:6])
            + ". Correct these issues."
        )
    cleaned_text = clean_input_text(text) or "[no textual content]"
    return f"""You create retrieval-oriented process supervision for ELDER.
Analyze ONLY the current {side} input. The paired item, label, positive ID, file
path, and task answer are deliberately unavailable. Never guess or invent them.
This is retrieval reasoning, not answer generation.

Produce exactly four distinct semantic checkpoints:
1. modality-grounded content perception: objective entities, attributes, text,
   regions, layout, actions, or events visible/present in this input;
2. retrieval evidence selection: the most discriminative evidence and irrelevant
   background to ignore;
3. relation and constraint reasoning: bindings, spatial/temporal/order/numeric
   relations, combined constraints, or explicitly say that no extra relation is
   present;
4. retrieval semantics synthesis: compact matching semantics. {role_instruction}

{media_instruction}
Input text:
{cleaned_text}

Before returning, privately verify that every claim is grounded in the attached media or input text. Preserve ambiguous wording exactly instead of resolving it with invented context.
Return one JSON object and nothing else, with exactly these string keys:
{{"stage_1":"...","stage_2":"...","stage_3":"...","stage_4":"..."}}
Use concise English. Aim for 24-80 tokens per stage and no more than 320 tokens
total. Do not include a direct answer, paired-item content, IDs, page numbers,
filenames, paths, markdown, XML tags, or additional keys.{correction}"""


def _json_objects(text: str) -> Iterable[dict]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_cot_json(raw_text: str) -> dict[str, str]:
    """Extract the final strict four-stage object from a Thinking-model reply."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("empty_generation")
    cleaned = _THINK_BLOCK.sub("", raw_text).strip()
    candidates: list[str] = [cleaned]
    candidates.extend(match.group(1).strip() for match in _CODE_FENCE.finditer(cleaned))
    parsed = []
    for candidate in candidates:
        parsed.extend(_json_objects(candidate))
    exact = [obj for obj in parsed if tuple(obj.keys()) == STAGE_KEYS]
    if not exact:
        shape_matches = [obj for obj in parsed if set(obj) == set(STAGE_KEYS)]
        if shape_matches:
            exact = [{key: shape_matches[-1][key] for key in STAGE_KEYS}]
    if not exact:
        raise ValueError("no_exact_four_stage_json")
    result = exact[-1]
    if not all(isinstance(result[key], str) for key in STAGE_KEYS):
        raise ValueError("stage_values_must_be_strings")
    return {key: result[key].strip() for key in STAGE_KEYS}


def _normalized_similarity(left: str, right: str) -> float:
    normalize = lambda value: re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", value.lower()).strip()
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def _contains_forbidden(text: str, terms: Iterable[str]) -> str | None:
    lowered = text.lower()
    for raw_term in terms:
        term = clean_input_text(raw_term).lower().strip()
        if len(term) < 12 or approximate_token_count(term) < 3:
            continue
        if term in lowered:
            return raw_term
    return None


def validate_cot(
    stages: Mapping[str, object],
    *,
    side: str,
    forbidden_terms: Iterable[str] = (),
    token_counter: Callable[[str], int] | None = None,
    max_stage_tokens: int = 96,
    max_total_tokens: int = 384,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if side not in {"query", "candidate"}:
        errors.append("invalid_side")
    if set(stages) != set(STAGE_KEYS):
        errors.append("schema_keys_mismatch")
    values: list[str] = []
    for key in STAGE_KEYS:
        value = stages.get(key, "")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key}_empty_or_not_string")
            values.append("")
        else:
            values.append(value.strip())

    counter = token_counter or approximate_token_count
    counts = [counter(value) for value in values]
    for key, count, value in zip(STAGE_KEYS, counts, values):
        explicit_no_relation = (
            key == "stage_3"
            and count >= 4
            and "no " in value.lower()
            and any(marker in value.lower() for marker in ("relation", "constraint"))
        )
        if count < 8 and not explicit_no_relation:
            errors.append(f"{key}_too_short:{count}")
        elif count < 24:
            warnings.append(f"{key}_below_recommended_length:{count}")
        if count > max_stage_tokens:
            errors.append(f"{key}_too_long:{count}")
    total = sum(counts)
    if total > max_total_tokens:
        errors.append(f"cot_too_long:{total}")
    elif total > 320:
        warnings.append(f"cot_above_recommended_length:{total}")

    for index in range(3):
        similarity = _normalized_similarity(values[index], values[index + 1])
        if similarity >= 0.88:
            errors.append(f"adjacent_stage_duplication:{index + 1}-{index + 2}:{similarity:.3f}")

    stage_2 = values[1].lower()
    if not any(
        marker in stage_2
        for marker in (
            "evidence", "discriminative", "relevant", "focus", "ignore",
            "signal", "field", "region", "frame", "attribute", "entity",
        )
    ):
        errors.append("stage_2_missing_evidence_selection")
    stage_3 = values[2].lower()
    if not any(
        marker in stage_3
        for marker in (
            "relation", "constraint", "binding", "spatial", "temporal",
            "sequence", "order", "compare", "combined", "simultaneously",
            "no additional", "no extra", "no explicit",
        )
    ):
        errors.append("stage_3_missing_relation_or_constraint")
    stage_4 = values[3].lower()
    stage_4_markers = (
        ("retrieve", "candidate", "target", "must", "should")
        if side == "query"
        else ("candidate", "evidence", "support", "contains", "retrieval need")
    )
    if not any(marker in stage_4 for marker in stage_4_markers):
        errors.append(f"stage_4_wrong_{side}_role")

    joined = " ".join(values)
    if re.search(r"\b(correct|final|right) answer\s+(?:is|:)\b", joined, re.IGNORECASE):
        errors.append("direct_answer_phrase")
    if re.search(r"(?:/root/|[A-Za-z]:\\|\.(?:jpg|jpeg|png|webp|mp4|pdf)\b)", joined, re.IGNORECASE):
        errors.append("path_or_filename_leak")
    if re.search(r"\bpage(?: number)?\s*#?\d+\b", joined, re.IGNORECASE):
        errors.append("page_number_leak")
    leaked = _contains_forbidden(joined, forbidden_terms)
    if leaked is not None:
        errors.append("paired_content_leak")

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        stage_token_counts=counts,
        total_tokens=total,
    )


def cot_side_payload(
    *,
    text: str,
    media: Sequence[Mapping[str, object]],
    gate: GateDecision,
    stages: Mapping[str, str] | None = None,
) -> dict:
    enabled = gate.enabled
    ordered_stages = [stages[key] for key in STAGE_KEYS] if stages else []
    return {
        "text": clean_input_text(text),
        "media": [dict(item) for item in media],
        "cot_enabled": enabled,
        "cot_steps": ordered_stages,
        "cot_step_mask": [1, 1, 1, 1] if enabled and stages else [0, 0, 0, 0],
        "cot_gate_reason": gate.reason,
    }


def metadata_payload(*, model_name: str, retries: Mapping[str, int]) -> dict:
    return {
        "generator": str(Path(model_name)),
        "prompt_version": PROMPT_VERSION,
        "schema_version": COT_SCHEMA_VERSION,
        "stage_schema": STAGE_SCHEMA,
        "num_steps": 4,
        "retries": dict(retries),
    }
