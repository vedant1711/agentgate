"""Outcome (final-answer) metrics — B1.

Note the deliberate ordering of preference: ``outcome.task_success`` runs a programmatic checker
against end state where one exists, and only falls back to answer inspection when the task is
stateless. String metrics (`exact_match`, `f1_token`) are reported alongside, never instead —
they measure phrasing agreement, which is a different question from whether the work got done.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, ClassVar

from agentgate.metrics.base import (
    BaseMetric,
    Scored,
    ScoredSample,
    binary,
    f1_score,
    ratio,
)
from agentgate.metrics.checkers import (
    answer_tokens,
    extract_number,
    is_abstention,
    normalize_answer,
    run_checker,
)
from agentgate.metrics.matching import cosine_of
from agentgate.metrics.registry import register
from agentgate.schemas.common import Direction, DType, MetricFamily, Requirement

OUTCOME = MetricFamily.OUTCOME


@register
class TaskSuccess(BaseMetric):
    """Did the task actually get done, judged by its programmatic checker (B1, E2)?"""

    name = "outcome.task_success"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "binary"
    direction: ClassVar[Direction] = "higher_is_better"
    description: ClassVar[str] = (
        "Task-defined programmatic checker. Prefers tau-bench-style end-state comparison; "
        "falls back to answer checks for stateless tasks."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Run the task's checker."""
        if sample.task.checker is None:
            return Scored.skip("task declares no checker")
        ok, detail = run_checker(sample)
        return Scored(value=binary(ok), detail=detail)


@register
class ExactMatch(BaseMetric):
    """Normalised string equality against the reference answer."""

    name = "outcome.exact_match"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "binary"
    requires: ClassVar[set[Requirement]] = {Requirement.REFERENCE_ANSWER}
    description: ClassVar[str] = "SQuAD-style normalised equality vs the reference answer."

    def compute(self, sample: ScoredSample) -> Scored:
        """Compare normalised answers."""
        candidates = _reference_answers(sample)
        predicted = normalize_answer(sample.answer)
        matched = next(
            (answer for answer in candidates if normalize_answer(answer) == predicted), None
        )
        return Scored(
            value=binary(matched is not None),
            detail={"normalized": predicted, "matched": matched},
        )


@register
class TokenF1(BaseMetric):
    """Token-level F1 against the reference answer (SQuAD-style partial credit)."""

    name = "outcome.f1_token"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "proportion"
    requires: ClassVar[set[Requirement]] = {Requirement.REFERENCE_ANSWER}
    description: ClassVar[str] = "Best token-level F1 across all accepted reference answers."

    def compute(self, sample: ScoredSample) -> Scored:
        """Take the best F1 across accepted answers."""
        predicted = answer_tokens(sample.answer)
        best = 0.0
        best_answer = ""
        for candidate in _reference_answers(sample):
            score = _token_f1(predicted, answer_tokens(candidate))
            if score > best:
                best, best_answer = score, candidate
        return Scored(value=best, detail={"best_against": best_answer, "n_tokens": len(predicted)})


@register
class NumericAccuracy(BaseMetric):
    """Numeric answer within the task-declared tolerance."""

    name = "outcome.numeric_accuracy"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "binary"
    requires: ClassVar[set[Requirement]] = {Requirement.REFERENCE_ANSWER}
    description: ClassVar[str] = "First number in the answer, within `numeric_tolerance`."

    def compute(self, sample: ScoredSample) -> Scored:
        """Extract and compare the first number in the answer."""
        expected = sample.reference.numeric_answer
        if expected is None:
            return Scored.skip("task declares no numeric_answer")
        actual = extract_number(sample.answer)
        if actual is None:
            return Scored(value=0.0, detail={"reason": "no number in answer"})
        tolerance = sample.reference.numeric_tolerance
        return Scored(
            value=binary(abs(actual - expected) <= tolerance + 1e-12),
            detail={"expected": expected, "actual": actual, "tolerance": tolerance},
        )


@register
class SemanticSimilarity(BaseMetric):
    """Cosine similarity between the answer and the reference, under the local embedder."""

    name = "outcome.semantic_similarity"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "continuous"
    requires: ClassVar[set[Requirement]] = {Requirement.REFERENCE_ANSWER, Requirement.EMBEDDINGS}
    description: ClassVar[str] = (
        "Cosine similarity vs the reference answer. The embedder in use is named in the report; "
        "the default is lexical, not semantic (see docs/limitations.md)."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Take the best similarity across accepted answers."""
        embedder = sample.context.embedder
        if embedder is None:  # pragma: no cover - guarded by `requires`
            return Scored.skip("no embedder configured")
        best = 0.0
        for candidate in _reference_answers(sample):
            best = max(best, cosine_of(sample.answer, candidate, embedder))
        return Scored(value=best, detail={"embedder": embedder.name})


@register
class JsonValid(BaseMetric):
    """Does the answer parse as JSON, on tasks that asked for JSON?

    The requirement is load-bearing. Without it this metric scores every sample in every suite,
    so a suite whose answers are prose — tau2-bench retail, for one — comes back at 0.009 and
    looks like a catastrophic formatting failure. It is not a finding about the model; it is the
    harness asking a question the task never posed. A task declares that it wants JSON by
    carrying an ``output_schema``, so that is the gate.
    """

    name = "outcome.json_valid"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "binary"
    requires: ClassVar[set[Requirement]] = {Requirement.OUTPUT_SCHEMA}
    description: ClassVar[str] = "The final answer parses as JSON (fenced code blocks stripped)."

    def compute(self, sample: ScoredSample) -> Scored:
        """Parse the answer, tolerating a markdown fence around it."""
        payload, error = _parse_json(sample.answer)
        return Scored(value=binary(payload is not None), detail={"error": error})


@register
class SchemaCompliant(BaseMetric):
    """Does the answer validate against the task's declared JSON Schema?"""

    name = "outcome.schema_compliant"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "binary"
    requires: ClassVar[set[Requirement]] = {Requirement.OUTPUT_SCHEMA}
    description: ClassVar[str] = "The answer parses as JSON and satisfies the task's schema."

    def compute(self, sample: ScoredSample) -> Scored:
        """Parse then structurally validate."""
        schema = sample.reference.output_schema or {}
        payload, error = _parse_json(sample.answer)
        if payload is None:
            return Scored(value=0.0, detail={"error": error})
        problems = validate_against_schema(payload, schema)
        return Scored(value=binary(not problems), detail={"problems": problems})


@register
class Abstained(BaseMetric):
    """Did the agent decline to answer?

    Reported for every task, not just unanswerable ones: abstention on an answerable question is
    a failure mode worth watching, and refusing to guess on an unanswerable one is the goal.
    """

    name = "outcome.abstained"
    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "binary"
    direction: ClassVar[Direction] = "lower_is_better"
    description: ClassVar[str] = "The answer declines to answer rather than asserting something."

    def compute(self, sample: ScoredSample) -> Scored:
        """Look for abstention language."""
        abstained = is_abstention(sample.answer)
        return Scored(
            value=binary(abstained),
            detail={"unanswerable_task": sample.reference.unanswerable},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_answers(sample: ScoredSample) -> list[str]:
    reference = sample.reference
    return [answer for answer in [reference.answer, *reference.accepted_answers] if answer]


def _token_f1(predicted: list[str], expected: list[str]) -> float:
    """SQuAD token-level F1 over multisets of tokens."""
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = Counter(predicted) & Counter(expected)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0
    precision = ratio(shared, len(predicted))
    recall = ratio(shared, len(expected))
    return f1_score(precision, recall)


def _parse_json(text: str) -> tuple[Any, str | None]:
    """Parse JSON, stripping a surrounding markdown fence if present."""
    candidate = text.strip()
    if candidate.startswith("```"):
        body = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = body.rsplit("```", 1)[0].strip()
    if not candidate:
        return None, "empty answer"
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def validate_against_schema(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Validate ``payload`` against a JSON Schema subset.

    Supports the keywords task authors actually reach for — ``type``, ``required``,
    ``properties``, ``enum``, ``items``, ``minimum``/``maximum`` — without taking a dependency
    on a full validator. Unsupported keywords are ignored rather than failed, so a richer schema
    degrades to a weaker check instead of a false negative.

    Args:
        payload: Parsed JSON value.
        schema: JSON Schema fragment.

    Returns:
        Human-readable problems; empty when the payload conforms.
    """
    problems: list[str] = []
    _validate(payload, schema, "$", problems)
    return problems


_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _validate(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and expected_type in _TYPES:
        allowed = _TYPES[expected_type]
        matches = isinstance(value, allowed) and not (
            expected_type in ("number", "integer") and isinstance(value, bool)
        )
        if not matches:
            problems.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            return

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not in enum {schema['enum']}")

    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}: missing required property {key!r}")
        for key, subschema in (schema.get("properties") or {}).items():
            if key in value and isinstance(subschema, dict):
                _validate(value[key], subschema, f"{path}.{key}", problems)

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate(item, items, f"{path}[{index}]", problems)
