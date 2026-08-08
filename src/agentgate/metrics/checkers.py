"""Programmatic success checkers for ``outcome.task_success`` (B1).

The preferred checker is ``goal_state``: compare the sandbox's **end state** against the task's
annotated goal (tau-bench, E2). Whether the refund happened is a fact about the database; whether
the answer *sounds* like it happened is a fact about prose. Only the first is a success.

``answer_match`` and friends exist for stateless tasks, where there is nothing to inspect but
the answer.
"""

from __future__ import annotations

import math
import re
import string
from collections.abc import Callable
from typing import Any

from agentgate.metrics.base import ScoredSample
from agentgate.schemas.trajectory import RunStatus

CheckerResult = tuple[bool, dict[str, Any]]
CheckerFn = Callable[[ScoredSample, dict[str, Any]], CheckerResult]

ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "cannot determine",
    "no information",
    "unable to answer",
    "not in the provided",
    "does not contain",
)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: casefold, drop articles and punctuation, collapse whitespace."""
    lowered = text.casefold()
    without_punct = lowered.translate(_PUNCT)
    without_articles = _ARTICLES.sub(" ", without_punct)
    return " ".join(without_articles.split())


def answer_tokens(text: str) -> list[str]:
    """Normalised whitespace tokens, the unit of token-level F1."""
    return normalize_answer(text).split()


def extract_number(text: str) -> float | None:
    """Return the first number in ``text``, or ``None``."""
    match = _NUMBER.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:  # pragma: no cover - regex guarantees parseability
        return None


def is_abstention(text: str) -> bool:
    """True when the answer declines to answer rather than guessing."""
    lowered = text.casefold()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


def subset_match(expected: Any, actual: Any, *, tolerance: float = 1e-9) -> bool:
    """Check that ``actual`` contains everything ``expected`` declares.

    Goal states are written as *what should have changed*, not as a full state dump, so matching
    is subset-shaped:

    * mappings — every expected key must be present and match recursively;
    * sequences — every expected element must be matched by some element of ``actual``, which
      lets a goal state say "an email was sent to X" without pinning down the whole outbox;
    * numbers — compared with a tolerance, since a refund is a float;
    * everything else — equality.

    Args:
        expected: The declared goal fragment.
        actual: The observed end state.
        tolerance: Absolute tolerance for numeric leaves.

    Returns:
        True when ``actual`` satisfies ``expected``.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and subset_match(value, actual[key], tolerance=tolerance)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(
            any(subset_match(item, candidate, tolerance=tolerance) for candidate in actual)
            for item in expected
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) is bool(actual)
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance)
    return bool(expected == actual)


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------


def check_goal_state(sample: ScoredSample, params: dict[str, Any]) -> CheckerResult:
    """Compare the sandbox end state against the task's annotated goal state (E2)."""
    goal = sample.reference.goal_state
    if not goal:
        return False, {"reason": "task declares no goal_state"}
    state = sample.trajectory.final_state
    if state is None:
        return False, {"reason": "agent produced no end state"}
    tolerance = float(params.get("tolerance", 0.01))
    ok = subset_match(goal, state, tolerance=tolerance)
    mismatches = [key for key, value in goal.items() if not subset_match(value, state.get(key))]
    return ok, {"goal": goal, "state": state, "mismatched_keys": mismatches}


def check_answer_match(sample: ScoredSample, params: dict[str, Any]) -> CheckerResult:
    """Normalised equality against the reference answer or any accepted alternative."""
    reference = sample.reference
    candidates = [answer for answer in [reference.answer, *reference.accepted_answers] if answer]
    if not candidates:
        return False, {"reason": "task declares no reference answer"}
    normalized = normalize_answer(sample.answer)
    if params.get("contains", False):
        ok = any(normalize_answer(candidate) in normalized for candidate in candidates)
    else:
        ok = any(normalize_answer(candidate) == normalized for candidate in candidates)
    return ok, {"answer": sample.answer, "accepted": candidates}


def check_numeric(sample: ScoredSample, params: dict[str, Any]) -> CheckerResult:
    """Numeric answer within the task-declared tolerance."""
    reference = sample.reference
    expected = reference.numeric_answer
    if expected is None:
        return False, {"reason": "task declares no numeric_answer"}
    actual = extract_number(sample.answer)
    if actual is None:
        return False, {"reason": "no number in the answer", "answer": sample.answer}
    tolerance = float(params.get("tolerance", reference.numeric_tolerance))
    return abs(actual - expected) <= tolerance + 1e-12, {"expected": expected, "actual": actual}


def check_abstention(sample: ScoredSample, params: dict[str, Any]) -> CheckerResult:  # noqa: ARG001
    """For unanswerable tasks, success is declining to answer."""
    abstained = is_abstention(sample.answer)
    return abstained, {"abstained": abstained, "answer": sample.answer}


def check_contains(sample: ScoredSample, params: dict[str, Any]) -> CheckerResult:
    """Success when the answer contains every required substring."""
    required = [str(item) for item in params.get("required", [])]
    if not required:
        return False, {"reason": "checker 'contains' needs a non-empty `required` list"}
    lowered = sample.answer.casefold()
    missing = [item for item in required if item.casefold() not in lowered]
    return not missing, {"missing": missing}


CHECKERS: dict[str, CheckerFn] = {
    "goal_state": check_goal_state,
    "answer_match": check_answer_match,
    "numeric": check_numeric,
    "abstention": check_abstention,
    "contains": check_contains,
}
"""Registered checkers, referenced by name from a task's ``checker`` block."""


def run_checker(sample: ScoredSample) -> CheckerResult:
    """Run the task's declared checker.

    An agent that crashed, timed out, or exhausted its step budget fails regardless of what the
    checker would say about its (absent) answer.

    Args:
        sample: The scored sample.

    Returns:
        ``(success, detail)``.

    Raises:
        KeyError: When the task names a checker that is not registered.
    """
    checker = sample.task.checker
    if checker is None:
        return False, {"reason": "task declares no checker"}
    if sample.trajectory.status is not RunStatus.COMPLETED:
        return False, {"reason": f"run status {sample.trajectory.status.value}"}
    if checker.name not in CHECKERS:
        known = ", ".join(sorted(CHECKERS))
        msg = f"unknown checker {checker.name!r}; registered checkers: {known}"
        raise KeyError(msg)
    ok, detail = CHECKERS[checker.name](sample, dict(checker.params))
    return ok, {"checker": checker.name, **detail}
