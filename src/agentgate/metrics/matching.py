"""Reference-trajectory matching (B2).

A gold trajectory is not a string to diff. It is an ordered list of steps where each step admits
**allowed alternatives** (``search_web`` and ``search_docs`` may both be right), some steps are
**optional**, and consecutive steps may form an **unordered group** whose members can be
satisfied in any relative order. Getting this wrong penalises correct agents for cosmetic
differences, which is the fastest way to make a gate untrustworthy.

Two alignments are computed, because the metrics need different things:

* an **ordered** alignment, greedy subsequence matching, backing ``exact_match`` and
  ``in_order_match``;
* an **unordered** alignment, greedy bipartite matching, backing ``any_order_match``,
  precision/recall/F1 and argument correctness — order-insensitive partial credit.

``lcs_ratio`` is computed separately by dynamic programming, since it is the one metric that
awards *partial* order credit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from agentgate.metrics.base import Embedder, f1_score, ratio
from agentgate.schemas.task import ArgComparator, ReferenceStep, ReferenceTrajectory
from agentgate.schemas.trajectory import ToolInvocation


@dataclass(frozen=True, slots=True)
class ArgCheck:
    """The outcome of comparing one expected argument against what the agent supplied."""

    key: str
    expected: Any
    actual: Any
    ok: bool
    comparator: str


@dataclass(slots=True)
class TrajectoryMatch:
    """Everything the trajectory metric family reads."""

    n_reference: int
    """Non-optional reference steps — the recall denominator."""
    n_predicted: int
    ordered_pairs: list[tuple[int, int]] = field(default_factory=list)
    unordered_pairs: list[tuple[int, int]] = field(default_factory=list)
    matched_refs: set[int] = field(default_factory=set)
    matched_preds: set[int] = field(default_factory=set)
    missing_refs: list[int] = field(default_factory=list)
    extra_preds: list[int] = field(default_factory=list)
    ordered_matched_refs: set[int] = field(default_factory=set)
    allow_extra_calls: bool = True
    arg_checks: list[ArgCheck] = field(default_factory=list)

    @property
    def exact_match(self) -> bool:
        """Predicted sequence equals the reference sequence: same calls, same order, no extras."""
        return (
            not self.missing_refs
            and not self.extra_preds
            and len(self.ordered_matched_refs) == self.n_reference
            and len(self.matched_preds) == self.n_predicted
        )

    @property
    def in_order_match(self) -> bool:
        """Reference appears as a subsequence of the prediction; extras allowed by default."""
        if len(self.ordered_matched_refs) != self.n_reference:
            return False
        return self.allow_extra_calls or not self.extra_preds

    @property
    def any_order_match(self) -> bool:
        """Every required reference call is present somewhere, in any order."""
        if len(self.matched_refs) != self.n_reference:
            return False
        return self.allow_extra_calls or not self.extra_preds

    @property
    def precision(self) -> float:
        """Correct predicted calls / total predicted calls."""
        return ratio(len(self.matched_preds), self.n_predicted)

    @property
    def recall(self) -> float:
        """Reference calls recovered / total required reference calls."""
        return ratio(len(self.matched_refs), self.n_reference, default=1.0)

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        return f1_score(self.precision, self.recall)

    @property
    def argument_correctness(self) -> float | None:
        """Correct key-value pairs / expected pairs; ``None`` when nothing was expected."""
        if not self.arg_checks:
            return None
        correct = sum(1 for check in self.arg_checks if check.ok)
        return correct / len(self.arg_checks)


def _blocks(steps: list[ReferenceStep]) -> list[list[int]]:
    """Partition reference-step indices into ordered blocks.

    Consecutive steps sharing a ``group`` form one unordered block; everything else is a
    singleton block that must be satisfied in sequence.
    """
    blocks: list[list[int]] = []
    for index, step in enumerate(steps):
        if step.group is not None and blocks and steps[blocks[-1][0]].group == step.group:
            blocks[-1].append(index)
        else:
            blocks.append([index])
    return blocks


def _ordered_alignment(
    steps: list[ReferenceStep], predicted: list[ToolInvocation]
) -> list[tuple[int, int]]:
    """Greedy in-order subsequence match, honouring unordered groups."""
    pairs: list[tuple[int, int]] = []
    cursor = 0
    for block in _blocks(steps):
        if len(block) == 1:
            ref_index = block[0]
            step = steps[ref_index]
            found = next(
                (i for i in range(cursor, len(predicted)) if step.matches_tool(predicted[i].tool)),
                None,
            )
            if found is not None:
                pairs.append((ref_index, found))
                cursor = found + 1
            continue

        remaining = set(block)
        scan = cursor
        while remaining and scan < len(predicted):
            hit = next(
                (i for i in sorted(remaining) if steps[i].matches_tool(predicted[scan].tool)), None
            )
            if hit is not None:
                pairs.append((hit, scan))
                remaining.discard(hit)
                cursor = scan + 1
            scan += 1
    return pairs


def _unordered_alignment(
    steps: list[ReferenceStep], predicted: list[ToolInvocation]
) -> list[tuple[int, int]]:
    """Greedy bipartite match on tool name, each predicted call used at most once.

    Reference steps are matched in declaration order and prefer the *earliest* unused predicted
    call that satisfies them, which keeps the alignment stable and reproducible.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for ref_index, step in enumerate(steps):
        found = next(
            (
                i
                for i, call in enumerate(predicted)
                if i not in used and step.matches_tool(call.tool)
            ),
            None,
        )
        if found is not None:
            used.add(found)
            pairs.append((ref_index, found))
    return pairs


def lcs_length(steps: list[ReferenceStep], predicted: list[ToolInvocation]) -> int:
    """Longest common subsequence between reference steps and predicted tools.

    Equality is "the predicted tool is one of this step's allowed alternatives", so alternatives
    are honoured by the order-sensitive metric too.

    Args:
        steps: Reference steps.
        predicted: Predicted tool invocations.

    Returns:
        Length of the LCS.
    """
    n, m = len(steps), len(predicted)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if steps[i - 1].matches_tool(predicted[j - 1].tool):
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[n][m]


def match_trajectory(
    reference: ReferenceTrajectory,
    predicted: list[ToolInvocation],
    *,
    embedder: Embedder | None = None,
) -> TrajectoryMatch:
    """Align a predicted trajectory against its gold specification.

    Args:
        reference: Gold trajectory.
        predicted: Tool invocations the agent actually made.
        embedder: Used only by ``semantic`` argument comparators.

    Returns:
        A :class:`TrajectoryMatch` carrying every quantity the B2 metrics need.
    """
    steps = list(reference.steps)
    required = {index for index, step in enumerate(steps) if not step.optional}

    ordered = _ordered_alignment(steps, predicted)
    unordered = _unordered_alignment(steps, predicted)

    matched_refs = {ref for ref, _ in unordered}
    matched_preds = {pred for _, pred in unordered}
    ordered_required = {ref for ref, _ in ordered if ref in required}

    match = TrajectoryMatch(
        n_reference=len(required),
        n_predicted=len(predicted),
        ordered_pairs=ordered,
        unordered_pairs=unordered,
        matched_refs=matched_refs & required,
        matched_preds=matched_preds,
        missing_refs=sorted(required - matched_refs),
        extra_preds=sorted(set(range(len(predicted))) - matched_preds),
        ordered_matched_refs=ordered_required,
        allow_extra_calls=reference.allow_extra_calls,
    )
    match.arg_checks = _check_arguments(steps, predicted, unordered, embedder=embedder)
    return match


def _check_arguments(
    steps: list[ReferenceStep],
    predicted: list[ToolInvocation],
    pairs: list[tuple[int, int]],
    *,
    embedder: Embedder | None,
) -> list[ArgCheck]:
    """Compare expected arguments against what the agent supplied, per matched pair.

    Unmatched reference steps still contribute their expected keys as failures: an argument you
    never supplied is not an argument you got right.
    """
    matched = dict(pairs)
    checks: list[ArgCheck] = []
    for ref_index, step in enumerate(steps):
        if not step.args:
            continue
        call = predicted[matched[ref_index]] if ref_index in matched else None
        for key, expected in step.args.items():
            comparator = step.arg_comparators.get(key, ArgComparator())
            if comparator.kind == "ignore":
                continue
            actual = call.args.get(key) if call is not None else None
            ok = call is not None and compare_argument(expected, actual, comparator, embedder)
            checks.append(
                ArgCheck(
                    key=f"{step.tools[0]}.{key}",
                    expected=expected,
                    actual=actual,
                    ok=ok,
                    comparator=comparator.kind,
                )
            )
    return checks


def compare_argument(
    expected: Any,
    actual: Any,
    comparator: ArgComparator,
    embedder: Embedder | None = None,
) -> bool:
    """Compare one argument value under the declared comparator.

    Args:
        expected: Reference value.
        actual: Value the agent supplied.
        comparator: How to compare them.
        embedder: Required for ``kind='semantic'``; without one, semantic falls back to exact.

    Returns:
        True when the supplied value is acceptable.
    """
    if actual is None:
        return False
    if comparator.kind == "numeric":
        return _numeric_close(expected, actual, comparator.tolerance)
    if comparator.kind == "regex":
        pattern = comparator.pattern or ""
        flags = 0 if comparator.case_sensitive else re.IGNORECASE
        return bool(re.search(pattern, str(actual), flags))
    if comparator.kind == "semantic":
        if embedder is None:
            return _exact(expected, actual, case_sensitive=comparator.case_sensitive)
        return cosine_of(str(expected), str(actual), embedder) >= comparator.threshold
    return _exact(expected, actual, case_sensitive=comparator.case_sensitive)


def _numeric_close(expected: Any, actual: Any, tolerance: float) -> bool:
    try:
        return abs(float(expected) - float(actual)) <= tolerance + 1e-12
    except (TypeError, ValueError):
        return False


def _exact(expected: Any, actual: Any, *, case_sensitive: bool) -> bool:
    if isinstance(expected, str) and isinstance(actual, str) and not case_sensitive:
        return expected.strip().casefold() == actual.strip().casefold()
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) is bool(actual)
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-9)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()
    return bool(expected == actual)


def cosine_of(first: str, second: str, embedder: Embedder) -> float:
    """Cosine similarity of two strings under ``embedder``, clamped to [0, 1]."""
    vectors = embedder.encode([first, second])
    return cosine(vectors[0], vectors[1])


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, clamped to [0, 1].

    Negative similarity is clamped to 0: for the metrics that use this, "less related than
    unrelated" is not a meaningful distinction and would put a proportion metric out of range.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))
