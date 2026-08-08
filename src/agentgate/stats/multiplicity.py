"""Multiple-comparison control (C3).

A suite gating on 12 metrics at raw alpha = 0.05 has roughly a 46% chance of at least one false
alarm per comparison — which is how a statistical gate becomes a gate everyone learns to ignore.

Benjamini-Hochberg controls the **false discovery rate**: the expected share of *rejections*
that are false. That is the right error rate here. Bonferroni controls the probability of any
false rejection at all, which is stricter than a CI gate needs and would hide real regressions
behind a conservative threshold. BH is also uniformly less conservative — every metric Bonferroni
rejects, BH rejects too, which the property tests assert.

Raw and adjusted p-values both appear in the report. A reader must be able to see the correction
being applied rather than infer it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Adjustment:
    """The outcome of a multiplicity correction over one family of tests."""

    names: tuple[str, ...]
    raw: tuple[float, ...]
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    q: float
    method: str = "benjamini-hochberg"

    def by_name(self) -> dict[str, tuple[float, float, bool]]:
        """``{name: (raw p, adjusted p, rejected)}``."""
        return {
            name: (raw, adjusted, rejected)
            for name, raw, adjusted, rejected in zip(
                self.names, self.raw, self.adjusted, self.rejected, strict=True
            )
        }

    @property
    def n_rejected(self) -> int:
        """How many hypotheses were rejected."""
        return sum(self.rejected)


def benjamini_hochberg(
    p_values: Sequence[float], *, q: float = 0.05, names: Sequence[str] | None = None
) -> Adjustment:
    """Benjamini-Hochberg step-up FDR control (Benjamini & Hochberg, 1995).

    Sort the ``m`` p-values ascending; find the largest ``k`` with ``p_(k) <= k*q/m``; reject
    everything up to it. Adjusted p-values are the monotone step-up transform
    ``min over j >= i of (m/j) * p_(j)``, clipped at 1, so that "adjusted p <= q" reproduces the
    same decisions as the threshold rule.

    Args:
        p_values: Raw p-values for one family of tests.
        q: Target false discovery rate.
        names: Optional labels, aligned with ``p_values``.

    Returns:
        The adjustment, in the caller's original order.
    """
    raw = [min(1.0, max(0.0, float(p))) for p in p_values]
    m = len(raw)
    labels = tuple(names) if names is not None else tuple(f"p{i}" for i in range(m))
    if m == 0:
        return Adjustment(names=(), raw=(), adjusted=(), rejected=(), q=q)

    order = sorted(range(m), key=lambda index: raw[index])
    sorted_p = [raw[index] for index in order]

    adjusted_sorted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        candidate = min(1.0, sorted_p[rank - 1] * m / rank)
        running = min(running, candidate)
        adjusted_sorted[rank - 1] = running

    adjusted = [0.0] * m
    for position, index in enumerate(order):
        adjusted[index] = adjusted_sorted[position]

    # Rejections are derived from the adjusted p-values rather than re-deriving the step-up
    # threshold. The two rules are equivalent in exact arithmetic, but `p*m/j` and `j*q/m` can
    # disagree by one ULP at the boundary — and a report that prints "adjusted p = 0.05, not
    # rejected" at q = 0.05 is indefensible however correct its floating point.
    rejected = [value <= q for value in adjusted]

    return Adjustment(
        names=labels,
        raw=tuple(raw),
        adjusted=tuple(adjusted),
        rejected=tuple(rejected),
        q=q,
    )


def bonferroni(
    p_values: Sequence[float], *, alpha: float = 0.05, names: Sequence[str] | None = None
) -> Adjustment:
    """Bonferroni correction, provided for the comparison the report draws against BH."""
    raw = [min(1.0, max(0.0, float(p))) for p in p_values]
    m = len(raw)
    labels = tuple(names) if names is not None else tuple(f"p{i}" for i in range(m))
    adjusted = tuple(min(1.0, p * m) for p in raw)
    return Adjustment(
        names=labels,
        raw=tuple(raw),
        adjusted=adjusted,
        rejected=tuple(p <= alpha for p in adjusted),
        q=alpha,
        method="bonferroni",
    )


def expected_false_alarms(n_tests: int, alpha: float = 0.05) -> float:
    """Expected false positives from ``n_tests`` independent nulls at raw ``alpha``.

    Used in the report to say, in one number, why the correction exists.
    """
    return n_tests * alpha


def family_wise_error(n_tests: int, alpha: float = 0.05) -> float:
    """Probability of at least one false alarm across ``n_tests`` independent nulls."""
    return float(1.0 - np.power(1.0 - alpha, n_tests))
