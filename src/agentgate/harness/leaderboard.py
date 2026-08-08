"""Ranking models without claiming an order the evidence does not support.

A leaderboard is the most tempting place in this whole project to commit the exact error it was
built to prevent. Sorting models by point estimate and printing 1, 2, 3 asserts an ordering
between every adjacent pair — a large family of implicit comparisons, each made at no stated
confidence, on samples usually too small to support any of them.

So this module ranks into **tiers** instead of positions. Models are ordered by point estimate,
then grouped: a model joins the current tier while its interval still overlaps the tier leader's.
Within a tier the harness says *we cannot separate these*, which is nearly always the truthful
answer at the sample sizes a free-tier harness can afford.

**The tiering rule is conservative, and that asymmetry is deliberate.** Non-overlapping intervals
imply a significant difference, but the converse is false: two intervals can overlap while a
paired test comfortably rejects, because pairing removes the between-task variance that dominates
both marginal intervals. So a tier boundary is strong evidence of a real difference, while shared
tier membership is only an absence of evidence from *this* view. When two models ran the same
tasks, :func:`head_to_head` answers the question properly with the paired machinery, and it is
what the CLI points you at whenever a tier contains more than one model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from agentgate.errors import InsufficientDataError
from agentgate.harness.ledger import Ledger, LedgerEntry
from agentgate.providers.models import get_card
from agentgate.schemas.results import Estimate, MetricComparison
from agentgate.stats.aggregate import summarise_metric
from agentgate.stats.compare import ComparisonConfig, compare_metric

if TYPE_CHECKING:
    from agentgate.storage.duckdb_store import RunStore

DEFAULT_METRIC = "outcome.task_success"


@dataclass(frozen=True, slots=True)
class Standing:
    """One model's position on a leaderboard.

    Args:
        model_id: The model measured.
        label: Human-readable name from the catalogue, falling back to the id.
        run_id: The run this standing was computed from.
        recorded_at: When that run was made.
        estimate: The metric value with its uncertainty. Cluster-robust when the suite clusters.
        n_tasks: Tasks contributing to the estimate.
        k: Repetitions per task.
        completion_rate: Fraction of units the model actually finished.
        tier: 1-based tier. Equal tiers are not separated by this evidence.
    """

    model_id: str
    label: str
    run_id: str
    recorded_at: datetime
    estimate: Estimate
    n_tasks: int
    k: int
    completion_rate: float
    tier: int = 1

    @property
    def interval(self) -> tuple[float, float] | None:
        """The confidence interval, or ``None`` for a degenerate sample."""
        if self.estimate.ci_low is None or self.estimate.ci_high is None:
            return None
        return (self.estimate.ci_low, self.estimate.ci_high)


@dataclass(frozen=True, slots=True)
class Leaderboard:
    """Tiered standings for one metric on one suite."""

    suite: str
    metric: str
    direction: str
    standings: tuple[Standing, ...]
    excluded: tuple[tuple[str, str], ...] = ()
    """``(model_id, reason)`` for every model deliberately left off."""

    @property
    def n_tiers(self) -> int:
        """How many separable groups the evidence supports."""
        return max((standing.tier for standing in self.standings), default=0)

    @property
    def is_separable(self) -> bool:
        """True when the evidence distinguishes at least two groups of models."""
        return self.n_tiers > 1

    def tier(self, number: int) -> list[Standing]:
        """Every model in one tier."""
        return [standing for standing in self.standings if standing.tier == number]

    def verdict(self) -> str:
        """A one-line, honest summary of what the board establishes."""
        if not self.standings:
            return "No model has a complete, scored recording on this suite yet."
        if len(self.standings) == 1:
            only = self.standings[0]
            return f"Only {only.label} has been measured here; a leaderboard needs at least two."
        if not self.is_separable:
            return (
                f"{len(self.standings)} models measured, none separable on {self.metric}: "
                f"every interval overlaps the leader's. Run `agentgate head-to-head` for the "
                f"paired test, which is more powerful than these marginal intervals."
            )
        leaders = self.tier(1)
        names = ", ".join(standing.label for standing in leaders)
        tail = "" if len(leaders) == 1 else " (tied, not separated)"
        return f"Tier 1: {names}{tail} — {self.n_tiers} separable tiers of {len(self.standings)}."


def build_leaderboard(
    store: RunStore,
    *,
    suite: str,
    metric: str = DEFAULT_METRIC,
    ledger: Ledger | None = None,
    level: float = 0.95,
    require_complete: bool = True,
) -> Leaderboard:
    """Rank every model measured on ``suite`` by ``metric``, into tiers.

    Args:
        store: The run database.
        suite: Suite name to rank on.
        metric: Metric to rank by.
        ledger: Prebuilt ledger, to avoid re-reading the store.
        level: Confidence level for each model's interval.
        require_complete: Exclude models whose recording is partial. On by default: a model
            measured on 40 of 111 tasks is not comparable to one measured on all 111, and
            showing them side by side implies a comparison that was never made.

    Returns:
        The leaderboard, including an explicit list of who was excluded and why.
    """
    ledger = ledger or Ledger.from_store(store)
    standings: list[Standing] = []
    excluded: list[tuple[str, str]] = []
    direction = "higher_is_better"

    for entry in ledger.for_suite(suite):
        reason = _exclusion_reason(entry, require_complete=require_complete)
        if reason is not None:
            excluded.append((entry.cell.model_id, reason))
            continue

        results = store.load_scores(entry.run_id, metric=metric)
        summary = summarise_metric(results, clusters=store.clusters_for(entry.run_id), level=level)
        if summary is None:
            excluded.append((entry.cell.model_id, f"no scored samples for {metric}"))
            continue

        direction = summary.direction
        card = get_card(entry.cell.model_id)
        standings.append(
            Standing(
                model_id=entry.cell.model_id,
                label=card.label if card else entry.cell.model_id,
                run_id=entry.run_id,
                recorded_at=entry.recorded_at,
                # The cluster-robust interval is the honest one whenever the suite clusters;
                # summarise_metric only populates it when clustering is actually present.
                estimate=summary.clustered or summary.estimate,
                n_tasks=summary.n_tasks,
                k=summary.k,
                completion_rate=entry.completion_rate,
            )
        )

    ordered = _rank(standings, direction=direction)
    return Leaderboard(
        suite=suite,
        metric=metric,
        direction=direction,
        standings=tuple(assign_tiers(ordered)),
        excluded=tuple(sorted(excluded)),
    )


def _scored_entry(entries: dict[str, LedgerEntry], model_id: str, suite: str) -> LedgerEntry:
    """Return a model's scored entry for a suite, or explain precisely what is missing."""
    entry = entries.get(model_id)
    if entry is None:
        msg = f"{model_id!r} has no run on suite {suite!r}"
        raise InsufficientDataError(msg)
    if not entry.is_scored:
        msg = f"{model_id!r} was recorded on {suite!r} but never scored (run {entry.run_id})"
        raise InsufficientDataError(msg)
    return entry


def _exclusion_reason(entry: LedgerEntry, *, require_complete: bool) -> str | None:
    """Why this entry cannot enter a leaderboard, or ``None`` when it can."""
    if entry.cell.model_id == "unknown":
        return "run does not record which agent model produced it"
    if not entry.is_scored:
        return "recorded but never scored"
    if require_complete and not entry.is_complete:
        return f"partial recording: {entry.n_samples} of {entry.expected_samples} units"
    return None


def _rank(standings: list[Standing], *, direction: str) -> list[Standing]:
    """Order by point estimate, best first, respecting the metric's direction."""
    reverse = direction != "lower_is_better"
    return sorted(standings, key=lambda standing: standing.estimate.value, reverse=reverse)


def assign_tiers(ordered: list[Standing]) -> list[Standing]:
    """Group an already-ordered list into tiers of models this evidence cannot separate.

    A model stays in the current tier while its interval overlaps the **tier leader's** — not
    merely its immediate predecessor. Chaining to the predecessor would let a long run of
    pairwise-overlapping intervals collapse into one tier spanning models that clearly differ at
    the ends, which is the transitivity trap that makes naive tiering worse than no tiering.

    A model with no computable interval (n < 2) starts its own tier: nothing can be claimed about
    it, so it must not be silently absorbed into a claim about others.
    """
    tiers: list[Standing] = []
    tier = 0
    leader: Standing | None = None
    for standing in ordered:
        if leader is None or not _overlaps(leader, standing):
            tier += 1
            leader = standing
        tiers.append(_with_tier(standing, tier))
    return tiers


def _overlaps(left: Standing, right: Standing) -> bool:
    """True when two intervals share any value, so neither model is established as better."""
    a, b = left.interval, right.interval
    if a is None or b is None:
        return False
    return a[0] <= b[1] and b[0] <= a[1]


def _with_tier(standing: Standing, tier: int) -> Standing:
    """Return a copy carrying its tier."""
    return Standing(
        model_id=standing.model_id,
        label=standing.label,
        run_id=standing.run_id,
        recorded_at=standing.recorded_at,
        estimate=standing.estimate,
        n_tasks=standing.n_tasks,
        k=standing.k,
        completion_rate=standing.completion_rate,
        tier=tier,
    )


def head_to_head(
    store: RunStore,
    *,
    suite: str,
    baseline_model: str,
    candidate_model: str,
    metric: str = DEFAULT_METRIC,
    ledger: Ledger | None = None,
    config: ComparisonConfig | None = None,
) -> MetricComparison:
    """Compare two models with the paired machinery, on the tasks they both ran.

    This is the right way to separate two models and it is strictly more powerful than reading
    their marginal intervals off the leaderboard: both models faced identical task instances and
    identical seeds, so the between-task variance that dominates each model's own interval
    cancels in the difference. A pair the leaderboard shows as tied is routinely separated here.

    Args:
        store: The run database.
        suite: Suite both models ran.
        baseline_model: The reference model.
        candidate_model: The model being judged against it.
        metric: Metric to compare.
        ledger: Prebuilt ledger.
        config: Statistical configuration.

    Returns:
        The full paired comparison, with both one-sided tests and the interval.

    Raises:
        InsufficientDataError: When either model has no scored run on this suite, or the two runs
            share no task. Pairing across disjoint tasks is not a weaker comparison — it is a
            different and meaningless one, so it fails loudly instead.
    """
    ledger = ledger or Ledger.from_store(store)
    entries = {entry.cell.model_id: entry for entry in ledger.for_suite(suite)}

    baseline_entry = _scored_entry(entries, baseline_model, suite)
    candidate_entry = _scored_entry(entries, candidate_model, suite)

    return compare_metric(
        metric,
        store.load_scores(baseline_entry.run_id, metric=metric),
        store.load_scores(candidate_entry.run_id, metric=metric),
        clusters=store.clusters_for(baseline_entry.run_id),
        config=config or ComparisonConfig(),
    )
