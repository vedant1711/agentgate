"""Pairing guards for baseline-vs-candidate comparison (C2).

*"Runs to be compared MUST use identical task instances, identical K, and identical seeds; the
runner enforces this (comparisons across mismatched suites are refused, not warned)."*

That severity is deliberate. A paired test computes per-task differences; if the two runs saw
different tasks, the differences are not differences of anything and the resulting p-value is
noise wearing a decimal point. There is no safe way to downgrade this to a warning.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentgate.errors import SuiteMismatchError
from agentgate.schemas.results import RunManifest
from agentgate.schemas.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class PairedSample:
    """One task/repetition observed under both systems."""

    task_id: str
    rep: int
    baseline: Trajectory
    candidate: Trajectory

    @property
    def key(self) -> tuple[str, int]:
        """The pairing key."""
        return (self.task_id, self.rep)


def assert_comparable(baseline: RunManifest, candidate: RunManifest) -> None:
    """Refuse to compare two runs that are not paired on identical inputs.

    Args:
        baseline: Baseline run manifest.
        candidate: Candidate run manifest.

    Raises:
        SuiteMismatchError: When suite content, K, seed, or schema version differ.
    """
    ok, reason = baseline.is_comparable_to(candidate)
    if ok:
        return
    msg = (
        f"cannot compare {baseline.run_id!r} with {candidate.run_id!r}: {reason}. "
        f"A paired test requires identical task instances, identical K, and identical seeds "
        f"(spec C2) — re-run both systems against the same suite version."
    )
    raise SuiteMismatchError(msg)


def pair_trajectories(
    baseline: list[Trajectory], candidate: list[Trajectory]
) -> list[PairedSample]:
    """Align two runs' trajectories on ``(task_id, rep)``.

    Args:
        baseline: Baseline trajectories.
        candidate: Candidate trajectories.

    Returns:
        Paired samples in (task, rep) order.

    Raises:
        SuiteMismatchError: When either run has units the other lacks, or when a paired unit
            was executed under different seeds.
    """
    left = {(t.task_id, t.rep): t for t in baseline}
    right = {(t.task_id, t.rep): t for t in candidate}

    only_baseline = sorted(left.keys() - right.keys())
    only_candidate = sorted(right.keys() - left.keys())
    if only_baseline or only_candidate:
        msg = (
            f"runs are not paired: {len(only_baseline)} unit(s) only in baseline "
            f"(e.g. {only_baseline[:3]}), {len(only_candidate)} only in candidate "
            f"(e.g. {only_candidate[:3]})"
        )
        raise SuiteMismatchError(msg)

    pairs: list[PairedSample] = []
    for key in sorted(left):
        first, second = left[key], right[key]
        if first.seed != second.seed:
            msg = (
                f"unit {key[0]}#{key[1]} ran under different seeds "
                f"({first.seed} vs {second.seed}); pairing would compare unlike with unlike"
            )
            raise SuiteMismatchError(msg)
        pairs.append(PairedSample(task_id=key[0], rep=key[1], baseline=first, candidate=second))
    return pairs
