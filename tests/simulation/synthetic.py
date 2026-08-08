"""Synthetic agents with known statistical properties (H3).

The Monte-Carlo suite validates the statistics engine against ground truth it controls: agents
whose true success probability, task-difficulty correlation, cluster structure, and within-task
repetition variance are all set by the experimenter. If the engine's type-I error, coverage, and
power match what these generators imply, the engine works; if a real eval disagrees, the eval is
the surprising part.

The generators deliberately reproduce the structures the spec's evidence base calls out:
per-question correlation between two systems in the 0.3-0.7 band (E3), clustered tasks, and
repetition-level noise that makes pass^k decay (E2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SuiteDesign:
    """The shape of a synthetic suite."""

    n_tasks: int = 60
    k_reps: int = 4
    n_clusters: int = 0
    """0 means every task is its own cluster."""
    pairing_correlation: float = 0.5
    """Target correlation between the two systems' per-task scores (E3's 0.3-0.7)."""

    @property
    def cluster_ids(self) -> list[str]:
        """Cluster id per task."""
        if self.n_clusters <= 0:
            return [f"c{index}" for index in range(self.n_tasks)]
        return [f"c{index % self.n_clusters}" for index in range(self.n_tasks)]


@dataclass(slots=True)
class GaussianAgents:
    """Two systems whose per-task scores are correlated Gaussians.

    Args:
        design: Suite shape.
        baseline_mean: True baseline mean.
        effect: True difference ``candidate - baseline``. Zero is the null used for type-I error.
        task_sd: Between-task SD (task difficulty).
        rep_sd: Within-task SD across repetitions (agent inconsistency).
        cluster_sd: Extra SD shared by every task in a cluster — the dependence that makes naive
            SEs under-cover.
    """

    design: SuiteDesign = field(default_factory=SuiteDesign)
    baseline_mean: float = 0.7
    effect: float = 0.0
    task_sd: float = 0.15
    rep_sd: float = 0.10
    cluster_sd: float = 0.0

    def draw(
        self, rng: np.random.Generator
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, str]]:
        """Draw one synthetic experiment.

        Returns:
            ``(baseline scores, candidate scores, clusters)`` where each score map is
            ``{task_id: [score per repetition]}``.
        """
        design = self.design
        n, k = design.n_tasks, design.k_reps
        clusters = design.cluster_ids

        cluster_offsets = {
            cluster: rng.normal(0.0, self.cluster_sd) if self.cluster_sd else 0.0
            for cluster in set(clusters)
        }

        # Correlated task-level difficulty: a shared component plus independent components,
        # mixed to hit the requested correlation exactly in expectation.
        rho = max(0.0, min(0.999, design.pairing_correlation))
        shared = rng.normal(0.0, 1.0, n)
        own_base = rng.normal(0.0, 1.0, n)
        own_cand = rng.normal(0.0, 1.0, n)
        base_difficulty = self.task_sd * (np.sqrt(rho) * shared + np.sqrt(1 - rho) * own_base)
        cand_difficulty = self.task_sd * (np.sqrt(rho) * shared + np.sqrt(1 - rho) * own_cand)

        baseline: dict[str, list[float]] = {}
        candidate: dict[str, list[float]] = {}
        cluster_map: dict[str, str] = {}
        for index in range(n):
            task_id = f"t{index:03d}"
            cluster_map[task_id] = clusters[index]
            offset = cluster_offsets[clusters[index]]
            base_centre = self.baseline_mean + base_difficulty[index] + offset
            cand_centre = (
                base_centre - base_difficulty[index] + cand_difficulty[index] + self.effect
            )
            baseline[task_id] = (base_centre + rng.normal(0.0, self.rep_sd, k)).tolist()
            candidate[task_id] = (cand_centre + rng.normal(0.0, self.rep_sd, k)).tolist()
        return baseline, candidate, cluster_map


@dataclass(slots=True)
class BernoulliAgents:
    """Two systems whose repetitions are Bernoulli draws with per-task success probabilities.

    This is the generator pass^k unbiasedness is checked against: the true ``P(all k succeed)``
    for a task with probability ``p`` is exactly ``p^k``, so averaging the estimator over many
    replications must converge to ``E[p^k]``.
    """

    design: SuiteDesign = field(default_factory=SuiteDesign)
    baseline_p: float = 0.75
    effect: float = 0.0
    task_spread: float = 0.0
    """Beta-like spread of per-task success probabilities; 0 gives every task the same p."""

    def task_probabilities(self, rng: np.random.Generator, mean_p: float) -> np.ndarray:
        """Per-task success probabilities around ``mean_p``."""
        if self.task_spread <= 0.0:
            return np.full(self.design.n_tasks, mean_p)
        concentration = max(1e-3, (1.0 / self.task_spread) - 1.0)
        alpha = max(1e-3, mean_p * concentration)
        beta = max(1e-3, (1.0 - mean_p) * concentration)
        return rng.beta(alpha, beta, self.design.n_tasks)

    def draw(
        self, rng: np.random.Generator
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, str]]:
        """Draw one synthetic experiment of binary outcomes."""
        design = self.design
        clusters = design.cluster_ids
        base_p = self.task_probabilities(rng, self.baseline_p)
        cand_p = np.clip(base_p + self.effect, 0.0, 1.0)

        baseline: dict[str, list[float]] = {}
        candidate: dict[str, list[float]] = {}
        cluster_map: dict[str, str] = {}
        for index in range(design.n_tasks):
            task_id = f"t{index:03d}"
            cluster_map[task_id] = clusters[index]
            baseline[task_id] = rng.binomial(1, base_p[index], design.k_reps).astype(float).tolist()
            candidate[task_id] = (
                rng.binomial(1, cand_p[index], design.k_reps).astype(float).tolist()
            )
        return baseline, candidate, cluster_map


def task_level(scores: dict[str, list[float]]) -> list[float]:
    """Per-task means in task-id order — the analysis unit."""
    return [float(np.mean(values)) for _, values in sorted(scores.items())]


def paired_differences(
    baseline: dict[str, list[float]], candidate: dict[str, list[float]]
) -> list[float]:
    """Per-task ``candidate - baseline`` differences in task-id order."""
    base = task_level(baseline)
    cand = task_level(candidate)
    return [c - b for b, c in zip(base, cand, strict=True)]


def wilson_covers(successes: int, trials: int, truth: float, *, level: float = 0.95) -> bool:
    """Whether a Wilson interval covers the true proportion."""
    from agentgate.stats import wilson_interval

    estimate = wilson_interval(successes, trials, level=level)
    assert estimate.ci_low is not None and estimate.ci_high is not None
    return estimate.ci_low <= truth <= estimate.ci_high


def covers(estimate: object, truth: float) -> bool:
    """Whether an :class:`Estimate` covers ``truth``."""
    low = getattr(estimate, "ci_low", None)
    high = getattr(estimate, "ci_high", None)
    if low is None or high is None:
        return False
    return bool(low <= truth <= high)
