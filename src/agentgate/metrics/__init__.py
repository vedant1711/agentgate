"""Metrics engine (L4): every Part B metric as a plugin behind one protocol."""

from agentgate.metrics import registry
from agentgate.metrics.base import (
    BaseMetric,
    Embedder,
    Judge,
    JudgeVerdict,
    Metric,
    MetricContext,
    Scored,
    ScoredSample,
)
from agentgate.metrics.checkers import CHECKERS, normalize_answer, run_checker, subset_match
from agentgate.metrics.embeddings import HashingEmbedder, get_embedder
from agentgate.metrics.engine import (
    MetricsEngine,
    ScoringConfig,
    group_by_metric,
    score_run,
    scored_metrics,
)
from agentgate.metrics.lexical_judge import LexicalJudge
from agentgate.metrics.matching import TrajectoryMatch, cosine, match_trajectory
from agentgate.metrics.trajectory import SingleToolUse

__all__ = [
    "CHECKERS",
    "BaseMetric",
    "Embedder",
    "HashingEmbedder",
    "Judge",
    "JudgeVerdict",
    "LexicalJudge",
    "Metric",
    "MetricContext",
    "MetricsEngine",
    "Scored",
    "ScoredSample",
    "ScoringConfig",
    "SingleToolUse",
    "TrajectoryMatch",
    "cosine",
    "get_embedder",
    "group_by_metric",
    "match_trajectory",
    "normalize_answer",
    "registry",
    "run_checker",
    "score_run",
    "scored_metrics",
    "subset_match",
]
