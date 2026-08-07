"""Typed exceptions.

Every failure mode the harness can hit deliberately has a class here, so callers (and the CLI's
exit-code mapping) never have to string-match on messages.
"""

from __future__ import annotations


class AgentGateError(Exception):
    """Base class for every AgentGate error."""


class ConfigError(AgentGateError):
    """Malformed suite, policy, or lockfile."""


class SuiteMismatchError(AgentGateError):
    """Two runs cannot be paired (C2 refuses rather than warns)."""


class CacheMissError(AgentGateError):
    """``replay`` mode hit a request that is not in the cache."""


class BudgetExceededError(AgentGateError):
    """A run-level request/token/cost/wall cap was reached (A3.6)."""


class ProviderError(AgentGateError):
    """The underlying LLM provider failed after the retry budget was exhausted."""


class RateLimitError(ProviderError):
    """The provider returned 429 / throttled the request.

    Args:
        message: Human-readable detail.
        retry_after: Seconds the provider asked us to wait, when it said so. The retry policy
            honours this instead of its own backoff curve, because the provider knows better.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class JudgeError(AgentGateError):
    """The judge produced output that could not be parsed after retries (H1)."""


class JudgeIndependenceError(AgentGateError):
    """Judge and agent share a model family without an explicit override (D3)."""


class MetricRequirementError(AgentGateError):
    """A metric was asked to score a sample lacking its declared requirements."""


class InsufficientDataError(AgentGateError):
    """Not enough paired observations to compute the requested statistic."""


class GateBlockedError(AgentGateError):
    """The gate rendered a failing verdict (used when embedding the gate in scripts)."""
