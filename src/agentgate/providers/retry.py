"""Exponential backoff with jitter (A3.6).

Full jitter (``delay ~ U(0, min(cap, base * 2**attempt))``) is used rather than plain
exponential backoff because synchronised retries from a concurrent runner are exactly how a
free tier turns one 429 into a storm of them.

The sleeper and RNG are injectable, so the backoff *schedule* can be asserted in tests without
any test actually sleeping.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from agentgate.errors import ProviderError, RateLimitError

Sleeper = Callable[[float], Awaitable[None]]
JitterMode = Literal["full", "equal", "none"]


@dataclass(slots=True)
class RetryPolicy:
    """How hard to retry a transient provider failure."""

    max_attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: JitterMode = "full"
    retry_on: tuple[type[BaseException], ...] = field(default=(ProviderError,))

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        """Return True when ``exc`` on attempt ``attempt`` (0-based) is worth another try."""
        return attempt + 1 < self.max_attempts and isinstance(exc, self.retry_on)


def backoff_delay(
    attempt: int,
    policy: RetryPolicy,
    rng: random.Random,
    *,
    retry_after: float | None = None,
) -> float:
    """Compute the delay before retry number ``attempt`` (0-based).

    Args:
        attempt: Index of the attempt that just failed.
        policy: Backoff configuration.
        rng: Seeded RNG, so a test can reproduce a schedule exactly.
        retry_after: Provider-supplied hint, which overrides the computed curve.

    Returns:
        Seconds to sleep.
    """
    if retry_after is not None:
        return max(0.0, min(retry_after, policy.max_delay_s))
    ceiling = min(policy.max_delay_s, policy.base_delay_s * (2.0**attempt))
    if policy.jitter == "none":
        return ceiling
    if policy.jitter == "equal":
        return ceiling / 2.0 + rng.uniform(0.0, ceiling / 2.0)
    return rng.uniform(0.0, ceiling)


@dataclass(slots=True)
class RetryOutcome:
    """Bookkeeping about how much retrying a call needed."""

    attempts: int = 1
    retries: int = 0
    rate_limit_events: int = 0
    slept_s: float = 0.0


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    *,
    sleeper: Sleeper = asyncio.sleep,
    rng: random.Random | None = None,
    outcome: RetryOutcome | None = None,
) -> T:
    """Run ``operation``, retrying transient provider failures with jittered backoff.

    Args:
        operation: Zero-argument coroutine factory to (re)invoke.
        policy: Backoff configuration; defaults to :class:`RetryPolicy`.
        sleeper: Awaitable sleep.
        rng: Seeded RNG for jitter.
        outcome: Optional record updated in place with attempt counts.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        BaseException: The final failure, once the attempt budget is spent.
    """
    policy = policy or RetryPolicy()
    rng = rng or random.Random(0)
    record = outcome if outcome is not None else RetryOutcome()
    record.attempts = 0

    attempt = 0
    while True:
        record.attempts = attempt + 1
        try:
            return await operation()
        except BaseException as exc:
            if isinstance(exc, RateLimitError):
                record.rate_limit_events += 1
            if not policy.should_retry(exc, attempt):
                raise
            hint = exc.retry_after if isinstance(exc, RateLimitError) else None
            delay = backoff_delay(attempt, policy, rng, retry_after=hint)
            record.retries += 1
            record.slept_s += delay
            await sleeper(delay)
            attempt += 1
