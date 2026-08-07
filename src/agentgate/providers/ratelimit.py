"""Token-bucket rate limiting and per-provider concurrency caps (A3.6).

Free tiers throttle aggressively, and a throttled CI run is a red build for a reason that has
nothing to do with the agent under test. Shaping traffic locally is cheaper than discovering
the limit remotely.

The clock and sleeper are injectable so tests exercise the arithmetic without wall-clock waits.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from agentgate.providers.catalog import ProviderLimits, limits_for, provider_of

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class TokenBucket:
    """A classic token bucket.

    Args:
        rate_per_sec: Steady-state refill rate.
        burst: Maximum tokens the bucket holds, i.e. the allowed burst size.
        clock: Monotonic time source.
        sleeper: Awaitable sleep, injected for tests.
    """

    def __init__(
        self,
        rate_per_sec: float,
        burst: float,
        *,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        if rate_per_sec <= 0:
            msg = "rate_per_sec must be positive"
            raise ValueError(msg)
        self.rate_per_sec = rate_per_sec
        self.burst = max(burst, 1.0)
        self._clock = clock
        self._sleeper = sleeper
        self._tokens = self.burst
        self._updated_at = clock()
        self._lock = asyncio.Lock()
        self.total_wait_s = 0.0

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated_at)
        self._updated_at = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_sec)

    @property
    def available(self) -> float:
        """Tokens currently in the bucket (refilled to now)."""
        self._refill()
        return self._tokens

    async def acquire(self, tokens: float = 1.0) -> float:
        """Consume ``tokens``, sleeping until the bucket can supply them.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            Seconds spent waiting.
        """
        async with self._lock:
            waited = 0.0
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.total_wait_s += waited
                    return waited
                deficit = tokens - self._tokens
                delay = deficit / self.rate_per_sec
                await self._sleeper(delay)
                waited += delay


class RateLimiterRegistry:
    """Per-provider token buckets plus concurrency semaphores.

    Args:
        overrides: Provider slug to explicit limits, overriding the catalog defaults.
        clock: Monotonic time source.
        sleeper: Awaitable sleep.
    """

    def __init__(
        self,
        overrides: dict[str, ProviderLimits] | None = None,
        *,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._overrides = overrides or {}
        self._clock = clock
        self._sleeper = sleeper
        self._buckets: dict[str, TokenBucket] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def limits_for_model(self, model: str) -> ProviderLimits:
        """Return the limits governing ``model``."""
        provider = provider_of(model)
        return self._overrides.get(provider, limits_for(model))

    def bucket_for(self, model: str) -> TokenBucket:
        """Return (creating if needed) the token bucket for ``model``'s provider."""
        provider = provider_of(model)
        bucket = self._buckets.get(provider)
        if bucket is None:
            limits = self.limits_for_model(model)
            bucket = TokenBucket(
                rate_per_sec=limits.requests_per_minute / 60.0,
                burst=float(limits.burst),
                clock=self._clock,
                sleeper=self._sleeper,
            )
            self._buckets[provider] = bucket
        return bucket

    def semaphore_for(self, model: str) -> asyncio.Semaphore:
        """Return (creating if needed) the concurrency semaphore for ``model``'s provider."""
        provider = provider_of(model)
        semaphore = self._semaphores.get(provider)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.limits_for_model(model).max_concurrency)
            self._semaphores[provider] = semaphore
        return semaphore

    async def acquire(self, model: str) -> float:
        """Wait for permission to issue one request against ``model``.

        Args:
            model: Model id being called.

        Returns:
            Seconds spent waiting on the bucket.
        """
        return await self.bucket_for(model).acquire()

    @property
    def total_wait_s(self) -> float:
        """Total time this run spent shaped by rate limits."""
        return sum(bucket.total_wait_s for bucket in self._buckets.values())
