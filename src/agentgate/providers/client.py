"""The provider-agnostic LLM client (L1).

One object composes the four execution modes, the SQLite cache, per-provider rate limiting,
retry-with-jitter, and the run budget. Everything above this layer — agents, judges — sees only
:meth:`LLMClient.complete`, which is what makes the agent model, judge model, and embedding
model independently swappable (A3.2).

Mode semantics (A5):

===========  =========================================================================
``live``     Always call the provider. Never reads or writes the cache.
``cache``    Read the cache; on a miss call the provider and write through.
``replay``   Read the cache only. A miss raises — CI must never silently spend quota.
``mock``     Deterministic fixtures. No network, no cache, no keys.
===========  =========================================================================
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from agentgate.errors import CacheMissError, RateLimitError
from agentgate.providers.budget import BudgetTracker
from agentgate.providers.cache import ResponseCache
from agentgate.providers.catalog import estimate_cost
from agentgate.providers.mock import MockTransport
from agentgate.providers.ratelimit import RateLimiterRegistry
from agentgate.providers.retry import RetryOutcome, RetryPolicy, with_retry
from agentgate.providers.types import ChatRequest, ChatResponse, ClientStats, Transport
from agentgate.schemas.common import ProviderMode
from agentgate.schemas.results import BudgetSpec

DEFAULT_CACHE_PATH = ".agentgate/cache.sqlite"


@dataclass(slots=True)
class ClientConfig:
    """Everything that governs how requests reach a provider."""

    mode: ProviderMode = ProviderMode.REPLAY
    cache_path: str | Path = DEFAULT_CACHE_PATH
    namespace: str = ""
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    degrade_to_replay_on_throttle: bool = True
    """E2 quota safety: once the retry budget is spent on 429s, stop calling the provider."""
    rng_seed: int = 0


class LLMClient:
    """Executes chat completions under a mode, a cache, a rate limit, and a budget.

    Args:
        config: Mode and policy configuration.
        transport: Backend to call in ``live``/``cache``/``mock`` modes. Defaults to
            :class:`~agentgate.providers.mock.MockTransport` in mock mode and LiteLLM otherwise.
        cache: Response cache. Defaults to a SQLite cache at ``config.cache_path``.
        limiter: Per-provider token buckets.
        budget: Run budget tracker.
        sleeper: Awaitable sleep, injected so tests never wait.
        clock: Monotonic time source.
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        transport: Transport | None = None,
        cache: ResponseCache | None = None,
        limiter: RateLimiterRegistry | None = None,
        budget: BudgetTracker | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or ClientConfig()
        self._mode = self.config.mode
        self._transport = transport if transport is not None else self._default_transport()
        self._cache = cache if cache is not None else self._default_cache()
        self._limiter = limiter or RateLimiterRegistry(sleeper=sleeper, clock=clock)
        self._budget = budget or BudgetTracker(self.config.budget, clock=clock)
        self._sleeper = sleeper
        self._rng = random.Random(self.config.rng_seed)

        self._requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._retries = 0
        self._rate_limit_events = 0
        self._degraded = False

    # -- construction helpers ---------------------------------------------

    def _default_transport(self) -> Transport:
        if self.config.mode is ProviderMode.MOCK:
            return MockTransport()
        if self.config.mode is ProviderMode.REPLAY:
            return _RefusingTransport()
        from agentgate.providers.litellm_transport import LiteLLMTransport

        return LiteLLMTransport()

    def _default_cache(self) -> ResponseCache:
        path = ":memory:" if self.config.mode is ProviderMode.MOCK else self.config.cache_path
        return ResponseCache(path, namespace=self.config.namespace)

    @classmethod
    def mock(
        cls,
        transport: Transport | None = None,
        *,
        budget: BudgetTracker | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> LLMClient:
        """Build a mock-mode client — the constructor unit tests and agents use."""
        config = ClientConfig(mode=ProviderMode.MOCK, cache_path=":memory:")
        return cls(
            config,
            transport=transport or MockTransport(),
            budget=budget,
            sleeper=sleeper,
            clock=clock,
        )

    # -- accessors ---------------------------------------------------------

    @property
    def mode(self) -> ProviderMode:
        """Current mode, which may have degraded to ``replay`` after a 429 storm."""
        return self._mode

    @property
    def cache(self) -> ResponseCache:
        """The underlying response cache."""
        return self._cache

    @property
    def budget(self) -> BudgetTracker:
        """The run budget tracker."""
        return self._budget

    @property
    def degraded(self) -> bool:
        """True once throttling forced the client into replay-only operation."""
        return self._degraded

    @property
    def stats(self) -> ClientStats:
        """Snapshot of provider bookkeeping for the run summary."""
        return ClientStats(
            requests=self._requests,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            retries=self._retries,
            rate_limit_events=self._rate_limit_events,
            prompt_tokens=self._budget.usage.prompt_tokens,
            completion_tokens=self._budget.usage.completion_tokens,
            cost_usd=self._budget.cost_usd,
            degraded_to_replay=self._degraded,
        )

    # -- the one method everything above this layer uses -------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Execute ``request`` under the configured mode.

        Args:
            request: The completion request.

        Returns:
            A normalised response. ``cached`` is True when it came from the cache.

        Raises:
            CacheMissError: In ``replay`` mode when the request was never recorded.
            BudgetExceededError: When a run-level cap would be breached.
            ProviderError: When the provider failed after the retry budget was spent.
        """
        if self._mode is ProviderMode.MOCK:
            return await self._call_provider(request)

        if self._mode in (ProviderMode.CACHE, ProviderMode.REPLAY):
            hit = self._cache.get(request)
            if hit is not None:
                self._cache_hits += 1
                return hit
            self._cache_misses += 1

        if self._mode is ProviderMode.REPLAY:
            raise self._cache_miss_error(request)

        try:
            response = await self._call_provider(request)
        except RateLimitError:
            if self.config.degrade_to_replay_on_throttle:
                self._degrade()
            raise

        if self._mode is ProviderMode.CACHE:
            self._cache.put(request, response)
        return response

    async def _call_provider(self, request: ChatRequest) -> ChatResponse:
        """Rate-limit, budget-check, and issue one request with retries."""
        self._budget.check()
        await self._limiter.acquire(request.model)
        outcome = RetryOutcome()
        semaphore = self._limiter.semaphore_for(request.model)
        try:
            async with semaphore:
                response = await with_retry(
                    lambda: self._transport.complete(request),
                    self.config.retry,
                    sleeper=self._sleeper,
                    rng=self._rng,
                    outcome=outcome,
                )
        finally:
            self._retries += outcome.retries
            self._rate_limit_events += outcome.rate_limit_events
        self._requests += 1
        self._budget.record(
            usage=response.usage, cost_usd=estimate_cost(request.model, response.usage)
        )
        return response

    def _degrade(self) -> None:
        """Stop calling the provider for the rest of the run (E2 quota safety)."""
        self._degraded = True
        self._mode = ProviderMode.REPLAY

    def _cache_miss_error(self, request: ChatRequest) -> CacheMissError:
        reason = (
            "provider throttling forced this run into replay mode"
            if self._degraded
            else "replay mode reads the cache only"
        )
        return CacheMissError(
            f"cache miss for model={request.model} prompt={request.prompt_hash()} "
            f"key={self._cache.key_for(request)[:16]}: {reason}. "
            f"Record it first with `agentgate run --mode cache`, or ship the fixture cache."
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release the cache connection."""
        self._cache.close()

    def __enter__(self) -> Self:
        """Enter a context manager that closes the client on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the client."""
        self.close()


class _RefusingTransport:
    """Placeholder transport for replay mode: reaching it at all is a bug."""

    name = "refusing"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Always raise — replay mode must never reach a network transport."""
        msg = (
            f"replay mode attempted a live call to {request.model}. "
            f"This is a bug in the runner, not a cache problem."
        )
        raise CacheMissError(msg)
