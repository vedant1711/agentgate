"""Provider-layer tests (Phase 1 acceptance).

Covers the four claims the phase must prove: cache-hit determinism, replay strictness, clean
budget halt, and survival of a 429 storm without data loss.
"""

from __future__ import annotations

import random
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from agentgate.errors import BudgetExceededError, CacheMissError, ProviderError, RateLimitError
from agentgate.providers import (
    BudgetTracker,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ClientConfig,
    LLMClient,
    MockTransport,
    RateLimiterRegistry,
    ResponseCache,
    RetryOutcome,
    RetryPolicy,
    ScriptedTransport,
    TokenBucket,
    ToolSpec,
    backoff_delay,
    estimate_cost,
    estimate_tokens,
    price_for,
    provider_of,
    with_retry,
)
from agentgate.providers.catalog import ollama_available
from agentgate.providers.litellm_transport import (
    LiteLLMTransport,
    _normalise_error,
    _parse_arguments,
)
from agentgate.schemas.common import ProviderMode
from agentgate.schemas.results import BudgetSpec
from agentgate.schemas.trajectory import TokenUsage

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay

    @property
    def total_slept(self) -> float:
        return sum(self.slept)


class ThrottlingTransport:
    """Raises 429 for the first ``failures`` calls, then serves the real reply."""

    name = "throttling"

    def __init__(self, failures: int, *, retry_after: float | None = None) -> None:
        self.failures = failures
        self.retry_after = retry_after
        self.attempts = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RateLimitError(
                f"429 rate limit exceeded (attempt {self.attempts})", retry_after=self.retry_after
            )
        return ChatResponse(
            text="served after throttling",
            model=request.model,
            provider="throttling",
            latency_ms=42.0,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )


class BrokenTransport:
    """Always fails with a non-retryable-looking provider error."""

    name = "broken"

    def __init__(self) -> None:
        self.attempts = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.attempts += 1
        msg = f"upstream exploded for {request.model}"
        raise ProviderError(msg)


def make_request(prompt: str = "hello", *, model: str = "groq/llama-3.3-70b") -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        seed=7,
    )


def fast_client(
    transport: object,
    *,
    mode: ProviderMode = ProviderMode.CACHE,
    cache: ResponseCache | None = None,
    budget: BudgetSpec | None = None,
    retry: RetryPolicy | None = None,
    clock: FakeClock | None = None,
    degrade: bool = True,
) -> tuple[LLMClient, FakeClock]:
    """Build a client whose sleeps are instant and whose clock is deterministic."""
    fake = clock or FakeClock()
    config = ClientConfig(
        mode=mode,
        cache_path=":memory:",
        budget=budget or BudgetSpec(),
        retry=retry or RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=30.0),
        degrade_to_replay_on_throttle=degrade,
        rng_seed=1234,
    )
    client = LLMClient(
        config,
        transport=transport,  # type: ignore[arg-type]
        cache=cache if cache is not None else ResponseCache(":memory:"),
        limiter=RateLimiterRegistry(sleeper=fake.sleep, clock=fake),
        budget=BudgetTracker(config.budget, clock=fake),
        sleeper=fake.sleep,
        clock=fake,
    )
    return client, fake


# ---------------------------------------------------------------------------
# Cache determinism (acceptance 1)
# ---------------------------------------------------------------------------


def test_cache_key_is_stable_across_equal_requests() -> None:
    assert make_request().cache_key() == make_request().cache_key()


@pytest.mark.parametrize(
    "mutation",
    [
        {"temperature": 0.7},
        {"seed": 8},
        {"model": "groq/other"},
        {"max_tokens": 64},
        {"stop": ["\n"]},
        {"extra": {"top_p": 0.9}},
    ],
)
def test_cache_key_changes_with_any_request_parameter(mutation: dict[str, object]) -> None:
    base = make_request()
    assert base.model_copy(update=mutation).cache_key() != base.cache_key()


def test_cache_key_changes_with_tools() -> None:
    base = make_request()
    with_tools = base.model_copy(update={"tools": [ToolSpec(name="refund")]})
    assert with_tools.cache_key() != base.cache_key()


def test_cache_key_is_namespaced() -> None:
    request = make_request()
    assert request.cache_key("v2") != request.cache_key("v1")


async def test_cache_hit_is_deterministic_and_replays_recorded_latency() -> None:
    transport = MockTransport(lambda _: "canned reply", latency_ms=123.0)
    client, _ = fast_client(transport, mode=ProviderMode.CACHE)
    request = make_request()

    first = await client.complete(request)
    second = await client.complete(request)
    third = await client.complete(request)

    assert transport.calls == [request], "cache must serve repeats without calling the provider"
    assert first.cached is False
    assert second.cached is True and third.cached is True
    assert second.text == first.text == "canned reply"
    assert second.latency_ms == pytest.approx(123.0), "recorded latency must be replayed verbatim"
    assert second.model_dump(exclude={"cached"}) == third.model_dump(exclude={"cached"})
    assert client.stats.cache_hits == 2
    assert client.stats.cache_misses == 1


async def test_cache_survives_a_process_restart(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    request = make_request()

    warm_cache = ResponseCache(path)
    client, _ = fast_client(
        MockTransport(lambda _: "warm"), mode=ProviderMode.CACHE, cache=warm_cache
    )
    await client.complete(request)
    client.close()

    cold_cache = ResponseCache(path)
    replay, _ = fast_client(MockTransport(), mode=ProviderMode.REPLAY, cache=cold_cache)
    restored = await replay.complete(request)
    assert restored.text == "warm"
    assert restored.cached is True


def test_cache_export_import_round_trip(tmp_path: Path) -> None:
    source = ResponseCache(":memory:")
    request = make_request()
    source.put(request, ChatResponse(text="fixture", model=request.model))
    exported = source.export_jsonl(tmp_path / "fixtures.jsonl")

    target = ResponseCache(":memory:")
    imported = target.import_jsonl(tmp_path / "fixtures.jsonl")
    hit = target.get(request)

    assert exported == imported == 1
    assert hit is not None and hit.text == "fixture"
    assert target.models() == {request.model: 1}


def test_cache_put_is_idempotent_on_the_same_key() -> None:
    cache = ResponseCache(":memory:")
    request = make_request()
    cache.put(request, ChatResponse(text="first"))
    cache.put(request, ChatResponse(text="second"))
    hit = cache.get(request)
    assert len(cache) == 1
    assert hit is not None and hit.text == "second"


def test_cache_membership_and_clear() -> None:
    cache = ResponseCache(":memory:")
    request = make_request()
    assert request not in cache
    cache.put(request, ChatResponse(text="x"))
    assert request in cache
    cache.clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Replay strictness (acceptance 2)
# ---------------------------------------------------------------------------


async def test_replay_mode_raises_on_cache_miss() -> None:
    transport = MockTransport()
    client, _ = fast_client(transport, mode=ProviderMode.REPLAY)
    with pytest.raises(CacheMissError, match=re.escape("cache miss for model=groq/llama-3.3-70b")):
        await client.complete(make_request())
    assert transport.calls == [], "replay mode must never reach a transport"
    assert client.stats.cache_misses == 1


async def test_replay_mode_error_names_the_fix() -> None:
    client, _ = fast_client(MockTransport(), mode=ProviderMode.REPLAY)
    with pytest.raises(CacheMissError, match=r"agentgate run --mode cache"):
        await client.complete(make_request())


async def test_replay_mode_serves_recorded_requests() -> None:
    cache = ResponseCache(":memory:")
    request = make_request()
    cache.put(request, ChatResponse(text="recorded", latency_ms=7.5))
    client, _ = fast_client(MockTransport(), mode=ProviderMode.REPLAY, cache=cache)
    response = await client.complete(request)
    assert response.text == "recorded"
    assert response.latency_ms == pytest.approx(7.5)
    assert client.stats.hit_rate == 1.0


async def test_live_mode_never_reads_or_writes_the_cache() -> None:
    cache = ResponseCache(":memory:")
    transport = MockTransport(lambda _: "live reply")
    client, _ = fast_client(transport, mode=ProviderMode.LIVE, cache=cache)
    request = make_request()
    await client.complete(request)
    await client.complete(request)
    assert len(transport.calls) == 2
    assert len(cache) == 0


async def test_mock_mode_bypasses_the_cache_entirely() -> None:
    transport = MockTransport(lambda _: "mocked")
    client, _ = fast_client(transport, mode=ProviderMode.MOCK)
    request = make_request()
    assert (await client.complete(request)).text == "mocked"
    assert (await client.complete(request)).text == "mocked"
    assert len(transport.calls) == 2
    assert client.stats.cache_hits == 0


async def test_mock_transport_is_deterministic_by_prompt() -> None:
    client, _ = fast_client(MockTransport(), mode=ProviderMode.MOCK)
    a = await client.complete(make_request("same"))
    b = await client.complete(make_request("same"))
    c = await client.complete(make_request("different"))
    assert a.text == b.text
    assert a.text != c.text


async def test_scripted_transport_drives_an_exact_path() -> None:
    transport = ScriptedTransport(["one", "two"])
    client, _ = fast_client(transport, mode=ProviderMode.MOCK)
    assert (await client.complete(make_request())).text == "one"
    assert (await client.complete(make_request())).text == "two"
    assert transport.remaining == 0
    with pytest.raises(IndexError, match="scripted transport exhausted"):
        await client.complete(make_request())


# ---------------------------------------------------------------------------
# Budget caps (acceptance 3)
# ---------------------------------------------------------------------------


async def test_request_cap_halts_the_run_cleanly() -> None:
    transport = MockTransport(lambda _: "ok")
    client, _ = fast_client(transport, mode=ProviderMode.LIVE, budget=BudgetSpec(max_requests=2))
    await client.complete(make_request("a"))
    await client.complete(make_request("b"))

    with pytest.raises(BudgetExceededError, match="budget exhausted on requests: used 2 of 2"):
        await client.complete(make_request("c"))

    assert len(transport.calls) == 2, "the capped request must not reach the provider"
    assert client.stats.requests == 2, "accounting stays accurate after the halt"


async def test_token_cap_halts_the_run() -> None:
    client, _ = fast_client(
        MockTransport(lambda _: "a rather long reply " * 20),
        mode=ProviderMode.LIVE,
        budget=BudgetSpec(max_tokens=50),
    )
    await client.complete(make_request("a"))
    with pytest.raises(BudgetExceededError, match="on tokens"):
        await client.complete(make_request("b"))


async def test_wall_clock_cap_halts_the_run() -> None:
    clock = FakeClock()
    client, _ = fast_client(
        MockTransport(lambda _: "ok"),
        mode=ProviderMode.LIVE,
        budget=BudgetSpec(max_wall_s=10.0),
        clock=clock,
    )
    await client.complete(make_request("a"))
    clock.now += 11.0
    with pytest.raises(BudgetExceededError, match="on wall_s"):
        await client.complete(make_request("b"))


async def test_cache_hits_do_not_consume_the_request_budget() -> None:
    client, _ = fast_client(
        MockTransport(lambda _: "ok"), mode=ProviderMode.CACHE, budget=BudgetSpec(max_requests=1)
    )
    request = make_request()
    await client.complete(request)
    for _ in range(10):
        assert (await client.complete(request)).cached is True
    assert client.stats.requests == 1


def test_budget_reports_remaining_headroom() -> None:
    tracker = BudgetTracker(BudgetSpec(max_requests=10, max_tokens=1000))
    tracker.record(usage=TokenUsage(prompt_tokens=100, completion_tokens=50), cost_usd=0.0)
    remaining = tracker.remaining()
    assert remaining["requests"] == 9
    assert remaining["tokens"] == 850
    assert remaining["cost_usd"] == float("inf")
    assert not tracker.exhausted


def test_unlimited_budget_never_trips() -> None:
    tracker = BudgetTracker(BudgetSpec())
    for _ in range(1000):
        tracker.record(usage=TokenUsage(prompt_tokens=10_000, completion_tokens=10_000))
    tracker.check()
    assert not tracker.exhausted


# ---------------------------------------------------------------------------
# 429 storm (acceptance 4)
# ---------------------------------------------------------------------------


async def test_429_storm_backs_off_and_still_returns_the_data() -> None:
    transport = ThrottlingTransport(failures=3)
    client, clock = fast_client(transport, mode=ProviderMode.CACHE)

    response = await client.complete(make_request())

    assert response.text == "served after throttling", "no data lost across the retries"
    assert transport.attempts == 4
    assert client.stats.retries == 3
    assert client.stats.rate_limit_events == 3
    assert not client.degraded
    assert clock.slept and all(delay >= 0 for delay in clock.slept)
    assert clock.total_slept > 0, "backoff actually waited"


async def test_429_storm_result_is_cached_so_the_retry_cost_is_paid_once() -> None:
    transport = ThrottlingTransport(failures=2)
    client, _ = fast_client(transport, mode=ProviderMode.CACHE)
    request = make_request()
    await client.complete(request)
    again = await client.complete(request)
    assert again.cached is True
    assert transport.attempts == 3


async def test_exhausted_retry_budget_degrades_to_replay() -> None:
    transport = ThrottlingTransport(failures=99)
    client, _ = fast_client(
        transport, mode=ProviderMode.CACHE, retry=RetryPolicy(max_attempts=3, base_delay_s=0.1)
    )

    with pytest.raises(RateLimitError):
        await client.complete(make_request())

    assert client.degraded is True
    assert client.mode is ProviderMode.REPLAY
    assert client.stats.degraded_to_replay is True

    # Everything already recorded still serves; only uncached work is lost.
    client.cache.put(make_request("warm"), ChatResponse(text="from cache"))
    assert (await client.complete(make_request("warm"))).text == "from cache"
    assert transport.attempts == 3, "no further provider calls after degradation"


async def test_degradation_can_be_disabled() -> None:
    transport = ThrottlingTransport(failures=99)
    client, _ = fast_client(
        transport,
        mode=ProviderMode.CACHE,
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.1),
        degrade=False,
    )
    with pytest.raises(RateLimitError):
        await client.complete(make_request())
    assert client.degraded is False
    assert client.mode is ProviderMode.CACHE


async def test_retry_honours_provider_retry_after_hint() -> None:
    transport = ThrottlingTransport(failures=1, retry_after=4.0)
    client, clock = fast_client(transport, mode=ProviderMode.CACHE)
    await client.complete(make_request())
    assert clock.slept == [4.0], "provider's Retry-After beats our own curve"


async def test_non_rate_limit_provider_errors_also_retry_then_surface() -> None:
    transport = BrokenTransport()
    client, _ = fast_client(
        transport, mode=ProviderMode.CACHE, retry=RetryPolicy(max_attempts=3, base_delay_s=0.1)
    )
    with pytest.raises(ProviderError, match="upstream exploded"):
        await client.complete(make_request())
    assert transport.attempts == 3
    assert client.stats.rate_limit_events == 0
    assert client.degraded is False, "a plain provider error is not a quota problem"


# ---------------------------------------------------------------------------
# Backoff and rate-limit mechanics
# ---------------------------------------------------------------------------


def test_full_jitter_delays_stay_within_the_exponential_envelope() -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0, jitter="full")
    rng = random.Random(0)
    for attempt in range(6):
        ceiling = min(policy.max_delay_s, policy.base_delay_s * 2**attempt)
        for _ in range(50):
            assert 0.0 <= backoff_delay(attempt, policy, rng) <= ceiling


def test_jitter_modes_differ() -> None:
    policy_none = RetryPolicy(base_delay_s=1.0, jitter="none")
    policy_equal = RetryPolicy(base_delay_s=1.0, jitter="equal")
    rng = random.Random(0)
    assert backoff_delay(2, policy_none, rng) == pytest.approx(4.0)
    assert 2.0 <= backoff_delay(2, policy_equal, rng) <= 4.0


def test_backoff_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=3.0, jitter="none")
    assert backoff_delay(20, policy, random.Random(0)) == pytest.approx(3.0)


async def test_with_retry_records_its_own_work() -> None:
    clock = FakeClock()
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            msg = "429 slow down"
            raise RateLimitError(msg)
        return "done"

    outcome = RetryOutcome()
    result = await with_retry(
        flaky,
        RetryPolicy(max_attempts=5, base_delay_s=1.0),
        sleeper=clock.sleep,
        rng=random.Random(3),
        outcome=outcome,
    )
    assert result == "done"
    assert outcome.attempts == 3
    assert outcome.retries == 2
    assert outcome.rate_limit_events == 2
    assert outcome.slept_s == pytest.approx(clock.total_slept)


async def test_token_bucket_shapes_traffic_to_the_configured_rate() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=2.0, burst=2, clock=clock, sleeper=clock.sleep)

    assert await bucket.acquire() == 0.0
    assert await bucket.acquire() == 0.0
    third = await bucket.acquire()

    assert third == pytest.approx(0.5), "third request waits for the bucket to refill"
    assert clock.now == pytest.approx(0.5)


async def test_token_bucket_refills_over_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=1.0, burst=1, clock=clock, sleeper=clock.sleep)
    await bucket.acquire()
    clock.now += 5.0
    assert bucket.available == pytest.approx(1.0), "refill is capped at the burst size"
    assert await bucket.acquire() == 0.0


def test_token_bucket_rejects_a_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="rate_per_sec must be positive"):
        TokenBucket(rate_per_sec=0.0, burst=1)


async def test_limiter_uses_per_provider_buckets() -> None:
    clock = FakeClock()
    registry = RateLimiterRegistry(sleeper=clock.sleep, clock=clock)
    groq = registry.bucket_for("groq/llama-3.3-70b")
    gemini = registry.bucket_for("gemini/gemini-2.0-flash")
    assert groq is not gemini
    assert groq is registry.bucket_for("groq/other-model")
    assert gemini.rate_per_sec < groq.rate_per_sec, "gemini's free tier is tighter"


# ---------------------------------------------------------------------------
# Catalog: routing, prices, Ollama detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("groq/llama-3.3-70b", "groq"),
        ("gemini/gemini-2.0-flash", "gemini"),
        ("cerebras/llama3.1-8b", "cerebras"),
        ("openrouter/some/free-model", "openrouter"),
        ("ollama/qwen3:4b", "ollama"),
        ("mock/agent", "mock"),
        ("gpt-4o", "unknown"),
    ],
)
def test_provider_inference(model: str, expected: str) -> None:
    assert provider_of(model) == expected


def test_free_tier_models_cost_nothing() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    for model in ("groq/llama-3.3-70b", "gemini/gemini-2.0-flash", "ollama/qwen3:4b"):
        assert estimate_cost(model, usage) == 0.0


def test_price_table_supports_enterprise_projection() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert estimate_cost("gpt-4o", usage) == pytest.approx(12.50)
    assert estimate_cost("gpt-4o-mini", usage) == pytest.approx(0.75)


def test_longest_price_key_wins() -> None:
    assert price_for("gpt-4o-mini").prompt_per_mtok == pytest.approx(0.15)
    assert price_for("gpt-4o").prompt_per_mtok == pytest.approx(2.50)


def test_unpriced_models_report_zero_with_a_note() -> None:
    price = price_for("some/unknown-model")
    assert price.cost(TokenUsage(prompt_tokens=999)) == 0.0
    assert "unpriced" in price.note


def test_ollama_detection_is_a_boolean_and_never_raises() -> None:
    assert isinstance(ollama_available(timeout_s=0.05), bool)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("abcd", 1), ("abcde", 2), ("a" * 4000, 1000)],
)
def test_token_estimation(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


# ---------------------------------------------------------------------------
# LiteLLM mapping (no network, no litellm import)
# ---------------------------------------------------------------------------


def test_litellm_kwargs_include_tools_and_seed() -> None:
    transport = LiteLLMTransport()
    request = ChatRequest(
        model="groq/llama-3.3-70b",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[ToolSpec(name="refund", description="refund an order")],
        seed=11,
        max_tokens=256,
        stop=["<end>"],
    )
    kwargs = transport._build_kwargs(request)
    assert kwargs["model"] == "groq/llama-3.3-70b"
    assert kwargs["tools"][0]["function"]["name"] == "refund"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["seed"] == 11
    assert kwargs["max_tokens"] == 256
    assert kwargs["stop"] == ["<end>"]


def test_litellm_kwargs_point_ollama_at_the_local_daemon() -> None:
    kwargs = LiteLLMTransport()._build_kwargs(make_request(model="ollama/qwen3:4b"))
    assert kwargs["api_base"].startswith("http")


@pytest.mark.parametrize(
    "message",
    ["429 Too Many Requests", "RateLimit reached for model", "quota exceeded for project"],
)
def test_throttling_errors_are_normalised_to_rate_limit(message: str) -> None:
    assert isinstance(_normalise_error(RuntimeError(message)), RateLimitError)


def test_other_errors_are_normalised_to_provider_error() -> None:
    error = _normalise_error(RuntimeError("connection reset"))
    assert isinstance(error, ProviderError)
    assert not isinstance(error, RateLimitError)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"order_id": 42}', {"order_id": 42}),
        ({"order_id": 42}, {"order_id": 42}),
        ("", {}),
        ("not json at all", {"__unparsed__": "not json at all"}),
        ("[1, 2]", {"__value__": [1, 2]}),
    ],
)
def test_tool_argument_parsing_tolerates_model_output(
    raw: object, expected: dict[str, object]
) -> None:
    assert _parse_arguments(raw) == expected


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------


async def test_client_context_manager_closes_cleanly() -> None:
    with LLMClient.mock(MockTransport(lambda _: "ok")) as client:
        assert (await client.complete(make_request())).text == "ok"


async def test_replay_client_refuses_to_construct_a_network_transport() -> None:
    client = LLMClient(ClientConfig(mode=ProviderMode.REPLAY, cache_path=":memory:"))
    with pytest.raises(CacheMissError):
        await client.complete(make_request())
    client.close()


def test_stats_hit_rate_is_zero_before_any_lookup() -> None:
    client, _ = fast_client(MockTransport(), mode=ProviderMode.MOCK)
    assert client.stats.hit_rate == 0.0
    assert client.stats.total_tokens == 0


async def test_sleeper_injection_means_tests_never_actually_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard rail: a real asyncio.sleep in this module would make CI slow and flaky."""

    async def forbidden(_: float) -> None:
        pytest.fail("provider tests must not call asyncio.sleep")

    sleeper: Callable[[float], Awaitable[None]] = forbidden
    monkeypatch.setattr("asyncio.sleep", sleeper)
    transport = ThrottlingTransport(failures=2)
    client, _ = fast_client(transport, mode=ProviderMode.CACHE)
    assert (await client.complete(make_request())).text == "served after throttling"
