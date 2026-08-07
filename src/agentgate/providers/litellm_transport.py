"""LiteLLM-backed transport for ``live`` and ``cache`` modes (A3.2).

LiteLLM is imported lazily so that ``replay`` and ``mock`` runs — every CI job — never pay for
the import and never require the optional dependency. Provider exceptions are normalised into
:class:`~agentgate.errors.RateLimitError` / :class:`~agentgate.errors.ProviderError` so the
retry policy has one thing to reason about.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentgate.errors import ProviderError, RateLimitError
from agentgate.providers.catalog import ollama_base_url, provider_of
from agentgate.providers.tokens import estimate_tokens, estimate_tokens_of
from agentgate.providers.types import ChatRequest, ChatResponse, ToolCallRequest
from agentgate.schemas.trajectory import TokenUsage

_RATE_LIMIT_MARKERS = ("rate limit", "ratelimit", "429", "quota", "too many requests")


class LiteLLMTransport:
    """Calls a provider through LiteLLM's unified async completion API.

    Args:
        extra_kwargs: Passed straight through to ``litellm.acompletion`` (e.g. ``api_base``).
        timeout_s: Per-request timeout.
    """

    name = "litellm"

    def __init__(
        self, *, extra_kwargs: dict[str, Any] | None = None, timeout_s: float = 120.0
    ) -> None:
        self._extra = extra_kwargs or {}
        self._timeout_s = timeout_s

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Execute ``request`` and normalise the reply.

        Raises:
            RateLimitError: When the provider throttled us.
            ProviderError: For any other provider-side failure.
        """
        litellm = _import_litellm()
        kwargs = self._build_kwargs(request)
        started = time.perf_counter()
        try:
            raw = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise _normalise_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _to_response(raw, request, latency_ms)

    def _build_kwargs(self, request: ChatRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_openai() for message in request.messages],
            "temperature": request.temperature,
            "timeout": self._timeout_s,
            **self._extra,
        }
        if request.tools:
            kwargs["tools"] = [tool.to_openai() for tool in request.tools]
            kwargs["tool_choice"] = "auto"
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        if request.stop:
            kwargs["stop"] = request.stop
        if provider_of(request.model) == "ollama":
            kwargs.setdefault("api_base", ollama_base_url())
        kwargs.update(request.extra)
        return kwargs


def _import_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        msg = (
            "live/cache mode needs the optional 'live' extra: `uv sync --extra live`. "
            "CI runs in replay mode and does not require it."
        )
        raise ProviderError(msg) from exc
    litellm.drop_params = True
    litellm.suppress_debug_info = True
    return litellm


def _normalise_error(exc: Exception) -> ProviderError:
    """Map a LiteLLM/provider exception onto AgentGate's error hierarchy."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        retry_after = getattr(exc, "retry_after", None)
        return RateLimitError(str(exc), retry_after=_as_float(retry_after))
    return ProviderError(str(exc))


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_response(raw: Any, request: ChatRequest, latency_ms: float) -> ChatResponse:
    """Normalise a LiteLLM ``ModelResponse`` into a :class:`ChatResponse`."""
    choice = raw.choices[0]
    message = choice.message
    text = getattr(message, "content", None) or ""
    tool_calls = [
        ToolCallRequest(
            id=getattr(call, "id", f"call-{index}"),
            name=call.function.name,
            arguments=_parse_arguments(call.function.arguments),
        )
        for index, call in enumerate(getattr(message, "tool_calls", None) or [])
    ]
    raw_usage = getattr(raw, "usage", None)
    prompt_tokens = _int_attr(raw_usage, "prompt_tokens")
    completion_tokens = _int_attr(raw_usage, "completion_tokens")
    if prompt_tokens or completion_tokens:
        token_usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    else:
        # Some free tiers omit usage entirely; estimate rather than report a misleading zero.
        token_usage = TokenUsage(
            prompt_tokens=estimate_tokens_of(m.content for m in request.messages),
            completion_tokens=estimate_tokens(text),
        )
    return ChatResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=str(getattr(choice, "finish_reason", "stop") or "stop"),
        usage=token_usage,
        model=str(getattr(raw, "model", request.model) or request.model),
        provider=provider_of(request.model),
        latency_ms=latency_ms,
        response_id=getattr(raw, "id", None),
    )


def _int_attr(obj: object, name: str) -> int:
    """Read a non-negative integer attribute, tolerating absent or malformed values."""
    value = getattr(obj, name, None)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _parse_arguments(arguments: object) -> dict[str, Any]:
    """Parse a function-call argument blob, tolerating the malformed JSON models emit."""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"__unparsed__": arguments}
    return parsed if isinstance(parsed, dict) else {"__value__": parsed}
