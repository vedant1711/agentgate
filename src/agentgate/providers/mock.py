"""Deterministic transports for ``mock`` mode and unit tests.

``mock`` mode exists so the entire harness — agents, metrics, judges, gate — can be exercised
in CI with no network, no keys, and no quota (A3.2). Determinism is the whole point: the same
request always yields the same response, so a mock-mode run is byte-reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agentgate.providers.tokens import estimate_tokens, estimate_tokens_of
from agentgate.providers.types import ChatRequest, ChatResponse
from agentgate.schemas.trajectory import TokenUsage

Handler = Callable[[ChatRequest], ChatResponse | str]
"""A mock backend: maps a request to a full response, or to just the reply text."""


def usage_for(request: ChatRequest, reply: str) -> TokenUsage:
    """Estimate token usage for a mock exchange."""
    return TokenUsage(
        prompt_tokens=estimate_tokens_of(message.content for message in request.messages),
        completion_tokens=estimate_tokens(reply),
    )


def echo_handler(request: ChatRequest) -> str:
    """Default mock backend: a deterministic tag derived from the prompt hash."""
    return f"[mock:{request.prompt_hash()}]"


class MockTransport:
    """A transport that never touches the network.

    Args:
        handler: Maps requests to responses. Defaults to :func:`echo_handler`.
        latency_ms: Synthetic latency recorded on every response, so latency metrics have a
            stable value in mock mode rather than measuring Python function-call overhead.
        provider: Provider label recorded on responses.
    """

    name = "mock"

    def __init__(
        self,
        handler: Handler | None = None,
        *,
        latency_ms: float = 5.0,
        provider: str = "mock",
    ) -> None:
        self._handler = handler or echo_handler
        self._latency_ms = latency_ms
        self._provider = provider
        self.calls: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Return the handler's deterministic response for ``request``."""
        self.calls.append(request)
        produced = self._handler(request)
        if isinstance(produced, ChatResponse):
            return produced.model_copy(
                update={
                    "model": produced.model or request.model,
                    "provider": produced.provider or self._provider,
                    "latency_ms": produced.latency_ms or self._latency_ms,
                    "usage": produced.usage
                    if produced.usage.total_tokens
                    else usage_for(request, produced.text),
                }
            )
        return ChatResponse(
            text=produced,
            model=request.model,
            provider=self._provider,
            latency_ms=self._latency_ms,
            usage=usage_for(request, produced),
            response_id=f"mock-{request.prompt_hash()}",
        )


class ScriptedTransport:
    """Returns a fixed sequence of replies, one per call.

    Useful for driving a reference agent through an exact tool-calling path in tests.

    Args:
        replies: Responses (or reply strings) to return in order.
        loop: Restart the sequence when exhausted instead of raising.
        latency_ms: Synthetic latency per reply.
    """

    name = "scripted"

    def __init__(
        self,
        replies: Sequence[ChatResponse | str],
        *,
        loop: bool = False,
        latency_ms: float = 5.0,
    ) -> None:
        self._replies = list(replies)
        self._loop = loop
        self._latency_ms = latency_ms
        self._index = 0
        self.calls: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Return the next scripted reply.

        Raises:
            IndexError: When the script is exhausted and ``loop`` is False.
        """
        self.calls.append(request)
        if self._index >= len(self._replies):
            if not self._loop:
                msg = (
                    f"scripted transport exhausted after {len(self._replies)} replies; "
                    f"the agent asked for another completion"
                )
                raise IndexError(msg)
            self._index = 0
        reply = self._replies[self._index]
        self._index += 1
        if isinstance(reply, ChatResponse):
            return reply.model_copy(
                update={
                    "model": reply.model or request.model,
                    "provider": reply.provider or self.name,
                    "latency_ms": reply.latency_ms or self._latency_ms,
                    "usage": reply.usage
                    if reply.usage.total_tokens
                    else usage_for(request, reply.text),
                }
            )
        return ChatResponse(
            text=reply,
            model=request.model,
            provider=self.name,
            latency_ms=self._latency_ms,
            usage=usage_for(request, reply),
        )

    @property
    def remaining(self) -> int:
        """Replies left in the script."""
        return max(0, len(self._replies) - self._index)
