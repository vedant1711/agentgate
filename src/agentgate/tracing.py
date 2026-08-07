"""OpenTelemetry tracing with OpenInference semantic conventions (Phase 2).

Tracing is strictly optional and off by default: the JSONL trajectory is the source of truth
for every metric, and spans exist for *humans* debugging a failing task in Phoenix. Making the
harness depend on a collector would violate the zero-paid-resources constraint and make CI
flaky for no analytical gain.

Enable with ``AGENTGATE_TRACING=1`` and the ``tracing`` extra installed; point it at a
self-hosted Phoenix with ``OTEL_EXPORTER_OTLP_ENDPOINT`` (see ``docker-compose.phoenix.yml``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final

# --- OpenInference semantic conventions ------------------------------------
SPAN_KIND: Final = "openinference.span.kind"
KIND_AGENT: Final = "AGENT"
KIND_LLM: Final = "LLM"
KIND_TOOL: Final = "TOOL"
KIND_RETRIEVER: Final = "RETRIEVER"
KIND_CHAIN: Final = "CHAIN"

LLM_MODEL_NAME: Final = "llm.model_name"
LLM_TEMPERATURE: Final = "llm.invocation_parameters.temperature"
LLM_PROMPT_TOKENS: Final = "llm.token_count.prompt"
LLM_COMPLETION_TOKENS: Final = "llm.token_count.completion"
LLM_TOTAL_TOKENS: Final = "llm.token_count.total"
TOOL_NAME: Final = "tool.name"
TOOL_PARAMETERS: Final = "tool.parameters"
INPUT_VALUE: Final = "input.value"
OUTPUT_VALUE: Final = "output.value"
RETRIEVAL_DOCUMENTS: Final = "retrieval.documents"

AGENTGATE_TASK_ID: Final = "agentgate.task_id"
AGENTGATE_REP: Final = "agentgate.rep"
AGENTGATE_SYSTEM: Final = "agentgate.system"
AGENTGATE_SEED: Final = "agentgate.seed"

_ENV_FLAG: Final = "AGENTGATE_TRACING"


def tracing_enabled() -> bool:
    """Return True when tracing is switched on and the SDK is importable."""
    if os.environ.get(_ENV_FLAG, "").strip().lower() in ("", "0", "false", "no", "off"):
        return False
    try:
        import opentelemetry.trace  # noqa: F401
    except ImportError:
        return False
    return True


class _NoOpSpan:
    """Stands in for a span when tracing is off. Records nothing, costs nothing."""

    span_id: str | None = None

    def set_attribute(self, key: str, value: object) -> None:
        """Discard an attribute."""

    def record_error(self, error: str) -> None:
        """Discard an error."""


class _OtelSpan:
    """Thin adapter over a real OpenTelemetry span."""

    def __init__(self, span: Any) -> None:
        self._span = span
        context = span.get_span_context()
        self.span_id: str | None = format(context.span_id, "016x") if context else None

    def set_attribute(self, key: str, value: object) -> None:
        """Set an attribute, coercing unsupported types to their repr."""
        if isinstance(value, str | bool | int | float):
            self._span.set_attribute(key, value)
        else:
            self._span.set_attribute(key, repr(value))

    def record_error(self, error: str) -> None:
        """Mark the span as failed."""
        self._span.set_attribute("error.message", error)
        self._span.set_status(_status_error(error))


def _status_error(message: str) -> Any:
    from opentelemetry.trace import Status, StatusCode

    return Status(StatusCode.ERROR, message)


@contextmanager
def span(name: str, attributes: Mapping[str, object] | None = None) -> Iterator[Any]:
    """Open a span, or a no-op stand-in when tracing is disabled.

    Args:
        name: Span name.
        attributes: Initial attributes, typically including :data:`SPAN_KIND`.

    Yields:
        An object exposing ``set_attribute`` / ``record_error`` and a ``span_id``.
    """
    if not tracing_enabled():
        yield _NoOpSpan()
        return
    from opentelemetry import trace

    tracer = trace.get_tracer("agentgate")
    with tracer.start_as_current_span(name) as raw:
        wrapper = _OtelSpan(raw)
        for key, value in (attributes or {}).items():
            wrapper.set_attribute(key, value)
        yield wrapper


def configure_tracing(service_name: str = "agentgate") -> bool:
    """Install an OTLP exporter if tracing is enabled and the SDK is available.

    Args:
        service_name: Resource service name reported to the collector.

    Returns:
        True when a real exporter was installed.
    """
    if not tracing_enabled():
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover - requires the optional extra
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return True
