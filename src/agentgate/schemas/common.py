"""Shared enums, base models, and canonical-hashing helpers.

Every schema in AgentGate derives from :class:`AgentGateModel` so that two runs with the
same logical content always produce the same bytes (constraint A3.4: reproducibility is a
feature). Canonical hashing is done over ``mode="json"`` dumps with sorted keys, which makes
hashes stable across Python versions and dict-insertion orders.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION = 1
"""Version of the on-disk schema family. Bumped on any breaking field change."""


class MetricFamily(StrEnum):
    """Metric families from Part B of the specification."""

    OUTCOME = "outcome"
    TRAJECTORY = "trajectory"
    RAG = "rag"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"


class Requirement(StrEnum):
    """Inputs a metric needs before it can legally score a sample.

    The metrics engine skips (rather than zero-scores) any metric whose requirements a task
    does not satisfy; silently scoring 0 for a missing reference would bias suite means.
    """

    REFERENCE_ANSWER = "reference_answer"
    REFERENCE_TRAJECTORY = "reference_trajectory"
    GOAL_STATE = "goal_state"
    CONTEXTS = "contexts"
    JUDGE = "judge"
    EMBEDDINGS = "embeddings"
    SANDBOX_EVENTS = "sandbox_events"
    OUTPUT_SCHEMA = "output_schema"


Direction = Literal["higher_is_better", "lower_is_better"]
"""Which way is better for a metric. Statistics are direction-normalised before gating."""

DType = Literal["binary", "proportion", "continuous", "count"]
"""Measurement type. Determines which statistical machinery (Part C) is legal for a metric."""


class ProviderMode(StrEnum):
    """Execution mode for the provider layer (A5).

    ``live``   — call the provider, no cache writes.
    ``cache``  — call the provider, write through to the SQLite cache.
    ``replay`` — cache only; a miss is a hard error. CI default.
    ``mock``   — canned deterministic fixtures; used by unit tests, never touches the network.
    """

    LIVE = "live"
    CACHE = "cache"
    REPLAY = "replay"
    MOCK = "mock"


class Verdict(StrEnum):
    """Per-metric or overall gate outcome (C3)."""

    PASS = "PASS"
    REGRESSION = "REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNDERPOWERED = "UNDERPOWERED"
    SAFETY_FAIL = "SAFETY_FAIL"
    SKIPPED = "SKIPPED"


class AgentGateModel(BaseModel):
    """Base model: strict field validation plus canonical content hashing."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        ser_json_inf_nan="constants",
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        """Allow a model's own serialized output to be re-validated.

        ``extra="forbid"`` would otherwise reject computed fields (``config_hash`` and friends)
        on the way back in, which would break every round trip through JSON.
        """
        if isinstance(data, dict) and cls.model_computed_fields:
            computed = set(cls.model_computed_fields)
            if computed & data.keys():
                return {key: value for key, value in data.items() if key not in computed}
        return data

    def canonical_json(self) -> str:
        """Return a deterministic JSON encoding (sorted keys, no incidental whitespace)."""
        return canonical_dumps(self.model_dump(mode="json"))

    def content_digest(self) -> str:
        """Return the sha256 of :meth:`canonical_json`, truncated to 16 hex chars.

        Named ``content_digest`` rather than ``content_hash`` so that models are free to carry a
        persisted ``content_hash`` *field* without shadowing this helper.
        """
        return sha256_hex(self.canonical_json())[:16]

    @classmethod
    def model_validate_yaml(cls, text: str) -> Self:
        """Parse a YAML document into this model.

        Args:
            text: YAML source.

        Returns:
            The validated model instance.
        """
        import yaml

        payload: Any = yaml.safe_load(text)
        return cls.model_validate(payload)


class FrozenModel(AgentGateModel):
    """Immutable variant used for anything that participates in a content hash."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_inf_nan="constants",
    )


def canonical_dumps(payload: Any) -> str:
    """Serialise ``payload`` to canonical JSON (sorted keys, compact separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """Return the hex sha256 digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any, *, length: int = 16) -> str:
    """Hash arbitrary JSON-able data deterministically.

    Args:
        payload: Any JSON-serialisable structure.
        length: Number of leading hex characters to keep.

    Returns:
        Truncated hex digest.
    """
    return sha256_hex(canonical_dumps(payload))[:length]
