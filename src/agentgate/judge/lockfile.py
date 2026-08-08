"""``agentgate.lock`` — the pinned evaluation instrument (D4).

Judge model, judge version, temperature, J, rubric hash, anchor hash, and embedder. Anything in
here that changes changes what a score *means*, so the lockfile is the boundary between "the
agent got worse" and "the ruler got shorter".

The lockfile does not block a run. It reports, precisely, which part of the instrument moved —
which is what a reader needs in order to know whether a trend line is still one line.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import Field

from agentgate.judge.rubrics import rubrics_hash
from agentgate.schemas.common import SCHEMA_VERSION, FrozenModel

DEFAULT_LOCK_PATH = Path("agentgate.lock")


class JudgeLock(FrozenModel):
    """Everything about the measuring instrument that must not change silently."""

    schema_version: int = SCHEMA_VERSION
    judge_model: str = ""
    judge_version: str = Field(
        default="", description="Provider-reported version, when the provider reports one."
    )
    temperature: float = 0.3
    samples: int = Field(default=3, ge=1)
    rubrics_hash: str = ""
    anchor_hash: str = ""
    embedder: str = "hashing-bow"
    recorded_at: datetime | None = None
    note: str = ""

    @classmethod
    def current(
        cls,
        *,
        judge_model: str,
        temperature: float = 0.3,
        samples: int = 3,
        anchor_hash: str = "",
        embedder: str = "hashing-bow",
        judge_version: str = "",
    ) -> JudgeLock:
        """Snapshot the instrument as currently configured."""
        return cls(
            judge_model=judge_model,
            judge_version=judge_version,
            temperature=temperature,
            samples=samples,
            rubrics_hash=rubrics_hash(),
            anchor_hash=anchor_hash,
            embedder=embedder,
            recorded_at=datetime.now(UTC),
        )

    def differences(self, other: JudgeLock) -> list[str]:
        """Describe how another lock differs from this one.

        Args:
            other: The lock to compare against.

        Returns:
            One human-readable line per changed field, empty when the instrument is unchanged.
            ``recorded_at`` and ``note`` are ignored — they are bookkeeping, not measurement.
        """
        changes: list[str] = []
        fields = {
            "judge_model": "judge model",
            "judge_version": "judge version",
            "temperature": "judge temperature",
            "samples": "judge samples (J)",
            "rubrics_hash": "rubric definitions",
            "anchor_hash": "anchor set",
            "embedder": "embedder",
        }
        for field_name, label in fields.items():
            mine = getattr(self, field_name)
            theirs = getattr(other, field_name)
            if mine != theirs:
                changes.append(f"{label}: {mine!r} -> {theirs!r}")
        return changes

    def is_compatible_with(self, other: JudgeLock) -> tuple[bool, list[str]]:
        """Whether history recorded under ``self`` is comparable to runs under ``other``."""
        changes = self.differences(other)
        return not changes, changes

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LOCK_PATH) -> JudgeLock | None:
        """Read the lockfile, or ``None`` when there is none yet."""
        source = Path(path)
        if not source.exists():
            return None
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload.get("judge", payload))

    def save(self, path: str | Path = DEFAULT_LOCK_PATH) -> Path:
        """Write the lockfile as YAML.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "# agentgate.lock": "Pinned evaluation instrument. Changing any value here changes "
            "what a judge-backed score means; regenerate deliberately, not incidentally.",
            "judge": self.model_dump(mode="json"),
        }
        target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return target
