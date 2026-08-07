"""Export pydantic models to JSON Schema under ``schemas/``.

Editors pick these up for YAML autocomplete on suite and policy files, and CI asserts they are
in sync with the models so a schema change can never land undocumented.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from agentgate.schemas import (
    ComparisonResult,
    GatePolicy,
    GateVerdict,
    MetricResult,
    RunManifest,
    RunReport,
    SuiteSpec,
    TaskSpec,
    Trajectory,
)

EXPORTED_MODELS: dict[str, type[BaseModel]] = {
    "suite": SuiteSpec,
    "task": TaskSpec,
    "trajectory": Trajectory,
    "run_manifest": RunManifest,
    "run_report": RunReport,
    "metric_result": MetricResult,
    "comparison_result": ComparisonResult,
    "gate_policy": GatePolicy,
    "gate_verdict": GateVerdict,
}
"""Public contract surface. Adding a model here is how it becomes part of the API."""


def schema_for(model: type[BaseModel], name: str) -> dict[str, object]:
    """Build the JSON Schema document for one model.

    Args:
        model: The pydantic model class.
        name: Slug used in ``$id``.

    Returns:
        A JSON-Schema-2020-12 document.
    """
    schema = model.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://agentgate.dev/schemas/{name}.schema.json"
    return schema


def render_schemas() -> dict[str, str]:
    """Return ``{filename: json text}`` for every exported model."""
    return {
        f"{name}.schema.json": json.dumps(schema_for(model, name), indent=2, sort_keys=True) + "\n"
        for name, model in EXPORTED_MODELS.items()
    }


def export_schemas(target_dir: str | Path) -> list[Path]:
    """Write every JSON Schema to ``target_dir``.

    Args:
        target_dir: Directory to write into; created if missing.

    Returns:
        Paths written, sorted by name.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, text in render_schemas().items():
        path = target / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return sorted(written)


def schemas_are_current(target_dir: str | Path) -> list[str]:
    """Return the names of schema files that are missing or stale.

    Args:
        target_dir: Directory holding the committed schemas.

    Returns:
        Filenames needing regeneration; empty when everything is in sync.
    """
    target = Path(target_dir)
    stale: list[str] = []
    for filename, text in render_schemas().items():
        path = target / filename
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(filename)
    return sorted(stale)
