"""JSON Schema export is complete, valid, and committed in sync with the models."""

from __future__ import annotations

import json
from pathlib import Path

from agentgate.schema_export import (
    EXPORTED_MODELS,
    export_schemas,
    render_schemas,
    schemas_are_current,
)

REPO_SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


def test_every_exported_model_renders_a_valid_schema() -> None:
    rendered = render_schemas()
    assert len(rendered) == len(EXPORTED_MODELS)
    for filename, text in rendered.items():
        doc = json.loads(text)
        assert doc["$schema"].startswith("https://json-schema.org/")
        assert doc["$id"].endswith(filename)
        assert "properties" in doc or "$ref" in doc or "allOf" in doc


def test_export_writes_files(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert {p.name for p in written} == set(render_schemas())
    assert schemas_are_current(tmp_path) == []


def test_stale_schemas_are_detected(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    (tmp_path / "suite.schema.json").write_text("{}", encoding="utf-8")
    assert "suite.schema.json" in schemas_are_current(tmp_path)


def test_committed_schemas_are_up_to_date() -> None:
    stale = schemas_are_current(REPO_SCHEMAS)
    assert stale == [], f"run `agentgate schema export`; stale: {stale}"
