"""CLI smoke tests (Phase 0 acceptance: ``agentgate --help`` works)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentgate.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "statistical regression gate" in result.output
    assert "schema" in result.output


def test_version_reports_provenance() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "agentgate" in result.output


def test_schema_export_and_check_round_trip(tmp_path: Path) -> None:
    export = runner.invoke(app, ["schema", "export", "--target", str(tmp_path)])
    assert export.exit_code == 0
    check = runner.invoke(app, ["schema", "check", "--target", str(tmp_path)])
    assert check.exit_code == 0
    assert "up to date" in check.output


def test_schema_check_fails_on_stale_dir(tmp_path: Path) -> None:
    (tmp_path / "suite.schema.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["schema", "check", "--target", str(tmp_path)])
    assert result.exit_code == 1
    assert "stale schemas" in result.output
