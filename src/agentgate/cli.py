"""``agentgate`` command-line interface (E1).

Commands are added phase by phase; this module owns argument parsing, exit-code mapping, and
console rendering only — never analysis logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from agentgate import __version__
from agentgate.provenance import git_dirty, git_sha, host_info, library_versions
from agentgate.schema_export import EXPORTED_MODELS, export_schemas, schemas_are_current

app = typer.Typer(
    name="agentgate",
    help="A statistical regression gate for AI agents.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
schema_app = typer.Typer(
    name="schema", help="Inspect and export JSON Schemas.", no_args_is_help=True
)
app.add_typer(schema_app)

console = Console()


@app.command()
def version() -> None:
    """Print version and provenance information."""
    table = Table(show_header=False, box=None)
    table.add_row("agentgate", __version__)
    table.add_row("git sha", f"{git_sha()}{' (dirty)' if git_dirty() else ''}")
    for name, value in host_info().items():
        table.add_row(name, value)
    for name, value in library_versions().items():
        if name != "agentgate":
            table.add_row(name, value)
    console.print(table)


@schema_app.command("export")
def schema_export(
    target: Annotated[
        Path, typer.Option("--target", "-t", help="Directory to write JSON Schemas into.")
    ] = Path("schemas"),
) -> None:
    """Write JSON Schemas for every public model."""
    written = export_schemas(target)
    for path in written:
        console.print(f"[green]wrote[/green] {path}")


@schema_app.command("check")
def schema_check(
    target: Annotated[
        Path, typer.Option("--target", "-t", help="Directory holding committed schemas.")
    ] = Path("schemas"),
) -> None:
    """Fail when committed schemas are out of sync with the models."""
    stale = schemas_are_current(target)
    if stale:
        console.print(f"[red]stale schemas:[/red] {', '.join(stale)}")
        console.print("run [bold]agentgate schema export[/bold] and commit the result")
        raise typer.Exit(code=1)
    console.print(f"[green]ok[/green] {len(EXPORTED_MODELS)} schemas up to date")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    main()
