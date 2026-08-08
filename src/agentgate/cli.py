"""``agentgate`` command-line interface (E1).

Commands are added phase by phase; this module owns argument parsing, exit-code mapping, and
console rendering only — never analysis logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agentgate import __version__
from agentgate.provenance import git_dirty, git_sha, host_info, library_versions
from agentgate.runner.config import DEFAULT_BASE_SEED
from agentgate.schema_export import EXPORTED_MODELS, export_schemas, schemas_are_current
from agentgate.schemas.common import ProviderMode

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
docs_app = typer.Typer(
    name="docs", help="Generate documentation from the code.", no_args_is_help=True
)
app.add_typer(docs_app)

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


@app.command()
def run(
    suite: Annotated[Path, typer.Option("--suite", "-s", help="Suite file or directory.")],
    system: Annotated[
        str, typer.Option("--system", help="System-under-test label, e.g. baseline or candidate.")
    ] = "baseline",
    k: Annotated[
        int | None, typer.Option("--k", help="Repetitions per task. Defaults to the suite's.")
    ] = None,
    mode: Annotated[
        ProviderMode, typer.Option("--mode", help="Provider mode.", case_sensitive=False)
    ] = ProviderMode.MOCK,
    agent: Annotated[
        str | None, typer.Option("--agent", help="Override the suite's declared agent.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Base seed for the run.")] = DEFAULT_BASE_SEED,
    concurrency: Annotated[int, typer.Option("--concurrency", "-j", min=1)] = 4,
    model: Annotated[str, typer.Option("--model", help="Agent model id.")] = "mock/agent",
    faults_from_env: Annotated[
        bool, typer.Option("--faults-from-env/--no-faults", help="Read FAULT_* knobs from env.")
    ] = True,
    max_requests: Annotated[
        int, typer.Option("--max-requests", min=0, help="Provider request cap; 0 is unlimited.")
    ] = 0,
    store: Annotated[
        Path | None, typer.Option("--store", help="DuckDB file to persist the run into.")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    score: Annotated[
        bool, typer.Option("--score/--no-score", help="Score the run with the metrics engine.")
    ] = True,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Override the derived run id.")
    ] = None,
) -> None:
    """Execute a suite against one system, record its trajectories, and score them."""
    import asyncio

    from agentgate.faults import FaultConfig
    from agentgate.runner import RunConfig, Runner
    from agentgate.schemas.results import BudgetSpec

    config = RunConfig(
        suite_path=suite,
        system=system,
        agent=agent,
        k=k,
        mode=mode,
        base_seed=seed,
        concurrency=concurrency,
        model=model,
        budget=BudgetSpec(max_requests=max_requests),
        faults=FaultConfig.from_env() if faults_from_env else FaultConfig(),
        store_path=store,
        resume=resume,
        run_id=run_id,
    )
    runner = Runner(config)
    console.print(
        f"[bold]{runner.suite.name}@{runner.suite.version}[/bold] "
        f"· {len(runner.suite.tasks)} tasks x K={runner.k} "
        f"· agent={runner.agent_name} · mode={mode.value} · system={system}"
    )
    if config.faults.enabled:
        console.print(f"[yellow]faults active:[/yellow] {', '.join(config.faults.active())}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as bar:
        tracker = bar.add_task("running", total=len(runner.units()))

        def tick(completed: int, total: int, label: str) -> None:
            bar.update(tracker, completed=completed, total=total, description=label)

        result = asyncio.run(runner.run(progress=tick))

    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    if score:
        from agentgate.metrics import MetricsEngine

        engine = MetricsEngine()
        results = engine.score_run(runner.suite, result.trajectories, run_id=result.run_id)
        usable = [item for item in results if item.is_scored]
        console.print(
            f"scored {len(usable)}/{len(results)} metric samples across "
            f"{len({item.metric for item in usable})} metrics"
        )
        if store is not None:
            from agentgate.storage.duckdb_store import RunStore

            with RunStore(store) as run_store:
                run_store.save_scores(result.run_id, results)

    summary = result.summary
    table = Table(show_header=False, box=None)
    table.add_row("run id", result.run_id)
    table.add_row("config hash", result.manifest.config_hash)
    table.add_row("samples", str(summary.n_samples))
    table.add_row("status", ", ".join(f"{k}={v}" for k, v in sorted(summary.status_counts.items())))
    table.add_row("tokens", f"{summary.total_tokens:,}")
    table.add_row("est. cost", f"${summary.total_cost_usd:.4f}")
    table.add_row("cache", f"{summary.cache_hits} hits / {summary.cache_misses} misses")
    table.add_row("wall time", f"{summary.wall_seconds:.2f}s")
    table.add_row("trajectories", str(config.run_dir(result.run_id) / "trajectories.jsonl"))
    console.print(table)


@app.command("suites")
def list_suites(
    root: Annotated[Path, typer.Option("--root", help="Directory to scan.")] = Path("suites"),
) -> None:
    """List available suites and any validation warnings."""
    from agentgate.runner import discover_suites, load_suite, validate_suite

    found = discover_suites(root)
    if not found:
        console.print(f"[yellow]no suites found under {root}[/yellow]")
        raise typer.Exit(code=1)
    table = Table("suite", "version", "tasks", "clusters", "K", "agent", "warnings")
    for name, path in found.items():
        spec = load_suite(path)
        warnings = validate_suite(spec)
        table.add_row(
            name,
            spec.version,
            str(len(spec.tasks)),
            str(spec.n_clusters),
            str(spec.default_k),
            spec.agent,
            str(len(warnings)) if warnings else "-",
        )
    console.print(table)


@app.command("metrics")
def list_metrics(
    family: Annotated[
        str | None, typer.Option("--family", help="Filter to one metric family.")
    ] = None,
) -> None:
    """List the registered metrics and what each requires."""
    from agentgate.metrics import engine as _engine  # noqa: F401  populates the registry
    from agentgate.metrics import registry
    from agentgate.schemas.common import MetricFamily

    metrics = registry.by_family(MetricFamily(family)) if family else registry.all_metrics()
    table = Table("metric", "family", "dtype", "direction", "requires")
    for metric in metrics:
        table.add_row(
            metric.name,
            metric.family.value,
            metric.dtype,
            metric.direction,
            ", ".join(sorted(r.value for r in metric.requires)) or "-",
        )
    console.print(table)
    console.print(f"{len(metrics)} metrics")


@docs_app.command("metrics")
def docs_metrics(
    target: Annotated[Path, typer.Option("--target", "-t", help="Markdown file to write.")] = Path(
        "docs/metrics.md"
    ),
    check: Annotated[
        bool, typer.Option("--check", help="Fail when the committed catalogue is stale.")
    ] = False,
) -> None:
    """Generate the metric catalogue from the registry."""
    from agentgate.metrics import engine as _engine  # noqa: F401  populates the registry
    from agentgate.metrics.docgen import metrics_doc_is_current, write_metrics_doc

    if check:
        if metrics_doc_is_current(target):
            console.print(f"[green]ok[/green] {target} is up to date")
            return
        console.print(f"[red]stale[/red] {target}; run `agentgate docs metrics`")
        raise typer.Exit(code=1)
    console.print(f"[green]wrote[/green] {write_metrics_doc(target)}")


@app.command("corpus")
def corpus_build(
    target: Annotated[
        Path, typer.Option("--target", "-t", help="Directory to write the wiki corpus into.")
    ] = Path("datasets/corpus/wiki"),
) -> None:
    """Regenerate the synthetic wiki corpus the RAG agents read."""
    from agentgate.agents.corpus import write_corpus

    count = write_corpus(target)
    console.print(f"[green]wrote[/green] {count} documents to {target}")


@app.command("agents")
def list_agents() -> None:
    """List the registered reference agents."""
    from agentgate.agents.registry import AGENT_CLASSES, agent_names

    table = Table("agent", "description")
    for name in agent_names():
        doc = (AGENT_CLASSES[name].__doc__ or "").strip().splitlines()[0]
        table.add_row(name, doc)
    console.print(table)


@app.command("scenarios")
def list_scenarios() -> None:
    """List the fault-injection scenarios and what each simulates."""
    from agentgate.faults import SIGNATURES, scenario_names

    table = Table("scenario", "knob", "simulates")
    for name in scenario_names():
        signature = SIGNATURES[name]
        table.add_row(name, signature.knob, signature.simulates)
    console.print(table)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    main()
