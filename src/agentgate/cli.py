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
judge_app = typer.Typer(
    name="judge", help="Inspect and pin the evaluation instrument.", no_args_is_help=True
)
app.add_typer(judge_app)

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


@app.command("plan")
def plan(
    target_mde: Annotated[
        float,
        typer.Option("--target-mde", min=0.0001, help="Smallest regression you want to detect."),
    ],
    sigma_d: Annotated[
        float,
        typer.Option("--sigma-d", min=0.0, help="SD of per-task differences (from a pilot run)."),
    ] = 0.15,
    alpha: Annotated[float, typer.Option("--alpha", min=0.0001, max=0.5)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.5, max=0.999)] = 0.80,
    suite: Annotated[
        Path | None,
        typer.Option("--suite", help="Report what this suite can already detect."),
    ] = None,
) -> None:
    """Tell a suite author how many tasks to write (C4)."""
    from agentgate.stats import minimum_detectable_effect, paired_power, plan_suite_size

    summary = plan_suite_size(target_mde=target_mde, sigma_d=sigma_d, alpha=alpha, power=power)
    table = Table(show_header=False, box=None)
    table.add_row("target MDE", f"{target_mde:.4g}")
    table.add_row("assumed sigma_d", f"{sigma_d:.4g}")
    table.add_row("alpha / power", f"{alpha:.3g} / {power:.0%}")
    table.add_row("[bold]tasks required[/bold]", f"[bold]{int(summary['required_tasks'])}[/bold]")
    console.print(table)

    if suite is not None:
        from agentgate.runner import load_suite

        spec = load_suite(suite)
        n = len(spec.tasks)
        achievable = minimum_detectable_effect(sigma_d=sigma_d, n=n, alpha=alpha, power=power)
        achieved = paired_power(effect=target_mde, sigma_d=sigma_d, n=n, alpha=alpha)
        console.print(
            f"\n[bold]{spec.name}[/bold] has {n} tasks: it can detect {achievable:.4g} at "
            f"{power:.0%} power, and has {achieved:.0%} power for your {target_mde:.4g} target."
        )
        if achieved < power:
            shortfall = int(summary["required_tasks"]) - n
            console.print(
                f"[yellow]underpowered[/yellow]: write about {shortfall} more task(s), or widen "
                f"the margin to {achievable:.4g}"
            )
        else:
            console.print("[green]adequately powered[/green] for the target you named")


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


@judge_app.command("rubrics")
def judge_rubrics() -> None:
    """List the rubric criteria and their anchored scales."""
    from agentgate.judge import RUBRICS, rubrics_hash

    table = Table("criterion", "needs", "question")
    for name, rubric in RUBRICS.items():
        needs = ", ".join(
            filter(
                None,
                [
                    "reference" if rubric.needs_reference else "",
                    "contexts" if rubric.needs_contexts else "",
                ],
            )
        )
        table.add_row(name, needs or "-", rubric.question)
    console.print(table)
    console.print(f"rubrics hash: [bold]{rubrics_hash()}[/bold]")


@judge_app.command("lock")
def judge_lock(
    judge_model: Annotated[str, typer.Option("--judge-model", help="Judge model to pin.")],
    path: Annotated[Path, typer.Option("--path", help="Lockfile location.")] = Path(
        "agentgate.lock"
    ),
    anchors: Annotated[
        Path | None, typer.Option("--anchors", help="Anchor set to hash into the lock.")
    ] = None,
    check: Annotated[
        bool, typer.Option("--check", help="Report differences instead of writing.")
    ] = False,
) -> None:
    """Pin (or verify) the evaluation instrument in ``agentgate.lock``."""
    from agentgate.judge import AnchorSet, JudgeLock

    anchor_hash = AnchorSet.load(anchors).content_hash() if anchors else ""
    current = JudgeLock.current(judge_model=judge_model, anchor_hash=anchor_hash)
    existing = JudgeLock.load(path)

    if check:
        if existing is None:
            console.print(f"[red]no lockfile at {path}[/red]; run `agentgate judge lock`")
            raise typer.Exit(code=1)
        compatible, changes = existing.is_compatible_with(current)
        if compatible:
            console.print("[green]ok[/green] the evaluation instrument is unchanged")
            return
        console.print("[yellow]instrument changed since the lockfile was written:[/yellow]")
        for change in changes:
            console.print(f"  · {change}")
        console.print("history recorded under the old lock is not comparable to new runs")
        raise typer.Exit(code=1)

    if existing is not None:
        for change in existing.differences(current):
            console.print(f"[yellow]changing[/yellow] {change}")
    console.print(f"[green]wrote[/green] {current.save(path)}")


@app.command("label")
def label(
    label_set: Annotated[
        str, typer.Option("--set", help="Label set name; written to <dir>/<name>.jsonl.")
    ] = "calibration",
    transcript_path: Annotated[
        Path, typer.Option("--transcript", help="Judge transcript JSON to label against.")
    ] = Path("datasets/calibration/transcript.json"),
    directory: Annotated[Path, typer.Option("--dir", help="Where label sets live.")] = Path(
        "datasets/calibration"
    ),
    limit: Annotated[int, typer.Option("--limit", min=1, help="Items to offer this session.")] = 20,
) -> None:
    """Hand-label judged items so judge-human agreement can be measured (D4)."""
    from agentgate.judge import HumanLabel, JudgeTranscript, LabelSet, calibrate

    if not transcript_path.exists():
        console.print(f"[red]no transcript at {transcript_path}[/red]")
        console.print("run a judged suite first, or point --transcript at a saved transcript")
        raise typer.Exit(code=1)

    transcript = JudgeTranscript.model_validate_json(transcript_path.read_text(encoding="utf-8"))
    target = directory / f"{label_set}.jsonl"
    labels = LabelSet.load(target, name=label_set)
    labelled = {item.key for item in labels.labels}
    pending = [
        (key, entry) for key, entry in sorted(transcript.entries.items()) if key not in labelled
    ][:limit]

    if not pending:
        console.print("[green]nothing left to label[/green]")
    for index, (key, entry) in enumerate(pending, start=1):
        console.print(f"\n[bold]{index}/{len(pending)}[/bold] · criterion: {entry.criterion}")
        console.print(f"judge said: [dim]{entry.mean:.2f}[/dim] (hidden from your judgement above)")
        console.print(f"[dim]key {key[:12]}[/dim]")
        raw = typer.prompt("your score 1-5 (or 's' to skip, 'q' to stop)", default="s")
        if raw.strip().lower() == "q":
            break
        if not raw.strip().isdigit():
            continue
        score = int(raw)
        if not 1 <= score <= 5:
            console.print("[yellow]score must be 1-5; skipping[/yellow]")
            continue
        labels.add(
            HumanLabel(
                criterion=entry.criterion,
                prompt=entry.samples[0].reasoning[:200] if entry.samples else "",
                response="",
                score=score,
            )
        )

    written = labels.save(target)
    console.print(f"\n[green]saved[/green] {written} labels to {target}")
    report = calibrate(labels, transcript)
    for agreement in report.per_criterion:
        kappa = "n/a" if agreement.cohens_kappa is None else f"{agreement.cohens_kappa:.3f}"
        rho = "n/a" if agreement.spearman_rho is None else f"{agreement.spearman_rho:.3f}"
        console.print(f"{agreement.criterion}: n={agreement.n} kappa={kappa} rho={rho}")
    for warning in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


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
