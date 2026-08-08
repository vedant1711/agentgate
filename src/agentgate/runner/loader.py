"""Suite discovery, loading, and validation (Phase 3).

A suite is data, and data that silently means something different from what its author intended
is worse than data that fails loudly. The loader therefore *refuses* structurally broken suites
and *warns* about statistically dubious ones — a suite whose clusters are all singletons, or
whose tasks carry no reference trajectory, will still run but will not support the analysis its
author probably expects.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from agentgate.errors import ConfigError
from agentgate.schemas.task import SuiteSpec

SUITE_FILENAME = "suite.yaml"
DEFAULT_SUITE_ROOT = Path("suites")

MIN_TASKS_FOR_INFERENCE = 5
"""Below this, no interval is worth printing — the loader says so rather than pretending."""


def resolve_suite_path(path: str | Path) -> Path:
    """Resolve a suite argument to the YAML file that defines it.

    Args:
        path: A suite file, or a directory containing ``suite.yaml``.

    Returns:
        Path to the suite YAML.

    Raises:
        ConfigError: When nothing loadable is there.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        nested = candidate / SUITE_FILENAME
        if nested.is_file():
            return nested
        yamls = sorted(candidate.glob("*.yaml"))
        if len(yamls) == 1:
            return yamls[0]
        if not yamls:
            msg = f"no suite file in {candidate}: expected {SUITE_FILENAME}"
            raise ConfigError(msg)
        names = ", ".join(p.name for p in yamls)
        msg = f"{candidate} holds several YAML files ({names}); name one explicitly"
        raise ConfigError(msg)
    msg = f"suite path does not exist: {candidate}"
    raise ConfigError(msg)


def load_suite(path: str | Path) -> SuiteSpec:
    """Load and validate a suite.

    Args:
        path: A suite file or directory.

    Returns:
        The validated suite.

    Raises:
        ConfigError: On malformed YAML or schema violations, with the offending file named.
    """
    suite_file = resolve_suite_path(path)
    try:
        payload = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"{suite_file}: malformed YAML ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{suite_file}: a suite must be a YAML mapping, got {type(payload).__name__}"
        raise ConfigError(msg)
    try:
        return SuiteSpec.model_validate(payload)
    except ValidationError as exc:
        msg = f"{suite_file}: {_format_errors(exc)}"
        raise ConfigError(msg) from exc


def _format_errors(exc: ValidationError) -> str:
    """Render pydantic errors as one line per problem, located by field path."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"{location or '<root>'}: {error['msg']}")
    return "; ".join(lines)


def discover_suites(root: str | Path = DEFAULT_SUITE_ROOT) -> dict[str, Path]:
    """Find every suite under ``root``.

    Args:
        root: Directory to scan.

    Returns:
        ``{suite name: path}`` sorted by name. Unloadable directories are skipped silently so
        one broken suite never blocks the listing of the others.
    """
    base = Path(root)
    found: dict[str, Path] = {}
    if not base.is_dir():
        return found
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir():
            continue
        try:
            suite = load_suite(candidate)
        except ConfigError:
            continue
        found[suite.name] = resolve_suite_path(candidate)
    return found


def validate_suite(suite: SuiteSpec) -> list[str]:
    """Return warnings about a structurally valid but statistically questionable suite.

    These are warnings, not errors: the author may know exactly what they are doing. But a gate
    that quietly reports a confident verdict on six tasks is the failure mode this whole project
    exists to prevent, so the harness says it out loud.

    Args:
        suite: A loaded suite.

    Returns:
        Human-readable warnings, empty when the suite is in good shape.
    """
    warnings: list[str] = []
    n = len(suite.tasks)

    if n < MIN_TASKS_FOR_INFERENCE:
        warnings.append(
            f"suite has only {n} tasks; intervals will be very wide and the gate will "
            f"most likely report UNDERPOWERED"
        )

    cluster_sizes: dict[str, int] = {}
    for task in suite.tasks:
        cluster_sizes[task.cluster_id] = cluster_sizes.get(task.cluster_id, 0) + 1
    if suite.has_clusters and min(cluster_sizes.values()) == 1:
        singletons = sorted(name for name, size in cluster_sizes.items() if size == 1)
        warnings.append(
            f"{len(singletons)} cluster(s) contain a single task ({', '.join(singletons[:5])}"
            f"{'...' if len(singletons) > 5 else ''}); clustered standard errors will be close "
            f"to naive ones"
        )

    without_trajectory = [task.id for task in suite.tasks if task.reference.trajectory is None]
    if without_trajectory:
        warnings.append(
            f"{len(without_trajectory)} task(s) declare no reference trajectory; the whole "
            f"trajectory metric family will be skipped for them"
        )

    without_checker = [task.id for task in suite.tasks if task.checker is None]
    if without_checker:
        warnings.append(
            f"{len(without_checker)} task(s) declare no checker; outcome.task_success cannot "
            f"be scored for them"
        )

    prompts: dict[str, str] = {}
    for task in suite.tasks:
        previous = prompts.get(task.prompt)
        if previous is not None:
            warnings.append(f"tasks {previous!r} and {task.id!r} share an identical prompt")
        prompts[task.prompt] = task.id

    return warnings
