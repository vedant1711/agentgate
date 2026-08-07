"""Provenance capture for run manifests (A3.4).

Everything here answers "what exactly produced this number?". Volatile facts (host, Python
build) are captured but deliberately excluded from ``RunManifest.config_hash`` so two machines
replaying the same cache still hash identically.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from importlib import metadata

TRACKED_LIBRARIES = (
    "agentgate",
    "pydantic",
    "numpy",
    "scipy",
    "statsmodels",
    "duckdb",
    "litellm",
    "sentence-transformers",
)
"""Libraries whose versions can change a number, so they belong in the manifest."""


@lru_cache(maxsize=1)
def git_sha() -> str:
    """Return the current commit SHA, or ``'unknown'`` outside a git checkout."""
    return _git("rev-parse", "HEAD") or "unknown"


@lru_cache(maxsize=1)
def git_dirty() -> bool:
    """Return True when the working tree has uncommitted changes."""
    status = _git("status", "--porcelain")
    return bool(status)


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git available
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


@lru_cache(maxsize=1)
def library_versions() -> dict[str, str]:
    """Return installed versions of every library in :data:`TRACKED_LIBRARIES`."""
    versions: dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


@lru_cache(maxsize=1)
def host_info() -> dict[str, str]:
    """Return non-reproducibility-relevant platform details for debugging."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
