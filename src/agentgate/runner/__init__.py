"""Runner: suite loading, K-repetition scheduling, seeds, resume, and pairing guards."""

from agentgate.runner.config import DEFAULT_BASE_SEED, DEFAULT_RUN_ROOT, RunConfig
from agentgate.runner.loader import (
    discover_suites,
    load_suite,
    resolve_suite_path,
    validate_suite,
)
from agentgate.runner.pairing import PairedSample, assert_comparable, pair_trajectories
from agentgate.runner.scheduler import ProgressHook, Runner, RunResult, run_suite

__all__ = [
    "DEFAULT_BASE_SEED",
    "DEFAULT_RUN_ROOT",
    "PairedSample",
    "ProgressHook",
    "RunConfig",
    "RunResult",
    "Runner",
    "assert_comparable",
    "discover_suites",
    "load_suite",
    "pair_trajectories",
    "resolve_suite_path",
    "run_suite",
    "validate_suite",
]
