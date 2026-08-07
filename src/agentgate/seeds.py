"""Deterministic seed derivation.

Every source of randomness in a run — sampling, fault injection, bootstrap resampling — is
seeded from one ``base_seed`` plus the identity of the thing being seeded. That gives two
properties the gate depends on:

* **Reproducibility** (A3.4): the same manifest replays to the same numbers.
* **Independence**: two tasks, or two repetitions, never share a random stream. Sharing one is
  how a "flaky dependency" simulation ends up injecting the *same* failures everywhere and
  quietly stops being a simulation of flakiness at all.
"""

from __future__ import annotations

from agentgate.schemas.common import canonical_dumps, sha256_hex

_SEED_BITS = 60
"""Wide enough to never collide in practice, narrow enough to stay a comfortable Python int."""


def derive_seed(base_seed: int, *parts: str | int) -> int:
    """Derive a child seed from ``base_seed`` and an identity.

    Args:
        base_seed: The run's root seed, recorded in the manifest.
        parts: Identity of the stream, e.g. ``(task_id, rep)``.

    Returns:
        A non-negative integer seed, deterministic in its inputs and stable across platforms
        and Python versions (unlike :func:`hash`, which is salted per process).
    """
    payload = canonical_dumps([base_seed, [str(part) for part in parts]])
    return int(sha256_hex(payload)[: _SEED_BITS // 4], 16)
