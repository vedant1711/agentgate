"""Run-level budget enforcement (A3.6).

Caps are checked *before* a request is issued, so hitting one is a clean stop with an accurate
accounting rather than an overrun discovered afterwards. The runner catches
:class:`~agentgate.errors.BudgetExceededError` and marks the remaining samples
``BUDGET_EXHAUSTED`` instead of losing the work already done.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from agentgate.errors import BudgetExceededError
from agentgate.schemas.results import BudgetSpec
from agentgate.schemas.trajectory import TokenUsage


class BudgetTracker:
    """Tracks requests, tokens, cost, and wall time against a :class:`BudgetSpec`.

    Args:
        spec: The caps to enforce. Any field set to 0 means "unlimited".
        clock: Monotonic time source, injected for tests.
    """

    def __init__(
        self, spec: BudgetSpec | None = None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.spec = spec or BudgetSpec()
        self._clock = clock
        self._started_at = clock()
        self.requests = 0
        self.usage = TokenUsage()
        self.cost_usd = 0.0

    @property
    def elapsed_s(self) -> float:
        """Seconds since the tracker was created."""
        return self._clock() - self._started_at

    @property
    def tokens(self) -> int:
        """Total tokens consumed so far."""
        return self.usage.total_tokens

    def check(self, *, projected_requests: int = 1) -> None:
        """Raise if issuing ``projected_requests`` more requests would breach a cap.

        Args:
            projected_requests: How many requests the caller is about to make.

        Raises:
            BudgetExceededError: When any configured cap is already met or would be exceeded.
        """
        spec = self.spec
        if spec.max_requests and self.requests + projected_requests > spec.max_requests:
            self._fail("requests", self.requests, spec.max_requests)
        if spec.max_tokens and self.tokens >= spec.max_tokens:
            self._fail("tokens", self.tokens, spec.max_tokens)
        if spec.max_cost_usd and self.cost_usd >= spec.max_cost_usd:
            self._fail("cost_usd", self.cost_usd, spec.max_cost_usd)
        if spec.max_wall_s and self.elapsed_s >= spec.max_wall_s:
            self._fail("wall_s", self.elapsed_s, spec.max_wall_s)

    def _fail(self, dimension: str, used: float, cap: float) -> None:
        msg = (
            f"run budget exhausted on {dimension}: used {used:.4g} of {cap:.4g}. "
            f"Raise the cap in the run config or narrow the suite."
        )
        raise BudgetExceededError(msg)

    def record(self, *, usage: TokenUsage | None = None, cost_usd: float = 0.0) -> None:
        """Account for one completed request.

        Args:
            usage: Tokens consumed by the request.
            cost_usd: Projected cost of the request.
        """
        self.requests += 1
        if usage is not None:
            self.usage = self.usage + usage
        self.cost_usd += cost_usd

    @property
    def exhausted(self) -> bool:
        """True when any cap has been reached."""
        try:
            self.check()
        except BudgetExceededError:
            return True
        return False

    def remaining(self) -> dict[str, float]:
        """Return headroom per dimension; ``inf`` where no cap is configured."""
        spec = self.spec
        return {
            "requests": float(spec.max_requests - self.requests)
            if spec.max_requests
            else float("inf"),
            "tokens": float(spec.max_tokens - self.tokens) if spec.max_tokens else float("inf"),
            "cost_usd": (spec.max_cost_usd - self.cost_usd) if spec.max_cost_usd else float("inf"),
            "wall_s": (spec.max_wall_s - self.elapsed_s) if spec.max_wall_s else float("inf"),
        }
