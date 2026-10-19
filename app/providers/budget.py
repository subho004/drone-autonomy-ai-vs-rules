"""Hard spend cap backed by the ``budget_ledger`` SQLite table."""

from __future__ import annotations

import sqlite3

from app.core.config import settings
from app.core.logging import get_logger
from app.harness.store import utc_now

logger = get_logger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised before a paid call that would push total spend over the cap."""

    def __init__(self, spent_usd: float, cap_usd: float) -> None:
        super().__init__(
            f"Budget cap reached: spent ${spent_usd:.4f} of ${cap_usd:.2f}"
        )
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd


class BudgetLedger:
    """Append-only spend log shared by all providers and processes."""

    def __init__(self, conn: sqlite3.Connection, cap_usd: float | None = None) -> None:
        self._conn = conn
        self._cap_usd = settings.budget_usd if cap_usd is None else cap_usd

    @property
    def cap_usd(self) -> float:
        return self._cap_usd

    def spent_usd(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM budget_ledger"
        ).fetchone()
        return float(row["total"])

    def remaining_usd(self) -> float:
        return self._cap_usd - self.spent_usd()

    def check(self, estimated_cost_usd: float = 0.0) -> None:
        """Raise ``BudgetExceededError`` if the next call would exceed the cap."""

        spent = self.spent_usd()
        if spent + estimated_cost_usd > self._cap_usd:
            logger.error("Budget cap hit: spent %.4f cap %.2f", spent, self._cap_usd)
            raise BudgetExceededError(spent, self._cap_usd)

    def record(
        self, provider: str, model: str, cost_usd: float, calls: int = 1
    ) -> None:
        self._conn.execute(
            "INSERT INTO budget_ledger(ts, provider, model, calls, cost_usd) VALUES (?,?,?,?,?)",
            (utc_now(), provider, model, calls, float(cost_usd)),
        )


__all__ = ["BudgetExceededError", "BudgetLedger"]
