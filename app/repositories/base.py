"""Shared read helpers for repositories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger

logger = get_logger(__name__)

# A DB the workers have not created yet must look empty, not broken.
_MISSING_DB_MARKERS = ("no such table", "unable to open database", "no such column")

Row = Mapping[str, Any]


class BaseRepository:
    """Runs read-only SQL and maps rows to plain dicts."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _all(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> list[Row]:
        try:
            result = await self._session.execute(text(sql), dict(params or {}))
        except OperationalError as exc:
            if any(marker in str(exc.orig).lower() for marker in _MISSING_DB_MARKERS):
                logger.debug("Database not ready, returning no rows: %s", exc.orig)
                return []
            raise
        return [dict(row) for row in result.mappings().all()]

    async def _one(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Row | None:
        rows = await self._all(sql, params)
        return rows[0] if rows else None

    async def _scalar_int(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> int:
        row = await self._one(sql, params)
        if not row:
            return 0
        return int(next(iter(row.values())) or 0)


def page_bounds(page: int, page_size: int) -> tuple[int, int]:
    """Return ``(limit, offset)`` for 1-based pagination."""

    return page_size, (page - 1) * page_size


def rows_to_list(rows: Sequence[Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
