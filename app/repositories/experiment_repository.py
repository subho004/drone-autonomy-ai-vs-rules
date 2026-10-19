"""Experiment and progress queries."""

from __future__ import annotations

from app.repositories.base import BaseRepository, Row, page_bounds


class ExperimentRepository(BaseRepository):
    async def list_page(self, page: int, page_size: int) -> tuple[list[Row], int]:
        limit, offset = page_bounds(page, page_size)
        total = await self._scalar_int("SELECT COUNT(*) FROM experiments")
        rows = await self._all(
            "SELECT e.id, e.name, e.created_at, e.status, e.config_sha, "
            "(SELECT COUNT(*) FROM flights f WHERE f.experiment_id = e.id) AS n_flights, "
            "(SELECT COUNT(*) FROM flights f WHERE f.experiment_id = e.id "
            " AND f.status = 'complete') AS n_complete "
            "FROM experiments e ORDER BY e.id DESC LIMIT :limit OFFSET :offset",
            {"limit": limit, "offset": offset},
        )
        return rows, total

    async def get(self, experiment_id: int) -> Row | None:
        return await self._one(
            "SELECT id, name, created_at, status FROM experiments WHERE id = :id",
            {"id": experiment_id},
        )

    async def status_counts(self, experiment_id: int) -> dict[str, int]:
        rows = await self._all(
            "SELECT status, COUNT(*) AS n FROM flights WHERE experiment_id = :id "
            "GROUP BY status",
            {"id": experiment_id},
        )
        return {str(r["status"]): int(r["n"]) for r in rows}

    async def per_arm_counts(self, experiment_id: int) -> list[Row]:
        return await self._all(
            "SELECT arm_id, COUNT(*) AS total, "
            "SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) AS complete "
            "FROM flights WHERE experiment_id = :id GROUP BY arm_id ORDER BY arm_id",
            {"id": experiment_id},
        )

    async def recent_finish_times(self, experiment_id: int, limit: int) -> list[str]:
        rows = await self._all(
            "SELECT finished_at FROM flights WHERE experiment_id = :id "
            "AND status = 'complete' AND finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT :limit",
            {"id": experiment_id, "limit": limit},
        )
        return [str(r["finished_at"]) for r in rows]
