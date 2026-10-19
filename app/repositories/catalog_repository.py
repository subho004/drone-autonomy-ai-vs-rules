"""Arm and scenario catalogue queries."""

from __future__ import annotations

from app.repositories.base import BaseRepository, Row, page_bounds


class ArmRepository(BaseRepository):
    async def list_page(self, page: int, page_size: int) -> tuple[list[Row], int]:
        limit, offset = page_bounds(page, page_size)
        total = await self._scalar_int("SELECT COUNT(*) FROM arms")
        rows = await self._all(
            "SELECT id, kind, provider, model FROM arms ORDER BY id "
            "LIMIT :limit OFFSET :offset",
            {"limit": limit, "offset": offset},
        )
        return rows, total


class ScenarioRepository(BaseRepository):
    async def list_page(
        self,
        page: int,
        page_size: int,
        tier: str | None,
        event_type: str | None,
    ) -> tuple[list[Row], int]:
        limit, offset = page_bounds(page, page_size)
        where = "WHERE (:tier IS NULL OR tier = :tier) AND (:event IS NULL OR event_type = :event)"
        params = {"tier": tier, "event": event_type}
        total = await self._scalar_int(
            f"SELECT COUNT(*) FROM scenarios {where}", params
        )
        rows = await self._all(
            "SELECT id, suite, seed, event_type, tier, budget_s, certified FROM scenarios "
            f"{where} ORDER BY id LIMIT :limit OFFSET :offset",
            {**params, "limit": limit, "offset": offset},
        )
        return rows, total

    async def get(self, scenario_id: str) -> Row | None:
        return await self._one(
            "SELECT id, suite, seed, event_type, tier, budget_s, certified, spec_json "
            "FROM scenarios WHERE id = :id",
            {"id": scenario_id},
        )
