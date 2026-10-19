"""Arm and scenario catalogue use-cases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from app.core.exceptions import NotFoundError
from app.repositories.catalog_repository import ArmRepository, ScenarioRepository
from app.schemas.api import ArmOut, Page, ScenarioOut, ScenarioSpecOut
from app.services.paging import make_page


class CatalogService:
    def __init__(self, arms: ArmRepository, scenarios: ScenarioRepository) -> None:
        self._arms = arms
        self._scenarios = scenarios

    async def list_arms(self, page: int, page_size: int) -> Page[ArmOut]:
        rows, total = await self._arms.list_page(page, page_size)
        return make_page([ArmOut(**r) for r in rows], total, page, page_size)

    async def list_scenarios(
        self, page: int, page_size: int, tier: str | None, event_type: str | None
    ) -> Page[ScenarioOut]:
        rows, total = await self._scenarios.list_page(page, page_size, tier, event_type)
        items = [ScenarioOut(**{**r, "certified": bool(r["certified"])}) for r in rows]
        return make_page(items, total, page, page_size)

    async def get_scenario_spec(self, scenario_id: str) -> ScenarioSpecOut:
        row = await self._scenarios.get(scenario_id)
        if row is None:
            raise NotFoundError(f"Scenario '{scenario_id}' not found")
        return to_spec_out(row)


def to_spec_out(row: Mapping[str, Any]) -> ScenarioSpecOut:
    """Build a ScenarioSpecOut from a scenarios row (spec_json parsed leniently)."""

    data = dict(row)
    try:
        spec = json.loads(str(data.pop("spec_json", "{}")))
    except json.JSONDecodeError:
        spec = {}
    data["certified"] = bool(data["certified"])
    return ScenarioSpecOut(**data, spec=spec if isinstance(spec, dict) else {})
