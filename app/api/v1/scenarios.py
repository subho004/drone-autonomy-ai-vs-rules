"""Scenario catalogue."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.deps import CatalogServiceDep, PageParam, PageSizeParam
from utils.response import success_response

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.get("")
async def list_scenarios(
    service: CatalogServiceDep,
    page: PageParam = 1,
    page_size: PageSizeParam = 50,
    tier: str | None = None,
    event_type: str | None = None,
) -> JSONResponse:
    data = await service.list_scenarios(page, page_size, tier, event_type)
    return success_response(data=data, message="Scenarios retrieved")


@router.get("/{scenario_id}/spec")
async def get_scenario_spec(
    scenario_id: str,
    service: CatalogServiceDep,
) -> JSONResponse:
    return success_response(
        data=await service.get_scenario_spec(scenario_id), message="Scenario retrieved"
    )
