"""Flight listing, summaries and decisions."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.deps import FlightServiceDep, PageParam, PageSizeParam
from utils.response import success_response

router = APIRouter(tags=["flights"])


@router.get("/experiments/{experiment_id}/flights")
async def list_flights(
    experiment_id: int,
    service: FlightServiceDep,
    page: PageParam = 1,
    page_size: PageSizeParam = 20,
    arm_id: str | None = None,
    scenario_id: str | None = None,
    outcome: str | None = None,
    event_type: str | None = None,
) -> JSONResponse:
    data = await service.list_flights(
        experiment_id, page, page_size, arm_id, scenario_id, outcome, event_type
    )
    return success_response(data=data, message="Flights retrieved")


@router.get("/flights/{flight_id}/summary")
async def get_flight_summary(
    flight_id: int,
    service: FlightServiceDep,
) -> JSONResponse:
    return success_response(
        data=await service.get_summary(flight_id), message="Flight retrieved"
    )


@router.get("/flights/{flight_id}/decisions")
async def list_decisions(
    flight_id: int,
    service: FlightServiceDep,
    page: PageParam = 1,
    page_size: PageSizeParam = 100,
) -> JSONResponse:
    return success_response(
        data=await service.list_decisions(flight_id, page, page_size),
        message="Decisions retrieved",
    )
