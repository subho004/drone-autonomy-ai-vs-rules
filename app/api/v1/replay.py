"""Replay payload for one flight."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.deps import ReplayServiceDep
from utils.response import success_response

router = APIRouter(tags=["replay"])


@router.get("/flights/{flight_id}/replay")
async def get_replay(
    flight_id: int,
    service: ReplayServiceDep,
) -> JSONResponse:
    return success_response(
        data=await service.get_replay(flight_id), message="Replay retrieved"
    )
