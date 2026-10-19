"""Aggregated results, latency curves and latency distributions."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.deps import PageParam, PageSizeParam, ResultsServiceDep
from utils.response import success_response

router = APIRouter(prefix="/experiments", tags=["results"])


@router.get("/{experiment_id}/results")
async def get_results(
    experiment_id: int,
    service: ResultsServiceDep,
    page: PageParam = 1,
    page_size: PageSizeParam = 200,
    scope: Literal["challenge", "controls", "all"] = "challenge",
) -> JSONResponse:
    return success_response(
        data=await service.results(experiment_id, page, page_size, scope),
        message="Results retrieved",
    )


@router.get("/{experiment_id}/latency-curves")
async def get_latency_curves(
    experiment_id: int,
    service: ResultsServiceDep,
    scope: Literal["challenge", "controls", "all"] = "challenge",
) -> JSONResponse:
    return success_response(
        data=await service.latency_curves(experiment_id, scope),
        message="Latency curves retrieved",
    )


@router.get("/{experiment_id}/latency-distribution")
async def get_latency_distribution(
    experiment_id: int,
    service: ResultsServiceDep,
) -> JSONResponse:
    return success_response(
        data=await service.latency_distribution(experiment_id),
        message="Latency distribution retrieved",
    )
