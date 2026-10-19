"""Experiment listing and progress."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.deps import ExperimentServiceDep, PageParam, PageSizeParam
from utils.response import success_response

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("")
async def list_experiments(
    service: ExperimentServiceDep,
    page: PageParam = 1,
    page_size: PageSizeParam = 20,
) -> JSONResponse:
    return success_response(
        data=await service.list_experiments(page, page_size),
        message="Experiments retrieved",
    )


@router.get("/{experiment_id}/progress")
async def get_progress(
    experiment_id: int,
    service: ExperimentServiceDep,
) -> JSONResponse:
    return success_response(
        data=await service.get_progress(experiment_id), message="Progress retrieved"
    )
