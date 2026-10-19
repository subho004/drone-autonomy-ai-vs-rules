"""Arm catalogue."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.deps import CatalogServiceDep, PageParam, PageSizeParam
from utils.response import success_response

router = APIRouter(prefix="/arms", tags=["arms"])


@router.get("")
async def list_arms(
    service: CatalogServiceDep,
    page: PageParam = 1,
    page_size: PageSizeParam = 100,
) -> JSONResponse:
    return success_response(
        data=await service.list_arms(page, page_size), message="Arms retrieved"
    )
