"""Serves the single-file dashboard UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.exceptions import NotFoundError

DASHBOARD_FILE = Path(__file__).resolve().parents[3] / "static" / "dashboard.html"

router = APIRouter(tags=["dashboard"], include_in_schema=False)


@router.get("/dashboard")
async def dashboard() -> FileResponse:
    if not DASHBOARD_FILE.is_file():
        raise NotFoundError("Dashboard not built")
    return FileResponse(DASHBOARD_FILE, media_type="text/html")
