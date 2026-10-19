"""Server-sent events: live experiment progress."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.db.database import get_sessionmaker
from app.repositories.experiment_repository import ExperimentRepository
from app.schemas.api import ProgressOut
from app.services.experiment_service import ExperimentService

logger = get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["events"])

POLL_INTERVAL_S = 1.0


async def _fetch_progress(
    maker: async_sessionmaker[AsyncSession], experiment_id: int
) -> ProgressOut:
    # A fresh session per poll: SQLite read transactions pin a snapshot.
    async with maker() as session:
        return await ExperimentService(ExperimentRepository(session)).get_progress(
            experiment_id
        )


def _sse(event: str, payload: str) -> str:
    return f"event: {event}\ndata: {payload}\n\n"


async def _progress_stream(
    maker: async_sessionmaker[AsyncSession], first: ProgressOut
) -> AsyncIterator[str]:
    progress = first
    experiment_id = first.experiment_id
    try:
        while True:
            yield _sse("progress", progress.model_dump_json())
            if progress.complete:
                yield _sse("done", progress.model_dump_json())
                return
            await asyncio.sleep(POLL_INTERVAL_S)
            progress = await _fetch_progress(maker, experiment_id)
    except asyncio.CancelledError:
        logger.debug("SSE client disconnected from experiment %s", experiment_id)
        raise


@router.get("/{experiment_id}/events")
async def stream_events(
    experiment_id: int,
    maker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
) -> StreamingResponse:
    """Stream progress every second until the experiment completes."""

    first = await _fetch_progress(maker, experiment_id)  # 404 before the stream starts
    return StreamingResponse(
        _progress_stream(maker, first),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
