"""Async SQLite engine (aiosqlite) for the read-side API.

The API only reads: the database is opened with ``mode=ro`` so it can never
create an empty file or take a write lock. A missing file is handled by the
repositories (empty results, never a 500).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUSY_TIMEOUT_MS = 30_000

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve a repo-root-relative path (absolute paths pass through)."""

    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def build_engine(db_path: str | Path) -> AsyncEngine:
    """Create a read-only async engine for the SQLite file at ``db_path``."""

    url = URL.create(
        "sqlite+aiosqlite",
        database=f"file:{resolve_repo_path(db_path)}",
        query={"mode": "ro", "uri": "true"},
    )
    engine = create_async_engine(url)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]  # DBAPI types are untyped
        cursor = dbapi_connection.cursor()
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        cursor.close()

    return engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory (lazy; FastAPI dependency)."""

    global _engine, _sessionmaker
    if _sessionmaker is None:
        _engine = build_engine(settings.db_path)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
        logger.info("Read-only SQLite engine ready for %s", settings.db_path)
    return _sessionmaker


async def get_session(
    maker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
) -> AsyncIterator[AsyncSession]:
    """Yield one read session per request."""

    async with maker() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the engine (called from the FastAPI lifespan)."""

    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


__all__ = [
    "REPO_ROOT",
    "build_engine",
    "dispose_engine",
    "get_session",
    "get_sessionmaker",
    "resolve_repo_path",
]
