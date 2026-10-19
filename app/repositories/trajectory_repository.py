"""Parquet trajectory access (one file per flight)."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pyarrow.parquet as pq

from app.db.database import resolve_repo_path

# Columns the replay view needs (a subset of the full trajectory contract).
REPLAY_COLUMNS: tuple[str, ...] = (
    "t",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "action",
    "shield",
    "ox",
    "oy",
    "oz",
    "orad",
    "clearance",
)
MAX_REPLAY_HZ = 24.0


def _read_columns(path: Path) -> dict[str, list[float]]:
    table = pq.read_table(path, columns=list(REPLAY_COLUMNS))
    return {name: table.column(name).to_pylist() for name in REPLAY_COLUMNS}


def downsample(
    columns: dict[str, list[float]], max_hz: float = MAX_REPLAY_HZ
) -> tuple[dict[str, list[float | None]], float]:
    """Stride-downsample to <= ``max_hz`` keeping the last row; NaN becomes None."""

    t = columns["t"]
    n = len(t)
    if n < 2:
        return {k: [_clean(v) for v in vals] for k, vals in columns.items()}, 0.0
    native_hz = (n - 1) / max(t[-1] - t[0], 1e-9)
    stride = max(1, math.ceil(native_hz / max_hz - 1e-9))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    out = {k: [_clean(vals[i]) for i in idx] for k, vals in columns.items()}
    return out, native_hz / stride


def _clean(value: float | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 4)
    return value


class TrajectoryRepository:
    async def load(
        self, trajectory_uri: str
    ) -> tuple[dict[str, list[float | None]], float] | None:
        """Return downsampled column arrays, or None when the file is missing."""

        path = resolve_repo_path(trajectory_uri)
        if not path.is_file():
            return None
        columns = await asyncio.to_thread(_read_columns, path)
        return downsample(columns)
