"""Per-flight trajectory file contract (Parquet, one file per flight).

Path: ``<data_dir>/trajectories/<experiment_id>/<flight_id>.parquet``
(stored in ``flights.trajectory_uri`` relative to the repo root).
One row per control tick, sim time only (never wall time).
"""

from __future__ import annotations

TRAJ_COLUMNS: tuple[str, ...] = (
    "tick",  # int, control tick index
    "t",  # float, sim seconds
    "x",
    "y",
    "z",  # drone position (m)
    "vx",
    "vy",
    "vz",  # drone velocity (m/s)
    "rx",
    "ry",
    "rz",  # executor reference (carrot) position (m)
    "action",  # int, the Action applied to the executor this tick
    "shield",  # int, 1 when the reflex shield overrode the decision this tick
    "ox",
    "oy",
    "oz",  # obstacle position (m), NaN when no obstacle is active
    "orad",  # obstacle radius (m), NaN when none
    "clearance",  # float, surface-to-surface distance to the nearest obstacle (m), NaN if none
    "battery",  # float, percent
    "wind_x",
    "wind_y",  # applied wind acceleration (m/s^2)
)

TRAJ_RATE_HZ = 48

__all__ = ["TRAJ_COLUMNS", "TRAJ_RATE_HZ"]
