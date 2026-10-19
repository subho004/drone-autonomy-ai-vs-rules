"""Mission progress: visit and scan stations in order, then land on the pad.

Pure logic (no PyBullet), driven once per control tick. A station is *scanned* when the
drone holds inside its tolerance box, below a speed limit, for ``dwell_s`` seconds. A
station is *missed* when the drone flies past it by ``miss_margin`` (there is no
backwards action, so a missed station cannot be revisited). After the last station the
target becomes the landing pad.
"""

from __future__ import annotations

import math
from enum import IntEnum

from app.scenarios.spec import MissionSpec

NEAR_DX_M = 1.6
HERE_DX_M = 0.7
PAD_HERE_M = 0.5


class StationPhase(IntEnum):
    FAR = 0  # still approaching, more than NEAR_DX_M ahead
    NEAR = 1  # inside the braking zone: start slowing now
    HERE = 2  # inside the station's x window
    PASSED = 3  # flew past (will be marked missed)
    PAD_FAR = 4  # all stations handled; heading for the pad
    PAD_HERE = 5  # over the pad: land
    PAD_NEAR = 6  # in the braking zone before the pad
    PAD_ALIGN = 7  # over the pad's x position but off to the side


class MissionTracker:
    """Tracks which station is active and what has been scanned or missed."""

    def __init__(self, spec: MissionSpec) -> None:
        self.spec = spec
        self.index = 0
        self.scanned: list[bool] = [False] * len(spec.stations)
        self.missed: list[bool] = [False] * len(spec.stations)
        self.dwell_s = 0.0
        self.events: list[tuple[str, int]] = []  # (kind, station index) in order

    @property
    def n_stations(self) -> int:
        return len(self.spec.stations)

    @property
    def stations_done(self) -> bool:
        return self.index >= self.n_stations

    def target(self) -> tuple[float, float, float | None]:
        """Active target (x, y, z); z is None for the pad (landing keeps altitude)."""

        if self.stations_done:
            return self.spec.pad_x, self.spec.pad_y, None
        st = self.spec.stations[self.index]
        return st.x, st.y, st.z

    def update(
        self, pos: tuple[float, float, float], speed_xy: float, dt: float
    ) -> list[tuple[str, int]]:
        """Advance with the drone's true state; returns events fired this tick."""

        fired: list[tuple[str, int]] = []
        if self.stations_done:
            return fired
        s = self.spec
        st = s.stations[self.index]
        if s.gate_mode:  # a gate is passed (or missed) on crossing its plane
            if pos[0] >= st.x:
                ok = abs(pos[1] - st.y) <= s.tol_y and abs(pos[2] - st.z) <= s.tol_z
                fired.append(("scanned" if ok else "missed", self.index))
                self.scanned[self.index] = ok
                self.missed[self.index] = not ok
                self._advance()
            self.events.extend(fired)
            return fired
        inside = (
            abs(pos[0] - st.x) <= s.tol_x
            and abs(pos[1] - st.y) <= s.tol_y
            and abs(pos[2] - st.z) <= s.tol_z
        )
        if inside and speed_xy <= s.scan_speed_max:
            self.dwell_s += dt
        else:
            self.dwell_s = 0.0
        if self.dwell_s >= s.dwell_s:
            self.scanned[self.index] = True
            fired.append(("scanned", self.index))
            self._advance()
        elif pos[0] > st.x + s.miss_margin:
            self.missed[self.index] = True
            fired.append(("missed", self.index))
            self._advance()
        self.events.extend(fired)
        return fired

    def _advance(self) -> None:
        self.index += 1
        self.dwell_s = 0.0

    def phase(self, pos: tuple[float, float, float]) -> StationPhase:
        tx, ty, _ = self.target()
        dx = tx - pos[0]
        if self.stations_done:
            dy = abs(ty - pos[1])
            if abs(dx) <= PAD_HERE_M:
                return (
                    StationPhase.PAD_HERE
                    if dy <= PAD_HERE_M
                    else StationPhase.PAD_ALIGN
                )
            return (
                StationPhase.PAD_NEAR if 0 < dx <= NEAR_DX_M else StationPhase.PAD_FAR
            )
        if dx > NEAR_DX_M:
            return StationPhase.FAR
        if dx > HERE_DX_M:
            return StationPhase.NEAR
        if dx >= -HERE_DX_M:
            return StationPhase.HERE
        return StationPhase.PASSED

    def extras(
        self, pos: tuple[float, float, float], *, moving: bool
    ) -> dict[str, float]:
        """Numeric mission state attached to each Observation (``Observation.extras``)."""

        tx, ty, tz = self.target()
        return {
            "stn_i": float(min(self.index + 1, self.n_stations)),
            "stn_n": float(self.n_stations),
            "st_dx": tx - pos[0],
            "st_dy": ty - pos[1],
            "st_dz": 0.0 if tz is None else tz - pos[2],
            "st_phase": float(int(self.phase(pos))),
            "moving": 1.0 if moving else 0.0,
            "scan_s": self.dwell_s,
            "gate": 1.0 if self.spec.gate_mode else 0.0,
        }

    def landed_on_pad(self, pos: tuple[float, float, float]) -> bool:
        s = self.spec
        return math.hypot(pos[0] - s.pad_x, pos[1] - s.pad_y) <= s.pad_tol

    def score(self) -> tuple[int, int]:
        return sum(self.scanned), self.n_stations


__all__ = [
    "HERE_DX_M",
    "NEAR_DX_M",
    "PAD_HERE_M",
    "MissionTracker",
    "StationPhase",
]
