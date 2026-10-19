"""Task logic for the extra simulations: follow-me, restricted zones, landing sites.

Pure (no PyBullet), driven once per control tick by the flight runner. Each tracker
reports events, exposes numeric state for the observation (``Observation.extras``) and
returns graded metrics, so results are measured in metres and seconds, not just pass/fail.
"""

from __future__ import annotations

import math

import numpy as np

from app.scenarios.spec import SiteSpec, TargetSpec, ZoneSpec

BIG_SITE_RADIUS_M = 1.2
ZONE_SIDE_MARGIN_M = 0.7  # clearance wanted beside a zone when sidestepping
FAR_M = 99.0


class FollowTracker:
    """Keep a moving target inside a range band ahead; lock is lost after a while outside."""

    def __init__(self, spec: TargetSpec, start_x: float = 0.0) -> None:
        self.spec = spec
        self.start_x = start_x
        self.in_band_s = 0.0
        self.total_s = 0.0
        self.out_s = 0.0
        self.err_sum = 0.0
        self.lost = False
        self.overran = False
        self.max_gap = 0.0

    # ----------------------------------------------------------- target path
    def target_at(self, t: float) -> tuple[float, float, float]:
        s = self.spec
        if s.path == "stopgo":
            cycle = s.move_s + s.stop_s
            full, rem = divmod(t, cycle)
            dist = (full * s.move_s + min(rem, s.move_s)) * s.speed
        else:
            dist = s.speed * t
        y = s.start_y
        if s.path == "weave":
            y += s.weave_amp * math.sin(2.0 * math.pi * t / s.weave_period_s)
        return self.start_x + s.start_dx + dist, y, s.alt

    def target_vel(self, t: float) -> tuple[float, float]:
        s = self.spec
        vx = s.speed
        if s.path == "stopgo":
            vx = s.speed if (t % (s.move_s + s.stop_s)) < s.move_s else 0.0
        vy = 0.0
        if s.path == "weave":
            w = 2.0 * math.pi / s.weave_period_s
            vy = s.weave_amp * w * math.cos(w * t)
        return vx, vy

    # ---------------------------------------------------------------- update
    def update(
        self, pos: tuple[float, float, float], t: float, dt: float
    ) -> list[tuple[str, int]]:
        tx, ty, _ = self.target_at(t)
        dx, dy = tx - pos[0], ty - pos[1]
        s = self.spec
        in_band = s.band_min <= dx <= s.band_max and abs(dy) <= s.band_half_width
        self.total_s += dt
        self.in_band_s += dt if in_band else 0.0
        self.out_s = 0.0 if in_band else self.out_s + dt
        mid = 0.5 * (s.band_min + s.band_max)
        self.err_sum += abs(dx - mid) * dt
        self.max_gap = max(self.max_gap, dx)
        fired: list[tuple[str, int]] = []
        if dx < 0.3 and abs(dy) < 0.5 and not self.overran:
            self.overran = True
            fired.append(("overran", 0))
        if self.out_s >= s.lost_after_s and not self.lost:
            self.lost = True
            fired.append(("lost", 0))
        return fired

    def finished(self, t: float) -> bool:
        return t >= self.spec.duration_s

    # ------------------------------------------------------------- reporting
    def extras(
        self, pos: tuple[float, float, float], vel: tuple[float, float, float], t: float
    ) -> dict[str, float]:
        tx, ty, _ = self.target_at(t)
        tvx, tvy = self.target_vel(t)
        s = self.spec
        dx, dy = tx - pos[0], ty - pos[1]
        in_band = s.band_min <= dx <= s.band_max and abs(dy) <= s.band_half_width
        return {
            "fol_dx": dx,
            "fol_dy": dy,
            "fol_rvx": tvx - vel[0],
            "fol_rvy": tvy - vel[1],
            "fol_in": 1.0 if in_band else 0.0,
            "fol_out_s": self.out_s,
            "fol_bmin": s.band_min,
            "fol_bmax": s.band_max,
        }

    def metrics(self) -> dict[str, float | bool]:
        n = max(self.total_s, 1e-9)
        return {
            "in_band_frac": self.in_band_s / n,
            "mean_range_error_m": self.err_sum / n,
            "max_gap_m": self.max_gap,
            "lost": self.lost,
            "overran": self.overran,
        }


class ZoneMonitor:
    """Signed penetration of restricted zones; also summarises the nearest zone ahead."""

    def __init__(self, zones: list[ZoneSpec]) -> None:
        self.zones = zones
        self.max_depth = 0.0
        self.time_inside_s = 0.0
        self.entries = 0
        self._inside_prev = False

    @staticmethod
    def depth(z: ZoneSpec, x: float, y: float) -> float:
        """Distance inside the boundary (0 when outside)."""

        if not (z.x0 <= x <= z.x1 and z.y0 <= y <= z.y1):
            return 0.0
        return min(x - z.x0, z.x1 - x, y - z.y0, z.y1 - y)

    def update(
        self, pos: tuple[float, float, float], dt: float
    ) -> list[tuple[str, int]]:
        depth = max((self.depth(z, pos[0], pos[1]) for z in self.zones), default=0.0)
        inside = depth > 0.0
        fired: list[tuple[str, int]] = []
        if inside and not self._inside_prev:
            self.entries += 1
            fired.append(("breach", self.entries))
        self._inside_prev = inside
        self.max_depth = max(self.max_depth, depth)
        self.time_inside_s += dt if inside else 0.0
        return fired

    def nearest_ahead(self, x: float, y: float) -> tuple[float, bool, int]:
        """(distance to entry, drone path blocked, sidestep direction +1 left / -1 right)."""

        best: tuple[float, ZoneSpec] | None = None
        for z in self.zones:
            if z.x1 < x - 0.2:
                continue
            d = max(z.x0 - x, 0.0)
            if best is None or d < best[0]:
                best = (d, z)
        if best is None:
            return FAR_M, False, 0
        d, z = best
        lo, hi = z.y0 - ZONE_SIDE_MARGIN_M, z.y1 + ZONE_SIDE_MARGIN_M
        blocked = lo <= y <= hi
        direction = 0
        if blocked:
            direction = 1 if (hi - y) < (y - lo) else -1
        return d, blocked, direction

    def extras(self, pos: tuple[float, float, float]) -> dict[str, float]:
        d, blocked, direction = self.nearest_ahead(pos[0], pos[1])
        inside = (
            max((self.depth(z, pos[0], pos[1]) for z in self.zones), default=0.0) > 0
        )
        return {
            "zn_dx": d,
            "zn_blocked": 1.0 if blocked else 0.0,
            "zn_dir": float(direction),
            "zn_inside": 1.0 if inside else 0.0,
        }

    def predicts_breach(self, pos: tuple[float, float, float], speed: float) -> bool:
        """True if braking now would still leave the drone inside a zone it is heading into."""

        d, blocked, _ = self.nearest_ahead(pos[0], pos[1])
        stop = speed * speed / (2.0 * 4.0) + 0.3
        return blocked and d <= stop + 0.15

    def metrics(self) -> dict[str, float]:
        return {
            "max_breach_m": self.max_depth,
            "breach_time_s": self.time_inside_s,
            "breach_count": float(self.entries),
        }


class LandingTracker:
    """Candidate landing sites ahead; resolves where an irreversible landing ends up."""

    def __init__(self, sites: list[SiteSpec]) -> None:
        self.sites = sites
        self.chosen: int | None = None
        self.miss_m = math.inf

    def ahead(
        self, pos: tuple[float, float, float], n: int = 3
    ) -> list[tuple[int, SiteSpec]]:
        cands = [(i, s) for i, s in enumerate(self.sites) if s.x > pos[0] - 0.5]
        cands.sort(key=lambda it: math.hypot(it[1].x - pos[0], it[1].y - pos[1]))
        return cands[:n]

    def extras(self, pos: tuple[float, float, float]) -> dict[str, float]:
        out: dict[str, float] = {"ls_n": float(len(self.sites))}
        for k, (_, s) in enumerate(self.ahead(pos), start=1):
            out[f"ls{k}_dx"] = s.x - pos[0]
            out[f"ls{k}_dy"] = s.y - pos[1]
            out[f"ls{k}_r"] = s.radius
            out[f"ls{k}_hz"] = 1.0 if s.hazard else 0.0
        return out

    def resolve(self, pos: tuple[float, float, float]) -> str:
        """Outcome label for a landing at ``pos``."""

        dists = [math.hypot(s.x - pos[0], s.y - pos[1]) - s.radius for s in self.sites]
        if not dists:
            return "landed_off_site"
        k = int(np.argmin(dists))
        self.chosen, self.miss_m = k, float(dists[k])
        if dists[k] > 0.0:
            return "landed_off_site"
        return "landed_hazard" if self.sites[k].hazard else "landed_safe"

    def metrics(self) -> dict[str, float]:
        return {
            "chosen_site": -1.0 if self.chosen is None else float(self.chosen),
            "miss_m": self.miss_m if math.isfinite(self.miss_m) else -1.0,
        }


__all__ = [
    "BIG_SITE_RADIUS_M",
    "FollowTracker",
    "LandingTracker",
    "ZoneMonitor",
]
