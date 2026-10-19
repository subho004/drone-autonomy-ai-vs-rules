"""Rule-based pilots for the extra simulations (follow-me, zones, landing sites, gates).

Same interface as every other arm: observation in, one of the 11 actions out. Their
decision latency is their (tiny) compute time, so they are the zero-latency references that
the injected-latency sweeps and the live models are compared against.
"""

from __future__ import annotations

import time
from dataclasses import replace

from app.controllers.arms import FsmArm, MissionFsmArm, ReflexArm, _ArmBase
from app.domain.actions import Action
from app.domain.decision import Decision
from app.domain.observation import MAX_RANGE_M, Observation

LAT_STEP_THRESHOLD_M = 0.6  # sidestep when the target/site is further off than this
SITE_LAND_DX_M = 0.7
SITE_LAND_DY_M = 0.5
ZONE_LOOKAHEAD_M = 3.0


def _lateral(dy: float, threshold: float = LAT_STEP_THRESHOLD_M) -> Action | None:
    if dy > threshold:
        return Action.LEFT
    if dy < -threshold:
        return Action.RIGHT
    return None


class FollowArm(_ArmBase):
    """Keeps the target mid-band: FORWARD when it is ahead of the band centre, else slow."""

    def __init__(self) -> None:
        super().__init__("follow_pilot")

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        ex = obs.extras
        if "fol_dx" not in ex:
            return self._decision(Action.FORWARD, t0)
        mid = 0.5 * (ex["fol_bmin"] + ex["fol_bmax"])
        if not ex.get("moving"):
            side = _lateral(ex["fol_dy"])
            if side is not None:
                return self._decision(side, t0)
        action = Action.FORWARD if ex["fol_dx"] > mid else Action.HOVER
        return self._decision(action, t0)


class ZoneMissionArm(MissionFsmArm):
    """Mission pilot that sidesteps restricted zones before it reaches them."""

    def __init__(self) -> None:
        super().__init__()
        self._id = "zone_pilot"

    def decide(self, obs: Observation) -> Decision:
        ex = obs.extras
        if (
            ex.get("zn_blocked")
            and ex["zn_dx"] < ZONE_LOOKAHEAD_M
            and not ex["zn_inside"]
        ):
            t0 = time.perf_counter_ns()
            if ex.get("moving"):
                return self._decision(Action.HOVER, t0)  # let the sidestep finish first
            return self._decision(Action.LEFT if ex["zn_dir"] > 0 else Action.RIGHT, t0)
        return super().decide(obs)


class SiteArm(_ArmBase):
    """Lands on a candidate site once the battery is critical.

    ``nearest`` takes the closest safe site however small; ``best`` takes the safe site
    with the largest radius (more forgiving of a late decision, but further away).
    """

    def __init__(self, policy: str) -> None:
        super().__init__(f"site_{policy}")
        self._policy = policy

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        ex = obs.extras
        if "ls_n" not in ex or obs.battery_pct >= 8.0:
            return self._decision(Action.FORWARD, t0)
        sites = [
            (ex[f"ls{k}_dx"], ex[f"ls{k}_dy"], ex[f"ls{k}_r"])
            for k in (1, 2, 3)
            if f"ls{k}_dx" in ex and not ex[f"ls{k}_hz"]
        ]
        if not sites:
            return self._decision(Action.EMERGENCY_LAND, t0)
        dx, dy, _ = (
            min(sites, key=lambda s: s[0])
            if self._policy == "nearest"
            else max(sites, key=lambda s: (s[2], -s[0]))
        )
        if abs(dx) <= SITE_LAND_DX_M and abs(dy) <= SITE_LAND_DY_M:
            return self._decision(Action.EMERGENCY_LAND, t0)
        if not ex.get("moving", 0.0):
            side = _lateral(dy, SITE_LAND_DY_M)
            if side is not None and dx > 1.0:
                return self._decision(side, t0)
        return self._decision(Action.FORWARD, t0)


class GatePilot(MissionFsmArm):
    """Flies through gates without stopping: align early, never brake."""

    def __init__(self) -> None:
        super().__init__()
        self._id = "gate_pilot"

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        ex = obs.extras
        if not ex.get("gate"):
            return super().decide(obs)
        if ex["stn_i"] > ex["stn_n"] or int(ex["st_phase"]) >= 4:
            return self._decision(Action.HOVER, t0)
        if not ex.get("moving"):
            side = _lateral(ex["st_dy"], 0.35)
            if side is not None:
                return self._decision(side, t0)
            if ex["st_dz"] > 0.25:
                return self._decision(Action.ASCEND, t0)
            if ex["st_dz"] < -0.25:
                return self._decision(Action.DESCEND, t0)
        return self._decision(Action.FORWARD, t0)


ORB_DODGE_RANGE_M = 0.9
ORB_HOLD_S = 0.4
ORB_BEARING_LIMIT_DEG = 100.0
ORB_WATCH_RANGE_M = 9.0


class OrbDodger:
    """The late-sidestep rule shared by the orb pilot and the reflex layer.

    Orbs turn at a limited rate, so a dodge started early is out-turned; this one fires once
    the orb is within ``dodge_range_m`` and holds the same side until the orb opens up.
    """

    def __init__(
        self, dodge_range_m: float = ORB_DODGE_RANGE_M, hold_s: float = ORB_HOLD_S
    ) -> None:
        self.range = dodge_range_m
        self.hold = hold_s
        self.action: Action | None = None
        self.until = -1.0

    def reset(self) -> None:
        self.action, self.until = None, -1.0

    def dodge(
        self, t: float, rng: float, bearing: float, closing: float
    ) -> Action | None:
        engaged = self.action is not None and (
            t < self.until or (rng < self.range * 1.4 and closing > -0.1)
        )
        if engaged:
            return self.action
        if rng < self.range and closing > 0.2 and abs(bearing) < ORB_BEARING_LIMIT_DEG:
            self.action = Action.AVOID_RIGHT if bearing >= 0 else Action.AVOID_LEFT
            self.until = t + self.hold
            return self.action
        self.action = None
        return None


class OrbPilot(_ArmBase):
    """Dodges homing orbs with :class:`OrbDodger` and otherwise flies the mission to the
    pad, holding its lane while an orb is in view."""

    def __init__(
        self, dodge_range_m: float = ORB_DODGE_RANGE_M, hold_s: float = ORB_HOLD_S
    ) -> None:
        super().__init__("orb_pilot")
        self._dodger = OrbDodger(dodge_range_m, hold_s)

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        if obs.battery_pct < 8.0:
            return self._decision(Action.EMERGENCY_LAND, t0)
        ex = obs.extras
        rng = ex.get("orb_range", obs.nearest_range)
        bearing = ex.get("orb_bearing_deg", obs.nearest_bearing_deg)
        closing = ex.get("orb_closing", obs.closing_speed)
        act = self._dodger.dodge(obs.t_sim, rng, bearing, closing)
        if act is not None:
            return self._decision(act, t0)
        if (rng < ORB_WATCH_RANGE_M and closing > 0.1) or "stn_n" not in ex:
            return self._decision(
                Action.FORWARD, t0
            )  # no lateral corrections near an orb
        return self._decision(MissionFsmArm._mission_action(obs), t0)


class PredictiveArm(_ArmBase):
    """Decides on where the world will be when the answer lands, not where it was measured.

    Wraps a reflex arm and advances the obstacle by ``closing_speed * expected_ms`` before
    the inner arm sees the observation (constant-velocity extrapolation of range and
    time-to-collision). It cannot know the true latency; ``expected_ms`` is its guess.
    """

    def __init__(self, inner: _ArmBase, expected_ms: float) -> None:
        super().__init__(f"predict_{inner.provider_id}_{expected_ms:.0f}")
        self._inner = inner
        self._lead_s = expected_ms / 1000.0

    def decide(self, obs: Observation) -> Decision:
        return self._inner.decide(self.project(obs))

    def project(self, obs: Observation) -> Observation:
        """The observation advanced by the expected latency."""

        if (
            obs.fwd_range >= MAX_RANGE_M * 0.98
            and obs.nearest_range >= MAX_RANGE_M * 0.98
        ):
            return obs
        closing = max(obs.closing_speed, 0.0)
        shrink = closing * self._lead_s
        return replace(
            obs,
            fwd_range=max(obs.fwd_range - shrink, 0.05),
            nearest_range=max(obs.nearest_range - shrink, 0.05),
            ttc=max(obs.ttc - self._lead_s, 0.01),
        )


def build_predictive(inner: str, expected_ms: float) -> PredictiveArm:
    arms: dict[str, _ArmBase] = {"fsm": FsmArm(), "reflex_ttc": ReflexArm()}
    return PredictiveArm(arms[inner], expected_ms)


__all__ = [
    "FollowArm",
    "GatePilot",
    "OrbDodger",
    "OrbPilot",
    "PredictiveArm",
    "SiteArm",
    "ZoneMissionArm",
    "build_predictive",
]
