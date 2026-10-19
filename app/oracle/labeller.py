"""Ground-truth action labels (``label_v1``).

For one observation + privileged truth, returns the set of *acceptable*
actions, a preferred action and an urgency. Acceptability for threats comes
from short point-mass rollouts of each action under the executor's own
constants, so labels agree with the real controller to first order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.controllers.executor import (
    A_BRAKE,
    A_LAT,
    A_SLEW,
    LAT_SPEED,
    LAT_SPEED_AVOID,
    LAT_STEP,
    LAT_STEP_AVOID,
)
from app.domain.actions import Action
from app.domain.observation import Observation, TruthState

LABEL_VERSION = "label_v1"
DRONE_R = 0.06
SAFETY_M = 0.12
HORIZON_S = 2.6
SIM_DT = 0.05
TAU = 0.22  # drone velocity time constant measured in docs/research/11
GOAL_REACHED_M = 0.3
CRITICAL_PCT = 8.0

ACTIONS_THAT_DODGE = (Action.AVOID_LEFT, Action.AVOID_RIGHT, Action.LEFT, Action.RIGHT)
PREFERENCE = (
    Action.FORWARD,
    Action.AVOID_LEFT,
    Action.AVOID_RIGHT,
    Action.LEFT,
    Action.RIGHT,
    Action.BRAKE,
    Action.HOVER,
)


@dataclass(frozen=True, slots=True)
class Label:
    acceptable: frozenset[Action]
    preferred: Action
    urgency: float  # seconds of slack before the preferred action stops being safe (<=0: now)
    safe_now: bool


def _rollout_min_clearance(
    action: Action,
    pos: tuple[float, float],
    vel: tuple[float, float],
    obs_pos: tuple[float, float],
    obs_vel: tuple[float, float],
    obs_r: float,
    v_cruise: float,
    start_slow: bool,
) -> float:
    """Minimum surface clearance over the horizon for a latched ``action``."""

    x, y = pos
    vx, vy = vel
    ox, oy = obs_pos
    ovx, ovy = obs_vel
    lat_target: float | None = None
    lat_speed = LAT_SPEED
    slow_until = -1.0
    ref_vx = vx
    ref_vy = vy
    x_mode = "cruise"
    a_x = A_SLEW
    if action == Action.BRAKE:
        x_mode, a_x = "stop", A_BRAKE
    elif action == Action.HOVER:
        x_mode = "stop"
    elif action in (Action.AVOID_LEFT, Action.AVOID_RIGHT):
        d = 1.0 if action == Action.AVOID_LEFT else -1.0
        lat_target, lat_speed, slow_until = y + d * LAT_STEP_AVOID, LAT_SPEED_AVOID, 1.5
    elif action in (Action.LEFT, Action.RIGHT):
        d = 1.0 if action == Action.LEFT else -1.0
        lat_target, lat_speed = y + d * LAT_STEP, LAT_SPEED
    best = math.inf
    t = 0.0
    while t < HORIZON_S:
        vdx = 0.0 if x_mode == "stop" else v_cruise * (0.5 if t < slow_until else 1.0)
        vdy = 0.0
        if lat_target is not None:
            err = lat_target - y
            vdy = max(-lat_speed, min(lat_speed, err * 3.0))
        ref_vx += max(-a_x * SIM_DT, min(a_x * SIM_DT, vdx - ref_vx))
        ref_vy += max(-A_LAT * SIM_DT, min(A_LAT * SIM_DT, vdy - ref_vy))
        vx += (ref_vx - vx) * (SIM_DT / TAU)
        vy += (ref_vy - vy) * (SIM_DT / TAU)
        x += vx * SIM_DT
        y += vy * SIM_DT
        ox += ovx * SIM_DT
        oy += ovy * SIM_DT
        best = min(best, math.hypot(x - ox, y - oy) - obs_r - DRONE_R)
        t += SIM_DT
    _ = start_slow
    return best


def label_observation(
    obs: Observation,
    truth: TruthState | None,
    v_cruise: float,
) -> Label:
    """Return the ``label_v1`` label for one decision point."""

    if obs.battery_pct < CRITICAL_PCT:
        return Label(
            frozenset({Action.EMERGENCY_LAND, Action.RETURN_HOME}),
            Action.EMERGENCY_LAND,
            0.0,
            False,
        )
    if not obs.gps_ok:
        return Label(
            frozenset(
                {Action.HOVER, Action.BRAKE, Action.EMERGENCY_LAND, Action.RETURN_HOME}
            ),
            Action.HOVER,
            0.0,
            False,
        )
    if obs.goal_dx < GOAL_REACHED_M:
        return Label(frozenset({Action.HOVER, Action.BRAKE}), Action.HOVER, 0.0, True)
    if truth is None or truth.obstacle_pos is None or truth.obstacle_vel is None:
        return Label(frozenset({Action.FORWARD}), Action.FORWARD, math.inf, True)

    pos = (truth.pos[0], truth.pos[1])
    vel = (truth.vel[0], truth.vel[1])
    o_pos = (truth.obstacle_pos[0], truth.obstacle_pos[1])
    o_vel = (truth.obstacle_vel[0], truth.obstacle_vel[1])
    r = truth.obstacle_radius
    margin = SAFETY_M
    clear: dict[Action, float] = {}
    for a in (
        Action.FORWARD,
        Action.BRAKE,
        Action.HOVER,
        Action.AVOID_LEFT,
        Action.AVOID_RIGHT,
        Action.LEFT,
        Action.RIGHT,
    ):
        clear[a] = _rollout_min_clearance(a, pos, vel, o_pos, o_vel, r, v_cruise, False)
    safe = {a for a, c in clear.items() if c > margin}
    stopped = math.hypot(*vel) < 0.3
    ahead = o_pos[0] > pos[0] and abs(o_pos[1] - pos[1]) < r + 0.35
    if stopped and ahead and o_pos[0] - pos[0] < 4.0:
        # Blocked and stationary: waiting is safe but makes no progress; dodge.
        acceptable = frozenset(a for a in safe if a in ACTIONS_THAT_DODGE)
        if not acceptable:
            acceptable = frozenset({Action.AVOID_LEFT, Action.AVOID_RIGHT})
        preferred = _pick(acceptable, clear)
        return Label(acceptable, preferred, 0.0, False)
    if Action.FORWARD in safe:
        return Label(frozenset(safe), Action.FORWARD, _slack(clear), True)
    if safe:
        return Label(frozenset(safe), _pick(frozenset(safe), clear), 0.0, False)
    best = max(clear, key=lambda a: clear[a])
    return Label(frozenset({best}), best, -1.0, False)


def _pick(candidates: frozenset[Action], clear: dict[Action, float]) -> Action:
    for a in PREFERENCE:
        if a in candidates:
            if a in (Action.AVOID_LEFT, Action.AVOID_RIGHT):
                pair = [
                    c
                    for c in (Action.AVOID_LEFT, Action.AVOID_RIGHT)
                    if c in candidates
                ]
                return max(pair, key=lambda c: clear[c])
            return a
    return next(iter(candidates))


def _slack(clear: dict[Action, float]) -> float:
    """Crude slack: how far the FORWARD rollout stays from the safety margin."""

    return max(0.0, clear[Action.FORWARD] - SAFETY_M)


__all__ = ["LABEL_VERSION", "Label", "label_observation"]
