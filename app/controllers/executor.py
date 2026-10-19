"""Maps discrete actions onto setpoints for the low-level PID.

Design (see docs/research/11): a *carrot* reference ``(ref_pos, ref_vel)`` is
advanced every control tick under acceleration limits and handed to
DSLPIDControl as ``target_pos`` / ``target_vel``. Three latched channels:

* x (longitudinal): cruise | stop | home
* y (lateral): a displacement target that is reached and then held
* z (vertical): a displacement target that is reached and then held

The carrot stays ahead of the drone, so braking never latches a target behind
a moving drone (which makes the stock PID reverse). EMERGENCY_LAND is an
irreversible terminal mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.domain.actions import Action

A_SLEW = 3.0  # m/s^2 normal horizontal acceleration limit
A_BRAKE = 4.5  # m/s^2 BRAKE / shield deceleration limit
A_LAT = 5.0  # m/s^2 lateral acceleration limit
A_VERT = 3.0  # m/s^2
LAT_SPEED = 1.5  # m/s for LEFT / RIGHT
LAT_SPEED_AVOID = (
    1.5  # m/s for AVOID_* (2.5 m/s destabilises the stock PID in repeated dodges)
)
LAT_STEP = 1.0  # m
LAT_STEP_AVOID = 1.2  # m
LAT_LIMIT = 3.5  # |y| clamp for lateral targets
VERT_STEP = 0.5
VERT_SPEED = 1.0
Z_MIN, Z_MAX = 0.4, 2.5
LAND_SPEED = 0.7
AVOID_SLOW_SCALE = 0.5
AVOID_SLOW_TTL_S = 1.5
SIDE_COMMIT_S = 0.5
CARROT_LEAD_XY = 0.8
CARROT_LEAD_Z = 0.4
LAND_DONE_Z = 0.12


@dataclass(slots=True)
class ExecState:
    ref_pos: np.ndarray
    ref_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x_mode: str = "cruise"
    lat_target: float | None = None
    lat_speed: float = LAT_SPEED
    z_target: float | None = None
    slow_until_s: float = -1.0
    side_lock_until_s: float = -1.0
    side_lock_dir: int = 0
    braking: bool = False
    landing: bool = False
    landed: bool = False
    last_action: Action = Action.FORWARD


class ActionExecutor:
    """Stateful action -> (target_pos, target_vel) converter."""

    def __init__(
        self,
        start_pos: np.ndarray,
        dt: float,
        v_cruise: float,
        home: tuple[float, float, float],
        initial_action: Action = Action.FORWARD,
        keep_lateral_on_stop: bool = False,
    ) -> None:
        self._keep_lateral = (
            keep_lateral_on_stop  # missions: a hold lets a sidestep finish
        )
        self._dt = dt
        self._v_cruise = v_cruise
        self._home = np.array(home, dtype=float)
        self.state = ExecState(ref_pos=np.array(start_pos, dtype=float))
        self.state.last_action = initial_action

    # --------------------------------------------------------------- commands
    def apply(self, action: Action, t_sim: float) -> None:
        """Latch ``action``; semantics per docs/research/11 table."""

        s = self.state
        if s.landing:
            return  # EMERGENCY_LAND is irreversible
        if action != s.last_action:
            s.last_action = action
        if action == Action.FORWARD:
            s.x_mode = "cruise"
            s.braking = False
        elif action in (Action.LEFT, Action.RIGHT):
            direction = 1 if action == Action.LEFT else -1
            if t_sim < s.side_lock_until_s and direction != s.side_lock_dir:
                return  # side-commit hysteresis
            self._set_lateral(direction, LAT_STEP, LAT_SPEED, t_sim)
        elif action in (Action.AVOID_LEFT, Action.AVOID_RIGHT):
            direction = 1 if action == Action.AVOID_LEFT else -1
            self._set_lateral(direction, LAT_STEP_AVOID, LAT_SPEED_AVOID, t_sim)
            s.slow_until_s = t_sim + AVOID_SLOW_TTL_S
            s.x_mode = "cruise"
            s.braking = False
        elif action == Action.ASCEND:
            s.z_target = min(self._z_ref() + VERT_STEP, Z_MAX)
        elif action == Action.DESCEND:
            s.z_target = max(self._z_ref() - VERT_STEP, Z_MIN)
        elif action == Action.BRAKE:
            s.x_mode = "stop"
            s.braking = True
            s.lat_target = s.lat_target if self._keep_lateral else None
        elif action == Action.HOVER:
            s.x_mode = "stop"
            s.braking = False
            s.lat_target = s.lat_target if self._keep_lateral else None
        elif action == Action.RETURN_HOME:
            s.x_mode = "home"
            s.braking = False
            s.lat_target = float(self._home[1])
            s.lat_speed = LAT_SPEED
            s.z_target = float(self._home[2])
        elif action == Action.EMERGENCY_LAND:
            s.landing = True
            s.x_mode = "stop"
            s.lat_target = None
            s.z_target = None

    def force_brake(self) -> None:
        """Shield override: stop forward motion at A_BRAKE.

        Lateral motion is deliberately left alone, so a sidestep already under way
        is allowed to finish and clear the obstacle.
        """

        s = self.state
        if s.landing:
            return
        s.x_mode = "stop"
        s.braking = True

    def _z_ref(self) -> float:
        return float(self.state.ref_pos[2])

    def _set_lateral(
        self, direction: int, step: float, speed: float, t_sim: float
    ) -> None:
        s = self.state
        target = float(np.clip(s.ref_pos[1] + direction * step, -LAT_LIMIT, LAT_LIMIT))
        s.lat_target = target
        s.lat_speed = speed
        s.side_lock_until_s = t_sim + SIDE_COMMIT_S
        s.side_lock_dir = direction

    def _limit_carrot_lead(self, true_pos: np.ndarray) -> None:
        """Keep a *moving* carrot within a bounded lead of the drone.

        The limit only applies per axis where the carrot is ahead of the drone in its
        direction of travel. A stationary or trailing carrot (a hold point) is never
        dragged along, so a sagging or wind-pushed drone is pulled back to its setpoint
        instead of ratcheting the setpoint down with it.
        """

        s = self.state
        lead = np.array([CARROT_LEAD_XY, CARROT_LEAD_XY, CARROT_LEAD_Z])
        gap = s.ref_pos - true_pos
        ahead = (np.sign(gap) == np.sign(s.ref_vel)) & (np.abs(s.ref_vel) > 1e-6)
        s.ref_pos = np.where(ahead, true_pos + np.clip(gap, -lead, lead), s.ref_pos)

    # ------------------------------------------------------------------- tick
    def step(self, true_pos: np.ndarray, t_sim: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance the carrot one tick; returns (target_pos, target_vel)."""

        s = self.state
        dt = self._dt
        v_des = np.zeros(3)
        a_xy = A_BRAKE if s.braking else A_SLEW
        scale = AVOID_SLOW_SCALE if t_sim < s.slow_until_s else 1.0
        if s.landing:
            v_des[2] = -LAND_SPEED
            a_xy = A_BRAKE
        else:
            if s.x_mode == "cruise":
                v_des[0] = self._v_cruise * scale
            elif s.x_mode == "home":
                dx = self._home[0] - s.ref_pos[0]
                v_des[0] = float(np.clip(dx * 1.5, -self._v_cruise, self._v_cruise))
                if abs(dx) < 0.15:
                    v_des[0] = 0.0
            if s.lat_target is not None:
                err = s.lat_target - s.ref_pos[1]
                v_des[1] = float(np.clip(err * 3.0, -s.lat_speed, s.lat_speed))
                if abs(err) < 0.02:
                    s.lat_target = None
                    v_des[1] = 0.0
            if s.z_target is not None:
                err = s.z_target - s.ref_pos[2]
                v_des[2] = float(np.clip(err * 3.0, -VERT_SPEED, VERT_SPEED))
                if abs(err) < 0.02:
                    s.z_target = None
                    v_des[2] = 0.0
        limit = np.array([a_xy, max(a_xy, A_LAT), A_VERT]) * dt
        s.ref_vel = s.ref_vel + np.clip(v_des - s.ref_vel, -limit, limit)
        s.ref_pos = s.ref_pos + s.ref_vel * dt
        self._limit_carrot_lead(true_pos)
        s.ref_pos[2] = max(s.ref_pos[2], -0.2 if s.landing else Z_MIN - 0.05)
        if s.landing and true_pos[2] < LAND_DONE_Z:
            s.landed = True
        return s.ref_pos.copy(), s.ref_vel.copy()


__all__ = [
    "A_BRAKE",
    "A_SLEW",
    "ActionExecutor",
    "ExecState",
]
