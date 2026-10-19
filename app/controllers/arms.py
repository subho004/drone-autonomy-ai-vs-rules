"""Classical decision arms (no network): hover/random sanity floors, no-avoidance
waypoint, reflex TTC avoider, rule-based FSM over the same 11 actions, and the
privileged oracle. They implement ``DecisionProvider`` so the harness treats
them exactly like Jev or an LLM; their latency is the measured compute time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.controllers.executor import A_BRAKE
from app.domain.actions import Action
from app.domain.decision import Decision
from app.domain.observation import MAX_RANGE_M, Observation, TruthState
from app.oracle.labeller import label_observation

TUNED_PATH = Path(__file__).resolve().parents[2] / "configs" / "tuned_arms.json"
T_LAT_DESIGN_S = 0.10  # latency budget assumed when deriving thresholds
A_BRK_DESIGN = 0.7 * A_BRAKE
MARGIN_M = 0.55


def _ms(t0_ns: int) -> float:
    return (time.perf_counter_ns() - t0_ns) / 1e6


def _free_side(obs: Observation) -> Action:
    """AVOID toward the side with more clearance (left = +y = ray 4)."""

    r = obs.rays
    left = min(r[2], r[3], r[4], r[5])
    right = min(r[-2], r[-3], r[-4], r[-5])
    return Action.AVOID_LEFT if left >= right else Action.AVOID_RIGHT


class _ArmBase:
    needs_truth = False

    def __init__(self, provider_id: str) -> None:
        self._id = provider_id

    @property
    def provider_id(self) -> str:
        return self._id

    def set_truth(self, truth: TruthState) -> None:
        return None

    def close(self) -> None:
        return None

    def decide(self, obs: Observation) -> Decision:
        raise NotImplementedError

    def _decision(self, action: Action, t0: int) -> Decision:
        return Decision(
            action=action, latency_ms=_ms(t0), valid=True, model_id=self._id
        )


class WaypointArm(_ArmBase):
    """Arm A: fly to the goal, no avoidance at all."""

    def __init__(self) -> None:
        super().__init__("waypoint")

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        action = Action.HOVER if obs.goal_dx < 0.3 else Action.FORWARD
        return self._decision(action, t0)


class HoverArm(_ArmBase):
    """Z0: always HOVER. Never collides; never succeeds."""

    def __init__(self) -> None:
        super().__init__("always_hover")

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        return self._decision(Action.HOVER, t0)


class RandomArm(_ArmBase):
    """Z1: uniformly random action each decision."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__("random")
        self._rng = np.random.default_rng(seed)

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        return self._decision(Action(int(self._rng.integers(len(Action)))), t0)


@dataclass(frozen=True, slots=True)
class ArmParams:
    """Tunable thresholds shared by the reflex and FSM arms (tuned on a held-out suite)."""

    t_lat_s: float = T_LAT_DESIGN_S  # latency budget assumed when deriving d_stop
    a_brk: float = A_BRK_DESIGN  # deceleration assumed for d_stop
    margin_m: float = MARGIN_M  # radius + noise + slack
    trigger_extra_m: float = 1.0  # start the dodge this far before d_stop
    commit_s: float = 0.5  # hold the chosen evasive action this long
    near_m: float = 2.5  # when stopped, dodge a blocker closer than this
    min_closing: float = 0.3


def _d_stop(prm: ArmParams, closing: float) -> float:
    return closing * prm.t_lat_s + closing**2 / (2 * prm.a_brk) + prm.margin_m


def _threat(obs: Observation, prm: ArmParams) -> tuple[bool, float, float]:
    """(blocked?, forward range, effective closing speed) from the corridor sensor."""

    fwd = obs.fwd_range
    v = max(obs.vel[0], 0.0)
    seen = fwd < MAX_RANGE_M * 0.98
    return (
        seen and (obs.closing_speed > prm.min_closing or v < 0.3),
        fwd,
        max(obs.closing_speed, v),
    )


class ReflexArm(_ArmBase):
    """B1: time-to-collision dodge, decided every control tick, with a short commit."""

    def __init__(self, params: ArmParams | None = None) -> None:
        super().__init__("reflex_ttc")
        self._p = params or load_params("reflex_ttc")
        self._commit: Action | None = None
        self._commit_until = -1.0

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        special = _special_case(obs)
        if special is not None:
            return self._decision(special, t0)
        if self._commit is not None and obs.t_sim < self._commit_until:
            return self._decision(self._commit, t0)
        blocked, fwd, closing = _threat(obs, self._p)
        if blocked and fwd < max(
            _d_stop(self._p, closing) + self._p.trigger_extra_m, self._p.near_m
        ):
            self._commit = _free_side(obs)
            self._commit_until = obs.t_sim + self._p.commit_s
            return self._decision(self._commit, t0)
        self._commit = None
        return self._decision(Action.FORWARD, t0)


class FsmArm(_ArmBase):
    """Arm E: rule-based finite-state policy over the 11 actions.

    Thresholds derive from d_stop = v*t_lat + v^2/(2*a_brk) + margin. When there is
    still room it sidesteps (AVOID_*); when too close to dodge it BRAKEs first and
    dodges from the stop; afterwards it resumes FORWARD.
    """

    def __init__(self, params: ArmParams | None = None) -> None:
        super().__init__("fsm")
        self._p = params or load_params("fsm")
        self._state = "CRUISE"
        self._until = -1.0
        self._side = Action.AVOID_LEFT

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        special = _special_case(obs)
        if special is not None:
            return self._decision(special, t0)
        if self._state == "EVADE" and obs.t_sim < self._until:
            return self._decision(self._side, t0)
        self._state = "CRUISE"
        blocked, fwd, closing = _threat(obs, self._p)
        if not blocked:
            return self._decision(Action.FORWARD, t0)
        v = max(obs.vel[0], 0.0)
        d_stop = _d_stop(self._p, closing)
        if v > 0.5 and fwd < d_stop * 0.8:
            return self._decision(Action.BRAKE, t0)  # too close to dodge: stop first
        if (v > 0.5 and fwd < d_stop + self._p.trigger_extra_m) or (
            v <= 0.5 and fwd < self._p.near_m
        ):
            self._side = _free_side(obs)
            self._state, self._until = "EVADE", obs.t_sim + self._p.commit_s
            return self._decision(self._side, t0)
        return self._decision(Action.FORWARD, t0)


def _special_case(obs: Observation) -> Action | None:
    """Battery / GPS / goal handling shared by the rule-based arms."""

    if obs.battery_pct < 8.0:
        return Action.EMERGENCY_LAND
    if not obs.gps_ok or obs.goal_dx < 0.3:
        return Action.HOVER
    return None


def load_params(name: str) -> ArmParams:
    """Tuned parameters from configs/tuned_arms.json if present, else defaults."""

    if TUNED_PATH.exists():
        table = json.loads(TUNED_PATH.read_text()).get("arms", {})
        if name in table:
            return ArmParams(**table[name]["params"])
    return ArmParams()


class MissionFsmArm(_ArmBase):
    """Rule-based mission pilot: the reference every model is compared with.

    Obstacle handling is the tuned FSM's; mission rules follow the same prompt rules the
    models are given (align with LEFT/RIGHT/ASCEND/DESCEND, brake in the NEAR zone, hold
    to scan, land at the pad).
    """

    def __init__(self) -> None:
        super().__init__("mission_fsm")
        self._avoid = FsmArm()

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        ex = obs.extras
        if "stn_n" not in ex:
            return self._avoid.decide(obs)
        if obs.battery_pct < 8.0:
            return self._decision(Action.EMERGENCY_LAND, t0)
        reactive = self._avoid.decide(obs).action
        if reactive in (Action.AVOID_LEFT, Action.AVOID_RIGHT, Action.BRAKE):
            return self._decision(reactive, t0)
        return self._decision(self._mission_action(obs), t0)

    @staticmethod
    def _mission_action(obs: Observation) -> Action:
        ex = obs.extras
        phase = int(ex["st_phase"])
        dy, dz = ex["st_dy"], ex["st_dz"]
        lat = Action.LEFT if dy > 0.5 else Action.RIGHT if dy < -0.5 else None
        alt = Action.ASCEND if dz > 0.25 else Action.DESCEND if dz < -0.25 else None
        moving = bool(ex["moving"])
        if phase == 5:  # over the pad
            return Action.EMERGENCY_LAND
        if phase == 7:  # beside the pad: sidestep onto it
            return Action.HOVER if moving else (lat or Action.HOVER)
        if phase == 6:  # braking zone before the pad
            if moving:
                return Action.HOVER
            return lat or (Action.HOVER if obs.speed > 0.4 else Action.FORWARD)
        if phase == 4:  # heading for the pad
            return Action.FORWARD if moving or lat is None else lat
        if phase == 3:  # passed the station
            return Action.FORWARD
        if moving:
            return Action.HOVER if phase >= 1 else Action.FORWARD
        if phase == 0:
            return lat or alt or Action.FORWARD
        if phase == 1:
            return Action.HOVER if obs.speed > 0.4 else Action.FORWARD
        return lat or alt or Action.HOVER  # phase HERE: correct, then hold to scan


class OracleArm(_ArmBase):
    """O: privileged truth, preferred action of the label_v1 labeller."""

    needs_truth = True

    def __init__(self, v_cruise: float = 2.0) -> None:
        super().__init__("oracle")
        self._truth: TruthState | None = None
        self._v_cruise = v_cruise

    def set_truth(self, truth: TruthState) -> None:
        self._truth = truth

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        label = label_observation(obs, self._truth, self._v_cruise)
        return self._decision(label.preferred, t0)


ERROR_HOLD_S = 0.5
ERROR_CHOICES = (
    Action.FORWARD,
    Action.LEFT,
    Action.RIGHT,
    Action.ASCEND,
    Action.DESCEND,
    Action.BRAKE,
    Action.HOVER,
    Action.AVOID_LEFT,
    Action.AVOID_RIGHT,
)


class OracleErrorArm(_ArmBase):
    """Oracle that is wrong a fixed *fraction of the time* ``error_rate``.

    Wrong episodes last ERROR_HOLD_S of sim time and are separated by exponentially
    distributed correct stretches, so the share of time in error equals ``error_rate``
    regardless of how often decisions are made. This isolates smartness from speed:
    sweep latency at a fixed accuracy. It never emits the irreversible
    EMERGENCY_LAND / RETURN_HOME as an error (those would end the flight).
    """

    needs_truth = True

    def __init__(self, error_rate: float, seed: int = 0, v_cruise: float = 2.0) -> None:
        super().__init__(f"oracle_err{error_rate:g}")
        self._truth: TruthState | None = None
        self._v = v_cruise
        self._p = error_rate
        self._rng = np.random.default_rng(seed)
        self._wrong: Action | None = None
        self._wrong_until = -1.0
        self._next_error: float | None = None

    def set_truth(self, truth: TruthState) -> None:
        self._truth = truth

    def _gap(self) -> float:
        """Length of the next correct stretch (mean chosen so error share = error_rate)."""

        if self._p <= 0.0:
            return float("inf")
        mean = ERROR_HOLD_S * (1.0 - self._p) / self._p
        return float(self._rng.exponential(mean)) if mean > 0 else 0.0

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter_ns()
        t = obs.t_sim
        label = label_observation(obs, self._truth, self._v)
        if self._next_error is None:
            self._next_error = t + self._gap()
        if self._wrong is not None and t < self._wrong_until:
            return self._decision(self._wrong, t0)
        if t >= self._next_error and label.preferred not in (
            Action.EMERGENCY_LAND,
            Action.RETURN_HOME,
        ):
            options = [a for a in ERROR_CHOICES if a not in label.acceptable]
            if options:
                self._wrong = options[int(self._rng.integers(len(options)))]
                self._wrong_until = t + ERROR_HOLD_S
                self._next_error = self._wrong_until + self._gap()
                return self._decision(self._wrong, t0)
        return self._decision(label.preferred, t0)


def build_classical_arm(
    kind: str, seed: int = 0, config: dict[str, float] | None = None
) -> _ArmBase:
    """Factory for the built-in policies by name."""

    cfg = config or {}
    table: dict[str, _ArmBase] = {}
    from app.controllers import sim_arms  # lazy: sim_arms imports this module

    extra: dict[str, _ArmBase] = {
        "follow_pilot": sim_arms.FollowArm(),
        "zone_pilot": sim_arms.ZoneMissionArm(),
        "gate_pilot": sim_arms.GatePilot(),
        "orb_pilot": sim_arms.OrbPilot(
            float(cfg.get("dodge_range_m", sim_arms.ORB_DODGE_RANGE_M)),
            float(cfg.get("hold_s", sim_arms.ORB_HOLD_S)),
        ),
        "site_nearest": sim_arms.SiteArm("nearest"),
        "site_best": sim_arms.SiteArm("best"),
    }
    if kind in extra:
        return extra[kind]
    if kind.startswith("predict_"):
        return sim_arms.build_predictive(
            kind.removeprefix("predict_"), float(cfg.get("expected_ms", 300.0))
        )
    if kind == "oracle_err":
        return OracleErrorArm(float(cfg.get("error_rate", 0.2)), seed)
    table = {
        "waypoint": WaypointArm(),
        "always_hover": HoverArm(),
        "random": RandomArm(seed),
        "reflex_ttc": ReflexArm(),
        "fsm": FsmArm(),
        "oracle": OracleArm(),
        "mission_fsm": MissionFsmArm(),
    }
    if kind not in table:
        raise ValueError(f"unknown classical arm: {kind!r}")
    return table[kind]


__all__ = [
    "ArmParams",
    "FsmArm",
    "HoverArm",
    "MissionFsmArm",
    "OracleArm",
    "OracleErrorArm",
    "RandomArm",
    "ReflexArm",
    "WaypointArm",
    "build_classical_arm",
    "load_params",
]
