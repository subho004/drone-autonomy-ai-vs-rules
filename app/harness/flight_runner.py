"""Run one flight: scenario x arm x latency policy x shield -> metrics + records.

The sim advances in control ticks (sim time only). A decision requested at
tick ``t`` is computed from the observation at ``t`` and applied at
``t + ceil(L/dt)``; the previous action stays in force meanwhile. Provider
calls are made synchronously ("pause-and-charge"): wall time spent inside a
provider never advances the sim, it is converted to ticks by the policy.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.controllers.executor import ActionExecutor
from app.controllers.shield import ReflexShield
from app.controllers.sim_arms import OrbDodger
from app.core.logging import get_logger
from app.domain.actions import Action
from app.domain.decision import Decision, DecisionProvider, DecisionSource
from app.domain.observation import Observation
from app.domain.trajectory import TRAJ_COLUMNS
from app.harness.latency import LatencyPolicy
from app.harness.orb_control import OrbController
from app.oracle.labeller import label_observation
from app.scenarios.spec import ScenarioSpec
from app.sim.mission import MissionTracker
from app.sim.tasks import FollowTracker, LandingTracker, ZoneMonitor
from app.sim.world import BATTERY_CRITICAL_PCT, DT, HOME, DroneWorld, StepInfo

logger = get_logger(__name__)

CONSECUTIVE_INVALID_HOVER = 3
PERIODIC_TICKS = 5
LANDED_SPEED_MAX = 1.0
ZONE_TOL_M = (
    0.02  # a delivery counts as clean when it never enters a zone by more than this
)
OBSTACLE_EVENTS = (
    "clear",
    "obstacle_static",
    "obstacle_approach",
    "intruder_cross",
    "pursuit",
    "canyon",
)
DEADLINE_EVENTS = ("battery_critical", "gps_loss")
EVASIVE = frozenset(
    {
        Action.BRAKE,
        Action.HOVER,
        Action.AVOID_LEFT,
        Action.AVOID_RIGHT,
        Action.LEFT,
        Action.RIGHT,
        Action.EMERGENCY_LAND,
        Action.RETURN_HOME,
    }
)


@dataclass(slots=True)
class FlightConfig:
    latency: LatencyPolicy
    shield: bool = False
    cadence: str = "sequential"  # or "periodic"
    record_trajectory: bool = True
    label_decisions: bool = True
    orbs: OrbController | None = None  # steers externally controlled orbs


@dataclass(slots=True)
class _Pending:
    obs_tick: int
    apply_tick: int
    obs: Observation
    decision: Decision
    latency_ms: float
    latency_ticks: int
    acceptable: bool | None


@dataclass(slots=True)
class FlightResult:
    metrics: dict[str, Any]
    decisions: list[dict[str, Any]]
    trajectory: dict[str, np.ndarray] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def run_flight(
    world: DroneWorld, spec: ScenarioSpec, provider: DecisionProvider, cfg: FlightConfig
) -> FlightResult:
    """Fly ``spec`` with ``provider`` and return metrics, decision rows, trajectory."""

    return _Flight(world, spec, provider, cfg).run()


class _Flight:
    """Mutable state of one flight, advanced tick by tick."""

    def __init__(
        self,
        world: DroneWorld,
        spec: ScenarioSpec,
        provider: DecisionProvider,
        cfg: FlightConfig,
    ) -> None:
        self.world, self.spec, self.provider, self.cfg = world, spec, provider, cfg
        self.wall0 = time.perf_counter()
        world.reset(spec)
        cfg.latency.reset(spec.seed)
        if cfg.orbs is not None:
            cfg.orbs.reset(spec.seed)
        self.ex = ActionExecutor(
            np.array([0.0, 0.0, spec.start_z]),
            DT,
            spec.v_cruise,
            HOME,
            keep_lateral_on_stop=(
                spec.mission is not None or spec.follow is not None or bool(spec.sites)
            ),
        )
        # in orb games the protective layer is the orb reflex, not the forward-brake shield
        self.shield = ReflexShield(enabled=cfg.shield and not self._has_orbs(spec))
        self.needs_truth = bool(getattr(provider, "needs_truth", False))
        self.max_ticks = int(spec.max_time_s / DT)
        self.traj = (
            {
                c: np.full(self.max_ticks + 1, np.nan)
                for c in (*TRAJ_COLUMNS, *self._orb_columns(spec))
            }
            if cfg.record_trajectory
            else None
        )
        self.pending: list[_Pending] = []
        self.rows: list[dict[str, Any]] = []
        self.n = 0
        self.last_applied_obs_tick = -1
        self.consecutive_invalid = 0
        self.issue_next = 0
        self.current_action = Action.FORWARD
        self.action_switches = 0
        self.api_wall_s = 0.0
        self.cost = 0.0
        self.first_evasive_tick: int | None = None
        self.resume_tick: int | None = None
        self.path_len = 0.0
        self.prev_pos = np.array([0.0, 0.0, spec.start_z])
        self.min_clear = math.inf
        self.outcome = "timeout"
        self.event_onset_t: float | None = None
        self.deadline_ok: bool | None = None
        self.t_goal: float | None = None
        self.mission = MissionTracker(spec.mission) if spec.mission else None
        self.mission_events: list[dict[str, Any]] = []
        self.follow = FollowTracker(spec.follow) if spec.follow else None
        zones = spec.mission.zones if spec.mission else []
        self.zones = ZoneMonitor(zones) if zones else None
        self.landing = LandingTracker(spec.sites) if spec.sites else None
        self.geofence_overrides = 0
        self.orb_dodger = OrbDodger() if cfg.shield and self._has_orbs(spec) else None
        self.orb_overrides = 0

    # ----------------------------------------------------------------- loop
    def run(self) -> FlightResult:
        for n in range(self.max_ticks):
            self.n = n
            if self._tick():
                break
        return self._finish()

    def _tick(self) -> bool:
        w = self.world
        t = w.t_sim
        pos, vel, _, _ = w.true_state()
        _, fwd, closing = w.sense()  # advances the closing-speed filter every tick
        self._apply_due(t)
        if self.cfg.orbs is not None:
            self.cfg.orbs.tick(self.n, w)
        if self._should_issue():
            self._issue(t)
        shield_active = self.shield.should_brake(t, fwd, closing, float(vel[0]))
        if self.orb_dodger is not None:  # reflex layer: late sidestep, zero latency
            radar = w.orb_radar()
            act = (
                self.orb_dodger.dodge(t, radar[0], radar[1], radar[2])
                if radar is not None
                else self.orb_dodger.dodge(t, 99.0, 0.0, 0.0)
            )
            if act is not None:
                self.ex.apply(act, t)
                self.orb_overrides += 1
                shield_active = True
        if shield_active:
            self.ex.force_brake()
        if self._geofence_active(pos, vel):
            self.ex.force_brake()
            self.geofence_overrides += 1
        tpos, tvel = self.ex.step(pos, t)
        info = w.step(tpos, tvel)
        npos, nvel, _, _ = w.true_state()
        self._track(npos, nvel, info)
        if self.traj is not None:
            self._record(npos, nvel, tpos, shield_active, info.clearance)
        return self._terminated(info, npos, nvel)

    def _geofence_active(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        """With the protective layer on, brake before entering a restricted zone."""

        if self.zones is None or not self.cfg.shield:
            return False
        return self.zones.predicts_breach(
            (float(pos[0]), float(pos[1]), float(pos[2])),
            float(np.hypot(vel[0], vel[1])),
        )

    # ------------------------------------------------------------ decisions
    def _should_issue(self) -> bool:
        if self.cfg.cadence == "sequential":
            return not self.pending and self.n >= self.issue_next
        return self.n % PERIODIC_TICKS == 0

    def _issue(self, t: float) -> None:
        w = self.world
        obs = self._with_mission(w.observe(int(self.current_action)))
        truth = w.truth()
        if self.needs_truth:
            self.provider.set_truth(truth)  # type: ignore[attr-defined]
        d = self.provider.decide(obs)
        self.api_wall_s += d.latency_ms / 1000.0
        self.cost += d.cost_usd
        lat_ms, ticks = self.cfg.latency.resolve(d)
        acceptable: bool | None = None
        if self.cfg.label_decisions:
            lab = label_observation(obs, truth, self.spec.v_cruise)
            acceptable = d.action in lab.acceptable if d.action is not None else False
        item = _Pending(self.n, self.n + ticks, obs, d, lat_ms, ticks, acceptable)
        if ticks == 0:
            self._apply(item, t)
        else:
            self.pending.append(item)
        self.issue_next = self.n + max(ticks, 1)

    def _with_mission(self, obs: Observation) -> Observation:
        """Attach task state (mission, follow target, zones, landing sites) to the observation."""

        extras: dict[str, float] = dict(obs.extras)  # e.g. the orb radar
        goal_dx = obs.goal_dx
        st = self.ex.state
        if self.mission is not None and self.spec.mission is not None:
            moving = st.lat_target is not None or st.z_target is not None
            extras.update(self.mission.extras(obs.pos, moving=moving))
            goal_dx = self.spec.mission.pad_x - obs.pos[0]
        if self.zones is not None:
            extras.update(self.zones.extras(obs.pos))
        moving = st.lat_target is not None or st.z_target is not None
        if self.follow is not None:
            extras.update(self.follow.extras(obs.pos, obs.vel, obs.t_sim))
            extras["moving"] = 1.0 if moving else 0.0
            goal_dx = extras["fol_dx"]
        if self.landing is not None:
            extras.update(self.landing.extras(obs.pos))
            extras["moving"] = 1.0 if moving else 0.0
            near = self.landing.ahead(obs.pos, 1)
            goal_dx = near[0][1].x - obs.pos[0] if near else 99.0
        if not extras:
            return obs
        return dataclasses.replace(obs, goal_dx=goal_dx, extras=extras)

    def _apply_due(self, t: float) -> None:
        due = sorted(
            (p for p in self.pending if p.apply_tick <= self.n),
            key=lambda q: (q.apply_tick, q.obs_tick),
        )
        self.pending = [p for p in self.pending if p.apply_tick > self.n]
        for p in due:
            if (
                p.obs_tick >= self.last_applied_obs_tick
            ):  # drop stale out-of-order results
                self._apply(p, t)

    def _apply(self, p: _Pending, t: float) -> None:
        self.last_applied_obs_tick = p.obs_tick
        d = p.decision
        if d.valid and d.action is not None:
            self.consecutive_invalid = 0
            applied, source = d.action, DecisionSource.MODEL
        else:
            self.consecutive_invalid += 1
            hover = self.consecutive_invalid >= CONSECUTIVE_INVALID_HOVER
            applied = Action.HOVER if hover else self.current_action
            source = DecisionSource.FALLBACK
        if applied != self.current_action:
            self.action_switches += 1
            if (
                self.first_evasive_tick is None
                and applied in EVASIVE
                and self.world.spawn_tick is not None
            ):
                self.first_evasive_tick = self.n
        self.current_action = applied
        self.ex.apply(applied, t)
        onset, limit = self.event_onset_t, self.spec.deadline_s
        in_window = (
            onset is not None
            and limit is not None
            and p.obs.t_sim
            >= onset  # observed after the event, not merely applied after
            and t <= onset + limit
        )
        if self.deadline_ok is None and in_window and p.acceptable:
            self.deadline_ok = True
        self.rows.append(_decision_row(len(self.rows), p, applied, source, t))

    # ------------------------------------------------------------- bookkeeping
    def _track(self, npos: np.ndarray, nvel: np.ndarray, info: StepInfo) -> None:
        w, spec = self.world, self.spec
        self.path_len += float(np.linalg.norm(npos - self.prev_pos))
        self.prev_pos = npos
        self.min_clear = min(self.min_clear, info.clearance)
        if (
            w.spawn_tick is not None
            and self.event_onset_t is None
            and spec.obstacle is not None
        ):
            self.event_onset_t = w.spawn_tick * DT
        if (
            spec.event_type == "battery_critical"
            and w.battery_pct < BATTERY_CRITICAL_PCT
        ):
            self.event_onset_t = self.event_onset_t or w.t_sim
        gps_t = spec.sensors.gps_loss_at_s
        if spec.event_type == "gps_loss" and gps_t is not None and w.t_sim >= gps_t:
            self.event_onset_t = self.event_onset_t or gps_t
        if self.mission is not None:
            speed_xy = float(np.hypot(nvel[0], nvel[1]))
            for kind, idx in self.mission.update(
                (float(npos[0]), float(npos[1]), float(npos[2])), speed_xy, DT
            ):
                self.mission_events.append({"t": w.t_sim, "kind": kind, "station": idx})
        p3 = (float(npos[0]), float(npos[1]), float(npos[2]))
        if self.follow is not None:
            for kind, _ in self.follow.update(p3, w.t_sim, DT):
                self.mission_events.append({"t": w.t_sim, "kind": kind, "station": 0})
        if self.zones is not None:
            for kind, n in self.zones.update(p3, DT):
                self.mission_events.append({"t": w.t_sim, "kind": kind, "station": n})
        resumed = info.clearance > 1.0 and nvel[0] > 0.8 * spec.v_cruise
        if self.first_evasive_tick is not None and self.resume_tick is None and resumed:
            self.resume_tick = self.n

    def _terminated(self, info: StepInfo, npos: np.ndarray, nvel: np.ndarray) -> bool:
        spec = self.spec
        if info.collision:
            self.outcome = "collision"
        elif self.ex.state.landed:
            soft = float(np.linalg.norm(nvel)) <= LANDED_SPEED_MAX
            self.outcome = "landed" if soft else "crash"
            if self.mission is not None and soft:
                on_pad = self.mission.landed_on_pad(
                    (float(npos[0]), float(npos[1]), float(npos[2]))
                )
                self.outcome = "landed_on_pad" if on_pad else "landed_off_pad"
            if self.landing is not None:
                self.outcome = (
                    self.landing.resolve(
                        (float(npos[0]), float(npos[1]), float(npos[2]))
                    )
                    if soft
                    else "crash"
                )
        elif info.ground_contact or info.out_of_bounds:
            self.outcome = "crash"
        elif self.follow is not None and (
            self.follow.lost
            or self.follow.overran
            or self.follow.finished(self.world.t_sim)
        ):
            f = self.follow
            self.outcome = "lost" if f.lost else "overran" if f.overran else "completed"
        elif (
            self.mission is not None
            and spec.mission is not None
            and spec.mission.gate_mode
            and self.mission.stations_done
        ):
            self.outcome = "gates_done"
        elif spec.event_type in OBSTACLE_EVENTS and npos[0] >= spec.goal_x:
            self.outcome, self.t_goal = "success", self.world.t_sim
        elif self._deadline_window_over():
            self.outcome = "timeout"
        else:
            return False
        return True

    def _deadline_window_over(self) -> bool:
        spec, onset = self.spec, self.event_onset_t
        if (
            spec.event_type not in DEADLINE_EVENTS
            or spec.deadline_s is None
            or onset is None
        ):
            return False
        return self.world.t_sim > onset + spec.deadline_s + 3.0

    # ----------------------------------------------------------------- output
    @staticmethod
    def _has_orbs(spec: ScenarioSpec) -> bool:
        return bool(
            spec.mission and any(o.sense_radius > 0 for o in spec.mission.obstacles)
        )

    @staticmethod
    def _orb_columns(spec: ScenarioSpec) -> list[str]:
        n = len(spec.mission.obstacles) if spec.mission else 0
        if n < 2:
            return []
        return [f"orb{k}{a}" for k in range(n) for a in "xyz"]

    def _record(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        ref: np.ndarray,
        shield: bool,
        clearance: float,
    ) -> None:
        traj, w, i = self.traj, self.world, self.n
        assert traj is not None
        traj["tick"][i], traj["t"][i] = w.tick, w.t_sim
        traj["x"][i], traj["y"][i], traj["z"][i] = pos
        traj["vx"][i], traj["vy"][i], traj["vz"][i] = vel
        traj["rx"][i], traj["ry"][i], traj["rz"][i] = ref
        traj["action"][i] = int(self.current_action)
        traj["shield"][i] = 1.0 if shield else 0.0
        if w.obstacle_active:
            traj["ox"][i], traj["oy"][i], traj["oz"][i] = w.obstacle_pos
            traj["orad"][i], traj["clearance"][i] = w.obstacle_radius, clearance
        for k, op in enumerate(w.orb_positions()):
            if op is not None and f"orb{k}x" in traj:
                traj[f"orb{k}x"][i], traj[f"orb{k}y"][i], traj[f"orb{k}z"][i] = op
        traj["battery"][i] = w.battery_pct
        traj["wind_x"][i], traj["wind_y"][i] = w.wind_now

    def _finish(self) -> FlightResult:
        steps = self.n + 1
        traj = (
            None if self.traj is None else {k: v[:steps] for k, v in self.traj.items()}
        )
        return FlightResult(
            metrics=self._metrics(), decisions=self.rows, trajectory=traj
        )

    def _metrics(self) -> dict[str, Any]:
        spec, rows = self.spec, self.rows
        unsafe = self.outcome in ("collision", "crash")
        if self.follow is not None:
            success = (
                self.outcome == "completed"
                and self.follow.metrics()["in_band_frac"] >= 0.7
            )
        elif self.landing is not None:
            success = self.outcome == "landed_safe"
        elif (
            self.mission is not None
            and spec.mission is not None
            and spec.mission.gate_mode
        ):
            done, total = self.mission.score()
            success = self.outcome == "gates_done" and done == total
        elif self.mission is not None:
            done, total = self.mission.score()
            success = self.outcome == "landed_on_pad" and done == total
            if self.zones is not None:  # a delivery that enters a no-fly zone failed
                success = success and self.zones.metrics()["max_breach_m"] < ZONE_TOL_M
        elif spec.event_type in DEADLINE_EVENTS:
            soft_land = (
                self.outcome == "landed" and spec.event_type == "battery_critical"
            )
            success = (bool(self.deadline_ok) and not unsafe) or soft_land
        else:
            success = self.outcome == "success"
        lats = [r["latency_ms"] for r in rows]
        n = len(rows)
        spawn = self.world.spawn_tick
        reaction = None
        if self.first_evasive_tick is not None and spawn is not None:
            reaction = (self.first_evasive_tick - spawn) * DT
        recovery = None
        if self.resume_tick is not None and self.first_evasive_tick is not None:
            recovery = (self.resume_tick - self.first_evasive_tick) * DT
        return {
            "outcome": self.outcome,
            "success": int(success),
            "collision": int(unsafe),
            "t_end": self.world.t_sim,
            "time_to_goal": self.t_goal,
            "path_length": self.path_len,
            "min_clearance": None if math.isinf(self.min_clear) else self.min_clear,
            "reaction_time_s": reaction,
            "recovery_time_s": recovery,
            "n_decisions": n,
            "lat_mean_ms": float(np.mean(lats)) if lats else None,
            "lat_p95_ms": float(np.percentile(lats, 95)) if lats else None,
            "invalid_rate": sum(1 for r in rows if not r["valid"]) / n if n else 0.0,
            "fallback_rate": sum(1 for r in rows if r["source"] == "fallback") / n
            if n
            else 0.0,
            "shield_overrides": self.shield.overrides,
            "action_switches": self.action_switches,
            "cost_usd": self.cost,
            "sim_wall_s": time.perf_counter() - self.wall0,
            "api_wall_s": self.api_wall_s,
            "stations_scanned": self.mission.score()[0] if self.mission else None,
            "stations_total": self.mission.score()[1] if self.mission else None,
            "events_json": json.dumps(self.mission_events)
            if (self.mission or self.follow)
            else None,
            "task_json": self._task_json(),
        }

    def _task_json(self) -> str | None:
        data: dict[str, Any] = {}
        if self.follow is not None:
            data.update(self.follow.metrics())
        if self.zones is not None:
            data.update(self.zones.metrics())
            data["geofence_overrides"] = self.geofence_overrides
        if self.landing is not None:
            data.update(self.landing.metrics())
        return json.dumps(data) if data else None


def _decision_row(
    seq: int, p: _Pending, applied: Action, source: DecisionSource, t_apply: float
) -> dict[str, Any]:
    d = p.decision
    probs = (
        json.dumps({a.name: round(v, 4) for a, v in d.probabilities.items()})
        if d.probabilities
        else None
    )
    return {
        "seq": seq,
        "obs_tick": p.obs_tick,
        "apply_tick": p.apply_tick,
        "t_obs": p.obs.t_sim,
        "t_apply": t_apply,
        "action": int(d.action) if d.action is not None else None,
        "applied_action": int(applied),
        "source": source.value,
        "valid": int(d.valid),
        "latency_ms": p.latency_ms,
        "latency_ticks": p.latency_ticks,
        "confidence": d.confidence,
        "probs_json": probs,
        "acceptable": None if p.acceptable is None else int(p.acceptable),
        "tokens_in": d.tokens_in,
        "tokens_out": d.tokens_out,
        "cost_usd": d.cost_usd,
        "from_cache": int(d.from_cache),
        "raw": (d.raw or d.error or None) if not d.valid else None,
    }


__all__ = ["FlightConfig", "FlightResult", "run_flight"]
