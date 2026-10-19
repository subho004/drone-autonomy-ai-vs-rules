"""Two-drone shared airspace: headless PyBullet world, per-drone sensing and a pair runner.

Two Crazyflies (``gym-pybullet-drones`` with ``num_drones=2``, plain ``Physics.PYB``;
``PYB_DW`` crashed a drone in the feasibility spike) each run their own ``DSLPIDControl``
and ``ActionExecutor``. Every drone works in its *own* frame: x along its heading toward
its goal, y to its left, z up. That keeps the single-drone arms and prompt encoder
reusable unchanged. Frame conversion happens only at the PID boundary.

The other drone appears in the observation two ways: as a return in the 16-ray ring
(an analytic ray-disc test, because a real 0.06 m body would slip between 22.5 degree
rays) so ``ReflexArm`` / ``FsmArm`` react to it, and as ``Observation.extras``
(``other_*`` keys) for anything that wants the numbers directly.

Scenarios (seeded, deterministic):

* ``head_on``: drones 10 m apart at the same altitude, each flying toward the other's
  start point at 1.5 m/s; one drone's lane is shifted by a seeded 0 to 0.3 m.
* ``perpendicular``: A flies +x, B flies +y through the same crossing point, with a
  seeded 0 to 0.3 m lane offset and start-time (along-track) offset.

``handedness="shared"`` mirrors drone B's lateral axis so "left" means the same world
side for both drones, the case where a symmetric controller makes both dodge the same way.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from app.controllers.executor import ActionExecutor
from app.core.logging import get_logger
from app.domain.actions import Action
from app.domain.decision import Decision, DecisionProvider
from app.domain.observation import MAX_RANGE_M, N_RAYS, N_SECTORS, Observation
from app.harness.latency import LatencyMode, LatencyPolicy
from app.sim.rng import channel_generator

logger = get_logger(__name__)

PYB_FREQ = 240
CTRL_FREQ = 48
DT = 1.0 / CTRL_FREQ
SETTLE_TICKS = 24  # half a second of hover before the flight clock starts
DRONE_RADIUS = 0.06  # CF2X collision cylinder radius: contact at 0.12 m centre distance
SENSE_RADIUS_M = 0.45  # effective radius of the other drone in the ray ring
CORRIDOR_HALF_WIDTH = 0.6  # same forward corridor as the single-drone world
FORWARD_RAYS = (0, 1, 2, N_RAYS - 1, N_RAYS - 2)
N_DRONES = 2
GOAL_TOL_M = 0.5
GROUND_Z = 0.06
NO_TTC = 99.0
CONTACT_TOL_M = (
    0.02  # a ground crash after the drones came this close counts as a collision
)
NEAR_MISS_CLEARANCE_M = (
    0.3  # surface clearance below this, without contact, is a near miss
)

# Deadlock: both drones (not yet at goal) make < STUCK_PROGRESS_M of forward progress
# over STUCK_WINDOW_S while within STUCK_RANGE_M, continuously for DEADLOCK_S.
STUCK_RANGE_M = 3.0
STUCK_WINDOW_S = 1.0
STUCK_PROGRESS_M = 0.45
DEADLOCK_S = 4.0

LATERAL_ACTIONS = {
    Action.LEFT: 1,
    Action.AVOID_LEFT: 1,
    Action.RIGHT: -1,
    Action.AVOID_RIGHT: -1,
}

Scenario = Literal["head_on", "perpendicular"]
Handedness = Literal["own", "shared"]


@dataclass(frozen=True, slots=True)
class TwoDroneSpec:
    """One seeded two-drone encounter."""

    seed: int
    scenario: Scenario = "head_on"
    handedness: Handedness = "own"
    separation_m: float = 10.0
    v_cruise: float = 1.5
    start_z: float = 1.0
    max_offset_m: float = 0.3
    max_time_s: float = 20.0
    sense_range_m: float = (
        4.0  # the other drone is invisible beyond this centre-to-surface range
    )


@dataclass(frozen=True, slots=True)
class DroneLayout:
    """Where a drone starts and how its private frame sits in the world."""

    start: np.ndarray  # (3,) world
    heading: np.ndarray  # (2,) unit vector, local +x
    lateral: np.ndarray  # (2,) unit vector, local +y
    goal_dist: float

    @property
    def basis(self) -> np.ndarray:
        """Rows are the local x and y axes in world xy."""

        return np.stack([self.heading, self.lateral])


def build_layouts(spec: TwoDroneSpec) -> list[DroneLayout]:
    """Start points and frames of drones A and B (seeded, no physics)."""

    gen = channel_generator(spec.seed, "two_drone_layout")
    offset = float(gen.uniform(0.0, spec.max_offset_m)) * float(gen.choice([-1.0, 1.0]))
    along = float(gen.uniform(-spec.max_offset_m, spec.max_offset_m))
    h = spec.separation_m / 2.0
    z = spec.start_z
    if spec.scenario == "head_on":
        a_start, a_head = (-h, 0.0, z), (1.0, 0.0)
        b_start, b_head = (h, offset, z), (-1.0, 0.0)
    else:
        a_start, a_head = (-h, 0.0, z), (1.0, 0.0)
        b_start, b_head = (offset, -h + along, z), (0.0, 1.0)
    layouts: list[DroneLayout] = []
    for k, (start, head) in enumerate(((a_start, a_head), (b_start, b_head))):
        hv = np.array(head)
        left = np.array([-hv[1], hv[0]])
        if spec.handedness == "shared" and k == 1:
            left = -left
        layouts.append(DroneLayout(np.array(start), hv, left, spec.separation_m))
    return layouts


def ray_ring(rel_x: float, rel_y: float, visible: bool) -> np.ndarray:
    """Ranges of the 16-ray ring against a disc of ``SENSE_RADIUS_M`` at (rel_x, rel_y)."""

    out = np.full(N_RAYS, MAX_RANGE_M)
    if not visible:
        return out
    c2 = rel_x * rel_x + rel_y * rel_y
    if c2 < SENSE_RADIUS_M**2:
        return np.zeros(N_RAYS)
    ang = np.arange(N_RAYS) * (2.0 * math.pi / N_RAYS)
    b = rel_x * np.cos(ang) + rel_y * np.sin(ang)
    disc = b * b - (c2 - SENSE_RADIUS_M**2)
    hit = (disc >= 0.0) & (b > 0.0)
    t = b - np.sqrt(np.where(hit, disc, 0.0))
    return np.where(hit, np.clip(t, 0.0, MAX_RANGE_M), out)


def corridor_range(rays: np.ndarray) -> float:
    """Nearest return inside the forward corridor (mirrors the single-drone world)."""

    best = MAX_RANGE_M
    for k in FORWARD_RAYS:
        r = float(rays[k])
        if r >= MAX_RANGE_M * 0.98:
            continue
        ang = k * 2.0 * math.pi / N_RAYS
        if abs(r * math.sin(ang)) < CORRIDOR_HALF_WIDTH:
            best = min(best, r * math.cos(ang))
    return best


@dataclass(slots=True)
class PairStep:
    """What happened in one control tick."""

    collision: bool
    contact: bool
    ground: bool
    separation: float  # centre distance
    clearance: float  # centre distance minus two drone radii


class TwoDroneWorld:
    """One headless PyBullet client with two drones, flown scenario after scenario."""

    def __init__(self) -> None:
        self._env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=N_DRONES,
            initial_xyzs=np.array([[-5.0, 0.0, 1.0], [5.0, 0.0, 1.0]]),
            initial_rpys=np.zeros((N_DRONES, 3)),
            physics=Physics.PYB,
            pyb_freq=PYB_FREQ,
            ctrl_freq=CTRL_FREQ,
            gui=False,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        self._client = self._env.getPyBulletClient()
        self._ids = [int(i) for i in self._env.getDroneIds()]
        self._ctrls = [
            DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(N_DRONES)
        ]
        self._executors: list[ActionExecutor] = []
        self._layouts: list[DroneLayout] = []
        self._spec: TwoDroneSpec | None = None
        self.tick = 0

    # ------------------------------------------------------------------ setup
    def reset(self, spec: TwoDroneSpec) -> None:
        """Load ``spec``: reset physics, settle both drones at their start points."""

        self._spec = spec
        self._layouts = build_layouts(spec)
        self._env.INIT_XYZS = np.stack([lay.start for lay in self._layouts])
        self._env.reset(seed=spec.seed)
        for ctrl in self._ctrls:
            ctrl.reset()
        self.tick = 0
        self._executors = [
            ActionExecutor(
                np.array([0.0, 0.0, spec.start_z]),
                DT,
                spec.v_cruise,
                (0.0, 0.0, spec.start_z),
            )
            for _ in range(N_DRONES)
        ]
        hold = np.stack([lay.start for lay in self._layouts])
        for _ in range(SETTLE_TICKS):
            self._advance(hold, np.zeros((N_DRONES, 3)))

    @property
    def spec(self) -> TwoDroneSpec:
        if self._spec is None:
            raise RuntimeError("world not reset")
        return self._spec

    @property
    def layouts(self) -> list[DroneLayout]:
        return self._layouts

    @property
    def t_sim(self) -> float:
        return self.tick * DT

    def executor(self, i: int) -> ActionExecutor:
        return self._executors[i]

    def apply(self, i: int, action: Action) -> None:
        """Latch ``action`` on drone ``i``'s executor."""

        self._executors[i].apply(action, self.t_sim)

    # ------------------------------------------------------------------ frames
    def world_state(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self._env.pos[i].copy(), self._env.vel[i].copy()

    def _to_local(self, i: int, pos_w: np.ndarray) -> np.ndarray:
        lay = self._layouts[i]
        xy = lay.basis @ (pos_w[:2] - lay.start[:2])
        return np.array([xy[0], xy[1], pos_w[2]])

    def _vec_to_local(self, i: int, vec_w: np.ndarray) -> np.ndarray:
        xy = self._layouts[i].basis @ vec_w[:2]
        return np.array([xy[0], xy[1], vec_w[2]])

    def _to_world(self, i: int, pos_l: np.ndarray) -> np.ndarray:
        lay = self._layouts[i]
        xy = lay.start[:2] + lay.basis.T @ pos_l[:2]
        return np.array([xy[0], xy[1], pos_l[2]])

    def _vec_to_world(self, i: int, vec_l: np.ndarray) -> np.ndarray:
        xy = self._layouts[i].basis.T @ vec_l[:2]
        return np.array([xy[0], xy[1], vec_l[2]])

    def local_state(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        pos, vel = self.world_state(i)
        return self._to_local(i, pos), self._vec_to_local(i, vel)

    # ----------------------------------------------------------------- sensing
    def observe(
        self, i: int, last_action: int, decision_age_ms: float = 0.0
    ) -> Observation:
        """Drone ``i``'s observation: own state plus the other drone in rays and extras."""

        j = 1 - i
        pos_l, vel_l = self.local_state(i)
        rel_w = self._env.pos[j] - self._env.pos[i]
        rel_v = self._env.vel[j] - self._env.vel[i]
        rel_l = self._vec_to_local(i, rel_w)
        dist = float(np.linalg.norm(rel_w))
        closing = float(-np.dot(rel_w, rel_v) / dist) if dist > 1e-6 else 0.0
        visible = abs(float(rel_w[2])) < SENSE_RADIUS_M
        rays = ray_ring(float(rel_l[0]), float(rel_l[1]), visible)
        rays = np.where(rays > self.spec.sense_range_m, MAX_RANGE_M, rays)
        seen = bool(np.any(rays < MAX_RANGE_M * 0.98))
        if not seen:
            closing = 0.0
        fwd = corridor_range(rays)
        surface = max(dist - 2.0 * DRONE_RADIUS, 0.0)
        ttc = min(surface / closing, NO_TTC) if closing > 0.1 else NO_TTC
        bearing = math.degrees(math.atan2(float(rel_l[1]), float(rel_l[0])))
        k_near = int(np.argmin(rays))
        sectors = tuple(
            float(
                min(
                    rays[(2 * k - 1) % N_RAYS],
                    rays[2 * k % N_RAYS],
                    rays[(2 * k + 1) % N_RAYS],
                )
            )
            for k in range(N_SECTORS)
        )
        other_vel = self._vec_to_local(i, self._env.vel[j])
        extras = {"other_visible": float(seen)}
        extras.update(
            {
                "other_range": float(min(surface, MAX_RANGE_M)),
                "other_bearing_deg": bearing,
                "other_closing": closing,
                "other_ttc": float(ttc),
                "other_rel_x": float(rel_l[0]),
                "other_rel_y": float(rel_l[1]),
                "other_vx": float(other_vel[0]),
                "other_vy": float(other_vel[1]),
            }
            if seen
            else {
                "other_range": MAX_RANGE_M,
                "other_bearing_deg": 0.0,
                "other_closing": 0.0,
                "other_ttc": NO_TTC,
                "other_rel_x": MAX_RANGE_M,
                "other_rel_y": 0.0,
                "other_vx": 0.0,
                "other_vy": 0.0,
            }
        )
        return Observation(
            tick=self.tick,
            t_sim=self.t_sim,
            pos=(float(pos_l[0]), float(pos_l[1]), float(pos_l[2])),
            vel=(float(vel_l[0]), float(vel_l[1]), float(vel_l[2])),
            goal_dx=float(self._layouts[i].goal_dist - pos_l[0]),
            home_dist=float(math.hypot(pos_l[0], pos_l[1])),
            rays=tuple(float(r) for r in rays),
            sectors=sectors,
            nearest_range=float(rays[k_near]),
            nearest_bearing_deg=float(
                (math.degrees(k_near * 2.0 * math.pi / N_RAYS) + 180.0) % 360.0 - 180.0
            ),
            closing_speed=closing,
            ttc=float(ttc),
            battery_pct=100.0,
            gps_ok=True,
            last_action=last_action,
            decision_age_ms=decision_age_ms,
            fwd_range=float(fwd),
            extras=extras,
        )

    # -------------------------------------------------------------------- step
    def step(self) -> PairStep:
        """Advance one control tick: executors -> world targets -> PIDs -> physics."""

        t = self.t_sim
        tpos = np.zeros((N_DRONES, 3))
        tvel = np.zeros((N_DRONES, 3))
        for i in range(N_DRONES):
            pos_l, _ = self.local_state(i)
            ref_pos, ref_vel = self._executors[i].step(pos_l, t)
            tpos[i] = self._to_world(i, ref_pos)
            tvel[i] = self._vec_to_world(i, ref_vel)
        info = self._advance(tpos, tvel)
        self.tick += 1
        return info

    def _advance(self, tpos: np.ndarray, tvel: np.ndarray) -> PairStep:
        env = self._env
        rpms = np.zeros((N_DRONES, 4))
        for i in range(N_DRONES):
            rpm, _, _ = self._ctrls[i].computeControl(
                control_timestep=DT,
                cur_pos=env.pos[i],
                cur_quat=env.quat[i],
                cur_vel=env.vel[i],
                cur_ang_vel=env.ang_v[i],
                target_pos=tpos[i],
                target_rpy=np.zeros(3),
                target_vel=tvel[i],
            )
            rpms[i] = rpm
        env.step(rpms)
        return self._step_info()

    def _step_info(self) -> PairStep:
        pos = self._env.pos
        sep = float(np.linalg.norm(pos[0] - pos[1]))
        clearance = sep - 2.0 * DRONE_RADIUS
        contact = bool(
            p.getContactPoints(
                bodyA=self._ids[0], bodyB=self._ids[1], physicsClientId=self._client
            )
        )
        ground = bool(np.any(pos[:, 2] < GROUND_Z))
        return PairStep(contact or clearance <= 0.0, contact, ground, sep, clearance)

    def close(self) -> None:
        self._env.close()


# ====================================================================== flights
@dataclass(slots=True)
class Pilot:
    """One drone's decision provider and injected latency."""

    provider: DecisionProvider
    latency: LatencyPolicy
    name: str = ""


def fixed_latency_pilot(
    provider: DecisionProvider, latency_ms: float, name: str = ""
) -> Pilot:
    """Pilot whose every decision takes exactly ``latency_ms`` of sim time."""

    return Pilot(
        provider,
        LatencyPolicy(LatencyMode.FIXED, fixed_ms=latency_ms),
        name or provider.provider_id,
    )


@dataclass(slots=True)
class _Pending:
    obs_tick: int
    apply_tick: int
    decision: Decision
    latency_ms: float
    latency_ticks: int


@dataclass(slots=True)
class PairResult:
    """Outcome metrics, per-tick trajectory and decision log of one encounter."""

    metrics: dict[str, Any]
    traj: dict[str, np.ndarray]
    decisions: list[list[dict[str, Any]]] = field(default_factory=list)


class _DecisionStream:
    """One drone's sequential decision loop (the single-drone harness semantics).

    A request is issued only when none is in flight; the answer is applied
    ``max(1, ceil(L/dt))`` ticks later and the previous action stays in force meanwhile.
    """

    def __init__(self, pilot: Pilot, seed: int) -> None:
        self.pilot = pilot
        pilot.latency.reset(seed)
        self.pending: _Pending | None = None
        self.current = Action.FORWARD
        self.rows: list[dict[str, Any]] = []
        self.switches = 0

    def progress(self, n: int) -> float:
        """0..1 completion of the in-flight request, NaN when none."""

        pen = self.pending
        if pen is None or pen.apply_tick <= pen.obs_tick:
            return float("nan")
        return min((n - pen.obs_tick) / (pen.apply_tick - pen.obs_tick), 1.0)

    def tick(self, world: TwoDroneWorld, i: int, n: int) -> None:
        pen = self.pending
        if pen is not None and pen.apply_tick <= n:
            self._apply(world, i, pen, n)
            self.pending = None
        if self.pending is None:
            obs = world.observe(i, int(self.current))
            decision = self.pilot.provider.decide(obs)
            ms, ticks = self.pilot.latency.resolve(decision)
            item = _Pending(n, n + ticks, decision, ms, ticks)
            if ticks == 0:
                self._apply(world, i, item, n)
            else:
                self.pending = item

    def _apply(self, world: TwoDroneWorld, i: int, pen: _Pending, n: int) -> None:
        d = pen.decision
        applied = d.action if d.valid and d.action is not None else self.current
        if applied != self.current:
            self.switches += 1
        self.current = applied
        world.apply(i, applied)
        self.rows.append(
            {
                "obs_tick": pen.obs_tick,
                "apply_tick": n,
                "latency_ms": pen.latency_ms,
                "latency_ticks": pen.latency_ticks,
                "action": int(applied),
            }
        )


class _DeadlockMonitor:
    """Flags both drones making almost no forward progress while close for DEADLOCK_S."""

    def __init__(self) -> None:
        self._hist: list[list[float]] = [[], []]
        self.stuck_s = 0.0
        self.flagged_at: float | None = None

    def update(
        self, t: float, x_local: tuple[float, float], sep: float, reached: list[bool]
    ) -> None:
        win = int(STUCK_WINDOW_S / DT)
        stuck = sep < STUCK_RANGE_M and not any(reached)
        for i in range(N_DRONES):
            self._hist[i].append(x_local[i])
            if (
                len(self._hist[i]) <= win
                or x_local[i] - self._hist[i][-1 - win] >= STUCK_PROGRESS_M
            ):
                stuck = False
        self.stuck_s = self.stuck_s + DT if stuck else 0.0
        if self.flagged_at is None and self.stuck_s > DEADLOCK_S:
            self.flagged_at = t


def run_pair(
    world: TwoDroneWorld, spec: TwoDroneSpec, pilots: tuple[Pilot, Pilot]
) -> PairResult:
    """Fly one encounter with ``pilots`` (A, B) and return metrics, trajectory, decisions."""

    wall0 = time.perf_counter()
    world.reset(spec)
    streams = [_DecisionStream(pl, spec.seed) for pl in pilots]
    layouts = world.layouts
    max_ticks = int(spec.max_time_s / DT)
    mon = _DeadlockMonitor()
    rec: dict[str, list[Any]] = {
        k: [] for k in ("t", "pos", "vel", "action", "progress", "sep")
    }
    reached_t: list[float | None] = [None, None]
    first_side: list[int] = [0, 0]
    min_sep, min_clear, min_clear_t = math.inf, math.inf, 0.0
    outcome = "timeout"
    for n in range(max_ticks):
        for i in range(N_DRONES):
            streams[i].tick(world, i, n)
            if first_side[i] == 0:
                first_side[i] = LATERAL_ACTIONS.get(streams[i].current, 0)
        info = world.step()
        t = world.t_sim
        rec["t"].append(t)
        rec["pos"].append([world.world_state(i)[0] for i in range(N_DRONES)])
        rec["vel"].append([world.world_state(i)[1] for i in range(N_DRONES)])
        rec["action"].append([int(s.current) for s in streams])
        rec["progress"].append([s.progress(n + 1) for s in streams])
        rec["sep"].append(info.separation)
        min_sep = min(min_sep, info.separation)
        if info.clearance < min_clear:
            min_clear, min_clear_t = info.clearance, t
        xs = (world.local_state(0)[0][0], world.local_state(1)[0][0])
        for i in range(N_DRONES):
            if reached_t[i] is None and xs[i] >= layouts[i].goal_dist - GOAL_TOL_M:
                reached_t[i] = t
        mon.update(t, xs, info.separation, [r is not None for r in reached_t])
        if info.collision:
            outcome = "collision"
        elif info.ground:
            # a drone knocked to the floor by a touch the contact test just missed is a collision
            outcome = "collision" if min_clear < CONTACT_TOL_M else "crash"
        elif mon.flagged_at is not None:
            outcome = "deadlock"
        elif all(r is not None for r in reached_t):
            outcome = "success"
        else:
            continue
        break
    metrics = _metrics(
        world,
        outcome,
        min_sep,
        min_clear,
        min_clear_t,
        reached_t,
        mon,
        first_side,
        streams,
    )
    metrics["sim_wall_s"] = time.perf_counter() - wall0
    traj = {
        "t": np.array(rec["t"]),
        "pos": np.array(rec["pos"]),
        "vel": np.array(rec["vel"]),
        "action": np.array(rec["action"], dtype=int),
        "progress": np.array(rec["progress"]),
        "sep": np.array(rec["sep"]),
    }
    return PairResult(metrics, traj, [s.rows for s in streams])


def _metrics(
    world: TwoDroneWorld,
    outcome: str,
    min_sep: float,
    min_clear: float,
    min_clear_t: float,
    reached_t: list[float | None],
    mon: _DeadlockMonitor,
    first_side: list[int],
    streams: list[_DecisionStream],
) -> dict[str, Any]:
    lay = world.layouts
    dodge = [first_side[i] * lay[i].lateral for i in range(N_DRONES)]
    both_dodged = first_side[0] != 0 and first_side[1] != 0
    same_way = bool(both_dodged and float(np.dot(dodge[0], dodge[1])) > 0.5)
    return {
        "outcome": outcome,
        "collision": int(outcome == "collision"),
        "min_separation_m": min_sep,
        "min_clearance_m": min_clear,
        "min_clearance_t": min_clear_t,
        "near_miss": int(outcome != "collision" and min_clear < NEAR_MISS_CLEARANCE_M),
        "both_reached": int(all(r is not None for r in reached_t)),
        "reached_a": int(reached_t[0] is not None),
        "reached_b": int(reached_t[1] is not None),
        "deadlock": int(mon.flagged_at is not None),
        "deadlock_t": mon.flagged_at,
        "dodged_same_way": int(same_way),
        "dodged_a": first_side[0],
        "dodged_b": first_side[1],
        "t_end": world.t_sim,
        "n_decisions_a": len(streams[0].rows),
        "n_decisions_b": len(streams[1].rows),
        "switches_a": streams[0].switches,
        "switches_b": streams[1].switches,
    }


__all__ = [
    "CTRL_FREQ",
    "DRONE_RADIUS",
    "DT",
    "NEAR_MISS_CLEARANCE_M",
    "DroneLayout",
    "PairResult",
    "PairStep",
    "Pilot",
    "TwoDroneSpec",
    "TwoDroneWorld",
    "build_layouts",
    "corridor_range",
    "fixed_latency_pilot",
    "ray_ring",
    "run_pair",
]
