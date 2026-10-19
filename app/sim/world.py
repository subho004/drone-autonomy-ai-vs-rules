"""PyBullet world for one scenario at a time (headless, DIRECT mode).

Wraps a ``CtrlAviary`` subclass and adds: seeded wind, battery derating,
a parked-then-teleported spherical obstacle, a 16-ray range ring, and
contact-based collision detection. One ``DroneWorld`` is reused across
flights in a worker (``reset`` is ~10 ms; constructing a client is slower).

Frames: world axes, yaw held at zero, x forward toward the goal, y left, z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from app.core.logging import get_logger
from app.domain.observation import (
    MAX_RANGE_M,
    N_RAYS,
    N_SECTORS,
    Observation,
    TruthState,
)
from app.scenarios.spec import BoxSpec, ObstacleSpec, ScenarioSpec
from app.sim.rng import channel_generator, ou_process

logger = get_logger(__name__)

PYB_FREQ = 240
CTRL_FREQ = 48
DT = 1.0 / CTRL_FREQ
SETTLE_TICKS = 24  # half a second of hover before the flight clock starts
DRONE_RADIUS = 0.06
RAY_ORIGIN_OFFSET = 0.10
PARK_POS = (0.0, 0.0, -50.0)
BATTERY_CRITICAL_PCT = 8.0
BATTERY_LOW_PCT = 25.0
HOME = (0.0, 0.0, 1.0)
CORRIDOR_HALF_WIDTH = 0.6  # drone radius + obstacle radius + noise slack
_FORWARD_RAYS = (0, 1, 2, N_RAYS - 1, N_RAYS - 2)


class _Aviary(CtrlAviary):
    """CtrlAviary with an external wind acceleration and an RPM derating."""

    wind_accel: np.ndarray
    rpm_scale: float

    def __init__(self, **kwargs: object) -> None:
        self.wind_accel = np.zeros(3)
        self.rpm_scale = 1.0
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(action) * self.rpm_scale, 0.0, self.MAX_RPM)

    def _physics(self, rpm: np.ndarray, nth_drone: int) -> None:
        super()._physics(rpm, nth_drone)
        if self.wind_accel.any():
            force = (self.wind_accel * self.M).tolist()
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                -1,
                forceObj=force,
                posObj=self.pos[nth_drone].tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )


@dataclass(slots=True)
class _Obstacle:
    """One spherical obstacle: parked under the floor until its trigger fires."""

    spec: ObstacleSpec
    body_id: int
    active: bool = False
    pos: np.ndarray = field(default_factory=lambda: np.array(PARK_POS, dtype=float))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cmd: int = 0  # external steering: -1 right, 0 straight, +1 left

    @property
    def radius(self) -> float:
        return self.spec.radius


@dataclass(slots=True)
class StepInfo:
    """What happened in one control tick."""

    collision: bool
    ground_contact: bool
    out_of_bounds: bool
    clearance: float
    obstacle_active: bool


class DroneWorld:
    """One headless PyBullet client flying scenarios one after another."""

    def __init__(self) -> None:
        self._env = _Aviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=PYB_FREQ,
            ctrl_freq=CTRL_FREQ,
            gui=False,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        self._ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self._client = self._env.getPyBulletClient()
        self._drone_id = int(self._env.getDroneIds()[0])
        self._obstacles: list[_Obstacle] = []
        self._boxes: list[int] = []
        self._spec: ScenarioSpec | None = None
        self.tick = 0
        self.hover_rpm = float(self._env.HOVER_RPM)

    # ------------------------------------------------------------------ setup
    def reset(self, spec: ScenarioSpec) -> None:
        """Load ``spec``: reset physics, settle at hover, arm the event."""

        self._spec = spec
        self._env.INIT_XYZS = np.array([[0.0, 0.0, spec.start_z]])
        self._env.reset(seed=spec.seed)
        self._ctrl.reset()
        self._env.wind_accel = np.zeros(3)
        self._env.rpm_scale = 1.0
        self._build_obstacles(spec)
        self.tick = 0
        self._n_max = int(spec.max_time_s * CTRL_FREQ) + 2
        self._gen_noise(spec)
        self.obstacle_active = False
        self.obstacle_pos = np.array(PARK_POS, dtype=float)
        self.obstacle_vel = np.zeros(3)
        self.obstacle_radius = self._obstacles[0].radius if self._obstacles else 0.0
        self.spawn_tick: int | None = None
        self._fwd_prev = MAX_RANGE_M
        self._closing = 0.0
        self.landed = False
        self.battery_pct = spec.sensors.battery_start_pct
        self._critical_since: float | None = None
        self._frozen_pos: np.ndarray | None = None
        self._wind_now = np.zeros(2)
        self._last_rays = np.full(N_RAYS, MAX_RANGE_M)
        self._sense_cache: tuple[int, np.ndarray, float, float] | None = None
        self._hold_pos = np.array([0.0, 0.0, spec.start_z])
        # Settle: hover at the start point so the flight clock starts from rest.
        for _ in range(SETTLE_TICKS):
            self._advance(self._hold_pos, np.zeros(3), count=False)

    def _build_obstacles(self, spec: ScenarioSpec) -> None:
        """Create one parked sphere per obstacle (single-event and mission specs)."""

        self._obstacles = []
        self._boxes = [self._make_box(b) for b in spec.boxes]
        specs = ([spec.obstacle] if spec.obstacle else []) + (
            list(spec.mission.obstacles) if spec.mission else []
        )
        for ob in specs:
            shape = p.createCollisionShape(
                p.GEOM_SPHERE, radius=ob.radius, physicsClientId=self._client
            )
            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=ob.radius,
                rgbaColor=[0.9, 0.2, 0.2, 1],
                physicsClientId=self._client,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual,
                basePosition=PARK_POS,
                physicsClientId=self._client,
            )
            self._obstacles.append(_Obstacle(ob, int(body)))

    def _make_box(self, b: BoxSpec) -> int:
        half = [b.hx, b.hy, b.hz]
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half, physicsClientId=self._client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=[0.62, 0.64, 0.7, 1.0],
            physicsClientId=self._client,
        )
        return int(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[b.x, b.y, b.z],
                physicsClientId=self._client,
            )
        )

    def _gen_noise(self, spec: ScenarioSpec) -> None:
        n = self._n_max
        s = spec.seed
        self._ray_noise = channel_generator(s, "ray_noise").standard_normal((n, N_RAYS))
        self._ray_drop = channel_generator(s, "ray_dropout").random((n, N_RAYS))
        w = spec.wind
        gen = channel_generator(s, "wind")
        self._turb = np.stack(
            [ou_process(gen, n, DT, w.turb_sigma, w.turb_tau_s) for _ in range(2)],
            axis=1,
        )

    # ------------------------------------------------------------------ state
    @property
    def spec(self) -> ScenarioSpec:
        if self._spec is None:
            raise RuntimeError("world not reset")
        return self._spec

    @property
    def t_sim(self) -> float:
        return self.tick * DT

    def true_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        env = self._env
        return (
            env.pos[0].copy(),
            env.vel[0].copy(),
            env.quat[0].copy(),
            env.ang_v[0].copy(),
        )

    def clearance(self) -> float:
        """Surface-to-surface distance to the nearest active obstacle (inf if none)."""

        pos = self._env.pos[0]
        best = math.inf
        for ob in self._obstacles:
            if ob.active:
                d = float(np.linalg.norm(pos - ob.pos)) - ob.radius - DRONE_RADIUS
                best = min(best, d)
        for box in self._boxes:
            pts = p.getClosestPoints(
                self._drone_id, box, 3.0, physicsClientId=self._client
            )
            if pts:
                best = min(best, float(min(q[8] for q in pts)))
        return best

    def _refresh_primary(self) -> None:
        """Expose the nearest active obstacle through the legacy single-obstacle fields."""

        pos = self._env.pos[0]
        active = [o for o in self._obstacles if o.active]
        self.obstacle_active = bool(active) or bool(self._boxes)
        if not active:
            return
        near = min(active, key=lambda o: float(np.linalg.norm(pos - o.pos)))
        self.obstacle_pos, self.obstacle_vel = near.pos, near.vel
        self.obstacle_radius = near.radius

    # ----------------------------------------------------------------- events
    def _maybe_spawn(self, pos: np.ndarray) -> None:
        for o in self._obstacles:
            ob = o.spec
            if o.active or pos[0] < ob.trigger_x:
                continue
            if ob.kind == "cross":
                start = np.array(
                    [pos[0] + ob.d_spawn, ob.cross_side * ob.cross_start_y, pos[2]]
                )
                vel = np.array([0.0, -ob.cross_side * ob.speed, 0.0])
            else:
                start = np.array([pos[0] + ob.d_spawn, ob.lateral_offset, pos[2]])
                head = ob.kind in ("approach", "pursue")
                vel = np.array([-ob.speed if head else 0.0, 0.0, 0.0])
            o.active, o.pos, o.vel = True, start, vel
            if self.spawn_tick is None:
                self.spawn_tick = self.tick
            self._move(o)

    def _guide(
        self, o: _Obstacle, drone_pos: np.ndarray, drone_vel: np.ndarray
    ) -> None:
        """Proportional navigation: steer by the line-of-sight rate, constant speed."""

        r = (drone_pos - o.pos)[:2]
        dist = float(np.linalg.norm(r))
        if dist < 1e-3:
            return
        v_rel = (drone_vel - o.vel)[:2]
        lam_dot = float(r[0] * v_rel[1] - r[1] * v_rel[0]) / dist**2
        closing = max(-float(np.dot(r, v_rel)) / dist, 0.1)
        los = r / dist
        normal = np.array([-los[1], los[0]])
        accel = float(
            np.clip(
                o.spec.nav_gain * closing * lam_dot, -o.spec.max_accel, o.spec.max_accel
            )
        )
        v2 = o.vel[:2] + accel * normal * DT
        norm = float(np.linalg.norm(v2))
        if norm > 1e-6:
            v2 = v2 / norm * o.spec.speed
        o.vel = np.array([v2[0], v2[1], 0.0])

    def _turn(self, o: _Obstacle) -> None:
        """External steering: turn left/right at the obstacle's acceleration limit."""

        if o.cmd == 0:
            return
        v = o.vel[:2]
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return
        normal = np.array([-v[1], v[0]]) / speed  # left of the heading
        v2 = v + o.cmd * o.spec.max_accel * normal * DT
        v2 = v2 / float(np.linalg.norm(v2)) * o.spec.speed
        o.vel = np.array([v2[0], v2[1], 0.0])

    def set_orb_command(self, index: int, cmd: int) -> None:
        self._obstacles[index].cmd = max(-1, min(1, cmd))

    def orb_positions(self) -> list[np.ndarray | None]:
        return [o.pos.copy() if o.active else None for o in self._obstacles]

    def orb_views(self) -> list[tuple[int, float, float, float, float]]:
        """For each active externally steered orb: (index, forward, left, distance, closing)
        of the drone in the orb's own heading frame (closing > 0 when approaching)."""

        pos = self._env.pos[0]
        dvel = self._env.vel[0]
        out: list[tuple[int, float, float, float, float]] = []
        for i, o in enumerate(self._obstacles):
            if not (o.active and o.spec.ext_control):
                continue
            rel = (pos - o.pos)[:2]
            speed = float(np.linalg.norm(o.vel[:2]))
            head = o.vel[:2] / speed if speed > 1e-6 else np.array([1.0, 0.0])
            left = np.array([-head[1], head[0]])
            dist = float(np.linalg.norm(rel))
            closing = -float(np.dot(rel, (dvel - o.vel)[:2])) / max(dist, 1e-6)
            out.append((i, float(rel @ head), float(rel @ left), dist, closing))
        return out

    def _move(self, o: _Obstacle) -> None:
        p.resetBasePositionAndOrientation(
            o.body_id, o.pos.tolist(), [0, 0, 0, 1], physicsClientId=self._client
        )

    def _update_events(self, pos: np.ndarray) -> None:
        spec = self.spec
        t = self.t_sim
        self._maybe_spawn(pos)
        for o in self._obstacles:
            if o.active:
                if o.spec.kind == "pursue" and o.spec.ext_control:
                    self._turn(o)
                elif o.spec.kind == "pursue":
                    self._guide(o, pos, self._env.vel[0])
                o.pos = o.pos + o.vel * DT
                self._move(o)
        self._refresh_primary()
        w = spec.wind
        mean = np.array([w.mean_x, w.mean_y]) if t >= w.step_at_s else np.zeros(2)
        if w.gust_at_s is not None and w.gust_at_s <= t < w.gust_at_s + w.gust_dur_s:
            mean = mean + np.array([w.gust_x, w.gust_y])
        i = min(self.tick, self._n_max - 1)
        self._wind_now = mean + self._turb[i]
        self._env.wind_accel = np.array([self._wind_now[0], self._wind_now[1], 0.0])
        self._update_battery(t)
        gps_t = spec.sensors.gps_loss_at_s
        if gps_t is not None and t >= gps_t and self._frozen_pos is None:
            self._frozen_pos = pos.copy()  # dead position: frozen at the loss instant

    def _update_battery(self, t: float) -> None:
        s = self.spec.sensors
        self.battery_pct = max(0.0, self.battery_pct - s.battery_drain_pct_per_s * DT)
        if s.battery_drop_at_s is not None and abs(t - s.battery_drop_at_s) < DT / 2:
            self.battery_pct = min(self.battery_pct, s.battery_drop_to_pct)
        if self.battery_pct < BATTERY_CRITICAL_PCT and self._critical_since is None:
            self._critical_since = t
        if self._critical_since is not None and not self.landed:
            over = t - self._critical_since - s.battery_death_after_s
            self._env.rpm_scale = float(np.clip(1.0 - 0.25 * max(over, 0.0), 0.2, 1.0))

    # ---------------------------------------------------------------- sensing
    def sense(self) -> tuple[np.ndarray, float, float]:
        """Cast the ray ring; returns (noisy ranges, forward-arc min, closing speed).

        Cached per tick so the closing-speed filter advances exactly once per tick.
        """

        if self._sense_cache is not None and self._sense_cache[0] == self.tick:
            return self._sense_cache[1], self._sense_cache[2], self._sense_cache[3]
        pos = self._env.pos[0]
        angles = np.arange(N_RAYS) * (2.0 * math.pi / N_RAYS)
        dirs = np.stack([np.cos(angles), np.sin(angles), np.zeros(N_RAYS)], axis=1)
        starts = pos + dirs * RAY_ORIGIN_OFFSET
        ends = pos + dirs * (RAY_ORIGIN_OFFSET + MAX_RANGE_M)
        res = p.rayTestBatch(
            starts.tolist(), ends.tolist(), physicsClientId=self._client
        )
        rng = np.full(N_RAYS, MAX_RANGE_M)
        ids = {o.body_id for o in self._obstacles} | set(self._boxes)
        for k, r in enumerate(res):
            if r[0] in ids:
                rng[k] = RAY_ORIGIN_OFFSET + r[2] * MAX_RANGE_M
        i = min(self.tick, self._n_max - 1)
        sen = self.spec.sensors
        noisy = rng + self._ray_noise[i] * sen.ray_noise_sigma
        if sen.ray_dropout_p > 0:
            noisy = np.where(self._ray_drop[i] < sen.ray_dropout_p, MAX_RANGE_M, noisy)
        noisy = np.clip(noisy, 0.0, MAX_RANGE_M)
        radar = self._radar(pos)
        if radar is not None:  # small fast objects slip between rays: add a disc return
            k = round(radar["bearing_rad"] / (2.0 * math.pi / N_RAYS)) % N_RAYS
            for kk in (k - 1, k, k + 1):
                noisy[kk % N_RAYS] = min(noisy[kk % N_RAYS], radar["range"])
        fwd = self._corridor_range(noisy)
        if fwd < MAX_RANGE_M * 0.98 and self._fwd_prev < MAX_RANGE_M * 0.98:
            inst = (self._fwd_prev - fwd) / DT
            self._closing = 0.5 * self._closing + 0.5 * inst
        else:
            self._closing = 0.5 * self._closing
        self._fwd_prev = fwd
        self._last_rays = noisy
        self._sense_cache = (self.tick, noisy, fwd, self._closing)
        return noisy, fwd, self._closing

    def _radar(self, pos: np.ndarray) -> dict[str, float] | None:
        """Nearest active disc-sensed obstacle (``sense_radius`` > 0): range, bearing, rates."""

        best: dict[str, float] | None = None
        vel = self._env.vel[0]
        for o in self._obstacles:
            if not o.active or o.spec.sense_radius <= 0.0:
                continue
            rel = (o.pos - pos)[:2]
            dist = float(np.linalg.norm(rel))
            rng = dist - o.spec.sense_radius
            if rng > MAX_RANGE_M or (best is not None and rng >= best["range"]):
                continue
            rel_v = (o.vel - vel)[:2]
            best = {
                "range": max(rng, 0.0),
                "bearing_rad": math.atan2(rel[1], rel[0]) % (2.0 * math.pi),
                "closing": -float(np.dot(rel, rel_v)) / max(dist, 1e-6),
                "dx": float(rel[0]),
                "dy": float(rel[1]),
            }
        return best

    def _corridor_range(self, rays: np.ndarray) -> float:
        """Distance to the nearest return inside the forward corridor.

        A ray at angle theta with range r projects to (r cos, r sin); it counts only if
        it lies ahead and within CORRIDOR_HALF_WIDTH of the flight line, so an obstacle
        that is already being dodged to the side no longer looks like a threat.
        """

        best = MAX_RANGE_M
        for k in _FORWARD_RAYS:
            r = float(rays[k])
            if r >= MAX_RANGE_M * 0.98:
                continue
            ang = k * 2.0 * math.pi / N_RAYS
            if abs(r * math.sin(ang)) < CORRIDOR_HALF_WIDTH:
                best = min(best, r * math.cos(ang))
        return best

    def observe(self, last_action: int, decision_age_ms: float = 0.0) -> Observation:
        rays, fwd, closing = self.sense()
        pos, vel, _, _ = self.true_state()
        s = self.spec
        gps_ok = not (
            s.sensors.gps_loss_at_s is not None
            and self.t_sim >= s.sensors.gps_loss_at_s
        )
        seen = (
            self._frozen_pos if (not gps_ok and self._frozen_pos is not None) else pos
        )
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
        k_near = int(np.argmin(rays))
        near = float(rays[k_near])
        bearing = math.degrees(k_near * 2.0 * math.pi / N_RAYS)
        if bearing > 180.0:
            bearing -= 360.0
        ttc = fwd / closing if closing > 0.1 and fwd < MAX_RANGE_M * 0.98 else 99.0
        return Observation(
            tick=self.tick,
            t_sim=self.t_sim,
            pos=(float(seen[0]), float(seen[1]), float(seen[2])),
            vel=(float(vel[0]), float(vel[1]), float(vel[2])),
            goal_dx=float(s.goal_x - seen[0]),
            home_dist=float(math.hypot(seen[0] - HOME[0], seen[1] - HOME[1])),
            rays=tuple(float(r) for r in rays),
            sectors=sectors,
            nearest_range=near,
            nearest_bearing_deg=bearing,
            closing_speed=float(closing),
            ttc=float(min(ttc, 99.0)),
            battery_pct=float(self.battery_pct),
            gps_ok=gps_ok,
            last_action=last_action,
            decision_age_ms=decision_age_ms,
            fwd_range=float(fwd),
            extras=self._radar_extras(),
        )

    def orb_radar(self) -> tuple[float, float, float] | None:
        """(range, bearing in degrees with left positive, closing speed) of the nearest orb."""

        radar = self._radar(self._env.pos[0])
        if radar is None:
            return None
        bearing = math.degrees(radar["bearing_rad"])
        return (
            radar["range"],
            bearing - 360.0 if bearing > 180.0 else bearing,
            radar["closing"],
        )

    def _radar_extras(self) -> dict[str, float]:
        radar = self._radar(self._env.pos[0])
        if radar is None:
            return {}
        bearing = math.degrees(radar["bearing_rad"])
        return {
            "orb_range": radar["range"],
            "orb_bearing_deg": bearing - 360.0 if bearing > 180.0 else bearing,
            "orb_closing": radar["closing"],
            "orb_dx": radar["dx"],
            "orb_dy": radar["dy"],
        }

    def truth(self) -> TruthState:
        pos, vel, _, _ = self.true_state()
        return TruthState(
            tick=self.tick,
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            vel=(float(vel[0]), float(vel[1]), float(vel[2])),
            obstacle_pos=tuple(float(v) for v in self.obstacle_pos)
            if self.obstacle_active
            else None,  # type: ignore[arg-type]
            obstacle_vel=tuple(float(v) for v in self.obstacle_vel)
            if self.obstacle_active
            else None,  # type: ignore[arg-type]
            obstacle_radius=self.obstacle_radius,
            clearance=self.clearance(),
        )

    # ------------------------------------------------------------------- step
    def step(self, target_pos: np.ndarray, target_vel: np.ndarray) -> StepInfo:
        """Advance one control tick tracking (target_pos, target_vel)."""

        return self._advance(target_pos, target_vel, count=True)

    def _advance(
        self, target_pos: np.ndarray, target_vel: np.ndarray, *, count: bool
    ) -> StepInfo:
        env = self._env
        pos, vel, quat, ang_v = self.true_state()
        if count:
            self._update_events(pos)
        rpm, _, _ = self._ctrl.computeControl(
            control_timestep=DT,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_v,
            target_pos=target_pos,
            target_rpy=np.zeros(3),
            target_vel=target_vel,
        )
        env.step(rpm.reshape(1, 4))
        if count:
            self.tick += 1
        return self._step_info()

    def _step_info(self) -> StepInfo:
        pos = self._env.pos[0]
        collision = False
        for o in self._obstacles:
            if not o.active:
                continue
            contacts = p.getContactPoints(
                bodyA=self._drone_id, bodyB=o.body_id, physicsClientId=self._client
            )
            if contacts:
                collision = True
        for box in self._boxes:
            if p.getContactPoints(
                bodyA=self._drone_id, bodyB=box, physicsClientId=self._client
            ):
                collision = True
        collision = collision or self.clearance() <= 0.0
        ground = bool(pos[2] < 0.06)
        oob = bool(
            abs(pos[1]) > 6.0
            or pos[2] > 4.0
            or pos[0] < -3.0
            or pos[0] > self.spec.goal_x + 6.0
        )
        return StepInfo(collision, ground, oob, self.clearance(), self.obstacle_active)

    @property
    def wind_now(self) -> tuple[float, float]:
        return float(self._wind_now[0]), float(self._wind_now[1])

    def close(self) -> None:
        self._env.close()


__all__ = [
    "BATTERY_CRITICAL_PCT",
    "BATTERY_LOW_PCT",
    "CTRL_FREQ",
    "DT",
    "HOME",
    "DroneWorld",
    "StepInfo",
]
