"""Render a recorded flight of one of the Group A simulations to frames and video.

One renderer, six task families: what differs is the scene content (target, zones, sites,
gates, walls, intruder), the checklist and the optional gauge, chosen from the scenario spec.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.domain.actions import Action
from app.render.hud import (
    BLUE,
    GREEN,
    ORANGE,
    RED,
    YELLOW,
    Hud,
    HudState,
    Toast,
    sky_gradient,
)
from app.render.mission_video import (
    DISPLAY_NAMES,
    END_HOLD_S,
    FPS,
    HEIGHT,
    WIDTH,
    MissionRun,
    load_run,
)
from app.render.scene import obstacle_track
from app.render.sim_scene import (
    GATE_ACTIVE,
    GATE_DONE,
    GATE_MISSED,
    GATE_TODO,
    SimScene,
    box_extent,
)
from app.scenarios.spec import ScenarioSpec
from app.sim.tasks import FollowTracker
from app.sim.world import DT

FOLLOW_GAUGE_MAX_M = 8.0
BANNER_LEAD_S = 0.3
GHOST_RGBA = (170, 200, 255, 230)
ZONE_RGBA = (220, 60, 60, 150)
WALL_RGBA = (140, 146, 158, 255)
SITE_OK = (60, 190, 90, 255)
SITE_BAD = (220, 60, 60, 255)
ARM_NAMES = {
    "follow_pilot": "Follow pilot (rule-based)",
    "zone_pilot": "Zone-aware pilot",
    "mission_fsm": "Zone-blind pilot",
    "gate_pilot": "Gate pilot (rule-based)",
    "oracle": "Oracle pilot",
    "fsm": "Reflex FSM pilot",
    "reflex_ttc": "Reflex TTC pilot",
    "site_best": "Site-choosing pilot",
    "site_nearest": "Nearest-site pilot",
    "fsm_machine": "Reflex FSM pilot",
    "fsm_human": "Human-like pilot (FSM rule)",
    "predict_fsm_400": "Predictive FSM pilot",
    "orb_pilot": "Dodge pilot (rule-based)",
    "orb_nav": "Nav pilot + reflex dodge",
}
Overlay = Callable[["SimRenderer", int, float], None]


@dataclass(slots=True)
class SimRun:
    run: MissionRun
    task: dict[str, float]
    latency_label: str
    extras: dict[str, str] = field(default_factory=dict)


def load_sim_run(
    conn: sqlite3.Connection, flight_id: int, root: Path, latency_label: str = ""
) -> SimRun:
    run = load_run(conn, flight_id, root, require_mission=False)
    row = conn.execute(
        "SELECT task_json, latency_ms FROM flights WHERE id = ?", (flight_id,)
    ).fetchone()
    task = json.loads(row["task_json"] or "{}")
    if not latency_label:
        ms = row["latency_ms"]
        latency_label = (
            f"injected latency {ms:.0f} ms"
            if ms is not None and ms >= 0
            else f"mean decision time {run.mean_latency_ms:.0f} ms"
        )
    return SimRun(run, task, latency_label)


ORB_DRONES = {
    "pilot": ("orb_pilot", "orb_pilot"),
    "navpilot": ("orb_nav", "orb_nav"),
    "navreflex": ("orb_nav", "orb_nav"),
    "jev": ("jev", "typesafe/jev-1.13"),
    "mercury": ("mercury", "inception/mercury-decide:free"),
}
ORB_BRAINS = {
    "pn": "homing orbs",
    "pursuit": "pursuit orbs",
    "llm": "gpt-4o-mini orbs",
    "llm-gemini": "Gemini-Lite orbs",
}


def load_orb_run(name: str, root: Path, label: str = "") -> SimRun:
    """A flight saved by ``scripts/run_orb_game.py`` (``data/orb_game/<name>.npz/.json``)."""

    from app.scenarios.sims import draw_orbs, draw_orbs_random, draw_swarm

    base = root / "data" / "orb_game" / name
    meta = json.loads(base.with_suffix(".json").read_text())
    with np.load(base.with_suffix(".npz")) as z:
        traj = {k: z[k] for k in z.files}
    arm_id, model = ORB_DRONES[str(meta["drone"])]
    draw = (
        draw_orbs_random
        if meta.get("scenario") == "random"
        else draw_swarm
        if meta.get("swarm")
        else draw_orbs
    )
    spec = draw(int(meta["seed"]), ext=meta["orb"] != "pn")
    outcome = str(meta["outcome"])
    run = MissionRun(
        flight_id=0,
        arm_id=arm_id,
        model=model,
        shield=bool(meta.get("shield", False)),
        outcome=outcome,
        success=bool(meta["success"]),
        spec=spec,
        traj=traj,
        decisions=list(meta["decisions"]),
        events=[],
        mean_latency_ms=float(meta.get("drone_latency_ms") or 0.0),
        stations_scanned=0,
        stations_total=0,
        t_end=float(meta["t_end"]),
    )
    if not label:
        d_ms = meta.get("drone_ms")
        drone = (
            f"{d_ms:.0f} ms" if d_ms is not None else f"{run.mean_latency_ms:.0f} ms"
        )
        o_ms = meta.get("orb_ms")
        brain = ORB_BRAINS[str(meta["orb"])]
        if meta["orb"] == "llm" and meta.get("orb_latency_ms"):
            brain += f" ({meta['orb_latency_ms']:.0f} ms)"
        elif o_ms:
            brain += f" ({o_ms:.0f} ms)"
        label = f"drone {drone} | {brain}"
    return SimRun(run, {}, label)


def family_of(spec: ScenarioSpec) -> str:
    if spec.event_type == "orbs":
        return "orbs"
    if spec.follow is not None:
        return "follow"
    if spec.sites:
        return "landing"
    if spec.mission is not None and spec.mission.gate_mode:
        return "gates"
    if spec.mission is not None and spec.mission.zones:
        return "delivery"
    return "canyon" if spec.event_type == "canyon" else "pursuit"


TITLES = {
    "follow": "FOLLOW-ME",
    "delivery": "DELIVERY, NO-FLY ZONES",
    "pursuit": "STEERING INTRUDER",
    "landing": "ENGINE-OUT LANDING",
    "gates": "GATE SLALOM",
    "canyon": "URBAN CANYON",
    "orbs": "ORB DODGE",
}


class SimRenderer:
    """Turns a ``SimRun`` into composed RGB frames."""

    def __init__(
        self,
        sim: SimRun,
        width: int = WIDTH,
        height: int = HEIGHT,
        title: str | None = None,
        overlays: list[Overlay] | None = None,
        skin: str = "orb",
    ) -> None:
        self.sim, self.run = sim, sim.run
        self.spec = sim.run.spec
        self.family = family_of(self.spec)
        self.scene = SimScene(self.spec, width, height, skin)
        t = self.run.traj
        self.xyz = np.stack([t["x"], t["y"], t["z"]], axis=1)
        self.vel = np.stack([t["vx"], t["vy"], t["vz"]], axis=1)
        self.n_ticks = len(self.xyz)
        x_hi = max(self.spec.goal_x, float(self.xyz[:, 0].max())) + 1.5
        half = max(3.4, float(np.abs(self.xyz[:, 1]).max()) + 1.0)
        self.hud = Hud(width, height, (-1.0, x_hi), (-half, half))
        self.sky = sky_gradient(width, height)
        self.title = title or TITLES[self.family]
        self.overlays = overlays or []
        self.tracker = FollowTracker(self.spec.follow) if self.spec.follow else None
        self.toasts = self._toasts()
        self._first = True
        self.hud_extra: Callable[[HudState, int, float], None] | None = None
        self.ghost_ms: float | None = (
            None  # draw where things will be when an answer lands
        )
        self.tracks = (
            [obstacle_track(self.spec.obstacle, self.xyz, t["x"])]
            if self.spec.obstacle is not None
            else []
        )

    # ----------------------------------------------------------------- script
    def _toasts(self) -> list[Toast]:
        out: list[Toast] = []
        for e in self.run.events:
            n = int(e["station"]) + 1
            kind = str(e["kind"])
            if self.family == "gates":
                text = f"GATE {n} " + ("PASSED" if kind == "scanned" else "MISSED")
                out.append(
                    Toast(float(e["t"]), text, GREEN if kind == "scanned" else ORANGE)
                )
            elif kind == "breach":
                out.append(Toast(float(e["t"]), "NO-FLY ZONE BREACH", RED))
            elif kind == "lost":
                out.append(Toast(float(e["t"]), "TARGET LOST", RED))
            elif kind == "overran":
                out.append(Toast(float(e["t"]), "OVERRAN THE TARGET", RED))
        spec = self.spec
        if self.family == "orbs":
            out += self._orb_toasts()
        if spec.sensors.battery_drop_at_s is not None:
            out.append(Toast(spec.sensors.battery_drop_at_s, "BATTERY CRITICAL", RED))
        if spec.wind.gust_at_s is not None:
            out.append(Toast(spec.wind.gust_at_s, "GUST", ORANGE))
        for j, o in enumerate(spec.obstacle and [spec.obstacle] or []):
            k = obstacle_track(o, self.xyz, self.run.traj["x"])[0]
            if k is not None:
                label = "INTRUDER" if o.kind == "pursue" else "OBSTACLE AHEAD"
                out.append(Toast((k + 1) * DT, label, RED))
        return sorted(out, key=lambda x: x.t)

    def _orb_at(self, k: int, i: int) -> np.ndarray | None:
        tr = self.run.traj
        key = f"orb{k}x"
        if key not in tr or not np.isfinite(tr[key][i]):
            return None
        return np.array([tr[key][i], tr[f"orb{k}y"][i], tr[f"orb{k}z"][i]])

    def _orb_passed(self, k: int, i: int) -> bool:
        q = self._orb_at(k, i)
        return q is not None and bool(q[0] < self.xyz[i][0] - 0.6)

    def _orb_toasts(self) -> list[Toast]:
        out: list[Toast] = []
        for k in range(self.scene.n_orbs):
            seen = [j for j in range(self.n_ticks) if self._orb_at(k, j) is not None]
            if not seen:
                continue
            out.append(Toast(seen[0] * DT, f"ORB {k + 1} INCOMING", RED))
            passed = next((j for j in seen if self._orb_passed(k, j)), None)
            if passed is not None and self.run.outcome != "collision":
                out.append(Toast(passed * DT, "DODGED!", GREEN))
        return out

    def _index(self, t: float) -> int:
        return int(np.clip(round(t / DT) - 1, 0, self.n_ticks - 1))

    def _event_status(self, t: float, n: int) -> list[str]:
        status = ["todo"] * n
        handled = 0
        for e in self.run.events:
            if float(e["t"]) <= t and e["kind"] in ("scanned", "missed"):
                status[int(e["station"])] = (
                    "done" if e["kind"] == "scanned" else "missed"
                )
                handled += 1
        if handled < n:
            status[handled] = "active"
        return status

    def _decision_state(self, i: int) -> tuple[float | None, float | None]:
        for d in self.run.decisions:
            o, a = int(d["obs_tick"] or 0), int(d["apply_tick"] or 0)
            if o <= i < a and a > o:
                return (i - o) / (a - o), float(d["latency_ms"] or 0.0)
        return None, None

    # ------------------------------------------------------------------ frames
    def frame_at(self, t: float) -> Image.Image:
        i = self._index(t)
        pos, vel = self.xyz[i], self.vel[i]
        tt = float(self.run.traj["t"][i])
        self.scene.place_drone(pos, 0.0, (float(-0.18 * vel[1]), float(0.18 * vel[0])))
        self.scene.follow(pos, vel, first=self._first)
        self._first = False
        tgt = None
        if self.tracker is not None:
            tgt = np.array(self.tracker.target_at(tt))
        self.scene.place_sphere("target", tgt)
        obstacle_xy: list[tuple[float, float]] = []
        ox = self.run.traj["ox"][i]
        has_obstacle = bool(np.isfinite(ox))
        opos = np.array([ox, self.run.traj["oy"][i], self.run.traj["oz"][i]])
        self.scene.place_sphere("obstacle", opos if has_obstacle else None)
        if has_obstacle:
            obstacle_xy.append((float(opos[0]), float(opos[1])))
        if self.family == "orbs":
            self.scene.place_sphere("obstacle", None)
            obstacle_xy = []
            for k in range(self.scene.n_orbs):
                q = self._orb_at(k, i)
                self.scene.place_sphere(f"orb{k}", q)
                if q is not None:
                    obstacle_xy.append((float(q[0]), float(q[1])))
        gates = self.family == "gates"
        if gates and self.spec.mission is not None:
            status = self._event_status(t, len(self.spec.mission.stations))
            cols = {"done": GATE_DONE, "missed": GATE_MISSED, "active": GATE_ACTIVE}
            self.scene.set_gate_colors([cols.get(s, GATE_TODO) for s in status])
        for ov in self.overlays:
            ov(self, i, t)
        rgb, depth = self.scene.render()
        img = Image.fromarray(np.where((depth >= 0.9999)[..., None], self.sky, rgb))
        lookback = max(0, i - int(5.0 / DT))
        img = self.hud.trail_overlay(
            img, [self.scene.project(self.xyz[q]) for q in range(lookback, i + 1, 4)]
        )
        if tgt is not None:
            img = self._tether(img, pos, tgt)
        if self.ghost_ms is not None:
            img = self._ghosts(img, i, pos, vel, opos if has_obstacle else None)
        progress, lat = self._decision_state(i)
        st = HudState(
            t=t,
            model=ARM_NAMES.get(self.run.arm_id)
            or DISPLAY_NAMES.get(self.run.model, self.run.model),
            subtitle=self.sim.latency_label,
            action=Action(int(self.run.traj["action"][i])),
            shield=self.run.shield,
            deciding_progress=progress,
            deciding_ms=lat,
            station_status=[],
            pad_status="todo",
            toasts=self.toasts,
            trail=[
                (float(self.xyz[q][0]), float(self.xyz[q][1]))
                for q in range(lookback, i + 1, 6)
            ],
            drone_xy=(float(pos[0]), float(pos[1])),
            obstacles_xy=obstacle_xy,
            stations_xy=[],
            pad_xy=self._pad_xy(),
            outcome=self._outcome_key()
            if t >= self.run.t_end - BANNER_LEAD_S
            else None,
            title=self.title,
            show_pad=self.family != "follow" and self.family != "landing",
        )
        self._fill_family(st, i, t, pos, tgt)
        if self.hud_extra is not None:
            self.hud_extra(st, i, t)
        return self.hud.compose(img, st, self._labels())

    def _ghosts(
        self,
        img: Image.Image,
        i: int,
        pos: np.ndarray,
        vel: np.ndarray,
        opos: np.ndarray | None,
    ) -> Image.Image:
        """Rings at the predicted positions (drone and obstacle) one latency from now."""

        from PIL import ImageDraw

        assert self.ghost_ms is not None
        lead = self.ghost_ms / 1000.0
        pairs = [(pos, pos + vel * lead, f"drone in {self.ghost_ms:.0f} ms")]
        if opos is not None and i > 0 and np.isfinite(self.run.traj["ox"][i - 1]):
            prev = np.array(
                [self.run.traj["ox"][i - 1], self.run.traj["oy"][i - 1], opos[2]]
            )
            ovel = (opos - prev) / DT
            pairs.append(
                (opos, opos + ovel * lead, f"obstacle in {self.ghost_ms:.0f} ms")
            )
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for real, ghost, label in pairs:
            a, b = self.scene.project(real), self.scene.project(ghost)
            if not (a[2] and b[2]):
                continue
            d.line([(a[0], a[1]), (b[0], b[1])], fill=GHOST_RGBA, width=2)
            d.ellipse(
                (b[0] - 15, b[1] - 15, b[0] + 15, b[1] + 15),
                outline=GHOST_RGBA,
                width=3,
            )
            tw = d.textlength(label)
            d.rounded_rectangle(
                (b[0] + 18, b[1] - 12, b[0] + 30 + tw, b[1] + 6),
                5,
                fill=(14, 18, 26, 200),
            )
            d.text((b[0] + 24, b[1] - 9), label, fill=GHOST_RGBA)
        return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    def _pad_xy(self) -> tuple[float, float]:
        m = self.spec.mission
        return (m.pad_x, m.pad_y) if m is not None else (self.spec.goal_x, 0.0)

    def _tether(
        self, img: Image.Image, pos: np.ndarray, tgt: np.ndarray
    ) -> Image.Image:
        from PIL import ImageDraw

        a = self.scene.project(pos)
        b = self.scene.project(tgt)
        if not (a[2] and b[2]):
            return img
        gap = float(tgt[0] - pos[0])
        band = self.spec.follow
        assert band is not None
        ok = band.band_min <= gap <= band.band_max
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).line(
            [(a[0], a[1]), (b[0], b[1])],
            fill=(80, 220, 120, 220) if ok else (255, 90, 90, 230),
            width=4,
        )
        return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    def _fill_family(
        self,
        st: HudState,
        i: int,
        t: float,
        pos: np.ndarray,
        tgt: np.ndarray | None,
    ) -> None:
        fam = self.family
        ended = t >= self.run.t_end - 0.05
        failed = ended and not self.run.success
        if fam == "follow" and tgt is not None and self.spec.follow is not None:
            f = self.spec.follow
            gap = float(tgt[0] - pos[0])
            st.checklist_title = "TASK"
            frac = self._inband_so_far(i)
            row = (
                "done"
                if ended and self.run.success
                else "missed"
                if failed
                else "active"
            )
            st.checklist = [
                ("Keep target in range", row),
                (f"In band {frac:3.0%}", "todo"),
            ]
            st.gauge = (
                "TARGET DISTANCE AHEAD (green = allowed range)",
                gap / FOLLOW_GAUGE_MAX_M,
                f"{gap:4.1f} m   (allowed {f.band_min:.1f} to {f.band_max:.1f} m)",
                (f.band_min / FOLLOW_GAUGE_MAX_M, f.band_max / FOLLOW_GAUGE_MAX_M),
            )
            st.map_dots = [(float(tgt[0]), float(tgt[1]), (60, 200, 240, 255))]
        elif fam == "delivery":
            breached = any(
                e["kind"] == "breach" and float(e["t"]) <= t for e in self.run.events
            )
            land = "done" if ended and self.run.outcome == "landed_on_pad" else "active"
            st.checklist_title = "DELIVERY"
            st.checklist = [
                ("Stay out of red zones", "missed" if breached else "active"),
                ("Land on pad", land if not failed else "missed"),
            ]
            st.map_rects = [
                (z.x0, z.x1, z.y0, z.y1, ZONE_RGBA)
                for z in (self.spec.mission.zones if self.spec.mission else [])
            ]
        elif fam == "gates" and self.spec.mission is not None:
            n = len(self.spec.mission.stations)
            status = self._event_status(t, n)
            st.checklist_title = "GATES"
            st.checklist = [(f"Gate {k + 1}", s) for k, s in enumerate(status)]
            st.map_dots = [
                (s.x, s.y, (80, 170, 255, 255)) for s in self.spec.mission.stations
            ]
        elif fam == "landing":
            batt = float(self.run.traj["battery"][i])
            st.checklist_title = "ENGINE OUT"
            st.checklist = [
                ("Pick a landing site", "done" if self._landed(i) else "active"),
                (
                    "Land safely",
                    "done"
                    if ended and self.run.success
                    else "missed"
                    if failed
                    else "todo",
                ),
            ]
            st.gauge = (
                "BATTERY",
                batt / 100.0,
                f"{batt:3.0f}%",
                (0.15, 1.0),
            )
            st.map_dots = [
                (s.x, s.y, SITE_BAD if s.hazard else SITE_OK) for s in self.spec.sites
            ]
        elif fam == "orbs":
            collided = ended and self.run.outcome == "collision"
            rows: list[tuple[str, str]] = []
            near = (999.0, -1)
            for k in range(self.scene.n_orbs):
                q = self._orb_at(k, i)
                if q is None:
                    rows.append((f"Dodge orb {k + 1}", "todo"))
                    continue
                d = float(np.hypot(q[0] - pos[0], q[1] - pos[1]))
                near = min(near, (d, k))
                state = "done" if self._orb_passed(k, i) else "active"
                rows.append((f"Dodge orb {k + 1}", state))
            if collided and near[1] >= 0:
                rows[near[1]] = (rows[near[1]][0], "missed")
            rows.append(
                (
                    "Land on the pad",
                    "done"
                    if ended and self.run.success
                    else "missed"
                    if failed
                    else "todo",
                )
            )
            st.checklist_title = "SURVIVE"
            st.checklist = rows
            if near[1] >= 0:
                st.gauge = (
                    "NEAREST ORB",
                    min(near[0], 6.0) / 6.0,
                    f"{near[0]:.1f} m   (safe beyond 1.5 m)",
                    (1.5 / 6.0, 1.0),
                )
        else:  # pursuit, canyon
            steering = (
                self.spec.obstacle is not None and self.spec.obstacle.kind == "pursue"
            )
            what = (
                "Cross the canyon"
                if fam == "canyon"
                else "Avoid the intruder"
                if steering
                else "Avoid the obstacle"
            )
            collided = ended and self.run.outcome in ("collision", "crash")
            st.checklist_title = "TASK"
            st.checklist = [
                (what, "missed" if collided else "active"),
                (
                    "Reach the goal",
                    "done"
                    if ended and self.run.success
                    else "missed"
                    if failed
                    else "todo",
                ),
            ]
            st.map_rects = [(*box_extent(b), WALL_RGBA) for b in self.spec.boxes]

    def _landed(self, i: int) -> bool:
        return bool(self.xyz[i][2] < 0.6 and i > self.n_ticks * 0.4)

    def _inband_so_far(self, i: int) -> float:
        f = self.spec.follow
        assert f is not None and self.tracker is not None
        ts = self.run.traj["t"][: i + 1]
        tx = np.array([self.tracker.target_at(float(q))[0] for q in ts])
        ty = np.array([self.tracker.target_at(float(q))[1] for q in ts])
        gap = tx - self.xyz[: i + 1, 0]
        ok = (
            (gap >= f.band_min)
            & (gap <= f.band_max)
            & (np.abs(ty - self.xyz[: i + 1, 1]) <= f.band_half_width)
        )
        return float(ok.mean())

    def _outcome_key(self) -> str:
        r = self.run
        if r.success:
            return (
                "gates_done"
                if self.family == "gates"
                else (
                    "completed"
                    if self.family in ("follow", "pursuit", "canyon", "delivery")
                    else r.outcome
                )
            )
        if self.family == "gates":
            return "gates_missed"
        if self.family == "follow" and r.outcome == "completed":
            return "low_band"
        if self.family == "orbs":
            if r.outcome == "crash":
                last = self.xyz[-1]
                if abs(last[1]) > 5.5 or last[0] > self.spec.goal_x + 5.5:
                    return "left_arena"
            return "hit" if r.outcome == "collision" else r.outcome
        if self.family == "delivery" and r.outcome == "landed_on_pad":
            return "breach"
        return r.outcome

    def _labels(self) -> list[tuple[float, float, str]]:
        out: list[tuple[float, float, str]] = []
        if self.family == "landing":
            for s in self.spec.sites:
                x, y, ok = self.scene.project(np.array([s.x, s.y, 0.7]))
                if ok:
                    out.append((x, y, "HAZARD" if s.hazard else "SAFE"))
        return out

    def frames(self) -> Iterator[tuple[float, Image.Image]]:
        total = self.run.t_end + END_HOLD_S
        for f in range(int(total * FPS)):
            t = min(f / FPS, self.run.t_end)
            yield f / FPS, self.frame_at(t)

    def close(self) -> None:
        self.scene.close()


__all__ = ["BLUE", "YELLOW", "SimRenderer", "SimRun", "family_of", "load_sim_run"]
