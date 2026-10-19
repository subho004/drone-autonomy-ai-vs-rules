"""3D replay of a two-drone encounter (PyBullet's CPU renderer, headless) with a Pillow HUD.

Nothing here simulates physics: the scene is drawn from a recorded ``PairResult``
trajectory. Drone A is blue and drone B is magenta (body, ground ring and badge); each
model is drawn 4x larger than real so it reads on screen. A smoothed overview camera keeps
both drones in frame, and the video slows to 0.4x while they are within 2.5 m.

The HUD is a small new class (the mission HUD is built around stations): each drone gets a
panel with its current action and an in-flight decision bar, the top centre shows sim time
and the separation in a danger colour, and toasts announce dodges, near misses and contact.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

from app.domain.actions import Action
from app.render.hud import (
    ACTION_COLORS,
    FONT_BOLD,
    FONT_REGULAR,
    GRAY,
    GREEN,
    MUTED,
    ORANGE,
    PANEL,
    RED,
    TEXT,
    YELLOW,
    Toast,
    sky_gradient,
)
from app.render.mission_video import FPS, HEIGHT, WIDTH
from app.sim.two_drone import DRONE_RADIUS, DT, NEAR_MISS_CLEARANCE_M, TwoDroneSpec

DRONE_SCALE = 6.0
BEACON_RADIUS_M = 0.07
BEACON_LIFT_M = 0.16
COLORS: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.10, 0.52, 1.0),
    (0.93, 0.22, 0.72),
)
HUD_COLORS: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] = (
    (60, 150, 255, 255),
    (240, 70, 190, 255),
)
NAMES = ("A", "B")
GROUND_COLOR = (0.66, 0.74, 0.68)
LANE_COLOR = (0.96, 0.96, 0.96)
FOV_DEG = 50.0
CAM_PITCH_DEG = 50.0
CAM_AZIMUTH_DEG = {"head_on": 0.0, "perpendicular": 38.0}
CAM_DIST_MIN, CAM_DIST_MAX = 4.2, 8.6
CAM_SMOOTH = 0.07
LIGHT_DIR = (0.4, -0.5, 1.0)
SLOWMO_SEP_M = 2.5
SLOWMO_SPEED = 0.4
END_HOLD_S = 2.2
TRAIL_S = 4.0
DANGER_FAR_M = 1.5
DANGER_NEAR_M = 0.6
TOAST_S = 2.0
LATERAL = {
    Action.LEFT,
    Action.RIGHT,
    Action.AVOID_LEFT,
    Action.AVOID_RIGHT,
    Action.BRAKE,
}
ARM_LABELS = {"reflex_ttc": "Reflex pilot", "fsm": "FSM pilot"}


def _font(
    candidates: tuple[str, ...], size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


@dataclass(slots=True)
class PairRun:
    """A recorded encounter plus what the HUD needs to caption it."""

    spec: TwoDroneSpec
    traj: dict[str, np.ndarray]
    decisions: list[list[dict[str, Any]]]
    metrics: dict[str, Any]
    latencies_ms: tuple[float, float]
    arm_ids: tuple[str, str] = ("reflex_ttc", "reflex_ttc")
    goals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n_ticks(self) -> int:
        return len(self.traj["t"])

    @property
    def t_end(self) -> float:
        return float(self.traj["t"][-1])

    def tick_at(self, t: float) -> int:
        return int(np.clip(round(t / DT) - 1, 0, self.n_ticks - 1))

    def clearance(self, i: int) -> float:
        return float(self.traj["sep"][i]) - 2.0 * DRONE_RADIUS


@dataclass(slots=True)
class DroneHud:
    """One drone's panel contents for one frame."""

    action: Action
    progress: float | None
    latency_ms: float
    arm_label: str


@dataclass(slots=True)
class PairHudState:
    t: float
    speed: float
    clearance: float
    min_clearance: float
    drones: tuple[DroneHud, DroneHud]
    toasts: list[Toast]
    trails: tuple[list[tuple[float, float]], list[tuple[float, float]]]
    xy: tuple[tuple[float, float], tuple[float, float]]
    goals: list[tuple[float, float]]
    caption: str
    outcome: str | None = None
    outcome_detail: str = ""


def danger_color(clearance: float) -> tuple[int, int, int, int]:
    if clearance >= DANGER_FAR_M:
        return GREEN
    return ORANGE if clearance >= DANGER_NEAR_M else RED


class PairHud:
    """Pillow HUD for the two-drone video."""

    def __init__(
        self,
        width: int,
        height: int,
        world_x: tuple[float, float],
        world_y: tuple[float, float],
    ) -> None:
        self.w, self.h = width, height
        self.world_x, self.world_y = world_x, world_y
        self.f_big = _font(FONT_BOLD, 22)
        self.f_mid = _font(FONT_BOLD, 17)
        self.f_small = _font(FONT_REGULAR, 14)
        self.f_tiny = _font(FONT_REGULAR, 12)
        self.f_toast = _font(FONT_BOLD, 28)

    def compose(
        self,
        frame: Image.Image,
        st: PairHudState,
        badges: list[tuple[float, float, int]],
        link: tuple[tuple[float, float], tuple[float, float]] | None,
        trail3d: tuple[list[tuple[float, float]], list[tuple[float, float]]],
    ) -> Image.Image:
        layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, "RGBA")
        for i, pts in enumerate(trail3d):
            if len(pts) > 1:
                d.line(pts, fill=(*HUD_COLORS[i][:3], 150), width=3)
        if link is not None:
            col = danger_color(st.clearance)
            d.line(link, fill=(*col[:3], 220), width=3)
            mx, my = (link[0][0] + link[1][0]) / 2, (link[0][1] + link[1][1]) / 2
            self._chip(d, mx, my - 18, f"{max(st.clearance, 0.0):.2f} m", col)
        for x, y, i in badges:
            d.ellipse((x - 13, y - 13, x + 13, y + 13), fill=HUD_COLORS[i])
            d.text(
                (x - 6, y - 10), NAMES[i], font=self.f_mid, fill=(255, 255, 255, 255)
            )
        self._drone_panel(d, 0, st)
        self._drone_panel(d, 1, st)
        self._centre_panel(d, st)
        self._minimap(d, st)
        self._caption(d, st)
        self._toasts(d, st)
        return Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")

    # ----------------------------------------------------------------- pieces
    def _chip(
        self,
        d: ImageDraw.ImageDraw,
        x: float,
        y: float,
        text: str,
        color: tuple[int, int, int, int],
    ) -> None:
        tw = d.textlength(text, font=self.f_small)
        d.rounded_rectangle(
            (x - tw / 2 - 8, y - 11, x + tw / 2 + 8, y + 11), 8, fill=PANEL
        )
        d.text((x - tw / 2, y - 8), text, font=self.f_small, fill=color)

    def _drone_panel(self, d: ImageDraw.ImageDraw, i: int, st: PairHudState) -> None:
        dh = st.drones[i]
        left = 14 if i == 0 else self.w - 14 - 300
        d.rounded_rectangle((left, 14, left + 300, 158), 12, fill=PANEL)
        d.rounded_rectangle((left, 14, left + 8, 158), 4, fill=HUD_COLORS[i])
        d.text(
            (left + 22, 22), f"DRONE {NAMES[i]}", font=self.f_big, fill=HUD_COLORS[i]
        )
        d.text(
            (left + 22, 52),
            f"{dh.arm_label}, {dh.latency_ms:.0f} ms decisions",
            font=self.f_small,
            fill=MUTED,
        )
        color = ACTION_COLORS.get(dh.action, GRAY)
        label = dh.action.name.replace("_", " ")
        tw = d.textlength(label, font=self.f_mid)
        d.rounded_rectangle((left + 22, 80, left + 22 + tw + 28, 108), 14, fill=color)
        d.text((left + 36, 83), label, font=self.f_mid, fill=(12, 14, 20, 255))
        bar_w = 262
        d.rounded_rectangle(
            (left + 22, 120, left + 22 + bar_w, 130), 5, fill=(40, 48, 62, 255)
        )
        if dh.progress is not None and dh.latency_ms > 20:
            fill = int(bar_w * min(max(dh.progress, 0.0), 1.0))
            d.rounded_rectangle(
                (left + 22, 120, left + 22 + max(fill, 10), 130), 5, fill=YELLOW
            )
            txt = f"decision in flight {dh.latency_ms * dh.progress:.0f}/{dh.latency_ms:.0f} ms"
            ink = YELLOW
        else:
            txt = "no decision in flight" if dh.latency_ms > 20 else "instant decisions"
            ink = MUTED
        d.text((left + 22, 135), txt, font=self.f_tiny, fill=ink)

    def _centre_panel(self, d: ImageDraw.ImageDraw, st: PairHudState) -> None:
        cx = self.w / 2
        d.rounded_rectangle((cx - 120, 14, cx + 120, 96), 12, fill=PANEL)
        d.text((cx - 106, 20), "SIM TIME", font=self.f_tiny, fill=MUTED)
        d.text((cx - 106, 34), f"{st.t:5.1f} s", font=self.f_mid, fill=TEXT)
        if st.speed < 1.0:
            d.text(
                (cx + 40, 36), f"{st.speed:.1f}x slow", font=self.f_small, fill=YELLOW
            )
        col = danger_color(st.clearance)
        d.text((cx - 106, 60), "SEPARATION", font=self.f_tiny, fill=MUTED)
        d.text(
            (cx - 20, 56), f"{max(st.clearance, 0.0):.2f} m", font=self.f_big, fill=col
        )

    def _minimap(self, d: ImageDraw.ImageDraw, st: PairHudState) -> None:
        mw, mh = 300, 112
        x0, y0 = self.w - mw - 14, self.h - mh - 14
        d.rounded_rectangle((x0 - 8, y0 - 26, x0 + mw + 8, y0 + mh + 8), 12, fill=PANEL)
        d.text((x0, y0 - 20), "TOP VIEW", font=self.f_tiny, fill=MUTED)

        def m(x: float, y: float) -> tuple[float, float]:
            fx = (x - self.world_x[0]) / (self.world_x[1] - self.world_x[0])
            fy = (y - self.world_y[0]) / (self.world_y[1] - self.world_y[0])
            return x0 + fx * mw, y0 + (1 - fy) * mh

        d.rectangle((x0, y0, x0 + mw, y0 + mh), fill=(30, 38, 50, 255))
        for gx, gy in st.goals:
            cx, cy = m(gx, gy)
            d.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=GRAY, width=2)
        for i in range(2):
            if len(st.trails[i]) > 1:
                d.line(
                    [m(*q) for q in st.trails[i]],
                    fill=(*HUD_COLORS[i][:3], 200),
                    width=2,
                )
            cx, cy = m(*st.xy[i])
            d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=HUD_COLORS[i])

    def _caption(self, d: ImageDraw.ImageDraw, st: PairHudState) -> None:
        d.rounded_rectangle(
            (14, self.h - 14 - 30, 14 + 410, self.h - 14), 10, fill=PANEL
        )
        d.text((26, self.h - 14 - 24), st.caption, font=self.f_small, fill=TEXT)

    def _toasts(self, d: ImageDraw.ImageDraw, st: PairHudState) -> None:
        live = [t for t in st.toasts if 0 <= st.t - t.t < TOAST_S]
        for k, toast in enumerate(live[-2:]):
            age = st.t - toast.t
            alpha = int(255 * min(1.0, (TOAST_S - age) / 0.5, age / 0.12 + 0.2))
            tw = d.textlength(toast.text, font=self.f_toast)
            y = 112 + k * 50
            x = (self.w - tw) / 2
            d.rounded_rectangle(
                (x - 16, y - 6, x + tw + 16, y + 38),
                14,
                fill=(14, 18, 26, int(alpha * 0.8)),
            )
            d.text(
                (x, y), toast.text, font=self.f_toast, fill=(*toast.color[:3], alpha)
            )
        if st.outcome:
            tw = d.textlength(st.outcome, font=self.f_toast)
            bx, by = (self.w - tw) / 2, self.h * 0.64
            col = (
                RED
                if st.outcome == "COLLISION"
                else GREEN
                if "CLEAN" in st.outcome
                else ORANGE
            )
            d.rounded_rectangle(
                (bx - 26, by - 12, bx + tw + 26, by + 70), 16, fill=(14, 18, 26, 230)
            )
            d.text((bx, by), st.outcome, font=self.f_toast, fill=col)
            dw = d.textlength(st.outcome_detail, font=self.f_small)
            d.text(
                ((self.w - dw) / 2, by + 40),
                st.outcome_detail,
                font=self.f_small,
                fill=MUTED,
            )


class PairScene:
    """Headless PyBullet scene: ground, lane lines, goal rings, two coloured drones."""

    def __init__(self, run: PairRun, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.run = run
        self.width, self.height = width, height
        self._cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._cid
        )
        plane = p.loadURDF("plane.urdf", physicsClientId=self._cid)
        p.changeVisualShape(
            plane, -1, rgbaColor=[*GROUND_COLOR, 1], physicsClientId=self._cid
        )
        urdf = files("gym_pybullet_drones") / "assets" / "cf2x.urdf"
        self._drones = []
        for rgb in COLORS:
            body = p.loadURDF(
                str(urdf),
                [0, 0, 1],
                globalScaling=DRONE_SCALE,
                physicsClientId=self._cid,
            )
            p.changeVisualShape(
                body, -1, rgbaColor=[*rgb, 1.0], physicsClientId=self._cid
            )
            self._drones.append(int(body))
        self._rings = [self._disc(0.0, 0.0, 0.012, 0.3, rgb, 0.9) for rgb in COLORS]
        for (gx, gy), rgb in zip(run.goals, COLORS, strict=False):
            self._disc(gx, gy, 0.01, 0.7, rgb, 0.45)
        self._beacons = [self._beacon(rgb) for rgb in COLORS]
        self._lanes(run)
        self.eye = np.array([0.0, -6.0, 6.0])
        self.target = np.array([0.0, 0.0, 1.0])
        self._first = True

    def _beacon(self, rgb: tuple[float, float, float]) -> int:
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=BEACON_RADIUS_M,
            rgbaColor=[*rgb, 1.0],
            physicsClientId=self._cid,
        )
        return int(
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[0, 0, -50],
                physicsClientId=self._cid,
            )
        )

    def _disc(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        rgb: tuple[float, float, float],
        alpha: float,
    ) -> int:
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=0.02,
            rgbaColor=[*rgb, alpha],
            physicsClientId=self._cid,
        )
        return int(
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, z],
                physicsClientId=self._cid,
            )
        )

    def _lanes(self, run: PairRun) -> None:
        """Dashed centre line along the head-on corridor, two edge lines (visual references)."""

        if run.spec.scenario != "head_on":
            return
        half = run.spec.separation_m / 2.0 + 1.2
        for y in (-2.0, 0.0, 2.0):
            for x in np.arange(-half, half, 1.0):
                seg = 0.5 if y == 0.0 else 0.9
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[seg / 2, 0.025, 0.004],
                    rgbaColor=[*LANE_COLOR, 0.9],
                    physicsClientId=self._cid,
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=vis,
                    basePosition=[x + 0.5, y, 0.004],
                    physicsClientId=self._cid,
                )

    def place(self, i: int, pos: np.ndarray, vel: np.ndarray) -> None:
        tilt = (float(-0.18 * vel[1]), float(0.18 * vel[0]))
        quat = p.getQuaternionFromEuler([tilt[0], tilt[1], 0.0])
        p.resetBasePositionAndOrientation(
            self._drones[i], pos.tolist(), quat, physicsClientId=self._cid
        )
        p.resetBasePositionAndOrientation(
            self._beacons[i],
            [pos[0], pos[1], pos[2] + BEACON_LIFT_M],
            [0, 0, 0, 1],
            physicsClientId=self._cid,
        )
        p.resetBasePositionAndOrientation(
            self._rings[i],
            [pos[0], pos[1], 0.012],
            [0, 0, 0, 1],
            physicsClientId=self._cid,
        )

    def follow(self, a: np.ndarray, b: np.ndarray) -> None:
        """Smoothed overview: look at the midpoint, back off as they separate."""

        mid = (a + b) / 2.0
        sep = float(np.linalg.norm(a[:2] - b[:2]))
        dist = float(np.clip(3.0 + 0.56 * sep, CAM_DIST_MIN, CAM_DIST_MAX))
        az = np.radians(CAM_AZIMUTH_DEG.get(self.run.spec.scenario, 0.0))
        pitch = np.radians(CAM_PITCH_DEG)
        want_eye = mid + dist * np.array(
            [-np.sin(az) * np.cos(pitch), -np.cos(az) * np.cos(pitch), np.sin(pitch)]
        )
        want_tgt = np.array([mid[0], mid[1], 0.7])
        alpha = 1.0 if self._first else CAM_SMOOTH
        self._first = False
        self.eye = (1 - alpha) * self.eye + alpha * want_eye
        self.target = (1 - alpha) * self.target + alpha * want_tgt

    def matrices(self) -> tuple[list[float], list[float]]:
        view = p.computeViewMatrix(
            self.eye.tolist(),
            self.target.tolist(),
            [0, 0, 1],
            physicsClientId=self._cid,
        )
        proj = p.computeProjectionMatrixFOV(
            FOV_DEG, self.width / self.height, 0.1, 80.0, physicsClientId=self._cid
        )
        return list(view), list(proj)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        view, proj = self.matrices()
        _, _, rgb, depth, _ = p.getCameraImage(
            self.width,
            self.height,
            viewMatrix=view,
            projectionMatrix=proj,
            lightDirection=LIGHT_DIR,
            shadow=1,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._cid,
        )
        img = np.reshape(np.array(rgb, dtype=np.uint8), (self.height, self.width, 4))[
            :, :, :3
        ]
        return img, np.reshape(
            np.array(depth, dtype=np.float32), (self.height, self.width)
        )

    def project(self, point: np.ndarray) -> tuple[float, float, bool]:
        view, proj = self.matrices()
        v = np.array(view).reshape(4, 4, order="F")
        pr = np.array(proj).reshape(4, 4, order="F")
        clip = pr @ v @ np.array([point[0], point[1], point[2], 1.0])
        if clip[3] <= 1e-6:
            return 0.0, 0.0, False
        ndc = clip[:3] / clip[3]
        return (
            (ndc[0] * 0.5 + 0.5) * self.width,
            (1.0 - (ndc[1] * 0.5 + 0.5)) * self.height,
            bool(abs(ndc[0]) < 1.3 and abs(ndc[1]) < 1.3 and ndc[2] < 1.0),
        )

    def close(self) -> None:
        p.disconnect(self._cid)


class TwoDroneRenderer:
    """Turns a ``PairRun`` into a stream of composed RGB frames."""

    def __init__(self, run: PairRun, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.run = run
        if not run.goals:
            run.goals = _goal_points(run)
        self.scene = PairScene(run, width, height)
        xs = [g[0] for g in run.goals] + [
            float(run.traj["pos"][0, i, 0]) for i in range(2)
        ]
        ys = [g[1] for g in run.goals] + [
            float(run.traj["pos"][0, i, 1]) for i in range(2)
        ]
        self.hud = PairHud(
            width,
            height,
            (min(xs) - 1.0, max(xs) + 1.0),
            (min(min(ys), -3.0), max(max(ys), 3.0)),
        )
        self.sky = sky_gradient(width, height)
        self.toasts = self._toasts()
        self.caption = (
            f"{run.spec.scenario.replace('_', '-')}, seed {run.spec.seed}, "
            f"L_A={run.latencies_ms[0]:.0f} ms, L_B={run.latencies_ms[1]:.0f} ms"
        )
        self._times = self._frame_times()

    # ----------------------------------------------------------------- script
    def _toasts(self) -> list[Toast]:
        run, out = self.run, []
        for i, rows in enumerate(run.decisions):
            last = Action.FORWARD
            for r in rows:
                act = Action(int(r["action"]))
                if act != last and act in LATERAL:
                    out.append(
                        Toast(
                            float(r["apply_tick"]) * DT,
                            f"{NAMES[i]}: {act.name.replace('_', ' ')}",
                            HUD_COLORS[i],
                        )
                    )
                last = act
        clear = float(run.metrics["min_clearance_m"])
        if not run.metrics["collision"] and clear < NEAR_MISS_CLEARANCE_M:
            out.append(
                Toast(
                    float(run.metrics["min_clearance_t"]),
                    f"NEAR MISS {clear:.2f} m",
                    ORANGE,
                )
            )
        return sorted(out, key=lambda x: x.t)

    def _frame_times(self) -> list[float]:
        """Sim time of every video frame: 1x normally, 0.4x while the drones are close."""

        run, t, times = self.run, 0.0, []
        while t < run.t_end:
            times.append(t)
            t += (
                SLOWMO_SPEED / FPS
                if run.clearance(run.tick_at(t)) < SLOWMO_SEP_M
                else 1.0 / FPS
            )
        times += [run.t_end] * int(END_HOLD_S * FPS)
        return times

    @property
    def n_frames(self) -> int:
        return len(self._times)

    def frame_times(self) -> list[float]:
        return list(self._times)

    # ------------------------------------------------------------------ frames
    def _drone_hud(self, i: int, k: int) -> DroneHud:
        run = self.run
        prog = float(run.traj["progress"][k, i])
        return DroneHud(
            Action(int(run.traj["action"][k, i])),
            None if np.isnan(prog) else prog,
            run.latencies_ms[i],
            ARM_LABELS.get(run.arm_ids[i], run.arm_ids[i]),
        )

    def _outcome(self, t: float) -> tuple[str | None, str]:
        run = self.run
        if t < run.t_end - 1e-6:
            return None, ""
        clear = float(run.metrics["min_clearance_m"])
        if run.metrics["collision"]:
            return "COLLISION", "drones made contact"
        if clear < NEAR_MISS_CLEARANCE_M:
            return f"NEAR MISS {clear:.2f} m", "closest approach, no contact"
        return "CLEAN PASS", f"closest approach {clear:.2f} m"

    def frame_at(self, t: float, speed: float = 1.0) -> Image.Image:
        run, sc = self.run, self.scene
        k = run.tick_at(t)
        pos, vel = run.traj["pos"][k], run.traj["vel"][k]
        for i in range(2):
            sc.place(i, pos[i], vel[i])
        sc.follow(pos[0], pos[1])
        rgb, depth = sc.render()
        img = Image.fromarray(np.where((depth >= 0.9999)[..., None], self.sky, rgb))
        lo = max(0, k - int(TRAIL_S / DT))
        trail3d = tuple(
            [
                sc.project(run.traj["pos"][q, i])[:2]
                for q in range(lo, k + 1, 4)
                if sc.project(run.traj["pos"][q, i])[2]
            ]
            for i in range(2)
        )
        badges = []
        for i in range(2):
            x, y, ok = sc.project(pos[i] + np.array([0.0, 0.0, 0.75]))
            if ok:
                badges.append((x, y, i))
        a, b = sc.project(pos[0]), sc.project(pos[1])
        link = ((a[0], a[1]), (b[0], b[1])) if a[2] and b[2] else None
        history = [
            [
                (float(run.traj["pos"][q, i, 0]), float(run.traj["pos"][q, i, 1]))
                for q in range(lo, k + 1, 4)
            ]
            for i in range(2)
        ]
        outcome, detail = self._outcome(t)
        clear_now = run.clearance(k)
        state = PairHudState(
            t=float(run.traj["t"][k]),
            speed=speed,
            clearance=clear_now,
            min_clearance=float(np.min(run.traj["sep"][: k + 1])) - 2.0 * DRONE_RADIUS,
            drones=(self._drone_hud(0, k), self._drone_hud(1, k)),
            toasts=self.toasts,
            trails=(history[0], history[1]),
            xy=(
                (float(pos[0][0]), float(pos[0][1])),
                (float(pos[1][0]), float(pos[1][1])),
            ),
            goals=run.goals,
            caption=self.caption,
            outcome=outcome,
            outcome_detail=detail,
        )
        return self.hud.compose(img, state, badges, link, trail3d)  # type: ignore[arg-type]

    def frames(self) -> Iterator[tuple[float, Image.Image]]:
        for t in self._times:
            near = (
                self.run.clearance(self.run.tick_at(t)) < SLOWMO_SEP_M
                and t < self.run.t_end
            )
            yield t, self.frame_at(t, SLOWMO_SPEED if near else 1.0)

    def close(self) -> None:
        self.scene.close()


def _goal_points(run: PairRun) -> list[tuple[float, float]]:
    """World xy of each drone's goal: the other drone's start for head-on, else straight ahead."""

    pos0 = run.traj["pos"][0]
    sep = run.spec.separation_m
    if run.spec.scenario == "head_on":
        return [
            (float(pos0[1][0]), float(pos0[0][1])),
            (float(pos0[0][0]), float(pos0[1][1])),
        ]
    return [
        (float(pos0[0][0]) + sep, float(pos0[0][1])),
        (float(pos0[1][0]), float(pos0[1][1]) + sep),
    ]


__all__ = [
    "COLORS",
    "PairRun",
    "TwoDroneRenderer",
    "danger_color",
]
