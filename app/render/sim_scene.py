"""3D scene for the Group A simulations: zones, sites, gates, walls, a follow target.

Built on ``MissionScene`` (ground, drone, camera, projection) with no mission attached;
extra geometry is added from the scenario spec. Nothing here simulates physics.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from app.render.scene import MissionScene
from app.scenarios.spec import BoxSpec, ScenarioSpec

Rgb = tuple[float, float, float]
ZONE_COLOR: Rgb = (0.86, 0.18, 0.18)
WALL_COLOR: Rgb = (0.55, 0.57, 0.62)
SAFE_SITE: Rgb = (0.2, 0.75, 0.35)
HAZARD_SITE: Rgb = (0.85, 0.2, 0.2)
TARGET_COLOR: Rgb = (0.1, 0.8, 0.95)
GATE_TODO: Rgb = (0.15, 0.55, 0.95)
GATE_DONE: Rgb = (0.2, 0.8, 0.35)
GATE_MISSED: Rgb = (0.95, 0.5, 0.1)
GATE_ACTIVE: Rgb = (0.98, 0.78, 0.15)
GATE_HALF_W = 0.5
GATE_HALF_H = 0.35
BAR = 0.05
ZONE_POST_H = 1.4
GHOST_COLOR: Rgb = (0.75, 0.8, 0.95)
ORB_COLOR: Rgb = (1.0, 0.45, 0.1)
CHASER_SCALE = 5.0


class SimScene(MissionScene):
    """Scene content for one scenario family, placed from the spec."""

    def __init__(
        self,
        spec: ScenarioSpec,
        width: int = 960,
        height: int = 540,
        skin: str = "orb",
    ) -> None:
        super().__init__(None, width, height)
        self.spec = spec
        self._gate_bars: list[list[int]] = []
        self._sphere_ids: dict[str, int] = {}
        self._chasers: dict[str, int] = {}
        for b in spec.boxes:
            self._box(b.x, b.y, b.z, b.hx, b.hy, b.hz, WALL_COLOR)
        m = spec.mission
        if m is not None:
            self._disc(m.pad_x, m.pad_y, 0.012, 0.9, (0.25, 0.27, 0.32))
            self._disc(m.pad_x, m.pad_y, 0.02, 0.5, (0.95, 0.95, 0.95))
            for z in m.zones:
                self._zone(z.x0, z.x1, z.y0, z.y1)
            if m.gate_mode:
                self._gate_bars = [self._gate(s.x, s.y, s.z) for s in m.stations]
        elif spec.follow is None and not spec.sites:
            self._disc(spec.goal_x, 0.0, 0.012, 0.9, (0.25, 0.27, 0.32))
            self._disc(spec.goal_x, 0.0, 0.02, 0.5, (0.95, 0.95, 0.95))
        for s in spec.sites:
            self._disc(
                s.x, s.y, 0.015, s.radius, HAZARD_SITE if s.hazard else SAFE_SITE
            )
            self._disc(s.x, s.y, 0.025, s.radius * 0.45, (0.95, 0.95, 0.95))
        if spec.follow is not None:
            self.add_sphere("target", 0.2, TARGET_COLOR)
        self.n_orbs = 0
        if m is not None and spec.event_type == "orbs":
            self.n_orbs = len(m.obstacles)
            for k, o in enumerate(m.obstacles):
                if skin == "drone":
                    self.add_chaser(f"orb{k}")
                else:
                    self.add_sphere(f"orb{k}", o.radius, ORB_COLOR)
        radius = spec.obstacle.radius if spec.obstacle is not None else 0.3
        self.add_sphere("obstacle", radius, (0.92, 0.2, 0.2))

    # ------------------------------------------------------------- building
    def _box(
        self, x: float, y: float, z: float, hx: float, hy: float, hz: float, rgb: Rgb
    ) -> int:
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=[*rgb, 1.0],
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

    def _zone(self, x0: float, x1: float, y0: float, y1: float) -> None:
        cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        self._box(cx, cy, 0.02, hx, hy, 0.02, ZONE_COLOR)
        for px in (x0, x1):
            for py in (y0, y1):
                self._box(
                    px, py, ZONE_POST_H / 2, 0.04, 0.04, ZONE_POST_H / 2, ZONE_COLOR
                )

    def _gate(self, x: float, y: float, z: float) -> list[int]:
        hw, hh = GATE_HALF_W, GATE_HALF_H
        return [
            self._box(x, y, z + hh, BAR, hw + BAR, BAR, GATE_TODO),
            self._box(x, y, z - hh, BAR, hw + BAR, BAR, GATE_TODO),
            self._box(x, y + hw, z, BAR, BAR, hh, GATE_TODO),
            self._box(x, y - hw, z, BAR, BAR, hh, GATE_TODO),
        ]

    def add_sphere(self, name: str, radius: float, rgb: Rgb) -> None:
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[*rgb, 1.0],
            physicsClientId=self._cid,
        )
        self._sphere_ids[name] = int(
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[0, 0, -50],
                physicsClientId=self._cid,
            )
        )

    def add_chaser(self, name: str) -> None:
        """A rival drone model (purely visual; it moves exactly like an orb)."""

        from importlib.resources import files

        urdf = files("gym_pybullet_drones") / "assets" / "cf2x.urdf"
        body = int(
            p.loadURDF(
                str(urdf),
                [0, 0, -50],
                globalScaling=CHASER_SCALE,
                physicsClientId=self._cid,
            )
        )
        for link in range(-1, p.getNumJoints(body, physicsClientId=self._cid)):
            p.changeVisualShape(
                body, link, rgbaColor=[*ORB_COLOR, 1.0], physicsClientId=self._cid
            )
        self._chasers[name] = body

    # -------------------------------------------------------------- updating
    def place_sphere(self, name: str, pos: np.ndarray | None) -> None:
        if name in self._chasers:
            target = [0.0, 0.0, -50.0] if pos is None else pos.tolist()
            p.resetBasePositionAndOrientation(
                self._chasers[name], target, [0, 0, 0, 1], physicsClientId=self._cid
            )
            return
        if name not in self._sphere_ids:
            return
        target = [0.0, 0.0, -50.0] if pos is None else pos.tolist()
        p.resetBasePositionAndOrientation(
            self._sphere_ids[name], target, [0, 0, 0, 1], physicsClientId=self._cid
        )

    def set_gate_colors(self, colors: list[Rgb]) -> None:
        for bars, rgb in zip(self._gate_bars, colors, strict=True):
            for bar in bars:
                p.changeVisualShape(
                    bar, -1, rgbaColor=[*rgb, 1.0], physicsClientId=self._cid
                )


def box_extent(b: BoxSpec) -> tuple[float, float, float, float]:
    return b.x - b.hx, b.x + b.hx, b.y - b.hy, b.y + b.hy


__all__ = [
    "GATE_ACTIVE",
    "GATE_DONE",
    "GATE_MISSED",
    "GATE_TODO",
    "SimScene",
    "box_extent",
]
