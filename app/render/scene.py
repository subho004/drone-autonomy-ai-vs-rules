"""3D scene for rendering a recorded mission flight (PyBullet's CPU renderer, headless).

The scene is rebuilt from the scenario spec and the recorded trajectory only: nothing here
simulates physics. The drone is drawn at a larger scale than real so it reads on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pybullet as p
import pybullet_data

from app.scenarios.spec import MissionSpec, ObstacleSpec
from app.sim.world import DT

DRONE_SCALE = 4.0
PAD_RADIUS = 0.9
STATION_RADIUS = 0.6
RING_ALPHA = 0.45
STATION_COLOR = (0.15, 0.55, 0.95)
SCANNED_COLOR = (0.2, 0.8, 0.35)
MISSED_COLOR = (0.9, 0.55, 0.15)
ACTIVE_COLOR = (0.98, 0.78, 0.15)
OBSTACLE_COLOR = (0.92, 0.2, 0.2)
GROUND_COLOR = (0.62, 0.74, 0.64)
FOV_DEG = 62.0
CAM_BACK_M = 3.2
CAM_UP_M = 2.1
CAM_LOOK_AHEAD_M = 1.6
CAM_SMOOTH = 0.06
LIGHT_DIR = (0.4, -0.5, 1.0)


@dataclass(slots=True)
class CameraPose:
    eye: np.ndarray
    target: np.ndarray


class MissionScene:
    """Headless PyBullet scene with stations, pad, obstacles and the drone model."""

    def __init__(
        self, mission: MissionSpec | None = None, width: int = 960, height: int = 540
    ) -> None:
        self.mission = mission
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
        self._drone = p.loadURDF(
            str(urdf), [0, 0, 1], globalScaling=DRONE_SCALE, physicsClientId=self._cid
        )
        self._stations: list[list[int]] = []
        self._obstacles: list[int] = []
        if mission is not None:
            self._stations = [self._station(i) for i in range(len(mission.stations))]
            self._pad = self._disc(
                mission.pad_x, mission.pad_y, 0.012, PAD_RADIUS, (0.25, 0.27, 0.32)
            )
            self._pad_mark = self._disc(
                mission.pad_x,
                mission.pad_y,
                0.02,
                PAD_RADIUS * 0.55,
                (0.95, 0.95, 0.95),
            )
            self._obstacles = [self._sphere(o) for o in mission.obstacles]
        self.cam = CameraPose(
            np.array([-CAM_BACK_M, 0.0, CAM_UP_M + 1]), np.array([0.0, 0, 1])
        )

    # ------------------------------------------------------------- building
    def _disc(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        rgb: tuple[float, float, float],
        alpha: float = 1.0,
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

    def _station(self, i: int) -> list[int]:
        assert self.mission is not None
        st = self.mission.stations[i]
        ring = self._disc(st.x, st.y, st.z, STATION_RADIUS, STATION_COLOR, RING_ALPHA)
        pole_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.025,
            length=st.z,
            rgbaColor=[0.75, 0.78, 0.85, 1.0],
            physicsClientId=self._cid,
        )
        pole = int(
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=pole_vis,
                basePosition=[st.x, st.y, st.z / 2],
                physicsClientId=self._cid,
            )
        )
        return [ring, pole]

    def _sphere(self, ob: ObstacleSpec) -> int:
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=ob.radius,
            rgbaColor=[*OBSTACLE_COLOR, 1.0],
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

    # -------------------------------------------------------------- updating
    def set_station_colors(self, colors: list[tuple[float, float, float]]) -> None:
        for (ring, _), rgb in zip(self._stations, colors, strict=True):
            p.changeVisualShape(
                ring, -1, rgbaColor=[*rgb, 1.0], physicsClientId=self._cid
            )

    def place_drone(
        self, pos: np.ndarray, yaw: float = 0.0, tilt: tuple[float, float] = (0, 0)
    ) -> None:
        quat = p.getQuaternionFromEuler([tilt[0], tilt[1], yaw])
        p.resetBasePositionAndOrientation(
            self._drone, pos.tolist(), quat, physicsClientId=self._cid
        )

    def place_obstacle(self, i: int, pos: np.ndarray | None) -> None:
        target = [0.0, 0.0, -50.0] if pos is None else pos.tolist()
        p.resetBasePositionAndOrientation(
            self._obstacles[i], target, [0, 0, 0, 1], physicsClientId=self._cid
        )

    # ---------------------------------------------------------------- camera
    def follow(self, drone: np.ndarray, vel: np.ndarray, first: bool = False) -> None:
        """Smoothed chase camera: behind and above, looking slightly ahead of the drone."""

        speed = float(np.hypot(vel[0], vel[1]))
        heading = (
            np.array([vel[0], vel[1]]) / speed if speed > 0.25 else np.array([1.0, 0.0])
        )
        want_eye = np.array(
            [
                drone[0] - CAM_BACK_M * heading[0],
                drone[1] - CAM_BACK_M * heading[1] - 0.8,
                drone[2] + CAM_UP_M,
            ]
        )
        want_tgt = np.array(
            [
                drone[0] + CAM_LOOK_AHEAD_M * heading[0],
                drone[1] + CAM_LOOK_AHEAD_M * heading[1],
                max(drone[2] - 0.35, 0.3),
            ]
        )
        a = 1.0 if first else CAM_SMOOTH
        self.cam.eye = (1 - a) * self.cam.eye + a * want_eye
        self.cam.target = (1 - a) * self.cam.target + a * want_tgt

    def matrices(self) -> tuple[list[float], list[float]]:
        view = p.computeViewMatrix(
            self.cam.eye.tolist(),
            self.cam.target.tolist(),
            [0, 0, 1],
            physicsClientId=self._cid,
        )
        proj = p.computeProjectionMatrixFOV(
            FOV_DEG, self.width / self.height, 0.1, 80.0, physicsClientId=self._cid
        )
        return list(view), list(proj)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB frame (H, W, 3 uint8) and depth buffer (H, W float in [0, 1])."""

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
        """World point -> (pixel x, pixel y, in front of the camera)."""

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


def obstacle_track(
    spec: ObstacleSpec, drone_xyz: np.ndarray, x_trace: np.ndarray
) -> tuple[int | None, np.ndarray, np.ndarray]:
    """Spawn tick, start position and velocity of one obstacle, from the drone's x trace.

    Mirrors ``DroneWorld._maybe_spawn``: it fires at the first tick the drone's x reaches
    ``trigger_x`` and is placed relative to the drone at that tick.
    """

    idx = np.nonzero(x_trace >= spec.trigger_x)[0]
    if idx.size == 0:
        return None, np.zeros(3), np.zeros(3)
    k = int(idx[0])
    d = drone_xyz[k]
    if spec.kind == "cross":
        start = np.array(
            [d[0] + spec.d_spawn, spec.cross_side * spec.cross_start_y, d[2]]
        )
        vel = np.array([0.0, -spec.cross_side * spec.speed, 0.0])
    else:
        start = np.array([d[0] + spec.d_spawn, spec.lateral_offset, d[2]])
        vel = np.array([-spec.speed if spec.kind == "approach" else 0.0, 0.0, 0.0])
    return k, start, vel


def obstacle_position(
    k_spawn: int, start: np.ndarray, vel: np.ndarray, tick: int
) -> np.ndarray:
    """Position at control tick ``tick`` (moves by vel*DT on the spawn tick and every tick after)."""

    return start + vel * DT * (tick - k_spawn + 1)


__all__ = [
    "ACTIVE_COLOR",
    "MISSED_COLOR",
    "SCANNED_COLOR",
    "STATION_COLOR",
    "CameraPose",
    "MissionScene",
    "obstacle_position",
    "obstacle_track",
]
