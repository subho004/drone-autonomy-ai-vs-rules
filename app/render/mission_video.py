"""Render a recorded mission flight to frames, screenshots and video."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from app.core.logging import get_logger
from app.domain.actions import Action
from app.render.hud import GREEN, ORANGE, RED, Hud, HudState, Toast, sky_gradient
from app.render.scene import (
    ACTIVE_COLOR,
    MISSED_COLOR,
    SCANNED_COLOR,
    STATION_COLOR,
    MissionScene,
    obstacle_position,
    obstacle_track,
)
from app.scenarios.spec import ScenarioSpec
from app.sim.world import DT

logger = get_logger(__name__)

FPS = 24
WIDTH, HEIGHT = 960, 540
SETTLE_TICKS = 24  # the world's pre-flight hover; trajectory ticks start after it
END_HOLD_S = 2.5
LANDED_Z_M = 0.25  # below this the landing counts as completed on screen


@dataclass(slots=True)
class MissionRun:
    flight_id: int
    arm_id: str
    model: str
    shield: bool
    outcome: str
    success: bool
    spec: ScenarioSpec
    traj: dict[str, np.ndarray]
    decisions: list[dict[str, float | int | str | None]]
    events: list[dict[str, float | int | str]]
    mean_latency_ms: float
    stations_scanned: int
    stations_total: int
    t_end: float
    notes: list[str] = field(default_factory=list)


def load_run(
    conn: sqlite3.Connection, flight_id: int, root: Path, require_mission: bool = True
) -> MissionRun:
    """Load one mission flight (metrics, trajectory, decisions, events) from SQLite."""

    f = conn.execute(
        "SELECT f.*, a.model AS model_name, a.provider AS provider FROM flights f "
        "JOIN arms a ON a.id = f.arm_id WHERE f.id = ?",
        (flight_id,),
    ).fetchone()
    if f is None:
        raise ValueError(f"flight {flight_id} not found")
    spec = ScenarioSpec.model_validate_json(
        conn.execute(
            "SELECT spec_json FROM scenarios WHERE id = ?", (f["scenario_id"],)
        ).fetchone()[0]
    )
    if require_mission and spec.mission is None:
        raise ValueError(f"flight {flight_id} is not a mission flight")
    table = pq.read_table(root / f["trajectory_uri"])
    traj = {c: table.column(c).to_numpy() for c in table.column_names}
    rows = conn.execute(
        "SELECT obs_tick, apply_tick, applied_action, latency_ms, source, valid FROM decisions "
        "WHERE flight_id = ? ORDER BY seq",
        (flight_id,),
    ).fetchall()
    model = f["model_name"] or f["arm_id"]
    return MissionRun(
        flight_id=flight_id,
        arm_id=f["arm_id"],
        model=model,
        shield=bool(f["shield"]),
        outcome=f["outcome"],
        success=bool(f["success"]),
        spec=spec,
        traj=traj,
        decisions=[dict(r) for r in rows],
        events=json.loads(f["events_json"] or "[]"),
        mean_latency_ms=float(f["lat_mean_ms"] or 0.0),
        stations_scanned=int(f["stations_scanned"] or 0),
        stations_total=int(f["stations_total"] or 0),
        t_end=float(f["t_end"]),
    )


DISPLAY_NAMES = {
    "typesafe/jev-1.13": "Jev 1.13",
    "inception/mercury-decide:free": "Mercury Decide",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gpt-6-luna": "GPT-6 Luna",
}


class MissionRenderer:
    """Turns a ``MissionRun`` into a stream of composed RGB frames."""

    def __init__(
        self, run: MissionRun, width: int = WIDTH, height: int = HEIGHT
    ) -> None:
        self.run = run
        m = run.spec.mission
        assert m is not None
        self.m = m
        self.scene = MissionScene(m, width, height)
        self.hud = Hud(width, height, (-1.0, m.pad_x + 1.5), (-3.4, 3.4))
        self.sky = sky_gradient(width, height)
        t = run.traj
        self.xyz = np.stack([t["x"], t["y"], t["z"]], axis=1)
        self.vel = np.stack([t["vx"], t["vy"], t["vz"]], axis=1)
        self.tracks = [obstacle_track(o, self.xyz, t["x"]) for o in m.obstacles]
        self.toasts = self._toasts()
        self.n_ticks = len(self.xyz)
        self._first = True

    # ----------------------------------------------------------------- script
    def _toasts(self) -> list[Toast]:
        out: list[Toast] = []
        for e in self.run.events:
            n = int(e["station"]) + 1
            if e["kind"] == "scanned":
                out.append(Toast(float(e["t"]), f"STATION {n} SCANNED", GREEN))
            else:
                out.append(Toast(float(e["t"]), f"STATION {n} MISSED", ORANGE))
        for i, (k, _, _) in enumerate(self.tracks):
            if k is not None:
                kind = self.m.obstacles[i].kind
                label = "INTRUDER CROSSING" if kind == "cross" else "OBSTACLE AHEAD"
                out.append(Toast((k + 1) * DT, label, RED))
        last = None
        for d in self.run.decisions:
            act = int(d["applied_action"] or 0)
            if act in (Action.AVOID_LEFT, Action.AVOID_RIGHT) and act != last:
                name = Action(act).name.replace("_", " ")
                out.append(Toast(float(d["apply_tick"] or 0) * DT, name, ORANGE))
            last = act
        return sorted(out, key=lambda x: x.t)

    def _status(self, t: float) -> tuple[list[str], str]:
        status = ["todo"] * self.run.stations_total
        handled = 0
        for e in self.run.events:
            if float(e["t"]) <= t:
                status[int(e["station"])] = (
                    "done" if e["kind"] == "scanned" else "missed"
                )
                handled += 1
        if handled < len(status):
            status[handled] = "active"
        pad = "todo"
        if handled >= len(status):
            pad = "active"
        i = int(np.clip(round(t / DT) - 1, 0, self.n_ticks - 1))
        if self.run.outcome == "landed_on_pad" and self.xyz[i][2] < LANDED_Z_M:
            pad = "done"
        return status, pad

    def _decision_state(self, i: int) -> tuple[float | None, float | None]:
        for d in self.run.decisions:
            o, a = int(d["obs_tick"] or 0), int(d["apply_tick"] or 0)
            if o <= i < a and a > o:
                return (i - o) / (a - o), float(d["latency_ms"] or 0.0)
        return None, None

    # ------------------------------------------------------------------ frames
    def frame_at(self, t: float) -> Image.Image:
        i = int(np.clip(round(t / DT) - 1, 0, self.n_ticks - 1))
        pos, vel = self.xyz[i], self.vel[i]
        tilt = (float(-0.18 * vel[1]), float(0.18 * vel[0]))
        self.scene.place_drone(pos, 0.0, tilt)
        self.scene.follow(pos, vel, first=self._first)
        self._first = False
        status, pad = self._status(t)
        colors = [
            {"done": SCANNED_COLOR, "missed": MISSED_COLOR, "active": ACTIVE_COLOR}.get(
                s, STATION_COLOR
            )
            for s in status
        ]
        self.scene.set_station_colors(colors)
        obstacles_xy: list[tuple[float, float]] = []
        for j, (k, start, v) in enumerate(self.tracks):
            if k is not None and i >= k:
                op = obstacle_position(k, start, v, i)
                self.scene.place_obstacle(j, op)
                obstacles_xy.append((float(op[0]), float(op[1])))
            else:
                self.scene.place_obstacle(j, None)
        rgb, depth = self.scene.render()
        frame = np.where((depth >= 0.9999)[..., None], self.sky, rgb)
        img = Image.fromarray(frame)
        lookback = max(0, i - int(5.0 / DT))
        trail3d = [self.scene.project(self.xyz[q]) for q in range(lookback, i + 1, 4)]
        img = self.hud.trail_overlay(img, trail3d)
        labels = self._labels(status)
        progress, lat = self._decision_state(i)
        action = Action(int(self.run.traj["action"][i]))
        state = HudState(
            t=t,
            model=DISPLAY_NAMES.get(self.run.model, self.run.model),
            subtitle=f"mean decision time {self.run.mean_latency_ms:.0f} ms",
            action=action,
            shield=self.run.shield,
            deciding_progress=progress,
            deciding_ms=lat,
            station_status=status,
            pad_status=pad,
            toasts=self.toasts,
            trail=[
                (float(self.xyz[q][0]), float(self.xyz[q][1]))
                for q in range(lookback, i + 1, 6)
            ],
            drone_xy=(float(pos[0]), float(pos[1])),
            obstacles_xy=obstacles_xy,
            stations_xy=[(s.x, s.y) for s in self.m.stations],
            pad_xy=(self.m.pad_x, self.m.pad_y),
            outcome=self._outcome_key() if t >= self.run.t_end - END_HOLD_S else None,
        )
        return self.hud.compose(img, state, labels)

    def _outcome_key(self) -> str:
        if self.run.success:
            return "mission_complete"
        return "partial" if self.run.outcome == "landed_on_pad" else self.run.outcome

    def _labels(self, status: list[str]) -> list[tuple[float, float, str]]:
        """3D-anchored labels for the active target only (keeps the view uncluttered)."""

        out: list[tuple[float, float, str]] = []
        active = next((i for i, s in enumerate(status) if s == "active"), None)
        if active is not None:
            st = self.m.stations[active]
            x, y, ok = self.scene.project(np.array([st.x, st.y, st.z + 0.6]))
            if ok:
                out.append((x, y, f"STATION {active + 1}"))
        elif all(s != "active" for s in status):
            x, y, ok = self.scene.project(np.array([self.m.pad_x, self.m.pad_y, 0.55]))
            if ok:
                out.append((x, y, "LANDING PAD"))
        return out

    def frames(self) -> Iterator[tuple[float, Image.Image]]:
        total = self.run.t_end + END_HOLD_S
        for f in range(int(total * FPS)):
            t = min(f / FPS, self.run.t_end)
            yield f / FPS, self.frame_at(t)

    def close(self) -> None:
        self.scene.close()


def encode_video(
    frames: Iterator[tuple[float, Image.Image]],
    out_mp4: Path,
    fps: int = FPS,
    size: tuple[int, int] = (WIDTH, HEIGHT),
) -> int:
    """Pipe frames to ffmpeg (H.264). Returns the number of frames written."""

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    n = 0
    for _, img in frames:
        proc.stdin.write(img.tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    return n


def make_gif(
    mp4: Path,
    gif: Path,
    width: int = 560,
    fps: int = 8,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> None:
    """GIF from (a window of) the video, with a per-clip palette to keep it small."""

    palette = gif.with_suffix(".palette.png")
    window = ["-ss", f"{start_s:.2f}"] + (
        ["-t", f"{duration_s:.2f}"] if duration_s else []
    )
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *window,
            "-i",
            str(mp4),
            "-vf",
            f"{scale},palettegen=max_colors=64:stats_mode=diff",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *window,
            "-i",
            str(mp4),
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
            str(gif),
        ],
        check=True,
    )
    palette.unlink(missing_ok=True)


__all__ = [
    "DISPLAY_NAMES",
    "FPS",
    "MissionRenderer",
    "MissionRun",
    "encode_video",
    "load_run",
    "make_gif",
]
