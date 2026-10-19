"""3D render of an inspection flight: structure wall, drone-camera picture-in-picture,
verdict ticker and a live "defects found" checklist.

Composes the existing mission renderer (``MissionRenderer``/``MissionScene``/``Hud``)
without changing it: the wall is added to the scene's PyBullet client, the chase camera is
moved to look across the flight line at the wall, and the extra panels are drawn on top of
the finished mission frame. The picture-in-picture is a second, private client rendering
what the drone camera would see, at the study resolution (nearest-neighbour upscaled so the
actual pixels are visible).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from app.render.hud import (
    GRAY,
    GREEN,
    MUTED,
    PANEL,
    RED,
    TEXT,
    YELLOW,
    Hud,
    HudState,
)
from app.render.mission_video import END_HOLD_S, MissionRenderer, MissionRun
from app.render.scene import CAM_SMOOTH, MissionScene
from app.scenarios.spec import MissionSpec
from app.sim.inspection import InspectionScene, StructureSpec
from app.sim.world import DT

PIP_SIZE = 224
MINIMAP_BOTTOM_PX = (
    222  # the Hud draws its top view h-14 px from the bottom; this lifts it
)
LABEL_MIN_Y_PX = (
    245.0  # station labels right of LABEL_AVOID_X_PX sit below the top view
)
LABEL_AVOID_X_PX = 640.0
MIN_ANALYSE_S = 0.8  # the ticker shows the "asking" state at least this long
SUMMARY_DELAY_S = 1.0  # after landing
CHASE_BACK_M = 2.2
CHASE_SIDE_M = 1.9
CHASE_UP_M = 1.0
CHASE_LOOK_FWD_M = 0.8
CHASE_LOOK_SIDE_M = 1.3
SHORT_NAMES = {
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gpt-4o-mini": "GPT-4o-mini",
}


@dataclass(frozen=True, slots=True)
class Verdict:
    """One panel check: the camera frame at the scan moment and the model's answer."""

    station: int  # station index (0-based)
    panel_index: int
    t_event: float  # sim time the scan completed (the frame was captured here)
    model: str
    resolution: int
    answer: str
    verdict: bool | None  # True = defect reported
    truth: bool
    latency_ms: float
    cost_usd: float
    tokens_in: int
    tokens_out: int

    @property
    def correct(self) -> bool:
        return self.verdict is not None and self.verdict == self.truth


class InspectionMissionScene(MissionScene):
    """``MissionScene`` plus the wall, with a chase camera that looks across at it."""

    def __init__(
        self, mission: MissionSpec, structure: StructureSpec, width: int, height: int
    ) -> None:
        super().__init__(mission, width, height)
        self._wall = InspectionScene(structure, client=self._cid)

    def follow(self, drone: np.ndarray, vel: np.ndarray, first: bool = False) -> None:
        want_eye = np.array(
            [drone[0] - CHASE_BACK_M, drone[1] - CHASE_SIDE_M, drone[2] + CHASE_UP_M]
        )
        want_tgt = np.array(
            [drone[0] + CHASE_LOOK_FWD_M, drone[1] + CHASE_LOOK_SIDE_M, drone[2] + 0.1]
        )
        a = 1.0 if first else CAM_SMOOTH
        self.cam.eye = (1 - a) * self.cam.eye + a * want_eye
        self.cam.target = (1 - a) * self.cam.target + a * want_tgt

    def close(self) -> None:
        self._wall.close()
        super().close()


class InspectionHud(Hud):
    """Mission HUD with an inspection title and the top view lifted above the PiP."""

    def _panel_model(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        d.rounded_rectangle((14, 14, 336, 118), 12, fill=PANEL)
        d.text((28, 22), "STRUCTURE INSPECTION", font=self.f_tiny, fill=MUTED)
        d.text((28, 38), st.model, font=self.f_big, fill=TEXT)
        d.text((28, 70), st.subtitle, font=self.f_small, fill=MUTED)
        d.text(
            (28, 92),
            st.action.name.replace("_", " "),
            font=self.f_small,
            fill=YELLOW,
        )

    def _minimap(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        real_h = self.h
        self.h = MINIMAP_BOTTOM_PX
        try:
            super()._minimap(d, st)
        finally:
            self.h = real_h


class InspectionRenderer(MissionRenderer):
    """Turns a recorded inspection flight plus its verdicts into composed frames."""

    def __init__(
        self,
        run: MissionRun,
        structure: StructureSpec,
        panel_for_station: list[int],
        verdicts: list[Verdict],
        resolution: int,
        model: str,
        width: int = 960,
        height: int = 540,
    ) -> None:
        super().__init__(run, width, height)
        self.scene.close()
        assert run.spec.mission is not None
        self.scene = InspectionMissionScene(run.spec.mission, structure, width, height)
        self.hud = InspectionHud(
            width, height, (-1.0, run.spec.mission.pad_x + 1.5), (-3.4, 3.4)
        )
        self.camera = InspectionScene(structure)
        self.panels = panel_for_station
        self.verdicts = sorted(verdicts, key=lambda v: v.t_event)
        self.resolution = resolution
        self.model = model
        self.width, self.height = width, height

    # ------------------------------------------------------------------- helpers
    def station_index(self, t: float) -> int:
        """Station the camera is aimed at: the first one not yet scanned (or the last)."""

        done = sum(1 for e in self.run.events if float(e["t"]) <= t)
        return min(done, len(self.panels) - 1)

    def pose_at(self, t: float) -> tuple[float, float, float]:
        i = int(np.clip(round(t / DT) - 1, 0, self.n_ticks - 1))
        x, y, z = self.xyz[i]
        return float(x), float(y), float(z)

    def camera_frame(self, t: float) -> np.ndarray:
        """Live view of the active panel; after the last scan it holds the last capture."""

        scans = [float(e["t"]) for e in self.run.events if e["kind"] == "scanned"]
        if len(scans) >= len(self.panels):
            t = min(t, scans[len(self.panels) - 1])
        panel = self.panels[self.station_index(t)]
        return self.camera.render_camera(self.pose_at(t), panel, self.resolution)

    def _labels(self, status: list[str]) -> list[tuple[float, float, str]]:
        """Station labels, kept clear of the top-view panel in the upper right."""

        return [
            (x, max(y, LABEL_MIN_Y_PX) if x > LABEL_AVOID_X_PX else y, text)
            for x, y, text in super()._labels(status)
        ]

    # -------------------------------------------------------------------- frames
    def frame_at(self, t: float) -> Image.Image:
        base = super().frame_at(t)
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, "RGBA")
        self._pip(layer, d, t)
        self._checklist(d, t)
        self._ticker(d, t)
        self._summary(d, t)
        return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")

    def _pip(self, layer: Image.Image, d: ImageDraw.ImageDraw, t: float) -> None:
        x0, y0 = self.width - 14 - PIP_SIZE, self.height - 14 - PIP_SIZE
        d.rounded_rectangle(
            (x0 - 8, y0 - 30, x0 + PIP_SIZE + 8, y0 + PIP_SIZE + 8), 12, fill=PANEL
        )
        d.text(
            (x0, y0 - 24),
            f"DRONE CAMERA  {self.resolution}x{self.resolution}",
            font=self.hud.f_tiny,
            fill=MUTED,
        )
        frame = Image.fromarray(self.camera_frame(t)).resize(
            (PIP_SIZE, PIP_SIZE), Image.Resampling.NEAREST
        )
        layer.paste(frame.convert("RGBA"), (x0, y0))

    def _latest(self, t: float) -> Verdict | None:
        seen = [v for v in self.verdicts if v.t_event <= t]
        return seen[-1] if seen else None

    def _settled(self, v: Verdict, t: float) -> bool:
        return t - v.t_event >= max(v.latency_ms / 1000.0, MIN_ANALYSE_S)

    def _checklist(self, d: ImageDraw.ImageDraw, t: float) -> None:
        top = 130
        n = len(self.panels)
        found = sum(1 for v in self.verdicts if v.verdict and self._settled(v, t))
        d.rounded_rectangle((14, top, 250, top + 52 + 28 * n), 12, fill=PANEL)
        d.text((28, top + 8), "DEFECTS FOUND", font=self.hud.f_tiny, fill=MUTED)
        d.text(
            (190, top + 6), str(found), font=self.hud.f_mid, fill=RED if found else TEXT
        )
        for k in range(n):
            y = top + 34 + 28 * k
            v = next((q for q in self.verdicts if q.station == k), None)
            done = v is not None and v.t_event <= t and self._settled(v, t)
            label, color = ("pending", GRAY)
            if v is not None and done:
                label, color = (
                    ("no verdict", GRAY)
                    if v.verdict is None
                    else ("DEFECT", RED)
                    if v.verdict
                    else ("clear", GREEN)
                )
            d.ellipse((28, y + 3, 44, y + 19), outline=color, width=3)
            if done and color != GRAY:
                d.ellipse((32, y + 7, 40, y + 15), fill=color)
            d.text(
                (54, y), f"Panel {self.panels[k] + 1}", font=self.hud.f_small, fill=TEXT
            )
            d.text((140, y), label, font=self.hud.f_small, fill=color)

    def _ticker(self, d: ImageDraw.ImageDraw, t: float) -> None:
        x0, x1 = 250, self.width - PIP_SIZE - 36
        y0, y1 = self.height - 14 - 74, self.height - 14
        v = self._latest(t)
        if v is None:
            return
        d.rounded_rectangle((x0, y0, x1, y1), 12, fill=PANEL)
        name = SHORT_NAMES.get(v.model, v.model)
        elapsed_ms = (t - v.t_event) * 1000.0
        if not self._settled(v, t):
            frac = min(elapsed_ms / max(v.latency_ms, 1.0), 1.0)
            d.text(
                (x0 + 14, y0 + 8),
                f"PANEL {v.panel_index + 1}  asking {name} ...",
                font=self.hud.f_mid,
                fill=YELLOW,
            )
            d.rounded_rectangle(
                (x0 + 14, y0 + 44, x1 - 14, y0 + 58), 7, fill=(40, 48, 62, 255)
            )
            d.rounded_rectangle(
                (
                    x0 + 14,
                    y0 + 44,
                    x0 + 14 + max(10, int((x1 - x0 - 28) * frac)),
                    y0 + 58,
                ),
                7,
                fill=YELLOW,
            )
            d.text(
                (x0 + 20, y0 + 44),
                f"{min(elapsed_ms, v.latency_ms):.0f} ms",
                font=self.hud.f_tiny,
                fill=(14, 16, 22, 255),
            )
            return
        word, color = (
            ("NO VERDICT", GRAY)
            if v.verdict is None
            else ("DEFECT", RED)
            if v.verdict
            else ("NO DEFECT", GREEN)
        )
        d.text(
            (x0 + 14, y0 + 8),
            f"PANEL {v.panel_index + 1}",
            font=self.hud.f_mid,
            fill=TEXT,
        )
        tw = d.textlength(f"PANEL {v.panel_index + 1}", font=self.hud.f_mid)
        d.text((x0 + 28 + tw, y0 + 8), word, font=self.hud.f_mid, fill=color)
        d.text(
            (x0 + 14, y0 + 40),
            f"{name}  {v.resolution}px  {v.latency_ms:.0f} ms  ${v.cost_usd:.5f}",
            font=self.hud.f_small,
            fill=MUTED,
        )

    def _summary(self, d: ImageDraw.ImageDraw, t: float) -> None:
        if t < self.run.t_end - END_HOLD_S + SUMMARY_DELAY_S or not self.verdicts:
            return
        vs = self.verdicts
        reported = sum(1 for v in vs if v.verdict)
        real = sum(1 for v in vs if v.truth)
        right = sum(1 for v in vs if v.correct)
        lat = float(np.mean([v.latency_ms for v in vs]))
        cost = sum(v.cost_usd for v in vs)
        x0, y0, x1 = 250, 300, self.width - PIP_SIZE - 36
        d.rounded_rectangle((x0, y0, x1, y0 + 126), 14, fill=(14, 18, 26, 235))
        d.text((x0 + 16, y0 + 10), "INSPECTION SUMMARY", font=self.hud.f_mid, fill=TEXT)
        rows = (
            f"Defects reported {reported} | actually present {real}",
            f"Verdicts matching ground truth: {right} of {len(vs)}",
            f"Mean model latency {lat:.0f} ms | total cost ${cost:.5f}",
        )
        for k, line in enumerate(rows):
            d.text((x0 + 16, y0 + 42 + 26 * k), line, font=self.hud.f_small, fill=MUTED)

    def close(self) -> None:
        self.camera.close()
        super().close()


__all__ = [
    "InspectionHud",
    "InspectionMissionScene",
    "InspectionRenderer",
    "Verdict",
]
