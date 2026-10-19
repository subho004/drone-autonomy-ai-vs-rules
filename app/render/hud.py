"""Heads-up display drawn over the 3D frame (Pillow): model, action, decision latency,
mission checklist, event toasts, a top-down mini-map and 3D-anchored labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.domain.actions import Action

FONT_REGULAR = ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf")
FONT_BOLD = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
PANEL = (14, 18, 26, 200)
TEXT = (240, 244, 250, 255)
MUTED = (165, 175, 190, 255)
GREEN = (74, 214, 120, 255)
ORANGE = (255, 170, 60, 255)
RED = (255, 90, 90, 255)
BLUE = (90, 170, 255, 255)
YELLOW = (255, 214, 70, 255)
GRAY = (130, 140, 155, 255)
TOAST_S = 2.4
TRAIL_S = 5.0

ACTION_COLORS: dict[Action, tuple[int, int, int, int]] = {
    Action.FORWARD: BLUE,
    Action.LEFT: (80, 210, 200, 255),
    Action.RIGHT: (80, 210, 200, 255),
    Action.ASCEND: (170, 140, 255, 255),
    Action.DESCEND: (170, 140, 255, 255),
    Action.BRAKE: ORANGE,
    Action.HOVER: GRAY,
    Action.AVOID_LEFT: ORANGE,
    Action.AVOID_RIGHT: ORANGE,
    Action.RETURN_HOME: BLUE,
    Action.EMERGENCY_LAND: RED,
}


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
class Toast:
    t: float
    text: str
    color: tuple[int, int, int, int]


@dataclass(slots=True)
class HudState:
    """Everything the HUD needs for one frame."""

    t: float
    model: str
    subtitle: str
    action: Action
    shield: bool
    deciding_progress: float | None  # 0..1 while a request is in flight, else None
    deciding_ms: float | None
    station_status: list[str]  # "done" | "missed" | "active" | "todo"
    pad_status: str
    toasts: list[Toast]
    trail: list[tuple[float, float]]  # world x, y of the drone, oldest first
    drone_xy: tuple[float, float]
    obstacles_xy: list[tuple[float, float]]
    stations_xy: list[tuple[float, float]]
    pad_xy: tuple[float, float]
    outcome: str | None = None
    title: str = "MISSION DEMO"
    checklist_title: str = "MISSION"
    checklist: list[tuple[str, str]] | None = (
        None  # (label, status) rows; None = stations + pad
    )
    gauge: tuple[str, float, str, tuple[float, float]] | None = (
        None  # label, 0..1, text, band
    )
    map_rects: (
        list[tuple[float, float, float, float, tuple[int, int, int, int]]] | None
    ) = None
    map_dots: list[tuple[float, float, tuple[int, int, int, int]]] | None = None
    show_pad: bool = True


class Hud:
    def __init__(
        self,
        width: int,
        height: int,
        world_x: tuple[float, float],
        world_y: tuple[float, float],
    ) -> None:
        self.w, self.h = width, height
        self.world_x, self.world_y = world_x, world_y
        self.f_big = _font(FONT_BOLD, 26)
        self.f_mid = _font(FONT_BOLD, 18)
        self.f_small = _font(FONT_REGULAR, 14)
        self.f_tiny = _font(FONT_REGULAR, 12)
        self.f_toast = _font(FONT_BOLD, 30)

    # ----------------------------------------------------------------- frame
    def compose(
        self, frame: Image.Image, st: HudState, labels: list[tuple[float, float, str]]
    ) -> Image.Image:
        layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, "RGBA")
        self._labels(d, labels)
        self._panel_model(d, st)
        self._panel_mission(d, st)
        self._panel_time(d, st)
        self._minimap(d, st)
        self._gauge(d, st)
        self._toasts(d, st)
        return Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")

    # ----------------------------------------------------------------- pieces
    def _labels(
        self, d: ImageDraw.ImageDraw, labels: list[tuple[float, float, str]]
    ) -> None:
        for x, y, text in labels:
            tw = d.textlength(text, font=self.f_small)
            d.rounded_rectangle(
                (x - tw / 2 - 6, y - 11, x + tw / 2 + 6, y + 11), 6, fill=PANEL
            )
            d.text((x - tw / 2, y - 8), text, font=self.f_small, fill=TEXT)

    def _panel_model(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        d.rounded_rectangle((14, 14, 336, 168), 12, fill=PANEL)
        d.text((28, 22), st.title, font=self.f_tiny, fill=MUTED)
        d.text((28, 38), st.model, font=self.f_big, fill=TEXT)
        d.text((28, 70), st.subtitle, font=self.f_small, fill=MUTED)
        color = ACTION_COLORS.get(st.action, GRAY)
        label = st.action.name.replace("_", " ")
        tw = d.textlength(label, font=self.f_mid)
        d.rounded_rectangle((28, 98, 28 + tw + 28, 128), 15, fill=color)
        d.text((42, 102), label, font=self.f_mid, fill=(12, 14, 20, 255))
        if st.shield:
            d.rounded_rectangle(
                (28 + tw + 40, 98, 28 + tw + 130, 128), 15, outline=GREEN, width=2
            )
            d.text((28 + tw + 52, 103), "SHIELD", font=self.f_small, fill=GREEN)
        bar_w = 294
        d.rounded_rectangle((28, 142, 28 + bar_w, 156), 7, fill=(40, 48, 62, 255))
        if st.deciding_progress is not None and (st.deciding_ms or 0) > 20:
            fill = int(bar_w * min(max(st.deciding_progress, 0.0), 1.0))
            d.rounded_rectangle((28, 142, 28 + max(fill, 10), 156), 7, fill=YELLOW)
            txt = f"model deciding... {st.deciding_ms or 0:.0f} ms"
        else:
            txt = (
                "reflex pilot: no model latency"
                if st.deciding_ms is not None
                else "applying decision"
            )
        d.text(
            (32, 143),
            txt,
            font=self.f_tiny,
            fill=(14, 16, 22, 255) if st.deciding_progress else MUTED,
        )

    def _panel_mission(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        if st.checklist is not None:
            rows = st.checklist
        else:
            rows = [
                (f"Scan station {i + 1}", s) for i, s in enumerate(st.station_status)
            ]
            rows.append(("Land on pad", st.pad_status))
        n = len(rows)
        top = self.h - 30 - 30 * n
        d.rounded_rectangle((14, top - 38, 232, self.h - 14), 12, fill=PANEL)
        d.text((28, top - 28), st.checklist_title, font=self.f_tiny, fill=MUTED)
        for i, (name, status) in enumerate(rows):
            y = top + i * 30
            self._status_icon(d, 36, y + 8, status)
            color = TEXT if status in ("done", "active") else MUTED
            d.text((54, y), name, font=self.f_small, fill=color)

    def _status_icon(
        self, d: ImageDraw.ImageDraw, cx: int, cy: int, status: str
    ) -> None:
        if status == "done":
            d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=GREEN)
            d.line(
                (cx - 4, cy, cx - 1, cy + 4, cx + 5, cy - 4),
                fill=(10, 30, 16, 255),
                width=3,
            )
        elif status == "missed":
            d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=ORANGE)
            d.line((cx - 4, cy - 4, cx + 4, cy + 4), fill=(40, 22, 0, 255), width=3)
            d.line((cx - 4, cy + 4, cx + 4, cy - 4), fill=(40, 22, 0, 255), width=3)
        elif status == "active":
            d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), outline=YELLOW, width=3)
            d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=YELLOW)
        else:
            d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), outline=GRAY, width=2)

    def _panel_time(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        d.rounded_rectangle((self.w - 170, 14, self.w - 14, 56), 12, fill=PANEL)
        d.text((self.w - 156, 20), "SIM TIME", font=self.f_tiny, fill=MUTED)
        d.text((self.w - 156, 33), f"{st.t:5.1f} s", font=self.f_mid, fill=TEXT)

    def _minimap(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        mw, mh = 290, 118
        x0, y0 = self.w - mw - 14, self.h - mh - 14
        d.rounded_rectangle((x0 - 8, y0 - 26, x0 + mw + 8, y0 + mh + 8), 12, fill=PANEL)
        d.text((x0, y0 - 20), "TOP VIEW", font=self.f_tiny, fill=MUTED)

        def m(x: float, y: float) -> tuple[float, float]:
            fx = (x - self.world_x[0]) / (self.world_x[1] - self.world_x[0])
            fy = (y - self.world_y[0]) / (self.world_y[1] - self.world_y[0])
            return x0 + fx * mw, y0 + (1 - fy) * mh

        d.rectangle((x0, y0, x0 + mw, y0 + mh), fill=(30, 38, 50, 255))
        for rx0, rx1, ry0, ry1, col in st.map_rects or []:
            ax, ay = m(rx0, ry1)
            bx, by = m(rx1, ry0)
            d.rectangle((ax, ay, bx, by), fill=col)
        if st.show_pad:
            px, py = m(*st.pad_xy)
            d.ellipse(
                (px - 9, py - 9, px + 9, py + 9), fill=(70, 74, 86, 255), outline=TEXT
            )
        for gx, gy, col in st.map_dots or []:
            cx, cy = m(gx, gy)
            d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=col)
        for (sx, sy), status in zip(st.stations_xy, st.station_status, strict=True):
            cx, cy = m(sx, sy)
            col = {"done": GREEN, "missed": ORANGE, "active": YELLOW}.get(status, BLUE)
            d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=col, width=3)
        if len(st.trail) > 1:
            d.line([m(*q) for q in st.trail], fill=(255, 255, 255, 170), width=2)
        for ox, oy in st.obstacles_xy:
            cx, cy = m(ox, oy)
            d.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=RED)
        dx, dy = m(*st.drone_xy)
        d.polygon([(dx + 8, dy), (dx - 5, dy - 5), (dx - 5, dy + 5)], fill=YELLOW)

    def _gauge(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        """A horizontal gauge with a shaded target band (e.g. follow distance)."""

        if st.gauge is None:
            return
        label, value, text, band = st.gauge
        x0, y0, gw = 250, self.h - 52, 330
        d.rounded_rectangle((x0 - 12, y0 - 26, x0 + gw + 12, y0 + 34), 12, fill=PANEL)
        d.text((x0, y0 - 20), label, font=self.f_tiny, fill=MUTED)
        d.rounded_rectangle((x0, y0 + 2, x0 + gw, y0 + 16), 7, fill=(40, 48, 62, 255))
        d.rounded_rectangle(
            (x0 + band[0] * gw, y0 + 2, x0 + band[1] * gw, y0 + 16),
            7,
            fill=(60, 120, 80, 255),
        )
        inside = band[0] <= value <= band[1]
        cx = x0 + min(max(value, 0.0), 1.0) * gw
        d.ellipse((cx - 8, y0 - 1, cx + 8, y0 + 19), fill=GREEN if inside else RED)
        d.text((x0, y0 + 20), text, font=self.f_tiny, fill=TEXT)

    def _toasts(self, d: ImageDraw.ImageDraw, st: HudState) -> None:
        live = [t for t in st.toasts if 0 <= st.t - t.t < TOAST_S]
        for i, toast in enumerate(live[-2:]):
            age = st.t - toast.t
            alpha = int(255 * min(1.0, (TOAST_S - age) / 0.6, age / 0.15 + 0.2))
            tw = d.textlength(toast.text, font=self.f_toast)
            y = 70 + i * 52
            x = (self.w - tw) / 2
            d.rounded_rectangle(
                (x - 18, y - 6, x + tw + 18, y + 40),
                14,
                fill=(14, 18, 26, int(alpha * 0.8)),
            )
            d.text(
                (x, y), toast.text, font=self.f_toast, fill=(*toast.color[:3], alpha)
            )
        if st.outcome:
            self._banner(d, st.outcome)

    def _banner(self, d: ImageDraw.ImageDraw, outcome: str) -> None:
        text, color = {
            "mission_complete": ("MISSION COMPLETE", GREEN),
            "completed": ("TASK COMPLETE", GREEN),
            "gates_done": ("ALL GATES PASSED", GREEN),
            "gates_missed": ("GATES MISSED", RED),
            "survived": ("SURVIVED AND LANDED!", GREEN),
            "left_arena": ("LEFT THE ARENA", ORANGE),
            "hit": ("ORB GOT THE DRONE", RED),
            "low_band": ("OUT OF RANGE TOO LONG", ORANGE),
            "landed_safe": ("SAFE LANDING", GREEN),
            "landed_hazard": ("LANDED IN HAZARD", RED),
            "landed_off_site": ("MISSED THE SITES", RED),
            "lost": ("TARGET LOST", RED),
            "overran": ("OVERRAN THE TARGET", RED),
            "breach": ("ZONE BREACH", RED),
            "partial": ("LANDED ON PAD", GREEN),
            "collision": ("COLLISION", RED),
            "crash": ("CRASH", RED),
            "timeout": ("TIME OUT", ORANGE),
        }.get(outcome, (outcome.upper(), TEXT))
        tw = d.textlength(text, font=self.f_toast)
        x, y = (self.w - tw) / 2, self.h / 2 - 20
        d.rounded_rectangle(
            (x - 26, y - 12, x + tw + 26, y + 46), 16, fill=(14, 18, 26, 225)
        )
        d.text((x, y), text, font=self.f_toast, fill=color)

    def trail_overlay(
        self, d_img: Image.Image, points: list[tuple[float, float, bool]]
    ) -> Image.Image:
        layer = Image.new("RGBA", d_img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, "RGBA")
        pts = [(x, y) for x, y, ok in points if ok]
        if len(pts) > 1:
            d.line(pts, fill=(255, 255, 255, 140), width=3)
        return Image.alpha_composite(d_img.convert("RGBA"), layer).convert("RGB")


def sky_gradient(width: int, height: int) -> np.ndarray:
    """Vertical sky gradient used where the depth buffer shows no geometry."""

    top = np.array([96, 150, 215], dtype=np.float32)
    bottom = np.array([196, 222, 242], dtype=np.float32)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    return np.broadcast_to((top * (1 - t) + bottom * t), (height, width, 3)).astype(
        np.uint8
    )


__all__ = ["Hud", "HudState", "Toast", "sky_gradient"]
