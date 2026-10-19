"""Camera inspection scene: a wall of panels, some with defects, seen by a drone camera.

Pure-data side (``draw_structure``, ``paint_panel``, ``describe_panel``) is seeded and has no
PyBullet dependency, so labels and textures are reproducible. ``InspectionScene`` places the
painted panels on a vertical wall in a headless PyBullet client and renders RGB frames with
the tiny CPU renderer at any square size (64, 128, 224 px in the resolution study).

Frames: the wall is the plane ``y = WALL_Y_M`` facing the flight line (``y`` near 0); the
drone flies along ``x``. The camera is gimballed: it looks at the centre of the requested
panel from the drone's position, so the vehicle keeps yaw zero as everywhere else.
"""

from __future__ import annotations

import io
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFilter

from app.sim.rng import channel_generator

WALL_Y_M = 1.8
PANEL_W_M = 1.0
PANEL_H_M = 0.8
COL_PITCH_M = 1.5
ROW_PITCH_M = 1.4
ROW0_Z_M = 1.2
FIRST_COL_X_M = 0.0
WALL_MARGIN_M = 6.0
WALL_BELOW_M = 3.0  # the wall continues under the floor so no background shows
TEX_W = 640  # 640 texture pixels per metre of panel
TEX_H = 512
IMAGE_SIZES: tuple[int, ...] = (64, 128, 224)
CAMERA_FOV_DEG = 50.0
CAMERA_NEAR_M = 0.05
CAMERA_FAR_M = 12.0
DIST_RANGE_M = (1.3, 1.8)  # camera-to-wall distance used by ``sample_pose``
LATERAL_JITTER_M = 0.25
HEIGHT_JITTER_M = 0.2
DEFECT_SIZES_M = (
    0.12,
    0.25,
    0.5,
)  # small / medium / large (crack length or blotch width)
P_BOTH_DEFECTS = 0.25
LIGHT_DIR = (0.2, 1.0, 0.8)
WALL_RGBA = (0.42, 0.43, 0.46, 1.0)

DefectKind = Literal["crack", "corrosion"]
MarkKind = Literal["shadow", "sticker", "red_sticker", "smudge", "scratch", "rivets"]

CRACK_RGB = (95, 18, 18)
CORROSION_RGB = (172, 82, 30)
CORROSION_CORE_RGB = (122, 52, 20)
SMUDGE_RGB = (88, 82, 74)
SCRATCH_RGB = (58, 58, 60)
STICKER_RGBS = ((40, 90, 200), (40, 160, 90), (235, 200, 40))
RED_STICKER_RGB = (205, 32, 32)


@dataclass(frozen=True, slots=True)
class DefectSpec:
    """One defect; ``u``/``v`` are the start point in panel fractions (0 = left/top)."""

    kind: DefectKind
    u: float
    v: float
    size_m: float
    angle: float
    seed: int


@dataclass(frozen=True, slots=True)
class MarkSpec:
    """A harmless mark (shadow, sticker, smudge, scratch, rivets) that is not a defect."""

    kind: MarkKind
    u: float
    v: float
    size_m: float
    angle: float
    seed: int


@dataclass(frozen=True, slots=True)
class PanelSpec:
    index: int
    row: int
    col: int
    center: tuple[float, float, float]
    tone: int
    defects: tuple[DefectSpec, ...]
    marks: tuple[MarkSpec, ...]
    texture_seed: int

    @property
    def has_defect(self) -> bool:
        return bool(self.defects)

    @property
    def label(self) -> str:
        """Ground truth: ``none``, ``crack``, ``corrosion`` or ``both``."""

        kinds = {d.kind for d in self.defects}
        if not kinds:
            return "none"
        return "both" if len(kinds) == 2 else next(iter(kinds))

    @property
    def smallest_defect_m(self) -> float:
        return min((d.size_m for d in self.defects), default=0.0)


@dataclass(frozen=True, slots=True)
class StructureSpec:
    seed: int
    rows: int
    cols: int
    panels: tuple[PanelSpec, ...]

    def labels(self) -> list[bool]:
        return [pn.has_defect for pn in self.panels]

    def panel_at(self, row: int, col: int) -> PanelSpec:
        return self.panels[row * self.cols + col]


# ---------------------------------------------------------------------- drawing specs
def _draw_defect(gen: np.random.Generator, kind: DefectKind, seed: int) -> DefectSpec:
    return DefectSpec(
        kind=kind,
        u=float(gen.uniform(0.2, 0.8)),
        v=float(gen.uniform(0.25, 0.75)),
        size_m=float(gen.choice(DEFECT_SIZES_M)),
        angle=float(gen.uniform(0.0, math.pi)),
        seed=seed,
    )


def _draw_marks(gen: np.random.Generator, n: int, base_seed: int) -> list[MarkSpec]:
    kinds: tuple[MarkKind, ...] = (
        "shadow",
        "sticker",
        "red_sticker",
        "smudge",
        "scratch",
        "rivets",
    )
    size_by_kind = {
        "shadow": (0.3, 0.55),
        "sticker": (0.05, 0.10),
        "red_sticker": (0.05, 0.08),
        "smudge": (0.06, 0.15),
        "scratch": (0.10, 0.30),
        "rivets": (0.6, 0.8),
    }
    out: list[MarkSpec] = []
    for k in range(n):
        kind = kinds[int(gen.integers(len(kinds)))]
        lo, hi = size_by_kind[kind]
        out.append(
            MarkSpec(
                kind=kind,
                u=float(gen.uniform(0.12, 0.88)),
                v=float(gen.uniform(0.12, 0.88)),
                size_m=float(gen.uniform(lo, hi)),
                angle=float(gen.uniform(0.0, math.pi)),
                seed=base_seed + 31 * k,
            )
        )
    return out


def _draw_panel(
    structure_seed: int, row: int, col: int, cols: int, defective: bool
) -> PanelSpec:
    index = row * cols + col
    gen = channel_generator(structure_seed, f"panel{index}")
    defects: list[DefectSpec] = []
    if defective:
        both = gen.random() < P_BOTH_DEFECTS
        first: DefectKind = "crack" if gen.random() < 0.5 else "corrosion"
        defects.append(_draw_defect(gen, first, structure_seed * 1000 + index))
        if both:
            other: DefectKind = "corrosion" if first == "crack" else "crack"
            defects.append(_draw_defect(gen, other, structure_seed * 1000 + index + 1))
    n_marks = int(gen.integers(0, 3)) if defective else int(gen.integers(1, 4))
    marks = _draw_marks(gen, n_marks, structure_seed * 7919 + index)
    return PanelSpec(
        index=index,
        row=row,
        col=col,
        center=(
            FIRST_COL_X_M + col * COL_PITCH_M,
            WALL_Y_M,
            ROW0_Z_M + row * ROW_PITCH_M,
        ),
        tone=int(gen.integers(150, 188)),
        defects=tuple(defects),
        marks=tuple(marks),
        texture_seed=structure_seed * 104729 + index,
    )


def draw_structure(
    seed: int,
    rows: int,
    cols: int,
    *,
    n_defective: int | None = None,
    forced: dict[int, bool] | None = None,
) -> StructureSpec:
    """A wall of ``rows`` x ``cols`` panels (deterministic in ``seed``).

    ``n_defective`` panels (default: half, rounded down) carry defects, chosen by the seed;
    ``forced`` maps panel index to a required has-defect value and overrides the choice.
    """

    n = rows * cols
    k = n // 2 if n_defective is None else n_defective
    order = channel_generator(seed, "defective").permutation(n)
    flags = [False] * n
    for idx in order[:k]:
        flags[int(idx)] = True
    for idx, value in (forced or {}).items():
        flags[idx] = value
    panels = tuple(
        _draw_panel(seed, r, c, cols, flags[r * cols + c])
        for r in range(rows)
        for c in range(cols)
    )
    return StructureSpec(seed=seed, rows=rows, cols=cols, panels=panels)


# ---------------------------------------------------------------------------- painting
def _px(m: float) -> float:
    return m * TEX_W  # metres to texture pixels (same scale on both axes)


def _base_texture(panel: PanelSpec) -> Image.Image:
    gen = channel_generator(panel.texture_seed, "base")
    low = gen.normal(0.0, 1.0, (8, 10))
    low_img = Image.fromarray(
        np.clip(128 + low * 40, 0, 255).astype(np.uint8), "L"
    ).resize((TEX_W, TEX_H), Image.Resampling.BICUBIC)
    shade = (np.asarray(low_img, dtype=np.float32) - 128.0) * 0.18
    grain = gen.normal(0.0, 6.0, (TEX_H, TEX_W)).astype(np.float32)
    gray = np.clip(panel.tone + shade + grain, 0, 255)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    rgb[..., 2] = np.clip(rgb[..., 2].astype(np.int16) + 4, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    ImageDraw.Draw(img).rectangle(
        (0, 0, TEX_W - 1, TEX_H - 1), outline=(70, 70, 74), width=6
    )
    return img


def _blob_points(
    gen: np.random.Generator, cx: float, cy: float, radius: float, n: int = 9
) -> list[tuple[float, float]]:
    angles = np.sort(gen.uniform(0.0, 2 * math.pi, n))
    radii = radius * gen.uniform(0.55, 1.0, n)
    return [
        (cx + float(r * math.cos(a)), cy + float(r * math.sin(a)))
        for a, r in zip(angles, radii, strict=True)
    ]


def _layer() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    layer = Image.new("RGBA", (TEX_W, TEX_H), (0, 0, 0, 0))
    return layer, ImageDraw.Draw(layer, "RGBA")


def _paint_crack(img: Image.Image, d: DefectSpec) -> Image.Image:
    gen = channel_generator(d.seed, "crack")
    layer, draw = _layer()
    length = _px(d.size_m)
    width = max(3, round(length * 0.045))
    x, y, ang = d.u * TEX_W, d.v * TEX_H, d.angle
    pts = [(x, y)]
    n = 12
    for _ in range(n):
        ang += float(gen.normal(0.0, 0.45))
        x += math.cos(ang) * length / n
        y += math.sin(ang) * length / n
        pts.append((x, y))
    draw.line(pts, fill=(*CRACK_RGB, 255), width=width, joint="curve")
    bx, by = pts[n // 2]
    bang = ang + float(gen.choice([-0.9, 0.9]))
    branch = [
        (
            bx + math.cos(bang) * length * 0.25 * t,
            by + math.sin(bang) * length * 0.25 * t,
        )
        for t in (0.0, 0.5, 1.0)
    ]
    draw.line(branch, fill=(*CRACK_RGB, 255), width=max(2, width - 2), joint="curve")
    layer = layer.filter(ImageFilter.GaussianBlur(0.8))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def _paint_corrosion(img: Image.Image, d: DefectSpec) -> Image.Image:
    gen = channel_generator(d.seed, "corrosion")
    layer, draw = _layer()
    radius = _px(d.size_m) / 2.0
    cx, cy = d.u * TEX_W, d.v * TEX_H
    for k in range(4):
        ox, oy = gen.normal(0.0, radius * 0.35, 2)
        pts = _blob_points(
            gen, cx + float(ox), cy + float(oy), radius * (1.0 - 0.12 * k)
        )
        draw.polygon(pts, fill=(*CORROSION_RGB, 215))
    draw.polygon(
        _blob_points(gen, cx, cy, radius * 0.45), fill=(*CORROSION_CORE_RGB, 200)
    )
    for _ in range(int(radius)):  # rust speckle around the edge
        a = float(gen.uniform(0, 2 * math.pi))
        r = radius * float(gen.uniform(0.9, 1.5))
        sx, sy = cx + r * math.cos(a), cy + r * math.sin(a)
        draw.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=(*CORROSION_RGB, 180))
    layer = layer.filter(ImageFilter.GaussianBlur(1.5))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def _paint_mark(img: Image.Image, m: MarkSpec) -> Image.Image:
    gen = channel_generator(m.seed, "mark")
    layer, draw = _layer()
    cx, cy, size = m.u * TEX_W, m.v * TEX_H, _px(m.size_m)
    blur = 0.8
    if m.kind == "shadow":
        draw.polygon(_blob_points(gen, cx, cy, size / 2.0), fill=(20, 20, 24, 85))
        blur = size / 6.0
    elif m.kind in ("sticker", "red_sticker"):
        rgb = (
            RED_STICKER_RGB
            if m.kind == "red_sticker"
            else STICKER_RGBS[int(gen.integers(len(STICKER_RGBS)))]
        )
        r = size / 2.0
        box = (cx - r, cy - r, cx + r, cy + r)
        if m.kind == "red_sticker" or gen.random() < 0.5:
            draw.ellipse(box, fill=(*rgb, 255), outline=(245, 245, 245, 255), width=3)
        else:
            draw.rectangle(box, fill=(*rgb, 255), outline=(245, 245, 245, 255), width=3)
    elif m.kind == "smudge":
        draw.polygon(_blob_points(gen, cx, cy, size / 2.0), fill=(*SMUDGE_RGB, 120))
        blur = 3.0
    elif m.kind == "scratch":
        dx, dy = math.cos(m.angle) * size / 2.0, math.sin(m.angle) * size / 2.0
        draw.line(
            (cx - dx, cy - dy, cx + dx, cy + dy), fill=(*SCRATCH_RGB, 230), width=2
        )
    else:  # rivets: a row of small dark dots
        for k in range(6):
            sx = cx - size / 2.0 + k * size / 5.0
            draw.ellipse((sx - 5, cy - 5, sx + 5, cy + 5), fill=(60, 60, 64, 255))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def paint_panel(panel: PanelSpec) -> np.ndarray:
    """Panel texture ``(TEX_H, TEX_W, 3)`` uint8: base concrete, marks, then defects."""

    img = _base_texture(panel)
    for m in panel.marks:
        img = _paint_mark(img, m)
    for d in panel.defects:
        img = _paint_crack(img, d) if d.kind == "crack" else _paint_corrosion(img, d)
    return np.asarray(img, dtype=np.uint8)


def encode_png(rgb: np.ndarray) -> bytes:
    """Lossless PNG bytes of an RGB frame (what is sent to the vision models)."""

    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _size_word(size_m: float) -> str:
    return f"{size_m * 100:.0f} cm"


def describe_panel(panel: PanelSpec) -> str:
    """Neutral text description of a panel's visible features (the text-only control).

    Describes colour, shape and size only; it never uses the words defect, crack or
    corrosion, so the model has to make the same judgement it makes from the picture.
    """

    where = {-1: "left", 0: "centre", 1: "right"}

    def place(u: float, v: float) -> str:
        h = where[-1 if u < 0.38 else 1 if u > 0.62 else 0]
        vv = "upper" if v < 0.38 else "lower" if v > 0.62 else "middle"
        return f"{vv} {h}"

    parts: list[str] = []
    for d in panel.defects:
        if d.kind == "crack":
            parts.append(
                f"a thin jagged dark red line about {_size_word(d.size_m)} long "
                f"with one short branch, {place(d.u, d.v)}"
            )
        else:
            parts.append(
                f"an irregular orange-brown rough blotch about {_size_word(d.size_m)} "
                f"wide with rust speckles, {place(d.u, d.v)}"
            )
    texts = {
        "shadow": "a soft grey shadow",
        "sticker": "a flat coloured sticker with a white border",
        "red_sticker": "a round red sticker with a white border",
        "smudge": "a dull grey-brown smudge",
        "scratch": "a thin straight dark grey scratch",
        "rivets": "a row of small dark grey rivets",
    }
    for m in panel.marks:
        parts.append(f"{texts[m.kind]} about {_size_word(m.size_m)}, {place(m.u, m.v)}")
    body = "; ".join(parts) if parts else "no marks of any kind"
    return (
        f"Concrete wall panel {panel.index + 1} (grey, 1.0 m by 0.8 m). "
        f"Visible features: {body}."
    )


# ------------------------------------------------------------------------------ scene
_QUAD_OBJ = """v -0.5 0 -0.5
v 0.5 0 -0.5
v 0.5 0 0.5
v -0.5 0 0.5
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 -1 0
f 1/1/1 2/2/1 3/3/1 4/4/1
f 1/1/1 4/4/1 3/3/1 2/2/1
"""  # a unit quad in the x-z plane with UVs, double sided


def sample_pose(
    structure: StructureSpec, panel_index: int
) -> tuple[float, float, float]:
    """Seeded drone position in front of a panel (distance and offsets vary per panel)."""

    panel = structure.panels[panel_index]
    gen = channel_generator(structure.seed, f"pose{panel_index}")
    dist = float(gen.uniform(*DIST_RANGE_M))
    return (
        panel.center[0] + float(gen.uniform(-LATERAL_JITTER_M, LATERAL_JITTER_M)),
        WALL_Y_M - dist,
        panel.center[2] + float(gen.uniform(-HEIGHT_JITTER_M, HEIGHT_JITTER_M)),
    )


class InspectionScene:
    """A wall with painted panels in a PyBullet client; renders the drone camera view.

    With ``client=None`` it owns a private DIRECT client (and closes it); pass an existing
    client id to add the wall to a larger scene such as the 3D flight render.
    """

    def __init__(self, structure: StructureSpec, client: int | None = None) -> None:
        self.structure = structure
        self._owns_client = client is None
        self._cid = p.connect(p.DIRECT) if client is None else client
        self._tmp = Path(tempfile.mkdtemp(prefix="inspection_"))
        self._body_ids: list[int] = []
        self._build()

    def _build(self) -> None:
        st = self.structure
        cid = self._cid
        xs = [pn.center[0] for pn in st.panels]
        zs = [pn.center[2] for pn in st.panels]
        x0, x1 = min(xs) - WALL_MARGIN_M, max(xs) + WALL_MARGIN_M
        z1 = max(zs) + PANEL_H_M / 2 + 0.6
        wall = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[(x1 - x0) / 2, 0.1, (z1 + WALL_BELOW_M) / 2],
            rgbaColor=list(WALL_RGBA),
            physicsClientId=cid,
        )
        self._body_ids.append(
            int(
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=wall,
                    basePosition=[
                        (x0 + x1) / 2,
                        WALL_Y_M + 0.1,
                        (z1 - WALL_BELOW_M) / 2,
                    ],
                    physicsClientId=cid,
                )
            )
        )
        obj = self._tmp / "quad.obj"
        obj.write_text(_QUAD_OBJ)
        for pn in st.panels:
            png = self._tmp / f"panel{pn.index}.png"
            Image.fromarray(paint_panel(pn), "RGB").save(png)
            tex = p.loadTexture(str(png), physicsClientId=cid)
            vis = p.createVisualShape(
                p.GEOM_MESH,
                fileName=str(obj),
                meshScale=[PANEL_W_M, 1.0, PANEL_H_M],
                rgbaColor=[1, 1, 1, 1],
                physicsClientId=cid,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[pn.center[0], WALL_Y_M - 0.004, pn.center[2]],
                physicsClientId=cid,
            )
            p.changeVisualShape(body, -1, textureUniqueId=tex, physicsClientId=cid)
            self._body_ids.append(int(body))

    def render_camera(
        self, pose: tuple[float, float, float], panel_index: int, size: int
    ) -> np.ndarray:
        """RGB ``(size, size, 3)`` uint8 view from ``pose`` aimed at the panel centre."""

        target = self.structure.panels[panel_index].center
        view = p.computeViewMatrix(
            list(pose), list(target), [0, 0, 1], physicsClientId=self._cid
        )
        proj = p.computeProjectionMatrixFOV(
            CAMERA_FOV_DEG, 1.0, CAMERA_NEAR_M, CAMERA_FAR_M, physicsClientId=self._cid
        )
        _, _, rgb, _, _ = p.getCameraImage(
            size,
            size,
            viewMatrix=view,
            projectionMatrix=proj,
            lightDirection=list(LIGHT_DIR),
            shadow=0,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._cid,
        )
        return np.reshape(np.array(rgb, dtype=np.uint8), (size, size, 4))[:, :, :3]

    def close(self) -> None:
        for body in self._body_ids:
            p.removeBody(body, physicsClientId=self._cid)
        self._body_ids = []
        if self._owns_client:
            p.disconnect(self._cid)
        shutil.rmtree(self._tmp, ignore_errors=True)


__all__ = [
    "CAMERA_FOV_DEG",
    "COL_PITCH_M",
    "DEFECT_SIZES_M",
    "IMAGE_SIZES",
    "PANEL_H_M",
    "PANEL_W_M",
    "ROW0_Z_M",
    "ROW_PITCH_M",
    "WALL_Y_M",
    "DefectSpec",
    "InspectionScene",
    "MarkSpec",
    "PanelSpec",
    "StructureSpec",
    "describe_panel",
    "draw_structure",
    "encode_png",
    "paint_panel",
    "sample_pose",
]
