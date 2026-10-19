"""Render the Orb Dodge game: side-by-side videos, GIFs and screenshots under assets/sims/orbs/.

Cases compare saved flights (``data/orb_game``) on the same scenario seed, picked as the
median-ranked scenario of those where the first side survived and the others did not (or, for
the live cases, the first seeds with a mixed result). Constants at the top; set the
environment variable ``RENDER_CASES`` (comma separated names) to render a subset.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.core.logging import configure_logging, get_logger
from app.render.mission_video import FPS, encode_video, make_gif
from app.render.sim_video import SimRenderer, load_orb_run

RESULTS = ROOT_DIR / "assets" / "sims" / "orbs" / "results.json"
OUT_DIR = ROOT_DIR / "assets" / "sims" / "orbs"
FULL = (960, 540)
PANEL_BY_N = {1: (960, 540), 2: (720, 406), 3: (560, 316)}
GIF_WIDTH_BY_N = {1: 720, 2: 640, 3: 900}
GIF_SECONDS = 12.0
END_HOLD_S = 2.5

FOLDERS = {
    "tournament-llm-vs-llm": "tournament",
    "tournament-rule-vs-rule": "tournament",
    "tournament-mixed": "tournament",  # case name -> subfolder of assets/sims/orbs (default: the case name)
    "orb-duel": "duel",
    "rule-vs-models-homing": "rule-vs-models",
    "rule-vs-models-pursuit": "rule-vs-models",
    "swarm-models": "swarm",
    "swarm-drones": "swarm",
}

logger = get_logger("render_orbs")


@dataclass(frozen=True)
class Case:
    name: str
    part: str  # "A" or "B" in results.json
    sides: tuple[dict[str, object], ...]  # filters on result rows, one per panel
    pass_idx: tuple[int, ...]  # panels that must have survived (the rest must not)
    title: str | None = None
    skin: str = "orb"  # "drone" draws the chasers as rival drone models (visual only)


CASES: tuple[Case, ...] = (
    Case(
        "orb-dodge",
        "A",
        (
            {"name_prefix": "A_pn_d0_", "label": "drone 0 ms | homing orbs"},
            {"name_prefix": "A_pn_d300_", "label": "drone 300 ms | homing orbs"},
        ),
        (0,),
    ),
    Case(
        "orb-duel",
        "B",
        (
            {"name_prefix": "B_mercury_llm_", "label": ""},
            {"name_prefix": "B_mercury_pursuit_", "label": ""},
        ),
        (0,),
        title="STRONG MODEL vs SMALL MODEL",
    ),
    Case(
        "chaser-drones",
        "A",
        (
            {"name_prefix": "A_pn_d0_", "label": "drone 0 ms | rival chaser drones"},
            {
                "name_prefix": "A_pn_d300_",
                "label": "drone 300 ms | rival chaser drones",
            },
        ),
        (0,),
        title="RIVAL CHASER DRONES",
        skin="drone",
    ),
    Case(
        "rule-vs-models-homing",
        "A",
        (
            {"name_prefix": "A_pn_d0_", "label": "rule-based pilot, 0 ms"},
            {"name_prefix": "B_mercury_pn_", "part": "B", "label": ""},
            {"name_prefix": "B_jev_pn_", "part": "B", "label": ""},
        ),
        (0,),
        title="RULES vs MERCURY vs JEV",
    ),
    Case(
        "rule-vs-models-pursuit",
        "A",
        (
            {
                "name_prefix": "A_pursuit_d0_o0_",
                "label": "rule-based pilot, 0 ms | pursuit orbs",
            },
            {"name_prefix": "B_mercury_pursuit_", "part": "B", "label": ""},
            {"name_prefix": "B_jev_pursuit_", "part": "B", "label": ""},
        ),
        (0,),
        title="RULES vs MERCURY vs JEV",
    ),
    Case(
        "roles",
        "A",
        (
            {"name_prefix": "A_pursuit_d0_o0_", "label": "rule dodger vs rule chaser"},
            {"name_prefix": "C_pilot_llm_", "part": "C", "label": ""},
            {"name_prefix": "B_mercury_llm_", "part": "B", "label": ""},
        ),
        (0, 1),
        title="WHO CHASES, WHO DODGES",
    ),
    Case(
        "swarm",
        "S",
        (
            {"name_prefix": "S_pn_d0_", "label": "rule pilot 0 ms | orb swarm"},
            {"name_prefix": "S_pn_d300_", "label": "rule pilot 300 ms | orb swarm"},
        ),
        (0,),
        title="ORB SWARM",
    ),
    Case(
        "swarm-models",
        "S",
        (
            {"name_prefix": "S_pn_d0_", "label": "rule pilot 0 ms | orb swarm"},
            {"name_prefix": "SB_mercury_pn_", "label": ""},
            {"name_prefix": "SB_jev_pn_", "label": ""},
        ),
        (0,),
        title="SWARM: RULES vs MERCURY vs JEV",
    ),
    Case(
        "swarm-drones",
        "S",
        (
            {"name_prefix": "S_pn_d0_", "label": "rule pilot 0 ms | rival drone swarm"},
            {
                "name_prefix": "S_pn_d300_",
                "label": "rule pilot 300 ms | rival drone swarm",
            },
        ),
        (0,),
        title="RIVAL DRONE SWARM",
        skin="drone",
    ),
    Case(
        "tournament-llm-vs-llm",
        "T",
        (
            {"name_prefix": "T_mercury_llm_", "label": ""},
            {"name_prefix": "T_mercury_llm-gemini_", "label": ""},
            {"name_prefix": "T_jev_llm_", "label": ""},
        ),
        (0,),
        title="LLM vs LLM",
    ),
    Case(
        "tournament-rule-vs-rule",
        "T",
        (
            {"name_prefix": "T_pilot_pn_", "label": ""},
            {"name_prefix": "T_pilot_pursuit_", "label": ""},
            {"name_prefix": "T_navreflex_pursuit_", "label": ""},
        ),
        (0,),
        title="RULE vs RULE",
    ),
    Case(
        "tournament-mixed",
        "T",
        (
            {"name_prefix": "T_pilot_llm_", "label": ""},
            {"name_prefix": "T_mercury_pursuit_", "label": ""},
            {"name_prefix": "T_jev_pn_", "label": ""},
        ),
        (0,),
        title="MIXED MATCHUPS",
    ),
)


def _rows(
    data: dict[str, list[dict[str, object]]], case: Case
) -> list[list[dict[str, object]]]:
    return [
        sorted(
            (
                r
                for r in data[str(side.get("part", case.part))]
                if str(r["name"]).startswith(str(side["name_prefix"]))
            ),
            key=lambda r: int(r["seed"]),  # type: ignore[call-overload]
        )
        for side in case.sides
    ]


def pick(data: dict[str, list[dict[str, object]]], case: Case) -> list[str]:
    per_side = _rows(data, case)
    by_seed = [{int(r["seed"]): r for r in rows} for rows in per_side]  # type: ignore[call-overload]
    shared = sorted(set.intersection(*(set(d) for d in by_seed)))

    def ok(seed: int, i: int) -> bool:
        return bool(by_seed[i][seed]["success"])

    exact = [
        s
        for s in shared
        if all(ok(s, i) == (i in case.pass_idx) for i in range(len(by_seed)))
    ]
    pool = (
        exact or [s for s in shared if all(ok(s, i) for i in case.pass_idx)] or shared
    )
    seed = pool[len(pool) // 2]
    logger.info(
        "%s: %d shared, %d contrast; using seed %d",
        case.name,
        len(shared),
        len(exact),
        seed,
    )
    return [str(d[seed]["name"]) for d in by_seed]


def render_case(data: dict[str, list[dict[str, object]]], case: Case) -> None:
    names = pick(data, case)
    out = OUT_DIR / FOLDERS.get(case.name, case.name)
    out.mkdir(parents=True, exist_ok=True)
    n = len(names)
    panel = PANEL_BY_N[n]
    sims = [
        load_orb_run(nm, ROOT_DIR, str(side["label"]))
        for nm, side in zip(names, case.sides, strict=True)
    ]
    renderers = [SimRenderer(s, *FULL, title=case.title, skin=case.skin) for s in sims]
    total = max(s.run.t_end for s in sims) + END_HOLD_S
    shots = {
        (k, f"{case.name}-side{k + 1}-{tag}"): round(frac * s.run.t_end * FPS)
        for k, s in enumerate(sims)
        for tag, frac in (("early", 0.3), ("mid", 0.6), ("end", 0.995))
    }

    def frames() -> Iterator[tuple[float, Image.Image]]:
        for f in range(int(total * FPS)):
            canvas = Image.new("RGB", (panel[0] * n, panel[1]))
            for k, (r, s) in enumerate(zip(renderers, sims, strict=True)):
                img = r.frame_at(min(f / FPS, s.run.t_end))
                for (side_k, name), idx in shots.items():
                    if side_k == k and idx == f:
                        img.save(out / f"{name}.jpg", quality=88)
                canvas.paste(
                    img.resize(panel, Image.Resampling.LANCZOS), (panel[0] * k, 0)
                )
            yield f / FPS, canvas

    mp4 = out / f"{case.name}.mp4"
    encode_video(frames(), mp4, size=(panel[0] * n, panel[1]))
    for r in renderers:
        r.close()
    make_gif(
        mp4,
        out / f"{case.name}.gif",
        width=GIF_WIDTH_BY_N[n],
        duration_s=GIF_SECONDS,
    )
    meta = [
        {
            "flight": nm,
            "label": s.latency_label,
            "outcome": s.run.outcome,
            "success": s.run.success,
        }
        for nm, s in zip(names, sims, strict=True)
    ]
    (out / f"{case.name}.json").write_text(json.dumps(meta, indent=2))
    logger.info("%s: %s", case.name, [m["outcome"] for m in meta])


def main() -> None:
    configure_logging()
    t0 = time.perf_counter()
    wanted = set(filter(None, os.environ.get("RENDER_CASES", "").split(",")))
    data = json.loads(RESULTS.read_text())
    for case in CASES:
        if wanted and case.name not in wanted:
            continue
        render_case(data, case)
    logger.info("done in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
