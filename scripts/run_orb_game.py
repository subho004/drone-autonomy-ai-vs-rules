"""Orb Dodge, a game: homing orbs chase a drone that must dodge them all and land.

Part A (no network): a rule-based dodge pilot against orbs steered by proportional
navigation or by pure pursuit, sweeping the drone's decision latency and the orb brain's.
Part B (live models, a few cents): a strong model flies the drone while a small model steers
the orbs, against the same scenarios with rule-based orb brains.

Every flight's trajectory is saved to ``data/orb_game/`` for rendering; the summary goes to
``assets/sims/orbs/results.json``. Constants at the top instead of CLI args; set
``ORB_PARTS`` (e.g. ``A`` or ``AB``) in the environment to run a subset.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.controllers.arms import build_classical_arm
from app.core.logging import configure_logging, get_logger
from app.domain.decision import DecisionProvider
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.harness.orb_control import OrbController, PursuitSteerer
from app.scenarios.sims import draw_orbs, draw_orbs_random, draw_swarm
from app.sim.world import DroneWorld

OUT_DIR = ROOT_DIR / "data" / "orb_game"
RESULTS = ROOT_DIR / "assets" / "sims" / "orbs" / "results.json"
BASE_SEED = 6000
N_SCENARIOS = 24
N_LIVE = 8
SWARM_BASE_SEED = 7000
DRONE_LATENCIES_MS = (0, 300, 600)
ORB_LATENCIES_MS = (0, 300, 600, 1000)
LIVE_DRONES = {
    "jev": {"provider": "decisions_endpoint", "model": "typesafe/jev-1.13"},
    "mercury": {
        "provider": "decisions_endpoint",
        "model": "inception/mercury-decide:free",
    },
}
LIVE_ORB_MODEL = "gpt-4o-mini"
LIVE_WORKERS = 4
TOURNAMENT_SEED = 9000
N_TOURNAMENT = 12
TOURNAMENT_DODGERS = ("pilot", "navreflex", "mercury", "jev")
TOURNAMENT_CHASERS = ("pn", "pursuit", "llm", "llm-gemini")
TOURNAMENT_WORKERS = 6
SWEEP_WORKERS = 6

logger = get_logger("run_orb_game")
_WORLD: DroneWorld | None = None


def _world() -> DroneWorld:
    global _WORLD
    if _WORLD is None:
        _WORLD = DroneWorld()
    return _WORLD


def _save(name: str, spec_seed: int, result: Any, meta: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj = result.trajectory or {}
    np.savez_compressed(OUT_DIR / f"{name}.npz", **traj)
    rows = [
        {
            k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in r.items()
        }
        for r in result.decisions
    ]
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps({"seed": spec_seed, **meta, "decisions": rows}, default=str)
    )


def fly_case(job: dict[str, Any]) -> dict[str, Any]:
    """One flight. ``job``: seed, orb ('pn'|'pursuit'|'llm'), drone ('pilot'|'jev'|...),
    drone_ms / orb_ms (None = measured), name."""

    seed, orb, drone = int(job["seed"]), str(job["orb"]), str(job["drone"])
    swarm = bool(job.get("swarm", False))
    draw = (
        draw_orbs_random
        if job.get("scenario") == "random"
        else draw_swarm
        if swarm
        else draw_orbs
    )
    spec = draw(seed, ext=orb != "pn")
    provider: DecisionProvider
    if drone == "pilot":
        provider = build_classical_arm("orb_pilot")  # type: ignore[assignment]
    elif drone in (
        "navpilot",
        "navreflex",
    ):  # navigation only; dodging is left to the reflex layer
        provider = build_classical_arm("orb_pilot", config={"dodge_range_m": 0.0})  # type: ignore[assignment]
    else:
        from app.providers.factory import build_provider

        provider = build_provider(LIVE_DRONES[drone])
    drone_ms = job.get("drone_ms")
    lat = LatencyPolicy(
        LatencyMode.FIXED if drone_ms is not None else LatencyMode.MEASURED,
        fixed_ms=float(drone_ms or 0.0),
    )
    controller: OrbController | None = None
    steerer: Any = None
    if orb == "pursuit":
        orb_ms = job.get("orb_ms")
        controller = OrbController(
            PursuitSteerer(),
            LatencyPolicy(
                LatencyMode.FIXED if orb_ms else LatencyMode.LOCKSTEP,
                fixed_ms=float(orb_ms or 0.0),
            ),
        )
    elif orb in ("llm", "llm-gemini"):
        from app.providers.orb_llm import OrbLLMSteerer

        steerer = (
            OrbLLMSteerer("gemini-3.5-flash-lite", flavor="gemini")
            if orb == "llm-gemini"
            else OrbLLMSteerer(LIVE_ORB_MODEL)
        )
        controller = OrbController(steerer, LatencyPolicy(LatencyMode.MEASURED))
    cfg = FlightConfig(
        lat,
        shield=bool(job.get("shield", False)),
        record_trajectory=True,
        label_decisions=False,
        orbs=controller,
    )
    t0 = time.perf_counter()
    result = run_flight(_world(), spec, provider, cfg)
    m = result.metrics
    out = {
        "name": job["name"],
        "seed": seed,
        "orb": orb,
        "swarm": swarm,
        "scenario": job.get("scenario", "swarm" if swarm else "standard"),
        "shield": bool(job.get("shield", False)),
        "drone": drone,
        "drone_ms": drone_ms,
        "orb_ms": job.get("orb_ms"),
        "outcome": m["outcome"],
        "success": int(m["success"]),
        "collision": int(m["collision"]),
        "t_end": m["t_end"],
        "drone_latency_ms": m.get("lat_mean_ms"),
        "orb_latency_ms": (
            float(np.mean(controller.latencies_ms))
            if controller and controller.latencies_ms
            else None
        ),
        "orb_decisions": controller.decisions if controller else 0,
        "cost_usd": float(m.get("cost_usd") or 0.0)
        + (controller.cost_usd if controller else 0.0),
        "wall_s": time.perf_counter() - t0,
    }
    _save(job["name"], seed, result, out)
    if steerer is not None:
        steerer.close()
    return out


def certified_seeds(swarm: bool = False) -> list[int]:
    """Seeds the rule-based pilot completes against proportional-navigation orbs at 0 ms."""

    seeds: list[int] = []
    base = SWARM_BASE_SEED if swarm else BASE_SEED
    draw = draw_swarm if swarm else draw_orbs
    s = base
    world = _world()
    cfg = FlightConfig(
        LatencyPolicy(LatencyMode.MEASURED),
        shield=False,
        record_trajectory=False,
        label_decisions=False,
    )
    while len(seeds) < N_SCENARIOS and s < base + (80 if swarm else 40) * N_SCENARIOS:
        if run_flight(world, draw(s), build_classical_arm("orb_pilot"), cfg).metrics[
            "success"
        ]:
            seeds.append(s)
        s += 1
    return seeds


def run_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    import multiprocessing as mp

    out: list[dict[str, Any]] = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        for k, res in enumerate(pool.map(fly_case, jobs)):
            out.append(res)
            if (k + 1) % 20 == 0:
                logger.info("%d/%d flights done", k + 1, len(jobs))
    return out


def main() -> None:
    configure_logging()
    parts = os.environ.get("ORB_PARTS", "ABCS")
    t0 = time.perf_counter()
    seeds = certified_seeds()
    logger.info(
        "%d certified scenarios (seeds %d..%d)", len(seeds), seeds[0], seeds[-1]
    )
    prior: dict[str, Any] = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    prior["seeds"] = seeds
    if "A" in parts:
        jobs = [
            {
                "name": f"A_pn_d{dl}_s{s}",
                "seed": s,
                "orb": "pn",
                "drone": "pilot",
                "drone_ms": dl,
            }
            for s in seeds
            for dl in DRONE_LATENCIES_MS
        ] + [
            {
                "name": f"A_pursuit_d{dl}_o{ol}_s{s}",
                "seed": s,
                "orb": "pursuit",
                "drone": "pilot",
                "drone_ms": dl,
                "orb_ms": ol,
            }
            for s in seeds
            for dl in DRONE_LATENCIES_MS
            for ol in ORB_LATENCIES_MS
        ]
        jobs += [
            {
                "name": f"A2_{orb}_d{dl}_s{s}",
                "seed": s,
                "orb": orb,
                "drone": "navpilot",
                "drone_ms": dl,
                "shield": True,
                "orb_ms": 0,
            }
            for s in seeds
            for dl in DRONE_LATENCIES_MS
            for orb in ("pn", "pursuit")
        ]
        prior["A"] = run_jobs(jobs, SWEEP_WORKERS)
    if "B" in parts:
        live = seeds[:N_LIVE]
        jobs = [
            {
                "name": f"B_{drone}_{orb}_s{s}",
                "seed": s,
                "orb": orb,
                "drone": drone,
                "shield": True,
            }
            for s in live
            for drone in LIVE_DRONES
            for orb in ("pn", "pursuit", "llm")
        ]
        prior["B"] = run_jobs(jobs, LIVE_WORKERS)
    if "C" in parts:  # role swap: a small model chases, the rule-based pilot dodges
        c_jobs: list[dict[str, Any]] = [
            {"name": f"C_pilot_llm_s{s}", "seed": s, "orb": "llm", "drone": "pilot"}
            for s in seeds[:N_LIVE]
        ]
        prior["C"] = run_jobs(c_jobs, LIVE_WORKERS)
    if "S" in parts:  # swarms: waves of two orbs, six in all
        swarm_seeds = certified_seeds(swarm=True)
        prior["swarm_seeds"] = swarm_seeds
        s_jobs: list[dict[str, Any]] = [
            {
                "name": f"S_{orb}_d{dl}_s{s}",
                "seed": s,
                "orb": orb,
                "drone": "pilot",
                "drone_ms": dl,
                "swarm": True,
            }
            for s in swarm_seeds
            for orb in ("pn", "pursuit")
            for dl in (0, 300)
        ] + [
            {
                "name": f"SB_{drone}_{orb}_s{s}",
                "seed": s,
                "orb": orb,
                "drone": drone,
                "shield": True,
                "swarm": True,
            }
            for s in swarm_seeds[:N_LIVE]
            for drone in LIVE_DRONES
            for orb in ("pn", "pursuit")
        ]
        prior["S"] = run_jobs(s_jobs, LIVE_WORKERS)
    if "T" in parts:  # randomised round robin: every dodger against every chaser
        t_jobs: list[dict[str, Any]] = [
            {
                "name": f"T_{drone}_{orb}_s{s}",
                "seed": s,
                "orb": orb,
                "drone": drone,
                "scenario": "random",
                "shield": drone != "pilot",
            }
            for s in range(TOURNAMENT_SEED, TOURNAMENT_SEED + N_TOURNAMENT)
            for drone in TOURNAMENT_DODGERS
            for orb in TOURNAMENT_CHASERS
        ]
        prior["T"] = run_jobs(t_jobs, TOURNAMENT_WORKERS)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(prior, indent=2))
    logger.info("done in %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
