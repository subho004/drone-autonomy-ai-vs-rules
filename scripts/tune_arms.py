"""Tune the rule-based arms (FSM, reflex) with an equal random-search budget.

Protocol (docs/research/05): tune on a held-out 'tune' suite (disjoint seeds from the
evaluation suite), same trial budget for both arms, objective = collision rate plus a
progress penalty averaged over a few latencies. Writes configs/tuned_arms.json.
Constants at the top instead of CLI args.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.controllers.arms import ArmParams, FsmArm, ReflexArm
from app.core.logging import configure_logging, get_logger
from app.harness.experiment import ensure_calibration
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.scenarios.generator import CHALLENGE_EVENTS, ScenarioGenerator
from app.scenarios.spec import ScenarioSpec
from app.sim.world import DroneWorld

TUNE_SUITE = "tune"
TUNE_SEED = 50_000  # disjoint from the evaluation suites (1000+, 500000+)
N_SCENARIOS = 60
LATENCIES_MS = (0.0, 200.0)
N_TRIALS = 60
WORKERS = 6
PENALTY_NO_PROGRESS = 0.3
OUT_PATH = ROOT_DIR / "configs" / "tuned_arms.json"
SPACE: dict[str, tuple[float, float]] = {
    "t_lat_s": (0.05, 0.40),
    "a_brk": (1.5, 4.5),
    "margin_m": (0.30, 1.00),
    "trigger_extra_m": (0.3, 2.5),
    "commit_s": (0.3, 1.2),
    "near_m": (1.5, 4.0),
}

_WORLD: DroneWorld | None = None


def _evaluate(
    args: tuple[str, dict[str, float], list[str]],
) -> tuple[float, float, float]:
    """Return (objective, collision rate, success rate) for one parameter set."""

    global _WORLD
    kind, params, spec_jsons = args
    if _WORLD is None:
        _WORLD = DroneWorld()
    prm = ArmParams(**params)
    cols, succ, n = 0, 0, 0
    for blob in spec_jsons:
        spec = ScenarioSpec.model_validate_json(blob)
        for lat in LATENCIES_MS:
            arm = FsmArm(prm) if kind == "fsm" else ReflexArm(prm)
            cfg = FlightConfig(
                LatencyPolicy(LatencyMode.FIXED, fixed_ms=lat),
                shield=False,
                record_trajectory=False,
                label_decisions=False,
            )
            m = run_flight(_WORLD, spec, arm, cfg).metrics
            cols += m["collision"]
            succ += m["success"]
            n += 1
    col, suc = cols / n, succ / n
    return col + PENALTY_NO_PROGRESS * (1.0 - suc), col, suc


def _sample(rng: np.random.Generator) -> dict[str, float]:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SPACE.items()}


def main() -> None:
    configure_logging()
    logger = get_logger("tune_arms")
    t0 = time.perf_counter()
    world = DroneWorld()
    gen = ScenarioGenerator(ensure_calibration(world))
    specs, _ = gen.generate_certified(
        world, N_SCENARIOS, TUNE_SEED, TUNE_SUITE, events=CHALLENGE_EVENTS
    )
    world.close()
    blobs = [s.model_dump_json() for s in specs]
    rng = np.random.default_rng(TUNE_SEED)
    trials = [asdict(ArmParams())] + [
        _sample(rng) for _ in range(N_TRIALS)
    ]  # default + random
    result: dict[str, dict[str, object]] = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        for kind in ("fsm", "reflex_ttc"):
            scores = pool.map(_evaluate, [(kind, p, blobs) for p in trials])
            best = int(np.argmin([s[0] for s in scores]))
            result[kind] = {
                "params": trials[best],
                "objective": scores[best][0],
                "collision_rate": scores[best][1],
                "success_rate": scores[best][2],
                "default_objective": scores[0][0],
                "default_collision_rate": scores[0][1],
                "default_success_rate": scores[0][2],
                "trials": len(trials),
            }
            logger.info(
                "%s: default obj %.3f (col %.2f suc %.2f) -> tuned %.3f (col %.2f suc %.2f)",
                kind,
                scores[0][0],
                scores[0][1],
                scores[0][2],
                *scores[best],
            )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "protocol": "random search, equal budget, held-out tune suite",
                "tune_suite": TUNE_SUITE,
                "tune_seed": TUNE_SEED,
                "n_scenarios": N_SCENARIOS,
                "latencies_ms": list(LATENCIES_MS),
                "n_trials": N_TRIALS,
                "arms": result,
            },
            indent=2,
        )
    )
    logger.info("wrote %s in %.0fs", OUT_PATH, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
