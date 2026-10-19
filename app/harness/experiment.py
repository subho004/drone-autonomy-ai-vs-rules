"""Experiment preparation: scenarios, arms and the flight grid -> SQLite queue."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from app.controllers.arms import MissionFsmArm, build_classical_arm
from app.core.logging import get_logger
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.harness.store import Store
from app.scenarios.calibration import calibrate, load_calibration, save_calibration
from app.scenarios.generator import ScenarioGenerator
from app.scenarios.missions import generate_missions
from app.scenarios.sims import DRAWERS, draw_canyon, draw_pursuit
from app.scenarios.spec import ScenarioSpec
from app.schemas.experiment import ArmCfg, ExperimentConfig
from app.sim.world import DroneWorld

logger = get_logger(__name__)

# Zero-latency reference pilot that certifies each task family is completable.
FAMILY_PILOTS: dict[str, str] = {
    "follow": "follow_pilot",
    "delivery": "zone_pilot",
    "pursuit": "oracle",
    "orbs": "orb_pilot",
    "landing": "site_best",
    "gates": "gate_pilot",
    "canyon": "fsm",
}
CALIBRATION_PATH = Path("data/calibration.json")
IMPOSSIBLE_SUITE_SUFFIX = "-control"


def ensure_calibration(
    world: DroneWorld, path: Path = CALIBRATION_PATH
) -> dict[str, float]:
    """Load the avoidance table, calibrating (about 5 s) if it does not exist."""

    if path.exists():
        return load_calibration(path)
    table = calibrate(world)
    save_calibration(table, path)
    return table


def _scenario_row(spec: ScenarioSpec) -> dict[str, Any]:
    return {
        "id": spec.scenario_id,
        "suite": spec.suite,
        "seed": spec.seed,
        "event_type": spec.event_type,
        "tier": spec.tier,
        "budget_s": spec.budget_s,
        "certified": int(spec.certified),
        "spec_json": spec.model_dump_json(),
    }


def _mission_certifier(world: DroneWorld) -> Callable[[ScenarioSpec], bool]:
    """A mission is feasible when the rule-based pilot completes it with no latency."""

    def certify(spec: ScenarioSpec) -> bool:
        flight = FlightConfig(
            LatencyPolicy(LatencyMode.MEASURED),
            shield=False,
            record_trajectory=False,
            label_decisions=False,
        )
        return bool(run_flight(world, spec, MissionFsmArm(), flight).metrics["success"])

    return certify


def generate_family(
    world: DroneWorld, kind: str, n: int, base_seed: int, suite: str
) -> list[ScenarioSpec]:
    """``n`` scenarios of a task family that its zero-latency reference pilot completes."""

    arm_kind = FAMILY_PILOTS[kind]
    out: list[ScenarioSpec] = []
    seed = base_seed
    cfg = FlightConfig(
        LatencyPolicy(LatencyMode.MEASURED),
        shield=False,
        record_trajectory=False,
        label_decisions=False,
    )
    while len(out) < n and seed - base_seed < 40 * n:
        spec = family_spec(kind, seed, suite)
        seed += 1
        arm = build_classical_arm(arm_kind)
        if run_flight(world, spec, arm, cfg).metrics["success"]:
            spec.certified = True
            out.append(spec)
    return out


def family_spec(kind: str, seed: int, suite: str) -> ScenarioSpec:
    if kind == "canyon":
        return draw_canyon(seed, suite, block=seed % 2 == 0)
    return DRAWERS[kind](seed, suite)


def ensure_scenarios(
    store: Store, world: DroneWorld, cfg: ExperimentConfig
) -> list[str]:
    """Generate+certify the suite once (cached in SQLite); return ordered scenario ids."""

    suite = cfg.suite
    rows = store.conn.execute(
        "SELECT id FROM scenarios WHERE suite=? AND certified=1 AND budget_s >= 0 ORDER BY seed",
        (suite.name,),
    ).fetchall()
    if len(rows) < suite.n and suite.kind == "mission":
        specs = generate_missions(
            suite.n, suite.base_seed, suite.name, _mission_certifier(world)
        )
        store.upsert_scenarios([_scenario_row(s) for s in specs])
        logger.info("generated %d certified missions", len(specs))
    elif len(rows) < suite.n and suite.kind in FAMILY_PILOTS:
        specs = generate_family(world, suite.kind, suite.n, suite.base_seed, suite.name)
        store.upsert_scenarios([_scenario_row(s) for s in specs])
        logger.info("generated %d certified %s scenarios", len(specs), suite.kind)
        if suite.kind == "pursuit":  # paired straight-line intruders on the same seeds
            paired = [
                draw_pursuit(s.seed, suite.name + "-approach", "approach")
                for s in specs
            ]
            for p_ in paired:
                p_.certified = True
            store.upsert_scenarios([_scenario_row(s) for s in paired])
    elif len(rows) < suite.n:
        gen = ScenarioGenerator(ensure_calibration(world))
        rng = (
            (suite.budget_min_s, suite.budget_max_s)
            if suite.budget_min_s is not None and suite.budget_max_s is not None
            else None
        )
        specs, stats = gen.generate_certified(
            world, suite.n, suite.base_seed, suite.name, mix=suite.mix, budget_range=rng
        )
        store.upsert_scenarios([_scenario_row(s) for s in specs])
        logger.info("generated %d scenarios (rejected %d)", len(specs), stats.rejected)
    if suite.n_impossible:
        _ensure_controls(store, world, cfg)
    rows = store.conn.execute(
        "SELECT id FROM scenarios WHERE suite=? AND certified=1 AND budget_s >= 0 ORDER BY seed "
        "LIMIT ?",
        (suite.name, suite.n),
    ).fetchall()
    return [r["id"] for r in rows]


def _ensure_controls(store: Store, world: DroneWorld, cfg: ExperimentConfig) -> None:
    name = cfg.suite.name + IMPOSSIBLE_SUITE_SUFFIX
    have = store.conn.execute(
        "SELECT COUNT(*) AS n FROM scenarios WHERE suite=?", (name,)
    ).fetchone()
    if int(have["n"]) >= cfg.suite.n_impossible:
        return
    gen = ScenarioGenerator(ensure_calibration(world))
    specs, _ = gen.generate_certified(
        world,
        cfg.suite.n_impossible,
        cfg.suite.base_seed + 500_000,
        name,
        impossible=True,
        events=("obstacle_static", "obstacle_approach"),
    )
    store.upsert_scenarios([_scenario_row(s) for s in specs])


def _latency_cells(arm: ArmCfg, cfg: ExperimentConfig) -> Iterator[tuple[str, float]]:
    if arm.sweep:
        for ms in cfg.latency_sweep_ms:
            yield LatencyMode.FIXED.value, float(ms)
    if arm.sampled:
        for ms in cfg.latency_sampled_ms:
            yield LatencyMode.SAMPLED.value, float(ms)
    if arm.measured:
        yield LatencyMode.MEASURED.value, -1.0


def prepare_experiment(store: Store, world: DroneWorld, cfg: ExperimentConfig) -> int:
    """Create/refresh the experiment and enqueue its full grid. Idempotent."""

    exp_id = store.upsert_experiment(cfg.name, cfg.model_dump(mode="json"), cfg.sha())
    scenario_ids = ensure_scenarios(store, world, cfg)
    control_ids = [
        r["id"]
        for r in store.conn.execute(
            "SELECT id FROM scenarios WHERE suite=? ORDER BY seed",
            (cfg.suite.name + IMPOSSIBLE_SUITE_SUFFIX,),
        )
    ][: cfg.suite.n_impossible]
    flights: list[dict[str, Any]] = []
    n_total = len(scenario_ids)
    for arm in cfg.arms:
        store.upsert_arm(
            arm.id,
            arm.kind or arm.provider,
            arm.provider,
            arm.model,
            {**arm.config, "kind": arm.kind},
        )
        ids = scenario_ids[: arm.subsample] if arm.subsample else scenario_ids
        for idx, sid in enumerate(ids + control_ids):
            for mode, ms in _latency_cells(arm, cfg):
                for shield in cfg.shield:
                    for cadence in cfg.cadence:
                        for rep in range(cfg.reps):
                            flights.append(
                                {
                                    "experiment_id": exp_id,
                                    "scenario_id": sid,
                                    "arm_id": arm.id,
                                    "latency_mode": mode,
                                    "latency_ms": ms,
                                    "shield": shield,
                                    "cadence": cadence,
                                    "rep": rep,
                                    "pool": arm.pool,
                                    "priority": n_total - idx,
                                }
                            )
    added = store.enqueue_flights(flights)
    logger.info(
        "experiment %s: %d flights enqueued (%d new)", cfg.name, len(flights), added
    )
    store.conn.execute("UPDATE experiments SET status='queued' WHERE id=?", (exp_id,))
    return exp_id


def manifest(cfg: ExperimentConfig, extra: dict[str, Any] | None = None) -> str:
    return json.dumps({"config_sha": cfg.sha(), **(extra or {})}, sort_keys=True)


__all__ = [
    "CALIBRATION_PATH",
    "ensure_calibration",
    "ensure_scenarios",
    "manifest",
    "prepare_experiment",
]
