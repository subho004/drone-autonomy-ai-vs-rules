"""Worker process: claim queued flights, fly them, persist results."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from app.controllers.arms import build_classical_arm
from app.core.config import settings
from app.core.logging import get_logger
from app.domain.decision import DecisionProvider
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import (
    DEFAULT_SIGMA,
    LatencyMode,
    LatencyPolicy,
    lognormal_trace,
)
from app.harness.store import Store
from app.providers.budget import BudgetLedger
from app.providers.cache import ResponseCache
from app.providers.factory import build_provider
from app.scenarios.spec import ScenarioSpec
from app.sim.world import DroneWorld

logger = get_logger(__name__)

TRAJ_DIR = "trajectories"


def trajectory_path(data_dir: str, experiment_id: int, flight_id: int) -> Path:
    return Path(data_dir) / TRAJ_DIR / str(experiment_id) / f"{flight_id}.parquet"


def write_trajectory(path: Path, traj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(traj), path, compression="zstd")


class ProviderPool:
    """Lazily builds one provider per arm and reuses it (keeps HTTP/2 connections)."""

    def __init__(self, store: Store, use_cache: bool) -> None:
        self._store = store
        self._providers: dict[str, DecisionProvider] = {}
        self._cache = ResponseCache(store.conn) if use_cache else None
        self._ledger = BudgetLedger(store.conn)

    def get(self, arm_id: str, seed: int) -> DecisionProvider:
        if arm_id not in self._providers:
            cfg = self._store.arm_config(arm_id)
            if cfg["provider"] == "classical":
                kind = str(cfg.get("kind") or arm_id)
                self._providers[arm_id] = build_classical_arm(kind, seed, cfg)  # type: ignore[assignment]
            else:
                self._providers[arm_id] = build_provider(
                    cfg, cache=self._cache, ledger=self._ledger
                )
        return self._providers[arm_id]

    def reset_stateful(self, arm_id: str, seed: int) -> DecisionProvider:
        """Stateful classical arms (FSM commit, RNG) restart each flight."""

        cfg = self._store.arm_config(arm_id)
        if cfg["provider"] == "classical":
            self._providers.pop(arm_id, None)
        return self.get(arm_id, seed)

    def close(self) -> None:
        for p in self._providers.values():
            p.close()


def process_flight(
    store: Store, world: DroneWorld, pool: ProviderPool, row: Any, data_dir: str
) -> None:
    spec = ScenarioSpec.model_validate(store.scenario_spec(row["scenario_id"]))
    provider = pool.reset_stateful(row["arm_id"], spec.seed)
    mode = LatencyMode(row["latency_mode"])
    lat = LatencyPolicy(
        mode,
        fixed_ms=max(float(row["latency_ms"]), 0.0),
        seed=int(row["rep"]),
        trace_ms=(
            lognormal_trace(
                float(row["latency_ms"]),
                float(store.arm_config(row["arm_id"]).get("sigma", DEFAULT_SIGMA)),
            )
            if mode == LatencyMode.SAMPLED
            else []
        ),
    )
    cfg = FlightConfig(
        latency=lat,
        shield=bool(row["shield"]),
        cadence=row["cadence"],
        label_decisions=spec.mission is None,
    )
    result = run_flight(world, spec, provider, cfg)
    metrics = dict(result.metrics)
    if result.trajectory is not None:
        path = trajectory_path(data_dir, int(row["experiment_id"]), int(row["id"]))
        write_trajectory(path, result.trajectory)
        metrics["trajectory_uri"] = str(path)
    metrics["error"] = None
    store.complete_flight(int(row["id"]), metrics, result.decisions)


def worker_main(
    db_path: str, pool_name: str, worker_name: str, max_flights: int | None = None
) -> int:
    """Claim and run flights until the queue is empty; returns flights completed."""

    store = Store(db_path)
    world = DroneWorld()
    pool = ProviderPool(store, use_cache=False)
    done = 0
    try:
        while max_flights is None or done < max_flights:
            row = store.claim_flight(pool_name, worker_name)
            if row is None:
                break
            t0 = time.perf_counter()
            try:
                process_flight(store, world, pool, row, settings.data_dir)
                done += 1
            except Exception as exc:
                logger.exception("flight %s failed", row["id"])
                store.fail_flight(int(row["id"]), f"{type(exc).__name__}: {exc}")
                if "BudgetExceededError" in type(exc).__name__:
                    break
                traceback.clear_frames(exc.__traceback__)
            logger.debug(
                "flight %s in %.2fs (pid %s)",
                row["id"],
                time.perf_counter() - t0,
                os.getpid(),
            )
    finally:
        pool.close()
        world.close()
        store.close()
    return done


__all__ = [
    "ProviderPool",
    "process_flight",
    "trajectory_path",
    "worker_main",
    "write_trajectory",
]
