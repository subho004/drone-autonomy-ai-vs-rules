"""End-to-end: config -> scenarios -> queue -> worker -> metrics + trajectory files."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from app.domain.trajectory import TRAJ_COLUMNS
from app.harness.experiment import prepare_experiment
from app.harness.store import Store
from app.harness.worker import worker_main
from app.scenarios.calibration import cell_key
from app.schemas.experiment import ArmCfg, ExperimentConfig, SuiteCfg
from app.sim.world import DroneWorld


def test_pipeline_runs_and_is_resumable(
    world: DroneWorld, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    import json

    from app.core.config import settings

    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))
    cal = tmp_path / "cal.json"
    cal.write_text(
        json.dumps(
            {
                "d_min0": {
                    cell_key("static", 0.0): 1.14,
                    cell_key("approach", 1.0): 1.6,
                    cell_key("approach", 2.0): 2.1,
                    cell_key("cross", 1.0): 1.07,
                    cell_key("cross", 2.0): 1.2,
                }
            }
        )
    )
    monkeypatch.setattr("app.harness.experiment.CALIBRATION_PATH", cal)
    monkeypatch.setattr(
        "app.harness.experiment.ensure_calibration.__defaults__", (cal,)
    )
    cfg = ExperimentConfig(
        name="t",
        suite=SuiteCfg(name="t", n=4, base_seed=300, n_impossible=1),
        arms=[
            ArmCfg(id="oracle", kind="oracle", measured=False),
            ArmCfg(id="waypoint", kind="waypoint", measured=False),
        ],
        latency_sweep_ms=[0.0, 300.0],
        shield=[0, 1],
    )
    db = tmp_path / "p.db"
    store = Store(db)
    exp = prepare_experiment(store, world, cfg)
    total = store.progress(exp)["queued"]
    assert total == 2 * (4 + 1) * 2 * 2

    assert (
        worker_main(str(db), "cpu", "w0", max_flights=7) == 7
    )  # partial run, then "crash"
    prog = store.progress(exp)
    assert prog["complete"] == 7 and prog["queued"] == total - 7

    prepare_experiment(store, world, cfg)  # re-enqueue: idempotent, nothing duplicated
    assert sum(store.progress(exp).values()) == total
    assert worker_main(str(db), "cpu", "w0") == total - 7
    assert store.progress(exp) == {"complete": total}

    row = store.conn.execute(
        "SELECT trajectory_uri, n_decisions FROM flights WHERE arm_id='oracle' LIMIT 1"
    ).fetchone()
    table = pq.read_table(row["trajectory_uri"])
    assert set(table.column_names) == set(TRAJ_COLUMNS)
    n = store.conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
    assert n > 0
    paired = store.conn.execute(
        "SELECT AVG(collision) AS c FROM flights WHERE arm_id='waypoint' AND shield=0"
    ).fetchone()["c"]
    assert paired > 0.5  # no-avoidance arm collides on most scenarios
