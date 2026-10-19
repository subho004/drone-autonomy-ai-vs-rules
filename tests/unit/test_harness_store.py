from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.harness.store import MAX_ATTEMPTS, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "t.db")
    exp = s.upsert_experiment("e", {"a": 1}, "sha")
    s.upsert_scenarios(
        [
            {
                "id": f"s{i}",
                "suite": "x",
                "seed": i,
                "event_type": "clear",
                "tier": "C",
                "budget_s": 99.0,
                "certified": 1,
                "spec_json": "{}",
            }
            for i in range(3)
        ]
    )
    s.upsert_arm("arm", "k", "classical", "", {})
    s.enqueue_flights(
        [
            {
                "experiment_id": exp,
                "scenario_id": f"s{i}",
                "arm_id": "arm",
                "latency_mode": "fixed",
                "latency_ms": 0.0,
                "shield": 0,
                "cadence": "sequential",
                "rep": 0,
                "pool": "cpu",
                "priority": 3 - i,
            }
            for i in range(3)
        ]
    )
    return s


def test_enqueue_is_idempotent(store: Store) -> None:
    exp = store.conn.execute("SELECT id FROM experiments").fetchone()["id"]
    again = store.enqueue_flights(
        [
            {
                "experiment_id": exp,
                "scenario_id": "s0",
                "arm_id": "arm",
                "latency_mode": "fixed",
                "latency_ms": 0.0,
                "shield": 0,
                "cadence": "sequential",
                "rep": 0,
                "pool": "cpu",
                "priority": 9,
            }
        ]
    )
    assert again == 0
    assert store.progress(exp) == {"queued": 3}


def test_claims_follow_priority_and_never_duplicate(store: Store) -> None:
    first = store.claim_flight("cpu", "w1")
    second = store.claim_flight("cpu", "w2")
    assert first is not None and second is not None and first["id"] != second["id"]
    assert first["scenario_id"] == "s0"  # highest priority first
    assert store.claim_flight("api", "w3") is None  # other pool


def test_complete_writes_decisions_and_metrics_atomically(store: Store) -> None:
    row = store.claim_flight("cpu", "w")
    assert row is not None
    decision = {
        "seq": 0,
        "obs_tick": 0,
        "apply_tick": 1,
        "t_obs": 0.0,
        "t_apply": 0.02,
        "action": 0,
        "applied_action": 0,
        "source": "model",
        "valid": 1,
        "latency_ms": 1.0,
        "latency_ticks": 1,
        "confidence": None,
        "probs_json": None,
        "acceptable": 1,
        "tokens_in": None,
        "tokens_out": None,
        "cost_usd": 0.0,
        "from_cache": 0,
        "raw": None,
    }
    store.complete_flight(
        row["id"],
        {"outcome": "success", "success": 1, "collision": 0, "error": None},
        [decision],
    )
    done = store.conn.execute(
        "SELECT status, outcome FROM flights WHERE id=?", (row["id"],)
    ).fetchone()
    assert (done["status"], done["outcome"]) == ("complete", "success")
    assert (
        store.conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"] == 1
    )


def test_expired_lease_is_reclaimed_then_fails_after_max_attempts(store: Store) -> None:
    store.conn.execute("DELETE FROM flights WHERE scenario_id != 's0'")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        claimed = store.claim_flight("cpu", f"w{attempt}")
        assert claimed is not None and claimed["attempts"] == attempt
        store.conn.execute(
            "UPDATE flights SET lease_until=? WHERE id=?",
            (time.time() - 1, claimed["id"]),
        )
    assert store.claim_flight("cpu", "late") is None
    row = store.conn.execute("SELECT status, error FROM flights").fetchone()
    assert row["status"] == "failed" and "lease" in row["error"]


def test_heartbeat_extends_a_live_lease(store: Store) -> None:
    claimed = store.claim_flight("cpu", "w")
    assert claimed is not None
    store.conn.execute(
        "UPDATE flights SET lease_until=? WHERE id=?", (time.time() + 1, claimed["id"])
    )
    store.heartbeat(claimed["id"])
    lease = store.conn.execute(
        "SELECT lease_until FROM flights WHERE id=?", (claimed["id"],)
    ).fetchone()
    assert lease["lease_until"] > time.time() + 60


def test_scenario_and_arm_lookup(store: Store) -> None:
    assert store.scenario_spec("s1") == {}
    assert store.arm_config("arm")["provider"] == "classical"
    with pytest.raises(KeyError):
        store.scenario_spec("nope")
