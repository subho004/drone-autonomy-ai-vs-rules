from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator

import pytest

from app.analysis import stats, summary
from app.harness.store import connect

N_SCEN = 40
LATENCIES = (100.0, 400.0)


def _collide(arm: str, ms: float, i: int) -> int:
    if arm == "jev":
        return int(i % 8 == 0) if ms == 100.0 else int(i % 2 == 0)
    return int(i % 4 == 0) if ms == 100.0 else int(i % 4 <= 1)


def build_db() -> tuple[sqlite3.Connection, int]:
    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO experiments(id, name, created_at, config_json, config_sha) "
        "VALUES (1, 'unit', '2026-01-01', ?, 'abc123')",
        (json.dumps({"seed": 7}),),
    )
    for arm, kind in (("jev", "model"), ("fsm", "baseline")):
        conn.execute(
            "INSERT INTO arms(id, kind, provider) VALUES (?, ?, ?)", (arm, kind, arm)
        )
    for i in range(N_SCEN):
        conn.execute(
            "INSERT INTO scenarios(id, suite, seed, event_type, tier, budget_s, certified, spec_json) "
            "VALUES (?, 'test', ?, ?, ?, ?, 1, '{}')",
            (
                f"s{i}",
                i,
                "pop_up" if i % 2 else "gust",
                "T0" if i < 20 else "T1",
                1.0 if i % 2 else 2.0,
            ),
        )
    fid = 0
    for arm in ("jev", "fsm"):
        for ms in LATENCIES:
            for i in range(N_SCEN):
                fid += 1
                col = _collide(arm, ms, i)
                conn.execute(
                    "INSERT INTO flights(id, experiment_id, scenario_id, arm_id, latency_mode, latency_ms, "
                    "status, collision, success, time_to_goal, path_length, reaction_time_s, lat_mean_ms, "
                    "lat_p95_ms, invalid_rate, fallback_rate) VALUES (?, 1, ?, ?, 'injected', ?, "
                    "'complete', ?, ?, 10.0, 20.0, 0.5, ?, ?, 0.0, 0.1)",
                    (fid, f"s{i}", arm, ms, col, 1 - col, ms, ms * 1.5),
                )
                for seq, (lat, cached) in enumerate([(ms, 0), (ms * 2, 0), (999.0, 1)]):
                    conn.execute(
                        "INSERT INTO decisions(flight_id, seq, obs_tick, apply_tick, t_obs, t_apply, "
                        "applied_action, source, valid, latency_ms, latency_ticks, from_cache) "
                        "VALUES (?, ?, 0, 1, 0, 0.1, 0, 'model', 1, ?, 1, ?)",
                        (fid, seq, lat, cached),
                    )
    conn.execute(  # incomplete flight must be ignored
        "INSERT INTO flights(id, experiment_id, scenario_id, arm_id, latency_mode, latency_ms, rep, status, "
        "collision) VALUES (9999, 1, 's0', 'jev', 'injected', 100, 1, 'running', 1)"
    )
    return conn, 1


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    conn, _ = build_db()
    yield conn
    conn.close()


def test_arm_summary(db: sqlite3.Connection) -> None:
    rows = {(r.arm_id, r.latency_ms): r for r in summary.arm_summary(db, 1)}
    assert len(rows) == 4
    r = rows[("fsm", 100.0)]
    assert (r.n, r.collisions) == (N_SCEN, 10)
    assert r.collision_rate == pytest.approx(0.25)
    assert (r.collision_lo, r.collision_hi) == pytest.approx(stats.wilson(10, N_SCEN))
    assert r.success_rate == pytest.approx(0.75)
    assert r.lat_mean_ms == pytest.approx(100.0)
    assert r.lat_p95_ms == pytest.approx(150.0)
    assert r.fallback_rate == pytest.approx(0.1)
    assert rows[("jev", 100.0)].collisions == 5


def test_budget_distribution(db: sqlite3.Connection) -> None:
    b = summary.budget_distribution(db, 1)
    assert (b["min"], b["max"], b["n"]) == (1.0, 2.0, float(N_SCEN))


def test_paired_comparison_counts(db: sqlite3.Connection) -> None:
    res = {c.latency_ms: c for c in summary.paired_comparison(db, 1, "jev", "fsm")}
    c100 = res[100.0]
    # jev collides when i%8==0 (subset of fsm i%4==0): b=0, c=5
    assert (c100.n_pairs, c100.b, c100.c) == (N_SCEN, 0, 5)
    assert c100.p_value == pytest.approx(stats.exact_mcnemar(0, 5))
    assert c100.delta == pytest.approx(-0.125)
    only = summary.paired_comparison(db, 1, "jev", "fsm", latency_ms=400.0)
    assert [c.latency_ms for c in only] == [400.0]


def test_paired_family_holm(db: sqlite3.Connection) -> None:
    fam = summary.paired_family(db, 1, "fsm")
    assert len(fam) == 2
    raw = [c.p_value for c in fam]
    assert [c.p_holm for c in fam] == pytest.approx(stats.holm_correction(raw))
    assert all(c.p_holm is not None and c.p_holm >= c.p_value for c in fam)


def test_outcome_vs_rho(db: sqlite3.Connection) -> None:
    bins = summary.outcome_vs_rho(db, 1)
    assert sum(b.n for b in bins) == 4 * N_SCEN
    # 100 ms / 2 s = 0.05 -> bin [0, .1); 100 ms / 1 s = 0.1 -> bin [.1, .25)
    low = [b for b in bins if b.arm_id == "fsm" and b.rho_lo == 0.0]
    assert sum(b.n for b in low) == N_SCEN // 2
    for b in bins:
        assert 0.0 <= b.lo <= b.rate <= b.hi <= 1.0


def test_breakdowns(db: sqlite3.Connection) -> None:
    tiers = summary.tier_breakdown(db, 1, latency_ms=100.0)
    assert {t.stratum for t in tiers} == {"T0", "T1"}
    assert sum(t.n for t in tiers) == 2 * N_SCEN
    events = summary.event_breakdown(db, 1, latency_ms=100.0)
    assert {e.stratum for e in events} == {"pop_up", "gust"}
    # fsm collides at i%4==0 -> all even i -> all in 'gust'
    fsm_gust = next(e for e in events if e.arm_id == "fsm" and e.stratum == "gust")
    assert fsm_gust.collisions == 10


def test_latency_distribution_excludes_cache(db: sqlite3.Connection) -> None:
    d = summary.latency_distribution(db, 1, "jev")
    assert d.n == 2 * 2 * N_SCEN  # 2 non-cached decisions x 2 cells x scenarios
    assert 999.0 not in d.values_ms
    assert d.summary.quantiles["p50"].value > 0
