from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.report import build_report
from app.harness.store import connect
from tests.unit.test_analysis_summary import build_db


def test_report_smoke(tmp_path: Path) -> None:
    conn, exp_id = build_db()
    out = tmp_path / "sub" / "r.html"
    build_report(conn, exp_id, out, scope="all")
    html = out.read_text(encoding="utf-8")
    for needle in (
        "claims-ledger",
        'id="executive"',
        'id="latency-curves"',
        'id="rho"',
        'id="tiers"',
        'id="events"',
        'id="latency-ccdf"',
        'id="paired"',
        'id="manifest"',
        "abc123",
        "latency mode: injected",
        "budget B",
        "<svg",
        "prefers-color-scheme:dark",
    ):
        assert needle in html
    assert "NaN" not in html
    assert "nan" not in html.lower().replace("financ", "")
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "<script" not in html
    conn.close()


def test_report_empty_experiment_and_missing(tmp_path: Path) -> None:
    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO experiments(id, name, created_at, config_json, config_sha) "
        "VALUES (1, 'empty', 'now', '{}', 'sha')"
    )
    out = build_report(conn, 1, tmp_path / "e.html")
    assert "NaN" not in out.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        build_report(conn, 2, tmp_path / "x.html")
    conn.close()


def test_scoped_copy_keeps_only_the_requested_flights(tmp_path: Path) -> None:
    from app.analysis.report import scoped_copy

    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO experiments(id, name, created_at, config_json, config_sha) VALUES (1,'e','t','{}','s')"
    )
    conn.execute("INSERT INTO arms(id, kind, provider) VALUES ('a','k','p')")
    for sid, event, budget in (
        ("ok", "obstacle_static", 0.5),
        ("ctl", "obstacle_static", -0.2),
        ("clr", "clear", 99.0),
    ):
        conn.execute(
            "INSERT INTO scenarios(id, suite, seed, event_type, tier, budget_s, certified, spec_json) "
            "VALUES (?, 's', 1, ?, 'T', ?, 1, '{}')",
            (sid, event, budget),
        )
        conn.execute(
            "INSERT INTO flights(experiment_id, scenario_id, arm_id, latency_mode, status) "
            "VALUES (1, ?, 'a', 'fixed', 'complete')",
            (sid,),
        )

    def kept(scope: str) -> set[str]:
        mem = scoped_copy(conn, scope)
        return {r[0] for r in mem.execute("SELECT scenario_id FROM flights")}

    assert kept("challenge") == {"ok"}
    assert kept("controls") == {"ctl"}
    assert kept("all") == {"ok", "ctl", "clr"}
    with pytest.raises(ValueError):
        scoped_copy(conn, "nope")
