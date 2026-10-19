from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.scenarios.calibration import KAPPA_MARGIN, cell_key, closing_speed
from app.scenarios.generator import CHALLENGE_EVENTS, ScenarioGenerator, tier_for
from app.sim.world import DroneWorld

CAL = {
    cell_key("static", 0.0): 1.14,
    cell_key("approach", 1.0): 1.6,
    cell_key("approach", 2.0): 2.1,
    cell_key("cross", 1.0): 1.07,
    cell_key("cross", 2.0): 1.2,
}


def test_tier_boundaries() -> None:
    assert tier_for(-0.1) == "T-1"
    assert tier_for(0.05) == "T5"
    assert tier_for(0.1) == "T4"
    assert tier_for(0.7) == "T2"
    assert tier_for(2.5) == "T0"


def test_draw_is_deterministic_and_budget_matches_distance() -> None:
    gen = ScenarioGenerator(CAL)
    a, b = gen.draw(42, "t"), gen.draw(42, "t")
    assert a.scenario_id == b.scenario_id
    for seed in range(40):
        spec = gen.draw(seed, "t", events=CHALLENGE_EVENTS)
        ob = spec.obstacle
        assert ob is not None
        d0 = CAL[cell_key(ob.kind, ob.speed)] * (1 + KAPPA_MARGIN)
        expected = max(d0 + closing_speed(ob.kind, ob.speed) * spec.budget_s, 0.3)
        assert ob.d_spawn == pytest.approx(expected)
        assert spec.tier == tier_for(spec.budget_s)


def test_stratified_tier_draw_respects_the_tier() -> None:
    gen = ScenarioGenerator(CAL)
    for seed in range(20):
        spec = gen.draw(seed, "t", tier="T3", events=CHALLENGE_EVENTS)
        assert 0.25 <= spec.budget_s < 0.5


def test_impossible_controls_have_negative_budget() -> None:
    spec = ScenarioGenerator(CAL).draw(
        1, "t", impossible=True, events=("obstacle_static",)
    )
    assert spec.budget_s < 0 and spec.tier == "T-1"


def test_generated_scenarios_are_certified_avoidable(world: DroneWorld) -> None:
    gen = ScenarioGenerator(CAL)
    specs, stats = gen.generate_certified(world, 12, 7000, "t")
    assert len(specs) == 12 and all(s.certified for s in specs)
    assert stats.drawn >= 12
    ids = {s.scenario_id for s in specs}
    assert len(ids) == 12


def test_impossible_scenarios_actually_collide(world: DroneWorld) -> None:
    gen = ScenarioGenerator(CAL)
    specs, _ = gen.generate_certified(
        world, 3, 9000, "t-control", impossible=True, events=("obstacle_static",)
    )
    assert all(s.budget_s < 0 for s in specs)


def test_calibration_file_roundtrip(tmp_path: Path) -> None:
    from app.scenarios.calibration import (
        calibration_id,
        load_calibration,
        save_calibration,
    )

    path = tmp_path / "cal.json"
    cid = save_calibration(CAL, path)
    assert load_calibration(path) == CAL and cid == calibration_id(CAL)
    assert json.loads(path.read_text())["version"].startswith("cal_")
