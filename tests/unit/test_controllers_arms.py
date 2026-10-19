from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.controllers import arms as arms_mod
from app.controllers.arms import (
    ArmParams,
    FsmArm,
    HoverArm,
    OracleErrorArm,
    RandomArm,
    ReflexArm,
    WaypointArm,
    build_classical_arm,
    load_params,
)
from app.domain.actions import Action
from app.domain.observation import MAX_RANGE_M, Observation, TruthState


def _obs(**kw: object) -> Observation:
    base: dict[str, object] = {
        "tick": 0,
        "t_sim": 1.0,
        "pos": (5.0, 0.0, 1.0),
        "vel": (2.0, 0.0, 0.0),
        "goal_dx": 9.0,
        "home_dist": 5.0,
        "rays": (MAX_RANGE_M,) * 16,
        "sectors": (MAX_RANGE_M,) * 8,
        "nearest_range": MAX_RANGE_M,
        "nearest_bearing_deg": 0.0,
        "closing_speed": 0.0,
        "ttc": 99.0,
        "battery_pct": 80.0,
        "gps_ok": True,
        "last_action": 0,
        "fwd_range": MAX_RANGE_M,
    }
    base.update(kw)
    return Observation(**base)  # type: ignore[arg-type]


def _threat(fwd: float, closing: float = 2.0) -> Observation:
    rays = [MAX_RANGE_M] * 16
    rays[0] = fwd
    return _obs(
        rays=tuple(rays), fwd_range=fwd, closing_speed=closing, nearest_range=fwd
    )


def test_waypoint_always_flies_forward_until_the_goal() -> None:
    arm = WaypointArm()
    assert arm.decide(_threat(0.5)).action == Action.FORWARD  # ignores obstacles
    assert arm.decide(_obs(goal_dx=0.1)).action == Action.HOVER


def test_hover_and_random_floors() -> None:
    assert HoverArm().decide(_obs()).action == Action.HOVER
    seen = {RandomArm(3).decide(_obs()).action for _ in range(1)}
    assert len(seen) == 1
    many = RandomArm(5)
    assert len({many.decide(_obs()).action for _ in range(80)}) > 5


@pytest.mark.parametrize("cls", [FsmArm, ReflexArm])
def test_rule_arms_cruise_when_clear_and_dodge_when_close(cls: type) -> None:
    assert cls().decide(_obs()).action == Action.FORWARD
    d = cls().decide(_threat(1.2))
    assert d.action in (Action.AVOID_LEFT, Action.AVOID_RIGHT, Action.BRAKE)


@pytest.mark.parametrize("cls", [FsmArm, ReflexArm])
def test_rule_arms_handle_battery_gps_goal(cls: type) -> None:
    assert cls().decide(_obs(battery_pct=4.0)).action == Action.EMERGENCY_LAND
    assert cls().decide(_obs(gps_ok=False)).action == Action.HOVER
    assert cls().decide(_obs(goal_dx=0.0)).action == Action.HOVER


def test_arm_avoids_toward_the_freer_side() -> None:
    left_blocked = [MAX_RANGE_M] * 16
    left_blocked[0], left_blocked[3], left_blocked[4] = 2.0, 2.0, 2.0  # ray 4 = +y
    obs = _obs(rays=tuple(left_blocked), fwd_range=2.0, closing_speed=2.0)
    assert FsmArm(ArmParams()).decide(obs).action == Action.AVOID_RIGHT


def test_arm_commit_holds_the_evasive_action() -> None:
    arm = ReflexArm(ArmParams(commit_s=1.0))
    first = arm.decide(_threat(1.2, 2.0))
    later = arm.decide(
        _obs(t_sim=1.4)
    )  # threat gone but still inside the commit window
    assert later.action == first.action
    assert arm.decide(_obs(t_sim=2.5)).action == Action.FORWARD


def test_decisions_report_tiny_compute_latency() -> None:
    d = FsmArm().decide(_threat(2.0))
    assert 0.0 <= d.latency_ms < 5.0 and d.valid


def test_tuned_parameters_are_loaded_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tuned.json"
    path.write_text(
        json.dumps({"arms": {"fsm": {"params": {"margin_m": 0.9, "commit_s": 0.7}}}})
    )
    monkeypatch.setattr(arms_mod, "TUNED_PATH", path)
    assert load_params("fsm").margin_m == 0.9 and load_params("fsm").commit_s == 0.7
    assert load_params("reflex_ttc") == ArmParams()  # not in the file: defaults


def test_oracle_error_share_of_time_matches_rate_regardless_of_decision_rate() -> None:
    truth = TruthState(0, (5, 0, 1), (2, 0, 0), None, None, 0.0, float("inf"))
    for decisions_per_s in (5, 48):
        wrong = total = 0
        for seed in range(20):
            arm = OracleErrorArm(error_rate=0.3, seed=seed)
            arm.set_truth(truth)
            for i in range(decisions_per_s * 30):
                a = arm.decide(_obs(t_sim=i / decisions_per_s)).action
                wrong += a != Action.FORWARD
                total += 1
        assert abs(wrong / total - 0.3) < 0.06


def test_oracle_error_arm_errors_in_episodes_and_never_lands() -> None:
    truth = TruthState(0, (5, 0, 1), (2, 0, 0), None, None, 0.0, float("inf"))
    arm = OracleErrorArm(error_rate=1.0, seed=2)
    arm.set_truth(truth)
    first = arm.decide(_obs(t_sim=1.0)).action
    assert first != Action.FORWARD  # wrong: the only acceptable action is FORWARD
    assert arm.decide(_obs(t_sim=1.3)).action == first  # held for the episode
    seen = set()
    for i in range(200):
        a = OracleErrorArm(1.0, seed=i)
        a.set_truth(truth)
        seen.add(a.decide(_obs(t_sim=1.0)).action)
    assert Action.EMERGENCY_LAND not in seen and Action.RETURN_HOME not in seen


def test_factory() -> None:
    assert build_classical_arm("fsm").provider_id == "fsm"
    assert (
        build_classical_arm("oracle_err", config={"error_rate": 0.3}).provider_id
        == "oracle_err0.3"
    )
    with pytest.raises(ValueError):
        build_classical_arm("nope")
