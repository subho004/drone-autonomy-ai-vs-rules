from __future__ import annotations

from app.domain.actions import Action
from app.domain.observation import MAX_RANGE_M, Observation, TruthState
from app.oracle.labeller import label_observation


def _obs(**kw: object) -> Observation:
    base: dict[str, object] = {
        "tick": 0,
        "t_sim": 0.0,
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
    }
    base.update(kw)
    return Observation(**base)  # type: ignore[arg-type]


def _truth(ox: float, oy: float = 0.0, ovx: float = 0.0, vx: float = 2.0) -> TruthState:
    return TruthState(
        tick=0,
        pos=(5.0, 0.0, 1.0),
        vel=(vx, 0.0, 0.0),
        obstacle_pos=(ox, oy, 1.0),
        obstacle_vel=(ovx, 0.0, 0.0),
        obstacle_radius=0.3,
        clearance=ox - 5.0,
    )


def test_clear_sky_means_forward() -> None:
    lab = label_observation(_obs(), None, 2.0)
    assert (
        lab.acceptable == frozenset({Action.FORWARD})
        and lab.preferred == Action.FORWARD
    )


def test_far_obstacle_still_allows_forward_but_also_dodges() -> None:
    lab = label_observation(_obs(), _truth(ox=12.0), 2.0)
    assert Action.FORWARD in lab.acceptable and lab.safe_now


def test_close_obstacle_forbids_forward_and_prefers_a_dodge_or_brake() -> None:
    lab = label_observation(_obs(), _truth(ox=6.4), 2.0)
    assert Action.FORWARD not in lab.acceptable
    assert lab.acceptable & {Action.BRAKE, Action.AVOID_LEFT, Action.AVOID_RIGHT}


def test_hopeless_obstacle_is_flagged_unsafe() -> None:
    lab = label_observation(_obs(), _truth(ox=5.5), 2.0)
    assert not lab.safe_now and lab.urgency <= 0.0


def test_stopped_in_front_of_blocker_requires_a_dodge_not_waiting() -> None:
    lab = label_observation(_obs(vel=(0.0, 0.0, 0.0)), _truth(ox=7.0, vx=0.0), 2.0)
    assert lab.acceptable <= {
        Action.AVOID_LEFT,
        Action.AVOID_RIGHT,
        Action.LEFT,
        Action.RIGHT,
    }
    assert Action.HOVER not in lab.acceptable


def test_obstacle_on_the_left_prefers_dodging_right() -> None:
    lab = label_observation(_obs(), _truth(ox=6.6, oy=0.25), 2.0)
    if Action.AVOID_RIGHT in lab.acceptable and Action.AVOID_LEFT in lab.acceptable:
        assert lab.preferred in (Action.AVOID_RIGHT, Action.BRAKE)


def test_battery_critical_overrides_everything() -> None:
    lab = label_observation(_obs(battery_pct=5.0), _truth(ox=6.0), 2.0)
    assert lab.preferred == Action.EMERGENCY_LAND
    assert lab.acceptable == frozenset({Action.EMERGENCY_LAND, Action.RETURN_HOME})


def test_gps_loss_and_goal_reached() -> None:
    assert label_observation(_obs(gps_ok=False), None, 2.0).preferred == Action.HOVER
    assert label_observation(_obs(goal_dx=0.1), None, 2.0).acceptable == frozenset(
        {Action.HOVER, Action.BRAKE}
    )
