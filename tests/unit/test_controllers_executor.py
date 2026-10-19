from __future__ import annotations

import numpy as np

from app.controllers.executor import ActionExecutor
from app.domain.actions import Action

DT = 1 / 48
HOME = (0.0, 0.0, 1.0)


def _run(ex: ActionExecutor, seconds: float, t0: float = 0.0) -> float:
    pos = ex.state.ref_pos.copy()
    t = t0
    for _ in range(int(seconds / DT)):
        tp, _ = ex.step(pos, t)
        pos = tp  # perfect tracking: the drone sits on the carrot
        t += DT
    return t


def _ex(v: float = 2.0) -> ActionExecutor:
    return ActionExecutor(np.array([0.0, 0.0, 1.0]), DT, v, HOME)


def test_forward_reaches_cruise_speed_under_accel_limit() -> None:
    ex = _ex()
    _run(ex, 2.0)
    assert abs(ex.state.ref_vel[0] - 2.0) < 1e-6
    assert ex.state.ref_pos[0] > 2.0


def test_brake_stops_faster_than_hover() -> None:
    a, b = _ex(), _ex()
    _run(a, 2.0)
    _run(b, 2.0)
    a.apply(Action.BRAKE, 2.0)
    b.apply(Action.HOVER, 2.0)
    xa, xb = a.state.ref_pos[0], b.state.ref_pos[0]
    _run(a, 2.0, 2.0)
    _run(b, 2.0, 2.0)
    stop_brake = a.state.ref_pos[0] - xa
    stop_hover = b.state.ref_pos[0] - xb
    assert 0.0 < stop_brake < stop_hover
    assert abs(a.state.ref_vel[0]) < 1e-9


def test_avoid_shifts_lateral_and_slows_then_recovers_speed() -> None:
    ex = _ex()
    _run(ex, 1.5)
    ex.apply(Action.AVOID_LEFT, 1.5)
    t = _run(ex, 0.5, 1.5)
    assert ex.state.ref_vel[0] < 2.0  # slowed
    _run(ex, 2.5, t)
    assert abs(ex.state.ref_pos[1] - 1.2) < 0.05
    assert abs(ex.state.ref_vel[0] - 2.0) < 1e-6  # back to cruise after the TTL


def test_left_right_displacement_is_bounded() -> None:
    ex = _ex()
    for i in range(10):
        ex.apply(Action.LEFT, i * 1.0)
        _run(ex, 1.0, i * 1.0)
    assert ex.state.ref_pos[1] <= 3.5 + 1e-6


def test_side_commit_blocks_immediate_reversal() -> None:
    ex = _ex()
    ex.apply(Action.LEFT, 0.0)
    ex.apply(Action.RIGHT, 0.2)  # inside the 0.5 s commit window
    assert ex.state.lat_target is not None and ex.state.lat_target > 0


def test_vertical_bounds() -> None:
    ex = _ex()
    for i in range(10):
        ex.apply(Action.DESCEND, float(i))
        _run(ex, 1.0, float(i))
    assert ex.state.ref_pos[2] >= 0.4 - 1e-6


def test_emergency_land_is_irreversible() -> None:
    ex = _ex()
    ex.apply(Action.EMERGENCY_LAND, 0.0)
    ex.apply(Action.FORWARD, 0.1)
    _run(ex, 0.5)
    assert ex.state.landing and ex.state.x_mode == "stop"
    assert ex.state.ref_vel[2] < 0


def test_return_home_heads_back() -> None:
    ex = _ex()
    _run(ex, 3.0)
    x_before = ex.state.ref_pos[0]
    ex.apply(Action.RETURN_HOME, 3.0)
    _run(ex, 4.0, 3.0)
    assert ex.state.ref_pos[0] < x_before


def test_carrot_never_runs_away_from_a_stalled_drone() -> None:
    ex = _ex()
    pos = np.array([0.0, 0.0, 1.0])
    for i in range(480):
        tp, _ = ex.step(pos, i * DT)  # drone never moves
    assert np.all(np.abs(tp - pos) <= np.array([0.8, 0.8, 0.4]) + 1e-9)


def test_hold_point_is_not_dragged_down_by_a_sagging_drone() -> None:
    ex = _ex()
    ex.apply(Action.HOVER, 0.0)
    sag = np.array([0.0, 0.0, 0.2])  # drone has sagged 0.8 m below the hold altitude
    for i in range(240):
        tp, _ = ex.step(sag, i * DT)
    assert abs(tp[2] - 1.0) < 1e-6  # setpoint still at the original altitude


def test_moving_carrot_lead_is_still_bounded() -> None:
    ex = _ex()
    stalled = np.array([0.0, 0.0, 1.0])
    for i in range(480):
        tp, _ = ex.step(stalled, i * DT)
    assert tp[0] - stalled[0] <= 0.8 + 1e-9
