from __future__ import annotations

import pytest

from app.scenarios.spec import MissionSpec, StationSpec
from app.sim.mission import MissionTracker, StationPhase

DT = 1 / 48


def _tracker() -> MissionTracker:
    return MissionTracker(
        MissionSpec(
            stations=[StationSpec(x=4, y=1, z=1.0), StationSpec(x=8, y=-1, z=1.5)],
            pad_x=12,
        )
    )


def _hold(
    t: MissionTracker, pos: tuple[float, float, float], speed: float, secs: float
):
    out = []
    for _ in range(int(secs / DT) + 2):
        out += t.update(pos, speed, DT)
    return out


def test_station_is_scanned_after_a_slow_dwell_in_the_box() -> None:
    t = _tracker()
    assert _hold(t, (4.1, 1.1, 1.05), 0.2, 1.2) == [("scanned", 0)]
    assert t.index == 1 and t.scanned == [True, False]


def test_fast_fly_through_does_not_scan() -> None:
    t = _tracker()
    assert _hold(t, (4.0, 1.0, 1.0), 1.5, 3.0) == []
    assert not t.scanned[0]


def test_dwell_resets_when_leaving_the_box_or_speeding_up() -> None:
    t = _tracker()
    _hold(t, (4.0, 1.0, 1.0), 0.2, 0.6)
    t.update((4.0, 2.0, 1.0), 0.2, DT)  # drifted out laterally
    assert t.dwell_s == 0.0
    _hold(t, (4.0, 1.0, 1.0), 0.2, 0.6)
    assert not t.scanned[0]


def test_passing_a_station_marks_it_missed_and_moves_on() -> None:
    t = _tracker()
    assert t.update((5.2, 1.0, 1.0), 1.0, DT) == [("missed", 0)]
    assert t.index == 1 and t.missed == [True, False]


def test_target_switches_to_the_pad_after_the_last_station() -> None:
    t = _tracker()
    _hold(t, (4.0, 1.0, 1.0), 0.1, 1.2)
    _hold(t, (8.0, -1.0, 1.5), 0.1, 1.2)
    assert t.stations_done and t.target() == (12, 0.0, None)
    assert t.score() == (2, 2)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.5, StationPhase.FAR),
        (2.5, StationPhase.NEAR),
        (3.5, StationPhase.HERE),
        (5.5, StationPhase.PASSED),
    ],
)
def test_phase_buckets(x: float, expected: StationPhase) -> None:
    assert _tracker().phase((x, 1.0, 1.0)) == expected


def test_extras_report_relative_target_and_pad_phase() -> None:
    t = _tracker()
    ex = t.extras((2.0, 0.0, 1.0), moving=True)
    assert ex["st_dx"] == pytest.approx(2.0) and ex["st_dy"] == pytest.approx(1.0)
    assert ex["moving"] == 1.0 and ex["stn_i"] == 1.0 and ex["stn_n"] == 2.0
    t.index = 2
    assert t.phase((11.8, 0.1, 1.0)) == StationPhase.PAD_HERE
    assert t.landed_on_pad((11.9, 0.2, 0.1))
