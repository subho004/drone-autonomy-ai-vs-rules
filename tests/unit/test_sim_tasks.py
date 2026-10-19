from __future__ import annotations

import pytest

from app.scenarios.spec import SiteSpec, TargetSpec, ZoneSpec
from app.sim.tasks import FollowTracker, LandingTracker, ZoneMonitor

DT = 1 / 48


def test_target_paths() -> None:
    line = FollowTracker(TargetSpec(path="line", speed=1.0, start_dx=3.0))
    assert line.target_at(0.0)[:2] == (3.0, 0.0) and line.target_at(10.0)[
        0
    ] == pytest.approx(13.0)
    weave = FollowTracker(TargetSpec(path="weave", weave_amp=1.0, weave_period_s=8.0))
    assert weave.target_at(2.0)[1] == pytest.approx(1.0, abs=1e-6)  # quarter period
    stop = FollowTracker(
        TargetSpec(path="stopgo", speed=1.0, move_s=4.0, stop_s=2.0, start_dx=0)
    )
    assert stop.target_at(4.0)[0] == pytest.approx(4.0) and stop.target_at(6.0)[
        0
    ] == pytest.approx(4.0)
    assert stop.target_vel(5.0) == (0.0, 0.0)


def test_follow_in_band_lost_and_overrun() -> None:
    spec = TargetSpec(path="line", speed=0.0, start_dx=3.0, lost_after_s=1.0)
    tr = FollowTracker(spec)
    for i in range(48):
        assert tr.update((0.0, 0.0, 1.0), i * DT, DT) == []
    assert tr.metrics()["in_band_frac"] == pytest.approx(1.0) and not tr.lost
    far = FollowTracker(spec)
    fired = []
    for i in range(120):
        fired += far.update(
            (-5.0, 0.0, 1.0), i * DT, DT
        )  # target 8 m ahead: out of band
    assert ("lost", 0) in fired and far.lost
    over = FollowTracker(TargetSpec(path="line", speed=0.0, start_dx=0.1))
    assert over.update((0.0, 0.0, 1.0), 0.0, DT) == [("overran", 0)]


def test_follow_extras_report_relative_motion() -> None:
    tr = FollowTracker(TargetSpec(path="line", speed=1.0, start_dx=3.0))
    ex = tr.extras((0.0, 0.0, 1.0), (0.4, 0.0, 0.0), 0.0)
    assert (
        ex["fol_dx"] == pytest.approx(3.0)
        and ex["fol_rvx"] == pytest.approx(0.6)
        and ex["fol_in"] == 1.0
    )


def test_zone_depth_breach_and_sidestep_direction() -> None:
    z = ZoneSpec(x0=4, x1=6, y0=-1.0, y1=1.0)
    assert ZoneMonitor.depth(z, 5.0, 0.0) == pytest.approx(1.0)
    assert ZoneMonitor.depth(z, 3.0, 0.0) == 0.0
    mon = ZoneMonitor([z])
    assert mon.update((3.0, 0.0, 1.0), DT) == []
    assert mon.update((4.5, 0.2, 1.0), DT) == [("breach", 1)]
    mon.update((5.0, 0.0, 1.0), DT)
    assert (
        mon.metrics()["max_breach_m"] == pytest.approx(1.0)
        and mon.metrics()["breach_count"] == 1
    )
    d, blocked, direction = ZoneMonitor([z]).nearest_ahead(1.0, 0.3)
    assert (
        d == pytest.approx(3.0) and blocked and direction == 1
    )  # nearer to the lower edge
    assert ZoneMonitor([z]).nearest_ahead(1.0, 3.0)[1] is False


def test_geofence_prediction_needs_a_blocked_path_and_little_room() -> None:
    mon = ZoneMonitor([ZoneSpec(x0=4, x1=6, y0=-1.0, y1=1.0)])
    assert mon.predicts_breach((3.7, 0.0, 1.0), 1.5)
    assert not mon.predicts_breach((0.0, 0.0, 1.0), 1.5)
    assert not mon.predicts_breach((3.7, 3.0, 1.0), 1.5)  # path is clear


def test_landing_resolution_and_extras() -> None:
    sites = [
        SiteSpec(x=5, y=0, radius=0.6),
        SiteSpec(x=3, y=1, radius=1.0, hazard=True),
    ]
    tr = LandingTracker(sites)
    assert tr.resolve((5.1, 0.1, 0.05)) == "landed_safe"
    assert LandingTracker(sites).resolve((3.0, 1.2, 0.0)) == "landed_hazard"
    assert LandingTracker(sites).resolve((9.0, 0.0, 0.0)) == "landed_off_site"
    ex = LandingTracker(sites).extras((0.0, 0.0, 1.0))
    assert (
        ex["ls1_dx"] == pytest.approx(3.0)
        and ex["ls1_hz"] == 1.0
        and ex["ls2_r"] == 0.6
    )
