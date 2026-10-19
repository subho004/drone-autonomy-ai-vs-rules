from __future__ import annotations

import numpy as np

from app.controllers.arms import FsmArm, HoverArm, OracleArm, ReflexArm, WaypointArm
from app.domain.actions import Action
from app.domain.decision import Decision
from app.domain.observation import Observation
from app.domain.trajectory import TRAJ_COLUMNS
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.scenarios.spec import ObstacleSpec, ScenarioSpec
from app.sim.world import DroneWorld


def _static(d: float = 4.0, seed: int = 1) -> ScenarioSpec:
    return ScenarioSpec(seed=seed, obstacle=ObstacleSpec(trigger_x=4.0, d_spawn=d))


def _cfg(
    mode: LatencyMode = LatencyMode.MEASURED, ms: float = 0.0, shield: bool = False
) -> FlightConfig:
    return FlightConfig(LatencyPolicy(mode, fixed_ms=ms), shield=shield)


def test_clear_flight_succeeds(world: DroneWorld) -> None:
    r = run_flight(
        world, ScenarioSpec(seed=1, event_type="clear"), WaypointArm(), _cfg()
    )
    assert r.metrics["outcome"] == "success" and r.metrics["collision"] == 0
    assert 6.0 < r.metrics["time_to_goal"] < 9.0


def test_no_avoidance_collides_but_shield_prevents_it(world: DroneWorld) -> None:
    bare = run_flight(world, _static(), WaypointArm(), _cfg())
    shielded = run_flight(world, _static(), WaypointArm(), _cfg(shield=True))
    assert bare.metrics["collision"] == 1
    assert shielded.metrics["collision"] == 0 and shielded.metrics["success"] == 0
    assert shielded.metrics["shield_overrides"] >= 1


def test_hover_never_collides_never_succeeds(world: DroneWorld) -> None:
    r = run_flight(world, _static(), HoverArm(), _cfg())
    assert r.metrics["outcome"] == "timeout" and r.metrics["collision"] == 0


def test_reflex_and_oracle_dodge_a_far_obstacle(world: DroneWorld) -> None:
    for arm in (ReflexArm(), FsmArm(), OracleArm(2.0)):
        r = run_flight(world, _static(4.0), arm, _cfg())
        assert r.metrics["success"] == 1, arm.provider_id
        assert r.metrics["reaction_time_s"] is not None


def test_latency_injection_is_monotone_for_the_oracle(world: DroneWorld) -> None:
    rates = []
    for ms in (0.0, 300.0, 3000.0):
        hits = 0
        for seed in range(6):
            spec = _static(2.0 + 0.1 * seed, seed)
            r = run_flight(world, spec, OracleArm(2.0), _cfg(LatencyMode.FIXED, ms))
            hits += r.metrics["collision"]
        rates.append(hits / 6)
    assert rates[0] <= rates[1] <= rates[2]
    assert rates[2] > rates[0]


def test_lockstep_hides_latency_that_injection_reveals(world: DroneWorld) -> None:
    spec = _static(2.2)
    lock = run_flight(world, spec, OracleArm(2.0), _cfg(LatencyMode.LOCKSTEP))
    slow = run_flight(world, spec, OracleArm(2.0), _cfg(LatencyMode.FIXED, 2000.0))
    assert lock.metrics["collision"] == 0 and slow.metrics["collision"] == 1


def test_decision_rows_record_latency_ticks_and_staleness(world: DroneWorld) -> None:
    r = run_flight(world, _static(), OracleArm(2.0), _cfg(LatencyMode.FIXED, 200.0))
    row = r.decisions[0]
    assert row["latency_ticks"] == 10 and row["apply_tick"] - row["obs_tick"] == 10
    assert all(d["apply_tick"] >= d["obs_tick"] for d in r.decisions)


class _InvalidProvider:
    """Always returns unparseable output."""

    provider_id = "invalid"

    def decide(self, obs: Observation) -> Decision:
        return Decision(
            action=None, latency_ms=5.0, valid=False, error="invalid_output"
        )

    def close(self) -> None:
        return None


def test_invalid_decisions_fall_back_and_eventually_hover(world: DroneWorld) -> None:
    r = run_flight(world, _static(), _InvalidProvider(), _cfg())
    assert r.metrics["fallback_rate"] == 1.0 and r.metrics["invalid_rate"] == 1.0
    assert r.decisions[-1]["applied_action"] == int(Action.HOVER)


def test_trajectory_matches_contract_and_is_sim_time(world: DroneWorld) -> None:
    r = run_flight(world, _static(), ReflexArm(), _cfg())
    assert r.trajectory is not None and set(r.trajectory) == set(TRAJ_COLUMNS)
    t = r.trajectory["t"]
    assert np.all(np.diff(t) > 0) and abs((t[1] - t[0]) - 1 / 48) < 1e-9
    assert not np.isnan(r.trajectory["x"]).any()


def test_same_inputs_give_same_flight(world: DroneWorld) -> None:
    a = run_flight(world, _static(3.0), ReflexArm(), _cfg(LatencyMode.FIXED, 100.0))
    b = run_flight(world, _static(3.0), ReflexArm(), _cfg(LatencyMode.FIXED, 100.0))
    assert a.metrics["outcome"] == b.metrics["outcome"]
    assert a.metrics["path_length"] == b.metrics["path_length"]
    assert np.array_equal(a.trajectory["x"], b.trajectory["x"])  # type: ignore[index]


def test_battery_critical_requires_landing_or_returning(world: DroneWorld) -> None:
    spec = ScenarioSpec(seed=2, event_type="battery_critical", max_time_s=24.0)
    spec.sensors.battery_drop_at_s = 3.0
    spec.deadline_s = 6.0
    ok = run_flight(world, spec, OracleArm(2.0), _cfg())
    bad = run_flight(world, spec, WaypointArm(), _cfg())
    assert ok.metrics["success"] == 1 and ok.metrics["outcome"] == "landed"
    assert bad.metrics["success"] == 0


def test_gps_loss_rewards_a_timely_hover(world: DroneWorld) -> None:
    spec = ScenarioSpec(seed=3, event_type="gps_loss")
    spec.sensors.gps_loss_at_s = 3.0
    spec.deadline_s = 2.0
    good = run_flight(world, spec, ReflexArm(), _cfg())
    bad = run_flight(world, spec, WaypointArm(), _cfg())
    assert good.metrics["success"] == 1 and bad.metrics["success"] == 0
