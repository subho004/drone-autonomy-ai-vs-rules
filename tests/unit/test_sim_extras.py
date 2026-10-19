from __future__ import annotations

import json
import math

import numpy as np

from app.controllers.arms import build_classical_arm
from app.controllers.sim_arms import PredictiveArm, build_predictive
from app.domain.actions import Action
from app.domain.observation import Observation
from app.harness.experiment import _latency_cells
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy, lognormal_trace
from app.render.sim_video import family_of
from app.scenarios import sims
from app.schemas.experiment import ArmCfg, ExperimentConfig
from app.sim.world import DroneWorld


def _obs(fwd: float, closing: float, ttc: float) -> Observation:
    return Observation(
        tick=0, t_sim=1.0, pos=(1.0, 0.0, 1.0), vel=(1.0, 0.0, 0.0), goal_dx=8.0, home_dist=1.0,
        rays=(10.0,) * 16, sectors=(10.0,) * 8, nearest_range=fwd, nearest_bearing_deg=0.0,
        closing_speed=closing, ttc=ttc, battery_pct=80.0, gps_ok=True, last_action=0, fwd_range=fwd,
    )  # fmt: skip


def test_lognormal_trace_has_the_requested_median_and_a_heavy_tail() -> None:
    trace = np.array(lognormal_trace(300.0, 0.4))
    assert math.isclose(float(np.median(trace)), 300.0, rel_tol=0.05)
    assert np.percentile(trace, 99) > 2.0 * 300.0
    assert lognormal_trace(300.0, 0.4) == lognormal_trace(
        300.0, 0.4
    )  # same trace every flight


def test_sampled_policy_draws_from_the_trace_deterministically() -> None:
    from app.domain.decision import Decision

    pol = LatencyPolicy(LatencyMode.SAMPLED, trace_ms=lognormal_trace(300.0), seed=1)
    pol.reset(7)
    first = [
        pol.resolve(Decision(Action.HOVER, 1.0, model_id="x"))[0] for _ in range(5)
    ]
    pol.reset(7)
    again = [
        pol.resolve(Decision(Action.HOVER, 1.0, model_id="x"))[0] for _ in range(5)
    ]
    assert first == again and len(set(first)) > 1


def test_sampled_cells_are_enqueued_only_for_sampled_arms() -> None:
    cfg = ExperimentConfig(
        name="t", arms=[], latency_sweep_ms=[0.0], latency_sampled_ms=[200.0, 400.0]
    )
    plain = list(_latency_cells(ArmCfg(id="a", kind="fsm", measured=False), cfg))
    human = list(
        _latency_cells(
            ArmCfg(id="h", kind="fsm", sweep=False, measured=False, sampled=True), cfg
        )
    )
    assert plain == [("fixed", 0.0)]
    assert human == [("sampled", 200.0), ("sampled", 400.0)]


def test_predictive_arm_advances_range_and_ttc_by_the_expected_latency() -> None:
    arm = build_predictive("fsm", 400.0)
    assert isinstance(arm, PredictiveArm)
    seen = arm.project(_obs(fwd=3.0, closing=2.0, ttc=1.5))
    assert math.isclose(seen.fwd_range, 3.0 - 2.0 * 0.4) and math.isclose(seen.ttc, 1.1)
    clear = _obs(fwd=10.0, closing=0.0, ttc=99.0)
    assert arm.project(clear) is clear  # nothing in view: unchanged
    assert build_classical_arm(
        "predict_fsm", config={"expected_ms": 250.0}
    ).provider_id.endswith("250")


def test_delivery_that_enters_a_zone_is_not_a_success(world: DroneWorld) -> None:
    spec = sims.draw_delivery(2)
    cfg = FlightConfig(
        LatencyPolicy(LatencyMode.MEASURED),
        shield=False,
        record_trajectory=False,
        label_decisions=False,
    )
    blind = run_flight(world, spec, build_classical_arm("mission_fsm"), cfg).metrics
    assert json.loads(blind["task_json"])["max_breach_m"] > 0.2
    assert blind["success"] == 0  # it landed on the pad, but through the zone


def test_family_detection_for_rendering() -> None:
    assert family_of(sims.draw_follow(1)) == "follow"
    assert family_of(sims.draw_delivery(1)) == "delivery"
    assert family_of(sims.draw_landing(1)) == "landing"
    assert family_of(sims.draw_gates(1)) == "gates"
    assert family_of(sims.draw_pursuit(1)) == "pursuit"
    assert family_of(sims.draw_canyon(1)) == "canyon"
