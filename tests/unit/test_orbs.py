from __future__ import annotations

import numpy as np

from app.controllers.arms import build_classical_arm
from app.controllers.sim_arms import OrbDodger
from app.domain.actions import Action
from app.domain.observation import Observation
from app.harness.flight_runner import FlightConfig, run_flight
from app.harness.latency import LatencyMode, LatencyPolicy
from app.harness.orb_control import (
    LEFT,
    RIGHT,
    STRAIGHT,
    OrbChoice,
    OrbController,
    OrbView,
    PursuitSteerer,
)
from app.providers.encoding import ObservationEncoder
from app.providers.orb_llm import describe
from app.scenarios.sims import draw_orbs
from app.sim.world import DT, DroneWorld


def test_pursuit_steerer_turns_toward_the_target() -> None:
    s = PursuitSteerer()
    assert s.steer(OrbView(5.0, 0.2, 5.0, 1.0)).cmd == STRAIGHT
    assert s.steer(OrbView(3.0, 2.0, 3.6, 1.0)).cmd == LEFT
    assert s.steer(OrbView(3.0, -2.0, 3.6, 1.0)).cmd == RIGHT


def test_orb_dodger_waits_then_commits_to_one_side() -> None:
    d = OrbDodger(dodge_range_m=1.0, hold_s=0.4)
    assert d.dodge(0.0, 3.0, 20.0, 2.0) is None  # too far: no dodge yet
    assert (
        d.dodge(0.1, 0.9, 20.0, 2.0) == Action.AVOID_RIGHT
    )  # orb on the left: go right
    assert d.dodge(0.2, 0.8, -30.0, 2.0) == Action.AVOID_RIGHT  # keeps its side
    assert d.dodge(2.0, 3.0, 0.0, -1.0) is None  # orb opening: released


class _Const:
    name = "const"

    def __init__(self, cmd: int, ms: float) -> None:
        self.cmd, self.ms = cmd, ms

    def steer(self, view: OrbView) -> OrbChoice:
        return OrbChoice(self.cmd, self.ms)


def test_orb_command_arrives_after_its_latency(world: DroneWorld) -> None:
    spec = draw_orbs(3, ext=True)
    world.reset(spec)
    ctl = OrbController(_Const(LEFT, 500.0), LatencyPolicy(LatencyMode.MEASURED))
    ctl.reset(3)
    ref = np.array([0.0, 0.0, 1.0])
    first_cmd = None
    for n in range(700):
        ref[0] += 1.4 * DT
        ctl.tick(n, world)
        world.step(ref, np.array([1.4, 0.0, 0.0]))
        if ctl.commands and first_cmd is None:
            first_cmd = n
    assert ctl.decisions >= 1 and first_cmd is not None
    issued = ctl.latencies_ms[0]
    assert issued == 500.0 and first_cmd >= int(
        0.5 / DT
    )  # not before 500 ms of sim time


def test_radar_sees_an_orb_the_rays_would_miss(world: DroneWorld) -> None:
    spec = draw_orbs(3)
    world.reset(spec)
    ref = np.array([0.0, 0.0, 1.0])
    seen_far = False
    for _ in range(900):
        ref[0] += 1.4 * DT
        world.step(ref, np.array([1.4, 0.0, 0.0]))
        if world.obstacle_active:
            obs = world.observe(0)
            if "orb_range" in obs.extras and obs.nearest_range < 6.0:
                seen_far = True
                break
    assert seen_far  # detected well before the ~1 m where a 0.25 m ball hits a ray


def test_external_orbs_are_deterministic(world: DroneWorld) -> None:
    def fly() -> str:
        ctl = OrbController(PursuitSteerer(), LatencyPolicy(LatencyMode.LOCKSTEP))
        cfg = FlightConfig(
            LatencyPolicy(LatencyMode.MEASURED),
            record_trajectory=False,
            label_decisions=False,
            orbs=ctl,
        )
        m = run_flight(
            world, draw_orbs(6001, ext=True), build_classical_arm("orb_pilot"), cfg
        ).metrics
        return f"{m['outcome']}:{m['t_end']:.3f}"

    assert fly() == fly()


def test_orb_prompt_block_and_llm_view_text() -> None:
    obs = Observation(
        tick=0, t_sim=1.0, pos=(2.0, 0.0, 1.0), vel=(1.0, 0.0, 0.0), goal_dx=5.0, home_dist=2.0,
        rays=(10.0,) * 16, sectors=(10.0,) * 8, nearest_range=1.1, nearest_bearing_deg=30.0,
        closing_speed=2.0, ttc=3.0, battery_pct=80.0, gps_ok=True, last_action=0,
        extras={"orb_range": 1.1, "orb_bearing_deg": 40.0, "orb_closing": 2.0},
    )  # fmt: skip
    text = ObservationEncoder().encode(obs)
    assert "orb: range=1.1 dir=LEFT closing=FAST" in text and "dodge LATE" in text
    assert "LEFT" in describe(OrbView(2.0, 2.0, 2.8, 1.0)) and "CLOSING" in describe(
        OrbView(2.0, 2.0, 2.8, 1.0)
    )


def test_swarm_arrives_in_waves_of_two() -> None:
    from app.scenarios.sims import draw_swarm

    spec = draw_swarm(7003)
    assert spec.mission is not None
    orbs = spec.mission.obstacles
    assert len(orbs) == 6
    triggers = sorted(round(o.trigger_x, 1) for o in orbs)
    assert len({round(t / 2) for t in triggers[::2]}) == 3  # three waves
    assert spec.goal_x > max(o.trigger_x for o in orbs) + 3.0


def test_random_orb_scenarios_vary_but_repeat() -> None:
    from app.scenarios.sims import draw_orbs_random

    specs = [draw_orbs_random(s) for s in range(9000, 9012)]
    counts = {len(sp.mission.obstacles) for sp in specs if sp.mission}
    speeds = {round(sp.mission.obstacles[0].speed, 2) for sp in specs if sp.mission}
    assert counts <= {2, 3, 4, 5} and len(counts) > 1 and len(speeds) > 6
    assert draw_orbs_random(9003).scenario_id == draw_orbs_random(9003).scenario_id
