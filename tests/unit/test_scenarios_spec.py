from __future__ import annotations

import numpy as np

from app.scenarios.spec import ObstacleSpec, ScenarioSpec
from app.sim.rng import channel_generator, ou_process


def test_scenario_id_is_stable_and_sensitive() -> None:
    a = ScenarioSpec(seed=1, obstacle=ObstacleSpec(d_spawn=2.5))
    b = ScenarioSpec(seed=1, obstacle=ObstacleSpec(d_spawn=2.5))
    c = ScenarioSpec(seed=1, obstacle=ObstacleSpec(d_spawn=2.6))
    assert a.scenario_id == b.scenario_id
    assert a.scenario_id != c.scenario_id


def test_certified_flag_does_not_change_id() -> None:
    a = ScenarioSpec(seed=3)
    b = a.model_copy(update={"certified": True})
    assert a.scenario_id == b.scenario_id


def test_spec_roundtrips_through_json() -> None:
    a = ScenarioSpec(seed=9, obstacle=ObstacleSpec(kind="cross", speed=1.2))
    b = ScenarioSpec.model_validate_json(a.model_dump_json())
    assert a.scenario_id == b.scenario_id


def test_channels_are_independent_and_deterministic() -> None:
    x1 = channel_generator(5, "ray_noise").standard_normal(8)
    x2 = channel_generator(5, "ray_noise").standard_normal(8)
    y = channel_generator(5, "wind").standard_normal(8)
    z = channel_generator(6, "ray_noise").standard_normal(8)
    assert np.array_equal(x1, x2)
    assert not np.array_equal(x1, y)
    assert not np.array_equal(x1, z)


def test_ou_process_has_requested_std() -> None:
    gen = channel_generator(1, "ou")
    path = ou_process(gen, 20000, 1 / 48, sigma=0.5, tau=1.0)
    assert abs(float(path.std()) - 0.5) < 0.08
    assert np.all(ou_process(gen, 10, 1 / 48, 0.0, 1.0) == 0.0)
