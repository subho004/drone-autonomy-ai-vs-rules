"""Shared fixtures for unit tests (one PyBullet world per test module)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.sim.world import DroneWorld


@pytest.fixture(scope="module")
def world() -> Iterator[DroneWorld]:
    w = DroneWorld()
    yield w
    w.close()
