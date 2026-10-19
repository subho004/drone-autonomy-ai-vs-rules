"""External steering of game orbs: each orb is steered by a decision-maker that sees only
where the drone is in the orb's own heading frame and answers turn left, right or straight.

Orb decisions run on the same simulation clock as the drone's: a request issued at tick t
is applied after its own latency, and the orb keeps its previous command meanwhile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from app.domain.decision import Decision
from app.harness.latency import LatencyPolicy
from app.sim.world import DroneWorld

LEFT, STRAIGHT, RIGHT = 1, 0, -1
AHEAD_DEG = 12.0  # within this bearing the orb goes straight


@dataclass(frozen=True, slots=True)
class OrbView:
    forward: float  # drone ahead (+) of the orb's heading, metres
    left: float  # drone to the orb's left (+), metres
    distance: float
    closing: float  # > 0 while the gap is shrinking

    @property
    def bearing_deg(self) -> float:
        return math.degrees(math.atan2(self.left, self.forward))


@dataclass(frozen=True, slots=True)
class OrbChoice:
    cmd: int
    latency_ms: float
    cost_usd: float = 0.0


class OrbSteerer(Protocol):
    name: str

    def steer(self, view: OrbView) -> OrbChoice: ...


class PursuitSteerer:
    """Turn toward the drone (pure pursuit); the rule-based orb brain, zero latency."""

    name = "rule_pursuit"

    def steer(self, view: OrbView) -> OrbChoice:
        if abs(view.bearing_deg) < AHEAD_DEG and view.forward > 0:
            return OrbChoice(STRAIGHT, 0.0)
        return OrbChoice(LEFT if view.left > 0 else RIGHT, 0.0)


@dataclass(slots=True)
class _Flight:
    index: int
    apply_tick: int
    choice: OrbChoice


class OrbController:
    """Runs one steerer for every external orb, with sim-time latency."""

    def __init__(self, steerer: OrbSteerer, latency: LatencyPolicy) -> None:
        self.steerer = steerer
        self.latency = latency
        self._pending: list[_Flight] = []
        self.decisions = 0
        self.cost_usd = 0.0
        self.latencies_ms: list[float] = []
        self.commands: list[
            tuple[int, int, int]
        ] = []  # (tick, orb index, cmd) when applied

    def reset(self, seed: int) -> None:
        self.latency.reset(seed + 7919)
        self._pending.clear()
        self.decisions = 0
        self.cost_usd = 0.0
        self.latencies_ms.clear()
        self.commands.clear()

    def tick(self, n: int, world: DroneWorld) -> None:
        for item in [p for p in self._pending if p.apply_tick <= n]:
            world.set_orb_command(item.index, item.choice.cmd)
            self.commands.append((n, item.index, item.choice.cmd))
        self._pending = [p for p in self._pending if p.apply_tick > n]
        in_flight = {p.index for p in self._pending}
        for index, fwd, left, dist, closing in world.orb_views():
            if index in in_flight:
                continue
            choice = self.steerer.steer(OrbView(fwd, left, dist, closing))
            self.decisions += 1
            self.cost_usd += choice.cost_usd
            lat_ms, ticks = self.latency.resolve(Decision(None, choice.latency_ms))
            self.latencies_ms.append(lat_ms)
            if ticks == 0:
                world.set_orb_command(index, choice.cmd)
            else:
                self._pending.append(_Flight(index, n + ticks, choice))


__all__ = [
    "LEFT",
    "RIGHT",
    "STRAIGHT",
    "OrbChoice",
    "OrbController",
    "OrbSteerer",
    "OrbView",
    "PursuitSteerer",
]
