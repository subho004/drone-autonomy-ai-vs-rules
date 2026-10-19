"""Flight listing, decisions and replay assembly."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from app.core.exceptions import NotFoundError
from app.domain.actions import ACTION_NAMES
from app.repositories.catalog_repository import ScenarioRepository
from app.repositories.experiment_repository import ExperimentRepository
from app.repositories.flight_repository import FlightRepository
from app.repositories.trajectory_repository import TrajectoryRepository
from app.schemas.api import (
    DecisionOut,
    FlightOut,
    Page,
    ReplayEvent,
    ReplayOut,
    TrajectoryColumns,
)
from app.services.catalog_service import to_spec_out
from app.services.paging import make_page


def _action_name(code: int) -> str:
    return ACTION_NAMES[code] if 0 <= code < len(ACTION_NAMES) else f"ACTION_{code}"


def to_flight_out(row: Mapping[str, Any]) -> FlightOut:
    data = dict(row)
    uri = data.pop("trajectory_uri", None)
    return FlightOut(**data, has_trajectory=bool(uri))


def to_decision_out(row: Mapping[str, Any]) -> DecisionOut:
    return DecisionOut(
        **row, applied_action_name=_action_name(int(row["applied_action"]))
    )


class FlightService:
    def __init__(
        self, flights: FlightRepository, experiments: ExperimentRepository
    ) -> None:
        self._flights = flights
        self._experiments = experiments

    async def list_flights(
        self,
        experiment_id: int,
        page: int,
        page_size: int,
        arm_id: str | None,
        scenario_id: str | None,
        outcome: str | None,
        event_type: str | None,
    ) -> Page[FlightOut]:
        if await self._experiments.get(experiment_id) is None:
            raise NotFoundError(f"Experiment '{experiment_id}' not found")
        rows, total = await self._flights.list_page(
            experiment_id, page, page_size, arm_id, scenario_id, outcome, event_type
        )
        return make_page([to_flight_out(r) for r in rows], total, page, page_size)

    async def get_summary(self, flight_id: int) -> FlightOut:
        row = await self._flights.get(flight_id)
        if row is None:
            raise NotFoundError(f"Flight '{flight_id}' not found")
        return to_flight_out(row)

    async def list_decisions(
        self, flight_id: int, page: int, page_size: int
    ) -> Page[DecisionOut]:
        if await self._flights.get(flight_id) is None:
            raise NotFoundError(f"Flight '{flight_id}' not found")
        rows, total = await self._flights.decisions_page(flight_id, page, page_size)
        return make_page([to_decision_out(r) for r in rows], total, page, page_size)


class ReplayService:
    def __init__(
        self,
        flights: FlightRepository,
        scenarios: ScenarioRepository,
        trajectories: TrajectoryRepository,
    ) -> None:
        self._flights = flights
        self._scenarios = scenarios
        self._trajectories = trajectories

    async def get_replay(self, flight_id: int) -> ReplayOut:
        row = await self._flights.get(flight_id)
        if row is None:
            raise NotFoundError(f"Flight '{flight_id}' not found")
        uri = row.get("trajectory_uri")
        loaded = await self._trajectories.load(str(uri)) if uri else None
        if loaded is None:
            raise NotFoundError(f"No trajectory recorded for flight '{flight_id}'")
        columns, hz = loaded
        scenario_row = await self._scenarios.get(str(row["scenario_id"]))
        if scenario_row is None:
            raise NotFoundError(f"Scenario '{row['scenario_id']}' not found")
        decisions = [
            to_decision_out(r) for r in await self._flights.decisions_all(flight_id)
        ]
        flight = to_flight_out(row)
        return ReplayOut(
            flight=flight,
            scenario=to_spec_out(scenario_row),
            hz=hz,
            trajectory=TrajectoryColumns.model_validate(columns),
            decisions=decisions,
            events=build_events(flight, columns),
        )


def build_events(
    flight: FlightOut, columns: Mapping[str, list[Any]]
) -> list[ReplayEvent]:
    """Derive timeline markers: obstacle onset, shield overrides, end of flight."""

    events: list[ReplayEvent] = []
    t = columns["t"]
    orad = columns["orad"]
    for ti, rad in zip(t, orad, strict=True):
        if rad is not None and not math.isnan(rad):
            events.append(
                ReplayEvent(t=float(ti), kind="obstacle", label="Obstacle appears")
            )
            break
    previous = 0
    for ti, flag in zip(t, columns["shield"], strict=True):
        current = int(flag or 0)
        if current and not previous:
            events.append(
                ReplayEvent(t=float(ti), kind="shield", label="Shield override")
            )
        previous = current
    if t:
        end = float(t[-1])
        if flight.collision:
            events.append(ReplayEvent(t=end, kind="collision", label="Collision"))
        elif flight.success:
            events.append(ReplayEvent(t=end, kind="goal", label="Goal reached"))
        elif flight.outcome:
            events.append(
                ReplayEvent(t=end, kind="end", label=flight.outcome.capitalize())
            )
    return sorted(events, key=lambda e: e.t)
