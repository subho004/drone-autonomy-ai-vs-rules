"""Aggregated results, latency curves and latency distributions."""

from __future__ import annotations

from typing import Literal

from app.core.exceptions import NotFoundError
from app.repositories.base import Row
from app.repositories.experiment_repository import ExperimentRepository
from app.repositories.flight_repository import FlightRepository
from app.schemas.api import (
    BreakdownRow,
    CurveSeries,
    LatencyCurvesOut,
    LatencyDistribution,
    LatencyDistributionOut,
    Page,
    ResultRow,
)
from app.services import aggregation
from app.services.paging import make_page

Scope = Literal["challenge", "controls", "all"]
CHALLENGE_EVENTS = frozenset(
    {"obstacle_static", "obstacle_approach", "intruder_cross", "mission"}
)


def apply_scope(rows: list[Row], scope: Scope) -> list[Row]:
    """Headline = certified obstacle scenarios; controls = impossible (B < 0) scenarios."""

    if scope == "all":
        return rows
    if scope == "controls":
        return [r for r in rows if float(r["budget_s"]) < 0.0]
    return [
        r
        for r in rows
        if r["event_type"] in CHALLENGE_EVENTS and float(r["budget_s"]) >= 0.0
    ]


class ResultsService:
    def __init__(
        self, experiments: ExperimentRepository, flights: FlightRepository
    ) -> None:
        self._experiments = experiments
        self._flights = flights

    async def _require(self, experiment_id: int) -> None:
        if await self._experiments.get(experiment_id) is None:
            raise NotFoundError(f"Experiment '{experiment_id}' not found")

    async def results(
        self, experiment_id: int, page: int, page_size: int, scope: Scope = "challenge"
    ) -> Page[ResultRow]:
        await self._require(experiment_id)
        scoped = apply_scope(await self._flights.metric_rows(experiment_id), scope)
        rows = aggregation.results(scoped)
        start = (page - 1) * page_size
        items = [ResultRow(**r) for r in rows[start : start + page_size]]
        return make_page(items, len(rows), page, page_size)

    async def latency_curves(
        self, experiment_id: int, scope: Scope = "challenge"
    ) -> LatencyCurvesOut:
        await self._require(experiment_id)
        rows = apply_scope(await self._flights.metric_rows(experiment_id), scope)
        return LatencyCurvesOut(
            experiment_id=experiment_id,
            series=[CurveSeries(**s) for s in aggregation.curve_series(rows)],
            by_tier=[BreakdownRow(**b) for b in aggregation.breakdown(rows, "tier")],
            by_event=[
                BreakdownRow(**b) for b in aggregation.breakdown(rows, "event_type")
            ],
        )

    async def latency_distribution(self, experiment_id: int) -> LatencyDistributionOut:
        await self._require(experiment_id)
        rows = await self._flights.decision_latencies(experiment_id)
        return LatencyDistributionOut(
            experiment_id=experiment_id,
            series=[
                LatencyDistribution(**d) for d in aggregation.latency_distribution(rows)
            ],
        )
