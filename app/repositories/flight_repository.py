"""Flight, decision and aggregate-source queries."""

from __future__ import annotations

from app.repositories.base import BaseRepository, Row, page_bounds

_FLIGHT_COLUMNS = (
    "f.id, f.experiment_id, f.scenario_id, f.arm_id, f.latency_mode, f.latency_ms, "
    "f.shield, f.cadence, f.rep, f.status, f.outcome, f.success, f.collision, f.t_end, "
    "f.time_to_goal, f.path_length, f.min_clearance, f.reaction_time_s, f.n_decisions, "
    "f.lat_mean_ms, f.lat_p95_ms, f.invalid_rate, f.fallback_rate, f.shield_overrides, "
    "f.cost_usd, f.trajectory_uri, s.event_type, s.tier, s.budget_s"
)
_FLIGHT_FROM = "FROM flights f JOIN scenarios s ON s.id = f.scenario_id"

# Deterministic subsampling keeps the ECDF query bounded on huge decision tables.
MAX_LATENCY_SAMPLES = 200_000


class FlightRepository(BaseRepository):
    async def list_page(
        self,
        experiment_id: int,
        page: int,
        page_size: int,
        arm_id: str | None,
        scenario_id: str | None,
        outcome: str | None,
        event_type: str | None,
    ) -> tuple[list[Row], int]:
        limit, offset = page_bounds(page, page_size)
        where = (
            "WHERE f.experiment_id = :eid AND (:arm IS NULL OR f.arm_id = :arm) "
            "AND (:scn IS NULL OR f.scenario_id = :scn) "
            "AND (:outcome IS NULL OR f.outcome = :outcome) "
            "AND (:event IS NULL OR s.event_type = :event)"
        )
        params = {
            "eid": experiment_id,
            "arm": arm_id,
            "scn": scenario_id,
            "outcome": outcome,
            "event": event_type,
        }
        total = await self._scalar_int(
            f"SELECT COUNT(*) {_FLIGHT_FROM} {where}", params
        )
        rows = await self._all(
            f"SELECT {_FLIGHT_COLUMNS} {_FLIGHT_FROM} {where} "
            "ORDER BY f.id LIMIT :limit OFFSET :offset",
            {**params, "limit": limit, "offset": offset},
        )
        return rows, total

    async def get(self, flight_id: int) -> Row | None:
        return await self._one(
            f"SELECT {_FLIGHT_COLUMNS} {_FLIGHT_FROM} WHERE f.id = :id",
            {"id": flight_id},
        )

    async def decisions_page(
        self, flight_id: int, page: int, page_size: int
    ) -> tuple[list[Row], int]:
        limit, offset = page_bounds(page, page_size)
        total = await self._scalar_int(
            "SELECT COUNT(*) FROM decisions WHERE flight_id = :id", {"id": flight_id}
        )
        rows = await self._all(
            f"{_DECISION_SELECT} WHERE flight_id = :id ORDER BY seq LIMIT :limit OFFSET :offset",
            {"id": flight_id, "limit": limit, "offset": offset},
        )
        return rows, total

    async def decisions_all(self, flight_id: int) -> list[Row]:
        return await self._all(
            f"{_DECISION_SELECT} WHERE flight_id = :id ORDER BY seq", {"id": flight_id}
        )

    async def metric_rows(self, experiment_id: int) -> list[Row]:
        """Per-flight metrics of completed flights (source of all aggregates)."""

        return await self._all(
            "SELECT f.arm_id, f.latency_mode, f.latency_ms, f.shield, f.cadence, "
            "s.tier, s.event_type, s.budget_s, f.collision, f.success, f.lat_mean_ms, f.lat_p95_ms, "
            "f.path_length, f.time_to_goal, f.invalid_rate "
            f"{_FLIGHT_FROM} WHERE f.experiment_id = :eid AND f.status = 'complete'",
            {"eid": experiment_id},
        )

    async def decision_latencies(self, experiment_id: int) -> list[Row]:
        """Positive decision latencies with provider/model (deterministically bounded)."""

        total = await self._scalar_int(
            "SELECT COUNT(*) FROM decisions d JOIN flights f ON f.id = d.flight_id "
            "WHERE f.experiment_id = :eid",
            {"eid": experiment_id},
        )
        stride = max(1, -(-total // MAX_LATENCY_SAMPLES))
        return await self._all(
            "SELECT a.provider, a.model, d.latency_ms FROM decisions d "
            "JOIN flights f ON f.id = d.flight_id JOIN arms a ON a.id = f.arm_id "
            "WHERE f.experiment_id = :eid AND d.latency_ms > 0 "
            "AND ((d.flight_id * 7919 + d.seq) % :stride) = 0",
            {"eid": experiment_id, "stride": stride},
        )


_DECISION_SELECT = (
    "SELECT seq, obs_tick, apply_tick, t_obs, t_apply, action, applied_action, source, "
    "valid, latency_ms, latency_ticks, confidence, acceptable, from_cache, cost_usd "
    "FROM decisions"
)
