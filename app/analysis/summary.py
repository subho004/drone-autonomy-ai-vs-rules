"""Experiment summaries computed from the SQLite store (read-only).

Only ``status = 'complete'`` flights count. The scenario is the unit of
analysis for paired comparisons: replicates are averaged per scenario first.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median

from app.analysis import stats
from app.core.logging import get_logger

logger = get_logger(__name__)

RHO_EDGES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, math.inf)

CellKey = tuple[str, float, int, str]  # latency_mode, latency_ms, shield, cadence


# ----------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class ArmSummary:
    arm_id: str
    latency_mode: str
    latency_ms: float
    shield: int
    cadence: str
    n: int
    collisions: int
    collision_rate: float
    collision_lo: float
    collision_hi: float
    successes: int
    success_rate: float
    success_lo: float
    success_hi: float
    lat_mean_ms: float | None
    lat_p50_ms: float | None
    lat_p95_ms: float | None
    path_length: float | None
    time_to_goal: float | None
    invalid_rate: float | None
    fallback_rate: float | None
    reaction_time_s: float | None


@dataclass(frozen=True)
class PairedComparison:
    arm_a: str
    arm_b: str
    latency_mode: str
    latency_ms: float
    shield: int
    cadence: str
    n_pairs: int
    b: int
    c: int
    rate_a: float
    rate_b: float
    delta: float
    delta_lo: float
    delta_hi: float
    p_value: float
    p_holm: float | None = None


@dataclass(frozen=True)
class RhoBin:
    arm_id: str
    latency_mode: str
    rho_lo: float
    rho_hi: float
    n: int
    collisions: int
    rate: float
    lo: float
    hi: float


@dataclass(frozen=True)
class StratumSummary:
    arm_id: str
    latency_mode: str
    latency_ms: float
    shield: int
    cadence: str
    stratum: str
    n: int
    collisions: int
    rate: float
    lo: float
    hi: float


@dataclass(frozen=True)
class LatencyDistribution:
    arm_id: str
    n: int
    values_ms: list[float]
    summary: stats.PercentileSummary


# ----------------------------------------------------------------------- utils
def _mean_or_none(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return mean(vals) if vals else None


def _median_or_none(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return median(vals) if vals else None


def _rate(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = stats.wilson(k, n)
    return (k / n if n else math.nan), lo, hi


def _fetch_flights(conn: sqlite3.Connection, experiment_id: int) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT f.*, s.tier AS tier, s.event_type AS event_type, s.budget_s AS budget_s "
        "FROM flights f JOIN scenarios s ON s.id = f.scenario_id "
        "WHERE f.experiment_id = ? AND f.status = 'complete'",
        (experiment_id,),
    )
    return list(cur.fetchall())


def _cell(row: sqlite3.Row) -> CellKey:
    return (
        str(row["latency_mode"]),
        float(row["latency_ms"]),
        int(row["shield"]),
        str(row["cadence"]),
    )


def _matches(
    row: sqlite3.Row,
    latency_mode: str | None,
    latency_ms: float | None,
    shield: int | None,
    cadence: str | None,
) -> bool:
    return (
        (latency_mode is None or row["latency_mode"] == latency_mode)
        and (latency_ms is None or row["latency_ms"] == latency_ms)
        and (shield is None or row["shield"] == shield)
        and (cadence is None or row["cadence"] == cadence)
    )


# ------------------------------------------------------------------ arm summary
def arm_summary(conn: sqlite3.Connection, experiment_id: int) -> list[ArmSummary]:
    """One row per arm x latency_mode x latency_ms x shield x cadence.

    ``lat_mean_ms``/``lat_p50_ms`` are the mean/median of per-flight mean
    latency; ``lat_p95_ms`` is the mean of per-flight p95.
    """

    groups: dict[tuple[str, CellKey], list[sqlite3.Row]] = defaultdict(list)
    for row in _fetch_flights(conn, experiment_id):
        groups[(str(row["arm_id"]), _cell(row))].append(row)
    out: list[ArmSummary] = []
    for (arm_id, cell), rows in sorted(groups.items()):
        coll = [r["collision"] for r in rows if r["collision"] is not None]
        succ = [r["success"] for r in rows if r["success"] is not None]
        c_rate, c_lo, c_hi = _rate(int(sum(coll)), len(coll))
        s_rate, s_lo, s_hi = _rate(int(sum(succ)), len(succ))
        out.append(
            ArmSummary(
                arm_id, cell[0], cell[1], cell[2], cell[3],
                len(rows), int(sum(coll)), c_rate, c_lo, c_hi,
                int(sum(succ)), s_rate, s_lo, s_hi,
                _mean_or_none([r["lat_mean_ms"] for r in rows]),
                _median_or_none([r["lat_mean_ms"] for r in rows]),
                _mean_or_none([r["lat_p95_ms"] for r in rows]),
                _mean_or_none([r["path_length"] for r in rows]),
                _mean_or_none([r["time_to_goal"] for r in rows]),
                _mean_or_none([r["invalid_rate"] for r in rows]),
                _mean_or_none([r["fallback_rate"] for r in rows]),
                _mean_or_none([r["reaction_time_s"] for r in rows]),
            )
        )  # fmt: skip
    return out


def budget_distribution(
    conn: sqlite3.Connection, experiment_id: int
) -> dict[str, float]:
    """min/median/max/mean of ``budget_s`` over scenarios flown in the experiment."""

    cur = conn.execute(
        "SELECT DISTINCT s.id, s.budget_s FROM scenarios s JOIN flights f ON f.scenario_id = s.id "
        "WHERE f.experiment_id = ?",
        (experiment_id,),
    )
    budgets = [float(r["budget_s"]) for r in cur.fetchall()]
    if not budgets:
        return {"n": 0.0}
    return {
        "n": float(len(budgets)),
        "min": min(budgets),
        "median": median(budgets),
        "mean": mean(budgets),
        "max": max(budgets),
    }


# ------------------------------------------------------------- paired comparison
def _per_scenario_collision(
    rows: list[sqlite3.Row],
    arm_id: str,
    flt: tuple[str | None, float | None, int | None, str | None],
) -> dict[tuple[str, CellKey], int]:
    """Scenario x cell -> collision flag (replicate mean > 0.5)."""

    acc: dict[tuple[str, CellKey], list[int]] = defaultdict(list)
    for r in rows:
        if r["arm_id"] == arm_id and r["collision"] is not None and _matches(r, *flt):
            acc[(str(r["scenario_id"]), _cell(r))].append(int(r["collision"]))
    return {k: int(sum(v) / len(v) > 0.5) for k, v in acc.items()}


def paired_comparison(
    conn: sqlite3.Connection,
    experiment_id: int,
    arm_a: str,
    arm_b: str,
    latency_mode: str | None = None,
    latency_ms: float | None = None,
    shield: int | None = None,
    cadence: str | None = None,
) -> list[PairedComparison]:
    """Exact McNemar of collision, joining arms on scenario_id + latency cell.

    One result per matched cell (filters restrict which cells are compared).
    ``delta`` = collision rate(a) - collision rate(b).
    """

    rows = _fetch_flights(conn, experiment_id)
    flt = (latency_mode, latency_ms, shield, cadence)
    a_map = _per_scenario_collision(rows, arm_a, flt)
    b_map = _per_scenario_collision(rows, arm_b, flt)
    by_cell: dict[CellKey, list[tuple[int, int]]] = defaultdict(list)
    for (scenario_id, cell), a_val in sorted(a_map.items()):
        if (scenario_id, cell) in b_map:
            by_cell[cell].append((a_val, b_map[(scenario_id, cell)]))
    results: list[PairedComparison] = []
    for cell, pairs in sorted(by_cell.items()):
        res = stats.paired_mcnemar_from_outcomes(
            [p[0] for p in pairs], [p[1] for p in pairs]
        )
        results.append(
            PairedComparison(
                arm_a, arm_b, cell[0], cell[1], cell[2], cell[3], res.n, res.b, res.c,
                (res.b + res.both) / res.n, (res.c + res.both) / res.n,
                res.delta, res.ci_lo, res.ci_hi, res.p_value,
            )
        )  # fmt: skip
    return results


def with_holm(comparisons: Sequence[PairedComparison]) -> list[PairedComparison]:
    """Attach Holm-adjusted p-values treating ``comparisons`` as one family."""

    adjusted = stats.holm_correction([c.p_value for c in comparisons])
    return [
        PairedComparison(**{**c.__dict__, "p_holm": p})
        for c, p in zip(comparisons, adjusted, strict=True)
    ]


def paired_family(
    conn: sqlite3.Connection,
    experiment_id: int,
    reference_arm: str,
    arms: Sequence[str] | None = None,
    latency_mode: str | None = None,
    latency_ms: float | None = None,
    shield: int | None = None,
    cadence: str | None = None,
) -> list[PairedComparison]:
    """Every arm vs ``reference_arm`` (a - reference), Holm-adjusted as one family."""

    family: list[PairedComparison] = []
    for arm in arms if arms is not None else list_arms(conn, experiment_id):
        if arm != reference_arm:
            family.extend(
                paired_comparison(
                    conn, experiment_id, arm, reference_arm,
                    latency_mode, latency_ms, shield, cadence,
                )
            )  # fmt: skip
    return with_holm(family)


# ------------------------------------------------------------------ rho collapse
def outcome_vs_rho(
    conn: sqlite3.Connection,
    experiment_id: int,
    edges: Sequence[float] = RHO_EDGES,
) -> list[RhoBin]:
    """Collision rate binned by rho = L_eff / budget_s, per arm x latency_mode.

    L_eff is the injected ``flights.latency_ms`` when >= 0, else the measured
    ``lat_mean_ms``. Flights with no usable latency are skipped.
    """

    counts: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for r in _fetch_flights(conn, experiment_id):
        if r["collision"] is None:
            continue
        l_eff = stats.effective_latency_ms(float(r["latency_ms"]), r["lat_mean_ms"])
        if l_eff is None or r["budget_s"] <= 0:
            continue
        rho = stats.latency_budget_ratio(l_eff, float(r["budget_s"]))
        for i in range(len(edges) - 1):
            if edges[i] <= rho < edges[i + 1]:
                cell = counts[(str(r["arm_id"]), str(r["latency_mode"]), i)]
                cell[0] += 1
                cell[1] += int(r["collision"])
                break
    out: list[RhoBin] = []
    for (arm, mode, i), (n, k) in sorted(counts.items()):
        rate, lo, hi = _rate(k, n)
        out.append(RhoBin(arm, mode, edges[i], edges[i + 1], n, k, rate, lo, hi))
    return out


# -------------------------------------------------------------------- breakdowns
def _breakdown(
    conn: sqlite3.Connection,
    experiment_id: int,
    column: str,
    cell_filters: tuple[str | None, float | None, int | None, str | None],
) -> list[StratumSummary]:
    groups: dict[tuple[str, CellKey, str], list[int]] = defaultdict(list)
    for r in _fetch_flights(conn, experiment_id):
        if r["collision"] is not None and _matches(r, *cell_filters):
            groups[(str(r["arm_id"]), _cell(r), str(r[column]))].append(
                int(r["collision"])
            )
    out: list[StratumSummary] = []
    for (arm, cell, stratum), vals in sorted(groups.items()):
        rate, lo, hi = _rate(sum(vals), len(vals))
        out.append(
            StratumSummary(arm, cell[0], cell[1], cell[2], cell[3], stratum, len(vals), sum(vals), rate, lo, hi)
        )  # fmt: skip
    return out


def tier_breakdown(
    conn: sqlite3.Connection,
    experiment_id: int,
    latency_mode: str | None = None,
    latency_ms: float | None = None,
    shield: int | None = None,
    cadence: str | None = None,
) -> list[StratumSummary]:
    """Collision rate per arm x cell x scenario tier."""

    return _breakdown(
        conn, experiment_id, "tier", (latency_mode, latency_ms, shield, cadence)
    )


def event_breakdown(
    conn: sqlite3.Connection,
    experiment_id: int,
    latency_mode: str | None = None,
    latency_ms: float | None = None,
    shield: int | None = None,
    cadence: str | None = None,
) -> list[StratumSummary]:
    """Collision rate per arm x cell x adversarial event type."""

    return _breakdown(
        conn, experiment_id, "event_type", (latency_mode, latency_ms, shield, cadence)
    )


# ------------------------------------------------------------ decision latency
def latency_distribution(
    conn: sqlite3.Connection, experiment_id: int, arm_id: str
) -> LatencyDistribution:
    """Decision latencies (ms) for an arm, excluding cache-replayed rows."""

    cur = conn.execute(
        "SELECT d.latency_ms FROM decisions d JOIN flights f ON f.id = d.flight_id "
        "WHERE f.experiment_id = ? AND f.arm_id = ? AND f.status = 'complete' "
        "AND d.from_cache = 0",
        (experiment_id, arm_id),
    )
    values = sorted(float(r["latency_ms"]) for r in cur.fetchall())
    return LatencyDistribution(
        arm_id, len(values), values, stats.percentile_summary(values)
    )


def list_arms(conn: sqlite3.Connection, experiment_id: int) -> list[str]:
    """Arm ids with at least one completed flight."""

    cur = conn.execute(
        "SELECT DISTINCT arm_id FROM flights WHERE experiment_id = ? AND status = 'complete' "
        "ORDER BY arm_id",
        (experiment_id,),
    )
    return [str(r["arm_id"]) for r in cur.fetchall()]


__all__ = [
    "RHO_EDGES",
    "ArmSummary",
    "LatencyDistribution",
    "PairedComparison",
    "RhoBin",
    "StratumSummary",
    "arm_summary",
    "budget_distribution",
    "event_breakdown",
    "latency_distribution",
    "list_arms",
    "outcome_vs_rho",
    "paired_comparison",
    "paired_family",
    "tier_breakdown",
    "with_holm",
]
