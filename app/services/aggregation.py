"""Pure aggregation helpers over per-flight metric rows."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from utils.intervals import wilson_interval

Row = Mapping[str, Any]

CELL_FIELDS = ("arm_id", "latency_mode", "latency_ms", "shield", "cadence")
ECDF_POINTS = 60


def cell_key(row: Row) -> tuple[Any, ...]:
    return tuple(row[f] for f in CELL_FIELDS)


def cell_dict(key: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(CELL_FIELDS, key, strict=True))


def group_by[K](rows: Iterable[Row], key: Callable[[Row], K]) -> dict[K, list[Row]]:
    groups: dict[K, list[Row]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return dict(groups)


def mean_of(rows: Sequence[Row], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) is not None]
    return sum(values) / len(values) if values else None


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 1]) of pre-sorted values."""

    if not sorted_values:
        return math.nan
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def proportion(rows: Sequence[Row], field: str) -> tuple[int, int, float, float, float]:
    """Return ``(n, k, rate, ci_low, ci_high)`` for a 0/1 field (None counts as 0)."""

    n = len(rows)
    k = sum(1 for r in rows if r.get(field))
    low, high = wilson_interval(k, n)
    return n, k, (k / n if n else 0.0), low, high


def result_row(key: Sequence[Any], rows: Sequence[Row]) -> dict[str, Any]:
    n, collisions, c_rate, c_low, c_high = proportion(rows, "collision")
    _, successes, s_rate, s_low, s_high = proportion(rows, "success")
    return {
        **cell_dict(key),
        "n": n,
        "collisions": collisions,
        "collision_rate": c_rate,
        "collision_ci_low": c_low,
        "collision_ci_high": c_high,
        "successes": successes,
        "success_rate": s_rate,
        "success_ci_low": s_low,
        "success_ci_high": s_high,
        "lat_mean_ms": mean_of(rows, "lat_mean_ms"),
        "lat_p95_ms": mean_of(rows, "lat_p95_ms"),
        "path_length": mean_of(rows, "path_length"),
        "time_to_goal": mean_of(rows, "time_to_goal"),
        "invalid_rate": mean_of(rows, "invalid_rate"),
    }


def results(rows: Sequence[Row]) -> list[dict[str, Any]]:
    grouped = group_by(rows, cell_key)
    return [result_row(k, grouped[k]) for k in sorted(grouped, key=_sort_key)]


def _sort_key(key: Sequence[Any]) -> tuple[str, str, float, int, str]:
    arm, mode, ms, shield, cadence = key
    return (str(arm), str(mode), float(ms), int(shield), str(cadence))


def curve_series(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Collision rate vs latency per (arm, mode, shield, cadence).

    Fixed cells plot at their injected latency; sampled and measured cells at the mean
    wall-clock decision latency (a measured model is a single point on the curve).
    """

    usable = [r for r in rows if r["latency_mode"] in ("fixed", "sampled", "measured")]
    series_groups = group_by(
        usable, lambda r: (r["arm_id"], r["latency_mode"], r["shield"], r["cadence"])
    )
    out: list[dict[str, Any]] = []
    for (arm, mode, shield, cadence), series_rows in sorted(series_groups.items()):
        points: list[dict[str, Any]] = []
        for latency_ms, pt_rows in group_by(
            series_rows, lambda r: r["latency_ms"]
        ).items():
            measured = latency_ms < 0
            x = mean_of(pt_rows, "lat_mean_ms") if measured else latency_ms
            if x is None:
                continue
            n, k, rate, low, high = proportion(pt_rows, "collision")
            points.append(
                {
                    "latency_ms": float(x),
                    "latency_measured": measured,
                    "n": n,
                    "collisions": k,
                    "rate": rate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        points.sort(key=lambda p: p["latency_ms"])
        label = f"{arm} · {mode}" + (" · shield" if shield else "")
        out.append(
            {
                "arm_id": arm,
                "latency_mode": mode,
                "shield": shield,
                "cadence": cadence,
                "label": label,
                "points": points,
            }
        )
    return out


def breakdown(rows: Sequence[Row], group_field: str) -> list[dict[str, Any]]:
    """Collision rate per cell and ``group_field`` (tier or event_type)."""

    grouped = group_by(rows, lambda r: (*cell_key(r), r[group_field]))
    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda k: (*_sort_key(k[:5]), str(k[5]))):
        n, k, rate, low, high = proportion(grouped[key], "collision")
        out.append(
            {
                **cell_dict(key[:5]),
                "group": str(key[5]),
                "n": n,
                "collisions": k,
                "rate": rate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return out


def latency_distribution(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Percentiles and a log-spaced ECDF per (provider, model)."""

    out: list[dict[str, Any]] = []
    grouped = group_by(rows, lambda r: (r["provider"], r["model"]))
    for (provider, model), group in sorted(grouped.items()):
        values = sorted(float(r["latency_ms"]) for r in group)
        out.append(
            {
                "provider": provider,
                "model": model,
                "label": f"{provider}/{model}" if model else str(provider),
                "n": len(values),
                "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95),
                "p99_ms": percentile(values, 0.99),
                "max_ms": values[-1],
                "ecdf": _ecdf(values),
            }
        )
    return out


def _ecdf(values: Sequence[float]) -> list[dict[str, float]]:
    lo, hi = values[0], values[-1]
    if hi <= lo:
        return [{"ms": lo, "p": 1.0}]
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    points: list[dict[str, float]] = []
    j = 0
    n = len(values)
    for i in range(ECDF_POINTS):
        x = 10 ** (log_lo + (log_hi - log_lo) * i / (ECDF_POINTS - 1))
        while j < n and values[j] <= x * (1 + 1e-12):
            j += 1
        points.append({"ms": x, "p": j / n})
    return points
