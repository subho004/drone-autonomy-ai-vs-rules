"""Pure statistics helpers for the benchmark analysis (no DB, no I/O).

Definitions follow docs/research/08-benchmark-methodology-time-accounting-stats.md:
Wilson / Clopper-Pearson intervals, exact McNemar, Wilcoxon, Holm, Kaplan-Meier,
bootstrap percentile summaries and the section 9.3 power table method.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from utils.intervals import Z_95, wilson_interval

DEFAULT_ALPHA = 0.05
DEFAULT_N_BOOT = 2000
MAX_BOOT_CELLS = 20_000_000
MIN_BOOT = 200
DEFAULT_QUANTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)


# --------------------------------------------------------------- proportions
def wilson(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Wilson score interval (delegates to ``utils.intervals``)."""

    z = Z_95 if alpha == DEFAULT_ALPHA else float(sps.norm.ppf(1.0 - alpha / 2.0))
    return wilson_interval(successes, n, z)


def clopper_pearson(
    successes: int, n: int, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial interval; (0, 1) when ``n`` is 0."""

    if n <= 0:
        return 0.0, 1.0
    lo = (
        0.0
        if successes <= 0
        else float(sps.beta.ppf(alpha / 2.0, successes, n - successes + 1))
    )
    hi = (
        1.0
        if successes >= n
        else float(sps.beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    )
    return lo, hi


# ----------------------------------------------------------------- bootstrap
def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` over ``values`` (resampling scenarios)."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    n_boot = max(MIN_BOOT, min(n_boot, MAX_BOOT_CELLS // arr.size))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([stat(arr[row]) for row in idx], dtype=float)
    lo, hi = np.percentile(boots, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Quantile:
    """One percentile with its bootstrap CI."""

    q: float
    value: float
    lo: float
    hi: float


@dataclass(frozen=True)
class PercentileSummary:
    n: int
    mean: float
    maximum: float
    quantiles: dict[str, Quantile]


def percentile_summary(
    values: Sequence[float],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> PercentileSummary:
    """p50/p90/p95/p99 (keys ``p50`` ...) with vectorised bootstrap CIs."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return PercentileSummary(0, math.nan, math.nan, {})
    n_boot = max(MIN_BOOT, min(n_boot, MAX_BOOT_CELLS // arr.size))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = np.percentile(arr[idx], list(quantiles), axis=1)
    point = np.percentile(arr, list(quantiles))
    out: dict[str, Quantile] = {}
    for i, q in enumerate(quantiles):
        lo, hi = np.percentile(boot[i], [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
        out[f"p{q:g}"] = Quantile(q, float(point[i]), float(lo), float(hi))
    return PercentileSummary(int(arr.size), float(arr.mean()), float(arr.max()), out)


def ecdf(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Empirical CDF: sorted x and P(X <= x)."""

    xs = np.sort(np.asarray(values, dtype=float))
    n = xs.size
    return xs.tolist(), ((np.arange(n) + 1.0) / n).tolist() if n else []


def eccdf(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Empirical complementary CDF: sorted x and P(X >= x) (never 0, log-safe)."""

    xs = np.sort(np.asarray(values, dtype=float))
    n = xs.size
    return xs.tolist(), ((n - np.arange(n)) / n).tolist() if n else []


# ------------------------------------------------------------------- testing
def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts ``b`` and ``c``."""

    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * sps.binom.cdf(min(b, c), n, 0.5)))


@dataclass(frozen=True)
class McNemarResult:
    """Paired binary comparison; ``delta`` = rate(a) - rate(b)."""

    n: int
    b: int  # a=1, b=0
    c: int  # a=0, b=1
    both: int
    neither: int
    p_value: float
    delta: float
    ci_lo: float
    ci_hi: float


def paired_mcnemar_from_outcomes(
    a: Sequence[int], b: Sequence[int], alpha: float = DEFAULT_ALPHA
) -> McNemarResult:
    """Exact McNemar on aligned 0/1 outcomes with a paired Wald CI on the difference."""

    if len(a) != len(b):
        raise ValueError("outcome sequences must be aligned and equal length")
    n = len(a)
    n_b = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    n_c = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    both = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    if n == 0:
        return McNemarResult(0, 0, 0, 0, 0, 1.0, math.nan, math.nan, math.nan)
    delta = (n_b - n_c) / n
    se = math.sqrt(max(n_b + n_c - (n_b - n_c) ** 2 / n, 0.0)) / n
    z = float(sps.norm.ppf(1.0 - alpha / 2.0))
    return McNemarResult(
        n, n_b, n_c, both, n - n_b - n_c - both, exact_mcnemar(n_b, n_c),
        delta, delta - z * se, delta + z * se,
    )  # fmt: skip


@dataclass(frozen=True)
class WilcoxonResult:
    n_nonzero: int
    statistic: float
    p_value: float


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank on paired samples (all-zero diffs -> p = 1)."""

    if len(x) != len(y):
        raise ValueError("samples must be paired and equal length")
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    nonzero = int(np.count_nonzero(diff))
    if nonzero == 0:
        return WilcoxonResult(0, 0.0, 1.0)
    res = sps.wilcoxon(diff[diff != 0.0])
    return WilcoxonResult(nonzero, float(res.statistic), float(res.pvalue))


def holm_correction(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in input order."""

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# -------------------------------------------------------------- Kaplan-Meier
@dataclass(frozen=True)
class KMCurve:
    """Kaplan-Meier survival estimate at each distinct event time."""

    times: list[float]
    survival: list[float]
    at_risk: list[int]
    events: list[int]
    std_err: list[float]  # Greenwood


def kaplan_meier(durations: Sequence[float], events: Sequence[int]) -> KMCurve:
    """Kaplan-Meier estimator; ``events[i]`` is 1 if observed, 0 if censored."""

    if len(durations) != len(events):
        raise ValueError("durations and events must have equal length")
    d_arr = np.asarray(durations, dtype=float)
    e_arr = np.asarray(events, dtype=int)
    times: list[float] = []
    surv: list[float] = []
    risk: list[int] = []
    deaths: list[int] = []
    se: list[float] = []
    s = 1.0
    green = 0.0
    for t in np.unique(d_arr[e_arr > 0]):
        at_risk = int(np.sum(d_arr >= t))
        d = int(np.sum((d_arr == t) & (e_arr > 0)))
        s *= 1.0 - d / at_risk
        if at_risk > d:
            green += d / (at_risk * (at_risk - d))
        times.append(float(t))
        surv.append(s)
        risk.append(at_risk)
        deaths.append(d)
        se.append(s * math.sqrt(green))
    return KMCurve(times, surv, risk, deaths, se)


# --------------------------------------------------------------------- power
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size (arcsine transform) for two proportions."""

    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def sample_size_unpaired(
    p_base: float, delta: float, alpha: float = DEFAULT_ALPHA, power: float = 0.8
) -> int:
    """Flights per arm, two-sided, equal arms, normal approx on Cohen's h."""

    h = abs(cohens_h(p_base, p_base + delta))
    z = float(sps.norm.ppf(1.0 - alpha / 2.0) + sps.norm.ppf(power))
    return math.ceil(2.0 * (z / h) ** 2)


def discordant_probabilities(
    p_a: float, p_b: float, rho: float
) -> tuple[float, float] | None:
    """(P(a=1,b=0), P(a=0,b=1)) given marginals and phi correlation, or None if infeasible."""

    sd = math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    p11 = p_a * p_b + rho * sd
    p10, p01 = p_a - p11, p_b - p11
    if p10 < -1e-12 or p01 < -1e-12 or p11 < -1e-12:
        return None
    return max(p10, 0.0), max(p01, 0.0)


def mcnemar_power(
    n: int, p10: float, p01: float, alpha: float = DEFAULT_ALPHA
) -> float:
    """Exact power of the two-sided exact McNemar test with ``n`` paired scenarios."""

    p_disc = p10 + p01
    if n <= 0 or p_disc <= 0.0:
        return 0.0
    q = p10 / p_disc
    spread = 7.0 * math.sqrt(n * p_disc * (1 - p_disc)) + 5.0
    lo = max(1, int(n * p_disc - spread))
    hi = min(n, int(n * p_disc + spread))
    ms = np.arange(lo, hi + 1)
    pm = sps.binom.pmf(ms, n, p_disc)
    total = 0.0
    for m, w in zip(ms.tolist(), pm.tolist(), strict=True):
        # largest k with 2 * P(Bin(m, .5) <= k) <= alpha  -> reject if min(b, m-b) <= k
        k = int(sps.binom.ppf(alpha / 2.0, m, 0.5))
        while k >= 0 and 2.0 * sps.binom.cdf(k, m, 0.5) > alpha:
            k -= 1
        if k < 0:
            continue
        reject = sps.binom.cdf(k, m, q) + sps.binom.sf(m - k - 1, m, q)
        total += w * float(reject)
    return min(total, 1.0)


def sample_size_paired(
    p_base: float,
    delta: float,
    rho: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = 0.8,
    n_max: int = 20_000,
) -> int | None:
    """Smallest paired n (scenarios = flights per arm); None if rho is infeasible."""

    probs = discordant_probabilities(p_base, p_base + delta, rho)
    if probs is None:
        return None
    p10, p01 = probs
    if p10 + p01 <= 0.0:
        return None
    lo, hi = 1, 8
    while mcnemar_power(hi, p10, p01, alpha) < power:
        lo, hi = hi, hi * 2
        if hi > n_max:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mcnemar_power(mid, p10, p01, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


# ----------------------------------------------------------- rho = L_eff / B
def effective_latency_ms(injected_ms: float, measured_ms: float | None) -> float | None:
    """Injected latency when declared (>= 0), else the measured mean (or None)."""

    if injected_ms >= 0.0:
        return injected_ms
    if measured_ms is None or math.isnan(measured_ms):
        return None
    return measured_ms


def latency_budget_ratio(latency_ms: float, budget_s: float) -> float:
    """rho = L_eff / B with L in milliseconds and the reaction budget B in seconds."""

    return (latency_ms / 1000.0) / budget_s


__all__ = [
    "DEFAULT_ALPHA",
    "KMCurve",
    "McNemarResult",
    "PercentileSummary",
    "Quantile",
    "WilcoxonResult",
    "bootstrap_ci",
    "clopper_pearson",
    "cohens_h",
    "discordant_probabilities",
    "eccdf",
    "ecdf",
    "effective_latency_ms",
    "exact_mcnemar",
    "holm_correction",
    "kaplan_meier",
    "latency_budget_ratio",
    "mcnemar_power",
    "paired_mcnemar_from_outcomes",
    "percentile_summary",
    "sample_size_paired",
    "sample_size_unpaired",
    "wilcoxon_signed_rank",
    "wilson",
]
