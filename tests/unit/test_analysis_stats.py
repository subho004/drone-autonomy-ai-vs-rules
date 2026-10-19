from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint

from app.analysis import stats


def test_wilson_matches_statsmodels() -> None:
    lo, hi = stats.wilson(7, 20)
    s_lo, s_hi = proportion_confint(7, 20, alpha=0.05, method="wilson")
    assert lo == pytest.approx(s_lo, abs=1e-9)
    assert hi == pytest.approx(s_hi, abs=1e-9)
    assert stats.wilson(0, 0) == (0.0, 1.0)


def test_clopper_pearson_matches_statsmodels() -> None:
    for k, n in [(0, 10), (3, 10), (10, 10), (50, 200)]:
        lo, hi = stats.clopper_pearson(k, n)
        s_lo, s_hi = proportion_confint(k, n, alpha=0.05, method="beta")
        assert lo == pytest.approx(0.0 if k == 0 else s_lo, abs=1e-9)
        assert hi == pytest.approx(1.0 if k == n else s_hi, abs=1e-9)


@pytest.mark.parametrize(("b", "c"), [(0, 0), (3, 3), (2, 9), (15, 4), (0, 6)])
def test_exact_mcnemar_matches_binomtest(b: int, c: int) -> None:
    expected = 1.0 if b + c == 0 else sps.binomtest(b, b + c, 0.5).pvalue
    assert stats.exact_mcnemar(b, c) == pytest.approx(expected)


def test_paired_mcnemar_from_outcomes() -> None:
    a = [1, 1, 1, 0, 0, 1, 0, 0]
    b = [0, 0, 1, 0, 1, 0, 0, 0]
    res = stats.paired_mcnemar_from_outcomes(a, b)
    assert (res.n, res.b, res.c, res.both, res.neither) == (8, 3, 1, 1, 3)
    assert res.delta == pytest.approx(0.25)
    assert res.ci_lo < res.delta < res.ci_hi
    assert res.p_value == pytest.approx(stats.exact_mcnemar(3, 1))
    with pytest.raises(ValueError):
        stats.paired_mcnemar_from_outcomes([1], [1, 0])


def test_holm_matches_statsmodels() -> None:
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    expected = multipletests(p, method="holm")[1]
    assert stats.holm_correction(p) == pytest.approx(expected.tolist())


def test_wilcoxon_wrapper() -> None:
    x = [1.1, 2.3, 3.9, 4.2, 5.8, 6.1, 7.7]
    y = [1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0]
    res = stats.wilcoxon_signed_rank(x, y)
    assert res.p_value == pytest.approx(sps.wilcoxon(np.subtract(x, y)).pvalue)
    assert stats.wilcoxon_signed_rank([1, 2], [1, 2]).p_value == 1.0


def test_kaplan_meier_hand_example() -> None:
    km = stats.kaplan_meier([1, 2, 3, 4, 5], [1, 0, 1, 0, 1])
    assert km.times == [1.0, 3.0, 5.0]
    assert km.survival == pytest.approx([0.8, 0.8 * 2 / 3, 0.0])
    assert km.at_risk == [5, 3, 1]


def test_bootstrap_and_percentiles_are_deterministic() -> None:
    values = list(np.random.default_rng(1).lognormal(5, 0.5, 400))
    lo, hi = stats.bootstrap_ci(values, np.median, n_boot=300, seed=3)
    assert lo < float(np.median(values)) < hi
    assert stats.bootstrap_ci(values, np.median, n_boot=300, seed=3) == (lo, hi)
    summ = stats.percentile_summary(values, n_boot=300, seed=3)
    q95 = summ.quantiles["p95"]
    assert q95.lo <= q95.value <= q95.hi
    assert q95.value == pytest.approx(float(np.percentile(values, 95)))
    assert set(summ.quantiles) == {"p50", "p90", "p95", "p99"}
    assert math.isnan(stats.bootstrap_ci([])[0])


def test_ecdf_eccdf() -> None:
    xs, ps = stats.ecdf([3, 1, 2, 4])
    assert xs == [1, 2, 3, 4]
    assert ps == [0.25, 0.5, 0.75, 1.0]
    _, cs = stats.eccdf([3, 1, 2, 4])
    assert cs == [1.0, 0.75, 0.5, 0.25]


@pytest.mark.parametrize(
    ("p", "d"), [(0.05, 0.05), (0.1, 0.1), (0.3, 0.1), (0.3, 0.05), (0.5, 0.2)]
)
def test_unpaired_power_matches_statsmodels_and_doc_table(p: float, d: float) -> None:
    ours = stats.sample_size_unpaired(p, d)
    h = abs(stats.cohens_h(p, p + d))
    ref = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=1.0)
    assert abs(ours - ref) <= 1.0
    table = {(0.05, 0.05): 424, (0.3, 0.1): 356, (0.3, 0.05): 1376, (0.5, 0.2): 93}
    if (p, d) in table:
        assert abs(ours - table[(p, d)]) <= 2


@pytest.mark.parametrize(
    ("p", "d", "rho", "expected"),
    [
        (0.05, 0.05, 0.6, 197),
        (0.3, 0.1, 0.6, 161),
        (0.3, 0.1, 0.3, 270),
        (0.1, 0.1, 0.6, 91),
    ],
)
def test_paired_power_matches_doc_table(
    p: float, d: float, rho: float, expected: int
) -> None:
    got = stats.sample_size_paired(p, d, rho)
    assert got is not None
    assert abs(got - expected) <= max(3, 0.03 * expected)


def test_paired_power_infeasible_rho() -> None:
    assert stats.sample_size_paired(0.05, 0.05, 0.8) is None


def test_mcnemar_power_monotone_in_n() -> None:
    p10, p01 = 0.02, 0.10
    assert stats.mcnemar_power(30, p10, p01) < stats.mcnemar_power(300, p10, p01)
    assert stats.mcnemar_power(300, p10, p01) > 0.9


def test_rho_helpers() -> None:
    assert stats.latency_budget_ratio(500.0, 2.0) == pytest.approx(0.25)
    assert stats.effective_latency_ms(100.0, 250.0) == 100.0
    assert stats.effective_latency_ms(-1.0, 250.0) == 250.0
    assert stats.effective_latency_ms(-1.0, None) is None
