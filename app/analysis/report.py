"""Self-contained static HTML report (inline CSS, inline SVG, no external assets)."""

from __future__ import annotations

import json
import math
import platform
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from importlib import metadata
from pathlib import Path

from app.analysis import summary as sm
from app.core.logging import get_logger

logger = get_logger(__name__)

SVG_W = 640
SVG_H = 360
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 58, 16, 16, 46
MAX_ECCDF_POINTS = 250
MIN_FLIGHTS_PER_CELL = 300
PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pydantic",
    "fastapi",
    "pybullet",
    "gym-pybullet-drones",
    "statsmodels",
)
DASHES: tuple[str, ...] = ("", "6 3", "2 3", "8 3 2 3", "1 4", "10 4", "4 2", "3 6")

CSS = """
:root{color-scheme:light dark;--bg:#fafafa;--fg:#1c1c1e;--muted:#5f6368;--card:#fff;--line:#d9dce1;
--warn:#fff4d6;--warn-line:#c99a00;--c0:#1f6feb;--c1:#d9480f;--c2:#2b8a3e;--c3:#9c36b5;
--c4:#0b7285;--c5:#a61e4d;--c6:#5c940d;--c7:#6741d9}
@media (prefers-color-scheme:dark){:root{--bg:#121316;--fg:#e8e8ea;--muted:#9aa0a6;--card:#1b1d21;
--line:#34383f;--warn:#3a3114;--warn-line:#d9a900;--c0:#58a6ff;--c1:#ff8a4c;--c2:#56d364;
--c3:#d2a8ff;--c4:#4fd1e6;--c5:#ff7aa8;--c6:#a9d84e;--c7:#9d8cff}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px;margin:10px 0}
.claims{background:var(--warn);border:2px solid var(--warn-line);border-radius:8px;padding:12px;margin:14px 0}
.claims h3{margin:10px 0 4px;font-size:14px}.claims ul{margin:0;padding-left:20px}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}th{position:sticky;top:0;background:var(--card)}
figure{margin:12px 0}figcaption{color:var(--muted);font-size:12.5px}
svg{max-width:100%;height:auto;display:block}svg text{fill:var(--fg);font:11px system-ui,sans-serif}
.axis{stroke:var(--muted);stroke-width:1;fill:none}.grid{stroke:var(--line);stroke-width:1}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;margin-top:4px}
.sw{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;margin-right:4px}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;
font-size:12px}
"""

Point = tuple[float, float, float | None, float | None]  # x, y, lo, hi


@dataclass(frozen=True)
class Series:
    label: str
    points: list[Point]


# ------------------------------------------------------------------- formatting
def _finite(v: float | None) -> bool:
    return v is not None and math.isfinite(v)


def _fmt(v: float | None, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if v is not None and math.isfinite(v) else "—"


def _pct(v: float | None) -> str:
    return f"{100 * v:.1f}%" if v is not None and math.isfinite(v) else "—"


def _ci(rate: float, lo: float, hi: float) -> str:
    return f"{_pct(rate)} [{_pct(lo)}, {_pct(hi)}]"


def _cell_label(mode: str, ms: float, shield: int, cadence: str) -> str:
    lat = f"{ms:g} ms" if ms >= 0 else "measured"
    return f"{mode} · {lat} · {'shield' if shield else 'raw'} · {cadence}"


def _table(
    headers: Sequence[str], rows: Iterable[Sequence[str]], raw_cols: Sequence[int] = ()
) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{c if i in raw_cols else escape(c)}</td>" for i, c in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


# --------------------------------------------------------------------- SVG core
def _ticks(lo: float, hi: float, log: bool) -> list[float]:
    if log:
        out: list[float] = []
        decade = math.floor(math.log10(lo))
        while 10.0**decade <= hi * 1.0001:
            for m in (1, 2, 5):
                v = m * 10.0**decade
                if lo * 0.9999 <= v <= hi * 1.0001:
                    out.append(v)
            decade += 1
        return out or [lo, hi]
    span = hi - lo
    raw = span / 5.0
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    n = int((hi - start) / step + 1e-9) + 1
    return [round(start + i * step, 10) for i in range(n)]


SYMLOG_C = 20.0  # ms; x axis is log-like but defined at 0 ms (a real latency cell)
SYMLOG_TICKS = (0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)


def _styles(labels: Sequence[str]) -> list[tuple[str, str]]:
    """(css colour, dasharray) per series: one colour per arm, dashed when shield is on."""

    arms: dict[str, int] = {}
    out: list[tuple[str, str]] = []
    for lab in labels:
        arm = lab.split(" · ")[0]
        idx = arms.setdefault(arm, len(arms))
        out.append((f"var(--c{idx % 8})", "6 3" if " · shield" in lab else ""))
    return out


def _legend(labels: Sequence[str]) -> str:
    items = "".join(
        f'<span><span class="sw" style="border-color:{color};'
        f'border-top-style:{"dashed" if dash else "solid"}"></span>{escape(lab)}</span>'
        for (color, dash), lab in zip(_styles(labels), labels, strict=True)
    )
    return f'<div class="legend">{items}</div>'


def _chart(
    series: Sequence[Series],
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_log: bool = False,
    y_log: bool = False,
    y_range: tuple[float, float] | None = (0.0, 1.0),
    caption: str = "",
) -> str:
    """Line chart (optionally with ribbons) as an inline-SVG <figure>."""

    pts = [
        p
        for s in series
        for p in s.points
        if _finite(p[0])
        and _finite(p[1])
        and (not x_log or p[0] >= 0)
        and (not y_log or p[1] > 0)
    ]
    if not pts:
        return f"<figure><figcaption>{escape(title)}: no data.</figcaption></figure>"
    xs = [p[0] for p in pts]
    x_lo, x_hi = min(xs), max(xs)
    if x_lo == x_hi:
        x_lo, x_hi = (x_lo / 2, x_hi * 2) if x_log else (x_lo - 0.5, x_hi + 0.5)
    if y_range is not None:
        y_lo, y_hi = y_range
    else:
        ys = [p[1] for p in pts]
        y_lo, y_hi = min(ys), max(ys)
        if y_lo == y_hi:
            y_lo, y_hi = y_lo / 2, y_hi * 2
    pw, ph = SVG_W - MARGIN_L - MARGIN_R, SVG_H - MARGIN_T - MARGIN_B

    def xt(v: float) -> float:
        return math.log10(1.0 + max(v, 0.0) / SYMLOG_C) if x_log else v

    def sx(v: float) -> float:
        t = (xt(v) - xt(x_lo)) / (xt(x_hi) - xt(x_lo))
        return MARGIN_L + max(0.0, min(1.0, t)) * pw

    def sy(v: float) -> float:
        t = (
            (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))
            if y_log
            else (v - y_lo) / (y_hi - y_lo)
        )
        return MARGIN_T + (1.0 - max(0.0, min(1.0, t))) * ph

    out = [
        (
            f'<svg viewBox="0 0 {SVG_W} {SVG_H}" role="img" aria-label="{escape(title)}" '
            f'xmlns="http://www.w3.org/2000/svg">'
        )
    ]
    x_ticks = (
        [float(t) for t in SYMLOG_TICKS if x_lo <= t <= x_hi]
        if x_log
        else _ticks(x_lo, x_hi, x_log)
    )
    for t in x_ticks:
        x = sx(t)
        out.append(
            f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{MARGIN_T}" y2="{MARGIN_T + ph}"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{MARGIN_T + ph + 14}" text-anchor="middle">{t:g}</text>'
        )
    for t in _ticks(y_lo, y_hi, y_log):
        y = sy(t)
        out.append(
            f'<line class="grid" x1="{MARGIN_L}" x2="{MARGIN_L + pw}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text x="{MARGIN_L - 6}" y="{y + 4:.1f}" text-anchor="end">{t:g}</text>'
        )
    out.append(
        f'<rect class="axis" x="{MARGIN_L}" y="{MARGIN_T}" width="{pw}" height="{ph}"/>'
    )
    out.append(
        f'<text x="{MARGIN_L + pw / 2}" y="{SVG_H - 6}" text-anchor="middle">{escape(x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(13 {MARGIN_T + ph / 2}) rotate(-90)" text-anchor="middle">'
        f"{escape(y_label)}</text>"
    )
    styles = _styles([s.label for s in series])
    for s, (color, dash_css) in zip(series, styles, strict=True):
        sp = sorted(
            p for p in s.points
            if _finite(p[0]) and _finite(p[1]) and (not x_log or p[0] >= 0) and (not y_log or p[1] > 0)
        )  # fmt: skip
        if not sp:
            continue
        band = [p for p in sp if _finite(p[2]) and _finite(p[3])]
        if len(band) >= 2:
            upper = " ".join(
                f"{sx(p[0]):.1f},{sy(max(p[3] or 0.0, y_lo if not y_log else 1e-12)):.1f}"
                for p in band
            )
            lower = " ".join(
                f"{sx(p[0]):.1f},{sy(max(p[2] or 0.0, y_lo if not y_log else 1e-12)):.1f}"
                for p in reversed(band)
            )
            out.append(
                f'<polygon points="{upper} {lower}" fill="{color}" fill-opacity="0.16" stroke="none"/>'
            )
        path = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in sp)
        dash = f' stroke-dasharray="{dash_css}"' if dash_css else ""
        out.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        )
        if len(sp) <= 40:
            out.extend(
                f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="2.8" fill="{color}"/>'
                for p in sp
            )
    out.append("</svg>")
    cap = f"<figcaption>{escape(caption)}</figcaption>" if caption else ""
    return (
        f"<figure><strong>{escape(title)}</strong>{''.join(out)}"
        f"{_legend([s.label for s in series])}{cap}</figure>"
    )


def _mini_bar(rate: float, lo: float, hi: float) -> str:
    """Tiny 0..100% bar with a Wilson whisker for table rows."""

    if not (_finite(rate) and _finite(lo) and _finite(hi)):
        return "—"
    w = 120.0
    return (
        f'<svg width="{w:.0f}" height="12" viewBox="0 0 {w:.0f} 12" role="img" '
        f'aria-label="{_pct(rate)}"><rect x="0" y="3" width="{rate * w:.1f}" height="6" '
        f'fill="var(--c0)" fill-opacity="0.55"/><line x1="{lo * w:.1f}" x2="{hi * w:.1f}" y1="6" y2="6" '
        f'stroke="var(--fg)" stroke-width="1.5"/></svg>'
    )


# --------------------------------------------------------------------- sections
def _executive(rows: list[sm.ArmSummary], budget: dict[str, float]) -> str:
    table = _table(
        [
            "arm", "latency mode", "L (ms)", "shield", "cadence", "n", "collision [95% Wilson]",
            "success [95% Wilson]", "lat mean/p50/p95 (ms)", "time to goal (s)", "path (m)",
            "invalid", "fallback", "reaction (s)",
        ],
        (
            [
                r.arm_id, r.latency_mode, f"{r.latency_ms:g}" if r.latency_ms >= 0 else "measured",
                "on" if r.shield else "off", r.cadence, str(r.n),
                _ci(r.collision_rate, r.collision_lo, r.collision_hi),
                _ci(r.success_rate, r.success_lo, r.success_hi),
                f"{_fmt(r.lat_mean_ms, 1)} / {_fmt(r.lat_p50_ms, 1)} / {_fmt(r.lat_p95_ms, 1)}",
                _fmt(r.time_to_goal, 2), _fmt(r.path_length, 1), _pct(r.invalid_rate),
                _pct(r.fallback_rate), _fmt(r.reaction_time_s, 2),
            ]
            for r in rows
        ),
    )  # fmt: skip
    if budget.get("n"):
        cap = (
            f"Budget B distribution over {int(budget['n'])} scenarios (s): min {_fmt(budget['min'], 2)}, "
            f"median {_fmt(budget['median'], 2)}, mean {_fmt(budget['mean'], 2)}, max {_fmt(budget['max'], 2)}. "
            "Absolute collision rates are only meaningful together with this distribution."
        )
    else:
        cap = "No completed flights."
    return f'<h2 id="executive">Executive summary</h2>{table}<p class="muted">{escape(cap)}</p>'


def _curves(rows: list[sm.ArmSummary]) -> str:
    by_mode: dict[str, dict[str, list[Point]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        x = r.latency_ms if r.latency_ms > 0 else (r.lat_mean_ms or 0.0)
        label = f"{r.arm_id} · {'shield' if r.shield else 'raw'} · {r.cadence}"
        by_mode[r.latency_mode][label].append(
            (x, r.collision_rate, r.collision_lo, r.collision_hi)
        )
    figs = [
        _chart(
            [Series(lab, p) for lab, p in sorted(groups.items())],
            title=f"Collision rate vs decision latency [latency mode: {mode}]",
            x_label="decision latency L (ms, log scale)",
            y_label="collision rate",
            x_log=True,
            caption="Ribbon = 95% Wilson interval. Measured-mode cells are plotted at their mean latency.",
        )
        for mode, groups in sorted(by_mode.items())
    ]
    return '<h2 id="latency-curves">Collision vs latency</h2>' + "".join(figs)


def _rho_section(bins: list[sm.RhoBin]) -> str:
    groups: dict[str, list[Point]] = defaultdict(list)
    for b in bins:
        mid = b.rho_lo + 0.5 if math.isinf(b.rho_hi) else (b.rho_lo + b.rho_hi) / 2
        groups[f"{b.arm_id} [{b.latency_mode}]"].append((mid, b.rate, b.lo, b.hi))
    fig = _chart(
        [Series(lab, p) for lab, p in sorted(groups.items())],
        title="Collision rate vs rho = L_eff / budget B [latency modes as labelled]",
        x_label="rho = L_eff / B (bin midpoint; last bin open-ended)",
        y_label="collision rate",
        caption="If latency acts through the budget, curves from different scenarios collapse onto one.",
    )
    return f'<h2 id="rho">Outcome vs rho</h2>{fig}'


def _strata_section(section_id: str, title: str, rows: list[sm.StratumSummary]) -> str:
    table = _table(
        ["arm", "cell", title.split()[-1], "n", "collisions", "rate [95% Wilson]", ""],
        (
            [
                r.arm_id, _cell_label(r.latency_mode, r.latency_ms, r.shield, r.cadence),
                r.stratum, str(r.n), str(r.collisions), _ci(r.rate, r.lo, r.hi),
                _mini_bar(r.rate, r.lo, r.hi),
            ]
            for r in rows
        ),
        raw_cols=(6,),
    )  # fmt: skip
    return f'<h2 id="{section_id}">{escape(title)}</h2>{table}'


def _eccdf_section(dists: list[sm.LatencyDistribution]) -> str:
    series: list[Series] = []
    summary_rows: list[list[str]] = []
    for d in dists:
        if not d.n:
            continue
        step = max(1, d.n // MAX_ECCDF_POINTS)
        idx = sorted({*range(0, d.n, step), d.n - 1})
        series.append(
            Series(
                d.arm_id, [(d.values_ms[i], (d.n - i) / d.n, None, None) for i in idx]
            )
        )
        q = d.summary.quantiles
        summary_rows.append(
            [d.arm_id, str(d.n)]
            + [f"{_fmt(q[k].value, 1)} [{_fmt(q[k].lo, 1)}, {_fmt(q[k].hi, 1)}]" for k in ("p50", "p90", "p95", "p99")]
            + [_fmt(d.summary.maximum, 1)]
        )  # fmt: skip
    fig = _chart(
        series,
        title="Decision latency ECCDF per provider [recorded decisions.latency_ms, cache hits excluded]",
        x_label="latency (ms, log scale)",
        y_label="P(latency >= x) (log scale)",
        x_log=True,
        y_log=True,
        y_range=None,
        caption="Heavy tails matter: compare p95/p99, not means.",
    )
    table = _table(
        ["arm", "n", "p50 [CI]", "p90 [CI]", "p95 [CI]", "p99 [CI]", "max"],
        summary_rows,
    )
    return f'<h2 id="latency-ccdf">Decision latency distribution</h2>{fig}{table}'


def _paired_section(
    conn: sqlite3.Connection, experiment_id: int, rows: list[sm.ArmSummary]
) -> str:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.arm_id] += r.n
    if len(counts) < 2:
        return '<h2 id="paired">Paired comparisons</h2><p class="muted">Needs at least two arms.</p>'
    reference = min(counts, key=lambda a: (-counts[a], a))
    fam = sm.paired_family(conn, experiment_id, reference)
    table = _table(
        ["arm A", "reference B", "cell", "pairs", "b (A only)", "c (B only)", "A rate", "B rate",
         "delta A-B (pp) [95% CI]", "p (exact McNemar)", "p (Holm)"],
        (
            [
                c.arm_a, c.arm_b, _cell_label(c.latency_mode, c.latency_ms, c.shield, c.cadence),
                str(c.n_pairs), str(c.b), str(c.c), _pct(c.rate_a), _pct(c.rate_b),
                f"{100 * c.delta:+.1f} [{100 * c.delta_lo:+.1f}, {100 * c.delta_hi:+.1f}]",
                f"{c.p_value:.4g}", _fmt(c.p_holm, 4),
            ]
            for c in fam
        ),
    )  # fmt: skip
    note = (
        f"Reference arm: {reference} (most flights). Scenario is the unit of analysis; replicates are "
        f"averaged per scenario. Holm correction spans all {len(fam)} comparisons in this table. "
        "Effect sizes first, p-values second."
    )
    return f'<h2 id="paired">Paired comparisons</h2>{table}<p class="muted">{escape(note)}</p>'


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2, sort_keys=True)
    except json.JSONDecodeError:
        return raw


def _manifest_section(exp: sqlite3.Row) -> str:
    machine = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "n/a",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    versions = _table(["package", "version"], ([k, v] for k, v in _versions().items()))
    return (
        '<h2 id="manifest">Run manifest</h2>'
        f"<p>Experiment <strong>{escape(str(exp['name']))}</strong> (id {exp['id']}), status "
        f"{escape(str(exp['status']))}, created {escape(str(exp['created_at']))}.<br>"
        f"config_sha: <code>{escape(str(exp['config_sha']))}</code></p>"
        f"<h3>Experiment config</h3><pre>{escape(_pretty_json(str(exp['config_json'])))}</pre>"
        f"<h3>Manifest</h3><pre>{escape(_pretty_json(str(exp['manifest_json'])))}</pre>"
        f"<h3>Packages</h3>{versions}"
        f"<h3>Machine</h3><pre>{escape(json.dumps(machine, indent=2))}</pre>"
    )


def _claims_ledger(rows: list[sm.ArmSummary], budget: dict[str, float]) -> str:
    modes = sorted({r.latency_mode for r in rows})
    n_flights = sum(r.n for r in rows)
    small = sum(1 for r in rows if r.n < MIN_FLIGHTS_PER_CELL)
    this_data = [
        (
            f"{n_flights} completed flights over {len(rows)} arm x latency cells; latency modes present: "
            f"{', '.join(modes) or 'none'}."
        ),
        "Collision rate of an arm under the stated latency mode on this scenario set, with Wilson 95% CIs.",
    ]
    if budget.get("n"):
        this_data.append(
            f"Scenario budgets B span {_fmt(budget['min'], 2)}-{_fmt(budget['max'], 2)} s "
            f"(median {_fmt(budget['median'], 2)} s); quote any absolute rate together with this."
        )
    not_data = [
        "Anything about real-world or sim-to-real performance.",
        "Real-network behaviour unless a measured/real-time latency mode is present above.",
    ]
    if small:
        not_data.append(
            f"{small} cell(s) have fewer than {MIN_FLIGHTS_PER_CELL} flights: small effects there are underpowered."
        )
    if "lockstep" in modes:
        not_data.append(
            "Lockstep cells are a latency-free quality ablation, never a latency comparison."
        )
    safe = [
        "Under injected decision latency L, collision rate of arm X on scenario set S is P (Wilson 95% CI).",
        "Measured p50/p95/p99 from this machine, region and time (published figures labelled unmeasured).",
        "Collision results can flip between lockstep and latency-injected accounting for the same policy.",
    ]
    conditional = [
        '"Jev+shield is as safe as FSM+shield": only at tiers where tuned parity holds.',
        '"Better semantic decisions than a fast LLM": format-ablated, n >= 500, latency-matched arm.',
        '"Results hold with real latency": needs >= 90% real-time agreement.',
        '"A faster decision layer is always safer": only between the knee bounds.',
    ]
    never = [
        "Any absolute collision rate without the budget B distribution.",
        "Jev out-reacts PID/MPC; Jev is N x faster than LLMs without an arm definition.",
        "Slow LLMs are safest (a lockstep artefact).",
        "Wall-clock sim_speed including API time as a model property.",
        "Sim-to-real generalisation; reacts to all event types; Jev is first or only.",
    ]

    def ul(items: Sequence[str]) -> str:
        return "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"

    return (
        '<div class="claims" id="claims-ledger"><strong>Claims ledger</strong>'
        f"<h3>This data supports</h3>{ul(this_data)}"
        f"<h3>This data does not support</h3>{ul(not_data)}"
        f"<h3>Safe to claim (project rules)</h3>{ul(safe)}"
        f"<h3>Conditional</h3>{ul(conditional)}"
        f"<h3>Do not claim</h3>{ul(never)}</div>"
    )


# ------------------------------------------------------------------------ entry
CHALLENGE_EVENTS = ("obstacle_static", "obstacle_approach", "intruder_cross", "mission")
SCOPES = {
    "challenge": "certified obstacle scenarios (headline)",
    "controls": "impossible controls (budget B < 0)",
    "all": "all flights",
}


def scoped_copy(conn: sqlite3.Connection, scope: str) -> sqlite3.Connection:
    """In-memory copy of ``conn`` keeping only the flights in ``scope``.

    The summary functions then run unchanged. 'challenge' = certified obstacle
    scenarios with B >= 0; 'controls' = scenarios with B < 0 (unavoidable by design).
    """

    if scope not in SCOPES:
        raise ValueError(f"unknown scope: {scope!r}")
    mem = sqlite3.connect(":memory:")
    mem.row_factory = sqlite3.Row
    conn.backup(mem)
    if scope == "challenge":
        keep = "event_type IN ('obstacle_static','obstacle_approach','intruder_cross','mission') AND budget_s >= 0"
    elif scope == "controls":
        keep = "budget_s < 0"
    else:
        return mem
    mem.execute(
        "DELETE FROM decisions WHERE flight_id IN (SELECT f.id FROM flights f "
        f"JOIN scenarios s ON s.id = f.scenario_id WHERE NOT ({keep}))"
    )
    mem.execute(
        "DELETE FROM flights WHERE scenario_id IN "
        f"(SELECT id FROM scenarios WHERE NOT ({keep}))"
    )
    mem.execute(f"DELETE FROM scenarios WHERE NOT ({keep})")
    return mem


def build_report(
    conn: sqlite3.Connection,
    experiment_id: int,
    out_path: str | Path,
    scope: str = "challenge",
) -> Path:
    """Write one self-contained static HTML report for ``experiment_id`` and ``scope``."""

    conn = scoped_copy(conn, scope)

    exp = conn.execute(
        "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
    ).fetchone()
    if exp is None:
        raise ValueError(f"experiment {experiment_id} not found")
    logger.info("Building report for experiment %s", exp["name"])
    rows = sm.arm_summary(conn, experiment_id)
    budget = sm.budget_distribution(conn, experiment_id)
    dists = [
        sm.latency_distribution(conn, experiment_id, a)
        for a in sm.list_arms(conn, experiment_id)
    ]
    body = "".join(
        [
            f"<h1>Drone decision-latency benchmark: {escape(str(exp['name']))}</h1>",
            (
                f'<p class="muted">Static report; every figure states its latency mode. '
                f"Scope: {escape(SCOPES[scope])}. "
                f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}.</p>"
            ),
            _claims_ledger(rows, budget),
            _executive(rows, budget),
            _curves(rows),
            _rho_section(sm.outcome_vs_rho(conn, experiment_id)),
            _strata_section(
                "tiers",
                "Collision rate by scenario tier",
                sm.tier_breakdown(conn, experiment_id),
            ),
            _strata_section(
                "events",
                "Collision rate by event type",
                sm.event_breakdown(conn, experiment_id),
            ),
            _eccdf_section(dists),
            _paired_section(conn, experiment_id, rows),
            _manifest_section(exp),
        ]
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(str(exp['name']))} report</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("Report written to %s (%d bytes)", out, len(html))
    return out


__all__ = ["Series", "build_report"]
