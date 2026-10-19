"""Generate a FAKE but realistic demo experiment (SQLite + Parquet trajectories).

Everything here is synthetic: a kinematic drone flies +x toward a goal and an
obstacle appears ahead. Outcomes follow the budget-CDF logic of the plan:
a flight collides iff its effective reaction latency exceeds the scenario's
reaction budget (times lognormal noise), plus a per-arm "wrong action" rate.

Latency bookkeeping: ``decisions.latency_ms`` is the arm's own (measured)
latency; the effective delay used to place ``apply_tick`` is
``injected + measured`` in fixed mode and ``measured`` in sampled mode.

Usage (no CLI args; edit the constants):

    uv run python scripts/seed_demo_db.py
    DB_PATH=data/demo.db uv run uvicorn main:app --port 8765
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.core.logging import get_logger
from app.domain.actions import Action
from app.domain.trajectory import TRAJ_COLUMNS, TRAJ_RATE_HZ
from app.harness.store import connect

logger = get_logger("seed_demo_db")

# ----------------------------------------------------------------- constants
DEMO_DB_PATH = ROOT_DIR / "data" / "demo.db"
DEMO_DATA_DIR = ROOT_DIR / "data" / "demo"
EXPERIMENT_NAME = "demo-latency-sweep"
N_SCENARIOS = 300
N_TRAJECTORY_SCENARIOS = 12  # scenarios (all configs) that also get Parquet files
QUEUED_FRACTION = 0.04  # tail of the queue left unfinished so progress looks live
N_RUNNING = 4
N_FAILED = 2
RNG_SEED = 20261017
SECONDS_BETWEEN_FINISHES = 1.2

LATENCY_CELLS: tuple[tuple[str, float], ...] = (
    ("fixed", 25.0),
    ("fixed", 100.0),
    ("fixed", 300.0),
    ("fixed", 1000.0),
    ("sampled", -1.0),
)

CRUISE_MS = 3.0  # m/s
GOAL_X = 40.0
START_Z = 1.0
T_MAX = 20.0
DODGE_S = 0.5
BRAKE_S = 0.5  # braking time baked into the budget definition
DRONE_RADIUS = 0.1
HZ = TRAJ_RATE_HZ


@dataclass(frozen=True)
class ArmDef:
    arm_id: str
    kind: str
    provider: str
    model: str
    median_ms: float  # median of the arm's own decision latency
    sigma: float  # lognormal sigma of that latency
    period_s: float  # minimum spacing between observations
    wrong_rate: float  # P(arm ignores the obstacle regardless of latency)
    invalid_rate: float
    cost_usd: float
    shield_variant: bool = False


ARMS: tuple[ArmDef, ...] = (
    ArmDef(
        "pid_reactive", "classical", "local", "pid-avoid", 4, 0.2, 0.25, 0.06, 0.0, 0.0
    ),
    ArmDef("fsm_reflex", "classical", "local", "fsm-v1", 2, 0.2, 0.25, 0.04, 0.0, 0.0),
    ArmDef("jev", "jev", "jev", "jev-1", 70, 0.35, 0.25, 0.03, 0.01, 0.0002, True),
    ArmDef(
        "llm_fast",
        "llm",
        "openrouter",
        "gemini-flash",
        420,
        0.5,
        0.5,
        0.09,
        0.03,
        0.0004,
        True,
    ),
    ArmDef(
        "llm_slow", "llm", "openai", "gpt-large", 1900, 0.55, 0.5, 0.08, 0.04, 0.004
    ),
    ArmDef("oracle", "oracle", "local", "oracle", 1, 0.1, 0.25, 0.0, 0.0, 0.0),
)

EVENT_TYPES = (
    ("obstacle_approach", 0.55),
    ("intruder_cross", 0.25),
    ("obstacle_static", 0.20),
)
SHIELD_REACT_S = 0.06
SHIELD_FAIL_RATE = 0.10
TIMEOUT_RATE = 0.015
IMPOSSIBLE_RATE = 0.05  # control tier TX: budget < 0


@dataclass(frozen=True)
class ScenarioDef:
    index: int
    id: str
    seed: int
    event_type: str
    tier: str
    budget_s: float
    t_event: float
    radius: float
    x_obs: float
    vy_obs: float


def tier_for(budget_s: float) -> str:
    if budget_s < 0:
        return "TX"
    for tier, floor in (
        ("T0", 3.0),
        ("T1", 1.0),
        ("T2", 0.3),
        ("T3", 0.1),
        ("T4", 0.03),
    ):
        if budget_s >= floor:
            return tier
    return "T5"


def make_scenarios(n: int, rng: np.random.Generator) -> list[ScenarioDef]:
    names = [e for e, _ in EVENT_TYPES]
    probs = [p for _, p in EVENT_TYPES]
    out: list[ScenarioDef] = []
    for i in range(n):
        if rng.random() < IMPOSSIBLE_RATE:
            budget = -float(rng.uniform(0.05, 0.3))
        else:
            budget = float(math.exp(rng.uniform(math.log(0.01), math.log(5.0))))
        event = str(rng.choice(names, p=probs))
        t_event = float(rng.uniform(2.0, 4.0))
        radius = float(rng.uniform(0.4, 0.9))
        gap = CRUISE_MS * (budget + BRAKE_S)  # surface distance at onset
        x_obs = CRUISE_MS * t_event + gap + radius
        vy = float(rng.choice([-0.6, 0.6])) if event == "intruder_cross" else 0.0
        out.append(
            ScenarioDef(
                i,
                f"scn-{i:04d}",
                int(rng.integers(1, 2**31)),
                event,
                tier_for(budget),
                budget,
                t_event,
                radius,
                x_obs,
                vy,
            )
        )
    return out


def scenario_spec(sc: ScenarioDef) -> dict[str, object]:
    return {
        "start": [0.0, 0.0, START_Z],
        "goal": [GOAL_X, 0.0, START_Z],
        "ctrl_hz": HZ,
        "cruise_ms": CRUISE_MS,
        "budget_s": sc.budget_s,
        "event": {
            "type": sc.event_type,
            "t": sc.t_event,
            "obstacle": {
                "x": sc.x_obs,
                "y": 0.0,
                "z": START_Z,
                "radius": sc.radius,
                "vy": sc.vy_obs,
            },
        },
    }


# ------------------------------------------------------------- kinematics
def side_of(sc: ScenarioDef) -> float:
    """+1 dodges left (+y), -1 right; away from a crossing intruder."""

    if sc.vy_obs != 0:
        return -1.0 if sc.vy_obs > 0 else 1.0
    return 1.0 if sc.index % 2 == 0 else -1.0


def _smooth(u: np.ndarray) -> np.ndarray:
    c = np.clip(u, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


@dataclass
class Sim:
    cols: dict[str, np.ndarray]
    outcome: str
    t_end: float
    t_pass: float
    path_length: float
    min_clearance: float
    time_to_goal: float | None


def simulate(
    sc: ScenarioDef,
    t_app: float,
    dodge_full: bool,
    dodges: bool,
    shield_from: float | None,
    timeout: bool,
) -> Sim:
    """Vectorised kinematic flight; geometry (not a flag) decides the outcome."""

    n = int(T_MAX * HZ) + 1
    t = np.arange(n) / HZ
    side = side_of(sc)
    x = CRUISE_MS * t - (0.0 if not dodges else 0.9 * _smooth((t - t_app) / 1.5))
    if timeout:
        x = np.minimum(x, sc.x_obs + sc.radius + 3.0)
    t_pass = (sc.x_obs + sc.radius + 0.8) / CRUISE_MS
    target = side * (sc.radius + 0.45 if dodge_full else 0.8 * sc.radius)
    y = (target if dodges else 0.0) * _smooth((t - t_app) / DODGE_S)
    y = y * (1.0 - _smooth((t - t_pass) / 1.5))

    active = t >= sc.t_event
    t_cross = sc.x_obs / CRUISE_MS
    ox = np.where(active, sc.x_obs, np.nan)
    oy = np.where(active, sc.vy_obs * (t - t_cross), np.nan)
    clearance = np.hypot(x - ox, y - oy) - sc.radius - DRONE_RADIUS

    end = n - 1
    outcome = "timeout" if timeout else "success"
    hit = np.flatnonzero(active & (clearance <= 0.0))
    goal = np.flatnonzero(x >= GOAL_X)
    if hit.size:
        end, outcome = int(hit[0]), "collision"
    elif goal.size and not timeout:
        end = int(goal[0])
    sl = slice(0, end + 1)

    z = START_Z * _smooth(t / 1.5) + 0.02 * np.sin(2 * t)
    ry = np.interp(t + 0.25, t, y)
    in_avoid = (
        np.logical_and(t >= t_app, t < t_pass) if dodges else np.zeros(n, dtype=bool)
    )
    avoid = Action.AVOID_LEFT if side > 0 else Action.AVOID_RIGHT
    action = np.where(in_avoid, int(avoid), int(Action.FORWARD))
    if timeout:
        action = np.where(x >= sc.x_obs + sc.radius + 3.0, int(Action.HOVER), action)
    shield = np.zeros(n, dtype=np.int32)
    if shield_from is not None:
        win = (t >= shield_from) & (t < min(shield_from + 0.6, t_pass))
        shield[win] = 1
        action = np.where(win, int(avoid), action)
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    cols = {
        "tick": np.arange(n, dtype=np.int32),
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": np.gradient(z, t),
        "rx": x + 0.6,
        "ry": ry,
        "rz": np.full(n, START_Z),
        "action": action.astype(np.int32),
        "shield": shield,
        "ox": ox,
        "oy": oy,
        "oz": np.where(active, START_Z, np.nan),
        "orad": np.where(active, sc.radius, np.nan),
        "clearance": clearance,
        "battery": 100.0 - 0.2 * t,
        "wind_x": 0.05 * np.sin(0.7 * t),
        "wind_y": 0.05 * np.cos(0.5 * t),
    }
    cols = {k: v[sl] for k, v in cols.items()}
    steps = np.hypot(np.diff(cols["x"]), np.diff(cols["y"]))
    clear = clearance[sl]
    finite = clear[np.isfinite(clear)]
    t_goal = float(cols["t"][-1]) if outcome == "success" else None
    return Sim(
        cols,
        outcome,
        float(cols["t"][-1]),
        t_pass,
        float(steps.sum()),
        float(finite.min()) if finite.size else float("nan"),
        t_goal,
    )


def _arrow_cast(name: str, values: np.ndarray) -> np.ndarray:
    if name == "t":
        return values.astype(np.float64)
    return values.astype(np.int32 if values.dtype.kind == "i" else np.float32)


def write_trajectory(path: Path, sim: Sim) -> None:
    assert tuple(sim.cols) == TRAJ_COLUMNS, (
        "trajectory columns drifted from the contract"
    )
    arrays = {
        k: pa.array(
            v.astype(
                np.float64
                if k == "t"
                else v.dtype
                if v.dtype.kind == "i"
                else np.float32
            )
        )
        for k, v in sim.cols.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path, compression="zstd")


# --------------------------------------------------------------- decisions
def draw_effective_ms(
    arm: ArmDef, mode: str, injected_ms: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Return ``(measured_ms, effective_ms)`` with multiplicative noise."""

    measured = float(arm.median_ms * math.exp(rng.normal(0.0, arm.sigma)))
    base = measured + (injected_ms if mode == "fixed" else 0.0)
    return measured, base * float(math.exp(rng.normal(0.0, 0.15)))


@dataclass(frozen=True)
class DecisionPlan:
    """What the geometry assumed: when the avoid action took effect and from whom."""

    t_app: float
    model_eff_ms: float
    measured_ms: float
    dodges: bool
    shield_from: float | None


def build_decisions(
    flight_id: int,
    arm: ArmDef,
    mode: str,
    injected_ms: float,
    sc: ScenarioDef,
    sim: Sim,
    plan: DecisionPlan,
    rng: np.random.Generator,
) -> list[tuple[object, ...]]:
    """One row per decision request: periodic ones plus the obstacle-onset decision."""

    avoid = int(Action.AVOID_LEFT if side_of(sc) > 0 else Action.AVOID_RIGHT)
    raw: list[
        tuple[float, float, float, str, bool]
    ] = []  # t_obs, measured, eff, kind, valid
    t, event_added = 0.0, False
    while t < sim.t_end:
        if not event_added and t >= sc.t_event:
            raw.append((sc.t_event, plan.measured_ms, plan.model_eff_ms, "event", True))
            event_added = True
            t = max(sc.t_event + arm.period_s, sc.t_event + plan.model_eff_ms / 1000.0)
            continue
        measured, eff = draw_effective_ms(arm, mode, injected_ms, rng)
        raw.append((t, measured, eff, "normal", bool(rng.random() >= arm.invalid_rate)))
        t = max(t + arm.period_s, t + eff / 1000.0)
    if plan.shield_from is not None:
        raw.append(
            (
                sc.t_event,
                SHIELD_REACT_S * 1000.0,
                SHIELD_REACT_S * 1000.0,
                "shield",
                True,
            )
        )
    raw.sort(key=lambda r: (r[0], r[3] != "event"))

    rows: list[tuple[object, ...]] = []
    last_applied = int(Action.FORWARD)
    llm = arm.kind == "llm"
    for seq, (t_obs, measured, eff, kind, valid) in enumerate(raw):
        t_apply = t_obs + eff / 1000.0
        dodging = plan.dodges and plan.t_app <= t_apply < sim.t_pass
        applied = avoid if dodging or (kind == "shield") else int(Action.FORWARD)
        action: int | None = applied
        source = "model"
        if kind == "shield":
            source, action = "shield", None
        elif not valid:
            source, action, applied = "fallback", None, last_applied
        last_applied = applied
        late = kind == "event" and (not plan.dodges or eff / 1000.0 > sc.budget_s)
        rows.append(
            (
                flight_id,
                seq,
                round(t_obs * HZ),
                round(t_apply * HZ),
                t_obs,
                t_apply,
                action,
                applied,
                source,
                int(valid),
                measured,
                max(0, round(eff / 1000.0 * HZ)),
                float(rng.uniform(0.5, 0.99)) if valid and kind != "shield" else None,
                None,
                0 if late else 1,
                134 if llm and kind != "shield" else None,
                1 if llm and kind != "shield" else None,
                arm.cost_usd if kind != "shield" else 0.0,
                0,
                None,
            )
        )
    return rows


# ------------------------------------------------------------------ build
@dataclass
class ConfigDef:
    arm: ArmDef
    mode: str
    latency_ms: float
    shield: int


def configs() -> Iterator[ConfigDef]:
    for arm in ARMS:
        for shield in (0, 1) if arm.shield_variant else (0,):
            for mode, ms in LATENCY_CELLS:
                yield ConfigDef(arm, mode, ms, shield)


FLIGHT_COLUMNS = (
    "id",
    "experiment_id",
    "scenario_id",
    "arm_id",
    "latency_mode",
    "latency_ms",
    "shield",
    "cadence",
    "rep",
    "pool",
    "status",
    "attempts",
    "started_at",
    "finished_at",
    "outcome",
    "success",
    "collision",
    "t_end",
    "time_to_goal",
    "path_length",
    "min_clearance",
    "reaction_time_s",
    "recovery_time_s",
    "n_decisions",
    "lat_mean_ms",
    "lat_p95_ms",
    "invalid_rate",
    "fallback_rate",
    "shield_overrides",
    "action_switches",
    "cost_usd",
    "sim_wall_s",
    "api_wall_s",
    "trajectory_uri",
    "worker",
)
DECISION_COLUMNS = (
    "flight_id",
    "seq",
    "obs_tick",
    "apply_tick",
    "t_obs",
    "t_apply",
    "action",
    "applied_action",
    "source",
    "valid",
    "latency_ms",
    "latency_ticks",
    "confidence",
    "probs_json",
    "acceptable",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "from_cache",
    "raw",
)


def uri_for(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def build_demo_db(
    db_path: Path = DEMO_DB_PATH,
    data_dir: Path = DEMO_DATA_DIR,
    n_scenarios: int = N_SCENARIOS,
    n_trajectory_scenarios: int = N_TRAJECTORY_SCENARIOS,
    queued_fraction: float = QUEUED_FRACTION,
    seed: int = RNG_SEED,
) -> dict[str, int]:
    """Create the demo DB (replacing any existing file at ``db_path``)."""

    if db_path.name == "drone_jev.db":
        raise ValueError("refusing to overwrite the real experiment database")
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    shutil.rmtree(data_dir, ignore_errors=True)

    master = np.random.default_rng(seed)
    scenarios = make_scenarios(n_scenarios, master)
    conn = connect(db_path)
    conn.execute("BEGIN IMMEDIATE")
    exp_id = int(
        conn.execute(
            "INSERT INTO experiments(name, created_at, status, config_json, config_sha, manifest_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                EXPERIMENT_NAME,
                datetime.now(UTC).isoformat(timespec="seconds"),
                "running",
                json.dumps({"demo": True, "cells": [list(c) for c in LATENCY_CELLS]}),
                "demo0000",
                json.dumps(
                    {
                        "fake": True,
                        "note": "synthetic data from scripts/seed_demo_db.py",
                    }
                ),
            ),
        ).lastrowid
        or 0
    )
    conn.executemany(
        "INSERT INTO arms(id, kind, provider, model, config_json) VALUES (?, ?, ?, ?, '{}')",
        [(a.arm_id, a.kind, a.provider, a.model) for a in ARMS],
    )
    conn.executemany(
        "INSERT INTO scenarios(id, suite, seed, event_type, tier, budget_s, certified, spec_json) "
        "VALUES (?, 'demo', ?, ?, ?, ?, 1, ?)",
        [
            (
                s.id,
                s.seed,
                s.event_type,
                s.tier,
                s.budget_s,
                json.dumps(scenario_spec(s)),
            )
            for s in scenarios
        ],
    )

    cfgs = list(configs())
    total = len(scenarios) * len(cfgs)
    n_unfinished = int(total * queued_fraction)
    first_unfinished = total - n_unfinished
    now = datetime.now(UTC)
    flight_rows: list[tuple[object, ...]] = []
    decision_rows: list[tuple[object, ...]] = []
    flight_id = 0
    n_traj = 0
    for sc in scenarios:
        for ci, cfg in enumerate(cfgs):
            flight_id += 1
            if flight_id > first_unfinished:
                status = (
                    "running"
                    if flight_id <= first_unfinished + N_RUNNING
                    else ("failed" if flight_id > total - N_FAILED else "queued")
                )
                flight_rows.append(
                    _unfinished_row(exp_id, flight_id, sc, cfg, status, now)
                )
                continue
            rng = np.random.default_rng([seed, sc.index, ci])
            frow, drows, has_traj = _fly(
                exp_id,
                flight_id,
                sc,
                cfg,
                rng,
                data_dir,
                sc.index < n_trajectory_scenarios,
                now,
                (first_unfinished - flight_id) * SECONDS_BETWEEN_FINISHES,
            )
            flight_rows.append(frow)
            decision_rows.extend(drows)
            n_traj += int(has_traj)

    placeholders = ",".join("?" * len(FLIGHT_COLUMNS))
    conn.executemany(
        f"INSERT INTO flights(experiment_id_placeholder) VALUES ({placeholders})".replace(
            "experiment_id_placeholder", ",".join(FLIGHT_COLUMNS)
        ),
        flight_rows,
    )
    conn.executemany(
        f"INSERT INTO decisions({','.join(DECISION_COLUMNS)}) VALUES ({','.join('?' * len(DECISION_COLUMNS))})",
        decision_rows,
    )
    status = "running" if n_unfinished else "complete"
    conn.execute("UPDATE experiments SET status=? WHERE id=?", (status, exp_id))
    conn.execute("COMMIT")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return {
        "experiment_id": exp_id,
        "flights": total,
        "decisions": len(decision_rows),
        "trajectories": n_traj,
        "unfinished": n_unfinished,
    }


def _unfinished_row(
    exp_id: int,
    flight_id: int,
    sc: ScenarioDef,
    cfg: ConfigDef,
    status: str,
    now: datetime,
) -> tuple[object, ...]:
    started = now.isoformat(timespec="seconds") if status != "queued" else None
    finished = now.isoformat(timespec="seconds") if status == "failed" else None
    row: dict[str, object] = {
        "id": flight_id,
        "experiment_id": exp_id,
        "scenario_id": sc.id,
        "arm_id": cfg.arm.arm_id,
        "latency_mode": cfg.mode,
        "latency_ms": cfg.latency_ms,
        "shield": cfg.shield,
        "cadence": "sequential",
        "rep": 0,
        "pool": "api" if cfg.arm.kind == "llm" else "cpu",
        "status": status,
        "attempts": 0 if status == "queued" else 1,
        "started_at": started,
        "finished_at": finished,
    }
    return tuple(row.get(c) for c in FLIGHT_COLUMNS)


def _fly(
    exp_id: int,
    flight_id: int,
    sc: ScenarioDef,
    cfg: ConfigDef,
    rng: np.random.Generator,
    data_dir: Path,
    write_traj: bool,
    now: datetime,
    seconds_ago: float,
) -> tuple[tuple[object, ...], list[tuple[object, ...]], bool]:
    arm = cfg.arm
    measured, eff_ms = draw_effective_ms(arm, cfg.mode, cfg.latency_ms, rng)
    model_eff_ms = eff_ms
    shield_saves = bool(cfg.shield and rng.random() >= SHIELD_FAIL_RATE)
    if shield_saves:
        eff_ms = min(eff_ms, SHIELD_REACT_S * 1000.0)
    wrong = bool(rng.random() < arm.wrong_rate) and not shield_saves
    eff_s = eff_ms / 1000.0
    dodge_full = eff_s <= sc.budget_s
    shield_from = sc.t_event + SHIELD_REACT_S if shield_saves else None
    sim = simulate(
        sc,
        sc.t_event + eff_s,
        dodge_full,
        not wrong,
        shield_from,
        bool(rng.random() < TIMEOUT_RATE),
    )
    plan = DecisionPlan(
        sc.t_event + eff_s, model_eff_ms, measured, not wrong, shield_from
    )
    decisions = build_decisions(
        flight_id, arm, cfg.mode, cfg.latency_ms, sc, sim, plan, rng
    )
    uri: str | None = None
    if write_traj:
        path = data_dir / "trajectories" / str(exp_id) / f"{flight_id}.parquet"
        write_trajectory(path, sim)
        uri = uri_for(path)
    lats = np.array([d[10] for d in decisions], dtype=float)
    valid = np.array([d[9] for d in decisions], dtype=float)
    sources = [d[8] for d in decisions]
    applied = [int(str(d[7])) for d in decisions]
    collision = int(sim.outcome == "collision")
    finished = now - timedelta(seconds=seconds_ago)
    wall = sim.t_end / 120.0 + float(lats.sum()) / 1000.0 * (
        0.0 if arm.kind != "llm" else 1.0
    )
    row: dict[str, object] = {
        "id": flight_id,
        "experiment_id": exp_id,
        "scenario_id": sc.id,
        "arm_id": arm.arm_id,
        "latency_mode": cfg.mode,
        "latency_ms": cfg.latency_ms,
        "shield": cfg.shield,
        "cadence": "sequential",
        "rep": 0,
        "pool": "api" if arm.kind == "llm" else "cpu",
        "status": "complete",
        "attempts": 1,
        "started_at": (finished - timedelta(seconds=wall)).isoformat(
            timespec="seconds"
        ),
        "finished_at": finished.isoformat(timespec="seconds"),
        "outcome": sim.outcome,
        "success": int(sim.outcome == "success"),
        "collision": collision,
        "t_end": sim.t_end,
        "time_to_goal": sim.time_to_goal,
        "path_length": sim.path_length,
        "min_clearance": sim.min_clearance,
        "reaction_time_s": eff_s,
        "recovery_time_s": None if collision else sim.t_pass + 1.5 - sc.t_event,
        "n_decisions": len(decisions),
        "lat_mean_ms": float(lats.mean()),
        "lat_p95_ms": float(np.percentile(lats, 95)),
        "invalid_rate": float(1.0 - valid.mean()),
        "fallback_rate": sources.count("fallback") / len(sources),
        "shield_overrides": sources.count("shield"),
        "action_switches": int(
            sum(1 for pair in itertools.pairwise(applied) if pair[0] != pair[1])
        ),
        "cost_usd": sum(float(str(d[17])) for d in decisions),
        "sim_wall_s": sim.t_end / 120.0,
        "api_wall_s": wall if arm.kind == "llm" else 0.0,
        "trajectory_uri": uri,
        "worker": "demo",
    }
    return tuple(row[c] for c in FLIGHT_COLUMNS), decisions, uri is not None


def main() -> None:
    start = time.perf_counter()
    logger.info("Seeding demo DB at %s", DEMO_DB_PATH)
    summary = build_demo_db()
    logger.info("Seeded %s in %.1fs", summary, time.perf_counter() - start)


if __name__ == "__main__":
    main()
