"""Synchronous SQLite access for worker processes (stdlib ``sqlite3``).

One connection per process. A flight's decisions and metrics are written in
one ``BEGIN IMMEDIATE`` transaction, so partial flights never appear.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.harness.schema import DDL, MIGRATION_COLUMNS, PRAGMAS, SCHEMA_VERSION

logger = get_logger(__name__)

LEASE_SECONDS = 120.0
MAX_ATTEMPTS = 3


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a tuned connection (WAL, busy timeout) and ensure the schema."""

    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    for pragma in PRAGMAS:
        conn.execute(pragma)
    for stmt in DDL:
        conn.execute(stmt)
    _migrate(conn)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created (idempotent)."""

    for table, column, ctype in MIGRATION_COLUMNS:
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")


class Store:
    """Thin repository over one SQLite connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = connect(path)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ setup
    def upsert_experiment(
        self, name: str, config: Mapping[str, Any], config_sha: str
    ) -> int:
        self.conn.execute(
            "INSERT INTO experiments(name, created_at, config_json, config_sha) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "config_json=excluded.config_json, config_sha=excluded.config_sha",
            (name, utc_now(), json.dumps(config, sort_keys=True), config_sha),
        )
        row = self.conn.execute(
            "SELECT id FROM experiments WHERE name=?", (name,)
        ).fetchone()
        return int(row["id"])

    def upsert_scenarios(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.executemany(
                "INSERT OR IGNORE INTO scenarios(id, suite, seed, event_type, tier, budget_s, "
                "certified, spec_json) VALUES (:id, :suite, :seed, :event_type, :tier, "
                ":budget_s, :certified, :spec_json)",
                list(rows),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def upsert_arm(
        self,
        arm_id: str,
        kind: str,
        provider: str,
        model: str,
        config: Mapping[str, Any],
    ) -> None:
        self.conn.execute(
            "INSERT INTO arms(id, kind, provider, model, config_json) VALUES (?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET config_json=excluded.config_json",
            (arm_id, kind, provider, model, json.dumps(config, sort_keys=True)),
        )

    def enqueue_flights(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Insert queued flights; duplicates (idempotency key) are ignored."""

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self.conn.executemany(
                "INSERT OR IGNORE INTO flights(experiment_id, scenario_id, arm_id, latency_mode, "
                "latency_ms, shield, cadence, rep, pool, priority) VALUES (:experiment_id, "
                ":scenario_id, :arm_id, :latency_mode, :latency_ms, :shield, :cadence, :rep, "
                ":pool, :priority)",
                list(rows),
            )
            self.conn.execute("COMMIT")
            return cur.rowcount
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ queue
    def claim_flight(self, pool: str, worker: str) -> sqlite3.Row | None:
        """Atomically claim one queued flight, reclaiming expired leases first."""

        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "UPDATE flights SET status='queued', lease_until=NULL WHERE status='running' "
                "AND lease_until < ? AND attempts < ?",
                (now, MAX_ATTEMPTS),
            )
            self.conn.execute(
                "UPDATE flights SET status='failed', error='lease expired too often' "
                "WHERE status='running' AND lease_until < ? AND attempts >= ?",
                (now, MAX_ATTEMPTS),
            )
            row = self.conn.execute(
                "SELECT id FROM flights WHERE status='queued' AND pool=? "
                "ORDER BY priority DESC, id LIMIT 1",
                (pool,),
            ).fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            self.conn.execute(
                "UPDATE flights SET status='running', worker=?, lease_until=?, "
                "attempts=attempts+1, started_at=? WHERE id=?",
                (worker, now + LEASE_SECONDS, utc_now(), row["id"]),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return self.conn.execute(
            "SELECT * FROM flights WHERE id=?", (row["id"],)
        ).fetchone()

    def heartbeat(self, flight_id: int) -> None:
        self.conn.execute(
            "UPDATE flights SET lease_until=? WHERE id=? AND status='running'",
            (time.time() + LEASE_SECONDS, flight_id),
        )

    def complete_flight(
        self,
        flight_id: int,
        metrics: Mapping[str, Any],
        decisions: Sequence[Mapping[str, Any]],
    ) -> None:
        """Write decisions + metrics and mark complete in a single transaction."""

        cols = ", ".join(f"{k}=:{k}" for k in metrics)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("DELETE FROM decisions WHERE flight_id=?", (flight_id,))
            if decisions:
                self.conn.executemany(
                    "INSERT INTO decisions(flight_id, seq, obs_tick, apply_tick, t_obs, t_apply, "
                    "action, applied_action, source, valid, latency_ms, latency_ticks, confidence, "
                    "probs_json, acceptable, tokens_in, tokens_out, cost_usd, from_cache, raw) "
                    "VALUES (:flight_id, :seq, :obs_tick, :apply_tick, :t_obs, :t_apply, :action, "
                    ":applied_action, :source, :valid, :latency_ms, :latency_ticks, :confidence, "
                    ":probs_json, :acceptable, :tokens_in, :tokens_out, :cost_usd, :from_cache, :raw)",
                    [{**d, "flight_id": flight_id} for d in decisions],
                )
            self.conn.execute(
                f"UPDATE flights SET {cols}, status='complete', finished_at=:_fin, "
                "lease_until=NULL WHERE id=:_id",
                {**metrics, "_fin": utc_now(), "_id": flight_id},
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def fail_flight(self, flight_id: int, error: str) -> None:
        self.conn.execute(
            "UPDATE flights SET status='failed', error=?, finished_at=?, lease_until=NULL WHERE id=?",
            (error[:500], utc_now(), flight_id),
        )

    # ---------------------------------------------------------------- reading
    def progress(self, experiment_id: int) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM flights WHERE experiment_id=? GROUP BY status",
            (experiment_id,),
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def scenario_spec(self, scenario_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT spec_json FROM scenarios WHERE id=?", (scenario_id,)
        ).fetchone()
        if row is None:
            raise KeyError(scenario_id)
        spec: dict[str, Any] = json.loads(row["spec_json"])
        return spec

    def arm_config(self, arm_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT kind, provider, model, config_json FROM arms WHERE id=?", (arm_id,)
        ).fetchone()
        if row is None:
            raise KeyError(arm_id)
        cfg: dict[str, Any] = json.loads(row["config_json"])
        cfg.update(kind=row["kind"], provider=row["provider"], model=row["model"])
        return cfg


__all__ = ["LEASE_SECONDS", "MAX_ATTEMPTS", "Store", "connect", "utc_now"]
