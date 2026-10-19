"""SQLite schema (STRICT tables, WAL) shared by workers and the API."""

from __future__ import annotations

SCHEMA_VERSION = 1

PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=30000",
    "PRAGMA foreign_keys=ON",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-65536",
)

DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'created',
        config_json TEXT NOT NULL,
        config_sha TEXT NOT NULL,
        manifest_json TEXT NOT NULL DEFAULT '{}'
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS scenarios (
        id TEXT PRIMARY KEY,
        suite TEXT NOT NULL,
        seed INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        tier TEXT NOT NULL,
        budget_s REAL NOT NULL,
        certified INTEGER NOT NULL DEFAULT 0,
        spec_json TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS arms (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT '',
        config_json TEXT NOT NULL DEFAULT '{}'
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS flights (
        id INTEGER PRIMARY KEY,
        experiment_id INTEGER NOT NULL REFERENCES experiments(id),
        scenario_id TEXT NOT NULL REFERENCES scenarios(id),
        arm_id TEXT NOT NULL REFERENCES arms(id),
        latency_mode TEXT NOT NULL,
        latency_ms REAL NOT NULL DEFAULT -1,
        shield INTEGER NOT NULL DEFAULT 0,
        cadence TEXT NOT NULL DEFAULT 'sequential',
        rep INTEGER NOT NULL DEFAULT 0,
        pool TEXT NOT NULL DEFAULT 'cpu',
        priority INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'queued',
        attempts INTEGER NOT NULL DEFAULT 0,
        lease_until REAL,
        worker TEXT,
        started_at TEXT,
        finished_at TEXT,
        outcome TEXT,
        success INTEGER,
        collision INTEGER,
        t_end REAL,
        time_to_goal REAL,
        path_length REAL,
        min_clearance REAL,
        reaction_time_s REAL,
        recovery_time_s REAL,
        n_decisions INTEGER,
        lat_mean_ms REAL,
        lat_p95_ms REAL,
        invalid_rate REAL,
        fallback_rate REAL,
        shield_overrides INTEGER,
        action_switches INTEGER,
        cost_usd REAL,
        sim_wall_s REAL,
        api_wall_s REAL,
        trajectory_uri TEXT,
        error TEXT,
        stations_scanned INTEGER,
        stations_total INTEGER,
        events_json TEXT,
        task_json TEXT,
        UNIQUE (experiment_id, scenario_id, arm_id, latency_mode, latency_ms, shield, cadence, rep)
    ) STRICT
    """,
    "CREATE INDEX IF NOT EXISTS ix_flights_queue ON flights (pool, status, priority DESC, id) WHERE status = 'queued'",
    "CREATE INDEX IF NOT EXISTS ix_flights_lease ON flights (lease_until) WHERE status = 'running'",
    "CREATE INDEX IF NOT EXISTS ix_flights_exp_arm ON flights (experiment_id, arm_id)",
    "CREATE INDEX IF NOT EXISTS ix_flights_scenario ON flights (scenario_id)",
    """
    CREATE TABLE IF NOT EXISTS decisions (
        flight_id INTEGER NOT NULL REFERENCES flights(id),
        seq INTEGER NOT NULL,
        obs_tick INTEGER NOT NULL,
        apply_tick INTEGER NOT NULL,
        t_obs REAL NOT NULL,
        t_apply REAL NOT NULL,
        action INTEGER,
        applied_action INTEGER NOT NULL,
        source TEXT NOT NULL,
        valid INTEGER NOT NULL,
        latency_ms REAL NOT NULL,
        latency_ticks INTEGER NOT NULL,
        confidence REAL,
        probs_json TEXT,
        acceptable INTEGER,
        tokens_in INTEGER,
        tokens_out INTEGER,
        cost_usd REAL NOT NULL DEFAULT 0,
        from_cache INTEGER NOT NULL DEFAULT 0,
        raw TEXT,
        PRIMARY KEY (flight_id, seq)
    ) STRICT, WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS model_cache (
        key TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        response_json TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS budget_ledger (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        calls INTEGER NOT NULL,
        cost_usd REAL NOT NULL
    ) STRICT
    """,
    """
    CREATE VIEW IF NOT EXISTS v_flight_summary AS
    SELECT f.experiment_id, f.arm_id, f.latency_mode, f.latency_ms, f.shield, f.cadence,
           s.event_type, s.tier,
           COUNT(*) AS n,
           SUM(f.collision) AS collisions,
           SUM(f.success) AS successes,
           AVG(f.collision) AS collision_rate,
           AVG(f.success) AS success_rate,
           AVG(f.lat_mean_ms) AS lat_mean_ms,
           AVG(f.path_length) AS path_length,
           AVG(f.time_to_goal) AS time_to_goal
    FROM flights f JOIN scenarios s ON s.id = f.scenario_id
    WHERE f.status = 'complete'
    GROUP BY f.experiment_id, f.arm_id, f.latency_mode, f.latency_ms, f.shield, f.cadence,
             s.event_type, s.tier
    """,
)

# Columns added after the first release: applied to existing databases by ``connect``.
MIGRATION_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("flights", "stations_scanned", "INTEGER"),
    ("flights", "stations_total", "INTEGER"),
    ("flights", "events_json", "TEXT"),
    ("flights", "task_json", "TEXT"),
)

__all__ = ["DDL", "MIGRATION_COLUMNS", "PRAGMAS", "SCHEMA_VERSION"]
