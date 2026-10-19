# 10 - SQLite schema, write throughput, FastAPI template integration

Author: research agent 10. Date of measurements/access: 2026-09-30.
Hardware: Apple M4 (10 cores: 4 P + 6 E), macOS (Darwin 27.0.0), APFS SSD. Python 3.14.2 (uv-managed, arm64), stdlib sqlite3 = SQLite 3.50.4.
Scratch (not in repo): `/private/tmp/claude-501/-Users-subhajithait-Documents-testing-drone-jev/0ec4827c-5206-473e-b48d-298a2822fdb4/scratchpad/10/` (`spike.py`, `rep.py`, `sweep.py`, `traj_size.py`, `wr.py`, `ana.py`, `sa.py`, `ddl.sql`, `sketch/`).

---

## 1. Executive summary

1. **SQLite is not the bottleneck.** 8 processes writing decision rows into ONE WAL database sustain ~660k-870k rows/s in batches of 500-5000 (measured), ~41k rows/s even with one commit per row. The real workload needs ~10^2-10^4 rows/s (section 7). Choose the multi-writer strategy for *robustness and resumability*, not speed.
2. **Recommended strategy: shared WAL DB, each worker has its own sync `sqlite3` connection, one `BEGIN IMMEDIATE` transaction per completed flight** (decisions + metrics + `status='complete'` committed atomically), `busy_timeout=30000`, retry on `OperationalError`. Raw lock errors only appear with `busy_timeout=0` (measured ~2300 BUSY per ~3200 commits); with a 30 s timeout: 0 errors in every run.
3. **Do not store trajectories as SQLite rows.** Measured for a 30 s / 48 Hz flight (1440 samples x 13 float32): SQLite rows 204 KB/flight, Parquet zstd+byte-stream-split 54 KB, npz 70 KB. Store trajectories as per-chunk Parquet (or per-flight npz) files, referenced by `flights.trajectory_uri`.
4. **DB size for 10k flights: ~0.3-0.5 GB for the SQLite file** (300 decisions/flight, WITHOUT ROWID narrow table) plus ~0.15-0.7 GB trajectory files. 10k flights is tiny for SQLite.
5. **Python 3.14.2 dependency check (measured, scratch venv): everything installs from wheels.** aiosqlite 0.22.1, SQLAlchemy 2.1.1 (2.0.50 also resolvable), duckdb 1.5.6, pyarrow 25.0.1, pandas 3.0.6, alembic 1.20.0, numpy 2.5.3, greenlet 3.5.6. **Gotcha: SQLAlchemy 2.1.1 does not pull in greenlet; `sqlalchemy.ext.asyncio` raises ImportError until you install `sqlalchemy[asyncio]`.**
6. **The template needs real changes for SQLite** (section 9): `app/db/database.py` is an empty file, `database_url` defaults to `""`, there is no lifespan, no models/repositories/services/schemas dirs, and three dependencies (asyncpg, psycopg2-binary, boto3) are dead weight (grep shows zero imports of them, or of `requests`/`python-dotenv`/`python-multipart`).

---

## 2. What the repo actually contains (read in full)

Tracked files: `main.py`, `app/api/v1/{health,router}.py`, `app/core/{config,exceptions,logging}.py`, `app/db/database.py`, `utils/{response,save_file}.py`, `tests/{__init__,conftest,test_health}.py`, `pytest.ini`, `mypy.ini`, `requirements.txt`, `.env.sample`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.pre-commit-config.yaml`, `.python-version` (3.14.2), `README.md`, `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/project-instructions.mdc`, `.github/copilot-instructions.md`, `.vscode/settings.json`. `docs/` is untracked.

### Real conventions (verified by reading code)

| Area | Reality |
|---|---|
| Instruction files | `AGENTS.md`, `CLAUDE.md`, `.cursor/rules` all just point to `.github/copilot-instructions.md` (single source of truth). |
| App factory | `main.py`: `create_app()` + module-level `app = create_app()`; `configure_logging()` runs at import; CORS middleware; `register_exception_handlers(app)`; one `api_router`. **No lifespan**, no DB engine wiring. |
| Packages | `app/`, `app/api`, `app/core`, `app/db`, `utils/` have **no `__init__.py`** (namespace packages); `tests/` has one. mypy needs `explicit_package_bases` (already set in `mypy.ini`, `python_version = 3.14`, `ignore_missing_imports = true`). |
| Config | `app/core/config.py`: `Settings(BaseSettings)`, `env_file=".env"`, `extra="ignore"`, `cors_origins` uses `NoDecode` + before-validator, `database_url: str = ""`. Single `settings = Settings()` instance. |
| Errors | `app/core/exceptions.py`: `AppError(StarletteHTTPException)` + 8 typed subclasses; handlers for `StarletteHTTPException`, `RequestValidationError`, `Exception`. Uses `status.HTTP_422_UNPROCESSABLE_CONTENT`. |
| Responses | `utils/response.py`: `APIResponse[T]` (PEP 695 generic syntax), `success_response()`/`error_response()` return `JSONResponse` (not typed models). Health route returns `JSONResponse`. |
| Logging | `app/core/logging.py`: plain-text formatter, `configure_logging`, `get_logger`. |
| DB | `app/db/database.py` is **0 bytes**. No engine/session/Base, no models, no repositories, no services, no schemas, no Alembic. |
| Tests | `tests/conftest.py` imports `main.app` and builds an `httpx.AsyncClient(ASGITransport)` fixture; `pytest.ini`: `asyncio_mode = auto`. Only `test_health.py`. (Guide: tests optional unless asked.) |
| Tooling | `.pre-commit-config.yaml`: ruff (`--fix`), ruff-format, mypy mirror with only `pydantic`, `pydantic-settings` as additional deps. **No `pyproject.toml`, `ruff.toml`**; dependencies live in `requirements.txt`. |
| Deps | requests, python-dotenv, fastapi, uvicorn, pydantic, pydantic-settings, SQLAlchemy>=2.0.50, asyncpg, psycopg2-binary, boto3, python-multipart, ruff, mypy, pytest, pytest-asyncio, httpx, pre-commit. **No Alembic, no aiosqlite, no greenlet.** |
| Docker | `Dockerfile` (python:3.14.2-slim + uv), `docker-compose.yml` (api service, healthcheck), `.dockerignore`, README "Docker" section. |

### Doc-vs-code drifts (flag, do not silently fix; repo rules say flag smells)

1. Guide says `logging.py (JSON)`; code is plain text (latest commit e6c11f9 changed it); `main.py` comment and module docstring still say "structured (JSON)".
2. Guide layer table says `routes/`; actual dir is `app/api/v1/`.
3. Guide examples use `ApiResponse[...]`, `PaginatedResponse[...]` and "shared pagination utility"; repo has `APIResponse` (different casing) and **no pagination utility or `PaginatedResponse`**. Must be created before any list endpoint (decision/flight lists are large).
4. Guide says isort rules enforced by Ruff; with no ruff config file, Ruff runs default rules (E4/E7/E9/F) and **does not enforce import sorting (`I`)**. `utils/save_file.py` uses `Optional`/`Union` and a bare `except Exception` that re-raises (allowed) but it is not PEP 604 style.
5. Guide says UUID primary keys by default. For this schema I deviate (integer PKs) - justified in section 3.
6. Guide says `from app.core.config import settings` is the only way; fine, continue.
7. `response.py.__all__` omits `success_response`/`error_response`/`to_http_response` although they are the mandated API.

### Changes required for SQLite (minimum)

- `requirements.txt`: remove `asyncpg`, `psycopg2-binary`, `boto3`; add `aiosqlite`, `greenlet` (via `sqlalchemy[asyncio]`), `alembic`; later `duckdb`, `pyarrow`, `pandas`/`polars` for analysis (an `[analysis]` extra or separate requirements file, since workers do not need them). Candidates for removal (YAGNI, unused): `requests`, `python-dotenv` (pydantic-settings reads `.env` itself), `python-multipart` (no form/file endpoints yet). Keep them out of the worker env to keep it small.
- `app/core/config.py`: `database_url: str = "sqlite+aiosqlite:///data/jev.db"`, plus `data_dir: Path = Path("data")`, `sqlite_busy_timeout_ms: int = 30000`. Add `data/` to `.gitignore` (currently only `.venv`, `.env`, caches).
- `app/db/database.py` (empty now): async engine + `async_sessionmaker` + `get_session` dependency + `Base`; connect-event pragma hook (section 4). Put `Base` in `app/db/base.py` (shown below) and keep `database.py` for engine/session.
- `main.py`: add a `lifespan` that disposes the engine (and optionally runs `PRAGMA optimize`); this is a new pattern, not a refactor of existing code.
- `tests/conftest.py`: override `get_session` with a per-test **temp-file** DB (`tmp_path`), not `:memory:` (WAL does not apply to in-memory DBs, and multi-connection in-memory needs StaticPool). `app.dependency_overrides`, as the guide requires. Teardown via fixture `yield` (guide rule).
- mypy pre-commit hook: add `sqlalchemy`, `aiosqlite` to `additional_dependencies`, otherwise model types resolve to `Any` (because `ignore_missing_imports = true` silently hides it). Verified: the model sketch below passes `mypy --strict` when SQLAlchemy is installed.
- Dockerfile/compose/.dockerignore/README Docker section: local-only/no-Docker requirement means they are dead, not harmful. Decision for 14: leave untouched or delete in one explicit commit (they reference `requirements.txt`, which will still work).
- A worker CLI needs the `scripts/` convention from the guide (constants at top, `sys.path` append, log start/end/elapsed).

---

## 3. Schema design

### Principles
- Two-tier storage: **SQLite = metadata, decisions, metrics, job queue, caches; files (Parquet/npz) = dense trajectory samples.**
- `STRICT` tables (SQLite >= 3.37; here 3.50.4). STRICT only allows `INT/INTEGER/REAL/TEXT/BLOB/ANY` ([SQLite STRICT docs](https://www.sqlite.org/stricttables.html), accessed 2026-09-30).
- Integer PKs, not UUIDs (documented deviation from the guide's "UUID PKs unless schema differs"): single-file local DB, UUID text would be 36 bytes per FK in the hot table, and ids never leave the machine. Expose stable human keys (`experiments.name`, `scenarios.scenario_key` hash) instead.
- `decisions`, `model_calls`, `event_outcomes`, `scenario_events` are `WITHOUT ROWID` with composite PKs. **Measured (1M sequential rows, 8 columns): rowid + UNIQUE(flight_id, seq) = 46.1 MB / 1.19 s; WITHOUT ROWID PK = 30.2 MB / 1.07 s.** 35% smaller and no slower.
- Time: `*_tick` = simulation ticks (integer; ctrl_hz stored on flight), `*_ns` = `perf_counter_ns` deltas, `*_at` = unix epoch milliseconds (integer; avoids TEXT timestamps).
- Idempotency key for resumability: `UNIQUE(scenario_id, arm_id, latency_cell, rep)` on `flights` (NULL would defeat uniqueness in SQLite, hence `latency_cell TEXT NOT NULL DEFAULT 'native'`).
- Actions as small integers with an `actions` lookup table (mirror of the Python `IntEnum`).

### Full DDL (validated: executed on SQLite 3.50.4 with `foreign_keys=ON`; FK, UNIQUE idempotency, claim query, view and `integrity_check` all pass)

```sql
-- Connection pragmas are NOT in this file (per-connection); see section 4.

CREATE TABLE experiments (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    description   TEXT,
    git_sha       TEXT,
    config_json   TEXT    NOT NULL CHECK (json_valid(config_json)),
    manifest_json TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(manifest_json)),  -- versions, hw, seeds
    status        TEXT    NOT NULL DEFAULT 'draft'
                  CHECK (status IN ('draft','running','complete','aborted')),
    created_at    INTEGER NOT NULL DEFAULT (CAST(unixepoch('subsec') * 1000 AS INTEGER)),
    CONSTRAINT uq_experiments_name UNIQUE (name)
) STRICT;

CREATE TABLE actions (                      -- mirrors the Python IntEnum; seeded once
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    CONSTRAINT uq_actions_name UNIQUE (name)
) STRICT;
INSERT INTO actions (id, name) VALUES
 (0,'HOVER'),(1,'FORWARD'),(2,'LEFT'),(3,'RIGHT'),(4,'ASCEND'),(5,'DESCEND'),
 (6,'BRAKE'),(7,'AVOID_LEFT'),(8,'AVOID_RIGHT'),(9,'RETURN_HOME'),(10,'EMERGENCY_LAND');

CREATE TABLE arms (
    id            INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    name          TEXT    NOT NULL,         -- 'jev', 'llm_slow', 'pid_noavoid', 'fsm', ...
    kind          TEXT    NOT NULL
                  CHECK (kind IN ('pid','pid_reactive','mpc','rl','fsm','llm_slow','jev','oracle','random','hover')),
    provider      TEXT,                     -- 'openrouter', 'local-mlx', NULL for classical
    model_id      TEXT,
    config_json   TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(config_json)),
    CONSTRAINT uq_arms_experiment_id_name UNIQUE (experiment_id, name),
    CONSTRAINT fk_arms_experiment_id FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
) STRICT;

CREATE TABLE scenarios (                    -- one row = one seeded adversarial world, shared by all arms
    id            INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    scenario_key  TEXT    NOT NULL,         -- sha256 of canonical spec_json (paired-design identity)
    seed          INTEGER NOT NULL,
    family        TEXT    NOT NULL,         -- 'baseline','sudden_obstacle','gps_loss','multi_event',...
    difficulty    REAL,
    spec_json     TEXT    NOT NULL CHECK (json_valid(spec_json)),  -- full serialisable scenario
    CONSTRAINT uq_scenarios_experiment_id_scenario_key UNIQUE (experiment_id, scenario_key),
    CONSTRAINT fk_scenarios_experiment_id FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_scenarios_experiment_id_family ON scenarios (experiment_id, family);

CREATE TABLE scenario_events (              -- denormalised from spec_json for SQL group-bys
    scenario_id           INTEGER NOT NULL,
    seq                   INTEGER NOT NULL,
    kind                  TEXT    NOT NULL
                          CHECK (kind IN ('obstacle','wind','target_move','sensor_noise','battery_drop','gps_loss','other_drone')),
    t_start_tick          INTEGER NOT NULL,
    t_end_tick            INTEGER,
    severity              REAL,
    params_json           TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(params_json)),
    ground_truth_action   INTEGER,          -- oracle "correct" action during the event window
    avoidable             INTEGER NOT NULL DEFAULT 1 CHECK (avoidable IN (0,1)),  -- feasibility guarantee
    PRIMARY KEY (scenario_id, seq),
    CONSTRAINT fk_scenario_events_scenario_id FOREIGN KEY (scenario_id) REFERENCES scenarios (id) ON DELETE CASCADE,
    CONSTRAINT fk_scenario_events_ground_truth_action FOREIGN KEY (ground_truth_action) REFERENCES actions (id)
) STRICT, WITHOUT ROWID;
CREATE INDEX ix_scenario_events_kind ON scenario_events (kind);

CREATE TABLE flights (                      -- one row = one (scenario, arm, latency cell, repeat) run; also the job queue
    id              INTEGER PRIMARY KEY,
    experiment_id   INTEGER NOT NULL,
    scenario_id     INTEGER NOT NULL,
    arm_id          INTEGER NOT NULL,
    latency_cell    TEXT    NOT NULL DEFAULT 'native',   -- 'native' | 'inject:300ms' | 'replay:jev-2026-10-01'
    rep             INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued','running','complete','failed','skipped')),
    attempt         INTEGER NOT NULL DEFAULT 0,
    worker_id       TEXT,
    claimed_at      INTEGER,
    heartbeat_at    INTEGER,
    started_at      INTEGER,
    finished_at     INTEGER,
    ctrl_hz         INTEGER,
    sim_ticks       INTEGER,
    wall_ns         INTEGER,
    outcome         TEXT CHECK (outcome IN ('success','collision','timeout','crash','emergency_land','error')),
    error           TEXT,
    trajectory_uri  TEXT,                   -- relative path under data dir (parquet/npz), NULL until written
    trajectory_sha256 TEXT,
    CONSTRAINT uq_flights_scenario_arm_latency_rep UNIQUE (scenario_id, arm_id, latency_cell, rep),  -- idempotency key
    CONSTRAINT fk_flights_experiment_id FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
    CONSTRAINT fk_flights_scenario_id   FOREIGN KEY (scenario_id)   REFERENCES scenarios (id)   ON DELETE CASCADE,
    CONSTRAINT fk_flights_arm_id        FOREIGN KEY (arm_id)        REFERENCES arms (id)        ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_flights_queue ON flights (id) WHERE status = 'queued';                 -- O(1) claim
CREATE INDEX ix_flights_running_heartbeat ON flights (heartbeat_at) WHERE status = 'running';
CREATE INDEX ix_flights_experiment_id_arm_id_status ON flights (experiment_id, arm_id, status);

CREATE TABLE decisions (                    -- hot table: 10^6..10^7 rows; keep it narrow
    flight_id   INTEGER NOT NULL,
    seq         INTEGER NOT NULL,           -- 0..n per flight
    obs_tick    INTEGER NOT NULL,           -- tick of the observation the decision was based on
    issue_tick  INTEGER NOT NULL,           -- tick at which the request left the sim loop
    apply_tick  INTEGER,                    -- tick at which the action took effect (NULL = dropped/timeout)
    action      INTEGER NOT NULL,           -- actions.id actually applied (after parse/fallback)
    raw_action  INTEGER,                    -- actions.id the provider returned (NULL = unparseable)
    latency_ns  INTEGER NOT NULL,           -- end-to-end provider latency, perf_counter_ns delta
    ttft_ns     INTEGER,
    flags       INTEGER NOT NULL DEFAULT 0, -- bit0 invalid_output, bit1 timeout, bit2 cache_hit, bit3 shield_override, bit4 fallback_used
    PRIMARY KEY (flight_id, seq),
    CONSTRAINT fk_decisions_flight_id FOREIGN KEY (flight_id) REFERENCES flights (id) ON DELETE CASCADE,
    CONSTRAINT fk_decisions_action FOREIGN KEY (action) REFERENCES actions (id)
) STRICT, WITHOUT ROWID;

CREATE TABLE model_calls (                  -- only for provider-backed arms; one row per HTTP/local-model call
    flight_id         INTEGER NOT NULL,
    decision_seq      INTEGER NOT NULL,
    provider          TEXT    NOT NULL,
    model_id          TEXT    NOT NULL,
    prompt_hash       TEXT    NOT NULL,     -- -> model_cache.key_hash (prompt text is regenerable/deduped)
    request_ns        INTEGER NOT NULL,
    ttft_ns           INTEGER,
    total_ns          INTEGER NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    http_status       INTEGER,
    retries           INTEGER NOT NULL DEFAULT 0,
    cache_hit         INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0,1)),
    cost_usd          REAL,
    error             TEXT,
    PRIMARY KEY (flight_id, decision_seq),
    CONSTRAINT fk_model_calls_decision FOREIGN KEY (flight_id, decision_seq) REFERENCES decisions (flight_id, seq) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;
CREATE INDEX ix_model_calls_provider_model_id ON model_calls (provider, model_id);

CREATE TABLE model_cache (                  -- response cache (replay mode); NOT used for latency measurement
    key_hash            TEXT    PRIMARY KEY,  -- sha256(provider|model|params_json|prompt)
    provider            TEXT    NOT NULL,
    model_id            TEXT    NOT NULL,
    params_json         TEXT    NOT NULL CHECK (json_valid(params_json)),
    prompt_text         TEXT    NOT NULL,
    response_text       TEXT    NOT NULL,
    recorded_total_ns   INTEGER,              -- latency observed when the row was first recorded
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    hit_count           INTEGER NOT NULL DEFAULT 0,
    created_at          INTEGER NOT NULL DEFAULT (CAST(unixepoch('subsec') * 1000 AS INTEGER))
) STRICT, WITHOUT ROWID;

CREATE TABLE flight_metrics (               -- one row per completed flight; written in the same tx as status='complete'
    flight_id               INTEGER PRIMARY KEY,
    success                 INTEGER NOT NULL CHECK (success IN (0,1)),
    collision               INTEGER NOT NULL CHECK (collision IN (0,1)),
    collision_tick          INTEGER,
    near_miss_count         INTEGER NOT NULL DEFAULT 0,
    min_clearance_m         REAL,
    time_to_target_s        REAL,
    path_length_m           REAL,
    recovery_time_s         REAL,
    energy_j                REAL,
    oscillation_index       REAL,
    n_decisions             INTEGER NOT NULL,
    n_invalid               INTEGER NOT NULL DEFAULT 0,
    n_timeouts              INTEGER NOT NULL DEFAULT 0,
    latency_p50_ms          REAL,
    latency_p95_ms          REAL,
    latency_p99_ms          REAL,
    decisions_per_wall_s    REAL,
    sim_speed_x             REAL,           -- sim_seconds / wall_seconds (excl. API wait; see 08)
    CONSTRAINT fk_flight_metrics_flight_id FOREIGN KEY (flight_id) REFERENCES flights (id) ON DELETE CASCADE
) STRICT;

CREATE TABLE event_outcomes (               -- per (flight, scenario event) reaction breakdown
    flight_id           INTEGER NOT NULL,
    scenario_id         INTEGER NOT NULL,
    event_seq           INTEGER NOT NULL,
    detected_tick       INTEGER,
    first_correct_tick  INTEGER,
    reaction_ticks      INTEGER,
    collided            INTEGER NOT NULL DEFAULT 0 CHECK (collided IN (0,1)),
    recovery_time_s     REAL,
    PRIMARY KEY (flight_id, event_seq),
    CONSTRAINT fk_event_outcomes_flight_id FOREIGN KEY (flight_id) REFERENCES flights (id) ON DELETE CASCADE,
    CONSTRAINT fk_event_outcomes_event FOREIGN KEY (scenario_id, event_seq) REFERENCES scenario_events (scenario_id, seq)
) STRICT, WITHOUT ROWID;

CREATE VIEW v_flight_summary AS
SELECT f.id AS flight_id, e.name AS experiment, a.name AS arm, s.family, s.seed, f.latency_cell, f.rep,
       f.status, f.outcome, m.success, m.collision, m.time_to_target_s, m.path_length_m, m.recovery_time_s,
       m.latency_p50_ms, m.latency_p95_ms, m.decisions_per_wall_s, m.sim_speed_x
FROM flights f
JOIN experiments e ON e.id = f.experiment_id
JOIN arms a        ON a.id = f.arm_id
JOIN scenarios s   ON s.id = f.scenario_id
LEFT JOIN flight_metrics m ON m.flight_id = f.id;
```

Trajectory storage is deliberately **not a table** (see 3.2). Optional later (YAGNI until 12 needs it): `trajectory_chunks(id, uri, format, hz, n_flights, columns_json, sha256, bytes)` if many flights share one Parquet file.

### 3.1 Job-queue claim (atomic, resumable)

```sql
-- inside BEGIN IMMEDIATE; uses RETURNING (SQLite >= 3.35)
UPDATE flights SET status='running', worker_id=:w, attempt=attempt+1, claimed_at=:now, heartbeat_at=:now
 WHERE id = (SELECT id FROM flights WHERE status='queued' ORDER BY id LIMIT 1)
RETURNING id, scenario_id, arm_id, latency_cell, rep;
-- verified plan: SCAN flights USING COVERING INDEX ix_flights_queue (partial index, O(1))
-- crash recovery at startup: UPDATE flights SET status='queued' WHERE status='running' AND heartbeat_at < :now-:stale_ms;
```
Completion = one transaction: insert `decisions` (+`model_calls`, `event_outcomes`), insert `flight_metrics`, `UPDATE flights SET status='complete', trajectory_uri=...`. A crash mid-flight leaves no partial rows (nothing committed before the end), which is the main reason for the per-flight transaction. Write the trajectory file *before* the transaction (temp name + atomic rename) so `trajectory_uri` never points to a missing file; orphan files are harmless and GC-able.

### 3.2 Trajectory: rows vs files (measured, 30 s flight at 48 Hz = 1440 samples, 13 float32 channels t,pos,vel,rpy,omega, synthetic smooth-plus-noise signals; 100 flights)

| Layout | Bytes / flight | x10k flights |
|---|---|---|
| Raw float32 | 74,880 | 0.75 GB |
| SQLite row-per-sample (WITHOUT ROWID, REAL cols) | 203,735 | 2.0 GB |
| zlib'd float32 BLOB in SQLite | 69,959 | 0.70 GB |
| npz compressed per flight | 70,172 | 0.70 GB |
| Parquet zstd (chunk of 100 flights) | 98,864 | 0.99 GB |
| Parquet snappy | 105,465 | 1.05 GB |
| **Parquet zstd + `use_byte_stream_split=True`, no dictionary** | **54,134** | **0.54 GB** |

Caveat: synthetic signals; real PyBullet trajectories will compress similarly or slightly worse (quantisation noise in float32 is what byte-stream-split exploits). Recommendation: **Parquet per worker chunk (e.g. 50-200 flights, columns `flight_id, tick, ...`), zstd + byte-stream-split**, stored at a downsampled rate (12-24 Hz is enough for replay/plots; keep full-rate only for a debug subset). DuckDB queries Parquet globs directly and fast (section 8). Per-flight npz is the simplest alternative (one file per flight, trivially atomic) but creates 10k-100k small files; Parquet chunks avoid that. A zlib BLOB in SQLite also works and keeps one file, but blocks DuckDB/pandas from scanning columns and bloats the DB you back up/copy; not recommended.

---

## 4. Connection setup and pragma set

Applied per connection (pragmas other than `journal_mode` are connection-scoped; `journal_mode=WAL` persists in the file - [SQLite WAL docs](https://www.sqlite.org/wal.html), accessed 2026-09-30):

```sql
PRAGMA journal_mode = WAL;          -- readers do not block the writer; persists in file
PRAGMA synchronous = NORMAL;        -- WAL: no fsync per commit; only checkpoints sync.
PRAGMA busy_timeout = 30000;        -- ms; MUST be set on every connection (python sqlite3 `timeout=` does the same)
PRAGMA foreign_keys = ON;           -- off by default in SQLite, per connection
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -65536;         -- 64 MiB page cache per connection (negative = KiB); workers can use -16384
PRAGMA mmap_size = 268435456;       -- 256 MiB, read-heavy analysis connections only
PRAGMA wal_autocheckpoint = 4000;   -- pages (default 1000); fewer checkpoints during bulk writes
PRAGMA journal_size_limit = 67108864; -- cap the WAL file left behind after a checkpoint (64 MiB)
-- at shutdown of the orchestrator: PRAGMA wal_checkpoint(TRUNCATE); PRAGMA optimize;
```

Notes / trade-offs:
- `synchronous=NORMAL` in WAL: a committed transaction can be lost on power loss/OS crash (not on process crash) until a checkpoint syncs ([WAL docs](https://www.sqlite.org/wal.html)). Acceptable: flights are idempotently re-runnable from the `queued` state. **Measured: no meaningful throughput difference vs FULL at batch=500** (table below), so the choice is about the per-row commit case (41k vs 17.6k rows/s at batch=1). Keep NORMAL.
- WAL only works when all processes are on the same host and not on a network filesystem (same doc). The data dir must be on local APFS (not iCloud Drive/Dropbox-synced folders, which can corrupt/duplicate `-wal`/`-shm`; inference, not tested).
- Checkpoint starvation: a long-lived reader transaction stops checkpoints and lets the WAL grow unbounded ([WAL docs](https://www.sqlite.org/wal.html)). Rule: API/analysis readers open short read transactions; never hold a cursor open across a flight.
- Default in this build: `DEFAULT_SYNCHRONOUS=2`, `DEFAULT_WAL_AUTOCHECKPOINT=1000`, `DEFAULT_PAGE_SIZE=4096`, `THREADSAFE=1`, FTS5/JSON/math functions compiled in (from `PRAGMA compile_options` of uv's Python 3.14.2).
- Page size: leave at 4096 (changing requires VACUUM and gives marginal gain here).
- Always use **`BEGIN IMMEDIATE`** for write transactions. A deferred `BEGIN` that later upgrades from read to write can fail immediately with `SQLITE_BUSY`/`BUSY_SNAPSHOT` without invoking the busy handler; `IMMEDIATE` takes the write lock up front so `busy_timeout` applies (inference from WAL semantics; consistent with spike below).

### SQLAlchemy hook (aiosqlite for the API process; same hook pattern for sync engines)

```python
# app/db/database.py  (sketch; conventions: settings import, typed, no print, lazy logging)
@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    for pragma in PRAGMAS:  # tuple[str, ...] from section 4
        cur.execute(pragma)
    cur.close()
```
SQLAlchemy docs describe the pysqlite/aiosqlite legacy-transaction quirk (need `isolation_level=None` + explicit `BEGIN` event for correct SAVEPOINT/`BEGIN IMMEDIATE` behaviour; [SQLAlchemy 2.1 SQLite dialect docs](https://docs.sqlalchemy.org/en/21/dialects/sqlite.html), accessed 2026-09-30). For the API process (mostly reads), defaults suffice. **Workers should bypass SQLAlchemy ORM entirely** (next section).

---

## 5. Write-throughput spike (measured)

### Setup (`spike.py`)
- Row = `decisions`-like STRICT table with rowid PK + `UNIQUE(flight_id, seq)` (a worse case than the WITHOUT ROWID design above): 11 columns incl. ~25-byte text; measured ~88 bytes/row on disk. Rows generated in Python per worker (generation alone = ~618k rows/s/process, so **with 8 workers the generator, not SQLite, is a co-bottleneck at the high end; numbers are "at least" for SQLite**).
- 8 worker processes (`multiprocessing` spawn), 200,000 rows each = 1.6M rows, unless noted. Sync `sqlite3`, `isolation_level=None`, explicit `BEGIN IMMEDIATE ... COMMIT`, batch = rows per transaction.
- Strategies: (A) shared WAL DB, all 8 write directly; (B) per-worker DB files then `ATTACH` + `INSERT...SELECT` merge; (C) single writer process fed by `mp.Queue` (maxsize 64) of batches; (D) per-worker Parquet files (pyarrow zstd) for comparison.
- Machine was shared with other agents (load average 6-12 during the final runs; an earlier rerun of mine ran during load 30-90 after I launched a `rep.py` without a `__main__` guard under spawn, which fork-bombed; killed, discarded, and all quoted numbers are from clean reruns). Treat numbers as +-30% noise; the ordering of strategies is stable, small differences are not significant.

### Results (clean reruns; median of 3 where marked)

| Strategy | Batch | Sync | busy timeout | rows/s total | lock errors | size |
|---|---|---|---|---|---|---|
| 1 writer, shared WAL (baseline) | 500 | NORMAL | 30 s | 627k | 0 | 93.8 B/row |
| 8 writers, shared WAL | 1 (autocommit per row, 24k rows) | NORMAL | 30 s | 41k | 0 | 261 B/row* |
| 8 writers, shared WAL | 1 | FULL | 30 s | 17.6k | 0 | - |
| 8 writers, shared WAL | 10 | NORMAL | 30 s | 197k | 0 | - |
| 8 writers, shared WAL | 50 | NORMAL | 30 s | 457k | 0 | 88.5 B/row |
| **8 writers, shared WAL (median of 3)** | **500** | NORMAL | 30 s | **663k** (656/663/704k) | **0** | 88 B/row (141 MB) |
| 8 writers, shared WAL | 5000 | NORMAL | 30 s | 873k | 0 | 90.5 B/row |
| 8 writers, shared WAL (median of 3) | 500 | FULL | 30 s | 669k | 0 | - |
| 8 writers, shared WAL | 500 | OFF | 30 s | 722k | 0 | - |
| 8 writers, shared WAL, no text payload | 500 | NORMAL | 30 s | 691k | 0 | 64.7 B/row |
| 8 writers, shared WAL (median of 3) | 500 | NORMAL | **0 (no wait)** | 705k (only successful retries counted) | **~2,300 BUSY per run** (2314/2269/2335) | - |
| **8 per-worker DBs, then merge (median of 3)** | 500 | NORMAL | - | **1.00M** end-to-end (write phase 1.16 s + merge 1.17 s for 1.6M rows) | 0 | 2x disk transiently (275.8 MB total) |
| **Single writer + mp.Queue (median of 3)** | 500 | NORMAL | - | **599k** (745k/599k/523k; high variance) - a second run gave 875k | 0 | 85.5 B/row |
| 8 per-worker Parquet zstd (median of 3) | - | - | - | 833k (656k-874k) | 0 | **13.1 B/row** (21 MB for 1.6M rows; synthetic, highly compressible) |

\*small-n runs include process spawn time (~0.3-0.5 s), biasing rows/s low; the byte/row figure is inflated by WAL not yet checkpointed.

Key reads of the data:
- **Lock errors are a function of `busy_timeout`, not of writer count.** With `timeout=0` 8 writers hit BUSY ~2,300 times in ~3,200 commits (I retried with 5 ms sleep); with 30 s timeout: zero errors across every run. Always configure the timeout.
- Batch size matters below ~50 rows/tx; above 500 rows/tx the curve is flat (generator-limited). One transaction per flight (~300-1500 rows) is in the flat region.
- `synchronous` FULL vs NORMAL made no measurable difference at batch 500 (669k vs 663k) but 2.3x at per-row commits.
- Per-worker DBs + merge was fastest only because the writers never contend; but it needs a merge step, double temporary disk, loses live queryability, and the ID/uniqueness/idempotency logic must be replicated across files. Not worth it at this volume.
- Single-writer queue adds pickling/IPC cost and a single point of failure; the measured throughput was no better than shared WAL. It is only needed if you later see real lock contention (e.g. 50+ writers) - YAGNI.
- **SQLAlchemy overhead (measured, single process, aiosqlite, SQLAlchemy 2.1.1, loaded machine):** ORM `add_all` of 50k objects = 3.2 s (~15k rows/s); Core `executemany` insert of 200k dict rows = 0.15 s (~1.3M rows/s). **~85x gap.** Consequence: decision/trajectory-adjacent bulk writes must use Core/`sqlite3.executemany`, never per-row ORM objects. ORM is fine for experiments/arms/scenarios/flights (thousands of rows) and API reads.

### Conclusion: multi-writer strategy
1. **Workers** (sync `sqlite3`, or SQLAlchemy Core sync engine) open their own connection, WAL + `busy_timeout=30000`, claim a flight (`BEGIN IMMEDIATE`, tiny tx), simulate entirely in memory (decisions buffered in a list), then write everything in ONE `BEGIN IMMEDIATE` transaction at flight end (plus heartbeat updates every ~10 s, single-row).
2. **API/orchestrator** (FastAPI, `aiosqlite` through SQLAlchemy async) does job creation (bulk insert of flights), progress reads, analysis endpoints; it is a normal WAL reader/occasional writer.
3. Retry wrapper on `sqlite3.OperationalError("database is locked")` with jittered backoff (belt and braces; should never fire with the timeout).
4. Fallbacks, in order: bigger batches; per-worker DB + merge for *offline* giant backfills; single-writer queue only if contention appears. Not needed at 10k flights.
5. LLM-arm caveat: decisions for slow arms arrive at ~1-10/s per flight, so they are not write-limited either; buffer per flight, not per decision (no partial-flight visibility, atomic resume). If live progress is needed, publish counters via the `flights.heartbeat_at`/a small `progress` column, not by flushing decision rows.

---

## 6. Async SQLAlchemy + aiosqlite vs sync sqlite3 in workers

| Concern | Decision |
|---|---|
| API process | `create_async_engine("sqlite+aiosqlite:///data/jev.db")` + `async_sessionmaker(expire_on_commit=False)` + `get_session` via `Depends()` (guide rules: async only, sessions via `Depends()`). aiosqlite runs each connection in its own thread; fine for a local low-concurrency API. |
| Workers | **sync `sqlite3` (or sync SQLAlchemy Core)**. The sim loop is synchronous PyBullet; the worker has no event loop benefit; an aiosqlite thread hop per call is pure overhead and the flight-end batch write is one call. This is a consistent deviation from "async for all DB ops" which targets request handling; flag it in 14. The guide says repositories are class-based with injected deps: define `FlightWriter`/`DecisionRepository` with an injected connection, with a sync implementation for workers and the async SQLAlchemy one for API reads. |
| Async in workers for the API-arm (HTTP concurrency) | The async HTTP client loop (owner 09/01) and DB writes should not share a thread with the sim; write at flight end from the sync path (or `run_in_executor`). |
| Engine per process | Never share an engine/connection across `fork`/`spawn` - create inside the worker after spawn. |
| Pool | aiosqlite with file DB: default pool is fine; set `pool_size` small; for tests with `:memory:` use `StaticPool`. |
| `ondelete="CASCADE"` | needs `PRAGMA foreign_keys=ON` on every connection (hook above); ORM relationships use `passive_deletes=True`. |
| DB-bound types | Workers use ints/floats/text only (STRICT). JSON columns are TEXT with `CHECK (json_valid())`; JSON1 functions (`json_extract`) are built in (compile options show JSON, math functions). |

### SQLAlchemy 2.0 Mapped sketches (conventions-compliant; **verified**: `Base.metadata.create_all` succeeds on SQLite and `mypy --strict` passes on these files in the scratch env)

**Gotcha found by running it:** SQLAlchemy maps `str` annotations to `VARCHAR`, and SQLite STRICT rejects `VARCHAR` (`create_all` fails). Fix: `type_annotation_map = {str: Text}` on the Base (below). Table option is `__table_args__ = ({"sqlite_strict": True, "sqlite_with_rowid": False},)` ([SQLAlchemy SQLite docs](https://docs.sqlalchemy.org/en/21/dialects/sqlite.html), accessed 2026-09-30).

```python
# app/db/base.py
from __future__ import annotations

from sqlalchemy import MetaData, Text
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {  # matches guide: ix_ / uq_ / fk_
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # STRICT tables reject VARCHAR: map every `str` annotation to TEXT.
    type_annotation_map = {str: Text}
```

```python
# app/models/flight.py  (singular file name per guide; table plural)
from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Flight(Base):
    __tablename__ = "flights"
    __table_args__ = (
        UniqueConstraint("scenario_id", "arm_id", "latency_cell", "rep"),
        CheckConstraint(
            "status IN ('queued','running','complete','failed','skipped')",
            name="status",
        ),
        Index("ix_flights_queue", "id", sqlite_where=text("status = 'queued'")),
        {"sqlite_strict": True},
    )

    id: Mapped[int] = mapped_column(
        primary_key=True
    )  # integer PK: documented deviation from UUID default
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE")
    )
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE")
    )
    arm_id: Mapped[int] = mapped_column(ForeignKey("arms.id", ondelete="CASCADE"))
    latency_cell: Mapped[str] = mapped_column(default="native", server_default="native")
    rep: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    status: Mapped[str] = mapped_column(default="queued", server_default="queued")
    sim_ticks: Mapped[int | None]
    wall_ns: Mapped[int | None]
    trajectory_uri: Mapped[str | None]

    decisions: Mapped[list[Decision]] = relationship(
        back_populates="flight", passive_deletes=True
    )


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = ({"sqlite_strict": True, "sqlite_with_rowid": False},)

    flight_id: Mapped[int] = mapped_column(
        ForeignKey("flights.id", ondelete="CASCADE"), primary_key=True
    )
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    obs_tick: Mapped[int]
    issue_tick: Mapped[int]
    apply_tick: Mapped[int | None]
    action: Mapped[int]
    raw_action: Mapped[int | None]
    latency_ns: Mapped[int]
    ttft_ns: Mapped[int | None]
    flags: Mapped[int] = mapped_column(default=0, server_default=text("0"))

    flight: Mapped[Flight] = relationship(back_populates="decisions")
```
(Experiment, Scenario, Arm, ScenarioEvent, ModelCall, ModelCache, FlightMetric, EventOutcome follow the DDL one-to-one; in the sketch I exercised Experiment/Scenario/Arm/Flight/Decision. Use `server_default=text("0")` rather than the string `"0"` to avoid quoting `'0'` into the DDL default - the string form worked in SQLite via affinity but is sloppy.) Schemas (`schemas/`) are Pydantic DTOs suffixed per guide: `FlightResponse`, `FlightMetricsResponse`, `DecisionResponse`, `ExperimentCreate`.

### Alembic with SQLite
- Stack already lists Alembic but it is not installed or configured. Recommend: `alembic init -t async migrations`, `render_as_batch=True` in `env.py` (SQLite `ALTER` is limited; batch mode recreates tables), naming convention on `Base.metadata` so batch operations can target constraints by name, autogenerate the initial revision from the models, then **diff it against `ddl.sql`** (the DDL above is the target truth). Alembic 1.20.0 installs on 3.14.
- Scale-appropriate alternative (KISS): since experiment data is regenerable, allow "new experiment = new DB file" (`data/experiments/<name>.db`) and keep Alembic only for additive changes. Per-experiment DB files also make archive/delete trivial (`rm`) and keep each DB small. Trade-off: cross-experiment queries require `ATTACH`/DuckDB. I lean to **one DB per experiment campaign, Alembic for schema evolution**; decision deferred to 14 (settings would take `database_url` per run).

---

## 7. Volume estimates (10k flights)

Assumptions (state them; scale linearly): 30 s sim flights, 48 Hz control, decisions at 10 Hz for provider arms (300/flight) and 48 Hz for classical/FSM arms (1,440/flight; usually you would store action *changes* only). Row widths from measurement: `decisions` WITHOUT ROWID narrow row = **30.2 B/row** measured (8 cols); with the full 10-column DDL allow ~40 B/row; model_calls ~90 B/row estimated.

| Item | Rows | Bytes | Size |
|---|---|---|---|
| decisions, 10k flights x 300 (10 Hz) | 3.0M | 40 B | **~120 MB** (measured-derived) |
| decisions, same at 48 Hz, all-arm full | 14.4M | 40 B | ~576 MB |
| decisions if stored wide with text (my spike row) | 3.0M | 88 B | ~264 MB (rowid + unique index + text) |
| model_calls (provider arms only, ~40% of flights) | 1.2M | ~90 B | ~110 MB |
| model_cache (unique prompts; prompt text ~1.2 KB + response) | <= 1.2M worst, realistically 10-30% unique | ~1.4 KB | 0.2-1.7 GB (**cap it**; store prompt only for cache-enabled replay arms; otherwise only the hash) |
| flights | 10k | ~250 B | 2.5 MB |
| flight_metrics | 10k | ~200 B | 2 MB |
| scenarios (shared by ~7 arms x 7 latency cells => ~200-2,000 scenarios) | 2k | ~2 KB | 4 MB |
| scenario_events / event_outcomes | ~30k / ~300k | ~60 B | 20 MB |
| indexes beyond PKs (flights x3, partial) | - | - | ~2 MB |
| **SQLite total, 10 Hz decisions, no prompt text** | | | **~0.25-0.4 GB** |
| Trajectory Parquet @48 Hz, zstd+BSS (54 KB/flight measured) | 14.4M samples | | ~0.54 GB |
| Trajectory Parquet @12 Hz (recommended) | | | ~0.14-0.2 GB |
| Same trajectories as SQLite rows @48 Hz | | 204 KB/flight | 2.0 GB (rejected) |

100k flights = 10x: SQLite 2.5-4 GB, Parquet 1.4-5.4 GB. Internal SQLite limits are irrelevant (2^63 rows / 281 TB). The 460 GiB APFS volume has 81 GiB free (measured `df`), so disk is not a constraint at 10k-100k.

**Required write rate** (for comparison to measured capacity): 10k flights over, say, 3 hours = 0.93 flights/s; at 300 rows/flight = ~280 rows/s sustained; even a burst of 8 workers finishing simultaneously = 8 x 1,440 rows = 11.5k rows in one lock-serialised sequence (~20 ms at 600k rows/s). **Measured capacity exceeds need by ~2,000x.** If flights run 100x faster than real time with 48 Hz decisions, 8 workers still produce ~8 x 48 x 100 = 38k rows/s, well within the 41k-rows/s per-row-commit result and far under batch capacity.

---

## 8. Analysis tooling

Measured on the 1.6M-row spike DB/Parquet files (Python 3.14.2, pandas 3.0.6, duckdb 1.5.6):
- sqlite3 group-by over 1.6M rows: 0.47 s.
- `pandas.read_sql("select * ...")` of 1.6M rows: 2.8 s (full materialisation; avoid for big tables, aggregate in SQL).
- **DuckDB on Parquet, group-by + `quantile_cont` over 1.6M rows: 0.024 s.**
- DuckDB's SQLite scanner is **not bundled** in the wheel: `LOAD sqlite` fails unless the extension is installed (`INSTALL sqlite` downloads it from DuckDB's extension repo; not done here because it is a network download that needs user approval). Plan: analysis reads `flight_metrics`/`v_flight_summary` via pandas `read_sql` (small tables) and trajectories/decisions via DuckDB on Parquet; optionally export nightly `decisions` to Parquet (`COPY (...) TO`) with DuckDB after the sqlite extension is installed, or with `pandas.read_sql` chunks -> pyarrow. A SQLite -> Parquet export of `decisions` makes the analysis layer independent of SQLite locks.
- Stats (Wilson CIs, bootstrap, McNemar) belong in `app/analysis/` or `scripts/` using numpy/scipy/statsmodels; out of scope here (08 owns).
- Live progress for 12: `SELECT arm_id, status, count(*) FROM flights WHERE experiment_id=? GROUP BY 1,2` hits `ix_flights_experiment_id_arm_id_status` (covering) - cheap to poll every second; SSE endpoint reads this.

---

## 9. Python 3.14 dependency verification (measured: `uv venv --python 3.14` -> 3.14.2, `uv pip install`, wheels only, arm64)

| Package | Version installed | Result |
|---|---|---|
| Python | 3.14.2 (uv-managed, clang 21.1.4, macOS aarch64) | OK; stdlib sqlite3 = SQLite 3.50.4 |
| aiosqlite | 0.22.1 | installs/imports OK |
| SQLAlchemy | **2.1.1** resolved for `>=2.0.50` (2.0.50 also resolvable: `uv pip install --dry-run sqlalchemy==2.0.50` would install it) | OK; async + aiosqlite works (ORM and Core inserts run, `select sqlite_version()` returns 3.50.4) |
| greenlet | 3.5.6 (only after `sqlalchemy[asyncio]`) | **Not installed automatically by SQLAlchemy 2.1.1**; `create_async_engine` import fails with ImportError until added |
| alembic | 1.20.0 | OK |
| duckdb | 1.5.6 | OK (wheel) |
| pyarrow | 25.0.1 | OK (wheel) |
| pandas | 3.0.6 | OK |
| numpy | 2.5.3 | OK |
| pydantic / pydantic-settings / fastapi / starlette / httpx / pytest / pytest-asyncio | 2.13.5 / 2.15.0 / 0.142.2 / 1.7.1 / 0.28.1 / 9.1.1 / 1.4.0 | OK |
| ruff / mypy (scratch) | 0.16.9 / installed | OK; mypy --strict passes on the model sketch |

Important nuance: the repo's existing pins are `>=` floors, so a fresh `uv pip install -r requirements.txt` today would pick SQLAlchemy 2.1.x; 2.1 changes the asyncio extra and may have other behavioural changes (I did not audit the 2.1 changelog). Choose explicitly: either pin `SQLAlchemy[asyncio]>=2.0.50,<2.1` (known-good for the guide, which says "SQLAlchemy 2.x") or adopt 2.1 and add `[asyncio]`. My sketch and spike ran on 2.1.1. Not tested: gym-pybullet-drones/pybullet on 3.14 (owner 02); if the sim worker needs a separate 3.10-3.12 venv, workers must not import the FastAPI app layer - keep a stdlib-only (`sqlite3`) writer module (`app/harness/store.py`) importable from both environments, which is another reason to keep worker DB code sync and free of SQLAlchemy/pydantic imports.

---

## 10. FastAPI endpoints (shape, following guide: no trailing id, pagination everywhere, `success_response`)

Under `app/api/v1/` (router files plural nouns; add to `router.py`):
```
GET  /experiments/list                        -> PaginatedResponse[ExperimentResponse]   (page, page_size)
POST /experiments/create                      -> 201
GET  /experiments/{experiment_id}/summary     -> aggregated metrics per arm/latency cell (SQL group-by over flight_metrics)
GET  /experiments/{experiment_id}/progress    -> counts by arm/status (for polling/SSE)
GET  /experiments/{experiment_id}/flights     -> paginated flights (filters: arm, outcome, family)
GET  /flights/{flight_id}/details             -> flight + metrics + events
GET  /flights/{flight_id}/decisions           -> paginated (page_size default 200; 300-1,440 rows/flight)
GET  /flights/{flight_id}/trajectory          -> stream Parquet/npz slice (for web replay, owner 12)
POST /experiments/{experiment_id}/runs/start  -> launches runner subprocess (owner 09); 202
```
Prerequisites that do not exist yet and must be added in the same change: `utils/pagination.py` + `PaginatedResponse`; domain errors (`ExperimentNotFoundError`, `FlightNotFoundError`) subclassing `NotFoundError` in `app/core/exceptions.py`; a repository/service per aggregate with constructor-injected session; `get_session` dependency. Keep the batch runner a CLI (`scripts/`), not an API thread, so crashes/memory are isolated (09).

---

## 11. Risks / smells found (flag only)

- `utils/save_file.py`: unused so far, no `from typing import Optional` modernisation; leave.
- `ignore_missing_imports = true` in `mypy.ini` hides missing types for SQLAlchemy if the pre-commit mypy env lacks it (see section 2).
- `.gitignore` does not list `data/`, `*.db`, `*.db-wal`, `*.db-shm`, `*.parquet`; add before creating any DB (otherwise `check-added-large-files` is the only safeguard).
- WAL creates `-wal` and `-shm` side files; backups must use `sqlite3 .backup`/`VACUUM INTO`, not a naive file copy while workers are active.
- Stale `running` flights after a crash/sleep: heartbeat-based requeue (section 3.1); macOS sleep will stall heartbeats (use `caffeinate`, 09).
- Spike caveats: synthetic rows, rowid+unique layout (worse than the recommended WITHOUT ROWID), machine shared with other agents (load avg 6-12), generator-limited at the high end, single APFS SSD. Numbers are lower bounds for SQLite, not absolute guarantees.

---

## Sources (accessed 2026-09-30)
- SQLite WAL: https://www.sqlite.org/wal.html
- SQLite STRICT tables: https://www.sqlite.org/stricttables.html
- SQLAlchemy 2.1 SQLite dialect (aiosqlite, `sqlite_strict`/`sqlite_with_rowid`, transaction workaround): https://docs.sqlalchemy.org/en/21/dialects/sqlite.html
- Package versions/wheels: measured via `uv pip install` in the scratch venv (above), not web-sourced.

---

## Decisions recommended

1. **Storage split:** SQLite (STRICT, WAL) for metadata/decisions/metrics/queue/cache; Parquet chunks (zstd + byte-stream-split, 12-24 Hz) for trajectories, referenced by `flights.trajectory_uri`. No trajectory rows in SQLite.
2. **Schema:** adopt the DDL in section 3 (integer PKs, `WITHOUT ROWID` for decisions/model_calls/event_outcomes/scenario_events, idempotency key `UNIQUE(scenario_id, arm_id, latency_cell, rep)`, partial-index job queue, `flight_metrics` wide row, `v_flight_summary`).
3. **Writes:** shared WAL DB, one sync `sqlite3` connection per worker, `BEGIN IMMEDIATE`, `busy_timeout=30000`, one transaction per flight containing decisions + metrics + status='complete'; Core/`executemany` only for hot tables (ORM is ~85x slower).
4. **Pragmas:** WAL, synchronous=NORMAL, busy_timeout=30000, foreign_keys=ON, temp_store=MEMORY, cache_size=-65536, wal_autocheckpoint=4000, journal_size_limit=64 MiB; `PRAGMA optimize` + `wal_checkpoint(TRUNCATE)` on shutdown.
5. **Template changes:** drop `asyncpg`, `psycopg2-binary`, `boto3` (and consider `requests`, `python-dotenv`, `python-multipart`); add `aiosqlite`, `sqlalchemy[asyncio]`, `alembic`; `database_url` default to a local SQLite file; implement `app/db/{base,database}.py` (empty today), lifespan for engine disposal, `type_annotation_map={str: Text}` for STRICT; tests use temp-file DB + `dependency_overrides`; add `sqlalchemy`/`aiosqlite` to the mypy pre-commit hook; gitignore `data/`, `*.db*`, `*.parquet`.
6. **SQLAlchemy version:** pin explicitly (`SQLAlchemy[asyncio]>=2.0.50,<2.1` for the guide's "2.x", or go 2.1 with `[asyncio]`); do not leave `>=` floats that silently cross 2.1 and drop the greenlet auto-install.
7. **Sync in workers, async in API:** worker store is a stdlib-only module so it can run in a different Python env than the FastAPI app if pybullet forces 3.10-3.12.
8. **Analysis:** DuckDB over Parquet for decision/trajectory analytics; pandas `read_sql` only for small tables; nightly/at-end export of `decisions` to Parquet.

## Open questions

1. One DB per experiment campaign vs a single global DB (affects Alembic burden, `ATTACH` for cross-experiment analysis, settings shape). Needs 14's call.
2. Does the sim worker need a separate Python (<3.14) env (owner 02)? If so the stdlib-only store module is mandatory, and the `Decision`/`Flight` ORM models are API-side only.
3. Decision rate per arm and whether to store every control-tick decision for classical arms or only action changes (affects 5x row count; metrics like oscillation need the full stream).
4. Is prompt/response text retention required for audit (model_cache size 0.2-1.7 GB at 10k flights) or is prompt-hash + regenerable prompt enough? Depends on 01/07 prompt templating determinism.
5. Is the DuckDB sqlite extension install (network download) acceptable to the user, or do we standardise on Parquet export only?
6. SQLAlchemy 2.0 vs 2.1 decision (2.1 changelog not audited here).
7. Real trajectory compressibility (my Parquet numbers use synthetic signals) and the actual decisions/flight once 08/11 fix the decision rate.
8. Does anything need live (mid-flight) decision visibility in the UI (would change the per-flight-transaction rule to small periodic flushes)?

## Follow-up spikes (small, runnable experiments)

1. **Real-row write spike:** rerun `spike.py` with 8 workers where each worker does real PyBullet-free work (e.g. 2-5 ms CPU per "tick" loop) before each flight-end transaction, plus concurrent FastAPI/aiosqlite reader polling `/progress` every 500 ms; confirm no lock errors, WAL size bounded (`ls -l *.db-wal` every second), p99 commit latency < 100 ms.
2. **Checkpoint starvation test:** open a long read transaction in a second process during the 8-writer run; record WAL growth and whether `wal_autocheckpoint`/`journal_size_limit` keep it bounded; then verify `wal_checkpoint(TRUNCATE)` recovers it.
3. **Crash-recovery test:** `kill -9` a worker mid-flight and the orchestrator mid-run; restart; confirm `running` rows are requeued via heartbeat, no partial decisions exist, and re-running yields identical rows (idempotency key).
4. **Real trajectory compression:** once pybullet runs (02), dump 100 real 30 s flights at 48 Hz and compare Parquet zstd/BSS, npz, and 12 Hz downsample; replace the synthetic table in 3.2.
5. **SQLAlchemy 2.0.50 vs 2.1.1 parity:** run the `sketch/` models plus `create_all`, an Alembic autogenerate with `render_as_batch`, and the async session tests on both versions; compare generated DDL to `ddl.sql`.
6. **DuckDB sqlite extension:** with the user's approval, `INSTALL sqlite; ATTACH 'data/jev.db' (TYPE sqlite, READ_ONLY)` and time a group-by over 3M `decisions` rows while writers are active (read-only attach semantics under WAL).
7. **API `/decisions` pagination cost:** keyset pagination `WHERE flight_id=? AND seq>? ORDER BY seq LIMIT 200` vs OFFSET on the WITHOUT ROWID table; confirm O(log n).
