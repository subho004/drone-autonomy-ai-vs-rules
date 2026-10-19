# 14 — Repo architecture fit + project skeleton

Research agent 14 (last of 14). Date: 2026-09-30. Inputs: docs/research 00-12 read in full (01 and 02 scripts and 08 appendices skimmed, all "Decisions recommended" read), plus the whole template (`main.py`, `app/`, `utils/`, `tests/`, `mypy.ini`, `pytest.ini`, `requirements.txt`, `.pre-commit-config.yaml`, `Dockerfile`, `docker-compose.yml`, `.env.sample`, `.gitignore`, `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`). Nothing in the repo was changed except this file. Agent 13 (risks/roadmap) reads this.

Evidence tags: **[V]** verified by a number or file I read or ran; **[C]** cited from another agent's measurement (id in brackets); **[I]** my inference/design choice.

The one thing I executed myself: the Protocol/DTO sketch in section 5 was merged into one file and checked in a scratch Python 3.14.2 venv (`mypy --strict` clean, `ruff` with the repo's effective default rules clean, `py_compile` ok). The rest is design.

---

## 0. TL;DR

1. **Two-plane architecture on one SQLite file.** A **runner CLI** (`scripts/run_experiment.py`, `spawn` worker pools) executes flights; **FastAPI** is a thin control/monitor plane (create experiment, progress SSE, summaries, replay JSON) that never hosts a pool [C 09, 10]. Workers use **sync stdlib `sqlite3`**; the API uses async SQLAlchemy + aiosqlite.
2. **One interpreter, one venv, Python 3.14.2 stays.** Every needed package has a cp314 arm64 wheel; pybullet compiles from sdist with `CFLAGS="-Dfdopen=fdopen"` (built on 3.14.2 by 02, 11b, 12) [C]. The flag lives in `scripts/setup_sim_env.sh`, never in `requirements*.txt`. A separate sim env is a contingency only; the code is structured so it stays possible (stdlib-only store, numpy-only domain types, import-boundary lint).
3. **PyBullet (gym-pybullet-drones) is the only backend for v1.** The "Tier-0 kinematic sim" survives as a *surrogate model* (needed anyway for the feasibility certifier, oracle and labeller) and is promoted to a second `SimBackend` only at milestone M6, only if the wall-time measured at M3 justifies it, and only behind the rank-preservation gate from 03. Measured single-core cost: ~5.9k control ticks/s with PID [C 11b], so a 20 s flight at 48 Hz is ~0.2-0.3 s of CPU and a 560k-flight latency sweep is an overnight run, not a week.
3b. **Do not use `VelocityAviary`.** Subclass `CtrlAviary` and own a `DSLPIDControl` adapter (hover drift 1.0 m/10 s vs 0.015 m latched [C 11b]; 0.25 m/s speed cap [C 02, 11]).
4. **One `DecisionProvider` contract for every action-emitting arm** (Jev, chat LLMs, local stand-ins, FSM, RL, oracle, random, hover). Arms that emit setpoints (waypoint PID, reactive avoidance, MPC) implement `SetpointPolicy` instead. The harness, not the provider, owns latency, staleness, fallback and shield.
5. **Dependencies**: delete `asyncpg`, `psycopg2-binary`, `boto3`, `requests`, `python-dotenv`, `python-multipart` (zero imports each [C 10, verified by me below]); add `aiosqlite`, `alembic`, `httpx[http2]`, `PyYAML`; split into `requirements/{core,sim,ml,analysis,dev}.txt`. Docker files are dead (recommend one explicit delete commit).
6. **The guide conflicts with reality in 11 places** (section 3.4). None is fatal; each has an explicit resolution. Most important: `ApiResponse`/`PaginatedResponse`/pagination util do not exist, the `routes/` layer is `app/api/v1/`, the guide's "async for all DB ops" and "sessions only via Depends()" do not apply to worker processes, and the guide's "scripts: constants at top, no CLI args" is kept.

---

## 1. Template audit (what is actually in the repo)

All of this I read; it agrees with agent 10's audit and adds three items.

| Area | Reality | Consequence for this project |
|---|---|---|
| Layout | `app/api/v1/{health,router}.py`, `app/core/{config,exceptions,logging}.py`, `app/db/database.py` (**0 bytes**), `utils/{response,save_file}.py`, `tests/{conftest,test_health}.py`. No `models/ repositories/ services/ schemas/` dirs. `app/` namespace packages (no `__init__.py`); `tests/` has one. | Everything in section 4 is new except edits to `config.py`, `exceptions.py`, `router.py`, `main.py` (lifespan), `conftest.py`. New packages follow the namespace-package convention; `mypy --explicit-package-bases .` already configured. |
| App factory | `main.py`: `create_app()` + module-level `app`, `configure_logging()` at import, CORS, `register_exception_handlers`, one router. **No lifespan.** Comment says "structured (JSON)" but `logging.py` is plain text (latest commit e6c11f9). | Add a `lifespan` for engine disposal (M5). Workers must call `configure_logging()` explicitly (idempotent). Flag the stale comment, do not fix silently. |
| Config | `Settings(BaseSettings)`, `env_file=".env"`, `extra="ignore"`, `NoDecode` for `cors_origins`, `database_url: str = ""`, single `settings`. `.env.sample` has only `PORT, ENV, DEBUG`. | Extend `Settings` (section 7); keep science parameters out of it. |
| Errors | `AppError(StarletteHTTPException)` + 8 subclasses; 3 handlers. | Harness/provider errors are **not** HTTP errors; they get their own `DomainError` tree (section 3.4 item 6). |
| Responses | `utils/response.py`: `APIResponse[T]` (PEP 695 generics), `success_response()`/`error_response()` return `JSONResponse`; `__all__` omits the two helpers. Health route returns `JSONResponse`. | Follow the existing "return `JSONResponse` from helper" pattern. Create `utils/pagination.py` + `PaginatedResponse` before any list endpoint. |
| Tests | `httpx.ASGITransport` client fixture; `pytest.ini`: `asyncio_mode = auto`, `testpaths = tests`. | Extend, do not replace. Guide: tests only when asked. |
| Tooling | `.pre-commit-config.yaml`: ruff (`--fix`) + ruff-format + mypy mirror with only `pydantic`, `pydantic-settings`. **No `ruff.toml`/`pyproject.toml`** -> Ruff runs its defaults (E4, E7, E9, F), so the guide's "isort enforced by Ruff" is **not true today** [C 10, confirmed: no config file in `git ls-files`]. `mypy.ini` has `ignore_missing_imports = true`, which silently turns missing stubs into `Any`. | Recommend a `ruff.toml` (section 6.6) and more mypy hook deps; do not apply. |
| Deps | 17 lines of `>=` floors, no lock. Import grep (`grep -rE "^(import|from) (requests|dotenv|boto3|asyncpg|psycopg2|multipart|sqlalchemy)" app utils main.py tests`) returns nothing: **none of requests, python-dotenv, boto3, asyncpg, psycopg2-binary, python-multipart, SQLAlchemy is imported anywhere**. | Removal list in 6.2 is safe. SQLAlchemy stays (used from M1 on). |
| Docker | `Dockerfile` (python:3.14.2-slim + uv), `docker-compose.yml`, `.dockerignore` (mentions `research`, `initial-plan.txt`). | `python:slim` has no compiler, so pybullet cannot even build there; dead files. Delete in one commit, including the README "Docker" section. |
| `.gitignore` | `.venv`, `.env`, caches only. | Must add `data/`, `*.db*`, `*.parquet`, `*.npz`, `*.rrd`, `wheelhouse/`, `reports/` before the first DB exists. |
| `.python-version` | `3.14.2`; `mypy.ini` `python_version = 3.14`. | Keep. |

---

## 2. Conflict reconciliation (the decisions that hold the rest together)

Format: **conflict -> who said what -> choice -> reason.**

| # | Conflict | Positions | **Choice** | Reason / evidence |
|---|---|---|---|---|
| 1 | **Tier-0 kinematic sim vs PyBullet** | 03: two tiers, T0 numpy-batched for sweeps, T1 gpd for 5-10 % validation. 02/11b: PyBullet 5.9k ticks/s/core with PID. 08: point-mass prototype is what produced the latency curves. 05/06/11: a point-mass surrogate is required for oracle, feasibility and labels. 09: classical flights 33k/h (assumed 1 ms/tick, ~6x pessimistic vs measured). | **PyBullet-only for v1.** Build `sim/surrogate.py` at M2 (pybullet-free, numpy only) for certifier/oracle/labeller. Expose it as `KinematicBackend` at **M6**, only if M3's measured full-matrix wall time exceeds ~1 working day, and only after the rank-preservation gate (Spearman >= 0.9, knee within one grid step, kappa >= 0.6 [C 03]). | Sweep arithmetic [I from C 11b/09]: 10k scenarios x 7 latencies x 8 classical arms = 560k flights x ~0.3 s (incl. harness overhead) = ~47 core-h / 6.5 P-eq = ~7 h. LLM arms are bound by RPM/GPU, not the sim [C 09], so a fast tier buys nothing for them. A second "truth" costs validation effort and an explanation in every figure. |
| 2 | **Runner CLI vs FastAPI-managed jobs** | 09: CLI executes, FastAPI control plane. 10: same (launch via `Popen`, never a pool in uvicorn). | **CLI executes; FastAPI enqueues/monitors.** `Popen(start_new_session=True)` + PID file only if the user wants a start button. | Crash isolation, clean `spawn`, no event-loop jitter in latency measurements [C 09 section 2.2 and 9]. |
| 3 | **pybullet build flag** | 02/11b/12: `CFLAGS="-Dfdopen=fdopen"` works on 3.14.2. 11 (first pass, 3.12): patched vendored header instead. 07: plain build fails. 11b caveat: its install returned in 0.1 s (shared uv cache), compile time unobserved. | Flag, in `scripts/setup_sim_env.sh` with `--no-cache` and a post-install smoke test (`import pybullet; p.connect(p.DIRECT)`). **Not** in requirements. Treat "pybullet builds" as a Phase-0 gate. | Three independent agents reproduced it; build is 74 s-2 m 13 s [C 02, 12]. Keep the recipe out of `requirements*.txt` so a clean `uv pip install -r` does not silently fail. |
| 4 | **Python 3.14.2 pin** | 02, 05, 10, 11b, 12: everything installs on 3.14.2 arm64 (numpy 2.5.3, scipy 1.18.1, gymnasium 1.3.0, torch 2.14.0, SB3 2.9.0, pyarrow 25.0.1, duckdb 1.5.6, rerun-sdk 0.38.1, osqp 1.1.3). 03: Genesis (<3.14) and ML-Agents (<=3.10) rejected anyway. 07/08 used 3.12 scratch envs out of habit. | **Keep 3.14.2.** Contingency: `.venv-sim` on 3.12 if a macOS SDK update breaks the pybullet build on 3.14. gpd `pyproject` says `python = "^3.12"` and works on 3.14.2 [C 02]. | No dependency forces 3.12. The contingency is cheap because of the import-boundary rules in section 3.2. |
| 5 | **VelocityAviary vs DSLPID directly** | 02, 11: subclass `CtrlAviary`/`BaseAviary`; 11b: VelocityAviary `target_pos = cur_pos` gives hover drift F/0.2 m/s (1.0 m in 10 s at 0.02 N), and 0.25 m/s cap. 08 prototype assumed velocity command. | **`ScenarioAviary(CtrlAviary)` + our `DslPidController` adapter** implementing `LowLevelController`. Carrot mode (`target_pos = pos + v*0.3`, `target_vel = v`) for velocity setpoints; true position latch for HOVER. Never latch at current position while moving (72-108 % reversal [C 11b]); BRAKE = velocity-zero ramp, latch when `|v| < 0.1 m/s`. | Measured, three independent confirmations. `ctrl.reset()` must be called in `_housekeeping` (env `reset()` does not reset it [C 02]). |
| 6 | **Jev's typed-decision API vs chat-LLM shape** | 01: Jev = `state` + typed `questions` -> option probabilities, no sampling params; contract should be probabilistic. 07: text prompt formats A-F, one-token output. 11: bare enum output. 04: only 3 questions/call. | **One contract**: `DecisionRequest(observation)` -> `DecisionResult(action, probabilities, confidence, aux, timing, raw_output)`. Each provider owns an injected `ObservationEncoder` (Jev: dict of `state`; chat: string format F). Chat adapters map a one-token letter answer (+ `top_logprobs` where the pinned endpoint supports them) onto the same result. The action glossary is **one YAML file** rendered to (a) Jev `criteria` and (b) the chat system prompt. | Keeps arms paired and cacheable; probabilities are needed for confidence gating and calibration; glossary single-source avoids prompt drift (the brief's fairness requirement). |
| 7 | **Action integer codes** | 10's `actions` table: HOVER=0, FORWARD=1, ... . 11: FORWARD=0, LEFT=1, RIGHT=2, ASCEND=3, DESCEND=4, BRAKE=5, HOVER=6, AVOID_LEFT=7, AVOID_RIGHT=8, RETURN_HOME=9, EMERGENCY_LAND=10 (= brief order). 07: mnemonic aliases F,L,R,U,D,B,H,a,b,G,X (and warns `a`/`b` are a trap). 01/11: letters A-K. | Canonical code = **definition order of the `Action` enum = 11's numbering**. `actions` table is **seeded from the enum at migration time** (never hand-typed). Wire aliases = `A..K` in enum order. Verify single-token-ness per real tokenizer (07 spike S4) before freezing. | 10's numbering would silently disagree with every other agent's logs. |
| 8 | **Scenario time units** | 06: scenario events in **seconds** (`t_s`) or **range** triggers. 10: `scenario_events.t_start_tick`. 02/08: `ctrl_hz` is a run parameter. | Scenarios are **tick-rate independent** (seconds/metres). `scenario_events` stores `t_start_s`, `t_end_s`; `flights` stores `ctrl_hz`; only the hot `decisions` table stores ticks. `scenario_key = sha256(canonical_json(spec without certified/scenario_id))`, tick-rate excluded from the hash. | Lets the same scenario run at 48 Hz and at the 240 Hz fine profile for the latency floor. |
| 9 | **Job queue table vs flights table** | 09: separate `job` table (pool, priority, lease, mode, attempts). 10: `flights` *is* the queue (`queued/running/...`, heartbeat). | **One table, `flights`**, with 10's DDL plus 09's columns `pool`, `priority`, `mode`, `attempts` (rename `attempt`), and `lease_until` in place of `heartbeat_at`. Two-statement claim + partial indexes [C 09: 369/s -> 13-18k/s]. `arms.config_sha` is part of the natural key. | YAGNI: a second table duplicates the idempotency key. 09's measured claim performance applies unchanged. |
| 10 | **Who owns latency** | 01: `LatencyInjectingProvider` decorator stamps `sim_latency_ms`. 08: harness converts latency to ticks (`ceil(L/dt)`), monotonic apply, latency trace. 01 itself: "providers never know about sim time". | **Harness owns latency.** Drop the decorator. `app/harness/latency.py`: `LatencyModel` (fixed, lognormal, empirical trace) + `LatencyPolicy` with modes `LOCKSTEP, FIXED, SAMPLED, MEASURED, REALTIME`. `DecisionTiming` keeps wall facts only. One `np.random.Generator` family everywhere (01 used `random.Random`, 06 Philox). | Single clock model (08), no double bookkeeping; providers stay trivially testable with `httpx.MockTransport`. |
| 11 | **Async API client: inside the sim process (09) or separate process (01/08)?** | 09: asyncio per process with K interleaved flights. 08/01: sim worker and API client in different processes to avoid loop lag contaminating latency. 09's own test: loop-lag p99 92 ms at C=16 under load. | **K=1 flight per API worker process**, sim paused while awaiting (pause-and-charge, `MEASURED` mode); `W_api` = 5-8 processes reaches the RPM ceiling anyway (one flight = 67 calls x 0.3 s = 20 s => 180 fl/h/process; 6 processes ~1k fl/h vs the 900 fl/h RPM bound [C 09 table]). Each process has a private event loop used only for the HTTP client. Bulk/sweep modes use `SAMPLED`/`FIXED` with no network at all. A separate **probe process** (no sim) produces the published latency distributions. K>1 interleaving is deferred until a measured shortfall. | Clean measurement beats an optimisation nobody has needed yet. |
| 12 | **Trajectory storage** | 10: Parquet chunks of 50-200 flights. 12: one Parquet per flight, tick schema with ~45 columns. 09: NPZ/Parquet atomic write. | **Per-flight Parquet** (zstd, byte-stream-split, written to `.part` then `os.replace`), `pyarrow` in the `sim` extra; chunk compaction deferred (YAGNI). Record rate = control rate; decimate non-hero flights to 12-24 Hz only if disk matters (4.5 GB at 48 Hz for 10k scenarios x 6 arms [C 12]). | DuckDB globs per-flight files fine; 10k-60k small files on APFS is ok. One schema (12's) is the replay contract. |
| 13 | **Control rate** | 02: 240/48 (20.8 ms). 06 JSON example: 30 Hz. 08: dt <= 10 ms if the 10 ms sweep point must be distinct. 11/12: 48. | **240 Hz physics / 48 Hz control default**; a `fine` profile 240/240 (4.2 ms tick) for the latency-floor experiments. Latency always reported as `nominal_ms` **and** `effective_ticks * dt`. 10 ms and 20 ms both map to 1 tick at 48 Hz and that is stated in the figure. | Cost scales with PID calls (~125 us each [C 11b]); 240 Hz control is ~5x dearer, so restrict it to a subset. |
| 14 | **SQLAlchemy version, key type** | 10: sketches verified on 2.1.1 (needs `[asyncio]` for greenlet); the template says `>=2.0.50`; guide: UUID PKs "unless schema differs". | `SQLAlchemy[asyncio]>=2.1.1,<2.2` (the version actually exercised), integer PKs (documented deviation, 36-byte UUID in a 10^7-row hot table buys nothing), `type_annotation_map = {str: Text}` for STRICT tables. | 10's measured gotchas (VARCHAR rejected by STRICT; greenlet not auto-installed). Spike F5 of 10 (2.0.50 parity) stays open. |
| 15 | **Shield: common to all arms (05) vs experimental factor (11)** | 05: one CBF-QP shield for every arm + `--no-shield`. 11: S0 (envelope + failsafe only), S1 (brake-only collision shield), S2 (deflection = arm B). 07: report raw and shielded. | `ExperimentConfig.shield in {S0, S1}` is an **experiment factor**; headline figures at S0 (decision-driven) and S1 (deployment-realistic); **S2 is arm B, not a level**. S1 is the TTC/velocity-clip filter from 11; a CBF-QP (05) is a later drop-in behind the same `Shield` Protocol. The shield senses from the same noisy `Observation` as the providers (no oracle shield). | Otherwise a 20 ms reflex beats a 300 ms LLM with no "decision" involved [C 04, 11]. |
| 16 | **Arm kinds** | 05: arms A and C emit setpoints, others emit actions. 11: executor bypassed for A/C. | Two Protocols: `DecisionProvider` (action arms, including local FSM/RL/oracle/random/hover, each wrapped so **compute time is measured and converted to ticks**; no arm gets free zero latency [C 08]) and `SetpointPolicy` (A, A', B, C). Both pass through the same `Shield` and `LowLevelController`. | One latency path, one fairness rule. |
| 17 | **RL inference dependency** | 05: SB3/PPO needs torch. 09: keep torch/SB3 out of sim workers (memory). | Train in the `ml` extra (`scripts/train_rl.py`); `scripts/export_policy.py` writes the MLP weights to `.npz`; `providers/rl_policy.py` runs a 2x128 numpy forward pass (~0.1 ms). Workers never import torch. | Keeps worker RSS ~150 MB [C 09] and the import boundary simple. |
| 18 | **RNG** | 06: Philox channels keyed by the **first 8 bytes** of the channel name (`channel.encode().ljust(8, b"\0")[:8]`), collision-prone (e.g. any two channels sharing an 8-byte prefix). 08: `SeedSequence(scenario_id, stream)`. 09: `seed_lat = hash(seed, arm_id)` (Python `hash` is salted per process). | `app/sim/rng.py::channel_rng(master_seed, scenario_index, channel: str)`, key = first 8 bytes of `sha256(channel)`; latency stream keyed `(scenario_key, arm_config_sha, rep)`; never Python `hash()`. | Flagging a latent bug in 06's sketch: paired design breaks silently if two channels alias. |
| 19 | **Data directory location** | 10: `data/` in the repo. 09: not on an iCloud/Dropbox-synced path, exclude from Spotlight. Repo lives under `~/Documents`. | `data_dir` setting defaults to `data/` (gitignored) but the headline-run preflight **warns if the resolved path is under `~/Documents`, `Mobile Documents` or a synced folder**; `.env` sets `DATA_DIR=$HOME/drone-jev-data` for real runs. | WAL `-wal/-shm` under file-sync can corrupt or conflict [C 09, 10: inference]. |
| 20 | **One DB or one per experiment (10's open question)** | 10 leans per-experiment; needs 14's call. | **One global `data/jev.db`.** | The API lists experiments; FK cascade delete per experiment works; 0.3-0.5 GB per 10k flights [C 10]; archive with `sqlite3 .backup`. Revisit above ~100k flights. |
| 21 | **Decision record shape** | 10: narrow `decisions` (obs/issue/apply tick, action, raw_action, latency_ns, ttft_ns, flags). 12 wants `tick_response`, `latency_ticks_effective`, `latency_mode`, `outcome`, `source`, latency split, oracle labels, `shield_reason`. 06: labels computed offline and versioned. | `decisions` = 10's columns + `response_tick`, `latency_ticks`, `latency_mode`, `outcome`, `source` (small ints). Split latency parts live in `model_calls` (10). **Oracle labels are a separate table** `decision_labels(flight_id, seq, label_version, admissible_mask, preferred, urgency_s)` written offline. Shield overrides are visible per tick in the trajectory (`action_source`), not per decision. | Hot table stays narrow (30-40 B/row); labels can be regenerated without rerunning flights (06). |
| 22 | **`Observation.age_ms`** | 07: `age_ms` = staleness injected by the latency model, "filled by harness". 08: state is a snapshot at issue; the provider's latency is unknown until it returns. | `age_ms` at issue = **sensor age only**. The provider's own latency cannot be in its own prompt. An optional `expected_latency_ms` (running median of that arm's measured/declared latency) supports the "pre-compensated" ablation. | Avoids a causality bug (the decision cannot know how late it will be). |

---

## 3. Architecture

### 3.1 Processes (no Docker, no Ray)

```
uvicorn main:app (control/monitor plane; async SQLAlchemy + aiosqlite; read-mostly)
   |  POST /experiments (enqueue flights) | GET progress SSE | GET summary | GET replay JSON
   v
data/jev.db  (SQLite WAL, STRICT)  <----- claim / heartbeat / complete (sync sqlite3)
   ^
scripts/run_experiment.py  (supervisor; NOT inside uvicorn; `caffeinate -dimsu -w <pid>`)
   |-- pool "cpu"   : N spawn workers, 1 flight at a time (classical + local-policy arms)
   |-- pool "api"   : M spawn workers, K=1 flight, private asyncio loop for httpx (Jev / chat arms)
   |-- pool "local" : 1-2 workers + 1 local model server (Ollama/MLX), optional
   '-- probe        : scripts/probe_latency.py, no sim, publishes latency distributions
data/flights/<experiment>/<flight_id>.parquet   (trajectory+decisions per flight, atomic rename)
```

Worker lifecycle: `spawn` -> `configure_logging()` -> build backend/providers via `app/harness/factory.py` (composition root) -> loop {`claim_next` -> run flight (async coroutine, sim stepped synchronously inside) -> write `.part` -> `os.replace` -> one `BEGIN IMMEDIATE` transaction: insert decisions + metrics + `status='complete'` -> next}. Recycle the worker every ~500 flights as a leak guard [C 09].

Flight loop (one code path for all arms and all time modes; mode is data, 08's option (c)/(d)/(a)/(b)):

```python
async def run_flight(ctx: FlightContext) -> FlightResult:
    backend.reset(ctx.scenario, ctx.seed)
    for tick in range(ctx.max_ticks):
        for reply in queue.due(
            tick
        ):  # monotonic-apply: drop if issue older than applied
            executor.on_decision(reply.action, reply.obs_tick, tick, state)
        if cadence.should_issue(tick, queue):
            obs = backend.sense()  # rays at decision rate, not physics rate
            request = DecisionRequest(ctx.next_id(), tick, obs, ctx.deadline_ms)
            result = await safe_decide(
                provider, request
            )  # InvalidDecisionOutputError/Timeout -> KEEP
            queue.push(latency_policy.resolve(result, tick), request, result)
        sp = arm.setpoint(
            executor, policy, obs, state
        )  # executor for action arms, policy for setpoint arms
        sp = shield.filter(sp, obs, state).setpoint  # S0/S1, identical for every arm
        step = backend.step(sp)  # backend owns LowLevelController (DI)
        collectors.on_step(step)
        if step.truth.in_collision or landed:
            break
    return collectors.finalize()
```

`LatencyPolicy.resolve`: `LOCKSTEP` -> 0 ticks (labelled ablation only); `FIXED` -> `SimClock.ticks_for_ms(ms)`; `SAMPLED` -> `LatencyModel.sample_ms(rng)` (empirical trace from the probe); `MEASURED` -> `result.timing.wall_total_ms` (pause-and-charge); `REALTIME` -> real-time pacing in the validation set only. All conversions use `ceil`; any nonzero latency costs >= 1 tick [C 08].

### 3.2 Layering and import rules (what replaces "routes/services/repositories" for sim code)

```
L0  utils/         pure, stateless (hashing, braking, intervals, pagination, response, save_file)
L1  schemas/       pydantic I/O + config models (Action, ScenarioSpec, ExperimentConfig, DecisionResult, DTOs)
L2  domain/        frozen dataclasses + Protocols + DomainError (stdlib + numpy + schemas + utils)
L3  implementations (worker-side, never import fastapi/sqlalchemy):
      sim/ controllers/ providers/ oracle/ scenarios/
L4  harness/       flight loop, latency, metrics, recorder, store (sqlite3), worker, supervisor, factory
L5  scripts/       composition roots (argument-free CLIs)           api/ services/ repositories/ models/ db/  (API branch)
```

Rules (enforced by a stdlib-`ast` script, `scripts/validate/check_imports.py`, since Ruff is not configured for it):

1. Nothing in L0-L4 imports `fastapi`, `starlette`, `sqlalchemy`, `app.api`, `app.db`, `app.models`, `app.repositories`, `app.services`, or `torch`.
2. `pybullet` and `gym_pybullet_drones` are imported only by `sim/pybullet_backend.py`, `sim/scenario_aviary.py`, `controllers/dsl_pid.py`. `sim/surrogate.py`, `sim/rng.py` and all of `domain/` are pybullet-free (so the API, analysis and oracle can import them).
3. `providers/` imports `domain`, `schemas`, `utils`, `httpx` only (not `sim`, `controllers`, `harness`). The oracle provider lives in `oracle/` because it needs the surrogate and privileged state.
4. Concrete classes are wired only in `harness/factory.py` and `scripts/`; everything else depends on Protocols from `domain/protocols.py` (Dependency Inversion, guide SOLID rule).
5. The API branch (`api -> services -> repositories -> models/db`) may import `schemas`, `utils`, `scenarios` (generator), `analysis`; never `sim`, `controllers`, `providers`, `harness`. The harness never imports `models` or `repositories` (it speaks sqlite3 through `harness/store.py`, stdlib only, importable from any env).
6. `utils/` stays pure; numpy is allowed, state/IO/network are not.

### 3.3 How the guide's layer table maps onto sim/harness code

| Guide layer | Applies as-is | Sim/harness equivalent and the deviation |
|---|---|---|
| `routes/` (parse, call service, respond) | yes, path is `app/api/v1/` | Only for the control plane. No DB queries, no business logic. |
| `services/` (orchestrate; no HTTP types) | yes | `experiment_service` (enqueue, progress, pause), `report_service`, `replay_service`. Class-based, DI. |
| `repositories/` (all DB access) | yes for the **API side** (async SQLAlchemy) | Workers use `harness/store.py` (sync `sqlite3`): a second, stdlib repository implementing the `JobQueue` Protocol. Deviation flagged: "async for all DB ops" and "sessions only via `Depends()`" do not apply outside request handling [C 10]. |
| `schemas/` (Pydantic DTO, no DB/service imports) | yes | Also holds `ScenarioSpec`, `ExperimentConfig`, `Action`, `DecisionResult`. Hot-path per-tick objects (`Observation`, `Setpoint`, `DecisionRequest`) are **frozen `slots` dataclasses in `domain/`** instead of Pydantic (numpy arrays, 48 Hz x 10^6 ticks); deviation flagged. |
| `models/` (ORM) | yes | Metadata for Alembic and the API read paths only. Workers never touch the ORM (ORM inserts ~85x slower than `executemany` [C 10]). |
| `utils/` (pure, stateless) | yes | Braking/stopping math, Wilson/bootstrap CIs, canonical-JSON hashing, pagination. DRY rule: used in >= 2 places -> here. |
| "services and repositories are class-based with injected deps" | yes | Extended to `FlightRunner`, `ActionExecutor`, providers, backends. Pure numerics (surrogate rollout, labeller inner loop) may be module-level functions over arrays; the guide's <= 30-line limit applies, so the 60-90-line FSM is split into named rule functions. |
| "no `print`" | yes | gym-pybullet-drones `print`s 10+ lines per env construction and calls `exit()`/sets `KMP_DUPLICATE_LIB_OK` [C 02]: confined to `sim/scenario_aviary.py` and `controllers/dsl_pid.py` with `contextlib.redirect_stdout` into the logger; never leaks into the API process (rule 2 above). |
| "no bare `except Exception`" | yes | The worker's per-job crash guard catches `Exception`, calls `logger.exception`, records `status='failed'` with the error and continues (handle-and-record, not swallow); providers catch `httpx.HTTPError`/`TimeoutException` specifically. |

### 3.4 Where the guide conflicts with reality (flag, resolve, do not silently fix)

| # | Guide says | Reality | Resolution |
|---|---|---|---|
| 1 | Examples use `ApiResponse[T]`, `PaginatedResponse[T]`, "shared pagination utility" | Repo has `APIResponse` (different casing), no `PaginatedResponse`, no pagination util; helpers return `JSONResponse`, routes are not typed with `response_model` | Keep the `JSONResponse` pattern. Create `utils/pagination.py` (`PageParams`, `PaginatedResponse[T]`) in M5 before the first list endpoint; decisions/flights lists use `page`/`page_size` (<= 1,440 decisions/flight makes OFFSET a non-issue; 10's keyset idea is dropped). |
| 2 | Layer is `routes/` | Directory is `app/api/v1/` | Follow the directory; the guide's own tree says so. |
| 3 | `logging.py (JSON)` | Plain-text formatter since e6c11f9; `main.py` comment still says JSON | Leave; flag in the PR that touches `main.py`. |
| 4 | Imports sorted by Ruff isort rules | No ruff config; default rules do not include `I` | Recommend `ruff.toml` (6.6). Until then import order is a manual convention. |
| 5 | "UUID primary keys unless schema differs" | Integer PKs chosen (hot tables; single local file) | Documented deviation; expose stable human keys (`experiments.name`, `scenario_key`). |
| 6 | "All HTTP errors live in `exceptions.py`; subclass `AppError`" | Harness/provider failures are not HTTP errors (01 left this to 14) | `app/domain/errors.py`: `DomainError` -> `ProviderError` (`ProviderTimeoutError`, `ProviderRateLimitedError`, `InvalidDecisionOutputError` carrying the raw text), `SimulationError`, `ScenarioInfeasibleError`, `ConfigError`. API-facing not-found (`ExperimentNotFoundError`, `FlightNotFoundError`, `ScenarioNotFoundError`) subclass `NotFoundError` in `app/core/exceptions.py`; services translate `DomainError` -> `AppError` at the boundary only. |
| 7 | "Scripts: constants at top instead of CLI args (unless asked)" | The runner needs a config path and a few verbs | Keep the rule: one argument-free script per verb; the experiment config is a module constant or the `EXPERIMENT_CONFIG` env var (read through `Settings`). If the user wants `argparse`, they must ask. Scripts add `ROOT_DIR` to `sys.path` at module top, which also runs in `spawn` children (they re-import `__main__`); every multiprocessing entry needs `if __name__ == "__main__":`. |
| 8 | "Tests are optional, only when explicitly asked" | Research asks for a testing strategy | Section 10 defines it; nothing is written until requested. Scientific validity checks (determinism, calibration, feasibility certificates, paired-design invariants) are **validation scripts under `scripts/validate/`**, not pytest tests, so they are not "tests" in the guide's sense. |
| 9 | "Functions <= ~30 lines" and "avoid `Any`" | Numeric code and untyped pybullet/gpd | Split into named helpers; `NDArray[np.float64]` aliases (`Vec3`); `# type: ignore[...]` with reason confined to the three pybullet-touching files. |
| 10 | "Pagination: all list endpoints" | Replay JSON, trajectory and SSE are not lists | Paginate lists only; replay/trajectory endpoints return bounded, documented payloads. |
| 11 | Stack lists "Docker (slim + uv)" | Local-only, no Docker; slim image cannot compile pybullet | Recommend deleting the Docker files and updating the guide's Stack/Commands lines in the same change. |

---

## 4. File-level skeleton

Milestones (M0-M6) are defined in section 12. `[edit]` = exists today. Everything else is new. "W" = worker-side (no FastAPI/SQLAlchemy), "A" = API-side.

### 4.1 Repo root

| Path | Purpose | M |
|---|---|---|
| `main.py` [edit] | add `lifespan` (dispose engine, `PRAGMA optimize`); mount static `/demo` | M5 |
| `requirements.txt` [edit] | becomes `-r requirements/core.txt` (keeps the documented `uv pip install -r requirements.txt` command working) | M0 |
| `requirements/{core,sim,ml,analysis,dev}.txt` | dependency split (6.1) | M0 |
| `alembic.ini`, `migrations/` (`env.py` with `render_as_batch=True`, `versions/0001_initial.py`) | schema source of truth is the ORM; revision diffed against 10's DDL | M1 |
| `ruff.toml` (recommended, optional) | adds `I, UP, B, TID, T20` (6.6) | M0 |
| `configs/experiments/*.yaml` | committed experiment definitions: `smoke.yaml`, `latency_sweep_v1.yaml`, `headline_v1.yaml` | M1 |
| `configs/prompts/actions.yaml` | **single source** of action glossary (name, alias, description) -> Jev criteria + chat prompt | M4 |
| `configs/prompts/chat_system_v1.txt`, `chat_fewshot_v1.yaml` | versioned prompt pieces (hashed into the manifest) | M4 |
| `data/` (gitignored) | `jev.db`, `flights/`, `scenarios/`, `cache/`, `reports/`, `wheelhouse/` | - |
| `.env.sample` [edit] | add the keys in 7.2 | M0 |
| `.gitignore` [edit] | add `data/`, `*.db`, `*.db-wal`, `*.db-shm`, `*.parquet`, `*.npz`, `*.rrd`, `reports/`, `wheelhouse/` | M0 |

### 4.2 `app/core/`, `app/db/`, `app/models/` (DB and config)

| Path | Purpose | M |
|---|---|---|
| `app/core/config.py` [edit] | extend `Settings` (7.1) | M0 |
| `app/core/exceptions.py` [edit] | add `ExperimentNotFoundError`, `FlightNotFoundError`, `ScenarioNotFoundError` (subclass `NotFoundError`) | M5 |
| `app/db/base.py` | `Base` with naming convention and `type_annotation_map = {str: Text}` (10) | M1 |
| `app/db/database.py` [edit, empty] | async engine + `async_sessionmaker` + `get_session` + pragma `connect` hook | M1 |
| `app/models/{experiment,arm,scenario,scenario_event,flight,decision,model_call,model_cache,flight_metric,event_outcome,decision_label}.py` | `Mapped[T]` models, STRICT, `WITHOUT ROWID` where 10 says so; `flight.py` also carries `pool/priority/mode/attempts/lease_until` | M1 |

### 4.3 `app/schemas/` (L1, pydantic)

| Path | Purpose | M |
|---|---|---|
| `action.py` | `Action` StrEnum (definition order = code), `ACTION_CODES`, `WIRE_ALIASES` | M0 |
| `scenario.py` | `ScenarioSpec`, `EventSpec` discriminated union on `type` (06's names: `obstacle_pop_up`, `wind_step`, `target_move`, `sensor_degradation`, `battery_drop`, `gps_loss`, `intruder_drone`), `canonical_json()`, `scenario_key` | M1 |
| `experiment.py` | `ExperimentConfig`, `ArmConfig`, `SimProfile`, `TimeModelConfig`, `ShieldLevel`, `ExecutionConfig`, `BudgetConfig` (loaded from YAML) | M1 |
| `decision.py` | `DecisionResult`, `DecisionTiming`, `ProviderCapabilities` (from 01), `DecisionResponse` (API) | M0 |
| `metrics.py` | `FlightMetricsResponse`, `ArmSummaryResponse`, `EventOutcomeResponse` | M5 |
| `experiment_api.py`, `flight_api.py` | `ExperimentCreate/Response`, `FlightResponse`, progress DTO | M5 |

### 4.4 `app/domain/` (L2: types, Protocols, errors)

| Path | Purpose | M |
|---|---|---|
| `observation.py` | `Observation`, `Threat`, `DroneState`, `GroundTruth` (frozen, `slots`) | M0 |
| `setpoint.py` | `Setpoint` (mode VEL/POS, source PROVIDER/SHIELD/FAILSAFE/DEFAULT) | M0 |
| `records.py` | `DecisionRequest`, `DecisionRecord`, `StepResult`, `FlightMetrics`, `ClaimedJob`, `FlightResult` | M0 |
| `protocols.py` | every cross-layer interface (section 5) | M0 |
| `clocks.py` | `SimClock` (ticks<->s, `ticks_for_ms`), `WallClock` (`perf_counter_ns`) | M0 |
| `errors.py` | `DomainError` tree (3.4 item 6) | M0 |

### 4.5 `app/sim/` (worker; backend, events, sensors)

| Path | Purpose | M |
|---|---|---|
| `rng.py` | `channel_rng(master_seed, scenario_index, channel)` (sha256-keyed Philox); pre-generated per-step arrays so arms cannot shift each other's draws | M1 |
| `scenario_aviary.py` | `ScenarioAviary(CtrlAviary)`: `obstacles=False`, parked kinematic obstacle pool, `_physics` hook for wind + rpm modifiers, `_housekeeping` resets controllers/events/battery, collision via `getContactPoints`; wraps gpd `print`/`exit` | M1 |
| `pybullet_backend.py` | `PyBulletBackend` implements `SimBackend`; takes `LowLevelController`, `SensorSuite`, injectors via `__init__` | M1 |
| `sensors.py` | `SensorSuite`: 16-ray ring (22.5 deg, 10 m) via `rayTestBatch` at decision rate; derived `Threat`s by finite difference; separate RNG stream | M1 |
| `events/{obstacle,wind,target,sensor_noise,battery,gps,intruder}.py` | `EventInjector` implementations; wind = mean step + OU, battery = rpm scale by V/Vref, GPS/sensor faults in the observation layer only (v1) | M1 obstacle; M3 rest |
| `events/registry.py` | `EventSpec.type` -> injector class map (Open/Closed: a new event is a new file + one line) | M1 |
| `surrogate.py` | pybullet-free point-mass + first-order lag + accel-limit model; used by certifier, oracle, labeller (and `KinematicBackend` later) | M2 |
| `kinematic_backend.py` | `KinematicBackend` implements `SimBackend` over the surrogate; **gated** (M6) | M6 |
| `calibration.py` | step/brake/lateral/wind response fits -> `a_brake`, `a_lat`, `tau_a`, per-speed stopping distances (`calibration_id`) | M2 |

### 4.6 `app/controllers/` (worker)

| Path | Purpose | M |
|---|---|---|
| `dsl_pid.py` | `DslPidController` (`LowLevelController`): carrot/velocity and position-latch modes, yaw ramp by integrating `target_rpy[2]`, `ctrl.reset()`, `print` guard | M1 |
| `executor.py` | `ActionExecutor`: two-channel latch (H, V), AVOID modifiers with TTL, displacement-bounded LEFT/RIGHT/ASCEND/DESCEND, BRAKE-then-latch, terminal EMERGENCY_LAND, dwell/side-commit/slew (11) | M1 |
| `shield.py` | `Shield` implementations S0 (envelope + failsafe) and S1 (brake-only collision filter) | M3 |
| `waypoint.py` | arm A/A': straight-line waypoint `SetpointPolicy` | M3 |
| `reactive.py` | arm B1 (TTC-brake + lateral nudge); `apf.py`, `vfh.py` ablations | M3 / later |
| `mpc_osqp.py` | arm C: double-integrator MPC, raw OSQP, N=20, dt=0.1, half-space obstacles | M6 |

### 4.7 `app/providers/` (worker; all `DecisionProvider` implementations)

| Path | Purpose | M |
|---|---|---|
| `encoding.py` | `ObservationEncoder` implementations: `HybridTerseEncoder` (format F, default), `KeyValueEncoder` (B), `JsonEncoder` (A), `SectorEncoder` (C), `AsciiGridEncoder` (D), `NaturalLanguageEncoder` (E), `JevStateEncoder` (dict, semantic buckets) | M4 |
| `parsing.py` | action parser: exact -> unique extraction -> fail (never guess on ambiguity) | M4 |
| `prompts.py` | loads `configs/prompts/*`, renders Jev `criteria` and chat system prompt from the one glossary | M4 |
| `mock.py` | `MockProvider`: delegates to a callable (oracle/scripted) with a configurable `p_miss`; no sleeping, no network | M1 |
| `jev.py` | `JevDecisionProvider` (`POST /api/alpha/decisions` or `api.typesafe.ai/v1/systemone`), injected `httpx.AsyncClient(http2=True)`, pinned `typesafe/jev-1.13`, logs returned snapshot id | M4 |
| `chat.py` | `ChatDecisionProvider` (OpenRouter chat, letter answer, `max_tokens` 1-4, `provider.order` + `allow_fallbacks:false`, optional `top_logprobs`) | M4 |
| `system_one_local.py` | Ollama `/v1/systemone` stand-in (same body as Jev) | M4 |
| `replay.py` | `ReplayProvider` over `model_cache` (action replay; latency comes from the recorded trace) | M4 |
| `fsm.py` | arm E rule FSM over the 11 actions, split into rule functions | M3 |
| `baselines.py` | Z0 random, Z1 always-HOVER, Z2 always-FORWARD, Z3 panic | M3 |
| `rl_policy.py` | numpy-forward MLP policy loaded from `.npz` (arm D) | M6 |
| `factory.py` | `build_provider(ArmConfig, settings) -> DecisionProvider` (only place that maps `kind` to class) | M1 |

### 4.8 `app/oracle/`, `app/scenarios/` (worker and API-importable, pybullet-free)

| Path | Purpose | M |
|---|---|---|
| `oracle/labeller.py` | viability-set labels via surrogate rollouts: admissible set, preferred, urgency, validity; `label_version` | M2 |
| `oracle/provider.py` | arms O / O-with-latency as `DecisionProvider` | M3 |
| `scenarios/generator.py` | budget-first generator (`d = d_min0 + v_c*(B + t_det)`), tiers T0-T5 + T-1 control, range-triggered world-fixed spawns | M2 |
| `scenarios/certifier.py` | Oracle-0 replay in the real backend, rejection sampling, stores `certified` | M2 |
| `scenarios/suites.py` | named suites (`smoke`, `headline_v1`, `train`, `compound`), seed ranges (train/tune/test disjoint, 05) | M2 |

### 4.9 `app/harness/` (worker orchestration)

| Path | Purpose | M |
|---|---|---|
| `flight_runner.py` | `FlightRunner.run(ctx) -> FlightResult` (the loop in 3.1) | M1 |
| `decision_loop.py` | cadence policies (`sequential`, `periodic(P)`, `event_triggered`), `LatencyQueue` with monotonic-apply and timeout fallback | M1 |
| `latency.py` | `LatencyModel` impls (`FixedLatency`, `LogNormalLatency`, `EmpiricalLatency`) + `LatencyPolicy` (5 modes) | M1 |
| `metrics.py` | `MetricCollector` impls: collision/near-miss/clearance, reaction time, path length, oscillation, recovery, invalid/timeout rates, four throughput metrics (08) | M1 |
| `recorder.py` | per-tick Parquet writer (12's schema), `.part` -> atomic rename | M1 |
| `store.py` | **stdlib-only** `SqliteJobQueue` (`JobQueue`) + `FlightWriter`; pragmas, `BEGIN IMMEDIATE`, two-statement claim, lease/heartbeat | M1 |
| `worker.py` | worker main loop, crash guard, recycle counter | M1 |
| `supervisor.py` | spawn pools, restart dead workers, SIGINT graceful, `$` ledger hard stop, pool-size rules (9 cpu / 6+2+1 mixed) | M3 |
| `manifest.py` | reproducibility manifest (08 sec 12): git SHA + dirty flag, package versions, hardware, config hashes, prompt hashes, latency-trace hash | M1 |
| `budget.py` | token/cost ledger with `BUDGET_USD` hard stop | M4 |
| `cache.py` | `model_cache` read/write; `cache_hit` never contributes to latency stats | M4 |
| `factory.py` | composition root: `ExperimentConfig` -> backend, controller, shield, arm, collectors | M1 |

### 4.10 `app/analysis/`, `app/replay/`, API branch

| Path | Purpose | M |
|---|---|---|
| `analysis/views.sql` | DuckDB/SQLite views (`v_flight_summary` from 10 + per-arm aggregates) | M5 |
| `analysis/stats.py` | Wilson, Newcombe, exact McNemar, Wilcoxon, BCa bootstrap, Holm, KM; power tables | M5 |
| `analysis/figures.py` | P1-P14 of 12 (matplotlib Agg, fixed rcParams, SVG+PNG + caption JSON) | M5 |
| `analysis/report.py`, `analysis/templates/report.html.j2` | Jinja2 report with manifest, underpowered banner | M5 |
| `replay/demo_export.py`, `replay/rerun_export.py` | stored flights -> demo JSON / `.rrd` (one recording per scenario, all arms) | M5 |
| `static/demo/index.html` | single-file canvas player + SVG Gantt | M5 |
| `app/repositories/{experiment,flight,scenario,decision}_repository.py` | async SQLAlchemy read paths (+ bulk enqueue via Core) | M5 |
| `app/services/{experiment,report,replay}_service.py` | enqueue/progress/pause, report, replay JSON | M5 |
| `app/api/v1/{experiments,flights,scenarios,reports}.py`, `router.py` [edit] | `GET/POST /experiments`, `GET /experiments/{experiment_id}/{summary,progress,flights,events,report}`, `GET /flights/{flight_id}/{details,decisions,trajectory}`, `GET /scenarios/{scenario_id}/replay`. `progress` is SSE (`fastapi.sse.EventSourceResponse` exists in the installed FastAPI [C 12]). Every path ends with a resource word per the guide. | M5 |

### 4.11 `utils/`, `scripts/`, `tests/`

| Path | Purpose | M |
|---|---|---|
| `utils/hashing.py` | `canonical_json(obj)`, `sha256_hex`, float rounding (10 digits) for `scenario_key` | M1 |
| `utils/braking.py` | `stopping_distance(v, t_lat, a, tau, r, margin)`, `budget_s(...)` shared by generator, FSM, shield, labeller (DRY, >= 4 users) | M1 |
| `utils/intervals.py` | Wilson CI, Newcombe difference (used by analysis **and** the API summary) | M5 |
| `utils/pagination.py` | `PageParams`, `PaginatedResponse[T]` (flagged gap) | M5 |
| `scripts/setup_sim_env.sh` | pybullet with `CFLAGS`, gpd at pinned SHA with `--no-deps`, smoke test | M0 |
| `scripts/init_db.py` | `alembic upgrade head`, seed `actions` from the enum | M1 |
| `scripts/enqueue_experiment.py` | YAML -> scenarios + arms + flights rows (idempotent) | M1 |
| `scripts/run_experiment.py` | supervisor entry (constants at top) | M3 |
| `scripts/probe_latency.py` | from 01's benchmark; writes JSONL + fitted `EmpiricalLatency` trace | M4 |
| `scripts/calibrate_sim.py` | runs `sim/calibration.py`, writes `calibration_id` JSON | M2 |
| `scripts/generate_scenarios.py` | suite -> certified scenarios in DB | M2 |
| `scripts/bench_sim_throughput.py` | B1/B2 of 09 on a quiet machine (gate) | M1 |
| `scripts/train_rl.py`, `scripts/export_policy.py` | PPO (ml env) and numpy export | M6 |
| `scripts/make_report.py`, `scripts/export_replay.py` | analysis/replay entry points | M5 |
| `scripts/validate/{check_imports,check_determinism,check_paired_invariants,check_tiers,check_env}.py` | CI-less validation (section 11) | M0-M6 |
| `tests/unit/`, `tests/integration/`, `tests/api/` (+ existing `conftest.py`, `test_health.py`) | only when requested (section 10) | - |

---

## 5. Typing Protocols (deduplicated, type-checked)

Provenance: `DecisionProvider`/`DecisionResult`/`DecisionTiming`/`ProviderCapabilities`/`LatencyModel` from 01 (with `sim_latency_ms` and the decorator removed, `raw_output` added, RNG unified to numpy `Generator`); `Observation`/`Threat` from 07 (plus `age_ms` semantics from reconciliation #22); `SimBackend` from 03 (plus `sense()`/`state_estimate()` so the controller is injected, not owned by the caller); `Setpoint`, executor inputs, `Shield` from 11; `EventInjector` from 06/02 split by interface segregation; `MetricCollector` from 08; `JobQueue` from 09/10. Merged into one file, `mypy --strict` clean on 3.14.2 (scratch `protocols_sketch.py`); in the repo each block is its own module as listed in section 4.

```python
# ---- app/schemas/action.py -------------------------------------------------------------
class Action(StrEnum):
    """Definition order == canonical integer code (0..10). DB `actions` is seeded from this."""

    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ASCEND = "ASCEND"
    DESCEND = "DESCEND"
    BRAKE = "BRAKE"
    HOVER = "HOVER"
    AVOID_LEFT = "AVOID_LEFT"
    AVOID_RIGHT = "AVOID_RIGHT"
    RETURN_HOME = "RETURN_HOME"
    EMERGENCY_LAND = "EMERGENCY_LAND"  # (one member per line in the repo)


ACTION_CODES: dict[Action, int] = {a: i for i, a in enumerate(Action)}
WIRE_ALIASES: dict[Action, str] = dict(zip(Action, "ABCDEFGHIJK", strict=True))


# ---- app/schemas/decision.py -----------------------------------------------------------
class DecisionTiming(BaseModel):  # wall-clock facts only
    wall_total_ms: float
    wall_ttfb_ms: float | None = None
    wall_first_token_ms: float | None = None
    server_ms: float | None = None
    connection_reused: bool | None = None
    retries: int = 0


class DecisionResult(BaseModel):
    action: Action
    probabilities: dict[Action, float] = Field(default_factory=dict)  # empty if none
    confidence: float | None = None
    aux: dict[str, float] = Field(default_factory=dict)  # Jev noul/score side answers
    timing: DecisionTiming
    provider_id: str  # includes pinned snapshot, e.g. "jev:typesafe/jev-1.13-20260917"
    raw_output: str | None = None
    cache_hit: bool = False
    input_tokens: int | None = None
    cost_usd: float | None = None


class ProviderCapabilities(BaseModel):
    returns_probabilities: bool
    supports_aux_questions: bool
    deterministic: bool
    max_state_tokens: int
    is_remote: bool


# ---- app/domain/observation.py, setpoint.py, records.py --------------------------------
Vec3 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Threat:
    range_m: float
    bearing_deg: float
    closing_speed_mps: float
    ttc_s: float


@dataclass(frozen=True, slots=True)
class Observation:  # noisy, body frame, explicit None for lost sensors
    sim_tick: int
    age_ms: float  # sensor age at issue, NOT the provider's own latency
    vel_body_mps: Vec3
    rays_m: Vec3  # 16 rays, 10.0 = no hit
    nearest_obstacle: Threat | None
    other_drone: Threat | None
    target_body_m: Vec3 | None
    home_body_m: Vec3 | None
    battery_frac: float
    gps_ok: bool
    last_action: Action | None
    expected_latency_ms: float | None = (
        None  # optional "pre-compensation" ablation input
    )


@dataclass(frozen=True, slots=True)
class DroneState:  # world frame; truth (physics/metrics) or estimate (controllers)
    pos: Vec3
    quat: NDArray[np.float64]
    vel: Vec3
    ang_vel: Vec3
    yaw: float


@dataclass(frozen=True, slots=True)
class GroundTruth:  # privileged: oracle, labeller, metrics only
    drone: DroneState
    clearance_min_m: float
    in_collision: bool
    hazard_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Setpoint:
    mode: Literal["VEL", "POS"]
    pos: Vec3
    vel: Vec3
    yaw: float
    source: Literal["PROVIDER", "SHIELD", "FAILSAFE", "DEFAULT"]
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    request_id: str
    obs_tick: int
    observation: Observation
    deadline_ms: float
    allowed_actions: tuple[Action, ...] = tuple(Action)


@dataclass(slots=True)
class DecisionRecord:  # one row of `decisions`
    seq: int
    obs_tick: int
    issue_tick: int
    response_tick: int | None
    apply_tick: int | None
    action: Action
    raw_action: Action | None
    latency_ticks: int
    latency_ns: int
    outcome: Literal[
        "ok", "invalid", "timeout", "transport_error", "superseded", "shield_rejected"
    ]
    source: Literal["live", "cache", "local", "classical"]
    flags: int = 0


# ---- app/domain/clocks.py ---------------------------------------------------------------
class Clock(Protocol):
    def now_ns(self) -> int: ...  # WallClock = perf_counter_ns; FakeClock for tests


@dataclass(frozen=True, slots=True)
class SimClock:
    ctrl_hz: int
    pyb_hz: int

    @property
    def dt_s(self) -> float:
        return 1.0 / self.ctrl_hz

    def ticks_for_ms(
        self, ms: float
    ) -> int:  # ceil; any nonzero latency costs >= 1 tick
        return 0 if ms <= 0 else max(1, -int(-ms * self.ctrl_hz // 1000))


# ---- app/domain/protocols.py ------------------------------------------------------------
@runtime_checkable
class DecisionProvider(Protocol):
    """One stateless decision per call. Returns wall timing; never applies fallback, never sees sim time.
    Raises ProviderTimeoutError / ProviderRateLimitedError / InvalidDecisionOutputError."""

    @property
    def provider_id(self) -> str: ...
    @property
    def capabilities(self) -> ProviderCapabilities: ...
    async def decide(self, request: DecisionRequest) -> DecisionResult: ...
    async def aclose(self) -> None: ...


class ObservationEncoder[T](Protocol):  # Jev: T = dict[str, object]; chat: T = str
    @property
    def format_id(self) -> str: ...  # "F", "A", ... recorded in the manifest
    def encode(self, observation: Observation) -> T: ...


class SetpointPolicy(Protocol):  # arms A, A', B, C
    def compute(
        self, observation: Observation, state: DroneState, tick: int
    ) -> Setpoint: ...
    def reset(self) -> None: ...


class LowLevelController(Protocol):  # the ONE shared DSLPID adapter
    def compute(
        self, setpoint: Setpoint, state: DroneState, dt_s: float
    ) -> NDArray[np.float64]: ...
    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ShieldResult:
    setpoint: Setpoint
    overridden: bool
    reason: str = ""


class Shield(Protocol):  # S0 / S1 are implementations; same for every arm
    def filter(
        self, setpoint: Setpoint, observation: Observation, state: DroneState
    ) -> ShieldResult: ...
    def reset(self) -> None: ...


class ForceModel(Protocol):  # wind; indexed by physics step, never wall time
    def force_world_n(self, physics_step: int, state: DroneState) -> Vec3: ...


class RpmModifier(Protocol):  # battery derating
    def scale_rpm(
        self, physics_step: int, rpm: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


class ObservationFilter(Protocol):  # noise/dropout/GPS loss: observation path only
    def apply(self, tick: int, observation: Observation) -> Observation: ...


class EventInjector(Protocol):  # lifecycle; may also implement the three above
    @property
    def event_id(self) -> str: ...
    def reset(self, scenario: ScenarioSpec, seed: int) -> None: ...
    def on_tick(self, tick: int, state: DroneState) -> None: ...


class SimBackend(Protocol):  # PyBulletBackend (v1), KinematicBackend (M6, gated)
    @property
    def clock(self) -> SimClock: ...
    def reset(self, scenario: ScenarioSpec, seed: int) -> None: ...
    def sense(self) -> Observation: ...  # rays cast here, at decision rate
    def state_estimate(self) -> DroneState: ...
    def step(
        self, setpoint: Setpoint
    ) -> (
        StepResult
    ): ...  # events, physics substeps, contacts; owns the LowLevelController
    def close(self) -> None: ...


class MetricCollector(Protocol):
    def on_step(self, result: StepResult) -> None: ...
    def on_decision(self, record: DecisionRecord) -> None: ...
    def finalize(self) -> FlightMetrics: ...


class LatencyModel(
    Protocol
):  # fixed / lognormal / empirical trace; one numpy Generator family
    def sample_ms(self, rng: np.random.Generator) -> float: ...


class JobQueue(
    Protocol
):  # sync sqlite3 impl in harness/store.py; the seam where Ray would plug in
    def claim_next(
        self, pool: str, worker_id: str, now_ms: int
    ) -> ClaimedJob | None: ...
    def heartbeat(self, flight_id: int, now_ms: int) -> None: ...
    def complete(
        self,
        job: ClaimedJob,
        metrics: FlightMetrics,
        decisions: Sequence[DecisionRecord],
    ) -> None: ...
    def fail(self, job: ClaimedJob, error: str) -> None: ...
```

Notes on what was deduplicated or dropped:
- `Scenario` from the brief's list is `ScenarioSpec` (pydantic), not a Protocol (nothing has two implementations).
- `Clock` is a real Protocol only for wall time (testability); sim time is the concrete `SimClock`.
- 01's `LatencyInjectingProvider` and `sim_latency_ms` are removed (reconciliation #10). 03's `SimBackend.step(setpoint, n_ticks)` becomes one control tick per call; the loop owns cadence.
- `ActionExecutor` has no Protocol (single implementation, YAGNI); `ActionProfile` (speeds, TTLs, slew) is a pydantic model in `ExperimentConfig`.
- The Jev adapter renders `state` via `JevStateEncoder` and builds `questions` from `configs/prompts/actions.yaml`; the chat adapter renders format F and asks for one letter. Aux answers (`risk`, `obstacle_imminent`) ride in `aux`; they are never used by the harness for control unless an ablation says so.

---

## 6. Dependency strategy

### 6.1 Exact requirement files (floors = versions actually measured on 3.14.2/arm64 by other agents; `>=` style matches the template)

```
# requirements/core.txt   (API + harness + domain; what `requirements.txt` includes)
fastapi>=0.136.3
uvicorn>=0.49.0
pydantic>=2.13.4
pydantic-settings>=2.14.1
PyYAML>=6.0            # UNVERIFIED on 3.14.2 wheels; Phase-0 check. Alternative: JSON experiment configs.
SQLAlchemy[asyncio]>=2.1.1,<2.2
aiosqlite>=0.22.1
alembic>=1.20.0
httpx[http2]>=0.28.1   # moved from dev: providers need it; [http2] pulls h2
numpy>=2.5.3

# requirements/sim.txt    (worker env; pybullet + gpd are installed by scripts/setup_sim_env.sh)
-r core.txt
scipy>=1.18.1          # DSLPIDControl uses scipy Rotation
gymnasium>=1.3.0
transforms3d           # gpd hard dep
pyarrow>=25.0.1        # per-flight Parquet
osqp>=1.1.3            # arm C (raw OSQP; cvxpy/casadi/acados deliberately excluded)
# Phase-0 spike 6 (02): add only what `import gym_pybullet_drones.envs.CtrlAviary` actually demands,
# candidates: matplotlib, pillow, control. Do NOT add torch or stable-baselines3 here.

# requirements/ml.txt     (RL training; never imported by workers)
-r sim.txt
torch>=2.14.0
stable-baselines3>=2.9.0
# optional local-model arm (unmeasured): mlx-lm

# requirements/analysis.txt
-r core.txt
pandas>=3.0.6
pyarrow>=25.0.1
duckdb>=1.5.6
scipy>=1.18.1
statsmodels>=0.15.0
matplotlib>=3.11.2
jinja2>=3.1.6
rerun-sdk>=0.38.1      # optional engineer replay; disable analytics in the environment

# requirements/dev.txt
-r core.txt
ruff>=0.15.17
mypy>=2.1.0
pytest>=9.0.3
pytest-asyncio>=1.4.0
pre-commit>=4.6.0
types-PyYAML
```

Install commands (replace the single-line README command; all into one `.venv` by default):

```bash
uv venv --python 3.14.2
uv pip install -r requirements/dev.txt -r requirements/sim.txt -r requirements/analysis.txt
bash scripts/setup_sim_env.sh          # pybullet (CFLAGS) + gpd @ pinned SHA (--no-deps) + smoke test
uv pip install -r requirements/ml.txt  # only when training RL
```

`scripts/setup_sim_env.sh` (sketch): `CFLAGS="-Dfdopen=fdopen" uv pip install pybullet==3.2.7 --no-cache`; `uv pip install --no-deps "gym-pybullet-drones @ git+https://github.com/learnsyslab/gym-pybullet-drones@7ebad1ecabd28a7000add2d05f888aa2e837c2cc"`; smoke: `python -c "import pybullet as p; p.connect(p.DIRECT); from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary"`. gpd has a single GitHub release (v1.0.0, 2021) so **pin the SHA** [C 03]. Needs Xcode CLT (clang).

### 6.2 What to delete (recommend only)

| Item | Why safe |
|---|---|
| `asyncpg`, `psycopg2-binary` | Postgres drivers; SQLite only; zero imports [V]. |
| `boto3` | No S3; zero imports [V]; largest dependency tree in the file. |
| `requests` | Zero imports [V]; `httpx` is the client. |
| `python-dotenv` | `pydantic-settings` reads `.env` itself [C 10]. |
| `python-multipart` | No form/file endpoints; add back if one is ever added. |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore`, README "Docker" section | Local-only requirement; slim image cannot compile pybullet; one explicit commit, plus update the guide's Stack/Commands lines. |

Keep: `ruff`, `mypy`, `pytest`, `pytest-asyncio`, `pre-commit` (dev). Move `httpx` from "dev" to core. Not added on purpose: Ray, cvxpy, casadi, acados, Foxglove/mcap, Plotly/kaleido, `uvloop`, `sse-starlette` (native SSE exists), any DuckDB SQLite extension (needs a network install; export to Parquet instead [C 10]).

### 6.3 Separate Python env: decision and contingency

Not needed. If a future macOS/CLT update breaks the pybullet build on 3.14: create `.venv-sim` with Python 3.12 (`uv venv --python 3.12 .venv-sim`, same `requirements/sim.txt`), run **only** `scripts/run_experiment.py` workers in it; the API and analysis stay on 3.14.2. This works without code changes because (a) `harness/store.py` and `domain/` import only stdlib/numpy/pydantic, (b) workers never import `fastapi`/`sqlalchemy`, (c) the boundary is the SQLite file. Pybullet has no cp312+ wheel either way, so 3.12 does not remove the compile step [C 02].

### 6.4 pre-commit and mypy

`.pre-commit-config.yaml` mypy hook `additional_dependencies` should gain `sqlalchemy`, `aiosqlite`, `numpy`, `httpx`, `types-PyYAML` (only `pydantic`, `pydantic-settings` today). Otherwise `ignore_missing_imports = true` turns every SQLAlchemy/NumPy type into `Any` and the "full type annotations" rule checks nothing [C 10]. pybullet/gpd have no stubs: keep them `Any` behind the three adapter files.

### 6.5 Lockfile

The template has no lock; `>=` floors will float (SQLAlchemy would cross 2.1 and drop greenlet auto-install [C 10]). Recommend `uv pip compile requirements/dev.txt requirements/sim.txt requirements/analysis.txt -o requirements.lock` and record its hash in every run manifest; the Phase-0 versions above are the floors.

### 6.6 Recommended `ruff.toml` (not applied)

```toml
target-version = "py314"
[lint]
select = ["E4", "E7", "E9", "F", "I", "UP", "B", "T20", "TID"]   # I = isort (guide), T20 = no print
[lint.per-file-ignores]
"scripts/**" = ["T20"]            # scripts may print progress; guide allows the structured logger instead
```

`TID251` banned-api cannot express "only these files may import pybullet"; that stays in `check_imports.py`.

---

## 7. Configuration

Two layers with one rule: **`Settings` (env/.env) = machine, secrets, paths; YAML `ExperimentConfig` = science.** Secrets never appear in YAML or manifests.

### 7.1 `Settings` additions (existing fields kept; all typed, single `settings` instance)

```python
from pathlib import Path
from pydantic import SecretStr


class Settings(BaseSettings):
    ...  # existing app_name, env, debug, host, port, cors_origins
    database_url: str = "sqlite+aiosqlite:///data/jev.db"
    data_dir: Path = Path("data")  # flights/, scenarios/, cache/, reports/
    sqlite_busy_timeout_ms: int = 30_000
    experiments_dir: Path = Path("configs/experiments")
    prompts_dir: Path = Path("configs/prompts")
    experiment_config: Path | None = None  # EXPERIMENT_CONFIG for argument-free scripts
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai"
    typesafe_api_key: SecretStr | None = None  # optional direct Jev
    typesafe_base_url: str = "https://api.typesafe.ai"
    ollama_base_url: str = "http://localhost:11434"
```

`database_url` is cwd-relative; `data_dir` should be resolved once (`Path.resolve()`) in the composition root and the DB path derived from it so workers spawned from any cwd agree. Preflight (11.3) fails if `DATABASE_URL` and `DATA_DIR` disagree.

### 7.2 `.env` keys (`.env.sample` to gain these; `.env` stays gitignored)

```
PORT=8000
ENV=development
DEBUG=True
DATABASE_URL=sqlite+aiosqlite:///data/jev.db
DATA_DIR=data                       # for real runs: $HOME/drone-jev-data (outside synced ~/Documents)
OPENROUTER_API_KEY=                 # OpenRouter (Jev via /api/alpha/decisions; chat LLM arms)
TYPESAFE_API_KEY=                   # optional: direct api.typesafe.ai (F2 latency comparison)
OPENROUTER_BASE_URL=https://openrouter.ai
TYPESAFE_BASE_URL=https://api.typesafe.ai
OLLAMA_BASE_URL=http://localhost:11434
EXPERIMENT_CONFIG=configs/experiments/smoke.yaml
```

No key exists in this environment and none was used by any agent [C 00, 01]. Keys are `SecretStr`; providers log only provider id, snapshot id and status, never headers or bodies; `model_cache.prompt_text` is optional (hash-only by default) to bound size and avoid storing state text needlessly [C 10 open question 4].

### 7.3 Experiment YAML (parsed into `schemas/experiment.py`; hashed into the manifest)

```yaml
name: headline_v1
description: Latency-vs-collision, paired scenarios, S0 and S1 shield levels
sim:
  backend: pybullet              # kinematic only after the M6 rank-preservation gate
  drone_model: CF2X
  profile: {pyb_hz: 240, ctrl_hz: 48}          # `fine`: {pyb_hz: 240, ctrl_hz: 240}
  v_cap_mps: 2.0
  calibration_id: cal_2026-10-xx_cf2x_dslpid_v1
scenarios:
  suite: headline_v1
  master_seed: 20260930
  n_per_tier: 400                # stratified equal-N per tier (06); ~2000 scenarios for cheap arms
  tiers: [T0, T1, T2, T3, T4, T5, T-1]
  split: {train: [0, 9999], tune: [100000, 100499], test: [200000, null]}
time_model:
  mode: sampled                  # lockstep | fixed | sampled | measured | realtime
  cadence: {kind: sequential}    # or {kind: periodic, period_ticks: 5}
  latency_sweep_ms: [0, 10, 50, 100, 300, 1000, 3000]
  timeout_ms: 3000
  fallback: KEEP                 # KEEP | HOVER | BRAKE, identical across arms
  ttl_mode: hold_last            # hold_last | ttl_to_hover
shield: [S0, S1]                 # experiment factor, reported separately
arms:
  - {id: pid_a,   kind: setpoint_policy, policy: waypoint}
  - {id: reactive_b1, kind: setpoint_policy, policy: reactive_b1}
  - {id: fsm_e,   kind: provider, provider: fsm}
  - {id: oracle_o, kind: provider, provider: oracle}
  - id: jev
    kind: provider
    provider: jev
    model: typesafe/jev-1.13     # pinned; never ~typesafe/jev-latest
    encoder: jev_state_v1
    params: {aux_questions: [obstacle_imminent]}
  - id: llm_fast
    kind: provider
    provider: chat
    model: openai/gpt-oss-20b
    route: {order: [Groq], allow_fallbacks: false}
    encoder: F
    params: {max_tokens: 4, temperature: 0}
execution:
  pools: {cpu: 9, api: 6, local: 0}          # mixed campaigns: cpu 6 / api 2 / local 1 (09)
  api_subsample: {paired_n: 1000}            # API arms run a paired subset (08/09)
  budget_usd: 150
  real_time_validation: {n: 300, latencies_ms: [100, 300, 1000]}
```

Prompt/glossary files are referenced by id + content hash; the resolved config, `requirements.lock` hash, git SHA (+dirty flag), calibration id, latency-trace hash and hardware tag go into `experiments.manifest_json` [C 08 section 12].

---

## 8. Data layer deltas relative to agent 10's DDL

Adopt 10's DDL, with these changes (all arise from the reconciliations above):

1. `flights`: add `pool TEXT`, `priority INTEGER`, `mode TEXT`; rename `attempt` -> `attempts`; replace `heartbeat_at` by `lease_until`; add `latency_mode`, `ctrl_hz` (exists), `shield TEXT`. Partial indexes: `ix_flights_pending ON flights(pool, priority, id) WHERE status='queued'`, `ix_flights_lease ON flights(pool, lease_until) WHERE status='running'`. Claim = pending first, then expired lease, two statements (the `OR` form was 36x slower [C 09]). Natural key `UNIQUE(scenario_id, arm_id, latency_cell, rep, mode, shield)`.
2. `arms`: add `config_sha TEXT NOT NULL`; rows immutable once flights exist (a changed prompt/model creates a new arm row, never silently reuses results).
3. `scenario_events`: `t_start_s REAL`, `t_end_s REAL` instead of ticks; drop the `CHECK (kind IN (...))` (validated by the Pydantic discriminated union instead; the two lists had already diverged).
4. `decisions`: add `response_tick`, `latency_ticks`, `latency_mode`, `outcome`, `source` (small ints); action codes come from the enum.
5. New `decision_labels` (offline, versioned) as in reconciliation #21.
6. `actions` seeded from `Action` in `scripts/init_db.py`, never a hand-written INSERT.
7. Storage layout: `data/flights/<experiment>/<flight_id>.parquet` (12's tick schema), `flights.trajectory_uri` relative. DuckDB reads Parquet globs; the `decisions` table is exported to Parquet at the end of a run for analysis independent of SQLite locks [C 10].

---

## 9. Mapping agents' decisions onto modules (traceability)

| Agent decision | Lands in |
|---|---|
| 01: pin `typesafe/jev-1.13`, log snapshot id, HTTP/2 keep-alive, warm-up discard, `perf_counter_ns` | `providers/jev.py`, `harness/latency.py`, `scripts/probe_latency.py` |
| 02: `CtrlAviary` subclass, `obstacles=False`, `ctrl.reset()` in `_housekeeping`, wind in `_physics`, collisions via `getContactPoints`, never call `sync()` | `sim/scenario_aviary.py`, `controllers/dsl_pid.py` |
| 03: `SimBackend`, calibration + rank-preservation gate | `domain/protocols.py`, `sim/calibration.py`, `sim/kinematic_backend.py` (M6), `scripts/validate/check_tiers.py` |
| 04: reflex on/off ablation, cadence as a factor | `controllers/shield.py` (S0/S1), `decision_loop.py` cadence |
| 05: arms A-G, Z0-Z3, O, tuning-parity splits | `controllers/*`, `providers/{fsm,baselines,rl_policy}.py`, `oracle/*`, `scenarios/suites.py` |
| 06: budget-first generator, certifier, Philox channels, scenario JSON | `scenarios/*`, `sim/rng.py`, `schemas/scenario.py`, `sim/events/*` |
| 07: Observation, rays, format F, parser, fallback policy | `sim/sensors.py`, `providers/{encoding,parsing}.py`, `harness/decision_loop.py` |
| 08: time model, monotonic apply, metrics, manifest, stats | `harness/{latency,decision_loop,metrics,manifest}.py`, `analysis/stats.py` |
| 09: CLI + SQLite queue, pool sizes, cache caveat, budget ledger, preflight | `harness/{store,worker,supervisor,budget,cache}.py`, `scripts/` |
| 10: STRICT DDL, WAL pragmas, sync workers, Alembic batch | `app/models/*`, `app/db/*`, `harness/store.py`, `migrations/` |
| 11: executor, shield lattice, two-channel latch, oracle labeller | `controllers/{executor,shield}.py`, `oracle/*` |
| 12: no rendering in batch, replay contract, SSE, HTML player, Rerun | `harness/recorder.py`, `replay/*`, `static/demo/`, `api/v1/experiments.py` |

---

## 10. Testing strategy (defined now, written only when the user asks, per the guide)

Principle: **tests protect logic that can silently change the science.** Everything else is validated by scripts (section 11).

| Layer | What | How | Notes |
|---|---|---|---|
| Unit (pure, fast, no pybullet) | `ActionExecutor` state machine (dwell, side-commit, TTL, displacement bounds, landing latch, stale-reply drop); `LatencyQueue` monotonic-apply and out-of-order drop; `SimClock.ticks_for_ms` (ceil, >= 1 tick); `parsing.py` fuzz (zero false-exact parses); `utils/braking.py` vs the 06 table (`d_min0` for v = 0.5..12); `utils/intervals.py` Wilson coverage; `utils/hashing.py` canonical-JSON stability; generator budget formula `B` vs the surrogate to the ms | pytest, no fixtures beyond `tmp_path` | These are the 06/08/11 spikes turned into regressions. |
| Contract | every `DecisionProvider` implementation satisfies the Protocol and the error taxonomy; mock/HTTP providers tested with `httpx.MockTransport` (canned Jev, OpenRouter chat, 429/529/524, malformed bodies) | `@pytest.mark.parametrize` over a provider factory | **No real network in any test.** Real latency only via `scripts/probe_latency.py`. |
| Store/DB | `SqliteJobQueue`: claim exclusivity (N threads/processes, zero duplicates), lease expiry requeue, idempotent `complete`, crash mid-flight leaves no partial rows | temp-file DB per test (`tmp_path`; `:memory:` has no WAL), fixture `yield` teardown | Mirrors 09's `crash.py` and 10's spike at small scale. |
| Integration (real pybullet, marked `sim`, excluded from the default quick run) | `ScenarioAviary` determinism (same scenario + action log -> identical state hash in-process), `ctrl.reset()` regression, hover-drift acceptance (< 0.05 m in 10 s at 0.02 N), obstacle spawn visible to rays only after spawn, collision flag via contacts | one env per test module, `close()` in teardown | Needs the build from `setup_sim_env.sh`. |
| API | `/health`, `/experiments` create/list/progress/summary, pagination bounds, error shape | existing `ASGITransport` client + `app.dependency_overrides` for `get_session` with a temp-file DB | Never patch globals (guide). |

Conventions: add markers `sim` and `slow` to `pytest.ini`; default `uv run pytest -v` should be runnable without sim markers via `-m "not sim"` if the user does not want the pybullet build on the test path. **Leave no stale data**: every test that creates rows/files/env handles does so in a `yield` fixture or `tmp_path`; after a run, `data/` and the real DB are untouched (tests never read `settings.data_dir`; fixtures inject paths). Coverage target from the guide (>= 80 % on services/repos) applies to `app/services`, `app/repositories`, `harness/store.py`; the science modules are covered by the validation scripts instead.

---

## 11. CI-less validation checklist

There is no CI. Everything below runs locally, in this order, and gates a milestone.

### 11.1 After every change (guide steps, verbatim)

```bash
uv run python -m py_compile path/to/file.py
uv run ruff check .
uv run mypy --explicit-package-bases .
```
Plus `uv run python scripts/validate/check_imports.py` whenever a file is added or an import changes (the layer rules in 3.2). `pre-commit run --all-files` before each commit.

### 11.2 Milestone gates (scripts under `scripts/validate/` unless noted)

| Gate | Check | Pass criterion | Source |
|---|---|---|---|
| G0 | `setup_sim_env.sh` smoke; `check_env.py`; throughput on a quiet machine (`bench_sim_throughput.py`, `sysctl vm.loadavg < 2`) | pybullet imports, `CtrlAviary` hover 1000 ticks, >= 25k ctrl steps/s env-only and >= 7k with PID per core (02's target) | 02, 12, 09 B1 |
| G1 | minimal-deps env: import `CtrlAviary`/`DSLPIDControl` with only `sim.txt` (no torch/SB3) | imports succeed; record what extra packages gpd truly needs | 02 spike 6 |
| G2 | `check_determinism.py`: same scenario + same action log, two spawned processes and two interpreter launches | identical state hash (or documented tolerance) | 02, 08, 12 |
| G3 | `calibrate_sim.py` -> `a_brake`, `a_lat`, `tau_a`, stopping distances at 1/2/3 m/s; hover drift acceptance | fit residual < 10 % of `s_stop`; latched hover drift < 0.05 m/10 s at 0.02 N | 06 spike 1, 11b |
| G4 | `check_tiers.py` / generator: per-scenario feasibility certificate; Oracle-B at `L = B` sits at the boundary | |B_true - B_model| <= 10 % B; Oracle-0 >= 99 % collision-free on feasible scenarios | 05, 06 |
| G5 | `check_paired_invariants.py`: adding or removing an arm does not change any other arm's world (obstacle trajectories, wind, noise arrays identical by hash) | bit-identical | 06 spike 5, 08 |
| G6 | B8 dry run (09): 200 scenarios x 4 arms with a mock provider, `kill -9` at random times | final `flights` count exact, metrics byte-identical to an uninterrupted run, zero stuck `running` | 09 B8 |
| G7 | Latency-mechanism validation: `REALTIME` vs `SAMPLED/FIXED` on >= 300 flights x 3 latencies | per-flight agreement >= 90 %, collision rate within CI, tick lateness p99 < 0.25 dt | 08 |
| G8 (only for M6) | rank-preservation T0 vs T1 | Spearman >= 0.9, knee within one grid step, kappa >= 0.6 | 03 |

### 11.3 Preflight before any headline run (`harness/supervisor.py` refuses to start otherwise)

Plugged in and not thermally throttled (`pmset -g therm`); `sysctl vm.loadavg` < 2 for latency-probe and real-time runs; `DATA_DIR` not under `~/Documents`/`Mobile Documents`; free disk >= 20 GB; `git` clean or dirty flag recorded; experiment config hash unchanged since enqueue; `OPENROUTER_API_KEY` present for API arms and `budget_usd` set; provider snapshot string unchanged mid-run (abort on change, 01); canary prompt every 5 minutes (08); run under `caffeinate -dimsu -w <supervisor pid>` and Spotlight-excluded data dir (09).

### 11.4 Reporting hygiene

Every figure carries n, test, correction and exclusions in its caption sidecar; the report refuses headline comparisons without a paired design or below the power-analysis n without an "underpowered" banner; latency mode is printed on every demo frame (12); `LOCKSTEP` results only ever appear labelled as the "latency-free quality" ablation (08).

---

## 12. Module dependency graph and build order

### 12.1 Dependency graph (arrow = "imports / needs")

```
utils ---------------------------------------------------------------+
schemas  (action, scenario, experiment, decision) -> utils            |
domain   (observation, setpoint, records, protocols, clocks, errors) -> schemas, utils
   ^
   +-- sim/rng, sim/surrogate (pybullet-free) ------------------------+
   +-- sim/{scenario_aviary,pybullet_backend,sensors,events/*} -> pybullet, gpd, surrogate? (no), rng, domain
   +-- controllers/{dsl_pid (pybullet), executor, shield, waypoint, reactive, mpc_osqp} -> domain, utils
   +-- providers/{encoding,parsing,prompts,mock,jev,chat,system_one_local,replay,fsm,baselines,rl_policy} -> domain, schemas, httpx
   +-- oracle/{labeller,provider}  -> domain, sim/surrogate, utils
   +-- scenarios/{generator,certifier,suites} -> schemas, sim/surrogate, utils  (certifier also takes a SimBackend via Protocol)
   v
harness/{latency,decision_loop,metrics,recorder,flight_runner,store,worker,supervisor,factory,manifest,budget,cache}
   -> domain (Protocols), schemas, utils;  factory -> all L3 concrete classes
   v
scripts/*  (composition roots)              api/v1 -> services -> repositories -> models/db -> schemas, utils
analysis/*, replay/* -> schemas, utils, pandas/duckdb/matplotlib  (API may import analysis and scenarios, never sim/providers/harness)
```

### 12.2 Build order and milestones

Each milestone ends with a runnable artefact, a gate from 11.2, and is sized for the single developer-plus-assistant loop. Effort is order-of-magnitude [I].

| M | Scope (files) | Runnable outcome | Exit gate | Effort |
|---|---|---|---|---|
| **M0 Foundations** | `requirements/*`, delete list applied, `setup_sim_env.sh`, `.gitignore`, `.env.sample`, `Settings`, `schemas/{action,decision}`, `domain/*`, `errors`, `check_imports.py`, `check_env.py` | `uv run pytest -k health` still green; Protocol sketch checks clean; pybullet imports | G0, G1; 11.1 clean | 1-2 d |
| **M1 Vertical slice** | `rng`, `scenario_aviary`, `dsl_pid`, `pybullet_backend`, `sensors`, `events/obstacle`, `executor`, `schemas/{scenario,experiment}`, `utils/{hashing,braking}`, `models/*`, `db/*`, Alembic 0001, `store`, `recorder`, `metrics`, `latency` (FIXED), `decision_loop`, `flight_runner`, `worker`, `factory`, `mock` provider, `enqueue_experiment`, `init_db`, `bench_sim_throughput` | One scenario, one mock-provider arm, fixed latency: flight runs end to end, writes a DB row and a Parquet file; `kill -9` leaves no partial rows | G2 (determinism) | 1-2 wk |
| **M2 Scenario science** | `surrogate`, `calibration`, `generator`, `certifier`, `suites`, `oracle/labeller`, remaining `calibrate_sim`/`generate_scenarios` scripts | Certified headline suite in the DB; calibration id fixed; labels generated offline | G3, G4, G5 | 1-2 wk |
| **M3 Classical arms + sweep** | `shield` (S0/S1), `waypoint`, `reactive`, `fsm`, `baselines`, `oracle/provider`, remaining events (wind, battery, GPS, noise, target, intruder), `supervisor`, `run_experiment` | First **collision-vs-injected-latency curve** for arms A/B1/E/O/Z on PyBullet, shield S0 and S1; wall-time of the full matrix measured (decides M6 tier-0) | G6 | 1-2 wk |
| **M4 Providers** | `encoding`, `parsing`, `prompts`, `jev`, `chat`, `system_one_local`, `replay`, `cache`, `budget`, `probe_latency`, glossary + prompt files | Real latency distributions (needs a key); `MEASURED` (pause-and-charge) and `SAMPLED` runs for Jev and one fast chat arm on a paired subset; prompt-format stage-1 ablation on a local stand-in | G7 | 1-2 wk |
| **M5 Analysis + API + demo** | `analysis/*`, `replay/*`, `static/demo`, `repositories`, `services`, `api/v1/*`, `utils/{intervals,pagination}`, `main.py` lifespan, `make_report`, `export_replay` | HTML report with P1-P14, SSE progress page, replay player for one scenario, all arms | G-report (11.4) | 2 wk |
| **M6 Stretch** | `mpc_osqp` (arm C), `rl_policy` + `train_rl` + `export_policy` (arm D), `kinematic_backend` (**only if** M3 wall time > ~1 working day), real-time validation set, format-ablation stage 2, optional Rerun | Full arm set; tier-0 labelled "screening only" until G8 passes | G8 | 2+ wk |

Critical path: M0 -> M1 -> M2 -> M3 -> (M4 in parallel with M3 after M1, since `MockProvider` already exercises the loop) -> M5. M6 items are independent of each other. The earliest meaningful scientific output is the end of M3 (classical arms) using mock/oracle providers with injected latency; real Jev arrives at M4.

---

## Decisions recommended

1. **Two-plane architecture**: runner CLI (`spawn` pools, argument-free scripts) executes; FastAPI enqueues and monitors; one global WAL SQLite file; workers use stdlib `sqlite3`, API uses async SQLAlchemy + aiosqlite.
2. **Stay on Python 3.14.2 in a single venv**; pybullet via `CFLAGS="-Dfdopen=fdopen"` in `scripts/setup_sim_env.sh` (not in requirements); gpd pinned to SHA `7ebad1e...` with `--no-deps`; 3.12 `.venv-sim` is a documented contingency only.
3. **PyBullet-only for v1; Tier-0 kinematic is a surrogate first and a backend only at M6**, gated by measured M3 wall time and the rank-preservation test.
4. **Subclass `CtrlAviary`, own the `DslPidController`**; never `VelocityAviary`; BRAKE as velocity ramp then latch, HOVER as true position latch.
5. **One `DecisionProvider` contract (action + probabilities + confidence + aux + wall timing + raw output)** for every action arm including FSM/RL/oracle; `SetpointPolicy` for arms A/A'/B/C; `ObservationEncoder` per provider; action glossary single-sourced in `configs/prompts/actions.yaml`.
6. **The harness owns latency and staleness** (`LatencyPolicy`: lockstep, fixed, sampled, measured, realtime; `ceil` ticks; monotonic apply); the provider decorator from 01 is dropped; one numpy `Generator` family with sha256-keyed Philox channels.
7. **K=1 flight per API worker process** with pause-and-charge for real-API arms; bulk sweeps use no network; a separate probe process publishes latency distributions.
8. **Shield level is an experiment factor (S0/S1, reported separately)**; S2 is arm B, not a level.
9. **Schema**: 10's DDL with `flights` as the queue (+ `pool, priority, mode, attempts, lease_until`), `arms.config_sha`, second-based scenario events, a separate offline `decision_labels` table, `actions` seeded from the enum, canonical action codes = enum order (11's numbering).
10. **Dependencies**: requirements split into `core/sim/ml/analysis/dev`; delete `asyncpg, psycopg2-binary, boto3, requests, python-dotenv, python-multipart` and the Docker files; add `aiosqlite, alembic, httpx[http2], PyYAML`; pin `SQLAlchemy[asyncio]>=2.1.1,<2.2`; torch/SB3 confined to `ml`; RL policy exported to numpy for eval; lock with `uv pip compile`.
11. **Layering**: L0 utils < L1 schemas < L2 domain < L3 implementations < L4 harness < L5 scripts/API, enforced by an `ast` import-boundary script; hot-path types are frozen dataclasses, config/DTOs are Pydantic; workers never import FastAPI/SQLAlchemy/torch; only three files import pybullet/gpd.
12. **Keep the guide's rules where they apply** (no `print`, typed, class-based DI, `%s` logging, scripts with constants and no CLI args, tests only when asked) and record the 11 deviations in 3.4 as explicit, reviewed exceptions rather than silent ones.
13. **Testing**: a logic-protecting test set defined but unwritten until requested; scientific validity enforced by `scripts/validate/` gates G0-G8 and a headline-run preflight.

## Open questions

1. **Confirm Jev identity** (TypeSafe Jev 1.13) and whether the user wants a direct TypeSafe key in addition to OpenRouter (changes `TYPESAFE_API_KEY` from optional to used) [01].
2. **`PyYAML` on 3.14.2** (and any other package not measured by another agent): unverified here; if it has no wheel, switch experiment configs to JSON/TOML (stdlib `tomllib`) rather than adding a build dependency.
3. **What does gym-pybullet-drones actually import beyond numpy/scipy/gymnasium/pybullet/transforms3d?** Decides whether `matplotlib`, `pillow`, `control` belong in `sim.txt` (02 spike 6, unrun).
4. **Does `uv --no-cache` really force a fresh pybullet compile, and does a cached wheel built with the flag get reused correctly?** 11b observed a 0.1 s install; the wheel-caching strategy (`wheelhouse/`) is untested.
5. **SQLAlchemy 2.0.50 vs 2.1.1 parity** (10's F5) and Alembic autogenerate diff against the DDL; chosen 2.1.x because it is what was exercised.
6. **Is `ctrl_hz=48` acceptable as the headline tick?** A 10 ms sweep point is indistinguishable from 20 ms there; the 240 Hz fine profile is 5x dearer (PID calls). User/08 call.
7. **Import-lint placement**: is a bespoke `check_imports.py` acceptable, or would the user prefer adding Ruff `TID`/`flake8-tidy-imports` with `ruff.toml` (which cannot express per-file allow-lists)?
8. **Where does `experiment_config` come from for spawned workers**: `Settings` env var (chosen) versus an argument passed by the supervisor; matters only if the user relaxes the "no CLI args" script rule.
9. **Observation `age_ms` / `expected_latency_ms` semantics for the Jev prompt**: is a pre-compensated state an acceptable baseline for classical arms too (07 open question 7, needs 08's ruling)?
10. **Retention of prompt text** (`model_cache.prompt_text`): hash-only by default; is prompt/response audit needed (0.2-1.7 GB at 10k flights [C 10])?
11. **Sensing geometry** (omnidirectional 16-ray ring, yaw fixed) is assumed throughout; if a forward-facing FOV is chosen, `YAW_LEFT/RIGHT` actions and the path-frame semantics of FORWARD change (11 E2).

## Follow-up spikes (small, runnable experiments)

1. **Phase-0 env script (30 min)**: write and run `scripts/setup_sim_env.sh` in a fresh `uv venv --python 3.14.2`; record build time with `--no-cache`, then again after deleting the env and reinstalling (cache reuse); assert `import pybullet`, `CtrlAviary` hover 1000 ticks.
2. **Minimal-deps import test (15 min)**: venv with only `requirements/sim.txt` (+pybullet, gpd `--no-deps`); import `CtrlAviary`, `DSLPIDControl`; list missing modules; check worker RSS with and without `pyarrow` (memory budget ~150 MB/worker, 09).
3. **`PyYAML`/`h2` wheel check (5 min)**: `uv pip install --only-binary :all: PyYAML "httpx[http2]"` on 3.14.2.
4. **Protocol conformance skeleton (1-2 h)**: create `app/domain/*`, `app/schemas/{action,decision}.py` exactly as in section 5, plus a `MockProvider` and a `FakeBackend`; run the 11.1 checks; acceptance: `mypy --explicit-package-bases .` clean with the real `mypy.ini`.
5. **Import-boundary script (1 h)**: `scripts/validate/check_imports.py` with the six rules of 3.2; acceptance: it fails on a deliberately added `import fastapi` in `harness/` and on `import pybullet` in `providers/`.
6. **Vertical-slice time budget (M1 smoke)**: measure ms per tick for `FlightRunner` with mock provider, rays at 20 Hz, Parquet recording on; compare with 11b's 0.17 ms/tick bare loop to size the real per-flight overhead (feeds the M6 tier-0 decision and 09's flight/hour table).
7. **Sweep wall-time projection (after M3)**: run 500 scenarios x 7 latencies x 4 classical arms on 9 workers; extrapolate to 560k flights; decision rule: < ~1 working day -> no `KinematicBackend`.
8. **Single-writer contention check with the real store**: 9 workers committing one transaction per flight plus the API polling progress every 500 ms (10's F1); accept p99 commit < 100 ms and bounded WAL.
9. **Rename/renumber dry run**: generate the `actions` seed from the enum and diff against 10's hand-typed INSERT and 11's table to prove the codes and the `Action` order match before any data is written.
10. **Guide-gap patch list**: prepare (do not apply) a single diff for `.github/copilot-instructions.md` covering the 11 deviations in 3.4 (Stack without Docker, `app/api/v1` naming, `APIResponse` casing, pagination util status, worker DB rules) so the guide stays the single source of truth.
