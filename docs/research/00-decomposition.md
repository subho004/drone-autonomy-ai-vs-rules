# 00 — Task Decomposition (shared brief for all research agents)

Produced by the decomposition agent. Every research agent reads this file first, then does its own subtask.

## Project context

- Idea source: `docs/initial-plan.txt` (chat-derived). Use gym-pybullet-drones on an M4 MacBook Pro (16 GB, macOS), no Docker.
- "Jev" = a fast, low-latency decision model/API the user will reach via OpenRouter or directly from Jev. **What Jev is exactly is unverified and must be researched.** Nothing in the repo describes it.
- Jev picks discrete actions: FORWARD, LEFT, RIGHT, ASCEND, DESCEND, BRAKE, HOVER, AVOID_LEFT, AVOID_RIGHT, RETURN_HOME, EMERGENCY_LAND. A conventional PID/MPC controller does low-level stabilisation; Jev does NOT replace motor control.
- Adversarial events: obstacle suddenly appears, wind changes, target moves, sensor noise, battery drops, GPS disappears, another drone enters trajectory.
- Compare Jev vs PID/MPC vs conventional policy vs slower LLM on: decision_latency, collision_rate, time_to_target, path_length, recovery_time, decisions/sec, simulation_speed, over thousands of flights.
- Local-only: SQLite, no Docker. The repo is a FastAPI template (Python 3.14.2, uv, async SQLAlchemy, asyncpg/psycopg2/boto3 currently in requirements.txt, Dockerfile + docker-compose.yml present). Conventions live in `.github/copilot-instructions.md` (layers: routes/services/repositories/schemas/models/utils; class-based services with DI; `success_response`/`error_response`; ruff + mypy; no print; etc.).

## Rules for every research agent

1. Use real web research (WebSearch / WebFetch). Cite sources with URLs and access dates. Separate **verified facts** from **inference**, and give confidence levels.
2. Note anything that contradicts `docs/initial-plan.txt` (especially the "~20 kHz / 80 drones / 4 envs" throughput claim and the "Apple Silicon / macOS 26 tested" claim).
3. Do not modify any file in the repo other than your own output file under `docs/research/`. Temp work (venvs, scripts, downloads) goes in the session scratchpad dir, never `/tmp` and never the repo. Do not use any API keys; do not send user data anywhere.
4. Small runnable spikes (installing packages into a scratch venv, micro-benchmarks) are encouraged where your subtask says so — report measured numbers with hardware/Python/package versions, and the honest failure if something does not install.
5. No compromises on depth: be specific, quantitative, and opinionated. Prefer numbers, code-shaped interface sketches, tables, and explicit tradeoffs over prose.
6. End your file with exactly these sections: **Decisions recommended**, **Open questions**, **Follow-up spikes (small, runnable experiments)**.
7. Your final chat reply must be short: the output path, a 5-line summary of key findings, and any blocker.

## Global open questions (owners in brackets)

1. **Simulated time vs wall-clock (CRITICAL; owner 08, with 01, 02, 09).** If an API decision takes 300 ms+ and the sim runs faster than real time, options are: (a) lockstep/pause-the-world (latency invisible → meaningless), (b) real-time pacing with the drone continuing on its last action while the request is in flight (faithful but slow), (c) latency injection: sim at max speed, measured/modelled/sampled wall-clock latency converted to N control ticks of delay, (d) hybrid: record real latency distributions, replay them as sampled delay in fast runs, plus a smaller real-time validation set. Must recommend a primary + validation method, the state-staleness model, the async decision loop, what the PID does meanwhile, reproducibility under non-deterministic latency, and how to report decisions/sec and sim_speed without API-time confounding.
2. **What is "Jev"?** Model on OpenRouter? Vendor API? Distilled small model? Hardware-backed inference service? Typo? Verify with primary sources and state confidence. If unresolved, design a model-agnostic `DecisionProvider` with a stand-in (local small model on MLX/llama.cpp/Ollama, or a fast OpenRouter model).
3. Raw API latency vs end-to-end closed-loop reaction (sense→act): the answer drives metric definitions.
4. gym-pybullet-drones on Python 3.14.2: wheels for pybullet, numpy, gymnasium, stable-baselines3, torch? Fallback: separate Python 3.10–3.12 venv for the sim worker behind a process boundary.
5. Is "~20 kHz with 80 drones across 4 envs" accurate and hardware-specific; what does it mean for 1 drone on an M4 (DIRECT mode)?
6. Does the "Apple Silicon / macOS 26 tested" claim hold; is GUI (OpenGL) needed? Headless DIRECT mode needed for thousands of flights.
7. Which adversarial events are native (wind/drag/ground effect/downwash, obstacles via URDF) vs custom (GPS loss, battery, sensor noise, moving target, second drone)?
8. Determinism/seeding: PyBullet determinism across processes; seeded event schedules so all controllers face identical scenarios (paired design).
9. Cost, rate limits, ToS for thousands of flights × decisions × paid LLM calls; budget math; sub-sampling for API arms.
10. Do SQLite + no-Docker preferences conflict with the template (asyncpg, psycopg2, boto3, Dockerfile, compose)? Minimum clean modifications (aiosqlite, WAL, dropping Postgres deps).

## Subtasks (output files under `docs/research/`)

Dependencies: 08 and 09 consume 01/02/06/11; 13 and 14 run last and read the others' outputs. 01–12 run in parallel; 01 and 02 are on the critical path.

### 01 — Jev identity + OpenRouter/API latency realities → `01-jev-identity-openrouter-latency.md`
- What is Jev? Check OpenRouter's model list, company sites, GitHub, HuggingFace, arXiv, news. Model, chip, product, or typo? Confidence rating and alternatives.
- OpenRouter support and params (max_tokens, structured output/JSON schema, tool calls, logprobs, temperature=0, streaming, stop tokens, `provider` routing, `:nitro`/`:floor`).
- Real latency components: DNS/TLS/connection reuse, TTFT, tokens/sec, queueing, OpenRouter overhead (reported ~25–40 ms), geography; p50/p95/p99.
- Minimising latency: keep-alive HTTP/2, prompt size, prefix caching, single-token/enum outputs, max_tokens=1–3, logit bias, direct provider vs OpenRouter.
- Fastest providers for tiny outputs (Groq, Cerebras, SambaNova, Fireworks) vs local MLX/llama.cpp small models on M4 (Qwen 0.5–3B, Llama 3.2 1B/3B, Phi, Gemma).
- Rate limits, cost/decision, ToS, reproducibility (version drift, non-determinism).
- Design the `DecisionProvider` abstraction (Jev, slow big LLM, local models, configurable-latency mock).

### 02 — gym-pybullet-drones internals, API, Apple Silicon, performance → `02-gym-pybullet-drones-internals-performance.md`
- Repo state (latest release/commit, maintainer, license, macOS/ARM issues, Python version support, install via pip/uv, pybullet arm64 wheel/compile situation).
- Env classes (BaseAviary, CtrlAviary, VelocityAviary, HoverAviary, MultiHoverAviary, BaseRLAviary), ActionType, ObservationType, Physics modes, freq params (`pyb_freq`, `ctrl_freq`).
- Built-in controllers (DSLPIDControl, SimplePIDControl), VelocityAviary `SpeedFractionLimit`, obstacle mechanism, camera API and cost, `record`/`gui`/`obstacles` flags, `sync()` wall-clock pacing and how to disable it.
- Reality check of the "20 kHz / 80 drones / 4 envs" claim vs README; micro-benchmark steps/sec for 1 drone DIRECT on this M4 (propose AND run the script in a scratch venv).
- Reset cost, memory growth over thousands of episodes, multiprocess safety (PyBullet client IDs).
- Missing natively: wind, battery, GPS, actuator latency — how easy is `applyExternalForce`?
- Extension points: subclassing BaseAviary, overriding `_computeObs`, `_preprocessAction`, `_computeReward`, `_computeTerminated`.

### 03 — Alternative simulators comparison → `03-simulator-alternatives-comparison.md`
Compare gym-pybullet-drones vs Webots, AirSim/Colosseum, Isaac Sim/Lab, Gazebo+PX4/ArduPilot SITL, Flightmare, Aerial Gym, MuJoCo (+MJX), Genesis, RotorPy, CrazyS, Unity ML-Agents, custom minimal sims. Apple Silicon support, headless throughput, sensor fidelity, adversarial-event support, multi-drone, PX4/MPC integration, licensing, maintenance, install friction w/o Docker. Scored matrix, recommendation with fallback; assess a tiered design (fast kinematic sim for thousands of runs + PyBullet high-fidelity subset).

### 04 — Prior art: LLM/VLM-driven drone control, latency-hybrid architectures → `04-prior-art-llm-drone-control.md`
What has been built (TypeFly, PromptCraft/ChatGPT for Robotics, UAV-VLA, AerialVLN, OpenFly, CityNav, NavGPT, SayCan, Code-as-Policies, RT-2, OpenVLA, Helix/dual-system, DriveVLM-Dual, LLM-in-loop for driving…). How latency was handled (async planner + reactive layer, System1/System2, safety shields, action chunking). Discrete-action LLM policies and failure modes. What's missing in the literature this experiment could fill. What people built on gym-pybullet-drones specifically; any repo/blog doing this exact experiment.

### 05 — Baselines: PID/MPC, reactive avoidance, RL, "conventional policy" → `05-baselines-pid-mpc-avoidance-rl.md`
Define arms: (A) waypoint PID no avoidance; (B) PID + reactive avoidance (APF, VFH, ORCA, TTC braking, VO, CBF); (C) MPC (solve time on M4; cvxpy/OSQP, CasADi, acados); (D) RL (PPO/SAC via SB3 with discrete head, same obs/action space); (E) rule-based FSM over the same 11 actions (fairest non-ML baseline; tuning-effort parity); (F) slow LLM; (G) Jev. Latency treatment for classical arms (measure compute time, e.g. MPC solve). Install friction on Python 3.14/arm64. Oracle bound, random/always-hover sanity baselines.

### 06 — Adversarial events and disturbance modelling → `06-adversarial-events-disturbances.md`
For each event (sudden obstacle, wind/gusts, moving target, sensor noise, battery drop, GPS loss, second drone): physical/mathematical model (Dryden/von Kármán, OU gusts, battery sag/thrust derating, GPS dropout/drift, noise/bias/dropout, second-drone policies), PyBullet implementation, scheduling (Poisson/uniform windows, severity, curriculum), **feasibility guarantee** (avoidable given braking distance; reaction budget), ablation design, seeded serialisable JSON scenarios, ground-truth "correct action" labels.

### 07 — Sensors, perception, state representation for LLM prompts → `07-sensors-perception-state-representation.md`
Modalities (kinematic state, relative target vector, raycast "lidar" via `p.rayTestBatch` layouts, depth images cost on M4, noisy GPS, battery, nearest-obstacle range/closing speed, other-drone state). Encoding for text LLMs (compact JSON vs key:value vs sector array vs ASCII grid vs natural language), token minimisation, prompt-format ablation, frames/units/normalisation/history/decision-age field, multimodal as optional arm, a shared `Observation` dataclass with per-arm projections, prompt design (system prompt, few-shot, action glossary, constrained output, parse-failure handling, fallback action).

### 08 — Benchmark methodology, time accounting, latency fairness, statistics → `08-benchmark-methodology-time-accounting-stats.md` (CORE)
Resolve global question 1 in depth: formal clock model (sim/wall/decision time), async decision loop, staleness model, "effective latency ticks"; compare options (a)–(d); recommend primary design + validation; design latency sweep (0, 10, 50, 100, 300, 1000, 3000 ms) giving collision-rate-vs-latency curves separating speed from smartness. Precise metric definitions (decision_latency split network/TTFT/generation, reaction_time, time_to_target, path_length, collision, recovery_time, decisions/sec, sim_speed, success, near-miss, energy, oscillation, invalid-output rate, timeouts). Experimental design: paired scenarios, seeds, power analysis, CIs (Wilson/bootstrap), McNemar/Wilcoxon, multiple-comparison correction, heavy-tailed latency reporting, warm-up exclusion, network-variance control. Rigour: perf_counter_ns, GIL/event-loop jitter, asyncio vs thread vs process client, M4 P/E cores & thermals. Avoid "killer demo" claims that are time-model artifacts. Reproducibility manifest.

### 09 — Orchestration, parallelism, throughput on M4 16 GB → `09-orchestration-parallelism-throughput.md`
Process architecture w/o Docker (runner CLI vs FastAPI jobs; multiprocessing spawn; one PyBullet client per process; pool sizing on 4P+6E cores; 16 GB budget). Throughput model (flights/hour per arm), API-arm bottleneck (in-flight concurrency, httpx pooling, rate limits/backoff), budget & time for 1k–10k flights. SQLite-backed job queue, resumability, idempotent runs keyed (scenario_id, arm, seed). LLM response caching and its latency-measurement caveat. macOS specifics (caffeinate, thermals, taskpolicy/QoS, Python 3.14 free-threading/JIT relevance). Micro-benchmarks to run first and acceptance thresholds. Ray: for/against.

### 10 — SQLite schema, write throughput, FastAPI template integration → `10-sqlite-schema-fastapi-integration.md`
Tables (experiments, scenarios, events, arms, flights, decisions, trajectory samples — downsampled rows vs Parquet/NPZ blobs on disk, metrics, model-call cache), indexes, FKs, WAL + pragmas, batching, multi-writer strategy. Volume estimates and analysis tooling (DuckDB, pandas). Fit with the template: read `app/`, `main.py`, `utils/`, `tests/`, `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md` in full; async SQLAlchemy + aiosqlite vs sync sqlite3 in workers; Alembic; changes to `app/db`, settings; remove asyncpg/psycopg2/boto3; API endpoints; tests with in-memory DB. Python 3.14 compatibility of deps. Run a SQLite write-rate spike with 8 workers.

### 11 — Action-space to controller mapping → `11-action-to-controller-mapping.md`
Semantics of the 11 actions as setpoint transforms (velocity vs position targets; BRAKE, HOVER, AVOID_LEFT/RIGHT, RETURN_HOME, EMERGENCY_LAND); latched vs one-shot vs TTL; override vs modifier of the waypoint; safety shield/arbitration and timeout fallback; smoothing and hysteresis; control freq vs decision freq; per-arm mapping; shared enum + JSON schema; oracle labeller; gaps in the action set (speed, yaw, moving target); whether MPC should consume the action as cost/constraint.

### 12 — Visualisation, demo, analysis tooling → `12-visualization-demo-analysis.md`
The "killer demo": side-by-side synchronised replay with latency bars. Options: PyBullet GUI on macOS, offline getCameraImage→MP4, matplotlib 3D, Plotly/Three.js web replay served by FastAPI from stored trajectories, Rerun.io, Foxglove. Choose with rationale. Recording cost vs replay from stored trajectories with a decision-timeline overlay. Analysis report: latency CDFs, collision-vs-latency curves with CIs, Pareto plots, Gantt of decision loops, per-event breakdown; libraries; dashboard endpoint; live progress (SSE/WebSocket).

### 13 — Risks, bottlenecks, failure modes + phased roadmap → `13-risks-bottlenecks-failure-modes.md` (runs LAST; reads 01–12 and 14)
Ranked risk register (likelihood, impact, mitigation, early-warning test), top 5 bottlenecks with quantitative estimates, confounds (incl. latency "knee" where extra speed stops helping because of physical decel limits), phased roadmap (Phase 0 spikes … Phase 6 analysis/demo) with exit criteria.

### 14 — Repo architecture fit + project skeleton → `14-repo-architecture-fit.md` (runs LAST; reads 01–12)
Audit the template and propose locations (`app/sim/`, `app/controllers/`, `app/providers/`, `app/harness/`, models, API, `scripts/`, `tests/`), dependency strategy (keep/remove/add; extras; separate Python env if 3.14.2 can't support pybullet; Docker files), typing Protocols (`DecisionProvider`, `Controller`, `Scenario`, `EventInjector`, `MetricCollector`, `Clock`), config (pydantic-settings + YAML experiment configs, `.env` for keys), testing strategy, file-level skeleton, dependency graph, order of implementation — all while obeying `.github/copilot-instructions.md`.

## Mandatory micro-spikes (owners in brackets)
1. Install & run pybullet + gym-pybullet-drones in a Python 3.14.2 venv; measure steps/sec headless [02].
2. Real p50/p95 latency of fast vs slow models via OpenRouter for ~300-token prompt, 1–3 output tokens [01] — **no API key available, so use published benchmarks + document the exact script to run later**.
3. Latency-injection prototype showing collision-rate vs latency on one scenario [08] (may use a simple point-mass sim or gym-pybullet-drones if 02 has proven install).
4. SQLite write-rate test with 8 workers [10].
