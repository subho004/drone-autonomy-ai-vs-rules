# Drone + Jev: Master Implementation Plan

Synthesis of 14 research files in `docs/research/` (index in section 17). Date: 2026-09-30. Target machine: M4 MacBook Pro, 4P+6E cores, 16 GB, Python 3.14.2 via `uv`, no Docker, SQLite.

Evidence tags: **[measured]** = run on this M4 by a research agent; **[docs]** = read from primary docs or source; **[inferred]** = reasoning or estimate. All throughput numbers were measured on a heavily loaded machine (load average 6–95 while agents ran in parallel), so treat them as ±25–30% lower bounds until the Phase 0 idle-machine rerun.

---

## Implementation status (updated 2026-10-18)

Built and running; see `README.md` to run it and `docs/PROGRESS.md` for the log.

| Plan item | Status |
|---|---|
| PyBullet sim, executor, shield, latency injection, scenario calibration + certification | Done (about 200 unit tests) |
| Arms: waypoint, reflex, FSM, oracle, oracle+error, hover, random; Jev, mercury, GPT-4o-mini, GPT-6 Luna, Gemini | Done; live arms use the real APIs under a hard $15 cap |
| SQLite queue, resumable workers, Parquet trajectories, budget ledger, response cache | Done |
| FastAPI read API, SSE progress, single-file dashboard with side-by-side replay, static HTML report | Done; checked visually on real data (light, dark, phone width) |
| Tier-0 kinematic simulator, MPC arm, RL arm, multimodal arm, Rerun | Not built (deferred as the plan allowed) |
| Periodic (pipelined) decision cadence | Implemented in the runner; the experiments use sequential cadence |

Deviations and findings that changed the plan:

- **Latency in India is higher than published.** Jev measured about 530-600 ms p50 here (published 200-330 ms), so gate G3 is restated by region and every result carries its measured latency.
- **Cruise 2 m/s, dodge speed 1.5 m/s.** Fast repeated sidesteps (2.5 m/s) destabilised the stock PID in wind (crashes 40/40); at 1.5 m/s none do. Calibration, suites and tuned arms were regenerated after this change.
- **Shield semantics.** The shield only stops forward motion; it must not cancel a sidestep in progress (it did at first, stalling the drone). It also senses a forward *corridor*, not a wide arc.
- **Scenario scope.** Headline results cover certified obstacle scenarios only; impossible controls (B < 0), battery and GPS events are reported separately.
- **Accuracy arms use time-based errors.** Per-decision error draws made fast deciders look wrong almost all the time.
- **Surprise worth keeping:** at high latency a noisy policy that brakes randomly collides *less* than the perfect oracle, because stale precise dodges fail where braking is robust.

---

## 1. Executive summary

**The project is feasible and worth doing, but the headline in `initial-plan.txt` has to change.**

1. **The original headline, "Jev out-reacts PID", is almost certainly false.** Jev is a hosted typed-decision API with 110–330 ms p50 latency (530–700 ms p95 via OpenRouter). A PID/FSM reflex layer at 48 Hz acts in 1–2 control ticks, about 21–42 ms. With one request in flight, effective sense-to-act delay is about 1.5× the call latency. On pure reflex speed Jev is 5–15× slower than PID.
2. **The defensible headline** is a measured curve of **collision rate against effective decision latency, with the knee where extra speed stops helping**. It compares a Jev-class decision layer (0.1–0.4 s) against an LLM-class layer (1–3 s), always next to a reflex baseline, with the shield on and off.
3. **Jev is not a chat LLM.** It is TypeSafe's `typesafe/jev-1.13`: you send `state` plus typed `questions`, and it returns per-option probabilities in one parallel pass, with no text, tokens, temperature, seed, logprobs or streaming. "Jev vs LLM" is therefore a comparison of two API shapes, not two models behind one interface (details in section 3).
4. **The simulator is not the bottleneck.** One core runs about 5.9k PID control ticks/s, roughly 120× real time. The bottlenecks are the Jev rate limit (about 20 req/s), the serial decision chain per flight, measurement quality on a shared machine, and scope.
5. **Sim time versus wall-clock is the central methodological trap.** In a prototype, lockstep accounting gave a flat 0.2% collision rate at every latency. The same policy with latency injected as ticks gave 12% at 100 ms, 47% at 300 ms and 99% at 1 s. Lockstep also flipped the ranking of a fast-but-sloppy arm against a slow-but-accurate one. The recommended design is **latency injection on sim time, plus a small real-time-paced validation set** (section 6).
6. **Core MVP is about 36–47 solo working days.** Phase 0 (4–5 days) contains five go/no-go gates, each with a pre-declared fallback (section 13). Everything in Phase 5 is cuttable.

---

## 2. What changes versus `docs/initial-plan.txt`

| Plan statement | Finding | Source |
|---|---|---|
| Current version "explicitly lists Apple Silicon/macOS 26 as tested" | True in the main README, but **pybullet has no macOS wheel for any Python version**. Installing needs `CFLAGS="-Dfdopen=fdopen"` and a 1–3 min source build (works on 3.14.2 [measured]). The repo moved to `learnsyslab/gym-pybullet-drones`, is not on PyPI, and its only release is v1.0.0 (2021). Pin a SHA (`7ebad1e`). | 02, 03, 12 |
| "~20 kHz with 80 drones across 4 environments" | From the pre-2023 README only; an aggregate on 2018–2020 Intel laptops; the same table shows 0.8–0.95× real time for that case; absent from the current README. **Measured, 1 drone on this M4:** 130–170k raw physics steps/s, about 37k with PID, **5.9–7.3k control steps/s per core at 240/48 Hz with PID**. | 02, 11 |
| Jev "gets the state and has to react" | Jev takes typed questions and returns probabilities. Alpha endpoint, hidden from the default OpenRouter model list. | 01 |
| Latency bars: PID short, Jev shorter, LLM long | Reflex layers win on pure avoidance. The demo must use measured or modelled labels and a latency-matched ghost. | 01, 04, 08, 12 |
| "Velocity control" as a ready feature | `VelocityAviary` caps speed at **0.25 m/s** (stopping distance about 4 cm, so every event is trivially avoidable), and it holds position only through velocity damping (drifts 1.0 m in 10 s under 0.02 N of wind). | 02, 06, 11 |
| RGB/depth/segmentation sensors | Real but 2–60 ms per frame (CPU renderer), about 100× a physics step. Use `rayTestBatch` (about 9 µs for 16 rays). | 02, 07, 12 |
| Obstacles | Built-in obstacles are hard-coded URDFs with 85–126 ms reset. Create primitives and spawn them at event ticks. | 02 |
| Wind, battery drop, GPS loss | None are native. Wind needs `applyExternalForce` every physics sub-step. GPS loss is cosmetic without an estimator. Most events are stressors, not discrete-reaction events. | 02, 06 |
| Maximise "simulation speed × decisions × environmental changes" | Wrong objective for API arms. Decisions per second per flight is at most 1/L and the API is rate-limited. | 09, 01 |
| "Thousands of simulated flights" for Jev | 10k Jev flights take 9–28 h. Sub-sample API arms to 500–1,000 paired scenarios. | 01, 09 |
| Sim runs much faster than real time, suiting low-latency demos | That speed-up is exactly what makes naive latency timing meaningless. Latency must become sim ticks. | 08 |
| Compare Jev vs PID vs a conventional policy vs a slower model | Needs a no-avoidance PID *and* a competent conventional arm (an FSM over the same 11 actions), reflex shield on/off, oracle, and random/hover sanity arms. A PID with no avoidance is a strawman. | 05, 08, 11 |
| Don't start with Webots | Agreed. | 03 |
| Jev shouldn't replace the low-level controller | Agreed. | 05 |

---

## 3. What Jev actually is (and what is still unconfirmed)

**[docs], confidence about 0.9 that this is the Jev you mean. Please confirm (open question 1 in section 16).**

- **Product:** TypeSafe AI "System One" decision model. Verified on OpenRouter's public model API as `typesafe/jev-1.13`, modality `text->decisions`. It is hidden from the default model list and only appears with `output_modalities=all`. `typesafe/jev-router` is a different product (an LLM router).
- **Interface:** JSON `state` plus typed `questions` (Choice with up to 255 options, Score, yes/no). It returns per-option probabilities in one pass. OpenRouter path is `POST /api/alpha/decisions`; there is also direct access at `api.typesafe.ai/v1/systemone`. Status is alpha, launched 2026-09-15 in early access.
- **Missing features:** no `max_tokens`, temperature, seed, logprobs or streaming, and no documented determinism.
- **Weak at:** counting, arithmetic and raw numbers. So the harness must precompute bucketed semantic state (time-to-collision, bearing, sector minima) rather than send raw floats.
- **Published latency:**
  - Independent paper (arXiv 2609.22753): p50 315–320 ms, p95 530–625 ms via OpenRouter; 0.274 s median in another table.
  - OpenRouter blog: p50 194 ms, p95 633–688 ms.
  - `RomanSlack/jev-drone` project, best case: p50 110–118 ms, sequential ceiling 7.4 Hz.
  - Vendor: 70–500 ms, mostly about 100 ms from US West.
  - Versus a fast flash LLM the paper found only a 10–37% median advantage.
  - I measured only the network floor: about 25–30 ms warm HTTP/2, 67–96 ms cold. No key exists yet, so no real inference latency was measured.
- **Price and limits:** $0.042 per million input tokens. The rate limit is 40 req/s in docs and 20 req/s elsewhere; plan on 16–20.
- **Closest prior art:** `RomanSlack/jev-drone` (MIT, MuJoCo): Jev at about 2.5 Hz under a 50 Hz reflex veto. It has no slow-LLM arm, latency sweep, adversarial events or multi-flight statistics.
- **Local stand-in:** an open-source System-One-compatible server exists (Ollama `/v1/systemone` with `nimble` and `tev1`). Not run yet.
- **Unverified:** the exact live request shape, TypeSafe's contract terms on benchmark publication, and whether the alpha path is stable.

**Design consequence.** The decision contract is Jev-shaped (action, per-action probabilities, confidence, optional auxiliary scores). Chat LLMs are adapters onto it (one-token letter code plus `top_logprobs` where available). The arms stay comparable only through a shared `Observation` and action glossary, with a per-provider projection.

---

## 4. Architecture

### 4.1 Overview

```
                         ┌─────────────── one SQLite file (WAL) + Parquet trajectories ───────────────┐
                         │                                                                            │
  FastAPI (control/monitor plane)                     Runner CLI (execution plane, spawn pools)
  - create experiment, enqueue flights                - supervisor claims flights with leases
  - progress via SSE                                  - sim workers (PyBullet, one client each)
  - summaries, replay JSON                            - API workers (asyncio, Jev/LLM calls)
  - async SQLAlchemy + aiosqlite                      - latency probe process (quiet, separate)
  - never hosts a worker pool                         - sync stdlib sqlite3, BEGIN IMMEDIATE per flight
```

Inside a flight (per worker):

```
 PyBullet CtrlAviary subclass (240 Hz physics, 48 Hz control)
   -> sensors (16-ray ring, noisy pose/GPS/battery, observation layer faults)
   -> Observation (shared dataclass, per-arm projection)
   -> [harness] latency policy: request issued at tick t, result applied at t + ceil(L/dt)
   -> DecisionProvider (Jev | chat LLM | local | FSM | RL | oracle | mock)
   -> ActionExecutor (latch semantics, slew, dwell, hysteresis)
   -> Shield (S0 off / S1 on)
   -> DSLPIDControl adapter -> RPM -> drone
```

### 4.2 Decisions (resolved conflicts between research agents)

| Topic | Decision | Why |
|---|---|---|
| Simulator | **PyBullet via gym-pybullet-drones only for v1.** Subclass `CtrlAviary`; never `VelocityAviary`. | Measured about 120× real time per core, so the full 10k × 7 × 7 × 30 s matrix is about 5–6.5 h wall. File 03's custom kinematic "Tier-0" sim rested on a stale 5 kHz figure. |
| Tier-0 kinematic sim | Build only as a **surrogate** for the feasibility certifier, oracle and labeller. Promote to a second backend only if the measured M3 wall time exceeds about one working day **and** the rank-preservation test (Spearman ρ ≥ 0.9) passes. | Avoids a second simulator to maintain. |
| Python | **Stay on 3.14.2, one venv.** Every package needed has a cp314 arm64 wheel; pybullet builds from source with the flag. A separate 3.12 env is a documented contingency only. | 02, 12, 14 |
| pybullet build | Flag goes in `scripts/setup_sim_env.sh`, never in requirements. gpd installed `--no-deps` at a pinned SHA. | 14 |
| Execution engine | **Runner CLI with `spawn` pools.** FastAPI only enqueues and reports. Ray: no. Free-threading/JIT/subinterpreters: no. | 09 |
| Job queue | `flights` table **is** the queue (design from file 10), with lease and attempts semantics from file 09. Measured 13–18k claims/s, zero duplicates, SIGKILL recovery OK. | Resolves file 09 vs 10 conflict. |
| Latency ownership | **The harness owns latency, staleness, fallback and shield.** Providers do not. | Keeps arms fair. |
| Shield | A factor: S0 (off) and S1 (on), reported separately. | A 20 ms shield wins without any "decision" being involved. |
| Speed regime | Cruise **2 m/s** (range 1–2). Below that, events are trivial; at 4 m/s and above the stock PID tilts 73–89°. | 02, 06, 11 |
| Control tick | 240 Hz physics / 48 Hz control (20.8 ms). Latency points under 21 ms collapse to one tick. | 02, 08 |
| Wind severity cap | About 0.054 N (a_w ≤ 0.5 × a_brake) for certified scenarios. Stronger gusts only in an uncertified "stress" class. | File 02's "0.05–0.3 N" exceeds brake authority above 0.054 N. |
| Brake and hover | HOVER = true position latch. BRAKE = velocity-zero, or forward-offset latch. **Never latch at the current position while moving.** | Measured 72–108% velocity reversal at 1–3 m/s (11b). |

---

## 5. Experiment design

### 5.1 Arms

| Arm | What it is | Notes |
|---|---|---|
| A / A′ | Waypoint PID with no avoidance / plus shield | A is a lower bound, not a competitor |
| B1 | PID + TTC-brake + lateral nudge | Primary competent reflex arm. APF, VFH, ORCA as ablations |
| E | Rule-based FSM over the **same 11 actions** | Fairest non-ML baseline. Thresholds from `d_stop = v(t_lat + t_ctl) + v²/(2·a_brk) + r + margin` with a measured a_brk, derated to 0.7× |
| G | **Jev** (OpenRouter; direct optional) | Pinned to `typesafe/jev-1.13`; never the `-latest` alias |
| F | Fast chat LLM (hundreds of ms) and a slow LLM (1–3 s) | Two separate arms. Jev vs a fast LLM is a small gap; vs a large/reasoning model it is 5–20× |
| Local | System-One-compatible stand-in (Ollama) | Insurance against alpha API changes |
| Z0–Z3 | Always-hover, random, etc. | Sanity floors. Z1 should score zero collisions but zero success |
| O / O2 | Oracle with privileged state at zero latency / at latency L | Must be ≥ 99% collision-free on certified scenarios |
| C (MPC) | OSQP double-integrator, N=20, dt=0.1 on top of the PID | **Phase 5, optional.** Measured 0.11 ms median solve. cvxpy 28 ms; CasADi+IPOPT 65 ms median. acados rejected (not on PyPI) |
| D (RL) | SB3 PPO, discrete 11-way head | **Phase 5, optional.** 5M steps per policy, 3 seeds, 10–25 h laptop time. Pilot first |

**Tuning parity protocol:** disjoint train/tune/test seed ranges; three-currency tuning budget (effort, evaluations, compute); frozen, hashed configs; every arm including the classical ones gets the latency sweep; a shared shield, PID and action executor; an information-parity table (same noisy rays for every arm, no ground truth leaking into prompts or shield).

### 5.2 Scenarios and adversarial events

- **Core concept: the reaction budget** `B = (d_spawn − d_min0) / v_closing`. A reactor with effective latency L_eff avoids the obstacle iff `L_eff ≤ B`. So P(collision) equals the CDF of your chosen budgets by construction. Consequently:
  - Generate scenarios budget-first and sample `B ~ LogUniform(10 ms, 5 s)` for the headline suite, plus a stratified equal-N-per-tier suite for power.
  - **Never report a single aggregate collision rate.** Report outcomes against `ρ = L_eff / B`, stratified by tier T0 (B ≥ 3 s) to T5 (B < 30 ms), plus an impossible control tier (B < 0).
  - Measured braking: stop distance 0.25 / 0.56 / 0.88 m at 1 / 2 / 3 m/s; deceleration 2.0 / 3.5 / 4.9 m/s² (speed dependent); velocity time constant about 0.15–0.2 s. Use a **measured lookup**, not constants, with κ = 0.85 and a +0.10 m margin.
- **Certification:** replay every scenario with a 1-tick zero-latency oracle in the real sim; reject and resample on failure.
- **Seeding:** per-channel Philox streams keyed by sha256 of the channel name, pre-generated by step index, so randomness does not depend on the arm. Scenario id is a hash of canonical JSON. Events are timed in seconds, not ticks.
- **Spawns** are range-triggered and world-fixed, so every arm sees the same obstacle at the same range.

| Event | Model | Class |
|---|---|---|
| Sudden obstacle | Parked kinematic body teleported into place at a range trigger | **Discrete-reaction (deadline)** |
| Second drone | Scripted, non-adaptive intruder (kinematic sphere) in certified tiers | Discrete-reaction |
| Battery critical | rpm scale by V/V_ref (thrust ∝ V²), sag plus derating | Deadline |
| GPS loss | Observation-layer fault; flagged loss is a deadline | Deadline if flagged; cosmetic otherwise |
| Wind / gusts | Mean-wind steps + OU turbulence via per-sub-step `applyExternalForce`, applied at the drone's current position | **Stressor** (shrinks the margin) |
| Sensor noise / dropout | Observation layer, separate RNG stream | Stressor |
| Target moves | Waypoint trajectory | Stressor |

Report reaction results only for deadline events. Wind, noise, unflagged GPS drift and target moves are stressors with no discrete correct action, so "reacts to all 7 events" is an overclaim.

### 5.3 Ground-truth labels

Viability-set labels from point-mass rollouts over all 11 actions: a safe-action set, a preferred action and a deadline. Stored at observation time and at apply time, computed offline, versioned with `label_version`. Include decoys and ghosts for false-alarm measurement.

---

## 6. Time model (the critical design)

| Option | Description | Verdict |
|---|---|---|
| (a) Lockstep | Sim waits for the decision | **Never use for latency comparisons.** Keep only as a labelled "latency-free quality" ablation |
| (b) Real-time pacing | Sim runs at 1×, drone continues on its last action while the request is in flight | Faithful but slow. **Use as a validation set** (≥ 300 flights × 3 latencies) |
| (c) Latency injection | Sim at max speed; latency converted to `N = ceil(L/dt)` ticks; old action stays in force meanwhile | **Primary.** Record the per-decision latency trace so runs replay exactly |
| (d) Hybrid | Record real latency distributions, replay as sampled delay | The reproducibility half of (c) |

Rules:

- Every arm gets a non-zero, measured-or-declared latency, including PID/MPC/FSM/RL.
- Out-of-order results use a monotonic-apply rule; timeouts use the same fallback for every arm.
- **Cadence is an explicit factor:** sequential (primary) and periodic 100 ms (secondary). At L = 300 ms, pipelining cut collisions from 47% to 33% at 3× the calls.
- The real-Jev sweep runs only at native (recorded) latency. Latency × accuracy sweeps use an oracle-plus-error mock.
- **Throughput is four separate metrics:** decisions per sim-second, decisions per wall-second (total), decisions per wall-second (compute only), and `sim_speed` compute versus total. Never a single "decisions/sec".
- **Measurement hygiene:** API clients live in a separate process from the sim; `perf_counter_ns`; a loop-lag monitor; a dedicated quiet latency-probe process; interleaved arm ordering; a canary prompt to detect network drift; timed runs on an otherwise idle machine under `caffeinate`.

---

## 7. Metrics

- `decision_latency` (request to parsed action, split into network / TTFT / generation where available; Jev rows have null TTFT and tokens).
- `reaction_time` (event onset to first relevant action change), about 1.5 × L under sequential cadence.
- `collision` (contact via `getContactPoints`), near-miss and minimum distance, `time_to_target`, `path_length`, `recovery_time` (back to nominal within ε), success, energy, action-switch rate (oscillation), jerk, invalid-output rate, timeout and fallback rate, shield override rate.
- Raw and shielded collision rates are both reported.

**Statistics:** scenario is the unit of analysis; paired design with common random numbers; Wilson CIs and bootstrap; exact McNemar and Wilcoxon; Holm correction; Kaplan-Meier for censored times; latency as ECCDF plus percentiles with CIs.

**Power (80%, α = 0.05):** a 10 pp difference at a 30% baseline needs about 356 flights per arm unpaired, or 161–270 paired. A 5 pp difference needs about 1,400 unpaired or 600–1,000 paired. Plan **≥ 2,000 scenarios for cheap arms and 300–500 paired for live-API arms.**

**Reproducibility manifest (required):** git SHA, package lockfile hash, gpd SHA, model id and snapshot id per call, provider route, seed, config hash, scenario-suite hash, calibration id, latency mode, latency trace id, client region, machine load at start.

---

## 8. Action space and controller mapping

The 11 actions (FORWARD, LEFT, RIGHT, ASCEND, DESCEND, BRAKE, HOVER, AVOID_LEFT, AVOID_RIGHT, RETURN_HOME, EMERGENCY_LAND) are kept, with canonical integer codes in enum order.

- **Two-channel latch** (horizontal and vertical). FORWARD means "toward the active goal". LEFT/RIGHT/ASCEND/DESCEND are displacement-bounded so a late decision cannot run away. AVOID_* are TTL modifiers that return to the underlying action. EMERGENCY_LAND is an irreversible terminal latch.
- **Executor, identical for all arms:** slew 3 m/s², dwell 0.25 s, side-commit 0.5 s, Schmitt trigger 3.0 / 4.0 m, no-op suppression.
- `ttl_mode` default `hold_last`; `ttl_to_hover` (2 s) is a named second factor and the two are never mixed in one comparison.
- **Shield priority lattice** P0 envelope > P1 hard failsafe > P2 collision > P3 watchdog > P4 provider > P5 default. The shield sees noisy sensor estimates, not ground truth.
- **Yaw:** `target_rpy_rates` alone is not a yaw-rate command (9° transient, no sustained rotation). Integrate `target_rpy[2]` and add the rate as feed-forward.
- **Gaps noted but deferred:** an optional speed level (SLOW/NORMAL/FAST, "E1") only if Phase 4 shows it matters; yaw only if sensing is directional; TRACK_TARGET is redundant; AVOID_UP/DOWN only with overhead obstacle scenarios. Jev cannot slow down to compensate for its own latency, which goes in the limitations.
- **Library gotchas:** `env.reset()` does **not** reset the PID controllers, so call `ctrl.reset()`. `applyExternalForce` must be applied at the drone's position (at the world origin it crashed the drone). `SimplePIDControl` no longer exists; use `DSLPIDControl` only. There is no collision flag.

---

## 9. Observation, prompt and sensing

- **Sensing:** a 16-ray horizontal ring at 22.5°, range 10 m, Gaussian noise plus dropout, cast at decision rate (20–50 Hz), not physics rate. Cost about 9 µs p50 [measured]. **No depth images in the main comparison** (3–15 ms at 64×64; 18.5 ms at 320×240). Multimodal is an optional side arm.
- **Observation:** one shared dataclass, pure per-arm projections, ground truth isolated, explicit `None` for lost sensors, `decision_age_ms` as an ablation factor.
- **Default text format:** a hybrid terse encoding (about 134 tokens): body frame f/l/u, rounded numbers, 8-sector minima, and harness-precomputed `near / ttc / brg / d`. The alternatives measured: JSON 171, key=value 138, sector array 137, natural language 144, ASCII grid 89 tokens (cl100k; Jev's tokenizer is unknown, so this is a proxy).
- **The harness does all arithmetic** (closing speed, TTC, bearing, sector minima, extrapolation to decision time). Jev is documented as weak at numbers.
- **Prompt:** a fixed static system prompt of about 545 tokens first (action glossary, priority rules, 2 examples), the observation last, for prefix caching. Single-token wire aliases for actions; `max_tokens` 1–2 for chat LLMs; `temperature=0`; no chain-of-thought in the Jev arm.
- **Format ablation is mandatory.** Literature (FormatSpread) reports up to 76 points of accuracy swing from formatting alone. Stage 1 is offline on oracle-labelled observations with a mirror-symmetry test; Stage 2 is closed-loop on the best and worst formats. Claims are stated per format or as a min/max range.
- **Parser:** exact match, then unique extraction, then fail. It never guesses on ambiguity and has no in-loop retries. The fallback chain is hold-last-safe-action (TTL), then reflex shield, then HOVER, then abort after repeated failures.

---

## 10. Data and storage

- **SQLite (STRICT, WAL)** for metadata, decisions, metrics, the queue and the model-call cache. **Parquet chunks** (zstd + byte-stream-split, 12–24 Hz) for trajectories, referenced by `flights.trajectory_uri`. No trajectory rows in SQLite (about 204 KB per flight as rows versus 54 KB as Parquet).
- **Tables:** experiments, scenarios (seeded JSON), scenario_events, arms (with `config_sha`), flights (doubles as the queue: `pool, priority, mode, attempts, lease_until`), decisions, decision_labels (offline, separate), event_outcomes, model_calls (cache), flight_metrics (wide row), plus `v_flight_summary`. The `actions` table is seeded from the enum. Idempotency key: `UNIQUE(scenario_id, arm_id, latency_cell, rep)`, plus `mode`.
- **Writes:** one sync `sqlite3` connection per worker, one `BEGIN IMMEDIATE` transaction per flight (decisions + metrics + `status='complete'`). Pragmas: WAL, `synchronous=NORMAL`, `busy_timeout=30000`, `foreign_keys=ON`, `cache_size=-65536`. Use Core `executemany` for hot tables (the ORM was about 85× slower).
- **Measured:** 8 processes into one WAL DB about 660k rows/s at batch 500 (873k at batch 5000). You need about 300 rows/s. Lock errors only appeared with `busy_timeout=0`.
- **Size for 10k flights:** about 0.25–0.4 GB SQLite plus 0.14–0.54 GB Parquet. `WITHOUT ROWID` on decisions cut size from 46 MB to 30 MB per 1M rows.
- **Analysis:** DuckDB over Parquet; pandas `read_sql` for small tables. DuckDB's SQLite extension is not bundled, so export decisions to Parquet.
- **Gotchas:** STRICT tables reject the `VARCHAR` SQLAlchemy emits for `str`, so set `type_annotation_map={str: Text}`. SQLAlchemy 2.1.1 no longer pulls in greenlet, so use `sqlalchemy[asyncio]`. Keep `data/` off iCloud-synced `~/Documents` (WAL hazard) and out of Spotlight; add `data/`, `*.db*`, `*.parquet` to `.gitignore`.

---

## 11. Throughput, parallelism and cost

**Machine:** M4, 4P + 6E cores, 16 GiB. An E-core measured about 0.58× a P-core, so the chip is about 6.5 P-core equivalents after reserving one. No core pinning on macOS. Use **6–9 CPU workers** (spawn), never more than 10 processes; a mixed campaign is 6 CPU + 2 API (asyncio, about 16 flights each) + 1 local. Keep torch and SB3 out of sim workers.

| Arm | Flights/hour (planning, conservative) | Bottleneck |
|---|---|---|
| Classical PID | about 33k | CPU |
| MPC (5 ms solve) | about 6k | CPU |
| Local LLM | about 540 | GPU |
| API-fast (Jev) | about 900 | Rate limit |
| API-slow | about 7.5k | Rate limit |

File 09 assumed 1 ms per tick, which is 3–6× conservative versus the measured 0.17 ms. Replace with Phase 0 measurements.

- **Calls per flight** = `T_f × min(f_d, 1/L)`: about 200 at L ≤ 100 ms, 67 at 300 ms, 20 at 1 s, 7 at 3 s. A latency sweep multiplies real calls to about 893 per scenario over 7 cells, which is why the real-Jev sweep runs at native latency only.
- **Concurrency:** by Little's law, 20 req/s × 0.3 s ≈ 6 concurrent Jev flights maximum.
- **Time and money:** 10k Jev flights take 9–28 h and cost about $84 at 200 decisions each. 10k slow-LLM flights cost $64–128. Plan **API arms on about 1,000 paired flights**, total API spend under about $100–300. Wall clock and rate-limit tier dominate, not dollars. Only the Anthropic prices and the OpenRouter limits were verified from primary docs; Groq/Cerebras prices came from conflicting secondary sources. Build a `$` ledger with a hard `BUDGET_USD` stop.
- **Response cache:** yes, keyed by (provider, model, route/version, params, prompt). It stores the **original recorded latency**; cache hits are excluded from latency and decisions/sec statistics; never use it in probe or validation runs.
- **Queue:** leases, partial indexes and a two-statement claim (an `OR` in the claim query defeated the index and made it 36× slower).
- **RL (only if approved):** 45–60M env steps, 10–25 h of laptop time contending with the sim CPU pool. Pilot first.

---

## 12. Repository changes

The template is a FastAPI skeleton: `app/db/database.py` is empty, there are no models, repositories, services, schemas, Alembic or lifespan, and `database_url` defaults to `""`. These are the recommended changes; none have been applied.

**Layout (about 130 files, file-level detail in `14-repo-architecture-fit.md` section 4):**

```
app/core/        config.py (+settings), exceptions.py, logging.py
app/db/          base.py, database.py (aiosqlite async engine), Alembic
app/models/      SQLAlchemy Mapped models (STRICT tables)
app/schemas/     action, scenario, experiment, decision (pydantic)
app/domain/      observation, setpoint, records, Protocols, clocks, errors
app/sim/         rng, surrogate (pybullet-free), scenario_aviary, pybullet_backend, sensors, events/*
app/controllers/ dsl_pid adapter, executor, shield, waypoint, reactive, fsm, mpc_osqp
app/providers/   encoding, parsing, prompts, mock, jev, chat, system_one_local, replay
app/oracle/      labeller, provider
app/scenarios/   generator, certifier, suites
app/harness/     latency, decision_loop, metrics, recorder, flight_runner, store, worker, supervisor, manifest, budget, cache
app/analysis/, app/replay/, app/api/v1/ (experiments, flights, results, replay)
scripts/         run_experiment.py, setup_sim_env.sh, calibrate_sim.py, generate_scenarios.py, probe_latency.py, validate/*
configs/         experiment YAML, prompts/actions.yaml (single-sourced glossary)
```

**Layering:** the guide's routes/services/repositories/schemas/models/utils become L0 utils < L1 schemas < L2 domain < L3 implementations < L4 harness < L5 scripts/API. Six import rules are enforced by an `ast` script. Only three files may import pybullet. Workers never import FastAPI, SQLAlchemy or torch. Hot-path types are frozen dataclasses; config and DTOs are Pydantic.

**Dependencies:** split into `requirements/{core,sim,ml,analysis,dev}.txt`, floors taken from measured versions (numpy 2.5.3, scipy 1.18.1, gymnasium 1.3.0, SQLAlchemy 2.1.1, aiosqlite 0.22.1, duckdb 1.5.6, pyarrow 25.0.1, pandas 3.0.6, torch 2.14.0, SB3 2.9.0, rerun-sdk 0.38.1). Lock with `uv pip compile` and record the lock hash in every manifest.

- **Delete (recommend only):** `asyncpg`, `psycopg2-binary`, `boto3`, `requests`, `python-dotenv`, `python-multipart`, `Dockerfile`, `docker-compose.yml`, `.dockerignore` (none are imported; the slim Docker image cannot compile pybullet).
- **Add:** `aiosqlite`, `alembic`, `httpx[http2]` (moved from dev to core), `PyYAML` (**unverified on 3.14.2**; fall back to stdlib `tomllib` or JSON if it has no wheel).
- **Pre-commit:** add `sqlalchemy`, `aiosqlite`, `numpy`, `httpx`, `types-PyYAML` to the mypy hook's `additional_dependencies`, otherwise those types resolve to `Any`.

**Guide vs reality (flag, do not silently fix):** the guide says JSON logging but the code is plain text; it names `ApiResponse`, `PaginatedResponse` and a pagination utility that do not exist; it claims isort is enforced but there is no ruff config; it says `routes/` but the repo has `app/api/v1/`; and "async everywhere" does not apply to sim workers. File 14 lists 11 such deviations to record as reviewed exceptions. The guide's other rules stand: full type hints, no `print`, class-based DI, `%s` logging, constants at the top of scripts, tests only when asked, and the py_compile / ruff / mypy checklist after every change.

**Keys and config:** `.env` (gitignored) holds `OPENROUTER_API_KEY` and optionally `TYPESAFE_API_KEY`; never log or commit them. Experiment YAML is parsed into a schema and hashed into the manifest.

---

## 13. Roadmap (solo developer; effort in working days)

Calendar time is about 1.3× because measurement windows must run alone.

| Phase | Days | Content |
|---|---|---|
| **0 Spikes and gates** | 4–5 | Confirm Jev identity and get keys; clean 3.14.2 install; idle-machine benchmark; real Jev and LLM latency (direct vs OpenRouter, several times of day, 11-option Choice, rate-limit ramp); Jev offline accuracy on 100 labelled scenes; brake calibration plus latency-injection knee prototype on the real stack; local stand-in latency |
| **1 Core sim + harness** | 7–9 | CtrlAviary subclass, executor + shield, DSL adapter, scenario JSON + Philox RNG, latency injector, `DecisionProvider` + mock/oracle, metrics, SQLite store, runner CLI with leases, repo wiring |
| **2 Baselines, events, certification** | 8–10 | Arms A/A′, B1, E, Z0–Z3, oracle; intruder, wind, battery, noise, flagged GPS; certification; parity table. **Stop adding events here** |
| **3 Providers, latency, prompts** | 5–7 | Jev/chat/local providers, cache, budget ledger, probe process, two-stage format ablation |
| **4 Main experiment** | 6–8 | Pre-registered manifest; cheap arms on ≥ 2,000 scenarios × latency cells; API arms on 500–1,000 paired; shield S0 and S1; cadence factor; real-time validation set |
| **5 Optional** | 0–10 | MPC (2–3 d), RL (5–7 d + 10–25 h compute), Tier-0 backend, multimodal, Rerun. Each is go/no-go |
| **6 Analysis and demo** | 6–8 | DuckDB + matplotlib report, HTML replay player, FastAPI summary/SSE/replay endpoints, README with install flag and claims ledger |

Core MVP cumulative: **36–47 days**.

**Phase 0 gates, each with a fallback:**

- **G1 (install):** hover loop runs on 3.14.2 from a clean env in under 10 min. *Fallback:* custom MuJoCo quad (+2–3 days).
- **G2 (knee exists):** oracle-brake collision rate rises monotonically with injected latency over 0.1–1 s on scenarios certified feasible at zero latency, and the calibrated B matches the boundary within 10%. *Fallback:* change the speed regime or spawn distances.
- **G3 (Jev usable):** p50 ≤ 400 ms, p95 ≤ 1 s, error rate under 1%, at least 10 req/s sustained. *Fallback:* local stand-in for the sweeps; Jev at recorded latency only.
- **G4 (Jev competent):** acceptable-action rate ≥ 0.7 on clear-threat states, above always-hover and random with a CI. *Fallback:* redesign the state encoding (1 day); if still below 0.5, reframe as a latency-accuracy trade-off with Jev as a measured weak point.
- **G5 (framing):** headline is the collision-vs-latency curve and knee, not "Jev beats PID".

**Build milestones** (from file 14; earliest scientific output is the end of M3): M0 foundations (1–2 d), M1 vertical slice with a mock provider, M2 scenario science (1–2 wk), M3 classical arms + first collision-vs-latency curve (1–2 wk), M4 providers with real Jev (parallel with M3), M5 analysis + API + demo (2 wk), M6 stretch.

**Other phase exit criteria worth enforcing:** Phase 1 must reproduce `P(collision) = CDF_B(L_eff)` within ±3 pp on 500 paired oracle flights, show identical trajectory hashes across two spawned processes, and survive `kill -9` mid-run with an exact flight count and no duplicates. Phase 4 needs real-time versus injected per-flight agreement of at least 90%.

---

## 14. Top risks and bottlenecks

Full 33-row register with likelihood, impact, mitigation and early-warning tests is in `13-risks-bottlenecks-failure-modes.md`. The ones that matter most:

| Risk | Mitigation |
|---|---|
| "Jev out-reacts PID" is false | Reframe the headline now (G5). Always show the reflex baseline. |
| Scope creep (7 arms × 7 events × 3 time models × 2 sim tiers × RL × multimodal × 3 viz stacks × prompt factorial) | MVP = obstacle + intruder events, arms A′, B1, E, Jev, fast/slow LLM, Z0–Z1, oracle; time model (c) + (b) validation. Defer T0, RL, MPC, multimodal, Rerun. |
| Jev vs LLM are two API shapes | Shared Observation + glossary, per-provider projection, a Jev-style bridge arm |
| Scenario-distribution artefact: a perfect reactor's collision rate follows the budget CDF; an FSM vs Jev-via-OpenRouter gap of about 41 pp comes purely from where scenarios sit | Stratify by budget tier, report against ρ, publish the B distribution |
| Speed regime (0.25 m/s trivial, ≥ 4 m/s unstable) | 2 m/s cruise, CtrlAviary subclass, calibrated braking table |
| Time-model artefacts | Injection primary, real-time validation, lockstep as a labelled ablation only |
| Strawman or asymmetric baselines | FSM and B1 as competent arms; tuning-parity protocol |
| Shield contamination | S0 and S1 as a factor; claims must hold at both |
| Thermal and contention corrupting latency | Quiet probe process, loop-lag monitor, idle-machine timed runs |
| Prompt-format confound | Precompute derived quantities; two-stage ablation |
| pybullet macOS build | Flag in setup script, Phase 0 gate G1 |
| Rate limits, alpha-endpoint instability, version drift | Pin model and route; assert snapshot id on every response; cache; local stand-in; 16–20 req/s planning |
| Geography (origin reportedly US West; a client outside North America adds about 150–250 ms) | Record region in the manifest; run the API runner from US West as a labelled control if p50 is above 250 ms |
| Jev may simply be poor at this task | Precomputed buckets, confidence-gated shield fallback, G4 |

**Top 5 bottlenecks, quantified:**

1. The Jev serial chain plus rate limit (20 req/s, about 6 concurrent flights, 9–28 h for 10k flights).
2. Real-time validation and latency probing need a quiet machine at 1× wall clock (about 1–2 h per 1k flights).
3. PID compute inside the sim (about 100 µs per call, about three quarters of per-tick cost), but not binding.
4. RL training (10–25 h) if approved.
5. Measurement quality: event-loop lag p99 was 92 ms at 16 in-flight and 0.78 s at 512, against a 100–300 ms Jev signal. Concurrency per process must stay small.

**The latency knee** (at 2 m/s, from files 06, 08, 11): the floor is `v·τ_a + v²/2a` ≈ 0.95 m of spawn distance regardless of decision speed, so below roughly 50–100 ms extra speed buys nothing. The window where a reflex survives but Jev-via-OpenRouter p50 does not is only about 1.0–1.7 m of spawn distance. Going from 3 s to 0.3 s (LLM to Jev) moves the required distance by about 5 m, which is where the signal lies. The p95 tail matters more than p50.

---

## 15. Claims ledger

**Safe to claim (with the evidence noted):**

- Under injected decision latency L, collision rate of arm X on certified scenario set S is P (Wilson 95% CI); the curve rises between a and b ms, the knee set by braking distance.
- Jev measured p50/p95/p99 from this machine, region and time (until measured, use published bands labelled "published, unmeasured").
- A typed-decision API returns a choice plus per-option probabilities in one pass without text generation.
- A reflex layer reacts in 1–2 control ticks and decision layers cannot match that.
- Collision results flip between lockstep and latency-injected accounting for the same policies (reproduced in a prototype; re-run on gym-pybullet-drones).
- Jev is about 10–37% faster at the median than a flash chat LLM on one independent benchmark, and 5–20× faster than large or reasoning LLMs.
- gym-pybullet-drones runs headless on Python 3.14.2 on an M4 at about 5.9k PID control ticks/s per core after a source build of pybullet.
- "To our knowledge, as of Sept 2026, no published study combines a paired latency sweep, reflex on/off, an FSM baseline and feasibility-certified scenarios for a Jev-class model." Hedged, because the literature search was limited.

**Conditional:** "Jev+shield is as safe as FSM+shield" (holds at S0 and S1, tuned parity); "Jev makes better semantic decisions than a fast LLM" (format-ablated, n ≥ 500, latency-matched arm); "results hold with real latency" (≥ 90% real-time agreement); "a faster decision layer is always safer" (only between the knee bounds).

**Do not claim:** Jev out-reacts PID/MPC; Jev is N× faster than LLMs without an arm definition; Jev avoids obstacles in X ms from raw API latency; the simulator runs at 20 kHz or 80 drones; slow LLMs are safest (a lockstep artefact); any absolute collision rate without the B distribution; works on Apple Silicon out of the box; reacts to all 7 events; Jev is first or only; sim-to-real generalisation (a 27 g CF2X in PyBullet); wall-clock `sim_speed` including API time as a model property.

---

## 16. Open questions for you

1. **Is "Jev" = TypeSafe Jev 1.13 (`typesafe/jev-1.13`)?** And do you want OpenRouter only, or also a direct TypeSafe key?
2. **Where are you located (region/ISP)?** It changes every latency figure by up to 150–250 ms.
3. **Demo or publishable comparison?** This decides whether pre-registration and the real-time validation set are mandatory.
4. **Headline LLM arm:** fast chat LLM (hundreds of ms), slow reasoning LLM (seconds), or both? A fast LLM makes the story "Jev vs fast LLM", with a small gap.
5. **Scope of the first result:** is single obstacle/intruder enough, or are moving-target and multi-event scenarios required?
6. **Benchmark publication terms:** TypeSafe's terms were unreadable and OpenRouter's have no benchmarking clause. Is that acceptable for your use?
7. **Is `~/Documents` synced by iCloud?** If so, `data/` must live elsewhere.
8. **Are you OK with the tick resolution?** At 48 Hz a 10 ms and 20 ms latency point are indistinguishable; 240 Hz control costs about 5× more PID calls.
9. **Sensing geometry:** a 16-ray omnidirectional ring with fixed yaw is assumed. A forward-facing field of view changes the action set.

Unresolved technical items: every throughput number needs an idle-machine rerun; cross-process PyBullet determinism is untested; `PyYAML` on 3.14.2 and the exact `gym_pybullet_drones` import set beyond numpy/scipy/gymnasium/pybullet/transforms3d are unverified; the braking-distance column in file 11 differs from the 11b re-measurement by up to 0.14 m (definition mismatch, not resolved); some literature citations (SayCan, Code-as-Policies, RT-2, OpenVLA, NavGPT, UAV-VLA, PromptCraft, ORCA/APF/CBF) were cited from memory and not re-fetched; the point-mass prototypes (files 06, 08) must be re-run on the real stack and on 3.14.2.

---

## 17. Research file index

| File | Topic | Headline |
|---|---|---|
| `00-decomposition.md` | Task split and shared brief | 14 subtasks, 10 global open questions |
| `01-jev-identity-openrouter-latency.md` | Jev identity, API, latency | TypeSafe Jev 1.13; 110–330 ms; `DecisionProvider` design; benchmark script |
| `02-gym-pybullet-drones-internals-performance.md` | Simulator internals and speed | Works on 3.14.2 with the build flag; 5.9–7.3k ticks/s/core; 20 kHz claim stale |
| `03-simulator-alternatives-comparison.md` | Alternatives | PyBullet chosen; Isaac/AirSim/Flightmare excluded; MuJoCo is the fallback |
| `04-prior-art-llm-drone-control.md` | Literature | `jev-drone`, RRARA, FPX; novelty is the paired latency study |
| `05-baselines-pid-mpc-avoidance-rl.md` | Arms | FSM pseudocode; OSQP MPC 0.11 ms; RL budget |
| `06-adversarial-events-disturbances.md` | Events and feasibility | Reaction budget; tiers; seeded JSON scenarios; labels |
| `07-sensors-perception-state-representation.md` | Sensing and prompts | 16 rays; 6 encodings measured; parser and fallback |
| `08-benchmark-methodology-time-accounting-stats.md` | Time model and statistics | Lockstep artefact; injection design; power table; prototype code |
| `09-orchestration-parallelism-throughput.md` | Runner and throughput | Runner CLI; queue; budget; machine facts |
| `10-sqlite-schema-fastapi-integration.md` | Storage and template fit | Full DDL; 660k rows/s; repo audit |
| `11-action-to-controller-mapping.md` | Actions and controller | Latch semantics; shield lattice; section 16 spike results |
| `12-visualization-demo-analysis.md` | Demo and analysis | HTML player plus Rerun; replay contract; 14 plots |
| `13-risks-bottlenecks-failure-modes.md` | Risks, roadmap, claims | 33 risks; 20 contradictions; 15 plan corrections |
| `14-repo-architecture-fit.md` | Repo skeleton and deps | About 130-file skeleton; Protocols; requirements split; build order |

The two files to read before writing any code are **08** (time model) and **14** (skeleton and Protocols). The one to read before making any claim is **13** section 5.
