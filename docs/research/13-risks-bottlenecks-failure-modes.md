# 13 - Risks, bottlenecks, failure modes, contradictions, phased roadmap, claims ledger

Agent 13 of 14 (synthesis; runs last). Written 2026-09-30 after reading `00-decomposition.md`, `docs/initial-plan.txt` and files 01-12 in full (two long code listings inside 01 and 02 skimmed, not line-checked). `14-repo-architecture-fit.md` did not exist when this was written; anything that depends on it is flagged "reconcile with 14".

Notation: **[file NN]** = number or finding comes from that research file (I did not re-measure anything). **[I13]** = my inference/arithmetic from those files. Likelihood (L) and impact (I) are 1-5; score = L x I. "Owner phase" refers to the roadmap in section 4.

---

## 0. The five things that matter most

1. **The plan's headline ("Jev out-reacts PID/MPC") is almost certainly false and must be replaced.** Jev is a hosted typed-decision API with ~110-120 ms (direct, best case) to ~200-320 ms p50 / 530-700 ms p95 (via OpenRouter) [file 01]. A PID/TTC/FSM reflex layer at 48 Hz acts within 1-2 control ticks (21-42 ms) [files 02, 05, 06]. With sequential cadence the effective sense-to-act delay is ~1.5 x call latency [file 08], so Jev arrives 5-15x later than a reflex. The defensible story is: *a semantic decision layer at ~0.1-0.4 s vs ~1-3 s for a big LLM, with a measured collision-vs-latency curve and a knee.*
2. **Every physical-outcome conclusion depends on (a) the time model and (b) the scenario difficulty distribution**, and both are choices the experimenter makes, not findings. Lockstep gives a flat curve and inverts arm rankings [file 08]; choosing log-uniform reaction budgets makes P(collision) equal the budget CDF by construction [file 06]. Report curves against the reaction budget / margin ratio, never a single aggregate collision rate.
3. **The simulator is not the bottleneck; the API (rate limit, serial chain, cost of the latency sweep) and measurement quality are.** One core does ~5.9k PID control ticks/s [files 02, 11] (~120x real time); the Jev arm is bound by ~20 req/s and a per-flight serial chain.
4. **The environment defaults make the experiment trivial or unstable**: 0.25 m/s `VelocityAviary` cap makes events trivial; >= 4 m/s makes the stock PID tilt 73-89 degrees [file 02]. The usable window is ~1-2 m/s cruise, and scenarios must be engineered (spawn distance ~1-2 m at 2 m/s) to land in the region where decision latency matters.
5. **Scope is the largest controllable risk.** Eleven research files added ~7 arms, 7 event types, 3-4 time models, a two-tier simulator, RL, multimodal, a prompt-format factorial, Rerun + HTML demo + SSE. A solo developer needs a ruthless MVP (section 4).

---

## 1. Ranked risk register

Sorted by score (L x I), ties broken by how directly the risk threatens the headline claim. Early-warning tests are cheap and should be scheduled in Phase 0/1 (see section 4).

| Rank | ID | Risk | L | I | Score | Mitigation | Early-warning test | Owner phase |
|---|---|---|---|---|---|---|---|---|
| 1 | R02 | **"Jev out-reacts PID" claim is false.** Jev call latency 110-330 ms (p50), p95 530-700 ms via OpenRouter [01]; effective reaction ~1.5 x L [08]; reflex layer 21-42 ms. The plan's latency-bar demo ordering (PID < Jev < LLM) is true only for *reflex* latency, and then Jev is 5-15x slower than PID. A PID+reactive/shield arm beats Jev on pure avoidance | 5 | 5 | 25 | Reframe: claim "Jev-class latency (0.1-0.4 s) vs LLM-class (1-3 s) on decisions a reflex cannot make", plus a latency-matched comparison and shield S0/S1 ablation [11]. Never draw a bar chart with a mock Jev latency [08]. Keep PID+reactive in every figure as the reflex reference | P0 gate G3: measure real Jev p50/p95; compute `d_min(L)` table (section 2.3). If p50 x 1.5 < the budget of every scenario you built, the experiment cannot show a Jev-vs-LLM gap either | P0, P4 |
| 2 | R17 | **Scope creep** (7 arms x 7 events x 3 time models x 2 sim tiers x RL x multimodal x 3 viz stacks x prompt factorial) | 5 | 4 | 20 | MVP = obstacle + (intruder) events, arms A', B1, E, Jev, slow LLM, Z0-Z1, Oracle; time model (c) + small (b) validation; one viz path (custom HTML) + matplotlib report. Defer T0 tier, RL, MPC, multimodal, Rerun, GPS estimator stub, AVOID_UP/DOWN, yaw. Phase gates are go/no-go, not milestones | Count open arms/events at end of Phase 2; if > 8 arms or > 3 event types, cut | P0-P6 |
| 3 | R01 | **"Jev vs LLM" is two API shapes**, not two models on one interface: Jev takes `state` + typed `questions`, returns choice + per-option probabilities, no text/tokens/temperature/seed/logprobs/stream [01]; chat LLMs take a prompt and emit tokens. Arms F and G cannot be "identical except provider" [05 assumed this]; prompt-format (07) is a chat-LLM concept that does not apply to Jev's wire format | 5 | 4 | 20 | `DecisionProvider` returns the same `DecisionResult` (action + probabilities + timing) from both [01 sec 8]; share `Observation`, glossary and action semantics, not the wire format; declare "interface confound" in the report; add a bridging arm (local SemIf/letter-logit model or Ollama `/v1/systemone`) that has Jev-style output semantics without the network [01] | Write the Jev adapter and a chat adapter against the same 100 scripted scenes (F5); check the glossary survives both | P3 |
| 4 | R05 | **Scenario-distribution artefact**: `P(collision) = CDF_B(L_eff)` by construction [06]. With reaction budgets log-uniform on 10 ms-5 s, a perfect reactor's collision rate at L is `log(L/10 ms)/log(500)` (0.26 at 50 ms, 0.53 at 300 ms, 0.93 at 3 s). A FSM (L_eff ~ 30 ms) vs Jev-OR (L_eff ~ 375 ms) gap is ~41 pp *purely from where you put probability mass* [I13: log10(375/30)/2.7] | 5 | 4 | 20 | Stratify by tier/budget and plot outcome vs `rho = L_eff / B`; publish the B-distribution and headline per-tier rates; state that absolute collision rates are properties of the scenario set. Pre-register the set | Verify the curve collapse: flights from all arms at all latencies should fall on one sigmoid around rho = 1; deviations = policy quality [06 sec 4.1] | P1, P4 |
| 5 | R03 | **Speed regime**: default `VelocityAviary` 0.25 m/s makes stop distance 4 cm, latency irrelevant [02, 06, 11]; > 4 m/s gives 73-89 deg tilt and altitude sag on the stock PID [02]. Various files assume 3-6 m/s [05, 08] | 4 | 5 | 20 | Cruise 2 m/s (range 1-2), subclass `CtrlAviary` (not `VelocityAviary`), raise closing speed with moving obstacles/intruders, not drone speed; calibrate brake table (section 3 contradictions C2/C3). Re-run 08's prototype at 2 m/s | Phase 0 calibration: stop distance at 1/2/3 m/s with the real executor; confirm the knee lies in 0.1-1 s for the scenario spawn distances you chose | P0 |
| 6 | R04 | **Time-model artefacts**: lockstep flat curve and rank inversion [08]; ceil-quantisation (10 ms and 20 ms both become 1 tick at 48 Hz); sequential vs pipelined cadence moves collision 47% -> 33% at L = 300 ms [08]; sample-and-replay hides burst correlation; cached responses have ~0 latency [09] | 4 | 5 | 20 | Primary = (c) injection on `t_sim`; `ceil(L/dt)`; real-time-paced (b) validation of >= 300 flights x 3 latencies; lockstep only as a labelled "latency-free quality" ablation; four separate throughput metrics [08 sec 8] | Reproduce 08's lockstep-flat / injected-rises pattern on the real `gym-pybullet-drones` executor; (b) vs (c) agreement >= 90% per flight and within CI | P1, P4 |
| 7 | R07 | **Strawman or asymmetric baselines**: PID with no avoidance (arm A) is a lower bound, not a competitor; FSM/RL/prompt tuning effort unequal; thresholds leaked into the prompt from the FSM [05, 07] | 4 | 5 | 20 | FSM (arm E) and B1 as the "competent conventional" arms; three-currency tuning budget, disjoint seed ranges, frozen config hashes [05 sec 6]; Jev prompt and FSM tuned on same tune split; publish all trials | FSM alone must reach < 2% collisions on feasible scenarios at 1-tick latency (Phase 2 exit); if a 60-line FSM gets this, Jev must beat *it* | P2 |
| 8 | R21 | **Shield contamination**: a 20 ms collision shield wins without any "decision" being involved; with S1 shield latency effect shrinks to progress loss [11] | 4 | 5 | 20 | Shield level S0/S1/S2 as a factor; headline at S0 *and* S1 (claim must hold at both); log `shield_override_rate`, `first_override_latency`; shield uses the same noisy sensor estimate, not ground truth [11] | In Phase 1, run the Oracle-1-tick arm at S0, S1; if S1 collisions ~ 0 for every arm the S1 curve is uninformative (expected) | P1, P4 |
| 9 | R15 | **Thermal / contention corrupts latency measurement.** Measured load averages: 46-60 (02), 45-95 (09), 30-90 after one fork-bomb (10), 6-12 (10 final runs); ray-cast p95 was 10-50x p50 under load (07); in 09, event-loop lag p99 was 92 ms at 16 in-flight and 0.78 s at 512 while a single flight's Jev signal is 100-300 ms. Throughput numbers across files are +-25-30% and lower bounds | 4 | 4 | 16 | Dedicated quiet latency-probe process with no sim; sim workers in separate processes from the asyncio API client; loop-lag monitor, tag contaminated calls `loop_lag_p99 > 2 ms` [09]; real-time validation alone, <= 4 sim procs, `caffeinate`, AC power; interleave arms in time; canary prompt every 5 min [08] | `sysctl vm.loadavg` < 2 before any timed run; 30-min all-core soak with flights/min per 5-min window (decay < 15%, spike F6 in 09) | P0, P3, P4 |
| 10 | R18 | **Prompt-format / state-encoding confound**: format swings of tens of points are documented [07 via FormatSpread]; Jev documents weak numeric reasoning and accuracy loss with irrelevant state [01]; raw floats lead to mistakes; `decision_age_ms` effect unknown | 4 | 4 | 16 | Precompute ttc/closing/bearing in the harness; bucketed semantic state; two-stage ablation (offline labelled screening with mirror-symmetry test, then closed-loop for best/worst format) [07]; state headline per format or as min/max range | Offline accuracy (F5/S2): acceptable-action rate per format on 500 frozen states + mirror consistency > 85% | P3 |
| 11 | R10 | **pybullet macOS build**: no macOS wheel on PyPI for any Python, sdist only [02, 03, 12]; plain `uv pip install pybullet` fails on this SDK (zlib `fdopen` macro clash); `CFLAGS="-Dfdopen=fdopen"` works on 3.14.2 (build 74 s [12], 2m13s [02], 3m18s failed attempt [02]); one agent's install returned in 0.1 s from a shared cache [11]; `gym-pybullet-drones` is not on PyPI, only release is v1.0.0, torch + SB3 are hard deps [02]; a macOS SDK update can re-break it | 4 | 3 | 12 | Put the flag in a setup script/README, cache the built wheel in `wheelhouse/`, pin gpd commit `7ebad1e`, install `--no-deps -e` plus explicit minimal deps; add a fresh-venv install check to Phase 0. Fallback: MuJoCo quad (cp314 wheel verified, 110k steps/s) [03] | Fresh `uv venv --python 3.14.2` + install + 1000-tick hover + 200-reset memory check (spike 12-F1) | P0 |
| 12 | R12 | **Rate limits and call volume**: Jev 20 req/s (docs say 40, blogs 20) [01]; free OpenRouter tier 20 rpm / 1000 per day [09]; Little's law: in-flight = rps x L = 20 x 0.3 = 6 concurrent Jev flights max; a **latency sweep multiplies real calls** (calls per flight fall with injected L: 200 at <= 100 ms, 67 at 300, 20 at 1 s, 7 at 3 s = ~893 calls per scenario over 7 cells [I13 from 09 D = T_f x min(f_d, 1/L), T_f = 20 s, f_d = 10 Hz]) | 4 | 3 | 12 | Do not run real Jev at every sweep cell: use an oracle-plus-error mock (2-D latency x accuracy map) for the sweep and real Jev at its native latency (+ recorded-latency injection) [08]; sub-sample API arms to 500-1000 paired scenarios; token-bucket + `Retry-After`; `BUDGET_USD` ledger [09] | F6 rate-limit ramp (1, 4, 16, 32, 64 concurrent); measure 429/529 onset | P3 |
| 13 | R08 | **PID under-tuning / controller pathologies blamed on the decision layer**: stock DSL gains are tuned for gentle flight; `VelocityAviary` uses `target_pos = cur_pos`, so only the velocity D-term acts (hover drifts F/0.2 m/s under wind: 1.0 m in 10 s at 0.02 N, 2.5 m at 0.05 N [11b]); `env.reset()` does not reset PID state (episodes not reproducible unless `ctrl.reset()`) [02]; no tilt limit on far targets [11] | 3 | 4 | 12 | Own the controller in `controllers/`: `CtrlAviary` + `DSLPIDControl` with carrot setpoints, `ctrl.reset()` in housekeeping, slew limits; report the PID's own step/brake table | Hover-hold < 0.05 m in 10 s at 0.02 N; episode 3 hash == episode 1 hash after reset; reversal <= 35% on brake | P1 |
| 14 | R06 | **Infeasible scenarios / certification gap**: analytic budget uses assumed `a = 4 m/s2, tau_a = 0.08 s` [06] but measured decel is 2.0 / 3.5 / 4.9 m/s2 at 1 / 2 / 3 m/s with 0.15-0.2 s velocity time constant and 17-34% reversal [11b]; wind and battery derating shrink authority | 3 | 4 | 12 | Calibrate a brake/lateral lookup in the real stack; `kappa_a = 0.85`, +0.10 m; certify every scenario by Oracle-0 replay in the simulator, reject and resample [06 sec 3.7]; keep an `UNAVOIDABLE` control tier (excluded from headline, always reported) | Oracle-0 >= 99% collision-free on certified scenarios; Oracle at `L = 0.9 B` collides ~0%, at `1.1 B` ~100% (spike 06-2) | P1, P2 |
| 15 | R22 | **Jev may simply be poor at this task** (numbers, geometry, 11-way choice): vendor's own jaggedness notes [01]; UTS paper shows accuracy falling with wide contracts (0.53-0.58 vs 0.90-0.93) [04]; jev-drone reports "the state has to contain the answer" [04]; probabilities are coarse [01] | 3 | 4 | 12 | Pre-compute derived quantities; semantic buckets; confidence-gated fallback to shield (vendor pattern); accept a negative result as a valid outcome (Jev below FSM on accuracy, above LLM on latency) | F5: acceptable-action rate and reliability curve on 100 scripted scenes before building the harness around Jev. Threshold: acceptable-rate on clear-threat states >= 0.7, else redesign state/questions or pivot arm framing | P0 |
| 16 | R13 | **ToS / alpha endpoint instability**: `POST /api/alpha/decisions` (status alpha; docs print a different path, docs also list `/api/v1/systemone`) [01]; launched 2026-09-15 in early access [01]; limits "change without notice"; TypeSafe publication/benchmark terms unread; OpenRouter prohibits unauthorised red-teaming (keep "adversarial events" wording = physical sim events) [01] | 3 | 4 | 12 | Pin `typesafe/jev-1.13`; log `model` snapshot per call; response cache + recorded-latency traces; local Ollama `/v1/systemone` stand-in; ask sales@typesafe.ai before publishing; keep adapter path a one-line constant | F1/F2 succeed against the live path in Phase 0; re-probe weekly during Phase 4 | P0, P3 |
| 17 | R32 | **Geography**: origin reportedly US West; client outside North America adds ~150-250 ms [01, I]. Every latency figure is conditional on the user's location | 3 | 4 | 12 | Record client region/ISP in the manifest; if > 250 ms p50, run the API runner from a US-West VM as a labelled control or shift to injected latency from a published distribution | F1 from the actual machine at 2-3 times of day | P0 |
| 18 | R09 | **HOVER/BRAKE controller pitfalls**: latching at the current position while moving gives 72 / 101 / 108% reversal at 1 / 2 / 3 m/s and 0.16 m altitude sag; `velzero` brake still reverses 17-34% [11b]; `latch_stop` works >= 2 m/s but stops at 1.46 m instead of 0.88 m at 3 m/s; `target_rpy_rates` is not a yaw-rate command; wind via `applyExternalForce` with the wrong world point crashes the drone [11b]; forces must be applied every physics sub-step [02] | 4 | 3 | 12 | BRAKE = velocity-zero (or slewed ramp) until `|v| < 0.1` then position latch; never latch at the current position while moving; HOVER = position latch with integrator reset; apply wind at the drone's base position each sub-step; yaw by integrating `target_rpy[2]` | Executor unit tests (pure state machine) plus the brake/hover table reproduced on the real executor | P1 |
| 19 | R28 | **Decision cadence as a hidden factor**: sequential vs periodic(100 ms) changes collision at 300 ms from 47% to 33% and calls per flight 30 -> 90 [08]; files use 5, 10, 20, 50 Hz | 4 | 3 | 12 | Fix cadence as an explicit factor (primary sequential, secondary periodic 100 ms); state `f_d` and `T_f` in the manifest | Include cadence in the 2-D map; check Jev cost at both | P1 |
| 20 | R23 | **Overclaiming across event types**: wind, sensor noise, unflagged GPS drift and target moves are *stressors* with no discrete deadline; constant wind is rejected by the PID integral when latched [02, 06]; GPS loss is cosmetic unless an estimator sits between truth and PID [06] | 4 | 3 | 12 | Claim reaction results only for obstacle/intruder/battery-critical/GPS-loss; report others as margin-shrinkers (wind x obstacle interaction) | n/a (reporting discipline) | P4, P6 |
| 21 | R20 | **Information asymmetry / leakage**: prompt thresholds copy the FSM [07]; `age` pre-compensation gives the LLM a physics engine [07]; oracle/shield using ground truth [11]; arm A has no range sensors by design [05] | 3 | 4 | 12 | Information-parity table in the report; same noisy rays for all arms; shield sees estimates, not ground truth; decide once about thresholds-in-prompt (HINTS factor) | Checklist in Phase 2 exit | P2, P3 |
| 22 | R33 | **Jev identity wrong** (confidence ~0.9 it is TypeSafe Jev 1.13) [01]; "Jev Router" is a different product; Jev is absent from the normal model list [01] | 2 | 5 | 10 | Ask the user to confirm now; DecisionProvider abstraction means the rest survives | One question to the user at project start | P0 |
| 23 | R19 | **Nondeterminism**: Jev documents no determinism; `temperature 0` is not bitwise-deterministic elsewhere [01]; cross-process PyBullet determinism untested (in-process verified) [02]; cache changes latency semantics [09] | 3 | 3 | 9 | Cache actions keyed by (provider, snapshot, input hash) but take latency from recorded live values; consistency probe (same input x 100); test cross-process hash; Philox streams per channel so arms cannot shift each other's draws [06] | F3 consistency probe; 2-process hash test | P0, P1 |
| 24 | R14 | **Model version drift**: `~typesafe/jev-latest` alias moves; OpenRouter chat models can be served by other providers/quantisations [01] | 3 | 3 | 9 | Pin model + `provider.order` + `allow_fallbacks:false`; abort the run if snapshot string changes; record provider per call | Assert snapshot id == expected on every response | P3 |
| 25 | R31 | **LLM arm misconfiguration**: reasoning models spend hidden tokens and return empty output with small `max_tokens`; `logprobs`/`logit_bias` are provider-specific (Groq lacks them) [01]; structured output adds overhead | 3 | 3 | 9 | Use non-reasoning or `effort: low` with `max_tokens` ~64 for reasoning models; `require_parameters: true`; one-token letter outputs | Invalid-output rate < 2% in a 200-call smoke test | P3 |
| 26 | R25 | **Tiered-sim complexity vs value**: 03's case for T0 rested on ~5 kHz legacy figure (85 core-hours at 50x); measured PID ~120x real time per core [02, 11] makes the same 10k x 7 x 7 x 30 s matrix ~33 core-hours = ~5 h wall on 6.5 P-equivalents [I13] (see C1) | 3 | 3 | 9 | Skip T0 initially; keep a point-mass surrogate only for the oracle labeller and scenario generator; revisit only if Phase 4 runtime is > 1 day | Idle-machine benchmark (spike 02-1) | P0 |
| 27 | R11 | **Python 3.14 risk**: core deps have cp314 wheels [02, 05, 10, 12]; Genesis `<3.14`, ML-Agents `<=3.10`, Ray 3.14 unverified, free-threading/JIT/subinterpreters not recommended [03, 09]; SQLAlchemy 2.1.1 no longer auto-installs greenlet (`sqlalchemy[asyncio]` needed) [10]; prototypes were run on 3.12.12 and 3.14.6 in places [08, 06, 11] | 2 | 3 | 6 | Stay on 3.14.2; pin SQLAlchemy `<2.1` or add `[asyncio]`; re-run key prototypes on 3.14.2; no Ray, no free-threading | Fresh-venv resolve + import smoke in Phase 0 | P0 |
| 28 | R26 | **RL arm cost**: 45-60M env steps = 10-25 h laptop time, contending with the sim CPU pool; reward hacking; latency mismatch [05] | 3 | 3 | 9 | Defer to optional Phase 5 with a go/no-go; 500k-step pilot on the point-mass surrogate first | Pilot learning curve vs FSM | P5 |
| 29 | R27 | **Statistical pitfalls**: per-stratum n (7 events x severity) divides power; low-collision regimes need thousands of scenarios; pseudo-replication for stochastic arms; multiple comparisons [08] | 3 | 3 | 9 | Pre-register primary metric/latency; scenario = unit; Holm; per-stratum n planned (2,000 cheap arms, 300-500 paired for live arms, 10 pp detectable) | Power table (08 sec 9.3) checked against chosen N | P4 |
| 30 | R29 | **Novelty overclaim**: `RomanSlack/jev-drone` (MuJoCo + Jev + 50 Hz reflex), RRARA time-conversion, FPX latency-win curves already exist [04] | 3 | 3 | 9 | "To our knowledge, as of Sept 2026" language; cite them; differentiate on paired latency sweep, reflex on/off, FSM baseline, seeded feasibility-certified scenarios | Semantic Scholar sweep before publishing | P6 |
| 31 | R24 | **Action-set gaps**: no speed, no yaw, no vertical avoid; Jev cannot slow down to compensate its own latency [11] | 3 | 3 | 9 | Add optional `speed` field (E1) only if Phase 4 shows the gap matters; keep yaw fixed with omnidirectional rays | Note in limitations | P2 |
| 32 | R16 | **SQLite / storage**: low technical risk (8 writers 660-870k rows/s; need ~280 rows/s) [10]. Real risks: repo under `~/Documents` (possible iCloud sync of WAL files) [09, 10]; ORM 85x slower than Core for bulk rows; two competing queue designs (see C5); `.gitignore` lacks `data/`, `*.db*` | 2 | 3 | 6 | Local non-synced `data/`, sync `sqlite3` in workers, WAL + `BEGIN IMMEDIATE` + `busy_timeout=30000`; Parquet for trajectories | `brctl status`/check iCloud Desktop & Documents; kill -9 resume test | P1 |
| 33 | R30 | **Repo-convention friction**: `app/db/database.py` is empty, no pagination utility/`PaginatedResponse`, no lifespan, guide says `routes/` but repo has `app/api/v1/`, Docker files are dead weight, sync DB code in workers deviates from "async everywhere" [10] | 3 | 2 | 6 | Flag, do not silently fix; follow 14; keep workers stdlib-only so they can run outside the FastAPI env | n/a | P1, P6 |

### 1.1 Jev-specific failure modes not in the ranking (watch list)

- Jev limited to 3 questions per call per one vendor summary [04] vs 255-option Choice per API docs [01]: verify an 11-option Choice + 2 auxiliary questions works and does not shift latency (F3).
- Coarse probability plateaus (0.78/0.22/0) [01]: do not build thresholds needing fine resolution.
- Probabilities are not comparable across question types [01]: gate only on the Choice confidence.
- 524/529 timeouts are a real outcome: record as timeout, apply the harness fallback, never retry silently inside the timed path [01, 07, 09].
- Early-access pricing "sustainability unproven" [01]: cost is small ($8 per 1k flights at 200 decisions) but could change.

---

## 2. Top 5 bottlenecks (quantitative)

All numbers are measured under shared-machine load (see R15), so CPU figures are lower bounds and latency-variance figures are upper bounds.

### B1. Jev / fast-LLM serial decision chain plus rate limit (dominant wall-clock bottleneck)

| Quantity | Value | Source |
|---|---|---|
| Sequential ceiling per flight | 1/L = 3-5 Hz via OpenRouter; 7.4 Hz direct (jev-drone); pipelined 21 decisions/s but each 120-300 ms stale | [01] |
| Jev rate limit | 20 req/s (docs may allow 40); paid OpenRouter has no platform cap, providers do | [01] |
| Max concurrent Jev flights (Little) | rps x L = 20 x 0.3 = 6 (at p50 0.3 s); ~2.4 at 0.12 s | [I13 from 01, 09] |
| Calls per flight | 20 s x min(10 Hz, 1/L): 200 (L <= 100 ms), 67 (300 ms), 20 (1 s), 7 (3 s) | [I13 from 09] |
| 10,000 Jev flights at 200 calls | 2M calls = 28 h at 20 req/s (14 h at 40) | [01] |
| 1,000 paired scenarios, native latency, 67 calls | ~67k calls = ~1 h at 16-20 req/s | [I13] |
| 1,000 scenarios run through all 7 latency cells with the real model | ~893k calls = 12-15 h | [I13] |
| Cost | $0.000042 per Jev call = $8 per 1k flights at 200 calls; whole campaign < $100-300 | [01, 09] |

Consequence: the wall-clock budget is dominated by the number of *real* Jev calls. Run the latency sweep with an oracle-plus-error mock and use real Jev only at its native (recorded) latency; real-time (b) validation for the fast arm costs ~67 min-2 h per 1k flights [09].

### B2. Real-time validation and latency probing need a quiet machine and 1x wall clock

| Quantity | Value | Source |
|---|---|---|
| Validation set | >= 300 flights x 3 latencies = 900 flights x 20 s = 5 h serial; ~1.2-1.5 h at <= 4 concurrent sim processes [I13] | [08, 09] |
| Fast-arm RT validation of 1k flights | 67 min (rate-bound) to 1-2 h | [09] |
| Clean-machine requirement | load avg < 2 before any timed run; measured load was 45-95 during the research | [09] |
| Jitter floor on an idle loopback loop | +4-5 ms p50, +13-17 ms p99 over a 100 ms server; loop-lag p99 92 ms at 16 in-flight under load | [09] |

Consequence: timed runs serialise with everything else; schedule them in dedicated windows under `caffeinate`, AC power, nothing else running.

### B3. The PID controller inside the sim (dominant CPU cost, but not binding)

| Quantity | Value | Source |
|---|---|---|
| `DSLPIDControl.computeControl` | ~100-125 us per call = ~3/4 of a ~170 us control tick; env step alone ~44 us | [02, 11] |
| Control ticks/s/core with PID at 240/48 Hz | 5.9-7.3k (4.5k with default obstacles); 22-26k without PID | [02, 11] |
| Real-time factor per core | ~120-150x (PID, no obstacles); ~95x with obstacles | [02] |
| Aggregate 8-10 processes | 20-23k ctrl steps/s (scaling flat past 4 P cores; E core ~0.58 of a P core) | [02, 09] |
| Full matrix: 10k scenarios x 7 arms x 7 latency cells x 30 s | 1.47e7 sim-s = ~33 core-h = ~5-6.5 h wall on 20-23k ticks/s aggregate | [I13] |
| N = 2,000 scenario matrix (the power target in 08) | ~1.3 h wall | [I13] |
| Reset with default obstacles | 85-126 ms vs 8 ms for primitives; cameras 10 ms/frame (170x the sim cost if every tick) | [02, 12] |

Consequence: the T1 PyBullet matrix at N = 2,000 is an afternoon, not a reason for a custom tier. A leaner controller or primitive obstacles buy 2-3x if needed.

### B4. RL training (only if Phase 5 is approved)

45-60M env steps = ~10-25 h of the same CPU pool; 5M steps = 0.6-1.2 h with 8 envs [05, inference from measured PPO-only overhead of 893-1,867 steps/s]. It blocks measurement windows (R15).

### B5. Measurement-quality limit on concurrency (correctness bottleneck)

Latency is only valid when the asyncio client is not starved. In 09, throughput collapsed from an ideal 160 req/s at 16 in-flight to 60 req/s with p99 overhead 844 ms. Practical rule: API client process separate from sim workers, <= 6-9 sim workers total, <= 4 during any real-time run, dedicated probe process with no sim. Combined with B1 this caps useful API-arm parallelism at ~6 flights, far below the 9-worker CPU pool.

### Non-bottlenecks (do not optimise)

SQLite writes (capacity ~2,000x need, 660-870k rows/s vs ~280 rows/s) [10]; memory (~9-11 GB of 16 GB with 9 workers, no torch in workers) [09]; ray casts (4 us for 8 rays, 0.18 ms for 360) [02]; trajectory storage (0.14-0.7 GB per 10k flights in Parquet) [10]; dollars (< $300) [09]; DB size (0.25-0.4 GB per 10k flights) [10].

### 2.3 The latency knee and "where extra speed stops helping"

Minimum spawn distance `d_min(L) = d_min0(v) + v x L_eff`, using 06's table (obstacle 0.25 m radius, a = 4 m/s2, tau_a = 0.08 s; consistent with 11b's measured 0.56 m stop at 2 m/s + 0.31 m hit radius) and effective latency `L_eff ~ 1.5 x call latency` for sequential cadence [08]:

| Reactor | L_eff (s) | d_min at 1 m/s (d0 = 0.50) | at 2 m/s (d0 = 0.95) | at 3 m/s (d0 = 1.66) |
|---|---|---|---|---|
| Reflex (PID/FSM, 1-2 ticks at 48 Hz) | 0.02-0.04 | 0.53 m | 1.01 m | 1.75 m |
| Jev direct, p50 ~120 ms | 0.18 | 0.68 | 1.31 | 2.20 |
| Jev via OpenRouter, p50 ~250 ms | 0.375 | 0.88 | 1.70 | 2.79 |
| Jev via OpenRouter, p95 ~600 ms | 0.90 | 1.40 | 2.75 | 4.36 |
| Fast small chat LLM, p50 ~400 ms | 0.60 | 1.10 | 2.15 | 3.46 |
| Mid/slow LLM, p50 ~2 s | 3.0 | 3.50 | 6.95 | 10.66 |

[I13 arithmetic on 01, 06, 08 figures; all latency values are planning bands from 01, to be replaced by the F1 measurement.]

Reading the table:

- At 2 m/s, the spawn-distance window where a reflex survives but Jev-via-OpenRouter p50 does not is **1.01-1.70 m (0.35 s of budget)**. That window is narrow; a Jev-vs-FSM "gap" exists only if scenarios put substantial mass there (R05).
- Going from 300 ms to 100 ms of Jev latency moves `d_min` by ~0.4 m at 2 m/s. If most scenarios spawn at >= 2.2 m at 2 m/s the difference is zero; if most spawn at < 1.0 m nobody survives. Going from 3 s to 0.3 s (LLM -> Jev) moves it by ~5 m and is where the signal lies.
- The p95 tail (0.9 s) matters more than p50 for collisions; report ECCDF and run p95-indexed sweeps [08].
- Below about 50-100 ms (for 08's 6 m/s geometry) extra speed buys nothing; at 2 m/s the floor is `v tau_a + v^2/2a` = ~0.95 m of spawn distance regardless of decision speed [06, 08].
- Sensing range caps the slow arm's feasibility: a 3 s L_eff at 2 m/s needs ~7 m of range; 10 m rays [07] give budget, 4 m Multi-ranger class does not [06].

---

## 3. Contradictions between agents' findings, with resolutions

| # | Topic | Conflict | Resolution |
|---|---|---|---|
| C1 | Simulator throughput and the T0 tier | [03] sizes the tiers on the legacy "~5 kHz physics per env" (85 core-h at 50x); [02] measured 130-170k raw steps/s and ~37k with PID per core (~120x RT at 48 Hz control); [11] measured 5.9k ticks/s with PID; [12] measured ~10k ticks/s no PID; [09] assumed `t_tick` = 1 ms (conservative by 3-6x); [02] no-PID at 240/48 gave 26.5k ticks/s vs [12]'s 10k and [11]'s 22.7k (load-dependent) | Trust [02]/[11] (same order: 5.9-7.3k PID ticks/s/core). The legacy 5 kHz figure is stale. Defer T0; re-measure on an idle machine (spike 02-1). [09]'s flights/hour are conservative |
| C2 | Braking parameters | [06] assumes `a = 4`, `tau_a = 0.08`; [05] placeholder `a_brk = 4` at 3 m/s; [08] uses 5 m/s2, tau 80 ms; [11/11b] measured decel 2.0 / 3.5 / 4.9 m/s2 at 1 / 2 / 3 m/s, velocity time constant 0.15-0.2 s, stop distance 0.25 / 0.56 / 0.88 m, reversal 17-34%; [02] stop 0.22 / 0.54 / 1.28 / 2.38 m at 1 / 2 / 4 / 6 m/s (time to < 5%) | Use measured lookup, not constants; `kappa_a = 0.85`, +0.10 m margin; treat [08]'s 130-150 ms knee as geometry-specific and re-run on the real executor. [06]'s d0 at 2 m/s (0.95 m) agrees with measurement (0.56 + 0.31 = 0.87) within margin |
| C3 | Cruise speed | [05] 3 m/s (max 5); [08] 3-6 m/s; [06] `v_cap` 2, closing 0.5-5; [11] 2 (1-3); [02] 1-2, >= 4 at the edge (73-89 deg tilt) | Cruise 2 m/s; closing speeds > 3 via moving obstacles/intruders; drop 5-6 m/s cases unless the PID is retuned |
| C4 | Control tick and latency resolution | [02] 240/48 Hz; [06] 30 Hz; [08] 50 Hz and asks for dt <= 10 ms; [05] 48 or 30 | 240/48 (20.8 ms); latency points < 21 ms collapse to 1 tick: either drop the 10 ms point or label "<= 1 tick"; 240/240 (4.2 ms) only for a fine-sweep subset (~3-4x CPU) |
| C5 | Job queue and idempotency key | [09]: separate `job` table, key `(experiment, scenario, arm_id incl. config hash, seed, rep, mode)`, leases; [10]: `flights` is the queue, `UNIQUE(scenario_id, arm_id, latency_cell, rep)`, partial index, heartbeat | Pick one in 14. Recommendation: [10]'s flights-as-queue (fewer tables), add `mode` (fast/realtime_validation) and arm config hash into `arms.config_json` hashed into the key, use [09]'s lease + `attempts` cap semantics. Both designs were measured OK (13-18k claims/s; 660k rows/s) |
| C6 | What a "provider" is | [05] arms F/G "identical except provider"; [07] single-token aliases, `max_tokens`, `logit_bias`; [12] decision table with `ttft`/token columns; vs [01] Jev has none of these | Interface confound (R01): shared `Observation` + glossary, per-provider projection; Jev rows have null `ttft`/tokens; bridge arm with Jev-style output |
| C7 | Jev rate limit | [01] 40 req/s in docs vs 20 req/s in blogs; [09] generic 1000 RPM (16.7 req/s) | Plan 16-20 req/s until F6 |
| C8 | Decision cadence and calls/flight | [01] 10 Hz, 200 calls; [09] 67 calls at L = 0.3 s; [07] observations 20-50 Hz; [12] 10 Hz | Cadence is a factor (R28); budget using 67-200 calls/flight |
| C9 | Jev latency used in examples | [06] "Jev-like 10 ms / 50 ms", [08] fast_model 100 ms, [05] `t_dec` placeholders vs measured [01] 110-320 ms | Never label a 10-50 ms provider "Jev"; label mock arms "fast-model stand-in" |
| C10 | Jev advantage vs LLM | [01] arXiv 2609.22753: 10-37% faster median vs DeepSeek flash; [04] quotes 22.7-64.5% lower median latency from the same paper; vendor "40-200x" on classification | Quote the paper's per-table numbers; never a single multiplier; against reasoning/large models 5-20x [01] |
| C11 | Wind severity | [02] "useful 0.05-0.3 N"; [06] cap `a_w <= 0.5 a_brake` = ~2 m/s2 = ~0.054 N (w <= 5.4 m/s) | 0.3 N = 11 m/s2 exceeds brake authority. Use [06]'s cap for certified scenarios; bigger gusts only in an uncertified "stress" class |
| C12 | Hover semantics | [02] "after BRAKE the drone drifts to rest"; [11] first claimed position latch fixes reversal, [11b] refuted (72-108% reversal); [02] says PID integral rejects steady wind | Latched hover rejects wind (0.015 m drift at 10 s); `pos = cur` hover drifts F/0.2 m/s; BRAKE as in R09 |
| C13 | PID class names | [00]/[05] mention `SimplePIDControl`; [02] it no longer exists | `DSLPIDControl` only |
| C14 | pybullet install | [02] plain install fails at `zutil.c` (3m18s), flag build 2m13s; [11] header patch 2m3s; [12] flag build 74 s; [11b] 0.1 s (cache); [03]/[05] unverified; [07] fresh 3.12 venv failed | Consistent: flag works on 3.12 and 3.14.2; a separate 3.12 env buys nothing (no wheel at any version) [02]. Put flag in setup, cache wheel, Phase 0 gate |
| C15 | Python in prototypes | [08] 3.12.12; [06] 3.14.6; [11] 3.12.12; others 3.14.2 | Re-run the key prototypes (08 curves, 06 feasibility) on 3.14.2 |
| C16 | Jev questions per call | [04] up to 3 questions; [01] Choice up to 255 options, plus noul/score | Test 11-option Choice + 2 auxiliary in F3 |
| C17 | `gym-pybullet-drones` facts | Org `utiasDSL` (older sources) vs `learnsyslab`; not on PyPI; only release v1.0.0; "tested macOS 26" true but with flag | Pin SHA `7ebad1e`, `learnsyslab` |
| C18 | Endpoint names | [09] `POST /experiments`; [10] `/experiments/create`, `/list`; [12] `/runs/{run_id}/...` vs `/experiments/{experiment_id}/...` | Reconcile in 14; follow "end with a resource word" rule |
| C19 | Vision cost | [02] 10 ms (default scene) to 2 ms (empty) per 64x48 frame; [07] 3-15 ms; [12] 18.5 ms at 320x240 | Consistent; exclude images from the main experiment |
| C20 | `decisions/sec` definitions | [01] 1/L; [08] four separate metrics; [10] `sim_speed_x` excluding API wait; [12] both | Adopt [08]'s four metrics everywhere |

### 3.1 Corrections to `docs/initial-plan.txt`

| # | Plan statement | Correction | Source |
|---|---|---|---|
| 1 | "current maintained version explicitly lists Apple Silicon/macOS 26 as tested" | True for `main` README, but pybullet has no macOS wheel for any Python; install needs `CFLAGS="-Dfdopen=fdopen"` and a 1-3 min source build; gpd is not on PyPI, only release is v1.0.0 (2021); repo moved to `learnsyslab/`; pin a SHA | [02, 03, 12] |
| 2 | "~20 kHz PyBullet physics updates when parallelizing 80 drones across 4 environments" | From the pre-2023 README only, aggregate, 2020 Intel laptops; the same table gives 0.8-0.95x real time for that case; absent from the current README. Measured 1 drone: 130-170k raw steps/s, ~37k with PID per core; 5.9k PID control ticks/s at 48 Hz | [02, 03, 11] |
| 3 | "Jev gets the state and has to react" | Jev is a typed-decision API (state + questions, choice + probabilities, no text, no sampling params, alpha endpoint); hidden on the normal OpenRouter model list | [01] |
| 4 | "PID/MPC ~ms; Jev short bar; LLM long bar; killer demo" | Jev is 100-330 ms (p50), tens of ms is false; reflex layers win on pure avoidance; effective reaction ~1.5 L; demo bars need measured/modelled labels and a latency-matched ghost | [01, 04, 08, 12] |
| 5 | "velocity control" as a ready feature | `VelocityAviary` caps at 0.25 m/s, holds position only via velocity damping (drifts in wind), no collision flag | [02, 06, 11] |
| 6 | "RGB/depth/segmentation sensors" | Real but 2-60 ms per frame (CPU TinyRenderer), 170x the sim cost if every tick; use `rayTestBatch` (4 us / 8 rays) | [02, 07, 12] |
| 7 | "obstacles" | Built-in obstacles are hard-coded URDFs (samurai, duck, cube, sphere), 85-126 ms reset; create primitives, spawn at event ticks | [02] |
| 8 | "wind changes, battery drops, GPS disappears" | None native (only ground effect/drag/downwash); wind needs per-sub-step `applyExternalForce`; GPS loss is cosmetic without an estimator; most events are stressors, not discrete-reaction events | [02, 06] |
| 9 | "maximize simulation speed x number of decisions x environmental changes" | Wrong objective: API arms are bound by rate limit and latency, not sim speed; decisions/s is at most 1/L per flight | [09, 01] |
| 10 | "thousands of simulated flights" for Jev | 10k Jev flights = 9-28 h and real-time validation 1-2 h per 1k flights; API arms must be sub-sampled to ~500-1000 paired scenarios | [01, 09] |
| 11 | "collect decision_latency, decisions/sec, simulation_speed" as comparable metrics | Four distinct throughput metrics; sim speed excluding API wait vs including it; lockstep hides latency | [08] |
| 12 | "Jev vs PID vs a conventional policy vs a slower model" | Needs reflex-on/off (shield S0/S1), FSM over the same 11 actions, latency-matched arm, oracle and random/hover sanity arms; PID without avoidance is a strawman | [05, 08, 11] |
| 13 | "Jev shouldn't directly replace the low-level controller" | Agreed; note MPC should sit above the PID (point-mass OSQP, 0.11 ms median), not be a competitor on raw reaction | [05] |
| 14 | "Start with Webots? Don't" | Agreed (per-world startup seconds, wrong tool for 10^3-10^4 runs); wind in Webots is custom, not built in | [03] |
| 15 | "Sim can run much faster than real time... suitable for demonstrating Jev's low-latency" | The speed-up is precisely what makes naive latency comparisons meaningless; latency must be converted to sim ticks | [08] |

---

## 4. Phased roadmap (solo developer)

Effort in working days, including debugging; calendar time is ~1.3x because measurement windows must run alone. Go/no-go gates are decisions, not checklists: each has a pre-declared fallback.

### Phase 0 - Spikes and go/no-go gates (4-5 days)

| Item | Days | Notes |
|---|---|---|
| 0.1 Confirm Jev identity with the user; get keys (OpenRouter; TypeSafe direct optional) | 0.25 | R33 |
| 0.2 Install: fresh `uv venv --python 3.14.2`, pybullet with the flag, gpd `7ebad1e` `--no-deps -e`, minimal deps; 1000-tick hover; 200-reset memory check | 0.5 | G1 |
| 0.3 Idle-machine benchmark (spikes 02-1, 09 B1/B2, 09 F6 soak) | 0.5 | replace all contended numbers |
| 0.4 Real latency: run 01's `bench_decision_latency.py` (F1), direct vs OpenRouter (F2), determinism/consistency + 11-option Choice + 2 aux questions (F3), rate-limit ramp (F6); 2-3 times of day | 1.0 | ~$2; G3 |
| 0.5 Jev offline accuracy: 100 scripted scenes with oracle labels, bucketed vs raw-number encodings, mirror test (F5/S2) | 0.75 | G4 |
| 0.6 Calibration + knee prototype on the real stack: brake/lateral step at 1/2/3 m/s, latch rules, `CtrlAviary` + `DSLPIDControl` executor sketch, latency injection N = 0, 2, 5, 14, 24, 48 ticks on "box appears at 1.2-2 m, 2 m/s" | 1.0 | G2 |
| 0.7 Local stand-in latency (Ollama `tev1:0.8b`/`nimble`, MLX letter-logit) (F4) | 0.5 | optional arm |

**Exit criteria / gates**

- **G1 (install)**: hover loop runs on 3.14.2 from a clean env in < 10 min wall. *No-go fallback*: MuJoCo quad (+2-3 days).
- **G2 (knee exists)**: collision rate of an oracle-brake reactor rises monotonically with injected latency within the 0.1-1 s range for spawn distances you can certify as feasible at zero latency; `B` from the calibrated lookup agrees with the boundary within 10%. *No-go*: lower speed/raise spawn distance, or reconsider the speed regime.
- **G3 (Jev usable)**: measured Jev p50 <= 400 ms, p95 <= 1 s, error/timeout rate < 1%, >= 10 req/s sustained. *No-go*: use a local System-One stand-in for the main sweeps and report Jev at injected latency from the recorded distribution only.
- **G4 (Jev competent)**: acceptable-action rate >= 0.7 on clear-threat states (and above the always-HOVER/random baselines with CI). *No-go*: redesign state encoding/questions (1 day); if still < 0.5, reframe the project as "latency-accuracy trade-off" with Jev as a measured weak point.
- **G5 (framing)**: decide the headline is the collision-vs-latency curve + knee, not "Jev beats PID" (R02).

### Phase 1 - Core sim + harness skeleton (7-9 days, including 2 days repo integration)

Deliverables: `CtrlAviary` subclass with seeded primitive obstacles, `ActionExecutor` + shield (S0/S1) + slew/dwell per 11, `DSLPIDControl` adapter with `ctrl.reset()`, scenario JSON + Philox channel RNG + obstacle generator by reaction budget (06), latency injector (c-fixed/c-sampled) with monotonic-apply and ceil quantisation (08), `DecisionProvider` protocol + mock/oracle provider (01), metrics (collision via `getContactPoints`, near-miss, reaction_time, four throughput metrics), sync `sqlite3` store with per-flight transaction, runner CLI with leases, repo wiring per 14 (config, `.gitignore`, drop asyncpg/psycopg2/boto3, aiosqlite, pins).

Exit criteria:
- 500 paired flights with the oracle+mock provider reproduce `P(collision) = CDF_B(L_eff)` within +-3 pp and curves collapse on `rho`.
- Cross-process determinism: same scenario, same actions, identical trajectory hash in two spawned processes.
- `kill -9` of workers and supervisor mid-run, then resume: final flight count exact, zero duplicates, no partial decisions.
- One flight costs <= 0.3 s CPU; `ruff`, `mypy`, `py_compile` clean per the repo checklist.
- Lockstep ablation shows a flat curve (sanity of the artefact story).

Go/no-go: if determinism fails across processes, paired-design claims need a stated tolerance; if the knee from G2 does not reproduce inside the executor, fix before proceeding.

### Phase 2 - Baselines, events, certification (8-10 days)

Deliverables: arms A and A' (A + shield), B1 (TTC brake + lateral nudge), E (FSM, ~60-90 lines per 05), Z0-Z3, Oracle O with labeller (06/11); second event type (intruder, kinematic sphere); wind (OU + mean step), battery rpm scale, sensor noise/dropout and flagged GPS loss in the observation layer; tuning split and three-currency budget log; feasibility certification (Oracle-0 replay + `UNAVOIDABLE` control tier); information-parity table.

Exit criteria: Oracle >= 99% collision-free on certified scenarios; FSM < 2% collisions at 1-tick latency on feasible scenarios; Z1 shows 0 collisions but 0 success (jointly reported); reaction-time and label pipelines validated on hand-built states; events limited to obstacle, intruder, wind/noise as stressors, battery/GPS flag as deadlines. **Stop adding events here (R17).**

Go/no-go: if FSM already matches the oracle within 2 pp at every latency <= 300 ms, Jev cannot show a reaction advantage; the project is then a latency-curve study with Jev as one arm.

### Phase 3 - Providers, latency measurement, prompts (5-7 days)

Deliverables: `JevDecisionProvider` (OpenRouter; direct optional), `ChatDecisionProvider` (fast and slow configs), local stand-in provider, response cache/replay with recorded latency (cache hits excluded from latency stats), budget ledger, dedicated latency-probe process and loop-lag monitor, prompt-format two-stage ablation on a local model then 300 observations via API (07), invalid-output/timeout taxonomy.

Exit criteria: probe distributions (p50/p95/p99, ECCDF) per provider in >= 3 time-of-day blocks; invalid rate < 2% (R31); snapshot id asserted on every call; fixed format chosen with documented sensitivity; cost ledger under cap. Go/no-go on the bridge arm.

### Phase 4 - Main experiment (6-8 days incl. runs)

Deliverables: pre-registered manifest (primary metric = collision rate on the certified set at the chosen headline latency, Holm family, N); cheap arms on >= 2,000 scenarios x latency cells (T1 PyBullet; ~1.3 h wall); API arms on 500-1,000 paired scenarios at native/recorded latency; 2-D latency x accuracy map with the mock; shield S0 and S1; cadence sequential + periodic; real-time (b) validation >= 300 flights x 3 latencies.

Exit criteria: (b) vs (c) per-flight agreement >= 90% and collision rates within Wilson CIs; canary drift flags resolved; interleaved arm ordering; results reproduce from the manifest + latency trace + response cache. Go/no-go: if validation fails, headline claims are restricted to the validated regime and labelled "injected latency".

### Phase 5 - Optional arms (go/no-go each; 2-3 days MPC, 5-7 days RL + 10-25 h compute)

MPC via raw OSQP double integrator (0.11 ms median) only if reviewers need it; RL only if Phase 4 leaves time (pilot first). Tier T0 only if Phase 4 runtime exceeds ~1 day. Multimodal arm: skip unless requested.

### Phase 6 - Analysis and demo (6-8 days)

Deliverables: DuckDB + pandas + matplotlib report (P1-P14 subset: latency ECDF, collision vs latency with Wilson ribbons and budget band, reaction-time, Pareto, per-event grid), manifest, Jinja2 report; single-file HTML replay player from stored trajectories with decision Gantt and avoidance budget band (12); FastAPI summary/progress(SSE)/replay endpoints with pagination (repo conventions); README with install flag and claims ledger.

Exit criteria: every headline number traces to a CSV and manifest hash; every figure states latency mode (measured/sampled/fixed); demo scenario selected by rule (median paired difference), shown beside population statistics.

### Effort summary

| Phase | Days | Cumulative |
|---|---|---|
| 0 | 4-5 | 4-5 |
| 1 | 7-9 | 11-14 |
| 2 | 8-10 | 19-24 |
| 3 | 5-7 | 24-31 |
| 4 | 6-8 | 30-39 |
| 5 (optional) | 0-10 | 30-49 |
| 6 | 6-8 | 36-47 (core) |

Core MVP is ~36-47 working days solo; everything in Phase 5 is cuttable.

---

## 5. Claims ledger

### 5.1 Safe to claim (given the evidence listed)

| Claim | Evidence needed / status |
|---|---|
| "Under injected decision latency L, collision rate of arm X on certified scenario set S is P (Wilson 95% CI); the curve rises between ~a and ~b ms, the knee being set by braking distance" | Phase 1/4; state scenario-set and B-distribution |
| "Jev via OpenRouter measured p50/p95/p99 = ... from this machine/region/time" | F1; until then use 01's bands labelled "published, unmeasured" |
| "A typed-decision API returns a choice plus per-option probabilities in one pass without text generation" | Verified docs [01] |
| "A reflex layer (PID/FSM + shield) reacts in 1-2 control ticks; decision layers cannot match this" | Measured/derived [02, 05, 06] |
| "Collision results flip sign between lockstep and latency-injected accounting for the same policies" | Reproduced in prototype [08]; re-run on gpd |
| "Jev is ~10-37% faster at the median than a flash chat LLM on one independent benchmark, and 5-20x faster than large/reasoning LLMs" | [01, 04] with arm definitions |
| "gym-pybullet-drones runs headless on Python 3.14.2 / M4 at ~5.9k PID control ticks/s per core (~120x real time) after a source build of pybullet" | [02, 11] |
| "To our knowledge as of Sept 2026, no published study combines ... (paired latency sweep, reflex on/off, FSM baseline, certified scenarios) for a Jev-class model" | hedged; [04] |
| "API cost of the full campaign is under ~$100-300; wall clock and rate limits dominate" | [01, 09] |

### 5.2 Conditional claims (only with the listed evidence)

| Claim | Requires |
|---|---|
| "Jev+shield is as safe as FSM+shield" | Holds at S0 and S1, paired test, tuned parity |
| "Jev makes better semantic decisions than a fast LLM" | Format-ablated accuracy on labelled states, same glossary, n >= 500, latency-matched arm |
| "Decision latency matters for event type Y" | Only for obstacle/intruder/battery-critical/GPS-loss; others as stressors |
| "Results hold with real latency, not just injected" | (b) validation >= 90% agreement |
| "Faster decision layer is always safer" | Only between the knee bounds; state the floor and ceiling |

### 5.3 Overclaiming (do not make)

| Claim | Why it fails |
|---|---|
| "Jev out-reacts PID/MPC" | Reflex 21-42 ms vs 110-330 ms p50; reaction ~1.5 L [01, 04, 08] |
| "Jev is N times faster than LLMs" without an arm definition | Fast flash LLM gap is 10-37% at median [01] |
| "Jev avoids obstacles in X ms" from raw API latency | Reaction = issue wait + call + actuation + braking physics [08] |
| "Simulator runs at 20 kHz" / "80 drones" | Stale legacy figure; measured ~5.9k PID ticks/s/core [02] |
| "Jev slow LLM is safest" from a lockstep run | Artifact [08] |
| Any absolute collision rate without the scenario B-distribution | P(collision) = CDF of your chosen budgets [06] |
| "Works on Apple Silicon out of the box" | No macOS pybullet wheel [02] |
| "Reacts to all 7 adversarial events" | Wind/noise/drift/target are stressors [06] |
| "Jev is first/only to ..." | jev-drone, RRARA, FPX exist [04] |
| Sim-to-real or "drone safety" generalisations | 27 g CF2X point in PyBullet, no real hardware |
| Jev quality in general | State design, format and decision cadence are factors; Jev documented weaknesses [01, 07] |
| Wall-clock `sim_speed`/decisions/s including API time as a model property | Four separate metrics [08] |

---

## Decisions recommended

1. Reframe the headline to "collision rate vs effective decision latency, with the knee, for Jev-class (0.1-0.4 s) vs LLM-class (1-3 s) decision layers, against a reflex baseline"; drop "Jev out-reacts PID" entirely (R02).
2. Adopt the MVP scope in Phase 1-4: obstacle + intruder events, arms A', B1, E (FSM), Z0/Z1, Oracle, Jev, fast LLM, slow LLM, optional local stand-in; time model (c) + (b) validation; defer T0 tier, RL, MPC, multimodal, Rerun, GPS estimator until Phase 5.
3. Fix speed regime: CtrlAviary subclass, 2 m/s cruise, 240/48 Hz, calibrated brake lookup, spawn distances ~1-2 m at 2 m/s; reject 0.25 m/s and >= 4 m/s designs.
4. Report everything against reaction budget / `rho = L_eff / B`; publish the B-distribution; stratify by tier.
5. Run the real-Jev sweep only at native (recorded) latency; do latency x accuracy sweeps with an oracle-plus-error mock; sub-sample API arms to 500-1,000 paired scenarios; cap real-Jev concurrency at ~6 flights.
6. Treat measurement hygiene as infrastructure: probe process, loop-lag monitor, separate sim and API processes, idle-machine timed runs, `caffeinate`, interleaved arms, canary.
7. Resolve C5 (queue design), C2-C4 (calibration-first), C11 (wind cap) before Phase 1 code freezes; update the plan text with section 3.1 corrections.
8. Gate the project on G1-G5 in Phase 0; each has a stated fallback.

## Open questions

1. Does the user confirm "Jev" = TypeSafe Jev 1.13, and do they want OpenRouter only or also a TypeSafe direct key?
2. Where is the machine (region/ISP)? Changes all latency numbers by up to 150-250 ms.
3. TypeSafe/OpenRouter terms on benchmark publication; real rate limit (20 vs 40 req/s); is the alpha path stable?
4. What are the *idle-machine* throughput and determinism numbers (every figure in 02, 07, 09, 10 was measured at load 6-95)?
5. Is the intended audience a demo or a publishable comparison? This decides whether Phase 4 validation and pre-registration are mandatory.
6. Should the headline also include a fast-LLM arm (hundreds of ms) or only a slow-reasoning LLM arm (seconds)? Changes the story from "Jev vs LLM" to "Jev vs fast LLM" (small gap) [01].
7. Is `~/Documents` synced by iCloud (WAL hazard)? Which queue design (C5) does 14 adopt?
8. What are real Jev accuracy and determinism for 11-option drone decisions (F3/F5)?
9. Is a moving-target or multi-event scenario required, or is single obstacle/intruder enough for the first result?

## Follow-up spikes (small, runnable experiments)

1. **Gate harness (Phase 0, 1 day):** fresh-venv install + hover + reset-memory check, then the `d_min(L)` table (section 2.3) re-derived from the real executor's brake lookup at 1/2/3 m/s; compare with the estimates above.
2. **Knee + artefact demo (1 day):** on gpd, oracle-brake reactor, injected latency N = 0, 2, 5, 14, 24, 48 ticks, three spawn-distance sets (tight, moderate, relaxed) and log-uniform vs stratified B; show that the aggregate rate changes by tens of pp with the distribution while the `rho` plot is invariant (R04/R05).
3. **Lockstep vs injected vs real-time (0.5 day):** repeat 08's three-arm misranking and (b)-vs-(c) agreement on the real executor with a local sleep-injecting mock server, 300 flights x {100, 300, 1000} ms.
4. **Shield S0/S1/S2 on the knee scenario (0.5 day):** quantify how much of the Jev-vs-LLM gap survives S1 (R21).
5. **Call-volume estimate (0.25 day):** count calls per flight for each arm/latency cell from the knee runs and project API time/cost at 20 req/s; decide which cells use the real model (B1).
6. **Contention test (0.5 day):** 4/6/8 sim workers + 1 API client + local mock server; log loop-lag p99, tick lateness, latency inflation vs idle; set the max worker count for timed runs (B5/R15).
7. **FSM strength test (0.5 day):** implement arm E on the point-mass surrogate, then on gpd; report collision vs injected latency; this number decides whether there is any room for Jev to win on reaction.
8. **Jev offline accuracy (0.5 day, needs key):** F5 with encodings A-F, mirror test, reliability curve (R22).
9. **Literature refresh (0.25 day):** Semantic Scholar sweep for "LLM drone inference latency collision" and review arXiv 2609.30216 before finalising the novelty language (R29).
