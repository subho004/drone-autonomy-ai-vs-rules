# 08 — Benchmark Methodology, Time Accounting, Latency Fairness, Statistics (CORE)

Research agent 08. Date of research: 2026-09-30. Hardware for all measured numbers: Apple M4 (MacBook Pro), macOS (Darwin 27.0.0), Python 3.12.12 in a scratch uv venv (numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, matplotlib). Prototype code lives in the session scratchpad (`.../scratchpad/08/`) and is reproduced in full in Appendix A–C. Nothing in the repo other than this file was touched.

Confidence tags: **[V]** verified from a cited source this session, **[M]** measured by me here, **[I]** inference/engineering judgement.

---

## 0. TL;DR (what the experiment must do to be meaningful)

1. **The time model is the experiment.** A drone-vs-obstacle benchmark that ignores decision latency (option (a), lockstep) measures *decision quality only*. My prototype shows lockstep gives a **flat** collision-vs-latency curve (0.2% at every latency from 0 to 3000 ms) while the same policy, latency-injected, goes 0.2% → 12% (100 ms) → 47% (300 ms) → 99% (1 s) **[M]**. Worse, lockstep **inverts the ranking** of a fast-but-sloppy model vs a slow-but-accurate model (Section 7.4).
2. **Primary design = option (c) latency injection, sim at max speed, with sim-time as the only clock the physics sees**, plus a per-decision *recorded-latency trace* so runs replay deterministically (this is option (d) for reproducibility). **Validation = a small option-(b) real-time-paced set** (e.g. 300 flights × 3–4 latencies), which my prototype shows agrees with (c) to within ±3 pp collision rate / 95–100% per-flight agreement **[M]** (Section 7.6).
3. **Report "effective reaction latency", not API latency.** With one request in flight, expected sense→act delay is ≈ **1.5 × L** (issue-wait + flight + apply), not L; with a fixed-period pipelined loop it is **L + P/2** **[M]**. The "knee" (where collisions start to rise) sits where effective latency reaches the physics-tolerable reaction time: in the prototype p1/p5/p50 of tolerable reaction = **0.20 / 0.24 / 0.64 s** **[M]**.
4. **Separate speed from smartness with a 2-D sweep** (latency × decision accuracy) using a configurable mock `DecisionProvider`, then place real arms (Jev, big LLM, PID reactor, RL) on that map with their *measured* (L, accuracy).
5. **Statistics:** unit = scenario (paired across arms), exact McNemar + Wilcoxon signed-rank + paired bootstrap, Wilson/Newcombe CIs, Holm correction, IQM/performance profiles are *not* needed for flat-binary collision but bootstrap CIs on medians/p95 are. Power table in Section 9: detecting a 10 pp collision difference at 30% baseline needs **356 flights/arm unpaired, ~160–270 paired** (80% power, α=.05).

---

## 1. Literature grounding (what is established vs what we are extrapolating)

All URLs accessed 2026-09-30 via web search (snippets/abstract-level verification; I did not read full PDFs except where noted, so claims beyond the abstract are [I]).

| Source | What it gives us | Relevance |
|---|---|---|
| Ramstedt & Pal, *Real-Time Reinforcement Learning*, NeurIPS 2019 — https://papers.nips.cc/paper/8571-real-time-reinforcement-learning.pdf **[V]** | Argues standard MDPs wait for the agent; in real time the environment keeps evolving while the agent computes. Proposes RTAC (SAC variant). | Motivates why lockstep sim is a *different problem* from the real deployed one. The "real-time MDP" adds the previous action to the state so that action selection and state transition are concurrent. |
| Bouteiller, Ramstedt, Beltrame, Pal, Binas, *Reinforcement Learning with Random Delays*, ICLR 2021 — https://arxiv.org/pdf/2010.02966 **[V]** | Observation delay + action delay in an MDP; random delays handled by state augmentation with an action buffer (DCAC). | Our decision latency = observation delay + compute delay + action delay; jittery API latency = *random delay*. Supports reporting latency as a distribution. |
| Katsikopoulos & Engelbrecht, *MDPs with delays and asynchronous cost collection*, IEEE TAC 2003 — via https://www.semanticscholar.org/paper/Markov-decision-processes-with-delays-and-cost-Katsikopoulos-Engelbrecht/bcbae6f03d0d36371456b90d94d86ffdaed25390 **[V]** | A delayed MDP reduces to a delay-free MDP with augmented state (state + buffer of pending actions); the reduction enlarges state space. | Formal basis for our "state-staleness + action-in-flight" model: the delayed system is an ordinary MDP on (s_t, a_{t-N..t-1}). Optimal delayed policies are not a function of the current state alone. |
| Travnik, Mathewson, Sutton, Pilarski, *Reactive RL in asynchronous environments*, Front. Robot. AI 2018 — https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00079/full **[V]** | In asynchronous (dynamic) environments the env transitions while the agent thinks; reaction time directly drives performance; "act immediately after observing new state." | Our definition of reaction time and the "latched last action" behaviour. |
| *Real-Time Reasoning Agents in Evolving Environments* (Real-Time Reasoning Gym, AgileThinker), ICLR 2026 — https://arxiv.org/pdf/2511.04898 **[V abstract]** | Benchmark for LLM agents in environments that keep evolving; independent control of cognitive load and time pressure; **token count as a hardware-independent temporal measure**; dual-thread reactive + planning agent. | Closest LLM-agent precedent. Validates (i) the two-axis (latency × difficulty) design, (ii) a hardware-independent time unit: we should log tokens AND convert to ticks. |
| *Reflex First, Reflect Later: Latency-Aware Embodied LLM Agents* — https://arxiv.org/pdf/2506.07223 **[V snippet]** | Latency creates a gap between the observed state and the state the action executes in. | Same staleness framing. |
| *Win Fast or Lose Slow: Speed vs Accuracy in Latency-Sensitive Decisions of LLMs* — https://arxiv.org/html/2505.19481v1 **[V snippet]** | Benchmarks (HFT, StreetFighter) where slower/bigger LLMs lose to faster ones under time pressure. | Supports the predicted "rank flip" between lockstep and real-time accounting. |
| Henderson et al., *Deep RL that Matters*, AAAI 2018 — https://arxiv.org/pdf/1709.06560 **[V]** | Seeds, hyperparameters, and implementation details change conclusions; recommends significance testing (t-test, KS, bootstrap % difference CIs). | Seed/scenario hygiene, reporting protocol, tuning-effort parity for baselines. |
| Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021 — https://arxiv.org/pdf/2108.13264 **[V]** | Report interval estimates (stratified bootstrap CIs), IQM, performance profiles rather than point estimates from few runs; `rliable` library. | Our RL arm (training seeds) must follow this; eval flights use bootstrap/exact CIs. |
| AdaStop — https://arxiv.org/pdf/2306.10882 **[V title/abstract only]** | Adaptive sequential testing for comparing agents with limited runs. | Optional for sequentially stopping expensive API arms. |
| Exact McNemar / Connor sample size — summary from https://metricgate.com/docs/sample-size-mcnemar-paired/ **[V secondary]** | Only discordant pairs carry information; effective n = n × p_discordant. | Basis of the paired power table (Section 9), which I computed myself with exact binomial enumeration. |

**Key synthesis [I]:** delay-aware RL says the correct object is a *delayed MDP on an augmented state*. For an experiment (not training), we only need its consequences: (1) action applied at t was chosen from information of age ≥ N ticks; (2) the policy continues executing its previous command meanwhile; (3) random delays should be modelled as a distribution with correlation structure; (4) there is no "fair" latency-free comparison of a 50 ms and a 2 s decision-maker — the delay is part of the system under test.

**Contradiction with initial plan:** `docs/initial-plan.txt` says PyBullet "can run much faster than real time... ~20 kHz... makes it suitable for demonstrating Jev's low-latency decision making." The speed-up is a property of the *simulator*, and is precisely what makes naive lockstep/fast-sim timing meaningless unless decision latency is explicitly converted to simulated ticks. The plan's "killer demo" (latency bars PID/MPC vs Jev vs LLM) is only valid under (b), (c) or (d). The throughput claim itself is owned by agent 02.

---

## 2. Formal time model

### 2.1 Clocks

| Symbol | Clock | Advanced by | Used for |
|---|---|---|---|
| `t_sim` (s), tick `k`, `t_sim = k·Δ` | Simulated physical time | Physics/control stepping only. Δ = control tick (prototype: 20 ms / 50 Hz) | *All physical outcomes* (collision, path length, reaction time, recovery). Every physical metric is a function of `t_sim`. |
| `t_wall` (ns, `time.perf_counter_ns`) | Host monotonic clock | Real time | Throughput, latency *measurements*, sim_speed, cost. Never enters physics in (a),(c),(d). Enters physics in (b) (pacing). |
| `t_dec` | Decision-provider clock | Per call: `L_call` measured on wall clock | Converted to sim ticks by the latency policy. |

### 2.2 Latency decomposition

For decision `i` issued from the observation taken at tick `k_i`:

```
L_eff,i  = L_sense + W_issue,i + L_call,i + L_apply
L_call,i = L_serialize + L_net(DNS/TLS/RTT, queue) + TTFT + n_tok/tps + L_parse
```

- `L_sense`: sensor/estimator delay (perception pipeline, image/ray compute) — 0 in a pure-state sim unless modelled.
- `W_issue`: *waiting for the next decision slot* (sequential: waiting for the previous call to finish; pipelined: waiting for the next period boundary; event-triggered: waiting for the trigger). **This is the term people forget.** It is ~U(0, P) for a pipelined loop with period P and ~U(0, L) for a sequential loop.
- `L_apply`: controller/actuator delay from accepting a setpoint to it becoming acceleration (attitude lag τ; here 80 ms, acceleration saturates at A_MAX). Part of physics, not of the decision clock.
- Injected **effective latency in ticks**: `N_i = ceil(L_call,i / Δ)` (floor is also defensible; ceil is conservative and avoids "free" sub-tick speed-ups). Decision applies at tick `k_i + N_i`. Observation age at application = `N_i` ticks; total reaction `R = (k_apply − k_event)·Δ` includes `W_issue`.

Staleness model: the decision at `k_i` sees `obs(k_i)` only (snapshot), *never* `obs(k_i+j)`. Output applied at `k_i+N_i` replaces the latched setpoint. Out-of-order arrivals (pipelined, jitter) are **discarded if their issue index is older than the currently applied one** (monotonic-apply rule; implemented in the prototype with `buf_iss`).

What the low-level controller does meanwhile: **continues executing the last latched high-level command** (velocity setpoint from the last FORWARD/BRAKE etc.) with its own 48–240 Hz stabilisation loop, optionally guarded by a *local safety shield* whose own latency is a separate, small, configurable arm parameter (see 11's action mapping; agent 11 owns semantics). A TTL on latched commands is an experimental factor (TTL=∞ vs TTL=2 s→HOVER).

### 2.3 Delayed-MDP equivalence [I]

State for the policy: `x̃_k = (s_k, u_{k−N}, …, u_{k−1})`. Our injected-latency harness realises exactly this delayed system. The real-time-paced harness (b) realises it with the *actual* OS/network as the delay process. The lockstep harness realises the delay-free MDP.

---

## 3. Options (a)–(d): pseudo-code main loops

Shared: `env.step(setpoint)` advances one control tick Δ of physics; `provider.decide(obs) -> (action, meta)` is the (async) call.

### (a) Lockstep / pause-the-world

```python
k = 0
while not done:
    obs = env.observe(k)
    t0 = perf_counter_ns()
    action, meta = provider.decide(obs)  # blocks; sim time frozen
    wall_call = perf_counter_ns() - t0  # recorded, but never enters physics
    setpoint = action_to_setpoint(action, obs)
    for _ in range(ticks_per_decision):  # e.g. 5 ticks
        env.step(setpoint)
        k += 1
```
Latency is invisible to physics. Valid *only* for decision-quality metrics. Collision_rate is independent of provider speed by construction.

### (b) Real-time pacing, drone continues on last action

```python
next_deadline = perf_counter()
latched = HOVER
inflight = None
while not done:
    if inflight is None or inflight.done():
        if inflight:
            latched = inflight.result()  # arrives whenever it arrives
        inflight = loop.create_task(provider.decide(env.observe(k)))  # snapshot NOW
    env.step(action_to_setpoint(latched, obs_now))  # physics at wall-clock rate
    k += 1
    next_deadline += Δ
    await sleep_until(next_deadline)  # sim_time == wall_time
    lateness.append(
        perf_counter() - next_deadline
    )  # record jitter; if too big, flight is tainted
```
Faithful (includes real network, jitter, scheduler), but sim_speed = 1×. 10 000 flights × 9 s = 25 h per arm even single-stream (parallelisable across processes with care for timing contamination).

### (c) Latency injection (recommended primary)

```python
pending = PriorityQueue()  # (arrive_tick, issue_tick, action)
applied_issue = -1
latched = INITIAL
for k in range(T):
    # 1. issue (cadence: sequential | fixed period | event-triggered)
    if cadence.should_issue(k, pending):
        obs = env.observe(k)  # snapshot at issue tick
        action, meta = provider.decide_sim(
            obs, k
        )  # real call OR mock OR replay (see below)
        N = latency_model.ticks(meta, k)  # from measured L_call, sampled dist, or const
        pending.push((k + N, k, action))
        trace.log(k, N, meta)  # recorded latency makes it replayable
    # 2. deliver everything due (monotonic-apply rule)
    while pending and pending.peek().arrive <= k:
        arrive, issue, a = pending.pop()
        if issue > applied_issue:
            latched, applied_issue = a, issue
    # 3. physics at full speed, on the latched command
    env.step(action_to_setpoint(latched, env.state))
```
Variants for where `N` comes from:
- **c-measured**: real call is made synchronously at issue (*sim waits in wall time*, pause-and-charge), `N = ceil(L_measured/Δ)`. Valid for sequential cadence only (the sim cannot advance past `k+1` before knowing `N`, else it would have to roll back). Wall cost = Σ real calls; no speed-up for API arms, but physics time is correct.
- **c-sampled**: `N` drawn from a fitted/empirical latency distribution (provider-specific), call result from cache/mock/replay. Sim at full speed. Supports pipelining.
- **c-fixed**: `N` constant (the latency sweep).

### (d) Hybrid: record → replay + small real-time validation

```
Phase 1 (record, real-time-ish, small n): run arm live under (b) or c-measured, log per decision
        (issue_tick, obs_hash, prompt_hash, raw_response, action, L_ms split into net/TTFT/gen, provider, model, headers)
Phase 2 (bulk, fast): run c-sampled with N ~ empirical CDF of Phase-1 L_ms (bootstrap resample,
        optionally preserving autocorrelation via block resampling), action from live-call cache keyed on (obs_hash,prompt_hash) or from mock
Phase 3 (validate): rerun ~300 flights under (b); accept if collision rate (c) vs (b) agrees within CI and per-flight agreement >= 90%
```

### Comparison

| | (a) lockstep | (b) real-time | (c) injection | (d) hybrid |
|---|---|---|---|---|
| Latency visible to physics | No | Yes (real) | Yes (modelled/measured) | Yes (sampled from real) |
| Sim speed | max | 1× | max (c-sampled) / API-bound (c-measured) | max + small 1× set |
| Reproducible under non-deterministic API | Yes | **No** | Yes if latency trace recorded | Yes |
| Captures real jitter, GC, scheduler | No | Yes | No (only if sampled from real distribution) | Partly (+ validation set) |
| Captures latency *correlation/bursts* | No | Yes | Only with block-resampling | Partly |
| Thousands of flights feasible | Yes | No | Yes | Yes |
| Risk of "artifact" claims | **Very high** | Low | Medium (model assumptions) | Low–medium |
| Verdict | decision-quality-only ablation | validation set | **primary** | **primary + validation** |

---

## 4. Recommended design

**Primary: (c) with (d)'s recorded-latency discipline. Validation: (b).**

1. **Sim runs at max speed on `t_sim` only.** `sim_speed` and `decisions/sec` are reported in separate, explicitly-labelled variants (Section 8), never mixed.
2. **Two latency regimes per arm**:
   - *Synthetic sweep* (c-fixed) for the collision-vs-latency curve: L ∈ {0, 10, 50, 100, 300, 1000, 3000} ms (add 20, 200, 500, 700, 1500, 2000 where the curve bends; Section 6).
   - *Empirical* (c-sampled) per provider using measured distribution (p50/p95/p99 + tail), with the same provider's accuracy from its own outputs.
3. **Decision loop (async)**: sequential cadence as baseline ("issue as soon as the last finished", what a simple API client does) AND pipelined cadence with period P (e.g. 100 ms, bounded concurrency). Report both; effective reaction latency differs by ≈L/2 at L=300 ms in the prototype. Pipelining is an architectural feature of the arm under test, not a free lunch: it costs calls/dollars (decisions/flight 90 fixed vs 30 sequential at 300 ms) **[M]**.
4. **Latency is always logged per decision** (`issue_tick, arrive_tick, N, L_ms components`). A run is reproducible iff `(scenario_id, arm_config_hash, latency_trace_hash, response_cache)` are fixed. Re-running with *new* live latencies is a new replicate, not a repro.
5. **Live-API arms**: c-measured for a small set (pause-and-charge, sequential) gives the real mean/tail in sim time without waiting real-time; c-sampled then scales to thousands using those distributions.
6. **Validation set (b)**: ≥300 flights × {low, mid, high} latency under real-time pacing with loop-lateness logged; acceptance: |ΔCR| within Wilson CI overlap and ≥90% per-flight outcome agreement, tick lateness p99 < 0.25Δ (see 7.6 for what I measured).
7. **Always include a lockstep (a) run as an explicit ablation arm** ("decision quality with latency removed"), labelled as such — the difference between (a) and (c) *is* the latency cost, a useful number instead of a confound.
8. **Classical arms** (PID reactor, MPC, FSM, RL) get their *measured compute time* per decision (e.g. MPC solve p50/p95) converted via the same `ceil(L/Δ)` rule; a 2 ms reactor is 1 tick, not 0. No arm gets zero latency unless it is the lockstep ablation.

---

## 5. Decision loop details

### 5.1 Cadence policies
- `sequential`: next request issued when previous response arrives (W_issue ≈ U(0, L)); simplest; decisions/s = 1/L.
- `periodic(P)`: every P ticks regardless; W_issue ≈ U(0,P); multiple in flight; requires monotonic-apply.
- `event_triggered`: issue on trigger (e.g. obstacle range below threshold, anomaly); W_issue ≈ trigger latency; saves calls; must include a periodic heartbeat to avoid missing events it cannot sense.
- `dual-system` (fast reactor + slow planner, à la AgileThinker): fast thread uses latest planner output + local obs; planner cadence sequential. Out of scope for the core sweep, add as an arm.

### 5.2 Timeouts and invalid output
Request timeout `T_to` (e.g. 3× p99 of that provider; in sim = `ceil(T_to/Δ)` ticks): on timeout or unparseable output → apply `fallback_action` (HOVER or BRAKE; configured, identical across arms) at tick `issue + N_to` and count `timeout`/`invalid` flags. The stale-result discard rule handles late arrivals after timeout.

### 5.3 Quantisation
The injected delay resolution is Δ. In the prototype 10 ms and 20 ms both become 1 tick (identical results; confirmed in table). If the 10 ms point must be distinct, run the control tick at 100 Hz (Δ = 10 ms) or implement sub-tick interpolation. For the sweep I recommend Δ ≤ 10 ms for the setpoint loop while keeping physics at its native rate (agent 02 to confirm gym-pybullet-drones step cost). **[I]**

---

## 6. Latency-sweep design

- **Independent variable**: injected `L_call` (median) ∈ {0, 10, 50, 100, 300, 1000, 3000} ms (brief) + densification (20, 200, 500, 700, 1500, 2000 ms) around the knee.
- **Factors crossed**: cadence {sequential, periodic 100 ms}; jitter σ {0, 0.6 lognormal}; provider accuracy p_miss {0, 0.01, 0.05, 0.10, 0.20} using the mock provider driven by the oracle labeller (agent 11) with controlled corruption.
- **Fixed**: scenario set, seeds, wind/noise draws (common random numbers), obstacle feasibility bound.
- **Outputs**: (i) collision_rate(L) with Wilson CIs; (ii) effective reaction-time distribution vs L; (iii) 2-D heat map CR(L, p_miss) with iso-collision contours; (iv) real arms placed at their measured (L, p_miss).
- **Reading the map**: moving *left* = faster model; moving *down* = smarter model. If two arms lie on the same iso-contour, they are equivalent for collision; speed-vs-smartness tradeoff rate = slope of the contour. **The "knee" is where the contour turns vertical: extra accuracy cannot compensate for latency beyond the physical budget.** Below the knee, extra *speed* stops helping (contours horizontal) — the ceiling of the speed advantage.
- **Feasibility guarantee** (owner 06): every scenario is constructed so a perfect zero-latency reactor can avoid it (the prototype uses `d_app ≥ 1.1·v²/2a + 0.15v + r + 0.2`). Anything not feasible is excluded a priori, not post hoc.

---

## 7. MEASURED prototype: latency-injected braking-limited reactor

### 7.1 What was built
Vectorised (numpy) 3-D point mass with quadrotor-like limits: velocity command → acceleration `a_cmd = clip(Kp(v_cmd − v), ±5 m/s²)` → first-order attitude lag τ = 80 ms → Euler integration at Δ = 20 ms; OU-free Gaussian wind accel (σ = 0.3 m/s²). Drone cruises along +x at v ∈ U(3,6) m/s. At `t_app ∈ U(1,3) s` a spherical obstacle (R_tot = 0.65 m incl. drone radius, lateral offset ±0.3 m, i.e. in the path) becomes *known* at distance `d_app` = feasible minimum + U(0,4) m. Reactor policy: FORWARD until it *sees* the obstacle in a snapshot, then BRAKE (v_cmd = 0) — a minimal "braking-limited reactor". Decisions may be wrong with probability `p_miss` (perception dropout / model error → FORWARD). N = 4000 paired flights (common random numbers across every arm/latency; seed 20260930). Runtime: whole sweep ≈ 65 s on the M4 (critical-delay search ≈ 37 s).

Accounting modes implemented: `lockstep_*` (latency ticks forced to 0), `seq` (one in flight), `pipe` (period 5 ticks = 100 ms, many in flight, monotonic apply).

### 7.2 Collision rate vs injected latency (N=4000 flights, p_miss = 0.05 unless noted)

| L (ms) | (a) lockstep | (c) seq, σ=0 | (c) seq, σ=0.6 | (c) pipe P=100 ms, σ=0 | (c) seq, perfect perception (p_miss=0) | reaction p50/p95 s (seq σ=0) |
|---|---|---|---|---|---|---|
| 0 | 0.2% | 0.2% | 0.2% | 4.7% | 0.0% | 0.00 / 0.02 |
| 10 | 0.2% | 0.4% | 0.9% | 6.1% | 0.0% | 0.02 / 0.04 |
| 20 | 0.2% | 0.4% | 2.6% | 6.1% | 0.0% | 0.02 / 0.04 |
| 50 | 0.2% | 5.5% | 8.4% | 8.5% | 0.0% | 0.08 / 0.12 |
| 100 | 0.2% | 12.0% | 20.9% | 12.0% | 0.0% | 0.14 / 0.20 |
| 200 | 0.2% | 29.4% | 44.2% | 22.2% | 11.9% | 0.30 / 0.40 |
| 300 | 0.2% | 47.3% | 61.3% | 33.5% | 29.7% | 0.44 / 0.60 |
| 500 | 0.2% | 76.4% | 81.2% | 55.4% | 62.3% | 0.76 / 1.04 |
| 700 | 0.2% | 90.3% | 89.8% | 75.4% | 85.0% | 1.04 / 1.42 |
| 1000 | 0.2% | 98.7% | 95.7% | 95.2% | 98.2% | 1.50 / 2.08 |
| 1500 | 0.2% | 100% | 99.1% | 100% | 100% | 2.04 / 3.10 |
| 2000 | 0.2% | 100% | 99.5% | 100% | 100% | 3.06 / 3.98 |
| 3000 | 0.2% | 100% | 99.95% | 100% | 100% | 4.00 / 4.90 |

Wilson 95% CIs are in `curves.json` (half-widths ≤ ±1.6 pp, ≤ ±0.9 pp at extremes). Near-miss (clearance < 0.5 m but no contact) peaks at ~13% around 200–300 ms and vanishes at ≥1.5 s where everything is a collision.

Physics-only tolerable reaction (smallest brake-onset delay after obstacle appearance that still collides, perfect perception): quantiles p1 = 0.20 s, p5 = 0.24 s, p25 = 0.42 s, p50 = 0.64 s, p75 = 0.86 s.

### 7.3 Findings
1. **Lockstep is flat (0.2% everywhere)** — it can only show the decision-quality floor; the latency cost is invisible. This is the "time-model artifact" the brief warns about.
2. **Knee.** With perfect perception and sequential cadence, collisions are exactly 0 up to 100 ms and 12% at 200 ms: effective reaction ≈ 1.5 L (reaction p95 0.20 s at L = 100 ms) reaches the p1 tolerable reaction (0.20 s) around L ≈ 130–150 ms. The curve then rises steeply and saturates ~1 s, where effective delay exceeds almost all scenarios' physical budget. **Consequence: below ~50–100 ms (for this geometry: 6 m/s, 5 m/s² brake, ~4–9 m sensing slack) a faster model buys nothing; between 100 ms and 1 s every 100 ms costs ~15–25 pp collision; above ~1.5 s nothing saves you except anticipatory policies.** The numbers are geometry-specific; the *shape* (flat floor → steep rise → saturation) is general.
3. **Cadence matters as much as call latency.** Pipelining (P = 100 ms) at L = 300 ms cuts collisions 47% → 33%, at 3× the decisions/flight (90 vs 30). At L = 0 the pipelined loop still has 4.7% collisions because a miss is only repaired one period later: sample rate is part of "latency". Conversely, sequential at L ≤ 20 ms looks better (decides every tick).
4. **Jitter helps at high medians, hurts at low medians** (σ=0.6 lognormal with fixed *median*): 20 ms → 2.6% vs 0.4%; but 1000 ms → 95.7% vs 98.7%, because lucky fast draws rescue some flights. **Always compare tails and means of effective latency, not medians only;** define which statistic of the latency distribution the sweep is indexed by.
5. **A miss costs a full round trip** in sequential mode: p_miss 5% with L = 300 ms produces 47% vs 30% at perfect perception — accuracy and latency interact multiplicatively (why the 2-D map is needed).

### 7.4 Lockstep misranks arms (speed vs smartness)

Three stand-in arms, same 4000 flights, latency lognormal σ=0.4 around the median shown:

| Arm | median L | p_miss | CR lockstep (a) | CR injected (c, seq) |
|---|---|---|---|---|
| reactive_local (PID/TTC-like) | 2 ms (1 tick) | 0.05 | 0.20% | 0.43% |
| fast_model (Jev-like stand-in) | 100 ms | 0.08 | 4.88% | 25.9% |
| slow_llm | 1500 ms | 0.005 | **0.00%** | **99.4%** |

Paired exact McNemar, fast vs slow: lockstep → 195 flights where only fast collided, 0 where only slow did (p ≈ 4e-59, slow "wins"); injected → 0 vs 2937 (p ≈ 0, fast "wins"). **Same policies, same flights, the verdict flips purely from time accounting.** A benchmark report that says "slow LLM is safest" under lockstep is an artifact.

### 7.5 Reaction time is not call latency
Sequential cadence, σ=0: measured reaction (obstacle-known → brake-command-active) p50 = 0.14 s for L = 100 ms, 0.44 s for L = 300 ms, 1.50 s for L = 1 s → **R ≈ 1.45–1.5 L**. Report `reaction_time` (closed-loop) next to `decision_latency` (call), and let the gap be `W_issue`.

### 7.6 Validation of (c) against a true real-time-paced (b) run
Script in Appendix B: 300 flights as concurrent asyncio tasks, one 50 Hz tick per 20 ms `perf_counter` deadline (sim time == wall time), each flight's request is a real `asyncio.sleep(L)` (sequential cadence, perfect perception). Scenarios: obstacle appears at 0.4–0.9 s, horizon 4 s. Same scenarios run under (c).

| L (ms) | CR real-time (b) | CR injected (c) | per-flight agreement | tick lateness p50 / p99 / max (ms) |
|---|---|---|---|---|
| 100 | 0.0% | 0.0% | 100% | 1.10 / 51.2* / 82.3* |
| 200 | 10.7% | 10.3% | 95.0% | 1.10 / 2.4 / 6.1 |
| 300 | 26.7% | 27.0% | 96.3% | 1.10 / 1.75 / 2.4 |
| 500 | 56.7% | 59.0% | 95.0% | 1.10 / 1.62 / 1.72 |

\*first run includes task-creation/warm-up spike (300 tasks) — this is exactly why warm-up must be excluded (Section 10). Disagreements of 4–5% of flights are borderline cases (quantisation: real arrival mid-tick is applied on the *next* tick, i.e. (b) carries an extra ≤1 tick of delay; plus real sleep overshoot ~1.1 ms). Rates agree within ±2.3 pp (n=300, SE ≈ 2.6 pp). This supports (c) as a faithful proxy **for this latency mechanism**; it does not validate network behaviour (no network here) — that needs the live (b) set with the real provider.

### 7.7 What the prototype does NOT show (honest limits)
- Point mass, not PyBullet; real quadrotor attitude dynamics add coupling (thrust tilt vs altitude, yaw) → tolerable braking A_MAX is state-dependent. Re-run on gym-pybullet-drones once 02's install is proven.
- Reactor is 1-action BRAKE; no lateral dodge, no anticipatory behaviour. An AVOID_LEFT policy has a *different* latency knee (lateral accel vs longitudinal).
- One event type. Wind, moving target, GPS loss have different time budgets.
- Latency model is i.i.d. lognormal; real APIs have bursts/autocorrelation/timeouts.
- Numbers are illustrative of method, not predictions for Jev.

---

## 8. Metric definitions (exact)

Notation: flight = one (scenario_id, arm, replicate) rollout; Δ = control tick; `k_ev` = tick at which the *first adversarial event relevant to the flight becomes observable* (ground truth, from scenario JSON); `k_end` = terminal tick; all times in **sim seconds** unless suffixed `_wall`.

**Latency family**
- `decision_latency_ms` (per call): `t_response_complete − t_request_start` on `perf_counter_ns` (wall), measured inside the provider wrapper (includes serialise + network + queue + generation + parse). Split fields, from client instrumentation and response metadata (agent 01): `net_ms` (request start → first byte sent/ACK-ish proxy; with streaming, `ttft_ms − server_compute_hint`), `ttft_ms` (request start → first output token/first SSE chunk), `gen_ms` (first token → last token), `parse_ms`, `n_in_tok`, `n_out_tok`. Where true network/server split is unobservable, report `ttft_ms` and `gen_ms` only; do not invent a network number. Report p50/p90/p95/p99/max + mean, with bootstrap CI for percentiles; never only the mean.
- `effective_latency_ticks` (per decision): `N = applied_tick − issue_tick` (includes timeout fallback). `effective_latency_s = N·Δ`.
- `issue_wait_s`: `issue_tick − k_obs_available`; 0 for event-triggered at the instant of trigger.
- `reaction_time_s` (per flight, per event): `(k_first_correct_response − k_ev)·Δ`, where "correct response" = first tick at which the *applied* command belongs to the scenario's oracle-correct action set for that phase (oracle labeller, agent 11/06; e.g. BRAKE/AVOID_* for an obstacle), and stays correct ≥ 0.5 s (debounce). NaN (censored at `k_end`) if never; report censoring fraction and use Kaplan–Meier/ median with censoring, not silent dropping. In the prototype: first tick with BRAKE active after `t_app`.

**Safety/outcome**
- `collision` (bool): at any tick after takeoff, signed clearance `min_j (‖p − c_j‖ − r_j − r_drone) ≤ 0` for any obstacle j, other drone, ground contact with |vertical speed| > 0.5 m/s outside an EMERGENCY_LAND/landing phase, or workspace exit. Contact with the ground during commanded landing is *not* a collision. Evaluated at physics rate (not control rate) to avoid tunnelling at speed.
- `collision_rate` = collisions / flights on the feasible scenario set (Wilson CI; paired comparisons by exact McNemar).
- `near_miss` (bool, no collision): `min clearance < d_nm` (default 0.5 m, tied to a speed-scaled 0.5 s time-to-contact alternative as a sensitivity check).
- `success` (bool): reached target region (radius r_goal) **and** no collision **and** `t_sim ≤ T_max` **and** no EMERGENCY_LAND unless the scenario demands it. `timeout_flight` = neither success nor collision by T_max.
- `recovery_time_s`: after each event, time from `k_ev` until the drone returns to "nominal" = (clearance > d_safe and ‖v − v_ref‖ < ε and heading error < θ and back within corridor width w of the nominal path) continuously for 1 s; censored if collision occurs first.

**Efficiency**
- `time_to_target_s`: `t_sim` at first entry into goal region (successful flights only; report with success rate — never compare conditional-on-success times without it; use also `time_to_target_penalised` = T_max for failures, for Wilcoxon).
- `path_length_m`: Σ ‖p_{k+1} − p_k‖ from ground-truth positions at control rate; report also `path_efficiency = straight_line / path_length`.
- `energy_proxy_J`: Σ over ticks of Σ_i (k_T·ω_i³)… in gym-pybullet-drones, use `Σ_k Σ_i rpm_i³·Δ` normalised per scenario; state explicitly it is a proxy. (Battery events change available thrust, not the proxy definition.)
- `oscillation_index`: number of sign changes of commanded-action *class* per 10 s (e.g. LEFT↔RIGHT, FORWARD↔BRAKE), plus mean |jerk| of the velocity setpoint. Detects thrashing/hysteresis-free policies.
- `invalid_output_rate` = invalid or unparseable outputs / decisions; `timeout_rate` = timeouts / decisions; `fallback_activation_rate`; `decision_correctness` = fraction of decisions equal to an oracle-acceptable action (for the accuracy axis of the 2-D map).

**Throughput (must be reported separately, never a single "speed")**
- `decisions_per_sim_second` = n_decisions / t_sim (property of cadence design).
- `decisions_per_wall_second_total` = n_decisions / wall (includes API waits; fleet-level number).
- `decisions_per_wall_second_compute` = n_decisions / (wall − API wait time) — provider-independent harness overhead.
- `sim_speed_compute` = t_sim / wall_compute (physics + harness, **excluding** awaited API time) — the number comparable across arms and to the "20 kHz" claim.
- `sim_speed_total` = t_sim / wall_total (includes API waits); reported only for live c-measured/(b) runs.
- `ticks_per_second` (physics steps/s) for comparison with agent 02's micro-benchmark.

**Flight-level aggregation:** per-scenario summary over replicates first (mean for binary → fraction, median for times), then across scenarios. The scenario is the experimental unit (Section 9.1).

---

## 9. Experimental design & statistics

### 9.1 Design
- **Paired/blocked design.** A scenario = (geometry, event schedule with times/severities, wind/noise realisations, initial conditions), fully serialised (owner 06). Every arm × latency setting is run on **the same scenarios** with the same noise tensors (common random numbers, pre-drawn per tick as in the prototype). This is what makes the paired tests valid and cuts required n by 1.3–2.4× (power table).
- **Unit of analysis = scenario.** PyBullet is deterministic for a fixed process/inputs, so replicating a deterministic arm on the same scenario adds no information; replicates exist only for stochastic arms (LLM sampling, latency draws): K replicates per scenario, averaged first (cluster by scenario) — otherwise pseudo-replication inflates significance.
- **Scenario split**: tuning scenarios (for baseline tuning, prompt iteration, RL training) vs locked test scenarios with different seeds; test set touched once per arm version; pre-registered primary metric (collision_rate on feasible scenarios at the headline latency) and hypotheses in the experiment config before running (Henderson et al. practice).
- **Randomised, interleaved execution order** of arms across scenarios (for live-API arms, blocked by time) to decorrelate provider drift from arm identity.
- **Stratify scenarios** by event type and severity; report per-stratum results and avoid Simpson-type averaging.
- **Warm-up exclusion**: discard first W=20 decisions per process/connection (TLS, DNS, JIT, caches) from *latency stats*; run 3 throwaway flights per worker for physics warm-up. Record but flag them.

### 9.2 Tests and intervals
| Quantity | Interval | Test (paired) | Notes |
|---|---|---|---|
| collision_rate, success, near_miss (binary) | Wilson score per arm; Newcombe paired-difference CI (hybrid score) or bootstrap CI over scenarios | **Exact McNemar** (b vs c discordant counts); mid-p variant as sensitivity | Only discordant pairs matter |
| time_to_target, path_length, energy (continuous, skewed, censored by failure) | BCa/percentile bootstrap over scenarios (10 000 resamples) for median / mean diff | **Wilcoxon signed-rank** on paired differences (on successes-in-both; also penalised version) | Report conditional + unconditional |
| recovery_time, reaction_time (censored) | Kaplan–Meier + bootstrap | Paired permutation on restricted mean time, or log-rank stratified | Do not drop censored flights |
| latency (heavy-tailed) | Bootstrap CI on p50/p95/p99; report CDF/ECCDF on log axis | Compare via KS/Anderson–Darling descriptively; **do not t-test** | Report max and count of timeouts |
| multi-arm summary | Stratified bootstrap CIs on IQM/mean across event-type strata (Agarwal et al.), performance profiles | — | Mostly for RL-arm training seeds |
| dose-response (latency) | Logistic/GLMM: `collision ~ log(L) + arm + (1|scenario)` ; knee via changepoint/piecewise fit with bootstrap CI | Likelihood-ratio | |

**Multiple comparisons:** pre-declare families. Primary family (headline latency, each arm vs designated reference, k−1 comparisons): **Holm–Bonferroni** (uniformly more powerful than Bonferroni; Benjamini–Hochberg FDR for exploratory per-event-type tables). Report raw and adjusted p, and effect sizes (difference in pp with CI) first, p-values second.

**Early stopping for expensive arms:** fixed-N by default; if sequential looks are needed use a group-sequential boundary (e.g. O'Brien–Fleming) or AdaStop (https://arxiv.org/pdf/2306.10882), never "run until p<0.05".

### 9.3 Power analysis (computed here with scipy/statsmodels)
Unpaired: `statsmodels NormalIndPower` on Cohen's h (arcsine), two-sided. Paired: **exact** McNemar power by enumerating discordant counts m~Bin(n, p_d) and the exact two-sided binomial rejection region; smallest n found by bisection. Discordance derived from the marginals and the phi-correlation ρ between arms' collision outcomes on the same scenarios (prototype measured ρ ≈ 0.55–0.58 between adjacent latencies: L=100 vs 200 ms ρ = 0.55; 200 vs 300 ms ρ = 0.58 **[M]**). "infeasible" = that ρ is not attainable given the two marginals (max phi is bounded when marginals differ a lot).

Flights **per arm** (n paired scenarios = n flights per arm), α = 0.05 two-sided, power 80% unless stated:

| Baseline A vs B collision | Δ | Unpaired n/arm (80%) | Unpaired (90%) | Unpaired, Bonferroni α=.05/6 (80%) | Paired ρ=0.3 (n; p_disc) | Paired ρ=0.6 | Paired ρ=0.8 |
|---|---|---|---|---|---|---|---|
| 5% vs 10% | 5 pp | 424 | 568 | 654 | 339 (0.101) | 197 (0.062) | not attainable |
| 5% vs 15% | 10 pp | 133 | 178 | 205 | 112 (0.138) | n/a | n/a |
| 5% vs 25% | 20 pp | 45 | 60 | 69 | 42 (0.218) | n/a | n/a |
| 10% vs 15% | 5 pp | 681 | 911 | 1050 | 517 (0.156) | 308 (0.091) | n/a |
| 10% vs 20% | 10 pp | 195 | 261 | 301 | 158 (0.188) | 91 (0.116) | n/a |
| 10% vs 30% | 20 pp | 60 | 79 | 92 | 52 (0.258) | n/a | n/a |
| 30% vs 35% | 5 pp | 1376 | 1842 | 2123 | 1003 (0.309) | 587 (0.178) | 304 (0.090) |
| 30% vs 40% | 10 pp | 356 | 476 | 549 | 270 (0.325) | 161 (0.191) | 79 (0.101) |
| 30% vs 50% | 20 pp | 93 | 125 | 144 | 76 (0.363) | 44 (0.225) | n/a |
| 50% vs 55% | 5 pp | 1565 | 2095 | 2414 | 1135 (0.351) | 662 (0.202) | 343 (0.102) |
| 50% vs 60% | 10 pp | 388 | 519 | 598 | 291 (0.353) | 173 (0.206) | 84 (0.108) |
| 50% vs 70% | 20 pp | 93 | 125 | 144 | 76 (0.363) | 44 (0.225) | n/a |

Continuous metrics (paired t-test power, Wilcoxon ≈ n/0.955, Pitman ARE for normal data): standardised paired effect d_z = 0.1 → 787 (Wilcoxon ≈ 824); 0.2 → 199 (208); 0.3 → 90 (94); 0.5 → 34 (35).

Practical reading: **~2,000 scenarios per arm** comfortably detects 5 pp differences at 30–50% baselines unpaired and ~10 pp at any baseline with large margin; per-stratum analyses (7 event types × severity) divide this by the strata count — plan n per stratum, not n total. For small effects in the low-collision regime (1–2% vs 3–4%), the required n explodes (paired, ρ≈0.6: thousands); use one-sided non-inferiority margins or the latency sweep (where effects are tens of pp) rather than chasing them. **Budget rule**: live-API arms (expensive) target the ≥ 10 pp headline effect with paired n ≈ 300–500; cheap local arms run the ≥ 2,000 set.

### 9.4 Seeds
Record `base_seed` and derive every stochastic stream from `SeedSequence(base_seed).spawn(...)` keyed on (scenario_id, stream name) so adding an arm never shifts another arm's draws. PyBullet cross-process determinism is agent 02/09's check; state the tolerance rather than assume bit-exactness.

---

## 10. Measurement rigour

- **Timers**: `time.perf_counter_ns()` only (monotonic, on macOS backed by `mach_absolute_time`; resolution is ns-scale [I]). Never `time.time()` for intervals. Take request-start *immediately before* the `await client.post` and end *when the last byte/chunk is parsed*; record loop-lag separately.
- **Event-loop/GIL jitter**: in a shared asyncio loop with CPU-bound physics, measured latency includes queueing behind the loop. In my (b) validation the loop with 300 tasks + a 20 ms tick had p50 tick lateness 1.1 ms, p99 1.6–2.4 ms, but 51 ms on the warm-up run. Mitigations: (1) **separate process per role**: sim worker (CPU) vs API client (asyncio, I/O only) connected by a queue; (2) run a *loop-lag monitor* (sleep 10 ms, log overshoot) in the API process and flag calls overlapping lag > 5 ms; (3) httpx/HTTP2 keep-alive client reused, connection pool warmed; (4) threads add GIL contention (Python 3.14 free-threading is optional; don't depend on it — agent 09); (5) compare asyncio vs thread vs process client in a 1-hour A/B with a mock local echo server (Follow-up spike) to size client overhead; expect < 1 ms for a healthy loop.
- **M4 specifics [I]**: 4 P + 6 E cores (10-core M4): macOS schedules background/low-QoS threads to E-cores (~2× slower); keep latency-critical processes at user-interactive/initiated QoS, avoid `taskpolicy -b`, use `caffeinate -i`; MacBook thermals throttle slowly under sustained all-core load → (i) interleave arms in time, (ii) log `pmset -g thermlog`/CPU freq proxy or a fixed micro-benchmark every N flights as a health canary, (iii) never run arm A fully before arm B. Real-time (b) runs should use ≤ 4 concurrent sim processes so the loop has headroom.
- **Network variance control**: pin provider+model+region (OpenRouter `provider` routing with `allow_fallbacks:false`), record response headers/model-version strings, do probe pings to the endpoint every 60 s, interleave arms, randomise order within hour blocks, record local time-of-day and Wi-Fi vs wired, run a fixed "reference prompt" canary every 5 min to detect drift; exclude/flag periods where canary p50 deviates > 30% from baseline; report latency at the granularity of (arm, hour block).
- **Cache caveat**: response caching (agent 09) makes latency measurement meaningless for cache hits: log `cache_hit` and use cached responses only for *actions*, taking latency from the recorded live measurement (c-sampled), never from the cache lookup time.
- **Clock sanity**: log `t_wall` start/end per flight plus NTP-free monotonic deltas; flight wall time is diagnostic only.

## 11. Avoiding "killer demo" artifacts

| Trap | Why it misleads | Guard |
|---|---|---|
| Lockstep sim with latency bars drawn afterwards | Drone never experiences the latency; bars are decoration | Only (b)/(c)/(d) for headline; demo replay must show the drone flying on stale commands |
| Fast sim + measured API latency added to a *different clock* | "Jev 50 ms vs LLM 2 s" while physics saw 0 | Convert with `ceil(L/Δ)`, log applied_tick |
| Comparing a mock latency for Jev against a measured latency for the LLM | Fairness violation | Same provider interface and same latency source type for all arms; flag which are measured vs assumed |
| Giving classical controllers zero latency | Free win vs models | Assign measured compute time → ticks; or explicitly label "on-board local reactor" |
| Cherry-picked scenario for the demo | Anecdote | Demo scenario chosen by rule (e.g. median paired-difference flight), with population stats beside it |
| Speed claims (sim_speed, decisions/s) mixed with API wait | Confounded | Four separate throughput metrics (Section 8) |
| Tuning effort asymmetry (Jev prompt engineered, PID default gains) | Henderson et al. style hidden hyper-parameter advantage | Equal tuning budget and log it |
| Choosing the latency axis after seeing results | Forking paths | Pre-register sweep and primary latency |
| Claiming Jev "avoids obstacles in X ms" from raw API latency | Reaction is ≈1.5 L + actuation lag + braking physics | Report `reaction_time` and time-to-clear, plus the physics-tolerable budget distribution for the scenario set |
| Above-knee comparisons (everyone collides) and below-knee (everyone fine) | No discriminative power | Choose scenario difficulty so baseline collision ≈ 20–60% at the headline latency; report the sweep |

## 12. Reproducibility manifest (per experiment run; stored as JSON next to results)

```json
{
  "experiment_id": "...", "git_sha": "...", "dirty": false,
  "python": "3.14.2|3.12.x", "packages": {"pybullet": "...", "gym_pybullet_drones": "...", "numpy": "...", "httpx": "..."},
  "hardware": {"chip": "Apple M4", "os": "macOS ...", "cores": "4P+6E", "thermal_canary_hz": 0},
  "time_model": {"mode": "c-sampled|c-measured|b|a", "control_dt_s": 0.02, "physics_hz": 240,
                 "latency_quantisation": "ceil", "cadence": "sequential|periodic", "period_ticks": 5,
                 "monotonic_apply": true, "timeout_s": 3.0, "fallback": "HOVER", "latched_ttl_s": null},
  "scenario_set": {"version": "...", "sha256": "...", "n": 4000, "feasibility_rule": "..."},
  "seeds": {"base": 20260930, "spawn_scheme": "SeedSequence(scenario_id, stream)"},
  "arms": [{"id": "...", "provider": "...", "model": "...", "model_version_string": "...", "prompt_sha": "...",
            "temperature": 0, "max_tokens": 3, "config_sha": "...", "tuning_budget": "..."}],
  "latency": {"source": "fixed|empirical_trace", "trace_sha": "...", "warmup_excluded": 20, "distribution_summary": {"p50":0,"p95":0,"p99":0}},
  "analysis": {"primary_metric": "collision_rate", "tests": ["exact_mcnemar","wilcoxon"], "correction": "holm", "bootstrap_B": 10000, "alpha": 0.05},
  "run_log": {"start_utc": "...", "end_utc": "...", "canary_drift_flags": []}
}
```
A result is *reproducible* if re-running with the same manifest + recorded latency trace + response cache gives identical decisions and physical outcomes up to the stated PyBullet tolerance; *replicable* if a fresh live run falls inside bootstrap CIs.

---

## Appendix A — Prototype simulator (`latsim.py`)

```python
"""Latency-injection prototype: braking-limited reactor vs sudden obstacle.

Point-mass 3D quadrotor-like model: commanded velocity -> attitude-lag-limited
acceleration (|a|<=A_MAX, first-order lag TAU). Vectorised over flights (F,3).
Accounting modes:
  lockstep  : sim paused during the call  -> latency ticks forced to 0
  seq       : latency-injected, one request in flight (next issued on arrival)
  pipe      : latency-injected, fixed-period issue (P ticks), many in flight,
              out-of-order results older than the applied one are discarded
"""

from __future__ import annotations

import numpy as np

DT = 0.02  # 50 Hz control tick
A_MAX = 5.0  # m/s^2 braking/tilt limit
TAU = 0.08  # attitude lag (s)
KP = 4.0
R_TOT = 0.65  # obstacle radius + drone radius
NEAR = 0.5  # near-miss band beyond R_TOT
T_TICKS = 450  # 9 s horizon
FORWARD, BRAKE = 0, 1


def make_scenarios(n: int, seed: int, t_app_range=(1.0, 3.0), horizon=T_TICKS):
    """Seeded scenario set. All stochastic inputs are pre-drawn per flight so every
    arm/latency sees identical worlds (common random numbers, paired design)."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(3.0, 6.0, n)
    t_app = (rng.uniform(*t_app_range, n) / DT).astype(int)
    # feasibility guarantee: obstacle appears at >= braking distance*1.1 + lag + margin
    d_feas = 1.1 * v**2 / (2 * A_MAX) + 0.15 * v + R_TOT + 0.2
    d_app = d_feas + rng.uniform(0.0, 4.0, n)
    lat = rng.uniform(-0.3, 0.3, (n, 2))
    return dict(
        n=n,
        v=v,
        t_app=t_app,
        d_app=d_app,
        lat=lat,
        wind=rng.normal(0, 0.3, (n, horizon, 3)),  # accel disturbance
        u_miss=rng.uniform(size=(n, horizon)),  # decision-error draws
        z_lat=rng.normal(size=(n, horizon)),  # latency jitter draws
    )


def init_state(sc):
    n = sc["n"]
    x = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    vel[:, 0] = sc["v"]
    a = np.zeros((n, 3))
    # obstacle placed d_app ahead of where drone is when it appears
    obs = np.zeros((n, 3))
    obs[:, 0] = sc["v"] * sc["t_app"] * DT + sc["d_app"]
    obs[:, 1:] = sc["lat"]
    return x, vel, a, obs


def step(x, vel, a, v_cmd, wind_t):
    a_cmd = np.clip(KP * (v_cmd - vel), -A_MAX, A_MAX)
    a = a + (a_cmd - a) * (DT / TAU)
    vel = vel + (a + wind_t) * DT
    x = x + vel * DT
    return x, vel, a


def run(
    sc,
    mode: str,
    lat_ms: float,
    sigma: float = 0.0,
    p_miss: float = 0.0,
    period: int = 5,
    return_detail: bool = False,
):
    """mode in {'lockstep_seq','lockstep_pipe','seq','pipe'}."""
    n = sc["n"]
    T = T_TICKS
    lockstep = mode.startswith("lockstep")
    cadence = mode.split("_")[-1] if lockstep else mode
    x, vel, a, obs = init_state(sc)
    Tb = T + 400
    buf_act = np.zeros((n, Tb), dtype=np.int8)
    buf_iss = np.full((n, Tb), -1, dtype=np.int32)
    next_issue = np.zeros(n, dtype=int)
    cur = np.zeros(n, dtype=np.int8)
    applied = np.full(n, -1, dtype=np.int32)
    v_fwd = np.zeros((n, 3))
    v_fwd[:, 0] = sc["v"]
    collided = np.zeros(n, bool)
    min_d = np.full(n, np.inf)
    first_brake = np.full(n, -1)
    n_dec = np.zeros(n, int)
    ar = np.arange(n)
    for t in range(T):
        issue = (t >= next_issue) if cadence == "seq" else np.full(n, t % period == 0)
        if issue.any():
            known = t >= sc["t_app"]
            miss = sc["u_miss"][:, t] < p_miss
            act = np.where(known & ~miss, BRAKE, FORWARD).astype(np.int8)
            if lockstep or lat_ms == 0:
                lt = np.zeros(n, int)
            else:
                lt = np.ceil(
                    lat_ms * np.exp(sigma * sc["z_lat"][:, t]) / 1000 / DT
                ).astype(int)
            arr = np.minimum(t + lt, Tb - 1)
            idx = ar[issue]
            buf_act[idx, arr[idx]] = act[idx]
            buf_iss[idx, arr[idx]] = t
            n_dec[idx] += 1
            if cadence == "seq":
                next_issue[idx] = np.maximum(arr[idx], t + 1)
        bi = buf_iss[:, t]
        upd = bi > applied
        cur = np.where(upd, buf_act[:, t], cur)
        applied = np.where(upd, bi, applied)
        fb = (cur == BRAKE) & (first_brake < 0) & (t >= sc["t_app"])
        first_brake[fb] = t
        v_cmd = np.where((cur == BRAKE)[:, None], 0.0, v_fwd)
        x, vel, a = step(x, vel, a, v_cmd, sc["wind"][:, t])
        d = np.linalg.norm(x - obs, axis=1)
        live = t >= sc["t_app"]
        min_d = np.where(live, np.minimum(min_d, d), min_d)
        collided |= live & (d < R_TOT)
    react = np.where(first_brake >= 0, (first_brake - sc["t_app"]) * DT, np.nan)
    res = dict(
        collision=collided,
        near_miss=(~collided) & (min_d < R_TOT + NEAR),
        reaction_s=react,
        n_dec=n_dec,
        min_clear=min_d - R_TOT,
    )
    return res


def critical_onset_delay(sc):
    """Physics-only tolerable reaction: smallest brake-onset delay d (ticks after
    the obstacle appears) that causes a collision (perfect perception)."""
    n = sc["n"]
    crit = np.full(n, 10**6)
    for d in range(0, 200):
        x, vel, a, obs = init_state(sc)
        v_fwd = np.zeros((n, 3))
        v_fwd[:, 0] = sc["v"]
        col = np.zeros(n, bool)
        for t in range(T_TICKS):
            brake = t >= sc["t_app"] + d
            v_cmd = np.where(brake[:, None], 0.0, v_fwd)
            x, vel, a = step(x, vel, a, v_cmd, sc["wind"][:, t])
            col |= (t >= sc["t_app"]) & (np.linalg.norm(x - obs, axis=1) < R_TOT)
        newly = col & (crit == 10**6)
        crit[newly] = d
        if (crit < 10**6).all():
            break
    return crit * DT


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)
```

## Appendix B — Sweep/analysis driver and real-time validation

`exp_curves.py` (produces all Section 7.2–7.4 tables):

```python
import json, time
import numpy as np
from latsim import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 4000
sc = make_scenarios(N, seed=20260930)
t0 = time.perf_counter()
crit = critical_onset_delay(sc)
print(
    "critical onset delay quantiles (s) p1,p5,p25,p50,p75:",
    np.round(np.quantile(crit, [0.01, 0.05, 0.25, 0.5, 0.75]), 3),
    f"{time.perf_counter() - t0:.1f}s",
)
out = {
    "crit_quantiles": dict(
        zip(
            ["p01", "p05", "p25", "p50", "p75"],
            np.quantile(crit, [0.01, 0.05, 0.25, 0.5, 0.75]).tolist(),
        )
    )
}

LATS = [0, 10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]
curves = {}
cfgs = {
    "lockstep (a), p_miss=.05": ("lockstep_seq", 0.0),
    "inject seq, sigma=0": ("seq", 0.0),
    "inject seq, sigma=0.6 (heavy tail)": ("seq", 0.6),
    "inject pipe P=100ms, sigma=0": ("pipe", 0.0),
}
for name, (mode, sg) in cfgs.items():
    rows = []
    for L in LATS:
        r = run(sc, mode, L, sigma=sg, p_miss=0.05, period=5)
        k = int(r["collision"].sum())
        lo, hi = wilson(k, N)
        rows.append(
            dict(
                lat_ms=L,
                coll=k / N,
                lo=lo,
                hi=hi,
                react_p50=float(np.nanmedian(r["reaction_s"])),
                react_p95=float(np.nanquantile(r["reaction_s"], 0.95)),
                near=float(r["near_miss"].mean()),
                dec_per_flight=float(r["n_dec"].mean()),
            )
        )
    curves[name] = rows
    print(name)
    [print("  ", {k: round(v, 4) for k, v in row.items()}) for row in rows]
out["curves"] = curves

# ideal perception (p_miss=0) knee, seq
rows = []
for L in LATS:
    r = run(sc, "seq", L, 0.0, 0.0)
    k = int(r["collision"].sum())
    rows.append(
        dict(lat_ms=L, coll=k / N, react_p50=float(np.nanmedian(r["reaction_s"])))
    )
out["seq_perfect"] = rows
print("seq p_miss=0", [(r["lat_ms"], round(r["coll"], 3)) for r in rows])

# arms: lockstep vs injected ranking flip
arms = {
    "reactive_local (2ms, miss .05)": (2, 0.05),
    "fast_model Jev-like (100ms, miss .08)": (100, 0.08),
    "slow_llm (1500ms, miss .005)": (1500, 0.005),
}
arm_out = {}
for name, (L, pm) in arms.items():
    row = {}
    for mode in ["lockstep_seq", "seq"]:
        r = run(sc, mode, L, sigma=0.4, p_miss=pm)
        k = int(r["collision"].sum())
        lo, hi = wilson(k, N)
        row[mode] = dict(
            coll=k / N,
            lo=lo,
            hi=hi,
            coll_vec=r["collision"].tolist() if False else None,
        )
    arm_out[name] = row
    print(name, {m: round(v["coll"], 4) for m, v in row.items()})
out["arms"] = arm_out
# paired McNemar arms under injected
from statsmodels.stats.contingency_tables import mcnemar

ra = run(sc, "seq", 100, 0.4, 0.08)["collision"]
rb = run(sc, "seq", 1500, 0.4, 0.005)["collision"]
b = int((ra & ~rb).sum())
c = int((~ra & rb).sum())
out["mcnemar_fast_vs_slow_injected"] = dict(
    b_fast_only=b, c_slow_only=c, p=float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)
)
ra = run(sc, "lockstep_seq", 100, 0.4, 0.08)["collision"]
rb = run(sc, "lockstep_seq", 1500, 0.4, 0.005)["collision"]
b = int((ra & ~rb).sum())
c = int((~ra & rb).sum())
out["mcnemar_fast_vs_slow_lockstep"] = dict(
    b_fast_only=b, c_slow_only=c, p=float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)
)
print(out["mcnemar_fast_vs_slow_injected"], out["mcnemar_fast_vs_slow_lockstep"])

json.dump(out, open("curves.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, rows in curves.items():
    L = [r["lat_ms"] for r in rows]
    ax[0].plot(L, [r["coll"] for r in rows], "o-", label=name)
    ax[0].fill_between(L, [r["lo"] for r in rows], [r["hi"] for r in rows], alpha=0.15)
ax[0].set_xscale("symlog", linthresh=10)
ax[0].set_xlabel("injected decision latency (ms)")
ax[0].set_ylabel("collision rate")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=0.3)
ax[0].set_title("Collision vs latency (N=4000 paired flights)")
ax[1].hist(crit, bins=60)
ax[1].set_xlabel("physics-tolerable reaction time (s)")
ax[1].set_title("Per-scenario critical reaction delay")
plt.tight_layout()
plt.savefig("curve.png", dpi=110)
```

`rt_validate.py` (option (b) vs (c), Section 7.6):

```python
"""Validate option (c) latency injection against option (b) real-time pacing.
(b): sim time == wall time (one tick per 20 ms deadline on perf_counter), each flight's
decision loop is an asyncio task that really awaits asyncio.sleep(latency)."""

import asyncio, json, time
import numpy as np
import latsim
from latsim import *

F = 300
RT_T = 200  # 4 s
latsim.T_TICKS = RT_T
sc = make_scenarios(F, seed=777, t_app_range=(0.4, 0.9), horizon=RT_T)


async def rt_run(lat_ms: float):
    x, vel, a, obs = init_state(sc)
    v_fwd = np.zeros((F, 3))
    v_fwd[:, 0] = sc["v"]
    cur = np.zeros(F, np.int8)
    tick = [0]
    collided = np.zeros(F, bool)
    late = []

    async def agent(f):
        while tick[0] < RT_T:
            act = BRAKE if tick[0] >= sc["t_app"][f] else FORWARD  # snapshot at issue
            await asyncio.sleep(lat_ms / 1000)  # real request latency
            cur[f] = act  # result arrives

    tasks = [asyncio.create_task(agent(f)) for f in range(F)]
    t0 = time.perf_counter()
    for t in range(RT_T):
        tick[0] = t
        dl = t0 + t * DT
        await asyncio.sleep(max(0.0, dl - time.perf_counter()))
        late.append(time.perf_counter() - dl)
        v_cmd = np.where((cur == BRAKE)[:, None], 0.0, v_fwd)
        x, vel, a = step(x, vel, a, v_cmd, sc["wind"][:, t])
        collided |= (t >= sc["t_app"]) & (np.linalg.norm(x - obs, axis=1) < R_TOT)
    for tk in tasks:
        tk.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return collided, np.array(late)


out = {}
for L in [100, 200, 300, 500]:
    col_rt, late = asyncio.run(rt_run(L))
    if L < 1:
        continue
    col_inj = run(sc, "seq", L, 0.0, 0.0)["collision"]
    agree = float((col_rt == col_inj).mean())
    out[L] = dict(
        rt=float(col_rt.mean()),
        inj=float(col_inj.mean()),
        flight_agreement=agree,
        tick_late_p50_ms=float(np.quantile(late, 0.5) * 1e3),
        tick_late_p99_ms=float(np.quantile(late, 0.99) * 1e3),
        tick_late_max_ms=float(late.max() * 1e3),
    )
    print(L, out[L], flush=True)
json.dump(out, open("rt_validate.json", "w"), indent=1)
```

## Appendix C — Power analysis (`power.py`)

```python
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize


def unpaired_n(p1, p2, alpha, power):
    h = abs(proportion_effectsize(p1, p2))
    return int(
        np.ceil(
            NormalIndPower().solve_power(
                h, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
            )
        )
    )


_cache = {}


def crit(m, alpha):
    """largest b such that exact two-sided McNemar rejects for b<=c or b>=m-c"""
    k = (m, alpha)
    if k not in _cache:
        b = np.arange(0, m + 1)
        p = np.minimum(
            1.0,
            2 * np.minimum(stats.binom.cdf(b, m, 0.5), stats.binom.sf(b - 1, m, 0.5)),
        )
        rej = np.where(p <= alpha)[0]
        rej_low = rej[rej <= m / 2]
        _cache[k] = int(rej_low.max()) if len(rej_low) else -1
    return _cache[k]


def mcnemar_power(n, p10, p01, alpha):
    pd = p10 + p01
    q = p10 / pd
    m = np.arange(0, n + 1)
    pm = stats.binom.pmf(m, n, pd)
    # restrict to mass region
    keep = pm > 1e-12
    pw = 0.0
    for mm, w in zip(m[keep], pm[keep]):
        c = crit(int(mm), alpha)
        if c < 0:
            continue
        pw += w * (stats.binom.cdf(c, mm, q) + stats.binom.sf(mm - c - 1, mm, q))
    return pw


def paired_n(p1, p2, rho, alpha, power):
    s = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    p11 = p1 * p2 + rho * s
    p10 = p1 - p11
    p01 = p2 - p11
    if p10 < 0 or p01 < 0:
        return None
    lo, hi = 10, 20
    while mcnemar_power(hi, p10, p01, alpha) < power:
        lo, hi = hi, hi * 2
        if hi > 60000:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mcnemar_power(mid, p10, p01, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi, round(p10 + p01, 3)


rows = []
print(
    "| baseline p_A | delta | unpaired n/arm a=.05 pw=.8 | unpaired a=.05 pw=.9 | unpaired Bonf a=.05/6 pw=.8 | paired rho=.3 (n, disc) | paired rho=.6 | paired rho=.8 |"
)
print("|---|---|---|---|---|---|---|---|")
for p1 in [0.05, 0.10, 0.30, 0.50]:
    for d in [0.05, 0.10, 0.20]:
        p2 = p1 + d
        if p2 > 1:
            continue
        u = [
            unpaired_n(p1, p2, 0.05, 0.8),
            unpaired_n(p1, p2, 0.05, 0.9),
            unpaired_n(p1, p2, 0.05 / 6, 0.8),
        ]
        pr = [paired_n(p1, p2, r, 0.05, 0.8) for r in (0.3, 0.6, 0.8)]
        print(
            f"| {p1:.0%} vs {p2:.0%} | {d * 100:.0f} pp | {u[0]} | {u[1]} | {u[2]} | "
            + " | ".join(f"{x[0]} (pd={x[1]})" if x else "infeasible" for x in pr)
            + " |"
        )

# continuous metric: paired t / Wilcoxon (ARE 0.955 -> n/0.955)
print("\ncontinuous paired (time_to_target): d_z -> n (t-test), n_wilcoxon ~ n/0.955")
for dz in [0.1, 0.2, 0.3, 0.5]:
    n = TTestPower().solve_power(dz, alpha=0.05, power=0.8, alternative="two-sided")
    print(dz, int(np.ceil(n)), int(np.ceil(n / 0.955)))
```

Reproduce: `uv venv --python 3.12 .venv && uv pip install numpy scipy statsmodels matplotlib && python exp_curves.py && python rt_validate.py && python power.py` (≈2 min on M4; needs `latsim.py` alongside).

---

## Decisions recommended

1. **Primary time model = (c) latency injection on `t_sim`; recorded per-decision latency trace for exact replay; (b) real-time-paced validation set (≥300 flights × 3 latencies); (a) lockstep kept only as a labelled "latency-free quality" ablation arm.** Never publish a latency comparison from (a) alone.
2. Convert every latency with `N = ceil(L/Δ)`; give every arm (including PID/MPC/FSM/RL) a non-zero, measured-or-declared latency; choose control tick Δ ≤ 10 ms for the setpoint loop if sub-20 ms points matter.
3. Both **sequential** and **periodic** cadences are reported; monotonic-apply rule for out-of-order results; timeout fallback identical across arms.
4. Headline plot = collision-rate vs injected latency with Wilson CIs + 2-D (latency × accuracy) map; report `reaction_time` (≈1.5 L sequential) alongside `decision_latency`.
5. Scenario = unit of analysis; paired design with common random numbers; exact McNemar, Wilcoxon, bootstrap, Holm; KM for censored times; heavy-tailed latency as ECCDF + percentiles with CIs.
6. Sample sizes: ≥2,000 scenarios for cheap arms (5 pp detectable), 300–500 paired for live-API arms (≥10 pp), with per-stratum planning.
7. Throughput reported as four separate metrics (decisions/sim-s, decisions/wall-s total and compute, sim_speed_compute vs total).
8. Architecture: sim worker process separate from asyncio API-client process; `perf_counter_ns`; loop-lag monitor; interleaved arm ordering; canary prompt for network drift.
9. Adopt the reproducibility manifest (Section 12) as a required table/JSON in the experiment record.

## Open questions

1. Real decision-tick size and physics rate in gym-pybullet-drones (48 Hz ctrl default?) — determines latency resolution (agent 02).
2. What is Jev, and does it offer streaming/TTFT instrumentation and a tolerable jitter distribution (agent 01)? Until known, latency for "Jev" is an assumption parameter, not a measurement.
3. Which latency statistic indexes the sweep for jittery providers: median, mean, or p95? (Prototype shows the ordering depends on it.) Proposal: index by median and report p95 alongside; add a tail-sensitivity factor.
4. How much real-API (c-measured) data is enough to fit a latency distribution with autocorrelation (hundreds of calls per provider per time block)? Budget owned by 01/09.
5. Is the oracle "correct action set" (for reaction_time, decision_correctness) well-defined for multi-event situations (obstacle + wind + battery)? (agents 06/11)
6. Does a local safety shield belong in all arms (then latency of LLMs matters less) or as a separate factor? Affects the headline claim ("Jev avoids collisions" vs "Jev+shield").
7. Real quadrotor braking limits (attitude/thrust coupling, battery sag) — what A_MAX distribution replaces the constant 5 m/s² used here?
8. PyBullet cross-process determinism tolerance (for paired replay claims).

## Follow-up spikes (small, runnable experiments)

1. **Port the prototype to gym-pybullet-drones** (VelocityAviary or custom BaseAviary subclass with `applyExternalForce` disturbances; obstacle via `loadURDF` at t_app): reproduce the collision-vs-latency curve for N=500 flights; compare the knee to the point-mass knee. Exit: curve shape matches and knee location explained by measured braking distance.
2. **(b) vs (c) with a real network path**: local `httpx` client → local mock server with injectable sleep + jitter distribution (no API key), 300 flights × {100, 300, 1000} ms; measure agreement and client overhead (asyncio vs thread vs process).
3. **Latency-accuracy 2-D map**: implement mock `DecisionProvider(latency_model, p_miss, error_type)` and compute the 13×5 map with paired N=2000; estimate the iso-collision contour slopes.
4. **Power-check by simulation**: use the prototype's own paired collision vectors to verify McNemar power predictions (simulate 1000 subsample draws at n from the table; empirical power should be within ±3 pp of 80%).
5. **Jitter/tail study**: fit lognormal/Weibull/mixture to a published provider latency distribution (from agent 01's scripts once run), use block bootstrap for autocorrelated latency; check sensitivity of knee to tails.
6. **Loop-lag characterisation on M4 under load**: 4/6/8 sim worker processes + 1 API client process; log loop lag, tick lateness p99, thermal canary over 30 min; set the max concurrency where p99 lag < 2 ms.
7. **Sub-tick latency handling**: test Δ = 10 ms vs 20 ms vs fractional-tick interpolation on the 10/20/50 ms sweep points; decide resolution.
8. **Live c-measured pause-and-charge loop** for one provider when a key is available (owner 01): verify `N = ceil(L_meas/Δ)` logging, that sim wall time ≈ Σ calls, and that replaying the trace reproduces bit-identical decisions.
