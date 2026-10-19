# 05 — Baselines: PID/MPC, reactive avoidance, RL, conventional FSM policy

Agent 05 of 14. Date of research: 2026-09-30. Hardware for spikes: Apple M4 (arm64), uv 0.9.18, CPython 3.14.2 and 3.14.6 available, scratch venv `scratchpad/05/v314` (Python 3.14.x). Scripts: `scratchpad/05/mpc_bench.py`, `scratchpad/05/ppo_fps.py`.

Labels: **[V]** verified (source or measurement), **[I]** inference/design choice, confidence H/M/L.

---

## 0. Executive summary

1. Seven arms A–G plus sanity/oracle arms (Z0 random, Z1 always-HOVER, Z2 always-FORWARD, O oracle). All share one `Observation`, one 11-action enum, one low-level PID, one safety shield (or an explicit "no-shield" flag), one scenario set. Arms differ only in *who picks the action (or the setpoint) and how long that takes*.
2. The fairest non-ML baseline is arm E (FSM over the same 11 actions, rules derived from braking distance). It is the arm Jev must beat to claim "smart", while the latency sweep (doc 08) separates "fast" from "smart".
3. MPC is realistic only as a **linear/convex point-mass MPC (6-state double integrator, N=20, dt=0.1 s) on top of the PID attitude loop**, solved with OSQP directly. Measured on M4: **0.11 ms median, 0.52 ms p95** (raw OSQP, sparse, warm-started); cvxpy+OSQP wrapper 28 ms median (not DPP in my formulation, so avoidable overhead); CasADi+IPOPT nonconvex circular obstacle 65 ms median, 255 ms p95 (cold). Do not use acados (no PyPI package, not installable via uv on 3.14).
6. Install viability on Python 3.14 / arm64 **[V]**: numpy 2.5.3, scipy 1.18.1, osqp 1.1.3, cvxpy 1.9.3, clarabel 0.11.1, casadi 3.8.1, gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.14.0 (MPS available) all installed from binary wheels. **pybullet 3.2.7 has no cp314 wheel** (`--only-binary` fails; it must compile from sdist; README says macOS needs `CFLAGS="-Dfdopen=fdopen"`). `gym-pybullet-drones` is not on PyPI (install from git). acados_template not on PyPI. Owner 02 covers the pybullet compile; if it fails, sim worker lives in a 3.12 venv and the controller/harness code must be importable by both.
7. PPO: the repo's own `learn.py` trains hover with `int(1e7)` steps, 1 env, MlpPolicy **[V]**. For the discrete-11-action high-level task plan **5M steps per final policy (budget cap 10M), 3 seeds, MaskablePPO optional**. Measured PPO-only overhead on M4 CPU (CartPole, free env): 893 steps/s with 1 env, 1867 steps/s with 8 subprocess envs. With the sim in the loop expect ~1.0–2.5k env-steps/s → 5M steps ≈ 0.6–1.4 h per run. Total RL campaign ≈ 10–20 h laptop time; this is a real cost and must be budgeted as tuning effort.

Contradictions with `docs/initial-plan.txt`: README (fetched today) says "Tested on Intel x64/Ubuntu 24.04 and Apple Silicon/macOS 26" and recommends `conda create python=3.12`, so the "Apple Silicon tested" claim is **true [V]** but for Python 3.12 not 3.14; README content returned no 20 kHz / 80 drone figure in my fetch (doc 02 owns that claim). The plan also says Jev replaces nothing low-level: consistent with the arm design here.

---

## 1. Shared experimental substrate (applies to every arm)

### 1.1 Layers (so arms are comparable)

```
Scenario+events -> Sim (PyBullet / kinematic) -> Observation (shared dataclass)
                                                       |
                       arm policy  (A..G, O, Z*)  ---> Action in {11}   (or setpoint for A, C)
                                                       |
                              ActionMapper (doc 11): action -> velocity/position setpoint
                                                       |
                          SafetyShield (optional, same code for all arms, flag)
                                                       |
                        DSLPIDControl / SimplePIDControl -> RPM
```

Only arms A and C bypass the action enum (they emit setpoints directly), by design: they are the "conventional controller" references, not discrete-policy arms.

### 1.2 Reference physical constants used to derive thresholds **[I, M]**

These are *symbols*; doc 11 / doc 02 spikes must measure them for the CF2X model in gym-pybullet-drones. Placeholder numbers for worked examples:

| Symbol | Meaning | Placeholder |
|---|---|---|
| `v_cruise` | FORWARD speed | 3.0 m/s |
| `v_max` | speed cap | 5.0 m/s |
| `a_brk` | achievable deceleration under BRAKE (horizontal) **measured, use 70% of measured as design value** | 4.0 m/s² |
| `t_ctl` | control-loop latency (one control tick + attitude lag) | 0.05 s |
| `t_dec` | decision latency for the arm (age of the action in effect); arm-dependent | 0.02 s (E), Jev x, LLM y |
| `r_drone` | drone radius incl. prop guard | 0.10 m |
| `d_margin` | safety margin | 0.30 m |

Stopping distance from speed `v` (closing speed along the obstacle bearing):

```
d_stop(v, t_lat) = v * (t_lat + t_ctl)  +  v^2 / (2 * a_brk)  +  r_drone + d_margin
```

Example at v=3 m/s, t_lat=0.05: 3*0.10 + 9/8 + 0.4 = 1.825 m. Time-to-collision `ttc = d_obs / max(v_close, eps)`; braking is required when `ttc < t_brake_needed = v_close/a_brk + t_lat + t_ctl`.

**Feasibility link to doc 06:** a scenario is "avoidable by an arm with decision age t" iff obstacle appearance distance `d_appear >= d_stop(v, t)` *or* a lateral dodge fits. Oracle uses t = 0. The sweep on latency should show a knee exactly where `d_appear < d_stop(v, t_lat)`.

### 1.3 Observation consumed by policy arms (projection, doc 07 owns the format)

```
pos(3), vel(3), yaw, target_rel(3), home_rel(3), battery_frac, gps_ok(bool),
range_sectors[N=16 azimuth x 1 elevation band] (m, capped 10), min_range, min_range_bearing,
closing_speed (m/s along min-range bearing), other_drone_rel(3)+vel(3) or None, obs_age_s
```
Arm D (RL) gets exactly this vector normalised; arm E reads the same fields; arm F/G get a text rendering of the same fields. Arm B uses `range_sectors` (APF/VFH) or other_drone (ORCA). Arm C additionally gets the nearest obstacle as a sphere estimate (centre, radius) built from sectors.

---

## 2. Arm definitions

Notation: `T` = target waypoint, `p`,`v` = state, `ctrl_hz` = 48 (gym-pybullet-drones default ctrl_freq is commonly 30 or 48; **assumption I**), decisions at `dec_hz` per arm.

### Arm A — Waypoint PID, no avoidance
- Controller: `DSLPIDControl.computeControl(target_pos=T_wp, target_vel=...)` **[V that class exists per task brief; signature per doc 02]**; waypoints from straight-line or scenario path, advancing when `|p - wp| < 0.3 m`.
- No action enum, no shield, no reaction to obstacles, wind, GPS loss or battery. Failures are collisions (scored), timeouts.
- Purpose: lower bound on safety; upper bound on efficiency (shortest path) in obstacle-free scenarios. Latency: decision compute ~0, `decision_latency = 0`.
- Variant A′ (A + shield): A with the shared SafetyShield active, isolates how much the shield alone buys. Report both; A′ is the real "PID baseline" if the shield is on for all arms.

### Arm B — PID + reactive avoidance (three sub-arms, pick B1 as primary)
All output a **desired velocity** `v_des` fed to the PID in velocity mode (VelocityAviary-style / `DSLPIDControl` with `target_vel`). They run every control tick (latency ≈ one tick, compute << 1 ms in numpy) — this is the "no-latency classical reactive" reference.

**B1: TTC-brake + lateral nudge (simplest, recommended primary)**
```
v_goal   = v_cruise * unit(T - p)            # capped by d_target braking profile
for each sector i: d_i = range_sectors[i], u_i = bearing unit vec
  closing_i = max(0, dot(v, u_i))
  ttc_i     = d_i / max(closing_i, 0.05)
  stop_i    = d_stop(closing_i, t_lat=0)
danger = any(d_i < stop_i)  (only sectors within +-60 deg of velocity direction)
if danger: v_des = 0 along hazard axis (brake), add lateral component toward freer side:
            v_des = v_goal - dot(v_goal,u_h) u_h  ;  lateral = side * v_lat (v_lat = 1.5 m/s)
            side = sign of (sum free range right) - (sum free range left)
            scale v_des by clip((d_h - r_drone - d_margin)/d_stop, 0, 1)
else v_des = v_goal
```
**B2: APF (Khatib 1986 style)**
```
F_att = k_att * clip(T - p, max=v_cruise)           # k_att = 1.0 1/s
F_rep = sum_i  k_rep * (1/d_i - 1/d0) * (1/d_i^2) * (-u_i)   for d_i < d0
        d0 = d_stop(|v|,0) + 0.5 m (speed-adaptive influence radius), k_rep = 0.8
v_des = clip(F_att + F_rep, |.|<=v_max)
local-minimum escape: if |v_des|<0.2 and |T-p|>1 for >1.0 s: add tangential force 0.5*unit(perp)
```
Tuned params: `k_att, k_rep, d0 scale, tangential gain`. Known failure: local minima and oscillation near narrow gaps (inference from literature, H).

**B3: VFH (Borenstein & Koren 1991; VFH+ Ulrich & Borenstein 1998)**
```
polar histogram from range_sectors (16-36 bins); h_k = a - b * d_k  clipped at 0 (certainty ~ 1/d)
low-pass smooth; threshold tau; candidate valleys wider than (2*(r_drone+d_margin))/d_k rad
choose valley whose direction minimises  a1*|theta_k - theta_goal| + a2*|theta_k - theta_v| + a3*|theta_k - theta_prev|
speed = v_cruise * (1 - h_c/h_max); v_des = speed * dir(theta_k)
```
Tuned params: threshold `tau`, cost weights `a1..a3`, valley width.

**ORCA (for second-drone event only, variant B4):** ORCA (van den Berg, Guy, Lin, Manocha; "Reciprocal n-body collision avoidance") needs both agents to share the rule (reciprocity) which a non-cooperating intruder breaks; use the **non-reciprocal VO half-plane** (agent takes full responsibility, factor 1.0 instead of 0.5) and a LP in 3D (scipy `linprog` or 3-line custom half-plane clip; ≤ 1 ms). Only activated when `other_drone` present.
**CBF (control barrier function) option:** shield-level `h(x)=d - d_stop(v_closing)`; solve the 3-variable QP `min |u - u_nom|^2 s.t. dh/dt + alpha h >= 0` with OSQP (<0.3 ms). This is the recommended content of the SafetyShield (shared by all arms), not a separate arm, to avoid double counting. **[I, M]**

Report B1 as "B"; B2/B3/B4 as ablations (each ≤ 2 h tuning, identical budget rules §6).

### Arm C — MPC (see §3 for formulation and timing)
Linear MPC producing `a_cmd` / `v_des` for the PID; obstacle half-space constraints with linearised per-step separating planes; target tracking cost; wind treated as unknown (nominal model) unless a disturbance observer is added (ablation C+). Compute time measured per solve, injected as latency (§5).

### Arm D — RL (PPO, discrete 11 head)
- Same observation vector and same 11-action discrete head as E/F/G; same ActionMapper + PID under it. Action repeat = decision period.
- Algorithm: `stable-baselines3` PPO (`MlpPolicy`, 2x128 tanh, `n_envs=8`), or `sb3-contrib` `MaskablePPO` with the same validity mask the FSM has (e.g. RETURN_HOME illegal when already home); the mask is a fairness choice: apply it in D only if E/G also see it (I recommend no mask, pure policy).
- Reward (dense, **the main tuning surface; freeze before evaluation**): `+ 1.0*(d_prev - d_curr)/v_cruise` progress, `- 50` on collision (terminate), `+ 100` on goal, `- 0.01` per step, `- 0.3*max(0, 1 - d_min/d_stop)` shaping for proximity, `- 0.05` per action switch (oscillation), `- 20` crash from EMERGENCY_LAND when avoidable. Target-reached within 0.3 m and speed < 0.5.
- Training distribution: the scenario generator of doc 06 with **training seeds disjoint from tuning and test seeds**; curriculum: events off → obstacle only → wind → all, mixing 30% events-off throughout to prevent forgetting. Domain randomisation of latency in `{0..3 ticks}` to avoid a policy that only works at one delay.
- Latency: forward pass MLP ≈ 0.1–0.3 ms on CPU (inference, I); count as compute time. **Training-time latency must equal the evaluation-time latency model** or D is handicapped: train with its own small fixed delay (1 decision period), evaluate at the same.
- Variant D-cont (continuous velocity action SAC): not recommended (breaks action-space parity). Mention only as a footnote.

### Arm E — Rule-based FSM over the 11 actions (conventional policy)
See §4 for full pseudocode. Decision rate = control tick rate or 10–20 Hz (configurable; 0 ms compute to ~0.05 ms). Acts as the "conventional policy" named in the plan and as the main intelligence control.

### Arm F — Slow LLM
Same prompts/format as G with a slower/larger model; text observation; temperature 0; max_tokens 1–3; same fallback on parse failure. Decision latency from measured/modelled distribution (doc 01, 08). Between decisions the last action is held (action TTL per doc 11) — the drone does not freeze.

### Arm G — Jev
Same as F with the Jev provider (identity unresolved; doc 01 → stand-in). Arms F and G are *identical except provider and latency*, so any difference is explained by model quality + latency only.

### Sanity / oracle arms

| Arm | Definition | Expected | Purpose |
|---|---|---|---|
| Z0 random | uniform over 11 actions per decision | collides/timeouts almost always; success ≈ 0 | floor, checks metrics non-degenerate |
| Z1 always-HOVER | never moves | collision 0 except dynamic intruder/events; success 0; timeout 100% | shows "no collision" is not credit; collision rate must be reported jointly with success |
| Z2 always-FORWARD | ignores everything | ≈ arm A without PID smoothing | checks ActionMapper |
| Z3 "panic" | EMERGENCY_LAND at first event | lands safely, success 0 | safety floor metric |
| O oracle | uses **ground-truth scenario labels** (full event timeline, true obstacle state), zero latency, plans with the same ActionMapper: doc 06's correct-action labeller + a receding horizon search (see below) | best attainable collision/time given the action set | upper bound; gap A→O = headroom |
| O2 oracle-continuous | C-MPC/kinodynamic planner with ground-truth obstacles and zero latency, continuous setpoints | bounds what the discrete action set costs | quantifies action-space loss (doc 11 gaps) |

Oracle algorithm for O: at each decision, enumerate the 11 actions, roll out each for `H = 2 s` in a **fast internal kinematic model** (same ActionMapper + 1st-order velocity lag, true obstacle trajectories from the scenario JSON), pick the lowest-cost action (collision => inf; cost = distance-to-target at H + 0.1*switch penalty + battery feasibility). 11 actions × 40 steps = 440 model steps/decision, ≈ 0.2 ms in numpy vectorised. Because the oracle uses the same kinematic model for planning as the tiered fast sim (doc 03), keep an "oracle vs PyBullet" gap check (oracle should be ≥ 99% collision-free on feasible scenarios; if not, the feasibility guarantee of doc 06 is wrong).

---

## 3. MPC formulation (arm C) — measured feasibility

### 3.1 Recommended formulation **[I, M]**
- **Not** a full quadrotor NMPC through RPMs. The PID (DSLPIDControl) already handles attitude; MPC is an outer-loop translational planner, exactly the "MPC above PID" layering of the plan.
- State `x = [p(3), v(3)]`, input `u = a_cmd(3)` (double integrator, `x+ = A x + B u`, dt = 0.1 s, N = 20 → 2 s horizon). Output: either `v_des = v + a_cmd*dt` (velocity-tracking mode, recommended) or `p_des = x_1`.
- Cost: `sum (x_k - x_T)' Q (x_k - x_T) + u_k' R u_k + terminal 10Q`; `Q = diag(10,10,10,1,1,1)`, `R = 0.1 I`.
- Constraints: `|v| <= v_max` (box), `|a| <= a_max` (box; `a_max` = measured tilt-limited accel ×0.7), altitude floor/ceiling, **obstacle avoidance as linearised separating half-spaces** `n_k'(p_k - c_k) >= r_obs + r_drone + margin` per obstacle per step, re-linearised each solve around the previous predicted trajectory (sequential convexification); moving obstacles use constant-velocity predictions. Soft slack (penalty 1e4) on the obstacle rows to avoid infeasibility (→ in infeasible/slack-heavy case, fall through to the BRAKE setpoint and count as `mpc_infeasible`).
- Wind/battery: nominal model; thrust derating enters as shrinking `a_max` only if the MPC is given battery fraction (ablation C+ gets it, C does not; this matches what E/G are told).
- GPS loss: MPC state estimate freezes/dead-reckons same as all arms (shared estimator, doc 07).

### 3.2 Solver and measured solve time (M4, Python 3.14, script `scratchpad/05/mpc_bench.py`) **[V]**

| Setup | Size | Median | p95 | Notes |
|---|---|---|---|---|
| OSQP 1.1.3 direct, sparse full-state, warm start, eps 1e-3, `update(q,l,u)` | 186 vars | **0.11 ms** | **0.52 ms** (p99 0.79) | constraints: box v/a + 20 half-space rows; 200 closed-loop solves, status solved |
| cvxpy 1.9.3 + OSQP, parameterised | same | 28.2 ms | 40.1 ms | cvxpy warned "not DPP" (`quad_form` with parameters in my formulation), first solve 48 ms; a DPP-compliant rewrite (`sum_squares` with `Q^{1/2}`) would be faster but is still a wrapper tax, avoid in timing runs |
| CasADi 3.8.1 Opti + IPOPT, nonconvex circular obstacle, cold each time | same | 65 ms | 255 ms | IPOPT is bundled; usable for a nonconvex NMPC variant C-NL; heavy tail |

Interpretation: use **raw OSQP** for the headline arm C (sub-millisecond, compute latency is negligible at 10–20 Hz control rates; matrices prebuilt, only `q,l,u` updated each tick). cvxpy only for prototyping. C-NL (CasADi+IPOPT or `ca.nlpsol` with SQP/qrqp, better: `ca.qpsol` with `osqp`) as an ablation with 65–250 ms compute to show a realistic "classical optimal control has compute latency too" curve — this is honest and useful, but make it clear it is solver-tuning-dependent. Caveats: constraints tiny vs real quadrotor NMPC; my numbers are a point-mass benchmark, not the PyBullet loop. Solve time at N=40 or with 5 obstacles: expect 2–4× (inference, L).
- **acados**: `acados_template` not found in the package registry (uv dry-run error), requires building the C library from source (CMake, BLASFEO/HPIPM); reject for a laptop project with 3.14. **[V for PyPI absence; I for rest]**
- Replay note: compute time for C is **measured per solve with `perf_counter_ns` and converted to ticks of delay** exactly like provider latency (§5), no special casing.

### 3.3 MPC and the discrete actions (doc 11 question)
Keep C outside the enum (continuous setpoint). A second variant **C-act**: MPC as the ActionMapper of a discrete policy (action → cost/constraint set: BRAKE → target v=0, AVOID_LEFT → lateral offset constraint, etc.) is an ablation only; it tells whether an MPC-based mapper improves G/F. Not in the core matrix.

---

## 4. Arm E — FSM over the 11 actions (exact decision rules)

Design goals: (i) no learning, (ii) every rule derived from `d_stop`, (iii) tuned with the same budget as others, (iv) interpretable. Evaluated at `dec_hz = 20` (and swept with an artificial delay for the latency curves: FSM is the clean "zero-compute but fixed-rule" reference; see §5).

State carried: `mode`, `t_mode_entered`, `avoid_side`, `last_action`.

Parameters (all tunable, defaults in brackets; "P" means tuned in §6): `v_cruise`=3, `a_brk_design`=0.7*a_brk_measured, `t_lat_assumed`=0.05+0.1, `margin`=0.3, `hyst`=0.25 m, `min_dwell`=0.3 s, `batt_rth`=0.30, `batt_land`=0.12, `gps_loss_hover_s`=2.0, `gps_loss_land_s`=8.0, `clear_ttc`=3.0 s.

```python
def decide_fsm(obs, st) -> Action:
    v_close   = obs.closing_speed                      # >0 approaching, along min-range bearing
    d_obs     = obs.min_range
    d_stop    = v_abs * (T_LAT + T_CTL) + v_close**2/(2*A_BRK_DESIGN) + R_DRONE + MARGIN
    ttc       = d_obs / max(v_close, 0.05)
    d_tgt     = obs.dist_to_target
    d_home    = obs.dist_to_home
    # ---- 1. terminal safety (highest priority, latched until landed) ----
    if obs.battery <= BATT_LAND or obs.on_ground_fault or st.mode == EMERG:
        st.mode = EMERG;                       return EMERGENCY_LAND
    if obs.altitude < 0.3 and d_obs < 0.5:     return EMERGENCY_LAND   # unrecoverable
    # ---- 2. degraded navigation ----
    if not obs.gps_ok:
        if st.gps_lost_for >= GPS_LOSS_LAND_S: st.mode = EMERG; return EMERGENCY_LAND
        if st.gps_lost_for >= GPS_LOSS_HOVER_S or v_abs > 1.0:   return HOVER if v_abs < 0.3 else BRAKE
    if obs.battery <= BATT_RTH or st.mode == RTH:
        if d_home > 0.4 + feasibility(obs.battery, d_home, v_cruise): pass   # can't make it -> land
        else: st.mode = RTH ;  # falls through with target := home
    # ---- 3. collision avoidance (static & dynamic & second drone) ----
    hazard = d_obs < d_stop * (1 + HYST if st.mode == AVOID else 1.0)
    if hazard and v_close > 0.1:
        free_l, free_r = mean_range(sectors,-90..-20), mean_range(sectors,20..90)
        fits_dodge = lateral_clear(obs, need = R_DRONE+MARGIN) and d_obs >= v_abs*T_LAT + dodge_dist(v_abs)
        # choose: dodge if a lateral side has room and the obstacle is not too close to dodge; else stop
        if d_obs < v_abs*(T_LAT+T_CTL) + v_abs**2/(2*A_BRK_DESIGN):   return BRAKE    # committed stop
        if fits_dodge:
            st.avoid_side = st.avoid_side or (LEFT if free_l > free_r else RIGHT)   # hold side (hysteresis)
            return AVOID_LEFT if st.avoid_side == LEFT else AVOID_RIGHT
        vertical clear? -> return ASCEND if ceiling_room and up_clear else DESCEND if down_clear else BRAKE
    if ttc < CLEAR_TTC and st.mode == AVOID and time_in_mode < MIN_DWELL: keep last avoid action
    st.avoid_side = None
    # ---- 4. stability ----
    if v_abs > v_safe_for_wind(obs.wind_est) : return BRAKE
    # ---- 5. navigation ----
    e = obs.target_rel_body                       # (forward, left, up)
    if d_tgt < ARRIVE_R and v_abs < 0.5:          return HOVER
    if d_tgt < stop_dist_for_arrival(v_abs):      return BRAKE           # decel profile v^2/(2a) > d_tgt - arrive_r
    if abs(e.up) > ALT_DEAD and abs(e.up) >= 0.5*abs(e.fwd): return ASCEND if e.up>0 else DESCEND
    if abs(bearing(e)) > YAW_DEAD (20 deg):        return LEFT if e.left>0 else RIGHT
    return FORWARD
```
Rules worth stating explicitly:
- **Threshold derivation:** `BRAKE` trigger is `d_obs < v*(t_lat+t_ctl) + v²/(2 a_brk_design) + r + margin` (above); `AVOID` trigger uses the same with hysteresis `(1+HYST)` so it does not chatter; `AVOID` only if a dodge fits: lateral needs `dodge_dist(v) ≈ v*t_lat + sqrt(2*(r+margin)/a_lat)*v` (time to shift `r+margin` laterally at `a_lat`, design `a_lat` = 2.5 m/s²).
- **Dwell/debounce:** any action other than BRAKE/EMERGENCY must persist `MIN_DWELL` = 0.3 s; EMERGENCY latched; AVOID side latched until `ttc > CLEAR_TTC`.
- **Sensor noise:** range readings median-filtered over 3 ticks inside the shared estimator; FSM adds 1.15× inflation to `d_stop` when `obs.noise_flag` is raised.
- **Second drone:** `closing_speed`/`min_range` computed against other drone too; same rules (plus it prefers climbing if `e.up` free, to avoid both-left deadlock).
- **Moving target:** target vector is re-read every tick, so navigation rules need no change; `stop_dist_for_arrival` uses relative velocity.
- No learned component; ≈ 60–90 lines of code. Expected to be strong on physical-rule events (obstacle, battery, GPS); that is the point: Jev must beat a competent rule-based agent, not a strawman.

**Conventional-policy tuning parity:** E is tuned by the same budget and protocol (§6) — the FSM design above is "manual tuning of ≤ 8 parameters" while RL gets an equal number of GPU/CPU hours, and LLM arms get a prompt-engineering budget (≤ N prompt variants, evaluated on the tuning split).

---

## 5. Latency treatment for classical arms (input to doc 08)

All arms pass through one `DecisionClock`: `effective_delay_ticks = ceil((compute_time + transport_time)/tick)`; action becomes active at `t_decision_request + delay` and the **previous action is held meanwhile**; state staleness = observation age at decision. Specifically:
- A (PID only): no decision loop; control compute (PID) counted as 0 extra ticks (shared by all).
- B: measured compute of the avoidance step (numpy, ≤ 0.2 ms) → 0 ticks at 48 Hz; include `perf_counter_ns` anyway for reporting.
- C: measured solve time per call (0.1–0.5 ms OSQP → 0 ticks; C-NL 65–255 ms → 3–12 ticks at 48 Hz) — a good illustration that "classical" is not free either.
- D: network forward pass ≈ 0.1–0.3 ms → 0 ticks. Train at same delay.
- E: < 0.1 ms → 0 ticks; + *artificial injected delay sweep* 0,10,50,100,300,1000,3000 ms for every arm including A-E, so collision-vs-latency curves are available for **every** arm and the "speed vs smartness" separation needs no assumptions about the LLM. Report "intrinsic latency" separately from "sweep latency".
- F/G: modelled/measured latency per doc 01/08.
- Decisions/sec: A–E report the *loop* rate (capped by control ticks), F/G report API-limited completions; never compare them unqualified (doc 08).

---

## 6. Fairness / tuning-parity protocol

1. **Splits (by seed, disjoint):** `train` seeds 0–9,999 (RL only), `tune` seeds 100,000–100,499 (500 scenarios, all arms tune here), `test` seeds 200,000+ (frozen, touched once per arm at the end, doc 08 sets N). Scenario JSON includes seed; events generated from scenario seed only, not arm seed (paired design).
2. **Equal tuning budget, defined in three currencies:** (a) **human-effort:** ≤ 2 engineer-hours per arm for rule/classical arms (A′, B, C, E); (b) **evaluation budget:** ≤ 20 tuning configurations × 500 tune scenarios for each of A′, B, C, E (random search or manual; log all trials); (c) **compute budget:** RL: ≤ 10 M env steps total per configuration, ≤ 5 reward/hyperparameter configurations, 3 seeds each; LLM arms: ≤ 10 prompt variants × 500 tune scenarios (sub-sampled to 100 for paid API). Publish the budget table and every trial.
3. **Tuning objective, identical:** `J = collision_rate*10 + (1-success_rate)*5 + normalised_time + 0.5*oscillation`, evaluated on tune seeds at the arm's intrinsic latency and at 100 ms. No test-seed peeking; freeze configs before test run (hash configs into the reproducibility manifest).
4. **Same shield/same PID/same ActionMapper for all.** Any arm-specific parameters live in its config, nothing in code branches.
5. **Information parity:** arms get the same sensors; arm C gets obstacle sphere estimation derived from the same rays; oracle only sees ground truth. Matrix of info: A (target only), B (+range), C (+range, sphere fit), D/E/F/G (full Observation). Document it in a table in the final report. If arm A lacks range sensors that is intentional.
6. **Latency parity:** same clock model and same sweep; LLM output parse failures → fallback action (same as E's `HOVER`/`BRAKE` by policy) logged as `invalid_rate`.
7. **Randomness:** D uses 3 training seeds; report mean ± CI over seeds *and* scenarios (nested: seed is a cluster); do not pick the best seed. E, B, C have no training randomness.
8. **Report ablations of unfairness:** e.g. "D at 5M vs 10M steps", "E default vs tuned", so the reader sees the sensitivity to tuning effort.
9. **Report jointly:** collision_rate *and* success_rate (Z1 always-HOVER demonstrates why).

---

## 7. RL training budget and feasibility (arm D)

Evidence **[V]**: gym-pybullet-drones `learn.py`: PPO, `MlpPolicy`, 1 env, `int(1e7)` steps, eval every 1000 steps on 10 episodes, reward-threshold early stop; hover task `one_d_rpm` (source: raw.githubusercontent.com/utiasDSL/gym-pybullet-drones/main/gym_pybullet_drones/examples/learn.py, fetched 2026-09-30). Literature: end-to-end quadrotor PPO for gate/stabilisation tasks uses 20–40 M steps and ≈ 42 h for 20 M steps in one report (arXiv 2412.12442, Multi-Task RL for Quadrotors, via search 2026-09-30) — these are *low-level* control tasks; ours is a **high-level discrete policy over a stabilised quad** (like a game agent), which needs far fewer interactions. No study found for this exact task (confidence on the budget: L-M) — treat 5M as a starting point and confirm with learning-curve saturation.

Budget proposal **[I, M/L]**:
- RL env step = one decision (0.1 s, = ~5 physics/control ticks at 48 Hz). Episode ≈ 15–40 s → 150–400 RL steps. 5 M steps ≈ 12–33 k episodes (training set is seeds 0–9,999 reused with event sampling → ok).
- Expected convergence: basic navigation 0.3–0.5 M steps; obstacle avoidance robust to sudden events 2–5 M; full event mix (wind, battery, GPS, second drone) 5–10 M. Acceptance: success ≥ 0.9× FSM on tune seeds, training curve flat over last 20%.
- Throughput **[V for PPO overhead]** (CartPole with free env, M4 CPU, torch 2.14 single thread, SB3 2.9.0): 893 steps/s at 1 env (batch 1024), 1,867 steps/s with 8 SubprocVecEnv; with default batch 64 it drops to 519/846. Real sim step cost dominates: if the PyBullet drone gives ~3–8k physics steps/s DIRECT (doc 02) → 600–1.6k RL steps/s → combined ≈ 400–1,000 steps/s on 1 env, 1.2–2.5k with 6–8 workers (workers also compete with the evaluation runs; plan on 4 workers). **5 M steps ≈ 0.6–1.2 h (8 env), up to 3.5 h (1 env)**. Plan: 3 seeds × 5 reward configs at 2 M screening each = 30 M, plus 3 final seeds × 5–10 M = 15–30 M → ≈ 45–60 M steps ≈ 10–25 h total. Use overnight runs with `caffeinate`.
- **Tiered option:** train D on the fast kinematic sim (doc 03) — 50–100× faster (inference; no measurement) — then evaluate in PyBullet; expect a sim-to-sim gap, so include PyBullet fine-tuning of 1 M steps. Recommended if PyBullet steps/s < 3k.
- Device: CPU for SB3 MLP (MPS slower for small nets; torch 2.14 reports `mps.is_available()=True` [V] but not recommended for 2×128 MLP). Set `torch.set_num_threads(1)` per worker.
- Discrete head: SB3 PPO supports `Discrete(11)` natively. `sb3-contrib` `MaskablePPO` supports invalid-action masking [V docs]; optional.
- Risk: reward hacking (hover forever), collapse to BRAKE/HOVER, latency mismatch; mitigation: step penalty, timeouts, latency-randomised training, evaluation of Z1/Z0.

---

## 8. Install friction summary (uv dry-run + binary install, Python 3.14, arm64, 2026-09-30) **[V]**

| Package | Result | Version | Note |
|---|---|---|---|
| numpy | wheel | 2.5.3 | |
| scipy | wheel | 1.18.1 | |
| osqp | wheel | 1.1.3 | used in bench |
| cvxpy | wheel | 1.9.3 | pulls clarabel 0.11.1 (Rust wheel ok) |
| casadi | wheel | 3.8.1 | IPOPT bundled |
| gymnasium | wheel | 1.3.0 | |
| stable-baselines3 | wheel | 2.9.0 | |
| torch | wheel | 2.14.0 | MPS available |
| pybullet | **no cp314 wheel** (`--only-binary` fails; only ancient cp35m wheels) | 3.2.7 (sdist) | build from source; README macOS flag `CFLAGS="-Dfdopen=fdopen"`; not attempted by me (doc 02) |
| gym-pybullet-drones | not on PyPI | git only | README says Python 3.12 |
| acados_template | not on PyPI | - | reject |

(The `dry-run` alone reported pybullet "Would install 1 package" because it is resolving the sdist; the `--only-binary` attempt was the discriminating test.)

---

## 9. Metrics per arm (minimum set; doc 08 has definitions)
`collision_rate, success_rate, timeout_rate, time_to_target, path_length, recovery_time (after each event), min_clearance, near_miss_rate, energy, action_switch_rate (oscillation), invalid_rate, intrinsic_decision_latency (p50/p95/p99), decisions_per_sec, sim_speed, mpc_infeasible_rate, solver_iterations`.

---

## Decisions recommended

1. Adopt arms A (plus A′ = A+shield), B1 (primary; B2 APF, B3 VFH, B4 ORCA as ablations), C (OSQP linear MPC N=20, dt=0.1), D (PPO, discrete 11), E (FSM as specified), F, G, plus Z0–Z3, O and O2. Keep shield (CBF-QP) common to all arms with a `--no-shield` ablation.
2. MPC: raw OSQP on a double-integrator with linearised half-space obstacles; report solve time as measured and inject it as delay; reject acados; cvxpy only for prototypes; CasADi+IPOPT only as the "nonconvex, slow" ablation.
3. FSM thresholds all derived from `d_stop = v(t_lat+t_ctl)+v²/(2 a_brk)+r+margin` with measured `a_brk` (spike with doc 11); `a_brk_design = 0.7×measured`.
4. RL: SB3 PPO, 8 envs, 5 M steps final (cap 10 M), 3 seeds, latency-randomised training, tiered training on the kinematic sim if PyBullet is < 3k steps/s. Budget ≈ 10–25 h laptop time; decide early whether that is acceptable.
5. Tuning parity: three-currency budget (effort, evaluations, compute), disjoint train/tune/test seed ranges, frozen configs hashed in the manifest, publish sensitivity to tuning effort.
6. Give *every* arm the injected-latency sweep so speed and smartness can be separated without the LLM.
7. Oracle O with ground-truth and zero latency, using the doc 06 feasibility guarantee; validate that O is ≥ 99% collision-free on feasible scenarios.

## Open questions

1. Measured `a_brk`, `a_lat`, `v_max` of the CF2X model in gym-pybullet-drones under DSLPIDControl velocity targets (drives every threshold). Owner: 11/02.
2. Will pybullet build on 3.14.2 (clang, `-Dfdopen=fdopen`)? If not, arms must run across a process boundary from 3.12 (RL training too; SB3/torch also need to be installed there).
3. Decision rate: 10 vs 20 Hz for E/D, and 48 Hz vs 30 Hz control? Changes ticks-of-delay mapping. Owner 08.
4. Should D see the same action mask as E? (Chosen: no.) Does the action set suffice for moving targets/yaw (gaps doc 11)?
5. Does the intruder drone act cooperatively (ORCA valid) or adversarially? Owner 06.
6. Jev identity and its observation format (owner 01): if Jev is not a general text model, arms F/G may need different projections.
7. RL budget plausibility: my 5 M figure is inference without a directly comparable published result.

## Follow-up spikes (small, runnable experiments)

1. **Braking spike:** in a PyBullet env (or point-mass), command 3 m/s then BRAKE; measure stopping distance/time vs `v²/(2a)`; fit `a_brk`, `t_ctl`. 30 min.
2. **OSQP-in-loop:** run `mpc_bench.py` logic with real PyBullet states (DSLPIDControl beneath), N=20 vs 40, 1–5 obstacles; record p50/p95/p99 solve times and closed-loop tracking. 1 h.
3. **FSM smoke test:** implement §4 on the kinematic point-mass with the doc 06 obstacle scenario; expect collision < 2% on feasible scenarios, and plot collision-vs-latency (this is also mandatory spike 3 for doc 08). 2 h.
4. **PPO pilot:** 500k steps on the kinematic sim with discrete head, latency-randomised; check Z0/Z1 sanity and learning curve; extrapolate steps needed for ≥ 90% of FSM performance. 1–2 h wall.
5. **pybullet build test** in scratch: `CFLAGS="-Dfdopen=fdopen" uv pip install pybullet --no-binary pybullet` on 3.14.2 and 3.12; record success/time. 15 min.
6. **DPP-compliant cvxpy rewrite** to see whether the cvxpy wrapper can get < 5 ms; purely informational (OSQP raw is the chosen path). 30 min.

## Sources (accessed 2026-09-30)
- gym-pybullet-drones README (Python 3.12, "Tested on ... Apple Silicon/macOS 26", pybullet macOS CFLAGS): https://github.com/utiasDSL/gym-pybullet-drones
- `examples/learn.py` (PPO 1e7 steps, 1 env): https://raw.githubusercontent.com/utiasDSL/gym-pybullet-drones/main/gym_pybullet_drones/examples/learn.py
- Multi-Task RL for Quadrotors (20–40M steps reference): https://arxiv.org/html/2412.12442
- Sample Efficient Learning of Path Following and Obstacle Avoidance for Quadrotors: https://arxiv.org/pdf/1906.12082
- VFH / VFH+ background: https://en.wikipedia.org/wiki/Vector_Field_Histogram ; https://ieeexplore.ieee.org/abstract/document/677362
- sb3-contrib MaskablePPO: https://sb3-contrib.readthedocs.io/en/master/guide/examples.html
- Own measurements: `scratchpad/05/mpc_bench.py`, `scratchpad/05/ppo_fps.py` (Apple M4, uv 0.9.18, Python 3.14, numpy 2.5.3, osqp 1.1.3, casadi 3.8.1, cvxpy 1.9.3, torch 2.14.0, SB3 2.9.0).
- ORCA (van den Berg et al.), Khatib 1986 APF, Ames et al. CBF: cited from memory (not fetched this session), confidence M-H for content, unverified URLs.
