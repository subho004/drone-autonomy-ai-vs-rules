# 17b - Simulation ideas: engineering feasibility and ranking

Date: 2026-10-18. Companion to `17a-simulation-ideas-prior-art.md` (prior art, 20 ideas). This file asks a different question: **what can one developer build on top of this codebase, how long does it take, and which ideas make a good video and a falsifiable result.** Nothing in the repo was changed except this file. Spike scripts live in the session scratch directory (`.../scratchpad/17b`: `multi.py`, `head.py`, `yaw.py`, `track.py`, `cam.py`).

Tags: **[measured]** = ran in this session on this machine (Apple silicon, repo venv, gym-pybullet-drones 2.2.0, PyBullet 3.2.7). **[read]** = from reading source. **[web]** = from a search result I did not open in full (search snippet only; treat as unverified). **[inferred]** = my reasoning.

---

## 1. What the codebase gives us (and what it does not)

[read] unless tagged.

| Piece | State | Consequence for new ideas |
|---|---|---|
| `DroneWorld` (`app/sim/world.py`) | One drone, `CtrlAviary` subclass, 240 Hz physics / 48 Hz control, yaw target hard-wired to zero in `_advance` (`target_rpy=np.zeros(3)`), 16 horizontal rays in world axes, obstacles are spheres parked at z=-50 until a trigger, wind is a 2-D acceleration (step mean + OU turbulence) | Moving entities are easy (sphere + velocity already exists). Non-sphere geometry (walls, buildings, zones) needs new code in three places: body creation, `clearance()`, `_step_info()` |
| Executor (`app/controllers/executor.py`) | Carrot with latched x/y/z channels; lateral step 1.0 or 1.2 m at 1.5 m/s; lateral accel 5 m/s2; forward accel 3 and brake 4.5 m/s2 | No backward motion, no yaw, no continuous speed command. Everything is "go forward, nudge sideways" |
| Flight loop (`app/harness/flight_runner.py`) | Provider called synchronously ("pause and charge"), latency converted to ticks, applied at `t + ceil(L/dt)`. `sequential` (one request in flight) and `periodic` (new request every `PERIODIC_TICKS = 5`, constant in code, stale results dropped) | Pipelining is half built. The observation is built once per decision, so any "prediction" or "age" feature goes in `_issue()` |
| Observation / prompt | `decision_age_ms` is encoded as `age_ms` and the system prompt says "age = milliseconds since the data was measured" | **Finding: nothing ever sets it.** `_issue()` calls `w.observe(int(self.current_action))`, so every live prompt said `age_ms=0`. The only non-zero values are in the few-shot examples (`prompts.py` lines 82, 100). The models were told their data was fresh in every run |
| Mission layer | `MissionTracker` + stations + tolerance boxes + pad; `extras` dict flows into the prompt via `_mission_lines()` | Any new task (follow, deliver, land) can ride on `extras` and a tracker class without touching the provider layer |
| Reflex shield | Forward-corridor brake from rays only | A geofence or lateral shield is a sibling class |
| Storage / dashboard / render | Metrics dict -> SQLite; trajectory Parquet with a fixed `TRAJ_COLUMNS` tuple; `MissionScene` draws spheres, rings, pad, drone at 4x | New entities need new trajectory columns (or a JSON side column) and a scene primitive. `TRAJ_COLUMNS` is a contract used by the dashboard, so add, do not rename |
| Cadence in the grid | `cadence` is already a list in `ExperimentConfig` and a column in the `flights` UNIQUE key | A cadence sweep needs no schema change |
| Response cache | Replays the stored decision with its original latency | Re-running a prompt is free and deterministic, which helps any prompt-ablation idea |

Measured numbers from the brief held: 6.5k control ticks/s for the stock single-drone loop in my multi-drone test, 6.6k with `DroneWorld.step` [measured].

---

## 2. Spikes run for this study [measured]

All with `.venv/bin/python`, `PYTHONPATH` = repo root, headless.

### 2.1 Multi-drone (`multi.py`, `head.py`)

`CtrlAviary(num_drones=N)` plus one `DSLPIDControl` per drone, 480 ticks, drones in a line 0.8 m apart flying +x at 1.2 m/s.

| N | Physics.PYB ticks/s | drone-ticks/s | Physics.PYB_DW ticks/s | drone-ticks/s |
|---|---|---|---|---|
| 1 | 6524 | 6524 | 5907 | 5907 |
| 2 | 3705 | 7410 | 2813 | 5626 |
| 4 | 1862 | 7449 | 1226 | 4903 |
| 8 | 893 | 7140 | 584 | 4676 |

- Cost is linear in drone count (about 7.4k drone-ticks/s with plain `PYB`). Two drones run at about 3.7k ticks/s, so the 48 Hz sim runs 77x real time. Throughput of the experiment grid halves for two drones, which is acceptable.
- **`PYB_DW` (downwash model) crashed one drone in every N >= 2 run** (final z 0.01 to 0.06 m) in this side-by-side, same-altitude layout. I did not isolate the cause (spacing, the downwash coefficients, or my layout). **Use plain `PYB` and treat downwash as out of scope until a spike explains it.**
- Drone-drone contact works in plain `PYB`: head-on pair at 1.2 m/s each, first `getContactPoints` hit at t = 2.44 s with centre separation 0.122 m (i.e. at body contact; both drones then crashed, z about 0.01). So collision detection between drones needs no new code beyond looping over pairs.
- Not tested: a multi-drone `DroneWorld` with rays that see the other drones, or the harness with two providers. Those are the real work (section 4, idea 11).

### 2.2 Yaw (`yaw.py`)

Reproduced `DroneWorld._advance` with a yaw target ramp while flying 10 s at 1.2 m/s.

| yaw rate | result |
|---|---|
| 0, 30, 90, 180, 360 deg/s | no crash; max tilt 0.36 to 0.44 rad; position error max 0.15 to 0.19 m; actual yaw tracks the ramp (e.g. at 90 deg/s target 900 deg, actual 173 deg = 900 mod 360 minus 7 deg lag) |

**A yaw-capable executor is feasible.** `DSLPIDControl` accepts `target_rpy[2]` as an unbounded angle and tracks it without upsetting translation. What it costs: rays and the mission frame are in world axes ("body frame equals world frame" is an assumption in `observation.py`, `encoding.py` bearings, `labeller.py`, `arms.py::_free_side`). A yaw action would force body-frame rays (rotate `dirs` in `sense()` by yaw), a yaw state in `Observation`, and relabelling of the oracle. About 4 to 5 days including tests. I would **not** do it unless an idea needs it (only gates/racing and camera pointing do).

Same script, aggressive lateral reference (sinusoid added to the setpoint, no acceleration limit):

| lateral sine at 0.5 Hz | peak lateral accel | result |
|---|---|---|
| 0.3 m amplitude | 3.0 m/s2 | ok, tilt 0.65 rad |
| 0.6 m | 5.9 m/s2 | **crash at t = 2.3 s** |
| 1.0 m | 9.9 m/s2 | **crash at t = 0.8 s** |

This reproduces "the stock PID becomes unstable with fast repeated lateral moves" and gives a number: **keep commanded lateral acceleration under about 3 m/s2** for any new scenario. Forward-only speeds of 2, 3 and 4 m/s were fine (position error 0.17 m), so forward speed is not the limit; lateral agility is.

### 2.3 Moving-target tracking (`track.py`)

Setpoint moves on a circle (speed v, radius R) with feed-forward velocity; the drone is given the target position delayed by L.

| v (m/s), R (m) | centripetal accel | tracking error at L = 0 / 0.1 / 0.3 / 0.6 / 1.0 s | v*L |
|---|---|---|---|
| 1.0, 3 | 0.33 | 0.03 / 0.08 / 0.28 / 0.58 / 0.97 m | 0 / .1 / .3 / .6 / 1.0 |
| 2.0, 4 | 1.0 | 0.07 / 0.18 / 0.58 / 1.17 / 1.96 | 0 / .2 / .6 / 1.2 / 2.0 |
| 3.0, 5 | 1.8 | crash / 0.30 / 0.89 / 1.78 / crash | .3 / .9 / 1.8 |

Mean steady-state error equals v*L to within about 5 percent. The PID itself tracks a 2 m/s curved path to 0.07 m, so **in a tracking task the whole error budget is decision/observation staleness, not controller lag.** At 3 m/s on a 5 m circle the run is borderline (the L=0 and L=1.0 cases crashed, likely at the start-up jump; not investigated). Target speeds up to 2 m/s and accelerations up to 1 m/s2 are safe.

### 2.4 Camera cost (`cam.py`)

PyBullet `getCameraImage`, tiny renderer, scene with plane and six box "buildings":

| resolution | ms per frame |
|---|---|
| 64x64 | 0.7 |
| 128x128 | 1.9 |
| 224x224 | 5.0 |
| 320x240 | 8.1 |
| 640x480 | 37.4 |

The brief's "10 to 60 ms per frame" matches 320x240 to 640x480 plus encoding. **Rendering is never the bottleneck**; model latency is. At 224x224 the render is 5 ms, which would be 0.2 control ticks. Camera cost matters only for rollout throughput at 640x480.

### 2.5 State save/restore and rollouts (`cam.py`)

`p.saveState` 0.22 ms, `p.restoreState` below 0.01 ms, a one-second (48-tick) rollout plus restore **7.4 ms**. So a hypothetical "what happens in the next second if I do X" lookahead costs about one real-time second per 135 forks. Caveat: PyBullet state restore does not restore Python-side state (the PID integrator, the executor carrot, `DroneWorld.tick`, cached `env.pos`); a rollout needs a `copy.deepcopy` of those (cheap) [inferred; not tested end to end].

### 2.6 Things I did not test

Per-step cost of a box-obstacle world; ray cost with many bodies; the web dashboard with two drones; any VLM call. These are listed as spikes in section 8.

---

## 3. Ranking

Scores 1 to 5 (5 = best). **Value** = research value for the decision-latency question. **Demo** = video appeal. **Fit** = how little new machinery it needs and how low the instability/feasibility risk. **Cost** inverted: 5 = under 2 days. Total = sum of the four with Value counted double (max 25).

| # | Idea | Days | Value | Demo | Fit | Cost | Score |
|---|---|---|---|---|---|---|---|
| 1 | Latency-aware prompting + fix the always-zero `age_ms` | 1 | 5 | 2 | 5 | 5 | 22 |
| 2 | Pipelined (periodic) cadence sweep | 2 | 5 | 3 | 5 | 5 | 23 |
| 3 | Latency tails and jitter at equal mean | 1.5 | 5 | 2 | 5 | 5 | 22 |
| 4 | Reactive intruder using proportional navigation | 3 | 4 | 4 | 4 | 4 | 20 |
| 5 | Moving-target following | 5 | 5 | 5 | 4 | 3 | 22 |
| 6 | Package delivery with restricted zones (graded outcome) | 4 | 5 | 4 | 4 | 3 | 21 |
| 7 | Speculative / predictive-state decisions | 4 | 5 | 4 | 4 | 3 | 21 |
| 8 | Landing-site selection under a battery deadline | 3.5 | 4 | 4 | 4 | 4 | 20 |
| 9 | Hybrid three-tier (slow planner + fast model + reflex) | 6 | 5 | 4 | 3 | 2 | 19 |
| 10 | Human reaction-time baseline (synthetic, then playable) | 0.5 / 5 | 4 | 4 | 5 | 5 | 22 |
| 11 | Two-drone shared airspace | 7 | 4 | 5 | 2 | 2 | 17 |
| 12 | Convoy / formation following | 4 (after 11) | 3 | 5 | 2 | 3 | 16 |
| 13 | Gate slalom (fly-through stations) | 3 | 3 | 4 | 3 | 4 | 17 |
| 14 | Corridors and buildings with gusts | 4 | 3 | 4 | 3 | 3 | 16 |
| 15 | Structure inspection with camera + VLM | 6 | 3 | 5 | 2 | 2 | 15 |
| 16 | Sensor-fault diagnosis (stuck ray, phantom, drift) | 3.5 | 3 | 2 | 4 | 4 | 16 |
| 17 | Area coverage with limited battery | 5 | 2 | 3 | 3 | 2 | 12 |

(Numbers 2, 3 and 10 reorder to a different top-5 below because of dependencies and demo needs; the table score is a rough guide, not the build order.)

---

## 4. The ideas

Format: **Build** (files), **Days**, **Video**, **Question + prediction**, **New sensing/actions**, **Risks**, **Smallest first experiment**.

Effort days are solo-developer days including unit tests (the repo has about 2.8k lines of tests and a green suite as a convention) but excluding live API runs and write-up.

### Idea 1 - Latency-aware prompting, and fix the always-zero `age_ms`

- **Build.** `app/harness/flight_runner.py::_issue` passes `decision_age_ms` to `world.observe`. Define it as the **expected** staleness at application: the previous decision's measured latency (sequential) or a running median. Add prompt variants in `app/providers/prompts.py`: (a) status quo (age=0, a latent bug), (b) true age, (c) age plus a sentence "your action takes effect about N ms after the data; act on where things will be, not where they are", (d) a derived `ttc_at_apply` field (ttc minus age). Add `prompt_variant` to the arm config. The cache already makes repeat runs free.
- **Days.** 1.
- **Video.** Weak by itself. Good as an overlay in the live-results video: same model, same scenario, with and without "you are 600 ms late".
- **Question and prediction.** Does telling the model about its latency reduce collisions? **Prediction (falsifiable):** for the slow models (Gemini flash-lite, GPT-6 Luna), variant (d) reduces shield-off collisions on the 56 obstacle scenarios by at least 10 points (paired McNemar, same scenarios); for the reflex arm (which already has `t_lat_s` baked in) the effect is within noise. If variants (b) and (c) change nothing but (d) helps, the lesson is "give the model arithmetic done, not a number to reason about", consistent with the project's earlier finding that small models obey flags and ignore raw numbers (`encoding.py` header).
- **New sensing/actions.** None.
- **Risks.** The age is only known after the fact in pipelined mode (use the median). Mild: prompt changes shift the baseline numbers; keep (a) as the labelled control.
- **Smallest first experiment (no API cost).** In the existing FSM/reflex arms, `ArmParams.t_lat_s` is the latency the arm assumes (design 0.10 s). Run the reflex arm under fixed injected latency 0.1, 0.3, 0.6 s with `t_lat_s` = 0.1 (mismatched) versus `t_lat_s` = true. This is the classical upper bound on what "knowing your latency" can buy. One config, a few minutes of compute.

### Idea 2 - Pipelined (periodic) cadence sweep

- **Build.** Replace the constant `PERIODIC_TICKS` with a `FlightConfig.period_ticks` and allow cadence strings such as `periodic:3`, `periodic:5`, `periodic:10` (`ExperimentConfig.cadence` is already a list). Add `max_inflight` and a stale policy (`drop_older`, `apply_all`). Record in-flight count per tick. HUD already draws an in-flight bar; draw several overlapping bars.
- **Days.** 2.
- **Video.** A timeline strip showing overlapping requests and which result the drone obeys; side by side with the sequential run on the same scenario.
- **Question and prediction.** With latency L, how much does issuing every Delta ticks help, and where does it stop paying? A decision is never fresher than L, but the *interval between fresh decisions* falls from L (sequential) to Delta. **Prediction:** mean time from event onset to first evasive action falls by about 0.5 (L - Delta) (event phase is uniform within the update interval), so at L = 0.6 s and Delta = 0.1 s the reaction improves by about 0.25 s; collision rate versus Delta flattens below Delta of about L/4; API calls (cost) grow as L/Delta, so the cost-per-avoided-collision has an interior optimum. Falsified if collisions fall with no flattening, or do not fall at all (e.g. because stale decisions arriving out of order cause flip-flopping; `action_switches` already measures that).
- **New sensing/actions.** None.
- **Risks.** Out-of-order and stale-drop semantics already exist (`_apply_due` drops results older than the last applied observation). Flip-flop between AVOID_LEFT and AVOID_RIGHT in successive results may destabilise the PID (the executor has `SIDE_COMMIT_S` hysteresis for LEFT/RIGHT but AVOID_* bypasses it) [read].
- **Smallest first experiment.** FSM arm with FIXED latency 600 ms, cadence sequential versus periodic:2/5/10 on the 56-scenario suite with `mix` of static, approach, cross. Pure compute, no API. Then repeat once with Jev (about $0.05 at 5 calls/s; the live run cost $0.45 for 6.9k calls, so a 5x call rate on 60 flights is around $0.30).

### Idea 3 - Latency tails and jitter at equal mean

- **Build.** `LatencyMode.SAMPLED` already draws from a recorded trace. Add trace files from the live run's per-call latencies (16c probe JSON and the DB) and synthetic ones (constant, lognormal with matched mean, bimodal with a 5 percent 3 s timeout). A new analysis cut in `app/analysis/summary.py`: collisions versus p99 at fixed mean.
- **Days.** 1.5.
- **Video.** Histogram of the model's latency beside a drone that is usually fine and occasionally coasts into an obstacle during a slow call.
- **Question and prediction.** Is p50 the wrong number to report? **Prediction:** at equal mean latency, collisions rise monotonically with p99 (heavy-tail trace at least 10 points worse than a constant trace at 600 ms, on the same scenarios); the 3 s timeout trace is worst of all. Falsified if collision depends only on the mean.
- **New sensing/actions.** None.
- **Risks.** Low. Statistical power: each cell needs the same paired scenarios (the harness already pairs by `scenario_id`).
- **Smallest first experiment.** FSM arm, three traces with mean 600 ms (constant, lognormal sigma 0.5, 90 percent 300 ms + 10 percent 3.3 s), 56 scenarios, 3 repetitions of the trace seed.

### Idea 4 - Reactive intruder using proportional navigation

- **Build.** New `ObstacleSpec.kind = "pursue"` with `speed`, `max_accel` and `nav_gain` in `app/scenarios/spec.py`; velocity update in `DroneWorld._update_events` (it already integrates `o.pos` each tick and moves the body). Pursuer guidance: lead/pure pursuit then **proportional navigation**, `a_cmd = N * V_c * dλ/dt` on the line-of-sight rate, clamped to `max_accel`. The generator's oracle certification (`ScenarioGenerator`, label rollouts) must treat the obstacle as moving with unknown future, so certify by flying the zero-latency oracle against it (that mechanism exists for missions). Add `intruder_pursue` to `EventType` and to the `OBSTACLE_EVENTS` tuple in `flight_runner.py`.
- **Days.** 3.
- **Video.** A red drone-sized sphere that tracks the Crazyflie and curves into its path; the drone's dodge, then the intruder correcting. Easy to render (already a sphere).
- **Question and prediction.** A constant-velocity intruder can be dodged by moving out of its line; a pursuer that corrects makes late dodges useless. **Prediction:** the critical latency at which success falls below 50 percent is at least 2x lower (earlier) for the PN pursuer than for the existing constant-velocity `approach` intruder at the same speed; and a lead-aware arm (extrapolating the LOS rate) beats the plain TTC reflex arm at L >= 300 ms. Falsified if the two intruder types give the same latency curve.
- **New sensing/actions.** None. The ray ring already sees spheres; `closing_speed` and `ttc` are derived. Optionally expose bearing rate as an extra.
- **Risks.** A pursuer faster than the drone that always hits is infeasible: certify per scenario and keep pursuer speed at or below drone speed with `max_accel` 1.5 m/s2 (at 1.5 m/s cruise a lateral escape of 1.2 m takes about 1 s, so the pursuer needs a turn-rate limit to leave an escape). The lateral-instability limit from section 2.2 still caps how fast the drone can dodge. Delay-sensitivity of PN is documented for missiles [web: Dhananjay, Lum, Xu on delayed LOS rate, IEEE TCST 2013, via https://publications.iitm.ac.in/ search snippet]; no quadrotor-specific analysis found.
- **Smallest first experiment.** Add the pursuer, fly the oracle (0 ms) and the FSM at 0, 200, 400, 800 ms on 30 seeds; plot success versus L next to the existing `approach` scenario.

### Idea 5 - Moving-target following

- **Build.** `TargetSpec` in `spec.py` (path family: line with weave, circle, stop-and-go, figure-eight; speed 0.5 to 2 m/s; accel cap 1 m/s2 from section 2.3). New `app/sim/target.py` (pure, like `mission.py`) advancing the target and computing lock metrics. In `world.py`, either a visual-only body or reuse `_Obstacle` with collision disabled. `Observation.extras`: `tgt_dx`, `tgt_dy`, `tgt_dvx`, `tgt_dvy`, `lock`. `encoding.py::_mission_lines` equivalent for a target block. New `FollowTracker` next to `MissionTracker`. Metrics in `_metrics()`: mean and p95 tracking error, fraction of time inside the desired range band (1 to 3 m), time to lose lock (more than 5 m), `trail` of target for the renderer. Scene: a ground vehicle or sphere on a pole in `scene.py`.
- **Days.** 5.
- **Video.** The strongest simple story: "follow-me" with a wobbling drone trailing a target that weaves; the same chase at 0 ms, 300 ms and 1 s; the lag distance drawn as a line.
- **Question and prediction.** How does tracking error scale with decision latency? Section 2.3 gives the continuous-control floor: error = v_target * L. **Prediction:** with the 11 discrete actions the slope is larger than 1: mean tracking error is at least 1.5 x v_target x L plus a floor from the 1 m LEFT/RIGHT quantisation (about 0.3 to 0.5 m); the lost-lock rate crosses 50 percent where v_target x L approaches half the allowed range band (for a 2 m band and v = 1 m/s, L about 1 s). DroneCATS reports moving targets are lost and blames latency but has no sweep [per 17a, verified there].
- **New sensing/actions.** Target-relative state as above. Actions: the target must stay ahead, so use `FORWARD` (cruise), `BRAKE`/`HOVER` (slow to match), `LEFT`/`RIGHT`. **No backward action**: restrict targets to net forward motion with lateral weave (this also keeps `RETURN_HOME` semantics clean) or accept that overrunning the target is a failure mode. A `SLOW` or speed-match action would be a better fit but changes the 11-action contract in `actions.py`, the glossary, the prompts, parsers, DB codes and every comparison; I would add it only as a second action-set version.
- **Risks.** (1) Discrete steps cannot follow a target with y-velocity above about 0.5 m/s; keep lateral weave amplitude 1 m at 0.15 Hz (peak accel 0.9 m/s2) [inferred from section 2.2]. (2) Velocity is a prompt number; models are weak at raw numbers, so bucket it ("target pulling away", "target closing") like the mission block does. (3) Certification: use the zero-latency tracking arm for feasibility.
- **Smallest first experiment.** No new action. A scripted `TargetArm` (proportional controller on `tgt_dx/tgt_dy` choosing FORWARD/BRAKE/LEFT/RIGHT) against a 1 m/s weaving target at injected latency 0, 0.2, 0.5, 1.0 s; plot error versus L and compare to the v*L line from `track.py`. Two days of work gets this plot; the rest of the 5 days is rendering, prompts and LLM runs.

### Idea 6 - Package delivery with restricted zones (graded outcome)

- **Build.** `ZoneSpec` (axis-aligned box or cylinder, no-fly) in `MissionSpec`; `ZoneTracker` in `app/sim/mission.py` computing signed distance and **penetration depth** each tick; zones are not physical (rays do not see them), so no change to `sense()`. Extras: `zone_dist`, `zone_bearing_bucket`, `zone_state` (CLEAR/NEAR/INSIDE). Metrics: `max_breach_m`, `breach_time_s`, `breach_count`. Optional `GeofenceShield` beside `ReflexShield`: brakes forward motion when predicted stop point is inside a zone. Render: translucent red cylinders in `scene.py`.
- **Days.** 4.
- **Video.** Delivery drone threading between red no-fly zones to a pad, red glow and a breach meter when it clips a zone; compare "Jev" with and without the geofence layer.
- **Question and prediction.** Same reaction question as obstacle avoidance but with a **graded** outcome (penetration depth in metres) instead of a binary collision, so it has far higher statistical power at n = 12 to 60. **Prediction:** maximum breach depth rises linearly with decision latency with slope about equal to cruise speed (1.2 m/s in missions) over the range 0 to 1 s, i.e. a breach of about v*L plus braking distance (about v^2 / (2 * 4.5) = 0.16 m at 1.2 m/s); with the geofence shield on, the slope drops below 0.3 m/s. Falsified if breach depth is flat or if the shield does not change the slope. Aviation geofence runtime-assurance work [web: https://uh-ir.tdl.org/items/b7c29e04-4c39-4f0b-a530-3063b1e39974/full via 17a] motivates a shield.
- **New sensing/actions.** Extras only; same 11 actions.
- **Risks.** Low physics risk (no new bodies). Design risk: zone placement must leave a feasible path for a competent pilot; certify with `MissionFsmArm` extended to be zone-aware, exactly like current mission certification. The earlier finding that "the shield braked while the pilot was approaching a station" (MISSION-DEMO) is a warning: a geofence shield needs a zone-aware release.
- **Smallest first experiment.** One zone placed across the straight line to the pad; the existing `MissionFsmArm` plus one extra rule; breach depth versus injected L = 0, 0.3, 0.6, 1.2 s over 30 seeds.

### Idea 7 - Speculative and predictive-state decisions

- **Build.** In `flight_runner._issue`, before calling the provider, replace the observation with its **predicted state at apply time**: constant-velocity extrapolation of the obstacle and drone (ranges reduced by `closing * L_expected`), or a one-second physics forward-roll using `p.saveState` / `restoreState` (section 2.5; deepcopy executor and PID first). New `app/providers/predictive.py` wrapper provider ("prediction-compensated"), plus a **speculative** variant: issue requests for the state predicted under the top two actions in parallel and keep the one whose predicted state matches the realised state (hit rate is a reported metric), in the spirit of speculative actions for agents [web: https://arxiv.org/html/2510.04371v1 and https://iclr.cc/virtual/2026/poster/10009726; ICLR 2026, reported 55 percent next-action prediction accuracy and up to 20 percent latency reduction in games and web tasks]. The response cache in `app/providers/cache.py` can be keyed on the quantised predicted observation to turn repeated situations into zero-latency hits, but note it replays the original latency by design, so a speculative-cache variant needs a different mode.
- **Days.** 4 (constant-velocity extrapolation: 1 day; physics rollout: 2 more; speculative branches and metrics: 1).
- **Video.** A translucent "ghost" drone and ghost obstacle drawn where the model *thinks* things will be when its answer lands, beside the real ones.
- **Question and prediction.** Can the latency penalty be engineered away instead of waited out? **Prediction:** for constant-velocity intruders (the existing static/approach/cross kinds) extrapolating by the expected latency recovers at least 50 percent of the collision gap between measured-latency and zero-latency for L <= 600 ms; for the PN pursuer of idea 4 and any manoeuvring obstacle it recovers under 25 percent. Falsified if extrapolation helps equally against a manoeuvring intruder (then the model is not using the lead the way I assume).
- **New sensing/actions.** None; the extrapolated numbers go into the existing fields. Needs obstacle velocity estimated from rays (the `closing_speed` filter exists; lateral rate of a crossing obstacle does not exist - add from consecutive ray hits).
- **Risks.** Constant-velocity extrapolation of rays is easy only along the closing axis. Physics rollout requires copying Python state (the PID integrator and executor); not tested. Speculation doubles API cost.
- **Smallest first experiment.** FSM/reflex arm with injected L; compare feeding `fwd_range - closing*L` and `ttc - L` versus raw. This is almost idea 1's no-cost experiment from the other direction; run them together.

### Idea 8 - Landing-site selection under a battery deadline

- **Build.** `SiteSpec` list in `ScenarioSpec`: candidate landing sites with a ground hazard flag (people, water, slope), distance and size. `ScenarioSpec.sensors.battery_*` and `battery_death_after_s` already provide the clock (motors fade 8 s after CRITICAL; `rpm_scale` ramps down). Extras: nearest 3 sites as (dx, dy, hazard bucket, size bucket) - same bucketed style as the mission block. Outcome: `landed_safe`, `landed_hazard`, `crash`, `too_late`. Add a `SiteTracker` and a ground-patch renderer.
- **Days.** 3.5.
- **Video.** Engine-out style: battery bar falls, three green/red patches, the drone commits to one. Compare a fast model that picks the nearest patch with a slow model that picks the best patch but arrives late.
- **Question and prediction.** The project's central finding is that speed beat smartness. This is the experiment where smartness *should* win until latency eats the slack. **Prediction:** there is a crossover latency L* (a few hundred ms to about 1.5 s depending on the deadline) above which the fast "nearest site" policy has a higher safe-landing rate than a slower but better-informed policy; L* shifts to higher values as the deadline slack grows. Falsified if the better-informed model wins at every L tested or the fast one wins at every L (then there is no tradeoff in this action set).
- **New sensing/actions.** Site block in extras. Actions: existing `LEFT/RIGHT/FORWARD/BRAKE/EMERGENCY_LAND`. `EMERGENCY_LAND` is irreversible, which is exactly the commit point to analyse. Landing sites behind the drone are unreachable (no backward action); place sites ahead.
- **Risks.** Feasibility: certify with the oracle choosing the best reachable site at 0 ms. Battery-critical already has certification infrastructure (`deadline_s`).
- **Smallest first experiment.** Scripted two-policy comparison ("nearest" vs "best reachable") with injected latency 0 to 2 s on 40 seeds; find L* before involving any LLM.

### Idea 9 - Hybrid three-tier architecture (slow planner, fast decision model, reflex)

- **Build.** Multi-rate providers in the flight loop: today one provider and one `pending` list. Add `PlannerStream` (own provider, own period 2 to 5 s, own pending queue) writing a `PlanHint` (preferred side, speed cap, next-station tag, "danger ahead" flag) into `Observation.extras`, consumed by the fast model's prompt and by the reflex shield's threshold. Add `tiers` in the config: `[reflex]`, `[fast+reflex]`, `[planner+fast+reflex]`, `[planner+reflex]`. Cost accounting per tier. Related designs: asynchronous fast-slow VLA policies where the fast side no longer waits for the slow side [web: https://arxiv.org/pdf/2512.20188], Fast-in-Slow [web: https://arxiv.org/pdf/2506.01953], and RealtimeGym's AgileThinker that runs reactive and planning agents in parallel [per 17a, verified there: https://arxiv.org/html/2511.04898v1].
- **Days.** 6.
- **Video.** A three-lane timeline (planner / fast / reflex) with the drone, showing who is in control at each moment (the HUD already shows the current action and a request bar).
- **Question and prediction.** Which tier carries which metric? **Prediction (ablation):** (a) collisions: the reflex tier removes most of the gap (as the live data suggested: with shield on, models indistinguishable, p >= 0.3); adding a planner changes collisions by less than 5 points; (b) mission completion: planner+fast+reflex completes at least 2x more missions than fast+reflex alone on the existing 12 mission seeds (5/12 for mercury+shield today), because the planner removes the "single bad decision" failure by pre-resolving station approach. Falsified if the planner adds nothing on missions, which would mean the failures are execution-rate, not planning.
- **New sensing/actions.** None for actions. Plan hints are a new extras block.
- **Risks.** Most open-ended idea; the prompt and the hint schema are design choices that can dominate results. Cost: two API streams. Keep the planner a rule-based script first (a "perfect planner" upper bound), then replace by an LLM.
- **Smallest first experiment.** Rule-based planner at 3 s latency feeding `MissionFsmArm` run at injected 600 ms, with and without the shield; no API.

### Idea 10 - Human reaction-time baseline

- **Build (synthetic, 0.5 day).** A `SAMPLED` trace of human-like response times (lognormal, median 250 to 350 ms for a simple-reaction task and 400 to 500 ms for a 2-to-11-way choice, using Hick-style growth with log2(n) [web: course-note summaries only; no peer-reviewed number verified; https://psychology.hanover.edu/classes/hfnotes2/tsld038.html]) applied to the reflex/FSM decision logic: "perfect decision rule, human delay". A `human_rt` mode in `LatencyPolicy` is a small addition (the class already supports traces).
- **Build (playable, +4.5 days).** A browser mini-game in `static/` using the existing FastAPI app: the sim runs at wall-clock pace (48 Hz, 77x faster than needed), the page streams position over WebSocket, keys map to the 11 actions, the real human latency is the delay.
- **Video.** "Human vs Jev vs reflex" on the same scenario, picture-in-picture, with the person's reaction time shown. Memorable and easy to grasp.
- **Question and prediction.** Where does a human sit on the latency curve? **Prediction:** a 250 ms human with the FSM decision rule collides at a rate between the 0 ms FSM (9 percent) and Jev at 600 ms (43 percent) - closer to the FSM, because latency (not the rule) dominates; real playable humans collide more than the synthetic human at the same median because of the choice among 11 actions and attention lapses (expect 2x). Falsified if the synthetic human is worse than Jev.
- **Risks.** The playable version brings pause/wall-clock mismatch and browser jitter (the measurement then includes network jitter); keep it a demo, not a measurement, or run the page locally.
- **Smallest first experiment.** The synthetic human, three traces (median 200, 300, 450 ms), FSM decision rule, 56 scenarios, zero API cost.

### Idea 11 - Two-drone shared airspace

- **Build.** The largest refactor: `DroneWorld` is single-drone throughout (`num_drones=1`, `self._env.pos[0]`, one `_ctrl`, `rays` ignoring other drones, `clearance` over obstacles only). New `MultiDroneWorld` (or generalise with a drone index) with per-drone `DSLPIDControl`, per-drone executor and provider, pairwise contact checks, rays that also hit other drones' bodies, other drones' states in `Observation.extras` (nearest other: range, bearing, closing speed). The flight loop becomes N interleaved decision streams with independent latency per drone. Storage: a `drone_id` column in decisions and trajectory (additive). Scenario: crossing lanes, head-on corridor, merge at a pad.
- **Days.** 7 (the spike in section 2.1 covers the physics; the harness and storage are the cost).
- **Video.** The best flagship: two coloured drones, one driven by Jev and one by a reflex or another model, negotiating a crossing; side-by-side camera.
- **Question and prediction.** (a) Does collision risk depend on the sum or on each drone's latency? **Prediction:** collision probability is a function of L_A + L_B, so (L, 0) and (L/2, L/2) give statistically equal rates; (b) symmetric controllers with equal latency produce mirror deadlock/oscillation (both choose the same free side) in at least 10 percent of head-on cases at L >= 400 ms. Falsified if the asymmetric case is significantly worse (the slow drone dominates). Robust MADER measured real multi-UAV collision safety under 0 to 100 ms message delay [web: https://arxiv.org/pdf/2209.13667 via 17a].
- **New sensing/actions.** Other-drone entries in the ray ring (free once other bodies are in the ray test) and an extras block.
- **Risks.** (1) `PYB_DW` crashed a drone in my spike - use `PYB`. (2) Run-time doubles. (3) The `Observation` and `TruthState` types are single-obstacle; the legacy "nearest obstacle" fields would be wrong for two drones; (4) the dashboard and report assume one trajectory per flight. (5) No yaw means drones cannot "face" each other; irrelevant for rays but confusing in video.
- **Smallest first experiment.** Two scripted drones with the reflex arm at independent injected latencies in a head-on corridor, no harness changes: write a 100-line standalone script in `scripts/` using the `multi.py` pattern; sweep (L_A, L_B) on a 5 by 5 grid and plot the collision heat map.

### Idea 12 - Convoy / formation following

- **Build.** On top of idea 11: a scripted leader on a path, followers choose actions to hold a longitudinal gap and a lateral offset (the follow-target extras of idea 5 applied to a drone). Metrics: per-rank gap error; amplification between ranks (string stability).
- **Days.** 4 after 11.
- **Video.** A chain of three drones weaving, with the gap errors growing down the chain at high latency.
- **Question and prediction.** Hypothesis, with no source checked: errors amplify down a chain when each follower reacts to the one in front with delay. **Prediction:** the third follower's peak gap error is at least 1.3x the first follower's at L >= 500 ms, and within 1.1x at L <= 100 ms. Falsified if amplification is absent (the 11-action quantisation could hide it).
- **Risks.** Depends on idea 11 and on idea 5's discrete-action following; quantised actions give limit cycles that mimic amplification, so run a continuous-control control group (scripted follower with the same delay).
- **Smallest first experiment.** After idea 11, a leader plus one follower with the scripted `TargetArm` of idea 5.

### Idea 13 - Gate slalom (fly-through stations)

- **Build.** A station with `dwell_s = 0` and no speed cap is a gate: pass inside a tolerance box without stopping. `MissionTracker` already tracks `tol_x/y/z` and `miss_margin`. Add `gate_mode` to `MissionSpec` and a pass test on plane crossing. Render: rings (already drawn for stations).
- **Days.** 3.
- **Video.** The drone threading rings at speed; misses glowing orange.
- **Question and prediction.** Pass rate versus latency at fixed cruise speed. **Prediction:** with tolerance half-width w, pass rate drops from near 100 percent to near 0 over L in [w/(2 v), 2w/v]: at v = 1.2 m/s and w = 0.5 m that is about 0.2 s to 0.8 s. Falsified if the threshold sits at the upper end only (lateral quantisation dominating).
- **New sensing/actions.** Same mission extras. True racing (3 m/s, tight rings) needs yaw, a continuous waypoint action, and a different PID; section 2.2 shows lateral agility is the wall (3 m/s2).
- **Risks.** With 1.0 m lateral steps and a 0.5 m half-width tolerance, the quantisation error is comparable to the ring itself; rings must be at least as wide as one lateral step or the pilot at 0 ms already fails certification. This is a "slalom at walking pace", not racing.
- **Smallest first experiment.** Existing `MissionFsmArm` with `dwell_s = 0` stations at injected L in {0, 0.2, 0.4, 0.8}.

### Idea 14 - Corridors and buildings with gusts

- **Build.** Box obstacles (`BoxSpec`) in the scenario; `world.py::_build_obstacles` creating `GEOM_BOX` bodies at scenario start (not parked), added to the ray-hit id set in `sense()`; `clearance()` using `p.getClosestPoints` for boxes; `_step_info` contact checks. Gust events: use the existing `WindSpec` step plus OU turbulence, plus a short high-amplitude gust burst.
- **Days.** 4.
- **Video.** An urban canyon with a gust shoving the drone toward a wall. Visually the most "real".
- **Question and prediction.** Does a wind gust make decision latency matter? Honest hypothesis: **the PID already rejects wind through position feedback, so decision latency should not matter for gust rejection**, only for steering choices. **Prediction:** collisions in a straight corridor with gusts up to 1.5 m/s2 are independent of L (slope within noise over 0 to 1 s); latency matters only when the corridor requires a decision (a branching corridor or a blocked passage). That is a useful negative control: it shows which scenarios actually test the decision layer. Falsified if collisions depend on L even with a pure straight corridor.
- **Risks.** Wind above about 2 m/s2 saturates the thrust margin and the PID (existing calibration retuned for wind at 1.5 m/s lateral); box clearance code is new and untested; many bodies slow rays (not measured).
- **Smallest first experiment.** One wall at y = 1.0 m, gust at t = 4 s of 1.5 m/s2 toward the wall, FSM at injected L in {0, 0.3, 0.6, 1.2}; confirm flatness before building the rest.

### Idea 15 - Structure inspection with camera + vision-language model

- **Build.** `app/sim/camera.py` rendering a forward (or fixed-side) frame with the tiny renderer (section 2.4: 2 ms at 128x128, 5 ms at 224x224); a structure with coloured defect patches visible only inside a range band; image attached by a new provider path (`app/providers/openai_chat.py`/`gemini.py` need image parts, prompts need a "what is on the panel" question); scoring of reported defects. Token cost: OpenAI low detail is reported as a flat 85 tokens and Gemini small images as about 258 tokens, but a forum thread reports 305 observed for 4o, so measure rather than assume [web: https://api.ifq.ai/blog/insights/image-detail-low-cost/, https://community.openai.com/t/consuming-more-tokens-than-expected-for-image-vision-gpt-4o/1097772; not official docs].
- **Days.** 6 for a station-only version (a mission station with a "read the panel" query at hold), 10+ for flying the inspection route from images.
- **Video.** Picture-in-picture of what the drone "sees", a defect list filling in, per-frame latency and cost ticker.
- **Question and prediction.** Is pixel resolution a latency tax? **Prediction:** defect recognition accuracy saturates at about 128x128 while response latency keeps rising with image size, so the cost-effective operating point is the smallest image that reaches the plateau; text-only state is faster than image+state by at least 100 ms for the same model. Falsified if accuracy keeps improving up to 224x224 or if image size does not change latency (API processing dominated by fixed overhead).
- **Risks.** Off the project's core story (flight under latency); BALROG-style results suggest images often lower VLM performance [web, via 17a]; image prompts break the response cache's cheap determinism (large payloads); privacy of nothing sensitive here, but each frame costs tokens.
- **Smallest first experiment.** Not flight: render 100 stills of a defect panel at 64/128/224 px, ask the three cheapest VLMs a yes/no question, record latency and accuracy. About $0.20 and one day; decides whether the rest is worth building.

### Idea 16 - Sensor-fault diagnosis (stuck ray, phantom obstacle, GPS drift)

- **Build.** In `world.sense()` add fault injectors: `stuck_ray` (a ray reports a fixed return), `phantom` (a ghost range appearing ahead for N seconds), `ray_bias`; GPS drift beyond the current freeze (`gps_loss_at_s`); a `diagnose` flag in `extras` only after the fact. Ground truth for the oracle label (`labeller.py`) must say the phantom is not real. New metrics: false-brake time, detection lag.
- **Days.** 3.5.
- **Video.** Low (nothing visually dramatic); useful as a graph.
- **Question and prediction.** What does a false alarm cost versus a missed real obstacle when latency is high? **Prediction:** at high L, a model that reacts to every phantom completes fewer scenarios than one that ignores a ray that disagrees with its neighbours, and the penalty grows with L because the model cannot revisit its decision before the drone has stopped or dodged; falsified if phantom reaction cost is independent of L. 17a ideas 8 and 13 give prior art (state-channel injection and false alarms).
- **New actions.** Adding a `REPORT_FAULT` output is outside the 11-action contract; score diagnosis via the chosen action instead.
- **Risks.** Labeller and certification need the truth/perception split; easy to introduce label noise.
- **Smallest first experiment.** A phantom at 2 m for 1 s, FSM and reflex arms, measure forward progress lost.

### Idea 17 - Area coverage with a limited battery

- **Build.** Lawnmower rows across a region (y offsets by LEFT/RIGHT, x by FORWARD), coverage grid in a tracker, battery drain raised to make the budget bind, RETURN_HOME at the point of no return. Coverage metric and a heatmap render.
- **Days.** 5.
- **Question and prediction.** A late return decision costs about v*L of extra distance, which at 0.2 percent/s drain and 1.2 m/s is a battery-percent error of drain * L, about 0.1 to 0.2 percent per second of latency: tiny. **Prediction:** coverage-at-landing is insensitive to L over 0 to 1 s (within noise) unless drain is raised by about 10x. Falsified if there is measurable sensitivity at the default drain. This is mostly a negative result; I rank it last because the effect is small unless the scenario is contrived, and 17a (idea 10) covers a sharper version.
- **Smallest first experiment.** Not worth building before ideas 1 to 8.

---

## 5. Cross-cutting engineering notes

1. **Add trajectory columns, do not rename.** `TRAJ_COLUMNS` is a contract (dashboard, report, renderer). Moving targets, zones and second drones need either extra columns or a JSON `entities` blob. A single `entities_json` per flight plus Parquet columns for `tx, ty, tz` is the least disruptive.
2. **Generalise obstacles once.** Ideas 4, 5, 8, 11, 14 all need entities that move or have shapes. A small `Entity` abstraction in `world.py` (kind, shape, motion policy, collidable) beats five special cases; `_Obstacle` is already close. Do it in idea 4/5, not later.
3. **Certification.** Every scenario family needs a zero-latency feasibility check. The mission certifier (`_mission_certifier` in `experiment.py`) is the pattern: fly the reference pilot at 0 ms and keep seeds that pass.
4. **Graded outcomes beat binary outcomes.** Breach depth (idea 6), tracking error (idea 5), site distance (idea 8) give continuous measures with far more power than collision yes/no at n = 12 to 60. The mission audition could not separate Jev and Mercury at n = 12.
5. **Lateral agility cap.** Section 2.2: do not command more than about 3 m/s2 lateral. Keep new scenario motion (targets, pursuers) inside this.
6. **Cost.** The live run cost $0.45 for 6.9k calls. A pipelined run issuing 5x more calls costs about 5x; still cents.
7. **No-yaw assumption is embedded** in observations, bearings, labels and `_free_side`. Yaw is feasible at the physics level (section 2.2) but expensive at the interface level; avoid it.
8. **Prompt bug to fix first.** The `age_ms` finding changes how the existing live results should be read: the models were told their data was current in every run, so their latency-induced misjudgment was not mitigated by the prompt. It does not invalidate the collision-versus-latency curve (injected latency is applied by the harness regardless) but it does mean idea 1's baseline is "uninformed".

---

## 6. Recommended top-5 build order

Rationale: cheapest decisive results first, then the two best demo stories, then the engineering-away-latency experiment, with the flagship multi-drone video held back because its refactor is large and its physics has one unexplained crash mode.

| Order | Package | Ideas | Days | Why now |
|---|---|---|---|---|
| 1 | **Prompt-honesty and cadence pack** | 1 + 2 + 3 | 4.5 | Fix `age_ms`, add prompt variants, parametrise cadence, add tail traces. All no-API-first, all reuse existing grid columns. Produces three falsifiable results and fixes a real flaw in the existing data |
| 2 | **Moving things** | 4 + 5 | 8 | Needs the generalised entity (note 2). The PN pursuer is cheap once moving entities exist; follow-the-target is the best single video. Section 2.3 already supplies the baseline curve |
| 3 | **Restricted zones** | 6 | 4 | Graded metric, no new physics, strong visual, extends missions |
| 4 | **Predictive and speculative decisions** | 7 | 4 | Tests whether latency can be mitigated rather than only measured; pairs naturally with idea 4 (a manoeuvring intruder is the counter-example). Visual "ghost drone" |
| 5 | **Human and landing-site baselines** | 10 (synthetic) + 8 | 4 | Half a day for the human baseline, 3.5 for landing-site selection; together they tell the "speed vs smartness vs human" story |

Total about 24.5 days. Stretch after that: idea 9 (hybrid tiers, 6 days) and idea 11 (two drones, 7 days; do the standalone heat-map script first).

---

## 7. Decisions recommended

1. Fix the `age_ms` wiring before any more live runs, and mark existing live results as "uninformed age" in `LIVE-RESULTS.md` (documentation change, outside this task).
2. Run `PYB`, never `PYB_DW`, for any multi-drone work until the crash in section 2.1 is explained.
3. Keep the 11-action contract. Handle follow, zones and landing within it; only fork an action-set version if the idea-5 spike shows quantisation dominating.
4. Prefer scenarios with a graded outcome metric.
5. Do not add yaw. If gates or camera pointing become a priority, treat it as a separate project (4 to 5 days plus relabelling).
6. Cap commanded lateral acceleration at 3 m/s2 in every new motion generator.
7. Build the generalised `Entity` first when starting idea 4/5, so zones, targets, buildings and drones share one mechanism.

## 8. Open questions

1. Why did `PYB_DW` crash one drone in every multi-drone run? (spacing 0.8 m at equal altitude; downwash coefficients; start-up transients.)
2. Does the 3 m/s circle-tracking crash in `track.py` come from the start-up jump (reference starts away from the drone) or from the dynamics? Only matters for fast targets.
3. How much does `getCameraImage` slow ray tests and stepping when many boxes are present?
4. Is the Jev endpoint's latency stable enough for pipelining at 5 requests/s (rate limits, free-tier throttling for the Mercury route)?
5. Will models act on `ttc_at_apply` (idea 1d) or ignore it, as they ignored raw `bat=5`? Only a live run tells.
6. Is the human reaction-time distribution used in idea 10 defensible? I did not find a peer-reviewed primary source in this session; the search returned lecture slides. Needs a cited study (simple visual RT and choice RT with Hick's law) before publication.
7. Image token costs: the numbers above are from secondary sources and conflict (85 vs about 305 tokens); check provider documentation.
8. A quadrotor-specific analysis of proportional navigation under delay was not found; the closest are missile engagements.

## 9. Follow-up spikes (each under one day)

1. **Age fix and reflex mismatch** (idea 1 smallest experiment): reflex arm with true versus design `t_lat_s` at injected L. Answers how much knowing the latency can ever be worth.
2. **Cadence ceiling:** FSM at FIXED 600 ms, sequential versus periodic:2/5/10, flip-flop count. Checks the executor under repeated stale AVOID_* switching.
3. **Entity generalisation prototype:** a box obstacle with `getClosestPoints` clearance, timing the ray ring with 1, 10 and 50 bodies.
4. **PN pursuer feasibility:** zero-latency oracle against a pursuer at 1.2 and 1.8 m/s with `max_accel` 1.0 and 1.5; report the fraction of seeds that are avoidable.
5. **Downwash crash:** repeat `multi.py` with spacing 0.5, 0.8, 1.5 m and with a staggered start; find the cause.
6. **Rollout fidelity:** deepcopy of PID and executor, `saveState`/`restoreState`, compare a 1-s forecast with the real next second under wind. Decides whether idea 7 needs physics rollouts or constant-velocity extrapolation is enough.
7. **VLM still-image test** (idea 15): 100 stills, three resolutions, three models; latency, tokens, accuracy.
8. **Two-drone head-on heat map** (idea 11 smallest experiment), standalone script.

---

## 10. Sources

Web results are search snippets unless noted; I did not open the full pages for most.

- Proportional navigation with delayed line-of-sight rate (Dhananjay, Lum, Xu, IEEE TCST 2013), quadrotor interception thesis comparing pursuit and PN: https://publications.iitm.ac.in/publication/capturability-of-augmented-proportional-navigation-apn-guidance ; https://apps.dtic.mil/sti/pdfs/AD1046289.pdf ; https://aerospace.technion.ac.il/?p=20112 (delay in pursuit-evasion)
- Time-to-contact / critical distance with latency in UAV avoidance: https://arxiv.org/pdf/2608.01855 (RADAR dynamic avoidance, `d_crit = v_rel (t_end + t_evade)`); https://arxiv.org/html/2512.00462v1 (acceleration CBF filter with latency); https://arxiv.org/pdf/2208.05873 (predictive angular potential field, speed from time-to-contact)
- Fast and slow dual-system agents: https://arxiv.org/pdf/2506.01953 (Fast-in-Slow); https://arxiv.org/pdf/2512.20188 (asynchronous fast-slow VLA); https://arxiv.org/html/2505.03912v1 (OpenHelix survey); https://arxiv.org/html/2511.04898v1 (RealtimeGym, AgileThinker; from 17a)
- Speculative actions for agents: https://arxiv.org/html/2510.04371v1 ; https://iclr.cc/virtual/2026/poster/10009726 ; https://arxiv.org/pdf/2607.12236
- Multi-drone delay: https://arxiv.org/pdf/2209.13667 (Robust MADER; from 17a)
- Runtime assurance / geofence: https://uh-ir.tdl.org/items/b7c29e04-4c39-4f0b-a530-3063b1e39974/full (from 17a)
- Reaction-time background (lecture notes only): https://psychology.hanover.edu/classes/hfnotes2/tsld038.html
- Image token cost (secondary): https://api.ifq.ai/blog/insights/image-detail-low-cost/ ; https://community.openai.com/t/consuming-more-tokens-than-expected-for-image-vision-gpt-4o/1097772
- gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones (installed version 2.2.0 inspected locally)
