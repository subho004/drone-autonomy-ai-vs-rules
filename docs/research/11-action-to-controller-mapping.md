# 11 - Action-space to controller mapping

Author: research agent 11. Access date for all web/source facts: 2026-09-30.
Scope: how the 11 discrete actions (FORWARD, LEFT, RIGHT, ASCEND, DESCEND, BRAKE, HOVER, AVOID_LEFT, AVOID_RIGHT, RETURN_HOME, EMERGENCY_LAND) become setpoints for a conventional controller in gym-pybullet-drones, what arbitrates between them, and what is wrong with the set.

Evidence key: **[V-code]** read in the real source (gym-pybullet-drones commit `7ebad1e`, 2026-09-06, package version 2.2.0, cloned to scratchpad); **[V-meas]** measured by me; **[INF]** inference; confidence H/M/L.

---

## 0. Executive summary (read this first)

1. **VelocityAviary is a poor base for this project as-is.** Its action is `[x, y, z, speed_fraction]`; it normalises xyz to a unit vector and scales by `SPEED_LIMIT = 0.03 * MAX_SPEED_KMH * (1000/3600)`. For the CF2X URDF (`max_speed_kmh="30"`) that is **0.25 m/s** [V-code, V-meas: default hold speed 0.239 m/s]. At 0.25 m/s the drone stops in 4 cm, so sudden-obstacle scenarios are trivially avoidable and **decision latency has no measurable effect** until seconds of delay. Any latency-vs-collision study must override `SPEED_LIMIT` (1-3 m/s) or bypass VelocityAviary.
2. VelocityAviary calls `DSLPIDControl.computeControl(target_pos=state[0:3], target_vel=...)`, i.e. **target_pos = current position every tick**. Then `pos_e == 0` exactly, the P and I position terms are identically zero, and only the **D term on velocity error** acts [V-code]. Consequences [INF, H from code reading]: (a) HOVER is "velocity damping", not position hold; under a steady wind force F the drone settles to a drift speed F/0.2 N per m/s instead of holding position (D_COEFF_FOR = 0.2, `GRAVITY`=0.265 N, so a 7.5 % of weight lateral wind force gives ~0.1 m/s drift and unbounded displacement); (b) braking by "zero velocity" reverses sign (measured undershoot -0.21 m/s from 1 m/s, see section 2.3). So HOVER must be a **position-latch** setpoint, which VelocityAviary cannot express. **[Spike 11b, section 16: (a) CONFIRMED and quantified, drift speed = F/0.2 exactly; the BRAKE half of this sentence was REFUTED: latching at the current position is much worse (72-108 % reversal); BRAKE needs a velocity ramp or a forward-offset latch, see 16.2.]** Recommendation: subclass `BaseAviary` (or `CtrlAviary`) and call `DSLPIDControl` ourselves with a full setpoint (pos, vel, yaw).
3. Model actions as a **two-channel latch** (horizontal channel, vertical channel) plus a TTL-bounded avoid modifier, with the executor, not the provider, owning hysteresis, slew limits and TTL. Keep the shield/arbiter identical across arms and log every override, otherwise the shield's own fast reflex contaminates the Jev-vs-LLM latency comparison.
4. The 11-action set is **sufficient for a v1** but has real gaps: no speed control (magnitude fixed, so stopping distance fixed), no yaw (only matters if sensing is directional), ASCEND/DESCEND semantics unspecified, no composite motion, no notion of "arrived". Minimal extensions recommended in section 9: a `speed` argument (3 levels), bounded-displacement semantics for lateral/vertical actions, conditional YAW_LEFT/YAW_RIGHT, and a clarifying redefinition of FORWARD as "toward active goal" which makes TRACK_TARGET redundant for moving targets.
5. ~~Not measured (blocked)~~ **Now measured (spike 11b, section 16, 2026-09-30).** The originally blocked spike (hover drift, position-latch braking, yaw ramp, control-tick throughput) was run by a follow-up agent on Python 3.14.2. Outcome: hover-drift claim confirmed; the claim that a position-latch brake removes the reversal is **refuted** for latch-at-current-position; yaw and throughput numbers added. Corrections are marked inline with "[Corrected by 11b]".

Side finding for agents 02/13: `pip install pybullet` on this M4 (macOS SDK, Python 3.12.12) **fails to build from sdist** (3.2.7 has no macOS wheels on PyPI; `zlib/zutil.h` `#define fdopen(fd,mode) NULL` collides with the macOS SDK's `stdio.h` declaration). I patched the vendored header (removed that define), built with `uv pip install --no-build-isolation .` (2 m 3 s), and it runs (`getAPIVersion() = 202010061`). Python 3.14 was not tried by me.

---

## 1. The real API surface (what we map onto)

### 1.1 DSLPIDControl [V-code]

```python
class DSLPIDControl(BaseControl):
    def __init__(self, drone_model: DroneModel, g: float = 9.8)   # only CF2X / CF2P; else print("[ERROR]") + exit()
    def reset(self)                                               # zeroes last_rpy, last_pos_e, integral_pos_e, integral_rpy_e
    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                       target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)) -> (rpm[4], pos_e[3], yaw_err)
```

Structure (two cascaded loops, both run every control tick):

- Position/velocity loop: `target_thrust = P_FOR*pos_e + I_FOR*integral_pos_e + D_FOR*(target_vel - cur_vel) + [0,0,m*g]`; gains `P=[.4,.4,1.25]`, `I=[.05,.05,.05]`, `D=[.2,.2,.5]`. Integral clipped to +-2 (z: +-0.15). Thrust is projected on body z; desired z-axis is `target_thrust/|target_thrust|`; yaw comes from `target_rpy[2]` only.
- Attitude loop: PID on rotation-matrix error to `target_euler`; `target_rpy_rates` enters **only** as the derivative reference (docstring explicit: not body rates, not ACRO); `cur_ang_vel` is unused. So there is **no yaw-rate command**; yaw rate must be produced by ramping `target_rpy[2]`.
- Output clip: PWM in [20000, 65535]. No tilt limit on `target_thrust` direction: a large `pos_e` produces a large commanded tilt, limited only by thrust saturation. So **never hand it a far position target**; use a carrot (`target_pos = cur_pos + v_des * T_look`) or velocity mode.
- Stateful: integrators and `last_rpy` persist across calls. Switching setpoint type abruptly (velocity-mode to position-latch) needs either `reset()` of the position integral or integrator anti-windup handling; otherwise a bump.
- Smell: `print()` calls inside the hot path (`[ERROR] ctrl it ...`) and `exit()` in `__init__`; wrap/monkey-guard in our controller adapter (repo rules forbid print; do not silently edit the vendored lib).
- Other controllers in the repo: `CTBRControl.py` (250 lines, socket + transforms3d imports, thrust/body-rate style; only header read, not evaluated), `MRACControl.py` (adaptive). Out of scope; DSLPID is the right choice for a PID baseline because the env already ships it.

### 1.2 VelocityAviary [V-code]

- Action space `Box(low=[-1,-1,-1,0], high=[1,1,1,1], shape=(N,4))`, values `[vx_dir, vy_dir, vz_dir, speed_fraction]`.
- `_preprocessAction`: per drone `v_unit = target_v[:3]/|target_v[:3]|` (zero vector maps to zero), `target_vel = SPEED_LIMIT * abs(target_v[3]) * v_unit`, `target_pos = state[0:3]`, `target_rpy = [0, 0, state[9]]` (keeps *current* yaw, so yaw is never commanded), then `computeControl`. The env does not clip the action (Box is advisory).
- `SPEED_LIMIT = 0.03 * MAX_SPEED_KMH * 1000/3600` set after `super().__init__()`, can be overwritten as an attribute (I did, `env.SPEED_LIMIT = v`).
- Observation: 20-vector `[pos3, quat4, rpy3, vel3, angvel3, last_rpm4]`; indices 0-2 pos, 7-9 rpy, 10-12 vel.
- Default `ctrl_freq=240`, `pyb_freq=240` (example `pid_velocity.py` uses 48 Hz control / 240 Hz physics; `BaseAviary` does `PYB_STEPS_PER_CTRL = pyb_freq/ctrl_freq`).
- Obstacles (`obstacles=True`) load fixed Bullet URDFs (samurai, duck, cube, sphere) at fixed positions, not a scenario mechanism (agent 06 owns dynamic obstacles).
- `sync()` in `utils/utils.py` only sleeps; it is called by example scripts, not by `step()`, so headless runs are not wall-clock paced unless we call it.

### 1.3 Physical envelope of the CF2X (URDF) [V-code]

mass 0.027 kg, thrust-to-weight 2.25, `max_speed_kmh=30` (8.33 m/s, the model's own cap; we will stay at <= 3-4 m/s), arm 0.0397 m, `kf=3.16e-10`. Max sustained horizontal accel is tilt-limited: with T/W 2.25, `a_max ~ g*sqrt(2.25^2-1) ~ 19.7 m/s^2` theoretical, but the PID/attitude lag gives far less (measured below).

---

## 2. Measurements (scratch venv, Python 3.12.12, pybullet 3.2.7 self-built, numpy/scipy from wheels; Apple M4; VelocityAviary, CF2X, Physics.PYB, pyb 240 Hz / ctrl 48 Hz; `SPEED_LIMIT` overridden; start z = 1.0 m; command `[1,0,0]`x fraction 1 held 4 s, then `[0,0,0,0]` for 3 s)

### 2.1 Step response and braking (velocity-zero braking, as VelocityAviary does it) [V-meas]

| SPEED_LIMIT (m/s) | v reached after 4 s | t to 90 % of v | braking distance | time to <5 % v | min velocity during brake (reversal) |
|---|---|---|---|---|---|
| 0.25 (default) | 0.239 | 0.29 s | 0.040 m | 0.33 s | -0.052 |
| 1.0 | 0.975 | 0.31 s | 0.159 m | 0.31 s | -0.210 |
| 2.0 | 1.969 | 0.38 s | 0.456 m | 0.42 s | -0.417 |
| 3.0 | 2.948 | 0.42 s | 0.861 m | 0.50 s | -0.333 |

Reading these [INF, M-H]:
- Steady-state velocity tracking error is ~1.5 % even with D-only control (gravity is feed-forward, no drag in Physics.PYB).
- Velocity time constant ~ 0.15-0.2 s (t90 ~ 0.3-0.4 s); effective braking decel from equivalent `v^2/(2d)`: 3.1 (1 m/s), 4.4 (2 m/s), 5.2 m/s^2 (3 m/s). **Use a_brake = 3 m/s^2 in planning** (conservative; 2 m/s stop = 0.67 m planned vs 0.46 m observed). **[Corrected by 11b: decel is speed dependent, implied v^2/(2d) = 2.0 / 3.5 / 4.9 m/s^2 at 1 / 2 / 3 m/s, so 3 m/s^2 is conservative at 2-3 m/s but NOT at 1 m/s (planned 0.17 m vs actual 0.25 m). Measured stopping distances for velocity-zero braking are 0.25 / 0.56 / 0.88 m at 1 / 2 / 3 m/s (16.2); use a lookup/interpolation on those, or re-fit in spike 2.]**
- The reversal (-10 to -21 % of entry speed) is a limit of "zero velocity + pos=cur" braking (attitude lag plus the target_pos being re-zeroed each tick so nothing pulls the drone back). ~~A position-latch brake should remove most of it (hypothesis).~~ **[Corrected by 11b: REFUTED. Latching at the current position gives 72 / 101 / 108 % reversal at 1 / 2 / 3 m/s; only a latch placed ahead at roughly the stopping point removes it. Also the 3 m/s reversal in the table above (-0.333) was wrong: re-measured -0.77 (VelocityAviary) / -0.98 (direct), i.e. 26-34 % of entry speed, and the 1-3 m/s range is 17-34 %, not 10-21 %. Braking distances were reproduced within ~0.02-0.07 m (note definition differences in 16.2). See 16.2.]**
- The knee for the project: stopping distance + travel during latency `d = v*(L + tau_reaction) + v^2/(2 a)`. At v = 2 m/s, L = 0.3 s: 0.6 + 0.67 = 1.3 m; L = 3 s: 6 + 0.67 m. For the default 0.25 m/s the same latency costs 0.075 m / 0.75 m: irrelevant. Pick scenario speeds and sensor ranges so that the latency knee lands inside the sweep (0-3000 ms); agent 08/13 should confirm.
- ~~Not measured: throughput (ticks/s) and hover drift.~~ [Measured by 11b, section 16.1 and 16.4.]

---

## 3. Architecture of the mapping

```
Provider(s) --Action(enum,args)--> ActionExecutor --Setpoint--> Shield/Arbiter --Setpoint'--> ControllerAdapter --rpm--> env.step
   (async, variable latency)        (control-tick rate)                                    (DSLPID | MPC | CBF-QP)
                                      ^ state estimate (noisy/GPS-loss-aware) ^
```

- **Setpoint** (controller-agnostic, the only thing the controller sees):
  `Setpoint(mode: VEL|POS, pos_des[3], vel_des[3], yaw_des, z_mode, reason)`; velocity mode is realised as `target_pos = pos + vel_des*T_look` (carrot, `T_look = 0.3 s`, keeps the P term alive, bounded tilt) with `target_vel = vel_des` as feed-forward; position-latch mode uses `target_pos = latch`, `target_vel = 0`. (A pure `target_pos = cur_pos` velocity mode, as in VelocityAviary, is kept as an option `vel_only` for parity testing.)
- **Frames**: world = PyBullet world, z up, metres, m/s; x/y arbitrary horizontal. Motion actions are expressed in the **path frame** `(f, l, u)`: `f = normalize(goal_xy - pos_xy)` (toward the active goal, recomputed every tick; for a moving target the goal moves, so FORWARD tracks it), `l = rot90(f)` (LEFT), `u = z`. Yaw is held at spawn yaw in v1 (see section 9 for when that changes). Rationale: body-frame LEFT/RIGHT would make a *stale* decision (issued at yaw psi0, executed at yaw psi1) mean different things; goal-bearing frame is stable as long as the goal is far; inside `r_goal_freeze = 1.0 m` freeze `f` to avoid rotation singularity.
- **Two-channel latch** [recommended design]: horizontal channel H in {STOP_HOLD, FORWARD, LEFT, RIGHT, RTH, LAND_DAMP} and vertical channel V in {HOLD_Z, UP, DOWN, LAND_DESCENT}. Horizontal actions write H only, ASCEND/DESCEND write V only, so "forward + climb" is expressible as two consecutive decisions without the first being lost. HOVER, BRAKE, RETURN_HOME, EMERGENCY_LAND write both. AVOID_* is a TTL modifier layered on H.

---

## 4. Per-action table

Default magnitude profile (parameters, scenario-level, not action-level): `v_cruise = 2.0 m/s`, `v_lat = 1.5`, `v_climb = 1.0`, `v_rth = 1.5`, `v_land = 0.5 m/s`, `a_max(slew) = 3.0 m/s^2`, `jerk` unlimited (slew is linear ramp). All are in `ActionProfile`, swept as a scenario parameter (speed matters for the latency knee).

Legend: VEL = velocity-carrot mode, POS = position-latch. TTL applies only when mode `ttl` is on (section 6.2); "hold-last" means persists until next decision.

| # | Action | Channel | Setpoint transform (v_des / p_des) | Frame | Magnitude | Latched / one-shot / TTL | Exit condition | Preemption rules (who may replace it) |
|---|---|---|---|---|---|---|---|---|
| 1 | FORWARD | H | `v_des = v_cruise * f` (vert. per V channel) | path (goal bearing) | 2.0 m/s, slewed 3 m/s^2 | latched; watchdog TTL 2 s (mode ttl) | goal reached (`|p-goal|<r_goal=0.3 m` for 0.5 s) -> auto STOP_HOLD and flag `arrived` | any provider action; shield P2 (collision) overrides; P1 |
| 2 | LEFT | H | `v_des = v_lat * l` | path, +90 deg | 1.5 m/s | **displacement-bounded**: active until lateral offset from entry point >= `d_step = 0.75 m` or TTL 1.0 s, then STOP_HOLD (lateral) and forward resumes? No: H returns to STOP_HOLD (provider must re-issue FORWARD) | bounded displacement | replaced by any H action; LEFT<->RIGHT reversal subject to hysteresis (section 7) |
| 3 | RIGHT | H | `v_des = -v_lat * l` | path, -90 deg | 1.5 m/s | as LEFT | as LEFT | as LEFT |
| 4 | ASCEND | V | `vz_des = +v_climb`; horizontal per H channel | world z | 1.0 m/s | displacement-bounded: ends at `dz = 0.5 m` or ceiling `z_max`; then V=HOLD_Z at reached altitude (p_z latch) | dz reached / z >= z_max (hard geofence) | replaced by V action; shield geofence clamp |
| 5 | DESCEND | V | `vz_des = -v_climb` | world z | 1.0 m/s | as ASCEND with floor `z_min = 0.3 m` (shield floor, prevents ground collision) | dz reached / z <= z_min | as ASCEND |
| 6 | BRAKE | H+V | phase 1: `v_des` ramps to 0 at `a_brake = 3.0 m/s^2` from current velocity; phase 2 (`|v|<0.1 m/s`): POS latch at current position [11b: phase 2 as written is fine because v is already ~0 at latch time; a direct latch at current position while moving is NOT; phase 1 as a 3 m/s^2 ramp was not measured, only the step-to-zero variant, see 16.2] | world | n/a | transient (<= 3 s), self-terminates into HOVER latch | `|v| < 0.1` -> becomes HOVER | any provider action; not preempted by shield unless P1 |
| 7 | HOVER | H+V | POS latch: `p_des = p at latch`, `v_des = 0`, integrator reset on entry, anti-drift under wind | world | n/a | latched indefinitely (mode hold-last) | replaced by any action | any provider action; watchdog fallback target of all other actions' TTL expiry |
| 8 | AVOID_LEFT | H modifier | `v_des = v_base*0.5 + v_avoid * l`, `v_avoid = 2.0 m/s` (lateral dominant), forward slowed to 50 % | path | 2.0 m/s lateral | **TTL modifier**: `T_avoid = 1.0 s` or until `d_lat >= d_clear (1.5 m)` or threat cleared (Schmitt, section 7); then revert to the *underlying* latched H (e.g. FORWARD) | TTL/clear | committed for `T_commit = 0.5 s` against opposite side; can be replaced by BRAKE/HOVER/EMERGENCY_LAND immediately; shield collision override beats it |
| 9 | AVOID_RIGHT | H modifier | mirror of 8 (`-l`) | path | 2.0 m/s | as 8 | as 8 | as 8 |
| 10 | RETURN_HOME | H+V | `v_des = v_rth * unit(home - p)` horizontal; vertical: climb to `z_rth = max(z, 1.5 m)` first then hold; home = takeoff point in **estimator frame** (dead-reckoned when GPS lost, so drift affects it; agent 06/07) | world | 1.5 m/s | latched until arrival | `|p_xy - home_xy| < 0.3 m` -> POS latch at home (HOVER_HOME), flag `rth_complete` | replaceable by provider (except after EMERGENCY_LAND); the shield may override for collision |
| 11 | EMERGENCY_LAND | H+V | horizontal: damp to 0 (`v_des=0`, POS latch xy at entry); vertical: `vz_des = -v_land` with ground-proximity slowdown (`<0.3 m`: 0.25 m/s) | world | 0.5 m/s descent | **terminal latch**: irreversible within the flight | `z < 0.06 m and |vz|<0.05` -> LANDED (flight ends; crash if touchdown `|vz| > 1.5 m/s` or tilt > 60 deg) | **not preemptible** by any provider action (except collision shield may still shift horizontally? No: shield may only brake/damp, never climb). Only scenario reset clears it |

Notes on the table:
- Decision D1: LEFT/RIGHT are **displacement-bounded**, not velocity latches. Rationale: a latched lateral velocity under 1-3 s decision latency drives the drone tens of metres sideways; displacement bounding makes every lateral decision's worst-case cost independent of latency, which is the property we want for a fair latency study and is also how human-facing "step" UIs work. The alternative (hold-last) is the `latch_mode=hold_last` ablation (section 6.2).
- Decision D2: AVOID_* return to the **underlying latched action** (modifier semantics) rather than to HOVER. If they returned to HOVER, every avoidance would cost one full decision latency of dead time, penalising slow arms twice; if they overrode the waypoint permanently the drone would leave the route. Modifier semantics keeps the waypoint goal.
- Decision D3: FORWARD/HOVER semantics on arrival: `arrived` triggers STOP_HOLD automatically; success is scored by the scenario (dwell in goal radius), not by an action.
- Decision D4: RETURN_HOME when GPS is lost uses the estimator (dead reckoning, drifting). It is a legitimate test: ground truth home vs estimated home error is logged.
- EMERGENCY_LAND "damp xy" uses POS latch so wind does not carry the drone into obstacles during descent.

### 4.1 Setpoint to DSLPID argument mapping (concrete)

| Setpoint mode | `target_pos` | `target_vel` | `target_rpy` | `target_rpy_rates` |
|---|---|---|---|---|
| VEL (carrot) | `pos + vel_des*0.3` | `vel_des` | `[0,0,yaw_des]` | `0` |
| POS latch | `p_latch` | `0` | `[0,0,yaw_des]` | `0` |
| vel_only (VelocityAviary parity) | `pos` | `vel_des` | `[0,0,yaw_cur]` | `0` |
| VEL with vertical channel | xy as above, z: `pos_z + vz_des*0.3` (UP/DOWN) or `z_latch` (HOLD_Z) | `[vx,vy,vz_des]` | | |

`target_rpy_rates` is deliberately left at zero: per the docstring it is only a D reference and non-zero values have unverified semantics [V-code]. Yaw rate (extension E2) is realised by integrating `yaw_des += r*dt` and feeding `target_rpy[2]`.

Velocity-setpoint slew limiter (executor-side, per tick): `v_cmd += clip(v_des - v_cmd, -a_max*dt, +a_max*dt)` per axis-norm; this is what makes "BRAKE ramp" and stale-decision steps well behaved and is identical for all arms.

---

## 5. Arbitration / shield layer

Priority order, highest first. Each layer returns `(setpoint, reason)`; first layer that asserts wins, lower layers are not evaluated for that tick, except that P0 always clamps the final result.

| Prio | Layer | Trigger (ground-truth/estimate used) | Action | Latching | Notes |
|---|---|---|---|---|---|
| P0 | **Envelope clamp** (always on, all arms) | speed > `v_max`, z < `z_floor` (0.3 m unless landing), z > `z_max`, outside geofence, NaN setpoint | clamp / project setpoint; never creates a new action | none | pure function, prevents physically absurd setpoints (tilt/speed) |
| P1 | **Hard failsafe** (always on) | battery <= `b_crit` (e.g. 10 %), attitude > 75 deg (flip), sustained state-estimate invalid > 2 s | force EMERGENCY_LAND latch | irreversible | keeps the crash metric about decisions, not about battery physics; make the rule explicit and identical for all arms |
| P2 | **Collision shield** (ablation switch: S0/S1/S2) | TTC/stopping-distance test: `d_obs - r_margin < v_closing*t_react + v_closing^2/(2*a_brake)` | S1: BRAKE override only (clamp approach velocity to safe value = speed clipped to `sqrt(2*a_brake*(d-r_margin))` along the obstacle direction, CBF-style velocity filter). S2: add lateral deflection. | override released after clear for `t_clear = 0.3 s` (Schmitt) | counts as `source=SHIELD` in logs |
| P3 | **Comms/decision watchdog** | no decision for `ttl` (mode ttl) or malformed output streak | fall back to BRAKE then HOVER (or to the **mission default**, section 5.2) | until next valid decision | TTL value is an experimental factor |
| P4 | **Provider action** (Jev / LLM / RL / rule FSM / oracle) | decision arrives | executor state machine (section 6) | per action | |
| P5 | **Mission default** | no provider assigned (PID arm) | position-carrot toward goal | | the "waypoint PID, no avoidance" baseline is literally P5 alone |

### 5.1 Shield levels as an experimental factor (critical for interpretation)

- **S0**: P0 + P1 only. Used for the main latency curve: collision outcome is driven purely by decisions. Collision rates will be high for slow arms at high speed, which is the signal.
- **S1**: S0 + P2 brake-only. Realistic deployment ("never fly into a known obstacle faster than you can stop"); expect the latency effect to shrink to "progress loss" rather than collisions. This is the honest version for claims about safety.
- **S2**: S0 + P2 with deflection. Turns the shield into a reactive avoider, i.e. the arm B baseline (agent 05). Report as its own arm, not as a setting of the decision arms.
- **Must-report metrics**: `shield_override_rate`, `time_under_shield`, `first_override_latency` per arm; an arm "wins" only if the claim holds at S0 and S1. Killer-demo hazard: a shield with a 20 ms reflex beats a 300 ms LLM without any "decision" being involved; making that visible is the point of S0 vs S1.
- Information used by P2 must be **the same sensor-derived estimate** given to providers (range sensors with noise, not ground truth), else the shield is an oracle. P1 battery uses the true battery (it is an onboard measurement).

### 5.2 Timeout fallback definition

Two modes (an experimental factor, `ttl_mode`):
- `hold_last` (primary for the latency study): last action persists; only a 10 s dead-man (P3) triggers HOVER. Isolates pure staleness.
- `ttl_to_hover`: each latched action carries `ttl = 2.0 s` (like PX4 offboard, where setpoints must stream continuously and loss triggers a failsafe mode; PX4 doc page fetched but excerpt did not show the exact rate/timeout, so the "2 Hz / 500 ms" recollection is **unverified**, treat as [INF, L]). On expiry: BRAKE -> HOVER. With TTL 2 s and LLM latency >= 2 s the drone alternates hover/act, which is a legitimate deployed-system outcome but changes the meaning of the curves, hence a separate factor.

---

## 6. ActionExecutor

### 6.1 Timing model (control tick vs decision tick)

- Physics 240 Hz, control 48 Hz (`dt_c = 20.83 ms`, 5 physics steps per control tick), decisions asynchronous at wall/sim-mapped latency `L`. Applied at the **first control tick on or after arrival**: `k_apply = ceil((t_obs + L) / dt_c)`; effective latency in ticks `n = k_apply - k_obs` (L = 300 ms -> 15 ticks; 3000 ms -> 144 ticks; 10 ms -> 1 tick floor since at least one tick passes between sense and act). Zero-order hold on actions between decisions. (Global time model is agent 08's; this is the executor's view.)

```
time ->   0     20.8   41.7   62.5   ...                     t_obs+L
control   |c0|---|c1|---|c2|---|c3|---|c4|---|c5|-- ... ---|ck|---|ck+1|
ticks      ^ each tick: read state -> shield -> executor.tick -> DSLPID -> 5x physics steps (240 Hz)
decision   [ snapshot@c0 ]------------ provider latency L (in flight) ---------->[ action arrives between c(k-1) and ck ]
                ^ obs age grows 0,1,2,..n ticks                                   ^ applied at ck (ZOH, latched)
           executor keeps running previous latched action (or BRAKE if TTL) during the whole flight of the request.
next req   fixed-period provider: new snapshot every T_d regardless (requests overlap, late replies drop if older than last applied obs)
           back-to-back provider: new snapshot issued at ck when the reply lands (decisions/s = 1/L)
```

Rules: (1) replies carry `obs_tick`; apply only if `obs_tick > last_applied_obs_tick` (drop out-of-order); (2) the decision carries `age_ticks = k_apply - obs_tick` into the log and, optionally, into the next prompt (agent 07); (3) the PID/MPC arms tick the executor at control rate with zero age.

### 6.2 Latch-mode factors

`latch_mode in {hold_last, ttl}` (section 5.2) and `lateral_semantics in {bounded, velocity_latch}`. Primary: `hold_last` + `bounded`. Report the other three combinations on a subset.

### 6.3 Pseudocode (Protocol-based, follows repo style; illustrative, not repo code)

```python
class Action(StrEnum):  # see section 10 for codes
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
    EMERGENCY_LAND = "EMERGENCY_LAND"


@dataclass(frozen=True)
class Setpoint:
    mode: Literal["VEL", "POS"]
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    source: Literal["PROVIDER", "SHIELD", "FAILSAFE", "DEFAULT"]
    reason: str


class ActionExecutor:
    def __init__(
        self, profile: ActionProfile, dt: float, rng: np.random.Generator | None = None
    ) -> None:
        self._p, self._dt = profile, dt
        self._h: HMode = HMode.STOP_HOLD
        self._v: VMode = VMode.HOLD_Z
        self._avoid: AvoidState | None = None  # (side, t_left, d_lat_done, base_h)
        self._latch_pos: np.ndarray | None = None
        self._v_cmd = np.zeros(3)  # slewed velocity command
        self._last_dec_tick = 0
        self._landing = False
        self._dwell: dict[str, int] = {}  # hysteresis counters

    def on_decision(
        self, act: Action, obs_tick: int, now_tick: int, st: State
    ) -> DecisionOutcome:
        if self._landing:  # EMERGENCY_LAND is terminal
            return DecisionOutcome(accepted=False, reason="landing_latched")
        if self._violates_hysteresis(
            act, now_tick
        ):  # section 7: min-dwell, side-commit
            return DecisionOutcome(accepted=False, reason="hysteresis")
        self._last_dec_tick = now_tick
        match act:
            case Action.FORWARD | Action.LEFT | Action.RIGHT:
                self._h = HMode[act.name]
                self._entry_pos = st.pos.copy()
                self._avoid = None
            case Action.ASCEND | Action.DESCEND:
                self._v = VMode[act.name]
                self._v_entry_z = st.pos[2]
            case Action.BRAKE:
                self._h, self._v, self._avoid = HMode.BRAKE, VMode.HOLD_Z, None
            case Action.HOVER:
                self._enter_hold(st)
            case Action.AVOID_LEFT | Action.AVOID_RIGHT:
                self._avoid = AvoidState(
                    side=+1 if act is Action.AVOID_LEFT else -1,
                    t_left=self._p.t_avoid,
                    d_done=0.0,
                    entry=st.pos.copy(),
                )
            case Action.RETURN_HOME:
                self._h, self._v, self._avoid = HMode.RTH, VMode.RTH_CLIMB, None
            case Action.EMERGENCY_LAND:
                self._landing = True
                self._latch_xy(st)
        return DecisionOutcome(accepted=True, reason="ok")

    def tick(self, st: State, now_tick: int, ttl_mode: bool) -> Setpoint:
        if (
            ttl_mode
            and now_tick - self._last_dec_tick > self._p.ttl_ticks
            and not self._landing
        ):
            self._h = HMode.BRAKE  # P3 watchdog -> brake -> hover
        f, l = self._path_frame(st)  # goal bearing; frozen inside r_goal_freeze
        v_des = np.zeros(3)
        if self._landing:
            v_des, mode = self._land_setpoint(st)
        else:
            v_des, mode = self._horizontal(
                st, f, l
            )  # FORWARD/LEFT/RIGHT/RTH/BRAKE/STOP_HOLD
            v_des[2] = self._vertical(st)  # UP/DOWN bounded by dz / ceiling / floor
            if self._avoid:
                v_des = self._apply_avoid(
                    v_des, f, l, st
                )  # TTL, d_clear, Schmitt clear
        self._v_cmd = self._slew(self._v_cmd, v_des, self._p.a_max * self._dt)
        self._update_exits(
            st
        )  # arrived, brake done -> hold latch, lateral bound reached
        if mode == "POS":
            return Setpoint(
                "POS", self._latch_pos, np.zeros(3), self._yaw, "PROVIDER", self._reason
            )
        return Setpoint(
            "VEL",
            st.pos + self._v_cmd * self._p.t_look,
            self._v_cmd,
            self._yaw,
            "PROVIDER",
            self._reason,
        )


class Shield:  # P0..P3; identical for every arm, configured by level S0/S1/S2
    def filter(
        self, sp: Setpoint, st: State, est: Estimate, exec_: ActionExecutor
    ) -> Setpoint: ...
```

Integration loop (per control tick, single process, deterministic given a latency schedule): `st = obs(); for reply in latency_queue.due(tick): exec.on_decision(...); sp = shield.filter(exec.tick(st), st, est); rpm = ctrl.compute(sp, st); obs = env.step(rpm)`. Controller swap (DSLPID / MPC / CBF-QP) lives behind `ControllerAdapter.compute(Setpoint, State) -> rpm`.

---

## 7. Smoothing and hysteresis rules

All applied in the **executor/shield**, never in a provider, so every arm gets identical stabilisation; raw provider decision sequences are also logged so oscillation is measurable from the source.

1. **Slew limit**: per-tick velocity-command change <= `a_max*dt` (3 m/s^2 -> 0.0625 m/s per tick at 48 Hz). Prevents steps in `target_vel` from saturating attitude.
2. **Min dwell**: a *different* non-emergency action is accepted only after `t_dwell = 0.25 s` (12 ticks) since the last accepted change, except BRAKE, HOVER, EMERGENCY_LAND and any AVOID_* while a P2 threat is active (safety actions bypass dwell). Dwell < decision period at fast arms is a no-op; it only bites a chattering provider.
3. **Side commitment**: after AVOID_LEFT (RIGHT), the opposite avoid is rejected for `T_commit = 0.5 s` unless the threat vector flips sign (obstacle passes). Prevents left-right dithering in front of an obstacle.
4. **Schmitt trigger for threat/clear**: threat asserts at `d < d_in = 3.0 m` (or TTC < 2 s) and clears at `d > d_out = 4.0 m` (TTC > 3 s) for 0.3 s; used by P2 and by AVOID exit. Thresholds scale with speed (`d_in = v*t_react + v^2/(2a) + margin`).
5. **No-op suppression**: repeating the current action resets nothing (no restart of TTL ramp) except refreshing `last_dec_tick` (feeds the watchdog). Crucial: re-issuing FORWARD every decision must not re-arm the slew/entry reference.
6. **Landing and hover transitions**: position-latch entry resets the position integrator (`ctrl.integral_pos_e = 0`) to avoid integrator kick from velocity mode; BRAKE phase 2 latches at the *current* position to avoid pulling back (valid only once |v| < 0.1 m/s; 11b measured that latching at the current position while still moving makes the drone fly back, section 16.2).
7. **Yaw hold**: yaw target continuous, rate-limited to 90 deg/s via ramping.
8. **Metrics hooks**: `n_action_switches`, `n_avoid_reversals`, `oscillation_index = switches per 10 s` from raw provider output and from accepted output separately.

---

## 8. Per-arm mapping

| Arm | How it produces Action / Setpoint | Notes |
|---|---|---|
| A: waypoint PID, no avoidance | No provider; P5 default: position carrot toward goal; equals "FORWARD latched then HOVER" | executor bypassed for the decision layer but **same shield level** and envelope (P0/P1) |
| B: PID + reactive avoidance (APF/VO/CBF) | Continuous velocity setpoint directly into Setpoint (no enum); or as shield S2 | arm B is conceptually "S2 shield with FORWARD default" |
| C: MPC | Two options: (i) tracks the same `Setpoint` (p_ref, v_ref) with constraint set from shield; (ii) consumes action as cost/constraint (below). Primary: (i) | latency = measured solve time, in ticks |
| D: RL (PPO discrete head) | `Discrete(11)` (or the extended enum) sampled at its decision rate (e.g. 10 Hz, frame-skipped) -> executor | same observation projection; RL policy sees `age_ticks` and last action to learn the executor |
| E: rule FSM | Same 11 actions via thresholds | fairest non-ML baseline (agent 05); uses same hysteresis |
| F: slow LLM | enum via JSON/constrained decode -> executor | latency via injection (agent 08) |
| G: Jev | as F with fast latency | parse failures -> `INVALID` -> keep last action (or BRAKE if ttl mode) |
| Oracle | labeller (section 11) with privileged state | upper bound and label source |

**MPC: action as cost/constraint or as setpoint?** Recommendation: treat the action as a **setpoint generator plus a constraint descriptor**, and have MPC track the setpoint. Action to MPC mapping [INF]: FORWARD -> `v_ref = v_cruise f` (tracking cost); BRAKE -> `v_N = 0` terminal constraint with `|a| <= a_brake`; AVOID_L/R -> half-space constraint on lateral position relative to the obstacle plus lateral velocity reference; ASCEND/DESCEND -> `vz_ref`; RETURN_HOME -> position reference; EMERGENCY_LAND -> terminal `z=0`, `|v|<=v_land`. The richer the mapping, the more the MPC becomes the intelligence rather than the stabiliser (the brief's premise is that Jev picks the action, the controller stabilises), and MPC arms get unequal information. Use the constraint variant only as a named ablation.

---

## 9. Critique of the 11-action set and minimal extensions

### 9.1 What the set gets right
- Small, discrete, enum-friendly: 11 symbols map to **1-2 output tokens** (good for latency, agent 01), easy to constrain-decode, easy for RL (Discrete(11)).
- Separation of concerns matches the brief: the set is high-level intent, the PID stabilises.
- Emergency vs navigation split allows a clean priority lattice.

### 9.2 Problems (ranked)

1. **No magnitude**. Fixed `v_cruise` fixes the stopping distance, so the agent cannot trade speed for safety; the best policy under latency is often "slow down" when uncertain, which is unrepresentable. Also the default VelocityAviary speed (0.25 m/s) collapses the problem. **Extension E1 (recommended, required): `speed` argument in {SLOW=0.5, NORMAL=1.0, FAST=1.5} x v_cruise** (`SPEED_LEVEL` field in JSON, default NORMAL). Justification: one extra token; creates the speed-vs-latency trade that the collision-vs-latency curve is supposed to reveal; without it a slow LLM cannot compensate for its own latency.
2. **Ambiguous FORWARD** (body x? heading? goal?). Fixed here as "toward active goal, path frame". This also neutralises the moving-target event without needing TRACK_TARGET. **TRACK_TARGET (E3) is NOT recommended for v1**: if the goal estimate is the target position, FORWARD already tracks it. Add it only if you want lead-pursuit (intercept) or separate "mission waypoint" vs "target of interest" semantics (two distinct references); then add a single `TRACK_TARGET` (goal := estimated target, with `p_ref = target + v_target*t_lead`).
3. **Channel composition**: flat single-label actions cannot express "forward and climb" or "avoid left while climbing". Fixed by two-channel latch + AVOID modifier (design choice, not an extra action). Alternative (rejected): add ASCEND_FORWARD etc.; doubles the set.
4. **Unbounded-cost actions under latency**: velocity-latched LEFT/RIGHT/ASCEND/DESCEND under 1-3 s staleness produce arbitrary displacement. Fixed by displacement-bounded semantics (D1). The `velocity_latch` variant is retained as an ablation.
5. **No yaw**. Only matters if perception is directional (forward-facing camera/lidar cone, or a second drone approaching from behind/side). Decision tied to agent 07: **if sensing is omnidirectional (sector raycasts 360 deg), keep yaw fixed and add no yaw actions** (recommended for v1, reduces action set). If forward-facing: add **E2: YAW_LEFT, YAW_RIGHT** (+-30 deg step, yaw-rate capped at 90 deg/s via ramped `target_rpy[2]`, since DSLPID has no rate command) and interpret FORWARD as body-forward (heading) rather than goal bearing. Do not add yaw in both ways.
6. **AVOID_LEFT vs LEFT redundancy**: the two differ by intent and speed profile (avoid = fast, lateral-dominant, forward slowed, TTL, returns to base action); LEFT = bounded lateral step that ends in stop-hold. Keep both, but document; else a model will treat them as synonyms. A provider-visible glossary (agent 07) with numeric magnitudes is needed.
7. **No vertical avoidance** (AVOID_UP/DOWN) for obstacles spanning the lateral corridor or second drones at similar altitude; ASCEND/DESCEND with bounded displacement are a slow substitute (0.5 m per decision). Acceptable for v1 if obstacles are floor-to-ceiling columns or spheres sized < lateral clearance; flag as a limitation, extension E4 = AVOID_UP / AVOID_DOWN if scenarios include overhead/walls.
8. **No "continue/no-op"**. A provider that wants to keep doing the same thing must re-emit it (costs tokens, and a slow provider's repeated FORWARD refreshes TTL). Suggest an internal `KEEP` (not exposed in the LLM enum, used for parse failures/timeouts) so invalid output does not become an implicit HOVER.
9. **Two terminal-ish actions** (RETURN_HOME, EMERGENCY_LAND) have mission semantics that the scorer must understand: RETURN_HOME ends the mission as failure/abort (not success); EMERGENCY_LAND ends in landed; define metrics accordingly (time_to_target undefined).
10. **EMERGENCY_LAND reversibility**: irreversible is simplest and safest but punishes false alarms permanently; good as the test of over-triggering. Keep irreversible.
11. **HOVER vs BRAKE duplicates**: BRAKE is a transient that ends in HOVER; provider-wise both are "stop". Keep both: BRAKE signals urgency (decel-limited stop at a_brake) while HOVER from speed executes the same stop at `a_max`? Define HOVER-from-motion as slew stop at `a_max = 3`, BRAKE at `a_brake = 3` ... identical numbers make them equal. **Decision**: BRAKE = maximum deceleration `a_brake = 5 m/s^2` (measured ability at 3 m/s was ~5.2), HOVER = gentle `a_max = 2 m/s^2`; document; otherwise redundant.
12. Risk of the DSLPID 'big error' failure: position-mode errors >~1 m produce steep tilt; carrot approach bounds it (section 4.1).

### 9.3 Recommended set

- **v1 core (11 actions, as given)** with: path-frame FORWARD, bounded LEFT/RIGHT/ASCEND/DESCEND, modifier AVOID_*, BRAKE/HOVER distinction by decel, terminal EMERGENCY_LAND.
- **E1 speed level** (3 values) **recommended for v1.5**; it is a JSON field, not a new enum member.
- **E2 YAW_LEFT/YAW_RIGHT** only if sensing is directional.
- **E3 TRACK_TARGET** only if lead pursuit is wanted.
- **E4 AVOID_UP/DOWN** only if vertical clearance scenarios exist.
- Internal-only: `KEEP`/`INVALID` (parse failure, timeout), never exposed to the model.

---

## 10. Shared enum and JSON schema

### 10.1 Canonical enum with integer codes (single-token-friendly)

| Code | Name | Code | Name |
|---|---|---|---|
| 0 | FORWARD | 6 | HOVER |
| 1 | LEFT | 7 | AVOID_LEFT |
| 2 | RIGHT | 8 | AVOID_RIGHT |
| 3 | ASCEND | 9 | RETURN_HOME |
| 4 | DESCEND | 10 | EMERGENCY_LAND |
| 5 | BRAKE | | (reserved 11 YAW_LEFT, 12 YAW_RIGHT, 13 TRACK_TARGET, 14 AVOID_UP, 15 AVOID_DOWN) |

Letter codes (A-K) or digits can be used when a provider is constrained to one token; names are canonical in storage and logs. (Tokenisation of names like `EMERGENCY_LAND` is multi-token; agent 01/07 decide name vs code on latency/accuracy; both are stored.)

### 10.2 Provider output schema (JSON Schema 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://drone-jev.local/schemas/action-command.json",
  "title": "ActionCommand",
  "type": "object",
  "additionalProperties": false,
  "required": ["action"],
  "properties": {
    "action": {
      "type": "string",
      "enum": ["FORWARD","LEFT","RIGHT","ASCEND","DESCEND","BRAKE","HOVER",
               "AVOID_LEFT","AVOID_RIGHT","RETURN_HOME","EMERGENCY_LAND"]
    },
    "speed": { "type": "string", "enum": ["SLOW","NORMAL","FAST"], "default": "NORMAL" },
    "reason": { "type": "string", "maxLength": 80 }
  }
}
```

Constraints: `speed` optional (E1, absent means NORMAL, ignored for EMERGENCY_LAND/BRAKE/HOVER); `reason` is optional telemetry for debugging runs and **must be off for the latency-critical Jev arm** (extra tokens); the minimal wire format is the bare enum string or code. Structured-output/`strict` support on the provider side is agent 01's; if unavailable, parse with a tolerant regex over the 11 names and treat no-match as `INVALID`.

### 10.3 Logged record (per decision, stored by agent 10)

`{decision_id, flight_id, obs_tick, issue_tick, recv_tick, apply_tick, age_ticks, raw_text, parsed_action, parse_status(OK|INVALID|TIMEOUT), accepted(bool), reject_reason(hysteresis|landing_latched|none), effective_action, source(PROVIDER|SHIELD|FAILSAFE|DEFAULT), shield_reason, oracle_action, oracle_admissible_set, oracle_urgency_s}`.

Python type (pydantic v2, placed in `app/schemas/`): `ActionName = StrEnum`, `ActionCommand(BaseModel, extra="forbid")`, to match the repo's Pydantic-at-the-boundary rule.

---

## 11. Oracle action labeller

Purpose: (a) upper-bound arm, (b) per-decision correctness and urgency scoring of every provider, (c) training/eval labels for distilled or RL policies, (d) sanity check that the scenario is **feasible** (there exists an admissible action sequence; owns the feasibility guarantee with agent 06).

### 11.1 Design: privileged-state rollout oracle (not a hand-written rule tree)

Inputs: ground-truth drone state, true obstacle/other-drone trajectories (from the seeded scenario), true wind, true battery, true home, goal. Model: a kinematic surrogate of the *executor+controller* (first-order velocity lag `tau = 0.15-0.2 s` from 2.1, accel limit 3/5 m/s^2, displacement-bounded primitives) calibrated from closed-loop PyBullet step responses (table 2.1). No PyBullet forking; surrogate rollouts are ~microseconds.

```
def label(state, scenario, t) -> Label:
    best = {}
    for a in ACTIONS (x speed levels if E1):
        traj = surrogate_rollout(state, executor_semantics(a), horizon=H=4.0 s, dt=dc)   # incl. latch continuation = "hold a then oracle default"
        cost = inf if collides(traj, margin=r_safe) or breaches_floor/geofence/battery_reserve(traj)
        else w_t*time_to_goal_est + w_e*energy + w_s*smoothness + w_h*(a != prev_action)*hyst
        best[a] = cost
    c_star = min(best.values())
    admissible = {a for a,c in best.items() if c <= c_star + eps}      # set, not a point
    urgency = min over a in admissible of "time margin": latest wait time delta such that after waiting
              delta (continuing current action) at least one admissible action remains collision-free   # seconds
    validity = earliest time the admissible set changes (label half-life)
    if battery_reserve_breached or geofence_unrecoverable: force {EMERGENCY_LAND} / {RETURN_HOME} per rule order
    return Label(action=argmin (tie-break: previous action, then smallest displacement), admissible, urgency, validity)
```

Priority rule for the oracle mirrors the shield lattice (battery critical -> EMERGENCY_LAND; GPS loss with distance > dead-reckoning budget -> RETURN_HOME else HOVER; imminent collision -> BRAKE/AVOID; else FORWARD). Because labels come from a rollout search and not from the same rules as the FSM arm (E), oracle and FSM are not circular.

### 11.2 How labels are used
- **Agreement scores**: `exact` (action == oracle argmin), `admissible` (action in set; the headline, since many actions are correct), `lethal` (action in set of collision-inducing actions given latency-adjusted state), `timely` (apply_time <= event_time + urgency).
- **Latency-aware variant**: evaluate provider action against the oracle computed at the *apply-time state* (what the action should have been when it lands) and at the *observation-time state*; the difference is the stale-label cost (feeds the decision-age analysis, agent 08).
- **Oracle with latency L** as an arm: re-run the oracle with `L` ticks delay; gives the physical upper bound on the collision-vs-latency curve (separates "latency" from "smartness"; the knee shown in agent 13's confound).
- **Labels for training**: store `(obs_projection, admissible_set, urgency)` per tick; train with set-valued loss (multi-label) rather than argmin to avoid label noise from ties.
- **Feasibility certificate**: if for any scenario `urgency < 0` at event onset (no admissible action), the scenario is infeasible and rejected by the generator (agent 06).

Cost: ~11 actions x 3 speeds x 200 surrogate steps x 10-50 Hz labelling ~ 0.2-1 ms per label vectorised in numpy (estimate [INF, L]; to be benchmarked).

---

## 12. Frame of open design risks

- Carrot `T_look=0.3 s` and `a_max` are tuned off a 1-D measurement at 48 Hz; 3-D diagonal + yaw coupling unmeasured.
- `DSLPIDControl` velocity-only drift under wind: deduced from code in this file, **measured by 11b (section 16.1): confirmed, drift speed = F/0.2 to 2 digits**.
- Physics.PYB has no drag/ground effect; `PYB_DRAG`/`PYB_GND_DRAG_DW` modes change decel and hover drift; calibrate the oracle surrogate per physics mode (agent 02/06).
- Control rate matters: at 48 Hz the shield reflex granularity is 20.8 ms; at 240 Hz (default) 4.2 ms; document the chosen `ctrl_freq` as part of the latency floor (a "zero-latency" arm still has >= 1 tick).
- Multi-drone (other-drone event): per-drone executors share nothing; VelocityAviary supports `num_drones` with one DSLPID per drone, which is convenient.
- Prompt-side glossary must state magnitudes (2 m/s, 0.75 m bounded steps, etc.) else providers can't reason about distances; owned by agent 07.

---

## 13. Decisions recommended

1. **Do not use VelocityAviary as-is.** Subclass `BaseAviary` (or `CtrlAviary`) and drive `DSLPIDControl.computeControl` ourselves with `target_pos`/`target_vel`/`target_rpy`; if VelocityAviary is reused, override `SPEED_LIMIT` (1-3 m/s) and accept the pos=cur limitation (no position hold).
2. Set default scenario speeds so the latency knee falls in the sweep: `v_cruise = 2 m/s` (range 1-3), CF2X, ctrl 48 Hz / physics 240 Hz, planning `a_brake = 3 m/s^2`, `tau_vel ~ 0.2 s` (measured, table 2.1).
3. Adopt the **two-channel latch** semantics: horizontal and vertical channels, FORWARD goal-bearing (path frame), AVOID_* as TTL modifiers that return to the underlying action, LEFT/RIGHT/ASCEND/DESCEND displacement-bounded, HOVER = position latch, BRAKE = velocity-zero or forward-offset latch at `p + v^2/(2 a_eff)` and only then latch in place [11b: never latch at the current position while moving], EMERGENCY_LAND = irreversible terminal latch.
4. Shield priority lattice P0 envelope > P1 hard failsafe > P2 collision > P3 watchdog > P4 provider > P5 default; **three shield levels S0/S1/S2** as an experimental factor; main latency curve at S0, deployment-realistic curve at S1; log shield override rate.
5. Hysteresis and slewing live in the executor, identical for all arms: slew 3 m/s^2, dwell 0.25 s, side-commit 0.5 s, Schmitt 3.0/4.0 m, no-op suppression.
6. `ttl_mode` default `hold_last`; `ttl_to_hover` (2 s) as a named second factor; the two must never be mixed in one comparison.
7. Add **E1 speed level (SLOW/NORMAL/FAST)** as an optional field; defer yaw (E2) unless directional sensing is chosen; do **not** add TRACK_TARGET in v1 (FORWARD = toward active goal covers a moving target); add AVOID_UP/DOWN only with overhead obstacle scenarios.
8. The action interface is `Action` enum + integer codes + JSON schema in 10.2; parse failure = `INVALID` -> `KEEP` (internal), logged with raw text.
9. Oracle = privileged kinematic-surrogate rollout labeller returning an admissible **set**, urgency and validity horizon, also used as a feasibility gate and as an "oracle with latency L" arm.
10. MPC tracks the same `Setpoint`; constraint-consuming MPC only as ablation. RL uses `Discrete(11)` into the same executor.

## 14. Open questions

1. Which sensing model (omnidirectional raycasts vs forward FOV)? Decides yaw actions and FORWARD frame (agent 07).
2. Is `hold_last` or `ttl_to_hover` the ruling timeout semantics in the benchmark protocol (agent 08)? Does the 2 s TTL value have a standard (PX4 offboard timeout figures were not verified from the fetched excerpt)?
3. ~~Real behaviour of position-latch braking, hover drift under wind, yaw ramp rate~~ (measured by 11b, section 16); diagonal-motion decel remains unmeasured.
4. Should RETURN_HOME require an obstacle-aware route or rely on S1/S2 shield? Affects arm fairness.
5. Decision cadence for the RL/FSM arms (every control tick vs 10 Hz): affects their effective latency floor; needs a protocol (agent 05/08).
6. Does Python 3.14.2 accept a pybullet build? **Answered by 11b: yes.** pybullet 3.2.7 installs and runs on CPython 3.14.2 (arm64 macOS) with `CFLAGS="-Dfdopen=fdopen" uv pip install pybullet` (no source patching), and gym-pybullet-drones `7ebad1e` runs on it; see 16.0 for a caveat about uv cache.
7. How does `Physics.PYB_GND_DRAG_DW` alter braking/hover enough to invalidate the surrogate? (Needs calibration per physics mode.)
8. Are LEFT/RIGHT displacement bounds (0.75 m) and AVOID `d_clear` (1.5 m) large enough to clear the obstacle sizes chosen by agent 06 (need obstacle radius distribution)?

## 15. Follow-up spikes (small, runnable experiments)

All in a scratch venv (`uv venv --python 3.12`; install pybullet from sdist after removing the `#define fdopen(fd, mode) NULL` lines in `examples/ThirdPartyLibs/zlib/zutil.h`, then numpy scipy gymnasium transforms3d matplotlib pillow, and `uv pip install --no-deps -e gym-pybullet-drones`). The second script below (my blocked spike) is to be run by the owner.

1. **Hover drift and brake comparison (the blocked spike; RUN by 11b, results in section 16; items C, the climb test, was not run).** Use `CtrlAviary` + `DSLPIDControl` directly at 48/240 Hz:
   - A: 10 s hover with `target_pos = cur_pos` (VelocityAviary semantics) vs `target_pos = latch`, with and without an `applyExternalForce` of 0.02 N; expected: drift vs hold.
   - B [DONE, 16.2]: from v = 1, 2, 3 m/s compare `velzero` braking to `poshold` (`target_pos = p + v*0.25`): measure stopping distance, reversal undershoot, final position error, altitude sag.
   - C: climb at `vz=1 m/s` 3 s, altitude tracking; D: yaw ramp at 90 deg/s via `target_rpy[2]`; E: control ticks/s for `DSLPIDControl + step` (shared with agent 02's benchmark).
   Acceptance: `poshold` undershoot < 5 % of entry speed; hover drift < 0.05 m in 10 s under 0.02 N wind.
2. **Surrogate calibration.** Fit `(tau, a_max)` of the first-order + accel-limit surrogate to PyBullet step/brake traces at 1-3 m/s for `Physics.PYB` and `PYB_GND_DRAG_DW`; acceptance: stopping-distance error < 15 %, used by the oracle.
3. **Latency-knee sketch.** One scenario: obstacle appears at range R = 6 m, drone at 2 m/s, FORWARD hold-last, executor+S0; latency 0/100/300/1000/3000 ms; collision vs latency curve; repeat at SPEED_LIMIT 0.25 (default) to show the null result. Shares work with agent 08's prototype.
4. **Executor unit tests without PyBullet** (pure state machine): dwell, side-commit, TTL, landing latch, displacement bounds, stale-reply drop.
5. **Shield S0/S1/S2 ablation** on the knee scenario (S1 should remove most collisions while raising time-to-target).
6. **Oracle microbenchmark**: labels/s for 11x3 surrogate rollouts vectorised; check agreement with a hand-written rule tree on 1k random states (expect `admissible` agreement > 95 %).

## 16. Spike results (measured)

Author: follow-up agent 11b, 2026-09-30. This section closes the spike that section 15 item 1 describes (A, B, D, E; the climb test C was **not** run). Anything in sections 0-14 contradicted by these numbers was edited in place and tagged "[Corrected by 11b]" or struck through.

### 16.0 Setup and honesty notes

- Hardware/software [V-meas]: Apple M4, macOS (Darwin 27.0.0), **CPython 3.14.2** (uv venv), pybullet 3.2.7 (`getAPIVersion()` 202010061), numpy/scipy/gymnasium/transforms3d from wheels, gym-pybullet-drones **commit `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`** (version 2.2.0), CF2X, `Physics.PYB`, pyb 240 Hz, ctrl 48 Hz (5 physics steps per control tick), `gui=False` (DIRECT), `record=False`, `obstacles=False`, start z = 1.0 m, 1 drone.
- Install: `CFLAGS="-Dfdopen=fdopen" uv pip install pybullet` succeeded in the 3.14.2 venv, then `uv pip install --no-deps -e gym-pybullet-drones`. **Caveat:** the pybullet install returned in ~0.1 s, so it almost certainly reused a wheel already in the shared uv cache (another agent, same flags) rather than compiling here; I did not observe a fresh compile, so compile time is unknown. The binary works on 3.14.2 regardless (import, stepping, all tests below).
- Script: `scratchpad/11b/spike/spike.py` (subcommands `hover | brake | yaw | tput`), outputs `*.json` next to it. One run per cell; the simulation is deterministic (no noise, `no-wind` runs show exactly 0.0 drift), so no repeats/CIs except throughput (3 reps).
- Wind method: a `WindMixin` subclass overrides `_physics()` and calls `p.applyExternalForce(drone, -1, [F,0,0], com, p.WORLD_FRAME)` **every physics substep** (PyBullet clears external forces after each `stepSimulation`, so a single call per control tick would only act for 1/5 of the time). **Pitfall found:** with `WORLD_FRAME` the position argument is a *world* point; passing `[0,0,0]` (the obvious copy-paste) applies the force at the world origin, creating a huge torque and crashing the drone (my first run: z dropped 0.94 m under even 0.02 N). The force must be applied at the drone's current base position. Anyone implementing agent 06's wind must do the same.
- Direct driving: `CtrlAviary` (action = 4 RPMs, clipped to `[0, MAX_RPM]`) + `DSLPIDControl.computeControl(target_pos, target_rpy, target_vel, target_rpy_rates)` called once per control tick from the observation of the previous tick. No speed cap exists on this path. To raise the speed cap **on VelocityAviary** I set `env.SPEED_LIMIT = v` after construction (it is a plain attribute read each step); on the direct path the cap is just `target_vel`.

### 16.1 Hover drift under constant lateral force (claim 1: CONFIRMED)

Protocol: 2 s settle at hover (no wind), then constant `F_x` for 10 s. Displacement along x relative to the start of the wind. Weight is 0.265 N, so 0.02 N = 7.5 % and 0.05 N = 19 % of weight.

| Setpoint | F = 0.02 N: x at 1 s / 5 s / 10 s | v_x last 2 s | F = 0.05 N: x at 1 s / 5 s / 10 s | v_x last 2 s |
|---|---|---|---|---|
| VelocityAviary, action `[0,0,0,0]` | 0.10 / 0.50 / **1.00 m** | 0.099 m/s | 0.25 / 1.24 / **2.51 m** | 0.254 m/s |
| Direct DSL, `target_pos = cur_pos` every tick, `target_vel = 0` | 0.10 / 0.52 / **1.00 m** | 0.100 m/s | 0.25 / 1.22 / **2.47 m** | 0.259 m/s |
| Direct DSL, **`target_pos` latched**, `target_vel = 0` | 0.05 / 0.03 / **0.015 m** | 0.000 m/s | 0.11 / 0.07 / **0.038 m** (peak 0.12 m) | -0.008 m/s |

- Drift speed = F / 0.2 N per (m/s) to two digits (0.02 N -> 0.10 m/s; 0.05 N -> 0.25 m/s), which is exactly `D_COEFF_FOR[x] = 0.2` acting alone. VelocityAviary zero-velocity and the direct `pos = cur` variant are indistinguishable, so the VelocityAviary behaviour is fully explained by `target_pos = cur_pos`. Altitude sag: 0.014-0.02 m in all wind cases (z P/I terms still see the z error of the tilted thrust; not an issue).
- Latched hold: settles within ~0.05 m (0.02 N) / ~0.12 m peak and 0.04 m residual (0.05 N, the integrator slowly removes the `F/P_COEFF = 0.05/0.4 = 0.125 m` proportional offset). **Acceptance (hover drift < 0.05 m in 10 s under 0.02 N wind): met by the latch (0.05 m peak, 0.015 m at 10 s), failed by VelocityAviary-style hover (1.0 m).** Under a 19 % weight force the un-latched hover drifts 2.5 m in 10 s and is still accelerating in displacement terms (constant 0.25 m/s), i.e. unbounded.
- Implication (confirms section 0 item 2, section 3, decision "HOVER = position latch"): VelocityAviary cannot be used as the HOVER/EMERGENCY_LAND-xy implementation in any wind scenario; with agent 06's wind this would show up as a fake "drift into obstacle" failure.

### 16.2 Braking: velocity-zero vs position-latch (claim 2: REFUTED for latch-at-current-position)

Protocol (direct DSL, VelocityAviary semantics for the approach so the cases are comparable): 1 s hover, then `target_vel = (v,0,0)` with `target_pos = cur` for 4 s (steady at ~99 % of v), then the brake variant for 6 s. Distances are along x from the brake-command point; "to 5 %" = first control tick where `v_x <= 0.05 v_entry`. `reversal` = most negative `v_x` as % of entry speed. A planning deceleration of 3 m/s^2 is used for `latch_stop`.

| v_entry | Variant | Dist. to 5 % v (m) | t to 5 % (s) | Max x reached (m) | Min v_x (m/s) / reversal | Final x at 6 s (m) | z sag / rise (m) |
|---|---|---|---|---|---|---|---|
| 0.99 | `velzero` (`pos = cur`, `vel = 0`) | 0.251 | 0.375 | 0.251 | -0.24 / 24 % | 0.191 | -0.002 / +0.011 |
| 0.99 | `latch_cur` (`target_pos = p0`) | 0.240 | 0.333 | 0.240 | **-0.72 / 72 %** | -0.005 (back at p0) | -0.003 / +0.011 |
| 0.99 | `latch_lead` (`p0 + 0.25 v`) | 0.276 | 0.438 | 0.276 (target 0.248) | -0.21 / 21 % | 0.247 | -0.001 / +0.004 |
| 0.99 | `latch_stop` (`p0 + v^2/(2*3)`, target 0.164) | 0.260 | 0.396 | 0.260 | -0.38 / 38 % | 0.168 | -0.002 / +0.006 |
| 1.98 | `velzero` | 0.563 | 0.438 | 0.563 | -0.34 / 17 % | 0.543 | 0.000 / +0.025 |
| 1.98 | `latch_cur` | 0.507 | 0.375 | 0.507 | **-1.99 / 101 %** (x min -0.24 m, i.e. flew back past start) | -0.007 | **-0.055** / +0.044 |
| 1.98 | `latch_lead` (target 0.494) | 0.581 | 0.458 | 0.581 | -0.47 / 24 % | 0.497 | -0.005 / +0.017 |
| 1.98 | `latch_stop` (target 0.651) | 0.629 | 0.521 | 0.674 | -0.15 / **7.8 %** | 0.664 | -0.003 / +0.011 |
| 2.93 | `velzero` | 0.875 | 0.458 | 0.875 | -0.98 / 34 % | 0.625 | -0.035 / +0.052 |
| 2.93 | `latch_cur` | 0.799 | 0.396 | 0.799 | **-3.15 / 108 %** (x min -0.59 m) | -0.012 | **-0.157** / +0.117 |
| 2.93 | `latch_lead` (target 0.732) | 0.892 | 0.458 | 0.893 | -0.86 / 29 % | 0.733 | -0.018 / +0.038 |
| 2.93 | `latch_stop` (target 1.430) | 1.464 | 1.458 | 1.491 | -0.05 / **1.6 %** | 1.462 | -0.003 / +0.009 |

Cross-check against VelocityAviary itself with `SPEED_LIMIT` raised (same protocol, command `[1,0,0,1]`): v_entry 0.972 / 1.968 / 2.934; distance to 5 % 0.224 / 0.566 / 0.882 m; min v_x -0.212 / -0.401 / -0.769 m/s; final x after 6 s 0.139 / 0.475 / 0.683 m. The direct `velzero` path reproduces these within ~0.03 m (differences are the slightly different entry speeds), so the direct DSL loop is a faithful stand-in for VelocityAviary.

Findings:
1. **`latch_cur` (latch at the current position while moving, what section 3/6/7 effectively assumed for BRAKE) is the worst option**: the P term (P_COEFF_FOR x = 0.4) fights the still-large velocity, the drone overshoots the latch by the full stopping distance (0.24 / 0.51 / 0.80 m), then is pulled back at up to 100-108 % of entry speed, with up to 0.16 m altitude sag at 3 m/s. **This refutes the hypothesis that a position-latch brake removes the reversal**; it makes it 3-6x larger. Stopping distance itself is only ~5-9 % shorter than `velzero`.
2. `velzero` (what VelocityAviary does): reversal 17-34 % of entry speed, no pull-back (final x stays where it stopped), no altitude problem. The previous "10-21 %" claim and the 3 m/s value (-0.333 m/s) in section 2.1 were under-estimates; re-measured -0.77 / -0.98 m/s at 3 m/s (two independent code paths).
3. `latch_lead` (`p0 + 0.25 v`, the variant written in section 15): the drone stops at the target (final error <= 0.003 m at 1-3 m/s vs target), overshoot past the target 0.03 / 0.09 / 0.16 m, but reversal 21-29 %, i.e. **no improvement over `velzero`; acceptance "undershoot < 5 %" fails**.
4. `latch_stop` (`p0 + v^2/(2 a_plan)`, `a_plan = 3`): reversal 38 % (1 m/s, target 0.164 m is *shorter* than the natural stopping distance 0.25 m, so the latch is behind the drone and pulls it back), 7.8 % (2 m/s), **1.6 %** (3 m/s, meets the < 5 % acceptance; overshoot 0.06 m). It also has the lowest altitude disturbance. The cost: at 3 m/s it stops at 1.46 m instead of 0.88 m and takes 1.46 s instead of 0.46 s (a long latch target saturates the position P term at the tilt limit, so it is slow rather than aggressive). So the latch target must sit **at or beyond the natural stopping point** for the drone to settle without reversal; a latch shorter than the natural stop (1 m/s case) reintroduces the pull-back.
5. The natural `velzero` stopping distances imply decelerations of 2.0 / 3.5 / 4.9 m/s^2 at 1 / 2 / 3 m/s (from `v^2/(2d)`); the deceleration is speed dependent, which corrects the constant `a_brake = 3` assumption (section 2.1 bullet, edited).
6. Definition caution: section 2.1's "braking distance" column (0.159 / 0.456 / 0.861 m) and this table's "distance to 5 %" (VelocityAviary rows: 0.224 / 0.566 / 0.882 m) are close at 3 m/s and differ up to 0.14 m at 1-2 m/s; I do not know agent 11's exact definition (likely a different threshold or start tick), so treat the two as the same order of magnitude, not as a contradiction of the stopping-distance conclusions. The reversal numbers are the substantive correction.

Design consequence for BRAKE (amends section 4 row 6, section 7 item 6, decision 3): implement as `velzero` (or a slewed velocity ramp, not measured here) until `|v| < 0.1 m/s`, then latch in place; or, if a no-reversal stop is required at >= 2 m/s, latch at `p + v^2/(2 a_eff)` with `a_eff` near or below the speed-dependent natural decel. Never latch at the current position while the speed is above ~0.1-0.2 m/s. Not measured: a 3 m/s^2 ramped velocity command (phase 1 as specified in the table), diagonal/vertical braking, braking with wind.

### 16.3 Yaw command: `target_rpy[2]` vs `target_rpy_rates` (claim 3)

Protocol: position latched, 1 s hover, then 3 s of yaw command at 90 deg/s (270 deg total, crosses +-180), 2 s after. Yaw unwrapped from the observation.

| Method | Yaw at end of ramp (command 270 deg) | Steady rate | Lag | Final yaw (2 s after) | Max xy deviation from latch |
|---|---|---|---|---|---|
| Integrate `target_rpy[2] += 90 deg/s * dt` each tick | 264.0 deg | 90.5 deg/s | **6.0 deg** (~0.067 s) | 270.6 deg | 0.011 m |
| Same + `target_rpy_rates = [0,0,1.571]` | 271.7 deg | 90.0 deg/s | -1.7 deg (slight lead) | 270.4 deg | 0.009 m |
| **`target_rpy_rates` only** (`target_rpy[2]` constant) | **9.1 deg, then returns to -0.9 deg** | 0 | not a rotation command | -0.9 deg | 0.011 m |
| Step of +90 deg in `target_rpy[2]` | t90 = 0.33 s | - | overshoot 1.9 deg | 90.5 deg | 0.011 m |

- **`target_rpy_rates` is not a yaw-rate command** [V-code, now V-meas]: the docstring says so and the measurement agrees: alone it produces a transient of ~9 deg and no sustained rotation. A yaw rate must be realised by integrating a yaw *angle* setpoint in the executor (`yaw_des += omega * dt`), optionally with `target_rpy_rates = omega` as a feed-forward that removes the 6 deg lag.
- Yaw is cleanly decoupled from translation in this model: position hold error stays under 1.2 cm during the ramp at 90 deg/s and across the +-180 deg wrap. A 90 deg turn takes ~0.33 s to 90 % with < 2 deg overshoot, so a discrete YAW_LEFT/RIGHT action (section 9) is cheap in time relative to a 1-3 s decision latency. The 3 s ramp is a single point; higher rates and yaw during forward flight were not tested.

### 16.4 Control ticks per second (claim 4)

1 drone, `gui=False`, `record=False`, `obstacles=False`, `Physics.PYB`, pyb 240 Hz / ctrl 48 Hz, 3000 ticks per rep, 3 reps, `time.perf_counter`, no `sync()` pacing (not called by `step`). Each tick includes 5 PyBullet substeps, `_updateAndStoreKinematicInformation`, observation, reward/terminated computation of the class.

| Loop | Ticks/s (3 reps) | Median | x real time at 48 Hz |
|---|---|---|---|
| `CtrlAviary` + `DSLPIDControl.computeControl` (my direct loop) | 5776 / 6081 / 5908 | **5.9 k** | ~123 x |
| `VelocityAviary.step` (internal DSLPID) | 5704 / 5711 / 5701 | 5.7 k | ~119 x |
| `CtrlAviary.step` with constant hover RPM (no PID) | 22696 / 22564 / 22678 | 22.7 k | ~472 x |

- **A full control loop costs ~170 us/tick; the env step alone ~44 us; so `DSLPIDControl` (scipy `Rotation` + numpy) is ~125 us, about 3/4 of the loop.** 5.9 k ticks/s is 28 k physics substeps/s with a PID, 113 k/s without. The "20 kHz" of the initial plan is therefore roughly reproducible only as physics-only stepping (and is unrelated to control rate); with the controller it is 5.9 kHz control ticks for one drone on one core. A 10 s flight at 48 Hz (480 ticks) costs ~0.08 s of wall time, so 10 k flights = ~14 min single process, before scenario events, sensing, DB writes and decision I/O (agents 02/09 should treat these as the upper bound for one core).
- Single process, Python 3.14.2; no thermal or multi-core scaling measured.

### 16.5 Summary of what changed in this file

| Earlier claim (where) | Status | Edit made |
|---|---|---|
| Un-latched pos=cur hover drifts at F/0.2 m/s (0, 3) | confirmed, exact | tags, no numeric change needed |
| VelocityAviary brake reverses, -0.21 m/s at 1 m/s (0, 2.1) | confirmed at 1-2 m/s; **3 m/s value wrong (-0.333 -> -0.77/-0.98)**; range 17-34 %, not 10-21 % | 2.1 bullet corrected |
| Position-latch braking removes reversal (2.1 hypothesis; 0 item 2; 7 item 6; 4 row 6; decision 3) | **refuted** for latch-at-current; `latch_stop` works at >= 2 m/s at the price of stopping distance | items amended |
| a_brake = 3 m/s^2 conservative (2.1) | conservative at 2-3 m/s, **not at 1 m/s** | 2.1 bullet corrected |
| `target_rpy_rates` as yaw-rate command (implicitly possible) | **false**; integrate `target_rpy[2]` | section 16.3 |
| Python 3.14.2 pybullet unknown (open question 6) | works (with the `CFLAGS` flag, cache caveat) | question 6 answered |
| Throughput not measured | 5.9 k ticks/s direct, 22.7 k env-only | 2.1 bullet, 16.4 |

## Sources (listed here for convenience; the three required closing sections above are Decisions, Open questions, Follow-up spikes)

- gym-pybullet-drones source (commit `7ebad1e`, 2026-09-06, pip version 2.2.0), https://github.com/utiasDSL/gym-pybullet-drones : `gym_pybullet_drones/control/DSLPIDControl.py`, `control/BaseControl.py`, `control/CTBRControl.py`, `envs/VelocityAviary.py`, `envs/BaseAviary.py`, `examples/pid_velocity.py`, `utils/utils.py`, `assets/cf2x.urdf` (read 2026-09-30).
- pybullet 3.2.7 sdist, https://pypi.org/project/pybullet/ (no macOS wheel; built locally, 2026-09-30).
- PX4 Offboard mode docs, https://docs.px4.io/main/en/flight_modes_mc/offboard.html (fetched 2026-09-30; excerpt did not include the setpoint-rate/timeout numbers, so none are claimed).
- Safety filters / CBFs for quadrotors, https://arxiv.org/pdf/2502.04101 (Composite CBF safety filter), https://arxiv.org/pdf/2303.15871 (collision-cone CBF), https://arxiv.org/pdf/2010.09819 (CBF vs APF) (search results only, abstracts not deeply read).
- LLM-robot shields, https://arxiv.org/pdf/2503.03911 (RoboGuard, reachability-based), https://arxiv.org/html/2503.07885v2 (Safety guardrails for LLM-enabled robots) (search results only).
