# 12 - Visualisation, demo, analysis tooling

Author: research agent 12. All web sources accessed 2026-09-30. Hardware for spikes: Apple M4 (macOS product version 27.0 / build 26A428, Command Line Tools SDK 26), uv 0.9.18, scratch venvs under `.../scratchpad/12` (Python 3.14.2 `v314`, Python 3.12.12 `v312`). No API keys used, no repo files touched except this one.

---

## 0. TL;DR decision

| Need | Choice | Why (one line) |
|---|---|---|
| Engineer replay/debug of one scenario, all arms side by side on one shared clock | **Rerun 0.38.1** (`.rrd` per scenario, one recording, entity roots per arm, saved blueprint) | Verified installs on 3.14.2/arm64, 100k ticks logged in 0.95 s, time-synchronised multi-view for free |
| The "killer demo" (Gantt of decision loops + replay + collision moment) | **Single-file HTML player (vanilla Canvas 2D + SVG Gantt), fed by a JSON export of stored trajectories, served by FastAPI static route**; three.js r186 is an optional phase-2 upgrade | Needs a custom decision-timeline overlay Rerun cannot draw natively; zero Python deps; works offline by double-click |
| Hero MP4 for slides/README | **Matplotlib Agg -> ffmpeg (imageio-ffmpeg)** for 2-3 curated clips; PyBullet camera render only if a "real" 3D look is wanted | Measured 7.2 fps render (10 s clip = ~42 s), deterministic, headless |
| Statistical report | **DuckDB + pandas -> matplotlib figures -> Jinja2 -> one self-contained HTML (+ PDF via browser print)** | All install on 3.14.2; no browser dependency for figures |
| Live progress | **FastAPI native SSE** (`fastapi.sse.EventSourceResponse`, present in the installed FastAPI 0.142.2) + tiny HTML page | No WebSocket needed, one-directional |
| Rejected as primary | PyBullet GUI, Foxglove, Plotly-animated 3D, live-rendered video at record time | See section 1 |

---

## 1. Option verification (measured where possible)

### 1.1 Rerun
- **Verified**: `rerun-sdk 0.38.1` (PyPI, released 2026-09-16 per search snippet; build stamp in `rerun --version`: "built 2026-09-16") installs into a **Python 3.14.2 venv on macOS arm64** via the wheel `rerun_sdk-0.38.1-cp310-abi3-macosx_11_0_arm64.whl` (abi3, so no per-interpreter build). `requires_python >=3.10`; changelog says 3.10 support is deprecated and dropped in 0.39. Source: https://pypi.org/project/rerun-sdk/ , https://rerun.io/docs/changelog/changeset-0-38
- **Measured** (v314, M4): `rr.log` per tick with 2 entities = 2000 ticks in 3.60 s (1.8 ms/tick: fine for live, too slow for bulk). `rr.send_columns` (columnar) for 100 000 ticks x (Points3D + Scalars) = **0.95 s, 4.8 MB `.rrd`** (random data, worst case for compression). Real trajectories will be smaller.
- Verified API surface in 0.38.1: `rr.save`, `rr.send_blueprint`, `rr.send_columns` with `rr.TimeColumn("sim_time", duration=...)`, `rr.serve_grpc`, `rr.serve_web_viewer`, blueprint `Spatial3DView`/`TimeSeriesView`/`Horizontal`/`Vertical`. I built and saved a blueprint with two side-by-side 3D views (`/arms/jev`, `/arms/llm`) plus a shared time-series panel and it wrote an `.rrd` (82 KB) without error. **Not verified**: pixel rendering of the viewer (no visual check; `rerun` native viewer is a separate binary `rerun-cli 0.38.1`, present and runs).
- 0.38 caveats from release notes/issues: viewer catalog now lazy-loads `.rrd`; known issue #12938 - several `.rrd` files that *share a recording ID* no longer merge (https://github.com/rerun-io/rerun/issues/12938). **Mitigation**: one `.rrd` per scenario containing all arms (what I recommend anyway), never split one recording over files.
- **Telemetry**: first run prints an anonymous-usage-analytics notice (opt-out with `rerun analytics disable`). The project is "local-only": disable it in the worker environment. (Inference: not a blocker; it says no data sent on first session.)
- License: open source (Apache-2/MIT dual, per project page; not re-verified this session).
- Weakness: no native Gantt/"bar" time-interval primitive; closest is `Scalars` step plots, `Boxes2D`, `TextLog`, `TextDocument`, `BarChart`. The latency-bar overlay is therefore approximated (step series + TextLog), not exact. -> why the demo is a custom page.

### 1.2 PyBullet GUI on macOS (and the bigger finding: install)
- **Install finding (important for agents 02/09/14)**: PyPI `pybullet 3.2.7` has **no macOS wheels at all** (only manylinux x86_64 + sdist; I listed every file). `uv pip install pybullet` compiles the sdist and **fails** on this machine on both 3.14.2 and 3.12.12: bundled zlib (`examples/ThirdPartyLibs/zlib/zutil.h:128`) `#define fdopen(fd,mode) NULL` collides with the macOS SDK prototype (`_stdio.h:322`, "expected identifier or '('"). **Workaround that worked**: `CFLAGS="-Dfdopen=fdopen" uv pip install pybullet` (makes the `#ifndef fdopen` guard false). Build time 74 s, then `p.connect(p.DIRECT)` OK on **Python 3.14.2**. gym-pybullet-drones 2.2.0 (from git) then installed in 21 s with gymnasium 1.3.0, numpy 2.5.3, stable-baselines3 2.9.0, torch 2.14.0 on the same 3.14.2 venv.
- **GUI**: `p.connect(p.GUI)` on the M4 succeeded and printed "Version = 4.1 Metal - 91.7 / Vendor = Apple / Renderer = Apple M4" (OpenGL 4.1 via Apple's GL-on-Metal shim), then **hung for >=40 s** at the first loadURDF/step loop when launched non-interactively from a subprocess (killed). Not proven to be a general bug (launched without a foreground GUI session), but it matches the long-standing reports: GUI freezes/pinwheel on Apple Silicon and "OpenGL on Mac must be in main thread; use a separate process" (https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=13281 ; https://github.com/bulletphysics/bullet3/issues/2014 ; https://github.com/bulletphysics/bullet3/discussions/3844). Also `getCameraImage` hangs after many calls with ER_TINY_RENDERER for some users (https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=13395).
- Conclusion: GUI is **not a dependency** of anything; do not build the demo on it. Keep it an optional manual debug aid behind a flag. The OpenGL-deprecation trajectory makes it a long-term risk (inference).
- **Headless render cost (measured, TinyRenderer CPU, M4, tiny scene)**: 320x240 = 18.5 ms/frame, 640x480 = 60.7 ms, 1280x720 = 193 ms. So 30 fps video at 720p is ~5 fps render.
- **DIRECT stepping (measured)**: gym-pybullet-drones `CtrlAviary`, 1 x CF2X, `pyb_freq=240`, `ctrl_freq=48`, hover RPM, no record/gui: **~10 000 control ticks/s = ~50 000 physics steps/s** (2000-tick run, includes `_getDroneStateVector` per tick). This is one data point for agent 02; it neither confirms nor refutes "20 kHz/80 drones/4 envs" (different config), but shows a 30 s flight (1440 ticks at 48 Hz) costs ~0.15 s of pure sim, so recording trajectories is essentially free and recording video is ~1000x more expensive than the physics.

### 1.3 Foxglove
- `foxglove-sdk 0.28.0` and `mcap 1.5.0` install on 3.14.2 (verified; a macOS wheel for cp314 or abi3 was resolved by uv, exact tag not inspected).
- **Status change**: Foxglove 2.0 made the studio **closed source going forward** (1.x source remains on GitHub), desktop app "not discontinued", local MCAP and websockets work offline but **first launch requires sign-in**; free plan = 3 users, 10 GB, 5 devices, 1 project. Sources: https://discourse.openrobotics.org/t/foxglove-2-0-integrated-ui-new-pricing-and-open-source-changes/36583 , https://foxglove.dev/pricing
- Verdict: good tool, account-gated and closed; poor fit for "local-only, reproducible, shareable" and adds a format (MCAP) duplicating our Parquet. **Rejected as primary.** Keep MCAP export as a possible later nicety (mcap lib installs).

### 1.4 Plotly
- `plotly 7.1.0` installs on 3.14.2. Static export via `kaleido 1.4.0` **worked (7.7 s for one PNG) only because Google Chrome is installed** (kaleido 1.x drives Chrome). Animated 3D via `Frame`s: embeds every frame in the HTML; fine for 1 marker, but for 3 arms x 1440 ticks x multiple traces it is tens of MB and sluggish (inference from Plotly's frame model, not measured at scale). **Decision**: use Plotly.js only inside the browser dashboard for interactive curves; do not use Plotly for replay and do not require kaleido for the report.

### 1.5 three.js
- Latest r186 / npm `three@0.186.x` (published 2026-09-08; release cadence 8-11 weeks) - https://github.com/mrdoob/three.js/releases/tag/r186 . Usable with no build step via ES modules + `importmap` from a CDN (or vendor the file into `app/static/`). Not needed for the first demo: a 2D top-down + side-elevation canvas conveys avoidance unambiguously and is ~300 lines. Phase-2 option if a 3D look is wanted.

### 1.6 Matplotlib / ffmpeg
- `matplotlib 3.11.2`, `imageio-ffmpeg 0.6.0` (bundled ffmpeg binary), `numpy 2.5.3`, `pandas 3.0.6`, `duckdb 1.5.6`, `scipy 1.18.1`, `statsmodels 0.15.0`, `polars 1.44.2`, `pyarrow 25.0.1`, `jinja2 3.1.6` all installed cleanly on 3.14.2/arm64.
- **Measured**: 3 top-down panels + Gantt strip at 1280x720, 300 frames via `FuncAnimation` + `FFMpegWriter` libx264 = **41.7 s (7.2 fps)**, 0.09 MB output. Acceptable for curated clips; too slow for mass video. imageio writer of pre-rendered 1280x720 frames: 120 frames encode in 1.22 s (encode is not the bottleneck, drawing is).

### 1.7 FastAPI SSE
- Installed FastAPI in scratch resolved to 0.142.2 (repo requires >=0.136.3); `fastapi.sse` exposes `EventSourceResponse`, `ServerSentEvent` natively; `sse-starlette 3.5.0` also installs (not needed). Verified by import only, not a live stream test.

---

## 2. Architecture: record cheap, render later

Principle: **the simulator never renders during a batch run.** It writes a compact trajectory + decision log. Everything visual (Rerun, HTML demo, MP4, plots) is derived from stored data, so visual options can change without re-running flights, and replay is byte-stable.

```
 sim worker (DIRECT, no camera) --> flights/<flight_id>.parquet   (ticks)
                                \-> decisions rows -> SQLite (see 10)
                                \-> scenario JSON (see 06)
 derive step (idempotent, keyed by scenario_id):
   - rerun_export   : scenarios/<scn>.rrd                     (engineer view)
   - demo_export    : demo/<scn>.json (+ gzip)                (HTML player)
   - mp4 (optional) : matplotlib Agg -> ffmpeg                (hero clips)
   - report         : DuckDB queries -> matplotlib -> HTML/PDF
 FastAPI: /api/v1/... JSON endpoints + /demo static + /runs/{run_id}/progress (SSE)
```

Optional "real 3D frame" path: for a hero clip only, open a DIRECT client, load the plane/obstacle/CF2X URDFs, for each stored pose call `resetBasePositionAndOrientation` and `getCameraImage` (no physics stepping needed; ~18 ms at 320x240, ~61 ms at 640x480) -> ffmpeg. 10 s @30 fps, 640x480 = ~18 s per arm.

Live rendering during the run (PyBullet `record=True`/GUI) is **rejected**: it cuts throughput by ~2-3 orders of magnitude and ties timing to the wall clock.

Replay clock rule (ties to agent 08): every viewer is driven by **sim_time**; wall-clock latency appears only as an *annotated bar* whose length is the **modelled/recorded latency converted to sim seconds**. Wall-clock API time must never drive animation speed.

---

## 3. Data contract for replay

Goal: a viewer must be able to reproduce, for every arm and every tick, exactly what the drone did, what it had been told, what it saw, what was in flight, and why it was overridden - **without re-simulating**.

### 3.1 Files / tables
- `scenario` JSON (already owned by agent 06): seed, world geometry, event schedule, target path. Copy its hash into every flight row.
- `flights/<flight_id>.parquet`: one row per **control tick** (ctrl_freq, e.g. 48 Hz). zstd, float32.
- `decisions` table (SQLite, agent 10) or `flights/<flight_id>.decisions.parquet`: one row per **decision request**.
- `entities` rows are included in the tick file (obstacles, target, other drone) so the viewer needs only one file per flight.

### 3.2 Tick table schema (`flights/<flight_id>.parquet`)

| column | dtype | note |
|---|---|---|
| tick | int32 | 0..N-1 |
| t_sim | float64 | seconds, `tick/ctrl_freq`; the ONLY replay clock |
| x,y,z | float32 x3 | world frame, m (ENU, z up, PyBullet world) |
| qx,qy,qz,qw | float32 x4 | attitude (skip roll/pitch plots otherwise) |
| vx,vy,vz | float32 x3 | m/s |
| wx,wy,wz | float32 x3 | rad/s (optional but cheap; needed for oscillation metric) |
| rpm0..rpm3 | float32 x4 | for energy and saturation detection |
| sp_x,sp_y,sp_z | float32 x3 | position setpoint (or NaN) fed to PID |
| sp_vx,sp_vy,sp_vz | float32 x3 | velocity setpoint (agent 11 semantics) |
| active_action | int8 | enum 0..10 currently *applied* (latched) |
| action_source | int8 | 0 none/initial, 1 provider, 2 shield override, 3 timeout fallback, 4 rule-FSM/baseline |
| active_decision_id | int32 | FK to decision that produced `active_action` (-1 if none) |
| inflight_decision_id | int32 | decision request currently pending (-1 if none); gives the bar overlap directly |
| obs_age_ticks | int16 | age of the observation the active action was derived from |
| battery_frac | float32 | true value |
| wind_x,wind_y,wind_z | float32 x3 | true applied wind |
| gps_ok | int8 | flag; `gps_x,gps_y,gps_z` float32 = what the agent was *told* (only if differing) |
| obs_noise_sigma | float32 | sensor noise level currently in effect |
| clearance_min | float32 | true min distance to any hazard surface (m) |
| clearance_obj_id | int16 | which hazard |
| time_to_collision | float32 | closing-speed-based estimate (inf -> large sentinel 1e3) |
| hazard_visible | int8 | hazard inside sensing envelope per the agent's sensors |
| in_collision | int8 | contact flag this tick |
| ent_<k>_x,y,z | float32 x3 per entity | obstacle(s), moving target, other drone; k = stable entity index; constant entities stored once in scenario JSON instead |

Size estimate: ~45 float32 columns x 1440 ticks (30 s at 48 Hz) = ~260 KB raw, **~60-90 KB zstd Parquet** (inference; smooth signals compress well). 10 000 scenarios x 6 arms x 75 KB = **~4.5 GB**. Policy: keep full-rate for all flights (4.5 GB is affordable on disk) OR decimate non-hero, non-collision flights to 12 Hz for a 4x saving; either way never store decisions decimated.

### 3.3 Decision table schema (one row per provider call, including timeouts and cache hits)

| column | dtype | note |
|---|---|---|
| decision_id | int32 | unique per flight |
| flight_id, arm | FK / text | |
| tick_obs | int32 | tick of the observation snapshot sent (sense time) |
| t_obs_sim | float64 | |
| tick_request | int32 | when request was issued (may be > tick_obs if batching) |
| tick_response | int32 | tick at which the result was *available* in sim time (request + modelled/injected latency ticks) |
| tick_applied | int32 | tick the action actually changed the setpoint (>= response; after shield/arbitration) |
| latency_ms_total | float32 | wall-clock measured (real API) OR sampled value, see `latency_mode` |
| latency_mode | int8 | 0 zero/lockstep, 1 fixed, 2 sampled from recorded distribution, 3 live measured real-time |
| lat_network_ms, lat_ttft_ms, lat_gen_ms, lat_queue_ms | float32 | split per agent 08; NaN if not measurable |
| latency_ticks_effective | int16 | what the sim actually used |
| action_raw | text | verbatim model output (truncated 256 chars) |
| action | int8 | parsed enum or -1 |
| valid | int8 | parsed+schema-valid |
| outcome | int8 | 0 ok, 1 parse_fail, 2 timeout, 3 http_error, 4 superseded (dropped because newer arrived), 5 shield_rejected |
| source | int8 | 0 live API, 1 cache hit, 2 local model, 3 classical controller, 4 FSM |
| fallback_action | int8 | action used instead if outcome != ok |
| obs_hash / prompt_hash | text(16) | dedupe + cache key |
| model_id, provider, route | text | as returned by API where available |
| tokens_in, tokens_out | int16 | |
| label_correct_action | int8 | oracle label at tick_obs (agent 11), -1 unknown |
| label_correct_at_apply | int8 | oracle label at tick_applied; difference = "stale decision" metric |
| shield_reason | int8 | 0 none, else code (TTC brake, geofence, battery, ...) |

### 3.4 Event table (from scenario; replicated per flight for joins)
`event_id, flight_id, type(enum: OBSTACLE_APPEAR, WIND_CHANGE, TARGET_MOVE, SENSOR_NOISE, BATTERY_DROP, GPS_LOSS, OTHER_DRONE), t_trigger_sim, tick_trigger, duration_s, params_json, severity, hazard_entity_idx, feasibility_margin_s` (the time budget = distance to hazard / closing speed minus physical braking time, agent 06).

### 3.5 Flight summary row (metrics; agent 08 definitions) - denormalised for plots
`flight_id, scenario_id, arm, seed, latency_setting_ms, success, collided, t_collision, time_to_target, path_length, min_clearance, near_miss, t_first_reaction_sim (first tick active_action changes to a hazard-appropriate action after the triggering event), reaction_time_s = that - t_trigger, recovery_time_s, n_decisions, n_invalid, n_timeouts, energy_proxy, oscillation_idx, sim_wall_s, sim_speed_x, decisions_per_wall_s, wall_clock_s_total, git_sha, package_versions_hash, hardware_tag`.

### 3.6 Invariants a replay reader may rely on
1. `tick_obs <= tick_request <= tick_response <= tick_applied`.
2. `active_decision_id` at tick t = the last decision with `tick_applied <= t` (or -1).
3. A flight file fully determines the viewer; no table outside {tick file, decision rows, scenario JSON} is required.
4. Reproduction check: rerunning (scenario JSON, action log with `tick_applied`) through the simulator reproduces the trajectory within tolerance (determinism test owned by agent 02/08; flag `trajectory_reproducible` per flight if verified).

---

## 4. The killer demo

### 4.1 Claim discipline (do not get caught by time-model artefacts)
The plan's sketch (bars: PID/MPC short, Jev shorter, LLM long) is a statement about latency, not about intelligence. The demo must show **the same scenario, same seed, same injected event, and latency values labelled as measured vs modelled**, plus a **latency-matched ghost** (Jev-quality policy given LLM latency, or LLM-quality policy given Jev latency) if available, so a viewer can see "speed alone" vs "decision quality". Also show the **avoidance budget** (time from hazard becoming visible to unavoidable impact). If the budget is shorter than the LLM latency, the collision is physics, and the demo should say so. Consistent with agent 08's "latency knee" warnings.

### 4.2 Layout (1280x720; HTML player; same layout for MP4)

```
+-----------------------------------------------------------------------------------+
| Scenario 0421  seed 91  event: OBSTACLE_APPEAR @ t=6.20s  budget 0.85 s  [x1 > ||] |
+----------------------+----------------------+-------------------------------------+
| PID + reactive       | JEV                  | SLOW LLM                            |
|  top-down (x,y)      |  top-down (x,y)      |  top-down (x,y)                     |
|      ..obstacle##    |      ..obstacle##    |      ..obstacle##                   |
|   o--->              |    o~~>              |    o--->   X  COLLISION t=7.02      |
|  clearance 0.12 m    |  clearance 0.94 m    |  clearance 0.00 m                   |
|  action: FORWARD     |  action: AVOID_LEFT  |  action: FORWARD (stale 0.81 s)     |
+----------------------+----------------------+-------------------------------------+
| side view z(t) mini strips                                                        |
+-----------------------------------------------------------------------------------+
| DECISION LOOP TIMELINE (shared sim_time axis, playhead |)                          |
| t:        5.0      5.5      6.0 |6.2   6.5      7.0      7.5                      |
| event                            v OBSTACLE_APPEAR                                |
| PID/MPC  [s|c]...[s|c][s|c]      [s|c]=======>HOVER/BRAKE? (compute 11 ms)        |
| JEV      [obs]-lat 48ms->[AVOID_LEFT applied @6.27]  ....                         |
| LLM      [obs]-------------------- lat 1.45 s -------------------->[AVOID @7.65] X|
| legend: [ ] = obs snapshot -> bar = in-flight request -> > = action applied       |
|         red tick = action was stale/incorrect at apply time; X = collision        |
| budget:  |<----------- 0.85 s avoidance budget ----------->|  (hazard visible -> |
|          last point where braking still avoids impact)                            |
+-----------------------------------------------------------------------------------+
| mini-metrics: reaction_time 0.07s | 0.06s | 1.45s   collision N|N|Y   path  m m m |
+-----------------------------------------------------------------------------------+
```

Element spec:
- Playback controls: play/pause, scrub, speed 0.1x/0.25x/1x/4x; arrow keys step **one control tick**; click a decision bar to jump to `tick_obs` and show the raw prompt/output (from `decisions`).
- Bars: start x = `t_obs_sim`, end x = `tick_response / ctrl_freq`, optional thin extension to `tick_applied` if the shield delayed it; colour by `outcome` (ok/parse_fail/timeout); red outline if `label_correct_at_apply != action`.
- Budget band: from `hazard_visible` first tick to `t_trigger + feasibility_margin_s`; shaded.
- Annotation strip under each top-down panel: current `active_action`, `obs_age_ticks` converted to ms, `action_source` (shield override badge).
- Footer tags: `latency_mode` for each arm ("measured" / "sampled from 300 real calls" / "fixed 1450 ms") - always visible.
- Colour: one categorical colour per arm, identical across demo and report plots; collisions in red only.

### 4.3 Variants
1. "Latency sweep film": same scenario, Jev arm replayed at 0/50/100/300/1000/3000 ms injected delay; 6 tiles; collision tile turns red at the knee.
2. "Gauntlet": 25 random scenarios tiled 5x5 per arm, collisions red, final tally - works because replays are just JSON.

### 4.4 Build plan (KISS)
`demo_export.py` (reads parquet + decisions -> `demo/<scn>.json`: per-arm downsampled to 30 Hz poses, decision intervals, events, budget) -> `app/static/demo/index.html` (canvas). ~300-400 lines JS, no bundler. FastAPI route `GET /api/v1/scenarios/{scenario_id}/replay` returns the JSON via `success_response`; static file mount for the page (follow the repo convention: URL ends with a resource word, hence `/replay`, `/scenarios/{id}/replay`).

---

## 5. Analysis plots (exact axes and stat overlays)

Conventions for all: colour per arm fixed; n printed in legend; every proportion CI = **Wilson 95%**; every continuous mean/median CI = **BCa bootstrap 95% (10 000 resamples, resample scenarios not flights for paired designs)**; paired tests use scenario as the pairing unit; p-values Holm-corrected within a figure family; heavy-tailed latency shown as quantiles, never mean +- sd; warm-up decisions (first N calls per process) excluded and the exclusion stated in the caption. Data from DuckDB views over `flights` and `decisions`.

| # | Plot | X axis | Y axis | Series | Overlays / stats |
|---|---|---|---|---|---|
| P1 | Decision-latency ECDF | `latency_ms_total`, **log x** 1 ms - 10 s | cumulative fraction 0-1 | one line per provider/model; dashed = network only, solid = total | vertical lines p50/p95/p99 markers with labels; DKW 95% band; inset table of p50/p95/p99/max; timeouts drawn as right-censored tick at the timeout value |
| P2 | Latency split stack | provider/model | ms (stacked median network / queue / TTFT / generation) | stacked bars | error bars = p25-p75 of total; marks N |
| P3 | **Collision rate vs injected latency** (core) | latency setting, ms, symlog: 0,10,50,100,300,1000,3000 | collision probability 0-1 | one line per policy quality (Jev-quality/FSM/PID+avoid/RL/LLM) | Wilson 95% ribbons; paired difference vs reference arm in a lower subpanel with BCa CI and McNemar p (Holm); vertical shaded band = the feasibility-budget distribution (p10-p90 of `feasibility_margin_s`) to expose the physical knee; grey line = always-hover baseline, black dashed = oracle |
| P4 | Reaction time distribution | arm | `reaction_time_s` (log y) | violin/box + jittered points (<= 500 pts) | median + BCa CI; horizontal line = budget median; Wilcoxon signed-rank p between paired arms; % of flights with reaction > budget annotated |
| P5 | Pareto: safety vs speed | median `time_to_target` (s) | collision rate (%) | one marker per arm-x-latency; lines connecting latency settings | 2-D error crosses (BCa x, Wilson y); highlight Pareto-optimal set; dominated arms faded; optional third channel = marker size = decisions/cost |
| P6 | Pareto: safety vs cost | cost per 1000 decisions (USD) or API cost per flight | collision rate | per provider | same crosses; log x |
| P7 | Per-event breakdown (heat grid) | event type (7 columns + "none") | arm (rows) | cell colour = collision rate (sequential palette), cell text = `k/n` | Wilson CI printed in a second small-font line; significance star vs reference arm (McNemar, Holm); second grid for median `recovery_time` |
| P8 | Severity dose-response | event severity bin (e.g. wind m/s, obstacle closing speed m/s, noise sigma) | collision rate | per arm | Wilson ribbons; logistic-fit line with 95% CI (statsmodels); point at which curves cross annotated |
| P9 | Decision-loop Gantt (single flight, from data, not mock) | sim_time s | rows: arms; sub-rows: in-flight decisions | bars = `[tick_obs/ctrl_freq, tick_response/ctrl_freq]`, colour = outcome | playhead-less static version of the demo strip; event marker and collision X; budget band. Produce for the 3 best/worst/median scenarios automatically |
| P10 | Throughput | arm | decisions / wall second (log y) and sim_speed_x (second panel) | bar + dots per worker process | p50/p95 over runs; annotated with "API-time excluded" vs "included" (agent 08 rule: report both, never mix) |
| P11 | Path-efficiency scatter | straight-line distance to target (m) | `path_length` (m) | per arm, colour by collided | y=x reference line; LOWESS per arm; points that failed hollow |
| P12 | Staleness impact | `obs_age_ticks`*1000/ctrl_freq (ms), 50 ms bins | fraction of decisions where `label_correct_at_apply == action` | per arm | binomial Wilson band; count histogram as secondary rug; shows decision "rot" vs age |
| P13 | Invalid/timeout | arm/model | rate (%) of `outcome in (parse_fail, timeout, http_error)` | bars | Wilson CI; break out by outcome type |
| P14 | Sanity panel | scenario index | metric | always-hover, random, oracle | shows bounds so readers can see headroom |

Every figure saved as SVG + PNG (200 dpi) with a caption sidecar JSON (n, test, correction, exclusion, git SHA, manifest hash).

Stats implementation: `scipy.stats.wilcoxon`, `statsmodels.stats.proportion.proportion_confint(method="wilson")`, `statsmodels.stats.contingency_tables.mcnemar(exact=True)` for small discordant counts, `scipy.stats.bootstrap(method="BCa")`, `statsmodels.stats.multitest.multipletests(method="holm")` - all available in the verified 3.14.2 env (scipy 1.18.1, statsmodels 0.15.0).

---

## 6. Report pipeline

```
scripts/make_report.py <run_id>           # constants at top (repo Scripts rule), no CLI args
  1. load   : DuckDB attach SQLite file (sqlite_scanner) or read exported Parquet
              SELECT ... FROM flights, decisions, events  (views defined once in report/views.sql)
  2. freeze : write reports/<run_id>/manifest.json
              (git SHA, uv.lock hash, python/pybullet/numpy versions, hardware, OS, run config,
               seeds, scenario-set hash, latency mode per arm, row counts)
  3. stats  : stats.py -> tidy DataFrames (one per figure) saved as CSV next to each figure
  4. figures: figs.py (matplotlib Agg, fixed rcParams, SVG+PNG)
  5. render : Jinja2 template -> reports/<run_id>/report.html (figures inlined as SVG, tables as HTML,
              methods + caveats section auto-filled from manifest)
  6. pdf    : browser "Print to PDF" of report.html (no extra dependency), or WeasyPrint only if asked
  7. extras : demo_export for top-N scenarios (best/median/worst per arm, all collisions for headline arm)
              rerun_export for the same; optional 2-3 MP4 hero clips
```

Guardrails: report refuses to render figures for a comparison lacking a paired design or with fewer than the power-analysis target n (agent 08) without a visible "underpowered" banner; every headline number links to its CSV; the report states the latency mode prominently; speed claims carry "sim time" vs "wall time" labels.

Layer placement (per repo guide): pure plotting/stat functions live in `utils/` (stateless) or an analysis package; DB reads go through a repository class; the FastAPI layer only serves results. Suggested (for agent 14 to reconcile): `app/analysis/{views.sql,stats.py,figures.py}`, `app/services/report_service.py`, `app/repositories/flight_repository.py`, `app/api/v1/reports.py`, `scripts/make_report.py`.

---

## 7. Dashboard endpoint and live progress

Endpoints (all responses wrapped with `success_response`/`error_response`; trailing path segment is a resource word per repo rule):

| Method + path | Purpose |
|---|---|
| `GET /api/v1/runs/{run_id}/summary` | per-arm counts, collision rate + Wilson CI, medians; JSON for the dashboard |
| `GET /api/v1/runs/{run_id}/progress` | **SSE** stream: `event: progress` every 1 s: `{flights_done, flights_total, per_arm:{...}, flights_per_hour, eta_s, api_inflight, rate_limited_429s, worker_states}` ; `event: done` |
| `GET /api/v1/runs/{run_id}/latency-quantiles` | p50/p95/p99 per provider (for P1/P2 in browser) |
| `GET /api/v1/scenarios/{scenario_id}/replay` | demo JSON (section 4.4); `?arms=jev,llm` |
| `GET /api/v1/flights/{flight_id}/decisions` | paginated decisions (shared pagination utility, never unbounded) |
| `GET /api/v1/runs/{run_id}/report` | serves the generated `report.html` |
| Static `/demo/` | the replay player; `/dashboard/` = live page |

SSE rather than WebSocket: one-way, auto-reconnect in the browser, FastAPI has native `EventSourceResponse` (verified import in the installed version). Progress source: the SQLite job table (agent 09/10) polled every 1 s by an async generator; workers never call the API. Dashboard page: vanilla JS + Plotly.js (CDN) for the running collision-rate-with-CI chart and a progress bar; no Node build.

---

## 8. Recording cost summary (measured/derived)

| Approach | Per-flight cost (30 s flight) | Verdict |
|---|---|---|
| Parquet trajectory+decisions (always on) | ~0.15 s sim + <5 ms write (derived; sim speed measured) | baseline, always |
| PyBullet `getCameraImage` 320x240 every tick @48 Hz during run | 1440 x 18.5 ms = **26.6 s** (170x the sim cost) | never in batch |
| Offline re-render from poses, 640x480 @30 fps | 900 x 60.7 ms = **55 s** | hero clips only |
| Matplotlib MP4, 3 panels+Gantt, 720p @30 fps | 900 frames / 7.2 fps = **125 s** | hero clips only |
| Rerun `.rrd` via `send_columns` | ~0.01-0.05 s (100k ticks = 0.95 s) | all interesting flights |
| HTML demo JSON | ms; 30 Hz decimated, ~100-200 KB gz per scenario (inference) | all interesting flights |

---

## 9. Contradictions / notes relative to `docs/initial-plan.txt`
- The plan assumes the simulator "just works" on Apple Silicon. **Measured reality**: `pybullet` has no macOS wheel; source build fails on this toolchain without `CFLAGS="-Dfdopen=fdopen"` (works after). Plain `uv pip install pybullet` or `uv pip install -r requirements.txt` will fail on a clean Mac; document the workaround in the repo README and CI.
- The plan treats "visually impressive simulator" as unnecessary; this agrees: nothing here needs the PyBullet GUI, which also hung in my non-interactive test.
- "~20 kHz with 80 drones/4 envs": not reproduced or refuted (different config); 1-drone hover DIRECT measured ~50 k physics steps/s (~10 k ctrl ticks/s at 48 Hz). Agent 02 owns this.
- "Apple Silicon / macOS 26 tested": the dependency that matters (pybullet) is not distributed as a macOS wheel; the claim does not cover the install problem above. (Not checked against the gym-pybullet-drones README in this task.)

Confidence: Rerun/pybullet/matplotlib/DuckDB install and timing results = **high** (measured). GUI hang root cause = **low** (launched non-interactively). File-size estimates = **medium** (derived). Foxglove license/account details = **medium** (single discourse post + pricing snippet).

Sources: https://pypi.org/project/rerun-sdk/ ; https://rerun.io/docs/changelog/changeset-0-38 ; https://github.com/rerun-io/rerun/issues/12938 ; https://foxglove.dev/pricing ; https://discourse.openrobotics.org/t/foxglove-2-0-integrated-ui-new-pricing-and-open-source-changes/36583 ; https://github.com/mrdoob/three.js/releases/tag/r186 ; https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=13281 ; https://github.com/bulletphysics/bullet3/issues/2014 ; https://github.com/bulletphysics/bullet3/discussions/3844 ; https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=13395 ; PyPI JSON API for pybullet file list (https://pypi.org/pypi/pybullet/json).

---

## Decisions recommended
1. Record trajectories + decisions only; **no rendering during batch runs**. All visuals derive from stored data keyed by `scenario_id`.
2. Replay data contract = section 3 (tick table at ctrl rate, decision table with obs/request/response/applied ticks, outcome, source, latency split, oracle labels, event table with feasibility margin). Replay clock is always `sim_time`.
3. **Rerun 0.38.1** for engineer replay: one `.rrd` per scenario with all arms as entity roots; generate via `send_columns`; saved blueprint; disable analytics.
4. **Custom single-file HTML player** for the killer demo (canvas + SVG Gantt), served by FastAPI; three.js r186 only as a later option.
5. Matplotlib for static report figures, Plotly.js (browser only) for the live dashboard; DuckDB + statsmodels/scipy for statistics; Jinja2 for the report; do not require kaleido.
6. Live progress via FastAPI native **SSE**.
7. Do not build on PyBullet GUI or Foxglove. Keep GUI as an optional debug flag.
8. Put the pybullet macOS build workaround (`CFLAGS="-Dfdopen=fdopen"`) in setup docs and treat "pybullet installs" as a Phase 0 gate; also pin exact versions in the manifest.
9. Demo must show a latency-matched ghost/variant and the avoidance budget, and label latency mode, to avoid time-model artefacts.

## Open questions
1. Is the trajectory reproducible bit-for-bit across processes (affects whether action-log re-simulation for video is trustworthy)? (agent 02/08)
2. Tick rate to store: 48 Hz vs 24 Hz vs decimating non-hero flights; depends on storage budget after agent 10's volume estimate.
3. Will the LLM/API arm be run in real-time mode or injected-latency mode for the hero scenario? The demo must show one or the other, not mix (agent 08).
4. Does the user want a 3D look (three.js/PyBullet render) or is top-down acceptable?
5. Does the PyBullet GUI hang also occur in an interactive foreground launch? (I could not test with a visible desktop session.)
6. Whether `rerun` blueprint-rendered output can be screenshotted headlessly for CI (not verified).
7. Should MCAP export for Foxglove be offered at all, given account gating?

## Follow-up spikes (small, runnable experiments)
1. **Pybullet install CI check**: fresh `uv venv --python 3.14.2`, `CFLAGS="-Dfdopen=fdopen" uv pip install pybullet gym-pybullet-drones`, run `CtrlAviary` hover 1000 ticks; fail if not green. Pin into a setup script.
2. **Interactive GUI test**: from a foreground Terminal, run `gui.py` (connect GUI, load r2d2, step 720 ticks, `getCameraImage` with `ER_BULLET_HARDWARE_OPENGL`) to classify GUI as working/hanging.
3. **Rerun fidelity**: build the `.rrd` with two arms, open in the native viewer, verify blueprint and scrubbing; measure load time for 40 scenarios in one session; test `rr.serve_web_viewer()` from a browser.
4. **Replay from poses**: DIRECT client + `resetBasePositionAndOrientation` + `getCameraImage` 640x480 for a stored 10 s trajectory; measure ms/frame and encode MP4 (estimate 55 s per 30 s flight).
5. **HTML player prototype** fed by synthetic JSON for 3 arms incl. Gantt; confirm 60 fps canvas at 1440 ticks x 3 arms and that stepping by tick works.
6. **Parquet size test**: write a realistic 1440-tick, 45-column file with zstd; confirm 60-90 KB; then 10 000-flight DuckDB query timing (`GROUP BY arm, latency_setting` with Wilson in SQL/pandas).
7. **SSE smoke test**: FastAPI `EventSourceResponse` streaming a SQLite progress row at 1 Hz to a browser `EventSource` for 10 min; check reconnect and CPU.
8. **Stats harness**: simulate paired Bernoulli collisions with known effect; verify Wilson/BCa coverage and McNemar power at the n targeted by agent 08.
