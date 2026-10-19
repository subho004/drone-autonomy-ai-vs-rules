# 02 — gym-pybullet-drones: internals, API, Apple Silicon, performance

Research agent 02. Author date: 2026-09-30. All numbers below were measured by me on this machine unless marked otherwise; source claims are from code I cloned and read (HEAD `7ebad1e`, 2026-09-06), not from memory.

## 0. TL;DR (verified vs inferred)

| # | Finding | Status | Confidence |
|---|---|---|---|
| 1 | gym-pybullet-drones (HEAD, package v2.2.0) **installs and runs on Python 3.14.2 / arm64 macOS**, including numpy 2.5.3, scipy 1.18.1, gymnasium 1.3.0, torch 2.14.0, stable-baselines3 2.9.0, matplotlib, pillow. The only friction is **pybullet**: there is no macOS wheel for any Python version on PyPI (only manylinux x86_64 + sdist), so it compiles from source, and the plain `pip/uv pip install pybullet` **fails** on this machine; `CFLAGS="-Dfdopen=fdopen"` (the README workaround) fixes it (2m13s build). | Measured | High |
| 2 | The "~20 kHz with 80 drones across 4 envs" claim is **real but stale**: it is in the pre-2023 README only (removed in commit `29fc5b1`, 2023-04-15). The current README contains **no performance claim at all**. The old figure was aggregate across 80 drones on 2020 i7 hardware, and the same old table reports only 0.8-0.95x real time for that case. On this M4, one process / one drone does **~130-170k raw PyBullet steps/s** and **~37k steps/s with PID in the loop**. So 20 kHz is neither relevant nor a limit for our use. | Verified from git history + measured | High |
| 3 | The "Apple Silicon/macOS 26 tested" claim **is present verbatim** in the current README ("Tested on Intel x64/Ubuntu 24.04 and Apple Silicon/macOS 26", commit `e6c5f61`, 2026-08-15) and my run agrees, *with the pybullet CFLAGS workaround*. The claim does not mention that there are no wheels. GUI/OpenGL is not needed for DIRECT mode. | Verified | High |
| 4 | Sim is **not** the bottleneck. 1 drone, PID in the loop, 240 Hz physics / 48 Hz control: ~7k control steps/s/core (~150x real time) without obstacles, ~4.5k with the built-in obstacle set. 4-8 worker processes scale to ~20k control steps/s aggregate (see caveat on machine load). A 20 s flight = 960 control steps = ~0.2 s of CPU. LLM/API latency will dominate by 3+ orders of magnitude. | Measured | Medium-high (machine was shared with other agents, see 3.1) |
| 5 | **Wind is trivial** (override `_physics`, call `p.applyExternalForce` per physics sub-step; +~9% step cost). Battery, GPS, sensor noise, actuator latency do not exist natively; all are thin wrappers over `_preprocessAction` / `_computeObs`. | Measured | High |
| 6 | Nasty gotchas found in code/measurement: `VelocityAviary` caps cruise at **0.25 m/s** by default; `env.reset()` does **not** reset the PID controllers (episodes are not reproducible unless you call `ctrl.reset()`); every `reset()` reloads all URDFs (85-126 ms with default obstacles vs ~8-12 ms without); no collision flag exists (use `getContactPoints`); constructor `print()`s. | Measured/read | High |

## 1. Repo state

- Repo: `https://github.com/learnsyslab/gym-pybullet-drones` (org moved from `utiasDSL`; README tip box points to sibling projects `safe-control-gym`, `crazyflow` (JAX, GPU), `aerial-autonomy-stack`). Maintainer in `pyproject.toml`: Jacopo Panerati. License: MIT. ~2.1k stars, ~571 forks, 110 open issues (WebFetch of the GitHub page, 2026-09-30).
- `main` is alive: 856 commits; recent: 2026-09-06 (pybullet install notes), 2026-08-15 "Update to Python 3.12, Ubuntu 24.04, Betaflight SITL fixes (#316)" and "tested on macos", 2026-04-14 "fix: initialize RNG with seed in BaseAviary.reset() (#300)", 2026-05-31 "Added DEP an ALL observation modes (#311)". `git log` read locally.
- **Not on PyPI** (my query of the PyPI JSON API returned no package record) and the only GitHub *release* is `v1.0.0` (2021-03-09). You install from a git checkout: `pip install -e .`. `pyproject.toml` (poetry): `version = "2.2.0"`, `python = "^3.12"`, deps `numpy ^2.5, scipy ^1.18, gymnasium ^1.3, stable-baselines3 ^2.9, torch ^2.13, pybullet ^3.2.7, matplotlib ^3.11, pillow ^12.3, transforms3d, control, setuptools >=83, pytest`. Note torch and SB3 are hard dependencies even if you never train.
- pybullet: latest on PyPI is 3.2.7 (2025-01-30); PyPI file list for 3.2.7 = `manylinux_2_17_x86_64` wheels for cp36-cp311 + pypy, and `.tar.gz`. **No macOS, no arm64, no cp312+ wheels.** So every Apple Silicon install compiles pybullet.
- Issues of interest (titles only, from the GitHub issues API): #102 "Does this work for ARM?" (2022), #130/#115 install problems, #216 ray 1.9. Nothing new about M-series on open issues list that I could see; the maintainers now document the build flag in the README.
- Changes vs the pre-2023 API that older blog posts still show: `SimplePIDControl` is **gone** (only `DSLPIDControl`, `MRACControl`, `CTBRControl`), `aggregate_phy_steps` is gone (replaced by `pyb_freq` + `ctrl_freq`), `VelocityAviary` takes `ctrl_freq`, `sync()` is now a helper in `utils/utils.py`. The decomposition brief mentions SimplePIDControl and `sync()`; only `sync()` still exists.

## 2. Install spike (Python 3.14.2, arm64, uv 0.9.18)

Machine: Apple M4 (4 P + 6 E cores = 10 logical), 16 GB, macOS `sw_vers` = ProductVersion 27.0 / build 26A428, clang from Xcode CLT. uv-managed `cpython-3.14.2-macos-aarch64`.

```
uv venv --python 3.14.2 v314
uv pip install --python v314/bin/python pybullet
   -> FAILS after 3m18s: build backend error compiling examples/ThirdPartyLibs/zlib/zutil.c
      (pybullet 3.2.7 sdist, -DBT_USE_DOUBLE_PRECISION ... lots of warnings then clang exits non-zero)
CFLAGS="-Dfdopen=fdopen" uv pip install --python v314/bin/python pybullet --no-cache
   -> OK, "Built pybullet==3.2.7", 2m13s wall (165 s user) while the machine was under heavy load
uv pip install numpy scipy gymnasium transforms3d matplotlib pillow control stable-baselines3 torch
   -> OK in seconds: numpy 2.5.3, scipy 1.18.1, gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.14.0, matplotlib 3.11.2, pillow 12.3.0
uv pip install --no-deps -e gym-pybullet-drones   -> gym-pybullet-drones 2.2.0
```

- Root cause of the plain-install failure is **inferred** (the README's own flag `-Dfdopen=fdopen` targets the bundled zlib `zutil.c`/minizip macro clash with the recent macOS SDK); I did not capture the exact clang diagnostic line. The fix worked, which is the evidence.
- Fallback Pythons (3.12/3.11/3.10) were **not needed** and not tested: 3.14.2 works. PyBullet itself has no cp312+ wheel anywhere, so moving to 3.12 does not remove the compile step; it would only matter if some other dep lacks 3.14 wheels (none did here). `torch.backends.mps.is_available()` was True on 3.14.
- Reproducibility note for the repo: cache the built wheel (`uv build`/`pip wheel pybullet` with the CFLAGS) in a private index or `wheelhouse/` so CI and other machines avoid the 2-3 minute compile; put the CFLAGS in the setup doc, **not** in `requirements.txt`.
- Smoke tests that passed on 3.14.2: `HoverAviary(obs=KIN, act=RPM|PID|VEL)` reset + 2000 `step()`s; all benchmarks below. I did not run the repo's `pytest` (it spawns training).
- Runtime warnings to silence in our harness: gymnasium `Box low's precision lowered by casting to float32` (harmless); `BaseAviary.__init__` `print()`s 10+ `[INFO]` lines per construction (I wrap construction in `contextlib.redirect_stdout`; our repo rules ban `print`, so wrap or patch).

## 3. Benchmarks (headless, `p.DIRECT`, `gui=False`)

### 3.1 Measurement caveat (be honest)
Fourteen research agents share this Mac. `uptime` load average was 46-60 during my first run (other agents compiling), ~6-10 during the final runs, and dropped to ~4 at the end. My first pass produced numbers 2-3x lower and erratic (e.g. the same config ranging 3.8k-9k steps/s). The tables below are the **final best-of-9** run (each run = 5 simulated seconds after 200-step warm-up, min wall time taken) started at load ~7.7 on 10 cores; single-thread numbers are therefore close to, but still probably below, an idle-machine figure. Parallel-scaling numbers are more contaminated; treat them as ±25%. Use `bench1`/`bench3` (code in section 9) on a quiet machine to get the authoritative figures.

Versions: Python 3.14.2, pybullet 3.2.7 (API 202010061), numpy 2.5.3, gymnasium 1.3.0, gym-pybullet-drones 2.2.0 @ `7ebad1e`. Drone = `CF2X` (0.027 kg, T/W 2.25). Loop = `CtrlAviary.step()` with constant hover RPM ("no PID") or `DSLPIDControl.computeControl` each control step ("PID", holding z=1 m). "ctrl steps/s" = `env.step()` calls per second; "phys steps/s" = ctrl steps x pyb/ctrl; "RT x" = sim seconds per wall second.

### 3.2 Single drone, one process (final run, load ~7)

| physics | obstacles | PID | pyb Hz | ctrl Hz | ctrl steps/s | phys steps/s | RT x |
|---|---|---|---|---|---|---|---|
| PYB | no | no | 240 | 240 | 72,456 | 72,456 | 302 |
| PYB | no | no | 240 | 48 | 26,506 | 132,530 | 552 |
| PYB | no | no | 240 | 30 | 18,443 | 147,542 | 615 |
| PYB | no | no | 500 | 50 | 15,557 | 155,567 | 311 |
| PYB | no | no | 1000 | 50 | 8,456 | 169,119 | 169 |
| PYB | no | no | 1000 | 100 | 15,616 | 156,157 | 156 |
| PYB | no | no | 2000 | 100 | 8,477 | 169,545 | 85 |
| PYB | no | **PID** | 240 | 240 | 9,038 | 9,038 | 38 |
| PYB | no | **PID** | 240 | 48 | 7,335 | 36,677 | 153 |
| PYB | no | **PID** | 240 | 30 | 6,331 | 50,648 | 211 |
| PYB | no | **PID** | 500 | 50 | 5,889 | 58,887 | 118 |
| PYB | no | **PID** | 1000 | 50 | 4,275 | 85,502 | 86 |
| PYB | no | **PID** | 1000 | 100 | 5,994 | 59,935 | 60 |
| PYB | no | **PID** | 2000 | 100 | 4,319 | 86,375 | 43 |
| PYB | default obstacles | no | 240 | 240 | 34,923 | 34,923 | 146 |
| PYB | default obstacles | no | 240 | 48 | 8,960 | 44,801 | 187 |
| PYB | default obstacles | no | 1000 | 50 | 2,486 | 49,719 | 50 |
| PYB | default obstacles | **PID** | 240 | 240 | 7,669 | 7,669 | 32 |
| PYB | default obstacles | **PID** | 240 | 48 | 4,548 | 22,738 | 95 |
| PYB | default obstacles | **PID** | 1000 | 50 | 1,870 | 37,391 | 37 |
| PYB | default obstacles | **PID** | 2000 | 100 | 1,902 | 38,048 | 19 |
| GND+DRAG+DW | no | no | 240 | 48 | 6,871 | 34,354 | 143 |
| GND+DRAG+DW | no | **PID** | 240 | 48 | 3,935 | 19,674 | 82 |
| GND+DRAG+DW | no | **PID** | 1000 | 50 | 1,480 | 29,607 | 30 |

What the numbers say:
- Raw PyBullet stepping of one quadrotor: ~130-170k steps/s per core, nearly independent of how you slice pyb/ctrl once `pyb_freq >= ctrl_freq*5`; fitting the table, cost per `env.step()` is roughly 7 µs fixed Python overhead plus ~6-7 µs per physics sub-step (240/240: 13.8 µs/step; 240/48: 37.7 µs/step; 1000/50: 118 µs/step).
- **`DSLPIDControl.computeControl` costs ~100 µs per call** (1/7,335 - 1/26,506 s at 240/48): it is pure numpy/scipy `Rotation` code. It dominates everything at 48-240 Hz. At 240/48 the PID costs ~2.6x the physics+env overhead (98 µs vs 38 µs). A leaner controller would probably give a large speedup (inference, untested).
- **The default obstacle set** (`obstacles=True` loads `samurai.urdf`, `duck_vhacd.urdf`, `cube_no_rotation.urdf`, `sphere2.urdf` at hard-coded positions) makes each step ~2-3x more expensive and each reset ~10x. Build our own obstacles from `p.createMultiBody` primitives.
- Extra physics (ground effect + drag + downwash) costs roughly 1.8x more with PID (3.9k vs 7.3k ctrl steps/s), because those call `getLinkStates`/`applyExternalForce` per sub-step.
- Comparison with the plan: **20 kHz** of physics updates vs ~37k (1 drone, PID, 240/48) or ~132k (raw) on one core here.

### 3.3 Multiprocess scaling (obstacles=True, PID, 240/48, 1 drone per process, `spawn`, barrier start, 30 sim-s each)

| processes | run A aggregate ctrl steps/s | run B aggregate ctrl steps/s | per-process (A) |
|---|---|---|---|
| 1 | 4,564 | 3,194 | 4,564 |
| 2 | 8,596 | 8,737 | 4,298 |
| 4 | 17,164 | 15,431 | 4,291 |
| 6 | 18,218 | 17,973 | 3,036 |
| 8 | 22,210 | 20,512 | 2,776 |
| 10 | 22,914 | 11,047 (other agents' jobs started) | 2,291 |

Scaling is ~linear up to 4 processes (the 4 P-cores), then flattens: 6-10 processes add only ~30% because the 6 E-cores are slower. Aggregate ~20-23k ctrl steps/s = ~100-115k physics steps/s = **~430-480x real time total** on 8-10 workers. Run B's 10-process collapse coincided with someone else's load, so pool size 6-8 is the practical sweet spot; leave 1-2 cores for the asyncio/API side.
- Flights/hour estimate (inference from measured rates): a 20 s flight at 48 Hz = 960 control steps. At ~2.5k steps/s/worker under 8-way load that is ~0.4 s per flight of sim CPU, so ~9k flights/hour/worker, ~70k flights/hour with 8 workers **excluding decision-provider time**.

### 3.4 Reset, construction, memory

| op | no obstacles | default obstacles |
|---|---|---|
| `env.reset()` (final run) | 7.9 ms | 85.5 ms (earlier loaded run: 12 ms / 126 ms) |
| construct + `close()` a whole new env | 7.6 ms | not measured |
| `p.loadURDF("samurai.urdf")` mid-episode | 242 ms | |
| `p.loadURDF("duck_vhacd.urdf")` | 8.8 ms | |
| `p.loadURDF("cube_small.urdf")` | 1.1 ms | |
| `p.createMultiBody` box (runtime obstacle) | 0.9 ms; `p.removeBody` 0.009 ms | |

- `BaseAviary.reset()` = `p.resetSimulation()` then `_housekeeping()`, which **reloads the plane, drone URDF and obstacles from disk every time** (see source). So per-episode reset cost scales with obstacle mesh complexity. With primitives only, reset is ~8 ms, i.e. about 2 s of reset for 250 flights: irrelevant.
- **Memory over 1000 resets** (each followed by 48 steps): RSS delta from baseline at reset #1/100/250/500/1000: no-obstacle env `+0.2 / +0.2 / -38.8 / -82.6 / -82.6 MB`; obstacle env `+0.1 / +8.1 / -8.8 / -14.7 / -16.9 MB`. **No growth observed.** RSS going *down* is macOS memory compression/purging and shows `psutil` RSS is a weak instrument on Darwin; I cannot claim "leak-free" beyond "no upward trend within 1000 resets". 200 construct/close cycles: +6.3 MB at 1, -43 MB at 200 (same caveat). A longer soak (10k resets, sampling `footprint` or `vmmap -summary`) is listed under follow-ups. Baseline RSS of a process that has imported the envs (numpy/scipy/gymnasium/pybullet, no torch) was ~208-240 MB, so budget ~0.25 GB per worker plus whatever the harness adds; 8 workers ~2 GB.
- Determinism: same process, same `reset(seed=123)` and same action sequence -> identical SHA-256 of 300 observations (True). `env.reset()` reuse: episode 1 hash != episode 2 hash **unless `ctrl.reset()` is called on each `DSLPIDControl` in `env.ctrl`**; with `ctrl.reset()` ep3 == ep1. So a worker that reuses one env across scenarios must reset the controllers (integrators/`last_*`). Cross-process determinism not yet tested (follow-up); on identical binary it is expected but unverified.

### 3.5 Sensors: camera, rays, contacts

| call | cost |
|---|---|
| `_getDroneImages` 64x48 (RGB+depth+seg), **with default obstacles** | 9.9 ms (9.1 ms in loaded run) |
| same, `segmentation=False` | 9.9 ms (no saving) |
| same, **no obstacles** | 2.1 ms |
| `p.getCameraImage` TinyRenderer 64x48 | 9.8 ms |
| 128x96 | 11.4 ms |
| 320x240 | 20.6 ms |
| 640x480 | 57.0 ms |
| `p.rayTestBatch` 8 / 16 / 64 / 360 rays | 4 / 8 / 32 / 181 µs total |
| `p.getContactPoints(drone)` | ~1 µs when no contact (contacts are only computed in `stepSimulation`) |
| `p.getClosestPoints(drone, body, 5 m)` | 12 µs |

- Camera frame cost is dominated by per-call scene sync, not pixels (64x48 at ~10 ms vs 640x480 at 57 ms), and by scene complexity (2 ms with only plane+drone). At the env's 24 FPS image cadence on the default scene that is ~0.24 s of camera CPU per sim second, i.e. more than the physics: **vision is ~30x costlier than a ray-cast "lidar"**. For the Jev experiment use `rayTestBatch` (360 rays cost 0.18 ms) and skip images unless a multimodal arm is explicitly wanted.
- Renderer: in DIRECT mode this is the CPU TinyRenderer; no EGL path on macOS (the EGL plugin line in `BaseAviary` is Linux-only and commented out). No GPU rendering is available to you headless on a Mac.

### 3.6 Wind via `applyExternalForce` (spike)

```python
class WindAviary(CtrlAviary):
    wind_force_n = np.zeros(3)

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            -1,
            forceObj=self.wind_force_n.tolist(),
            posObj=self.pos[nth_drone].tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT,
        )
```

- Why it must go in `_physics` (or a per-sub-step hook): PyBullet clears external forces after each `stepSimulation`, and `BaseAviary.step` calls `_physics` once per physics sub-step. Setting wind once per control step would apply it for only 1/5 of the time at 240/48.
- Overhead measured: 39.7 us -> 43.3 us per control step for the bare loop (+9%).
- Effect with `DSLPIDControl` holding hover (weight = 0.265 N): constant +x force 0.02 N: max excursion 5 cm, settled offset 1.4 cm; 0.05 N: 12 cm / 2.6 cm; 0.10 N (38% of weight): 24 cm / 5.4 cm; z stays within 2 mm. The PID integral term rejects steady wind quickly, so **a constant wind is a weak adversarial event for a hovering PID**; gusts, wind while translating at speed, or large force relative to T/W are needed to matter. With 2.25 T/W, horizontal wind above roughly 1.0-1.5 N would exceed authority; the useful regime is 0.05-0.3 N, time-varying.
- Drag is the only native air-relative model (`Physics.PYB_DRAG`), driven by ground speed, **not air-relative velocity**, so wind via drag would need a custom override anyway.
- Clean design: `WindModel` object (seeded OU/Dryden process) queried inside `_physics` with the *physics* time, so the force schedule is indexed by sim tick, not wall-clock.

### 3.7 VelocityAviary and stopping physics (spike, 48 Hz control, 1 m altitude)

`VelocityAviary.SPEED_LIMIT = 0.03 * MAX_SPEED_KMH * 1000/3600 = 0.25 m/s` (MAX_SPEED_KMH = 30 from the URDF). The action `[x, y, z, f]` normalises the direction and scales it by `SPEED_LIMIT*|f|`, so **by default you can never fly faster than 0.25 m/s**. It is an instance attribute, so `env.SPEED_LIMIT = v` after construction works.

| SPEED_LIMIT m/s | achieved cruise | brake time to <5% (s) | brake distance (m) | max tilt | min z (start 1.0) |
|---|---|---|---|---|---|
| 0.25 | 0.26 | 0.29 | 0.04 | 6.5 deg | 1.01 |
| 1.0 | 0.99 | 0.38 | 0.22 | 26 deg | 1.00 |
| 2.0 | 1.97 | 0.42 | 0.54 | 45 deg | 0.99 |
| 4.0 | 3.89 | 0.54 | 1.28 | 73 deg | 0.87 |
| 6.0 | 5.80 | 0.73 | 2.38 | 89 deg | 0.89 |

- The DSL PID tracks velocity commands up to about 2 m/s cleanly; 4 m/s and up is at the edge of controllability (73-89 deg tilt, altitude sag). Pick **1-2 m/s** cruise for the benchmark.
- Reaction physics for the experiment: at 2 m/s, every 100 ms of stale decision costs 0.2 m; brake distance is 0.54 m and time to stop 0.42 s. So an obstacle that appears within ~1 m at 2 m/s is avoidable only with <~250 ms total reaction. That is the regime where latency matters and what scenario feasibility design (file 06) should use.
- Collision detection: dropping a static `createMultiBody` box at x=1.0 and flying +x at 0.25 m/s gave `getContactPoints(drone, box)` non-empty after 168 control steps (x=0.839 m; the drone body radius is ~4-6 cm). The env itself has **no collision flag**: `_computeTerminated()` is a dummy `False` in `CtrlAviary`/`VelocityAviary`, and `HoverAviary` terminates only on reaching the target / truncates on tilt / range.
- Two envs in one process get `CLIENT` ids 0 and 1 and are isolated (x=0.209 vs 0.0 after commanding one to fly). So multi-env-per-process works, but one env per process is still the recommendation (multi-core scaling needs processes, and it bounds crashes/memory); thread-safety of one client across threads was not tested.

## 4. API map, read from source

### 4.1 Class hierarchy (`gym_pybullet_drones/envs/`, line counts from `wc -l`)

| Class | Role | Action space | Observation space | Notes |
|---|---|---|---|---|
| `BaseAviary(gym.Env)` (1150 l) | physics core, step loop, camera, URDF parsing | abstract | abstract | `_actionSpace/_observationSpace/_computeObs/_preprocessAction/_computeReward/_computeTerminated/_computeTruncated/_computeInfo` raise `NotImplementedError` |
| `CtrlAviary` (200 l) | "control applications"; you supply RPMs | `Box(N,4)` RPM in `[0, MAX_RPM]` | `Box(N,20)`: xyz, quat(4), rpy, vel, ang_vel, last RPMs | reward -1, never terminates, `info={"answer": 42}` |
| `VelocityAviary` (228 l) | high-level planning; internal `DSLPIDControl` per drone | `Box(N,4)`: `[vx,vy,vz, speed_fraction]` (direction normalised; speed = `SPEED_LIMIT*abs(f)`) | same 20-vector | `SPEED_LIMIT` = 0.25 m/s by default; holds yaw; dummy reward/terminated |
| `BaseRLAviary` (349 l) | RL base, single/multi-agent | `ActionType`: `RPM`(4), `PID`(3 = target xyz), `VEL`(4), `ONE_D_RPM`(1), `ONE_D_PID`(1), each in `[-1,1]` | `ObservationType.KIN`: 12 (xyz, rpy, vel, ang_vel) + `ctrl_freq//2` past actions; `RGB` (N,48,64,4); `DEP`; `ALL`=Dict | always passes `obstacles=True` but only adds 4 landmark URDFs for `RGB`; `ctrl_freq` default 30 |
| `HoverAviary` (132 l) | 1 drone hover at (0,0,1) | via `BaseRLAviary` | via `BaseRLAviary` | reward `max(0, 2 - dist**4)`; truncates at |x|,|y|>1.5, z>2, |roll|,|pitch|>0.4, or 8 s; "terminated" if within 0.1 mm |
| `MultiHoverAviary` (145 l) | N-drone hover | same | same | |
| `BetaAviary` (333 l) | Betaflight SITL | | | needs external SITL build; irrelevant here |

Enums (`utils/enums.py`): `DroneModel` {CF2X, CF2P, RACE}; `Physics` {PYB, DYN, PYB_GND, PYB_DRAG, PYB_DW, PYB_GND_DRAG_DW}; `ImageType`; `ActionType`; `ObservationType`.
Controllers (`control/`): `DSLPIDControl` (CF2X/CF2P only; the RACE model has none), `MRACControl`, `CTBRControl` (collective thrust + body rates). `SimplePIDControl` no longer exists.

### 4.2 Constructor parameters (`BaseAviary.__init__`)
`drone_model, num_drones=1, neighbourhood_radius=inf, initial_xyzs, initial_rpys, physics=Physics.PYB, pyb_freq=240, ctrl_freq=240, gui=False, record=False, obstacles=False, user_debug_gui=True, vision_attributes=False, output_folder='results'`.
- `pyb_freq` must be an integer multiple of `ctrl_freq` (`ValueError` otherwise); `PYB_STEPS_PER_CTRL = pyb/ctrl`. `env.step()` therefore advances `1/ctrl_freq` s of *sim time*; every `step()` runs that many `stepSimulation()` calls, applies the **same** RPMs for all of them (zero-order hold), and returns the state at the end.
- Kinematics are cached once per step in `env.pos/quat/rpy/vel/ang_v` (`_updateAndStoreKinematicInformation`); `_getDroneStateVector(i)` builds the 20-vector.
- `gui=False` -> `p.connect(p.DIRECT)`; `gui=True` -> `p.connect(p.GUI)` (needs OpenGL window; on macOS it works in principle but was not needed or tested here). `record=True` with `gui=False` saves a 640x480 PNG every `pyb_freq/24` steps via TinyRenderer (expensive, see 3.5); with `gui=True` it uses PyBullet's mp4 logger.
- `vision_attributes` is only exposed by `BaseAviary`/`BaseRLAviary` (RGB/DEP/ALL), not by `CtrlAviary`/`VelocityAviary`; to get camera output from those either call `env._getDroneImages(i)` yourself after setting `env.IMG_RES = np.array([64,48])` (as I did) or subclass.

### 4.3 Wall-clock pacing: `sync()` and real-time
- **The environment never sleeps.** `step()` has no wall-clock coupling; `time.time()` appears only in `RESET_TIME` for the text-only `render()` (prints sim time vs wall time, the "x real time" factor) and in video filenames. So headless runs are "as fast as possible" by default.
- `utils.utils.sync(i, start_time, timestep)`: `if timestep > .04 or i % int(1/(24*timestep)) == 0: sleep until i*timestep has elapsed`. It is a helper that the *examples* call in their loop (`examples/pid.py:167`); you disable it by simply not calling it. Note its quirk: with `timestep = 1/240` it sleeps only every 10th call.
- `gui=True` has no built-in throttling either (`p.setRealTimeSimulation(0)` is set in `_housekeeping`); the examples rely on `sync()`.
- For the project's latency model (file 08), this is ideal: the sim clock is `step_counter / PYB_FREQ` (and `env.step_counter` is incremented by `PYB_STEPS_PER_CTRL` per step), entirely decoupled from wall time, so "latency injection" can be implemented as N ticks of command delay without any pacing code.

### 4.4 Obstacle mechanism
- `obstacles=True` in `BaseAviary._housekeeping` calls `_addObstacles()`: loads Bullet-data URDFs `samurai.urdf` (origin), `duck_vhacd.urdf` (-.5,-.5,.05), `cube_no_rotation.urdf` (-.5,-2.5,.5), `sphere2.urdf` (0,2,.5). **Hard-coded; not configurable; not randomisable.** `BaseRLAviary._addObstacles` overrides it (4 landmarks only for `RGB`).
- Reset reloads all of this (85-126 ms). For the experiment: pass `obstacles=False` and override `_addObstacles()` (or add bodies in a wrapper after `reset()`), using `p.createMultiBody` with `GEOM_BOX/SPHERE/CYLINDER` (0.9 ms to create, 9 us to remove). Mid-episode "sudden appearance" = one `createMultiBody` call inside the step loop at the scheduled sim tick. (The repo's own `examples/pid.py` has a commented "make it rain rubber ducks" `loadURDF` mid-run hook, showing the pattern; `duck_vhacd.urdf` takes 8.8 ms, `samurai.urdf` 242 ms.)
- Ground truth collision: `p.getContactPoints(bodyA=drone, bodyB=obstacle)` after `stepSimulation`; `p.getClosestPoints` gives clearance (12 us); `p.rayTestBatch` for lidar-like sensing (181 us for 360 rays). Verified to work (section 3.7). Ground contacts also appear in `getContactPoints` (filter by `bodyB == PLANE_ID`).
- Drone collision geometry is a cylinder (`COLLISION_H`, `COLLISION_R` from the URDF); for `Physics.DYN` there is no PyBullet contact at all.

### 4.5 Physics modes (what is native)
- `PYB`: rotor thrust `kf*rpm^2` applied as `applyExternalForce` at each of 4 prop links, plus yaw torque.
- `PYB_GND`: per-propeller ground effect (Shi et al. 2019 inspired). `PYB_DRAG`: body drag from the sum of rotor speeds times ground velocity (Forster 2015). `PYB_DW`: downwash from other drones above (DSL experiments; only matters for multi-drone). `PYB_GND_DRAG_DW`: all. `DYN`: explicit analytic dynamics, no PyBullet stepping.
- **Not native**: wind/air-relative velocity, battery/thrust derating, GPS/IMU noise or dropout, actuator lag/latency, rotor-motor dynamics (RPM is applied instantly), sensor delays, moving obstacles, a second drone policy (multi-drone exists via `num_drones`, each driven by the caller). Nothing in the env cleans up after custom bodies or forces either.

### 4.6 PID API
```python
ctrl = DSLPIDControl(
    drone_model=DroneModel.CF2X
)  # one per drone, stateful (integrators, last_rpy)
rpm, pos_err, yaw_err = ctrl.computeControl(
    control_timestep,
    cur_pos,
    cur_quat,
    cur_vel,
    cur_ang_vel,
    target_pos,
    target_rpy=zeros(3),
    target_vel=zeros(3),
    target_rpy_rates=zeros(3),
)
ctrl.reset()  # REQUIRED between episodes; env.reset() does not do it
```
Gains (DSL): position P/I/D `[.4,.4,1.25]/[.05,.05,.05]/[.2,.2,.5]`, torque P `[70000,70000,60000]`, D `[20000,20000,12000]`. `VelocityAviary` calls it with `target_pos = current pos` and `target_vel`, so position error is zero and only the velocity term acts (so only the velocity-error path is active: it is a velocity controller, with no position hold while moving; after `BRAKE` the drone drifts to rest where it stops rather than returning to a point). `ActionType.PID` in `BaseRLAviary` feeds the controller a waypoint 1 m toward the target (`_calculateNextStep(step_size=1)`).

## 5. Extension points for this project

| Hook | Where | What we would do |
|---|---|---|
| `_preprocessAction(action)` | subclass of `BaseAviary` | map the 11-action enum (+latched setpoint, TTL, safety shield) to `target_pos/target_vel` -> `DSLPIDControl` -> RPM; also the natural place for **actuator lag / command delay** (hold a FIFO of past commands; `latency_ticks` = `round(latency_s*ctrl_freq)`) and **battery derating** (scale `MAX_RPM` or RPM output by `battery_factor(t)`) |
| `_physics(rpm, i)` / per-sub-step loop | override | wind/gust forces (3.6), thrust loss, motor fault; called once per physics sub-step |
| `_computeObs()` | override | emit our own `Observation`: add ray ranges, relative target vector, noisy/dropout GPS (seeded), battery, other-drone state, stale-state age. Must return what `observation_space` declares (override `_observationSpace`) or use a plain dataclass out-of-band since we do not need SB3 for most arms |
| `_computeReward/_computeTerminated/_computeTruncated/_computeInfo` | override | collision termination via `getContactPoints`; success radius; timeout; info dict carrying event ground truth and `sim_time`. The metrics collector should consume `info`, not the env internals |
| `_addObstacles()` | override | seeded scenario obstacles via `createMultiBody`; dynamic obstacle/second drone added at event ticks |
| `_housekeeping()` | override carefully | it runs on construct and every reset; reset the controllers, event schedule, battery state here (cleaner than relying on `reset()` callers) |
| `reset(seed, options)` | wrapper | `super().reset(seed=...)` seeds `self.np_random` (fix #300); pass the scenario via `options` |

Design advice (opinion): subclass `CtrlAviary` (raw RPM, no hidden controller) or `VelocityAviary`, not the RL classes; own the controller in our `controllers/` layer (so the PID is shared verbatim by all arms and its compute time is measurable), and use **one thin `SimWorker` process** that owns one env. Stepping at `pyb=240, ctrl=48` is a good default (20.8 ms control tick, 5 sub-steps); use `ctrl=60/240` if finer latency quantisation is wanted; 1000/50 is 2-4x slower for little benefit unless stiff contacts matter.

## 6. Reality check of README claims

### 6.1 "~20 kHz with 80 drones across 4 environments"
- Current README (HEAD `7ebad1e`): `grep -i "kHz|80 drones|faster|real.?time|parallel|speed"` returns **nothing**. There is no performance section.
- `git log -S"kHz" -- README.md` shows the text was deleted in commit `29fc5b1` ("Initial commit of the 2023 update", 2023-04-15). The old README (`29fc5b1^:README.md`) says, verbatim in the relevant parts: "~5kHz PyBullet physics (CPU-only) is faster than AirSim (1kHz) and more accurate than Flightmare's 35kHz simple single quadcopter dynamics" and "Exploiting parallel computation, i.e. multiple (80) drones in multiple (4) environments (see script `parallelism.sh`), achieves PyBullet physics updates at ~20kHz". The table in the same README: Lenovo P52 (i7-8850H) and a 2020 MacBook Pro (i7-1068NG7): single drone no vision 15.5x / 16.8x real time; single drone **with vision** 10.8x / **1.3x**; 10 drones no vision 2.1x / 2.3x; 5 drones with vision 2.5x / **0.2x**; **80 drones in 4 envs, no vision: 0.8x / 0.95x** real time. Measured at 240 Hz physics, 48 Hz control, with obstacles, and 24 FPS 64x48 6-channel capture where vision applies.
- Interpretation: (1) it is an **aggregate** throughput for a multi-drone, multi-process case on 2018-2020 Intel laptops, so it says nothing about a single drone; (2) those same numbers show **sub-real-time** (0.8-0.95x) for the 80-drone case; (3) it is from the pre-2023 codebase (the `aggregate_phy_steps`/`SimplePIDControl` era), not the current refactor. **The plan's sentence "The project reports ~20 kHz" is out of date** (repo no longer says that) and **confidence that it applies to M4/HEAD: none**. My single-process M4 number is ~130-170k raw steps/s (3.2) and ~37k with PID; with 8 procs, ~100-115k physics steps/s with PID and obstacles (3.3).
- More useful framing for this project: control ticks/s and x-real-time, since decisions happen per control tick: ~4.5-7k ctrl steps/s/core (PID, 48 Hz), or ~95-153x real time per worker.
- Caveat on the vision numbers: old vision cost (1.3x and 0.2x on Mac) is consistent with my finding that camera frames cost ~10 ms each on the default scene.

### 6.2 "Apple Silicon / macOS 26 tested"
- Present verbatim in README (line 20): "Tested on Intel x64/Ubuntu 24.04 and Apple Silicon/macOS 26." Backed by commits `dc23c5e` (2026-04-14 "macOS version") and `e6c5f61` (2026-08-15 "tested on macos"). The README further says "Beyond Python 3.10, `pybullet` has no pre-built wheel ... On macOS, build and install `pybullet` with `CFLAGS="-Dfdopen=fdopen" pip install pybullet --no-cache-dir`" and recommends `conda create -n drones python=3.12`.
- What the claim does not tell you: it is DIRECT/headless-capable; GUI path untested by me; the compile takes 2+ minutes and needs Xcode CLT; `uv` works just as well as conda. I reproduced "works on Apple Silicon" on Python 3.14.2 (newer than the README's 3.12) with the flag. **Confirmed, with the caveat.**

## 7. Risks and sharp edges specific to this library

1. **PID state leaks across episodes** (verified). Fix: reset controllers in the env `_housekeeping` override.
2. **`VelocityAviary` 0.25 m/s cap** (verified) makes naive use of it a non-adversarial toy; set `SPEED_LIMIT` to 1-2 m/s.
3. **Reset cost with default obstacles** (85-126 ms) and camera cost (10 ms/frame) scale with mesh complexity; use primitives.
4. **No timing model**: control is zero-order hold at `ctrl_freq`; motor RPM responds instantly; no sensor delay; if a "decision" arrives between control ticks we must quantise it to the tick.
5. **Noisy hard dependencies**: torch + SB3 are mandatory installs even for a pure-control sim worker (heavy; installs fine on 3.14 arm64 here). If the simulator worker lives in its own env, a `--no-deps` editable install + `numpy scipy gymnasium pybullet transforms3d` is sufficient for `CtrlAviary`/`VelocityAviary`/`DSLPIDControl` (I installed everything; I did not test the minimal set, and `matplotlib`/`control` may be imported by some modules; verify).
6. **Library is research-grade**: `print()` and `exit()` calls for errors (e.g. `BaseAviary.__init__` on bad `initial_xyzs` prints an error and continues; `_getDroneImages` calls `exit()`), bare prints in `render()`, `os.environ['KMP_DUPLICATE_LIB_OK']='True'` set by `VelocityAviary` and `BaseRLAviary` (an OpenMP duplicate-lib workaround, a global side effect inside our process). Wrap in an adapter and never let these escape into the FastAPI process; keep the sim in worker processes.
7. **Python-level throughput ceiling**: per control step the Python code (state vectorisation, np.array building, PID) costs more than PyBullet. Vectorised multi-drone is not relevant for one-drone flights; parallelise across flights with processes.
8. **Float32 space warnings** on every construction (cosmetic).
9. `BaseRLAviary.ACTION_BUFFER_SIZE = ctrl_freq//2` inflates the KIN observation with `ctrl_freq/2` past actions (72 dims at 30 Hz; 57 dims for PID actions). Irrelevant unless we use RL classes; if we train an RL arm, pick `ctrl_freq` with that in mind.
10. Unverified: behaviour of the GUI on macOS 27, the `BetaAviary`, `MRACControl`, `CTBRControl`, and cross-process bitwise determinism.

## 8. Mapping to project questions (from 00-decomposition)

- Global Q4 (Python 3.14.2 wheels): answered; all required packages have 3.14 arm64 wheels as of today except pybullet, which compiles in ~2 min with a flag. No need for a separate 3.10-3.12 venv **for this reason**. A process boundary for the sim worker remains recommended for robustness and to keep `KMP_DUPLICATE_LIB_OK`/`print`/`exit` out of the API process.
- Global Q5: see 6.1. Global Q6: confirmed with caveat; GUI not needed.
- Global Q7: native = drag, ground effect, downwash, URDF obstacles; custom = wind, sudden obstacle (spawn at tick), moving target (pure Python point), sensor noise, battery, GPS loss, second drone (num_drones=2 with our policy, downwash available), latency.
- Global Q8 (determinism): in-process determinism verified (same hash); `ctrl.reset()` required; cross-process untested; seed scenarios via our own `np.random.default_rng(seed)` for event schedules, not via env seed.
- Global Q1 (time): sim clock is decoupled from wall clock (4.3); control-tick quantisation = 20.8 ms at 48 Hz (4.2 ms at 240 Hz). Step costs are ~0.13 ms at 240/48 without PID, ~0.14-0.22 ms with PID -> sim step cost (<0.25 ms) is >2,000x smaller than a 300 ms LLM call (>20,000x vs 3 s), so a faster-than-real-time sim plus latency injection is cheap in compute terms, and a real-time-paced sim (a 20 s flight takes 20 s wall) costs ~100x more wall time, which is what the validation subset would cost.

## 9. Benchmark scripts (copied verbatim from the scratchpad; run with `PYTHONPATH=. python -W ignore <file>` from one directory)

Setup used:

```bash
uv venv --python 3.14.2 v314
CFLAGS="-Dfdopen=fdopen" uv pip install --python v314/bin/python pybullet --no-cache
uv pip install --python v314/bin/python numpy scipy gymnasium transforms3d matplotlib pillow control stable-baselines3 torch psutil
git clone https://github.com/learnsyslab/gym-pybullet-drones.git && uv pip install --python v314/bin/python --no-deps -e gym-pybullet-drones   # tested at 7ebad1e
```

### bench_common.py - Shared helpers

```python
"""Shared helpers for gym-pybullet-drones micro-benchmarks (headless DIRECT)."""

import contextlib
import io
import os
import time

import numpy as np
import psutil

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def quiet(fn, *a, **k):
    """BaseAviary.__init__ prints [INFO] lines via print(); silence them."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def make_ctrl_env(pyb, ctrl, obstacles=False, physics=Physics.PYB, n=1):
    xyz = np.array([[0.0, 0.0, 0.5 + 0.2 * i] for i in range(n)])
    return quiet(
        CtrlAviary,
        drone_model=DroneModel.CF2X,
        num_drones=n,
        initial_xyzs=xyz,
        physics=physics,
        pyb_freq=pyb,
        ctrl_freq=ctrl,
        gui=False,
        obstacles=obstacles,
        user_debug_gui=False,
    )


def run_steps(env, n_ctrl_steps, use_pid=False):
    """Hover-ish loop; returns (wall_s, cpu_s). use_pid=True runs DSLPIDControl at ctrl freq."""
    n = env.NUM_DRONES
    if use_pid:
        ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(n)]
        target = np.array([0.0, 0.0, 1.0])
    act = np.tile(np.ones(4) * env.HOVER_RPM, (n, 1))
    obs, _ = env.reset()
    t0 = time.perf_counter()
    c0 = time.process_time()
    for _ in range(n_ctrl_steps):
        if use_pid:
            for k in range(n):
                s = obs[k]
                act[k], _, _ = ctrls[k].computeControl(
                    control_timestep=env.CTRL_TIMESTEP,
                    cur_pos=s[0:3],
                    cur_quat=s[3:7],
                    cur_vel=s[10:13],
                    cur_ang_vel=s[13:16],
                    target_pos=target + np.array([0, 0, 0.2 * k]),
                )
        obs, *_ = env.step(act)
    return time.perf_counter() - t0, time.process_time() - c0
```

### bench1_steps.py - Benchmark 1: steps/sec matrix (3.2)

```python
"""Benchmark 1: steps/sec for 1 drone, varying pyb_freq/ctrl_freq, obstacles, physics, PID."""

import json
import sys

from bench_common import make_ctrl_env, run_steps, Physics

CONFIGS = [  # (pyb_freq, ctrl_freq)
    (240, 240),
    (240, 48),
    (240, 30),
    (500, 50),
    (1000, 50),
    (1000, 100),
    (2000, 100),
]
SIM_SECONDS = 10  # simulated seconds per measurement
REPEATS = 3
for physics in (Physics.PYB, Physics.PYB_GND_DRAG_DW):
    for obstacles in (False, True):
        for use_pid in (False, True):
            for pyb, ctrl in CONFIGS:
                if physics != Physics.PYB and (
                    obstacles or (pyb, ctrl) not in [(240, 48), (1000, 50)]
                ):
                    continue
                env = make_ctrl_env(pyb, ctrl, obstacles=obstacles, physics=physics)
                n = int(SIM_SECONDS * ctrl)
                run_steps(env, min(n, 200), use_pid)  # warm-up
                res = [run_steps(env, n, use_pid) for _ in range(REPEATS)]
                dt = min(r[0] for r in res)  # best-of-N wall
                cpu = min(
                    r[1] for r in res
                )  # best-of-N process CPU time (robust to contention)
                env.close()
                row = dict(
                    physics=physics.value,
                    obstacles=obstacles,
                    pid=use_pid,
                    pyb_hz=pyb,
                    ctrl_hz=ctrl,
                    ctrl_steps_per_s=round(n / dt),
                    phys_steps_per_s=round(n * pyb / ctrl / dt),
                    realtime_x=round(SIM_SECONDS / dt, 1),
                    cpu_ctrl_steps_per_s=round(n / cpu),
                    cpu_phys_steps_per_s=round(n * pyb / ctrl / cpu),
                )
                print(json.dumps(row))
                sys.stdout.flush()
```

### bench2_misc.py - Benchmark 2: reset, memory, camera, rays, contacts, wind (3.4-3.6)

```python
"""Benchmark 2: reset cost, memory growth, camera/raycast/contact cost, wind via applyExternalForce."""

import gc
import json
import sys
import time

import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel
from bench_common import make_ctrl_env, quiet, rss_mb, run_steps


def emit(**kw):
    print(json.dumps(kw))
    sys.stdout.flush()


def timeit(fn, n):
    """Return (wall_ms_per_call, cpu_ms_per_call), best of 3 batches."""
    best_w, best_c = 1e9, 1e9
    for _ in range(3):
        w0, c0 = time.perf_counter(), time.process_time()
        for _ in range(n):
            fn()
        best_w = min(best_w, (time.perf_counter() - w0) / n * 1e3)
        best_c = min(best_c, (time.process_time() - c0) / n * 1e3)
    return round(best_w, 3), round(best_c, 3)


# ---------- reset cost ----------
for obstacles in (False, True):
    env = make_ctrl_env(240, 48, obstacles=obstacles)
    env.reset()
    w, c = timeit(lambda: env.reset(), 200)
    emit(test="reset", obstacles=obstacles, wall_ms=w, cpu_ms=c)
    env.close()


# constructor (new env + p.connect) cost
def new_env():
    e = make_ctrl_env(240, 48, obstacles=False)
    e.close()


w, c = timeit(new_env, 20)
emit(test="construct+close", wall_ms=w, cpu_ms=c)

# ---------- memory growth over resets ----------
for obstacles in (False, True):
    env = make_ctrl_env(240, 48, obstacles=obstacles)
    act = np.tile(np.ones(4) * env.HOVER_RPM, (1, 1))
    gc.collect()
    base = rss_mb()
    marks = {}
    for i in range(1, 1001):
        env.reset()
        for _ in range(48):  # 1 s of sim per episode
            env.step(act)
        if i in (1, 100, 250, 500, 1000):
            gc.collect()
            marks[i] = round(rss_mb() - base, 1)
    emit(
        test="mem_growth_resets",
        obstacles=obstacles,
        base_rss_mb=round(base, 1),
        delta_mb_at_reset=marks,
    )
    env.close()

# memory growth over create/close cycles (alternative reset strategy)
gc.collect()
base = rss_mb()
marks = {}
for i in range(1, 201):
    e = make_ctrl_env(240, 48)
    e.reset()
    e.step(np.ones((1, 4)) * e.HOVER_RPM)
    e.close()
    if i in (1, 50, 100, 200):
        gc.collect()
        marks[i] = round(rss_mb() - base, 1)
emit(test="mem_growth_construct_close", delta_mb_at_iter=marks)

# ---------- camera / raycast / contact ----------
env = make_ctrl_env(240, 48, obstacles=True)
env.reset()
env.IMG_RES = np.array([64, 48])
w, c = timeit(lambda: env._getDroneImages(0, segmentation=True), 200)
emit(test="_getDroneImages 64x48 seg=True", wall_ms=w, cpu_ms=c)
w, c = timeit(lambda: env._getDroneImages(0, segmentation=False), 200)
emit(test="_getDroneImages 64x48 seg=False", wall_ms=w, cpu_ms=c)
view = p.computeViewMatrix([0, 0, 1], [1, 0, 1], [0, 0, 1])
for W, H in [(64, 48), (128, 96), (320, 240), (640, 480)]:
    proj = p.computeProjectionMatrixFOV(60, W / H, 0.05, 100)
    n = 100 if W < 320 else 20
    w, c = timeit(
        lambda: p.getCameraImage(
            W,
            H,
            view,
            proj,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=env.CLIENT,
        ),
        n,
    )
    emit(test=f"getCameraImage tiny {W}x{H} rgb+depth (no seg)", wall_ms=w, cpu_ms=c)
for n_rays in (8, 16, 64, 360):
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    frm = [[0, 0, 1]] * n_rays
    to = [[5 * np.cos(a), 5 * np.sin(a), 1] for a in ang]
    w, c = timeit(lambda: p.rayTestBatch(frm, to, physicsClientId=env.CLIENT), 500)
    emit(test=f"rayTestBatch {n_rays} rays", wall_ms=w, cpu_ms=c)
w, c = timeit(
    lambda: p.getContactPoints(bodyA=int(env.DRONE_IDS[0]), physicsClientId=env.CLIENT),
    2000,
)
emit(test="getContactPoints(drone)", wall_ms=w, cpu_ms=c)
w, c = timeit(
    lambda: p.getClosestPoints(
        int(env.DRONE_IDS[0]), 2, 5.0, physicsClientId=env.CLIENT
    ),
    500,
)
emit(test="getClosestPoints(drone, body2, 5m)", wall_ms=w, cpu_ms=c)
env.close()


# ---------- wind via applyExternalForce ----------
class WindAviary(CtrlAviary):
    """Adds a world-frame force to the base link every PyBullet sub-step (forces are cleared after each stepSimulation)."""

    wind_force_n = np.zeros(3)

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            -1,
            forceObj=self.wind_force_n.tolist(),
            posObj=self.pos[nth_drone].tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT,
        )


def wind_run(force, seconds=12, ctrl_hz=48):
    env = quiet(
        WindAviary,
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0, 0, 1.0]]),
        pyb_freq=240,
        ctrl_freq=ctrl_hz,
        gui=False,
        user_debug_gui=False,
    )
    env.wind_force_n = np.array(force)
    pid = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset()
    act = np.zeros((1, 4))
    w0 = time.perf_counter()
    traj = []
    for _ in range(int(seconds * ctrl_hz)):
        s = obs[0]
        act[0], _, _ = pid.computeControl(
            env.CTRL_TIMESTEP,
            s[0:3],
            s[3:7],
            s[10:13],
            s[13:16],
            target_pos=np.array([0, 0, 1.0]),
        )
        obs, *_ = env.step(act)
        traj.append(obs[0, 0:3].copy())
    wall = time.perf_counter() - w0
    traj = np.array(traj)
    env.close()
    return traj, wall, env.M * env.G


for label, f in [
    ("no wind", [0, 0, 0]),
    ("0.02 N +x (~7% weight)", [0.02, 0, 0]),
    ("0.05 N +x (~19% weight)", [0.05, 0, 0]),
    ("0.10 N +x (~38% weight)", [0.10, 0, 0]),
]:
    traj, wall, weight = wind_run(f)
    emit(
        test="wind_pid_hold",
        label=label,
        weight_n=round(weight, 3),
        final_xyz=np.round(traj[-1], 3).tolist(),
        max_abs_x_m=round(float(np.abs(traj[:, 0]).max()), 3),
        min_z=round(float(traj[:, 2].min()), 3),
        wall_s=round(wall, 3),
    )


# wind overhead on a bare loop, compared with CtrlAviary
def bare(cls_env, n=2000):
    obs, _ = cls_env.reset()
    a = np.ones((1, 4)) * cls_env.HOVER_RPM
    c0 = time.process_time()
    for _ in range(n):
        cls_env.step(a)
    return (time.process_time() - c0) / n * 1e6


e1 = make_ctrl_env(240, 48)
e2 = quiet(
    WindAviary,
    initial_xyzs=np.array([[0, 0, 1.0]]),
    pyb_freq=240,
    ctrl_freq=48,
    gui=False,
    user_debug_gui=False,
)
e2.wind_force_n = np.array([0.01, 0, 0])
bare(e1, 500), bare(e2, 500)
emit(
    test="wind_overhead_us_per_ctrl_step",
    base=round(min(bare(e1) for _ in range(3)), 1),
    wind=round(min(bare(e2) for _ in range(3)), 1),
)
```

### bench3_parallel.py - Benchmark 3: multiprocess scaling (3.3)

```python
"""Benchmark 3: aggregate throughput with N independent processes (spawn), PID hover loop, 1 drone each."""

import json
import multiprocessing as mp
import sys
import time

SIM_SECONDS = 30
PYB, CTRL = 240, 48


def worker(barrier, q):
    from bench_common import make_ctrl_env, run_steps

    env = make_ctrl_env(PYB, CTRL, obstacles=True)
    run_steps(env, 100, True)
    barrier.wait()  # start all at once
    n = SIM_SECONDS * CTRL
    t0 = time.perf_counter()
    wall, cpu = run_steps(env, n, True)
    q.put((n, wall, cpu, time.perf_counter() - t0))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    for nproc in (1, 2, 4, 6, 8, 10):
        barrier, q = ctx.Barrier(nproc), ctx.Queue()
        procs = [ctx.Process(target=worker, args=(barrier, q)) for _ in range(nproc)]
        for pr in procs:
            pr.start()
        res = [q.get() for _ in procs]
        for pr in procs:
            pr.join()
        agg_ctrl = sum(r[0] / r[1] for r in res)
        print(
            json.dumps(
                dict(
                    nproc=nproc,
                    agg_ctrl_steps_per_s=round(agg_ctrl),
                    agg_phys_steps_per_s=round(agg_ctrl * PYB / CTRL),
                    per_proc_ctrl_steps_per_s=round(agg_ctrl / nproc),
                    per_proc_cpu_ctrl_steps_per_s=round(
                        sum(r[0] / r[2] for r in res) / nproc
                    ),
                    agg_realtime_x=round(agg_ctrl / CTRL),
                )
            )
        )
        sys.stdout.flush()
```

### bench4_semantics.py - Benchmark 4: velocity/braking, collisions, spawn cost, determinism, multi-client (3.7)

```python
"""Benchmark 4: semantics spikes - VelocityAviary speed/braking, collision detection, runtime obstacle spawn,
determinism, multi-client-in-one-process, camera cost without obstacles."""

import hashlib
import json
import sys
import time

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from bench_common import make_ctrl_env, quiet


def emit(**kw):
    print(json.dumps(kw))
    sys.stdout.flush()


def make_vel(ctrl=48, pyb=240, obstacles=False, z=1.0):
    return quiet(
        VelocityAviary,
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0, 0, z]]),
        pyb_freq=pyb,
        ctrl_freq=ctrl,
        gui=False,
        obstacles=obstacles,
        user_debug_gui=False,
    )


# ---- VelocityAviary: speed limit, accel to cruise, braking distance ----
env = make_vel()
emit(
    test="vel_speed_limit",
    SPEED_LIMIT_mps=round(env.SPEED_LIMIT, 3),
    MAX_SPEED_KMH=env.MAX_SPEED_KMH,
    MAX_RPM=round(env.MAX_RPM),
    T2W=env.THRUST2WEIGHT_RATIO,
    mass_kg=env.M,
)
obs, _ = env.reset()
go = np.array([[1, 0, 0, 1.0]])
stop = np.array([[0, 0, 0, 0.0]])
t_log = []
for k in range(48 * 4):  # 4 s accelerate toward +x at full speed fraction
    obs, *_ = env.step(go)
    t_log.append((k / 48, obs[0, 0], obs[0, 10]))
v_cruise = obs[0, 10]
x_at_brake = obs[0, 0]
t90 = next(t for t, x, v in t_log if v >= 0.9 * v_cruise)
for k in range(48 * 4):  # then zero-velocity command (BRAKE)
    obs, *_ = env.step(stop)
    if abs(obs[0, 10]) < 0.05 * v_cruise:
        break
emit(
    test="vel_brake",
    v_cruise_mps=round(float(v_cruise), 3),
    t_to_90pct_s=round(t90, 2),
    brake_time_s=round((k + 1) / 48, 2),
    brake_distance_m=round(float(obs[0, 0] - x_at_brake), 3),
    z_after=round(float(obs[0, 2]), 3),
)
env.close()

# ---- collision detection with a runtime-spawned box (no built-in collision flag in the env) ----
env = make_vel(obstacles=False)
obs, _ = env.reset()
t0 = time.perf_counter()
box = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 1, 1]),
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.1, 1, 1], rgbaColor=[1, 0, 0, 1]
    ),
    basePosition=[1.0, 0, 1.0],
    physicsClientId=env.CLIENT,
)
spawn_ms = (time.perf_counter() - t0) * 1e3
hit_step = None
for k in range(48 * 5):
    obs, *_ = env.step(go)
    cps = p.getContactPoints(
        bodyA=int(env.DRONE_IDS[0]), bodyB=box, physicsClientId=env.CLIENT
    )
    if cps:
        hit_step = k
        break
emit(
    test="collision_detect_runtime_box",
    spawn_ms=round(spawn_ms, 3),
    hit_after_ctrl_steps=hit_step,
    hit_x=round(float(obs[0, 0]), 3),
    n_contacts=len(cps),
)
t0 = time.perf_counter()
p.removeBody(box, physicsClientId=env.CLIENT)
emit(test="removeBody_ms", ms=round((time.perf_counter() - t0) * 1e3, 3))
# does the env ignore the world-ground contact? drone starts on/near ground at z=0.1
env.close()

# ---- spawn cost of a bigger URDF mid-episode (as in examples/pid.py 'rubber ducks') ----
env = make_ctrl_env(240, 48)
env.reset()
import pybullet_data

p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=env.CLIENT)
for name in ("cube_small.urdf", "duck_vhacd.urdf", "samurai.urdf"):
    t0 = time.perf_counter()
    bid = p.loadURDF(name, [3, 3, 1], physicsClientId=env.CLIENT)
    ms = (time.perf_counter() - t0) * 1e3
    p.removeBody(bid, physicsClientId=env.CLIENT)
    emit(test="loadURDF_ms", urdf=name, ms=round(ms, 2))
env.close()


# ---- determinism: same actions, same seed, fresh process-local envs ----
def trajectory_hash():
    e = make_vel()
    o, _ = e.reset(seed=123)
    rng = np.random.default_rng(0)
    h = hashlib.sha256()
    for _ in range(300):
        a = np.array([[*rng.uniform(-1, 1, 3), rng.uniform(0, 1)]])
        o, *_ = e.step(a)
        h.update(o.tobytes())
    e.close()
    return h.hexdigest()[:16], o[0, :3].round(6).tolist()


r1, r2 = trajectory_hash(), trajectory_hash()
emit(test="determinism_same_process", equal=r1 == r2, h1=r1[0], h2=r2[0])
# reset-vs-fresh: does env.reset() reproduce the first episode exactly? (PID integrators are NOT reset by env.reset())
e = make_vel()


def episode(e):
    o, _ = e.reset(seed=1)
    h = hashlib.sha256()
    for k in range(150):
        o, *_ = e.step(np.array([[1, 0.5, 0, 1.0]]))
        h.update(o.tobytes())
    return h.hexdigest()[:16]


ha, hb = episode(e), episode(e)
for c in e.ctrl:
    c.reset()
hc = episode(e)
emit(
    test="determinism_reset_reuse",
    ep1=ha,
    ep2_no_ctrl_reset=hb,
    ep3_after_ctrl_reset=hc,
)
e.close()

# ---- several clients in one process (DIRECT): isolation and client ids ----
e1, e2 = make_vel(), make_vel()
o1, _ = e1.reset()
o2, _ = e2.reset()
for _ in range(48):
    e1.step(go)
    e2.step(stop)
emit(
    test="two_envs_one_process",
    client_ids=[e1.CLIENT, e2.CLIENT],
    x1=round(float(e1.pos[0, 0]), 3),
    x2=round(float(e2.pos[0, 0]), 3),
)
e1.close()
e2.close()

# ---- camera cost with no obstacles (scene complexity dependence) ----
env = make_ctrl_env(240, 48, obstacles=False)
env.reset()
env.IMG_RES = np.array([64, 48])
for _ in range(5):
    env._getDroneImages(0, segmentation=False)
n = 100
t0 = time.perf_counter()
for _ in range(n):
    env._getDroneImages(0, segmentation=False)
emit(
    test="_getDroneImages 64x48 no obstacles",
    ms=round((time.perf_counter() - t0) / n * 1e3, 3),
)
env.close()
```

### bench5_speed.py - Benchmark 5: SPEED_LIMIT sweep (3.7 table)

```python
"""Benchmark 5: raise VelocityAviary.SPEED_LIMIT; measure achieved cruise speed, tilt, braking time/distance (ctrl 48 Hz)."""

import json
import numpy as np
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel
from bench_common import quiet

for limit in (0.25, 1.0, 2.0, 4.0, 6.0):
    for ctrl in (48,):
        env = quiet(
            VelocityAviary,
            drone_model=DroneModel.CF2X,
            initial_xyzs=np.array([[0, 0, 1.0]]),
            pyb_freq=240,
            ctrl_freq=ctrl,
            gui=False,
            user_debug_gui=False,
        )
        env.SPEED_LIMIT = limit
        obs, _ = env.reset()
        go, stop = np.array([[1, 0, 0, 1.0]]), np.array([[0, 0, 0, 0.0]])
        max_tilt = 0.0
        for k in range(ctrl * 6):
            obs, *_ = env.step(go)
            max_tilt = max(max_tilt, abs(obs[0, 7:9]).max())
        v, x0, z_min = float(obs[0, 10]), float(obs[0, 0]), float(obs[0, 2])
        steps = 0
        for steps in range(1, ctrl * 10):
            obs, *_ = env.step(stop)
            z_min = min(z_min, float(obs[0, 2]))
            max_tilt = max(max_tilt, abs(obs[0, 7:9]).max())
            if abs(obs[0, 10]) < 0.05 * max(v, 1e-6):
                break
        print(
            json.dumps(
                dict(
                    speed_limit=limit,
                    ctrl_hz=ctrl,
                    cruise_mps=round(v, 2),
                    brake_time_s=round(steps / ctrl, 2),
                    brake_dist_m=round(float(obs[0, 0]) - x0, 2),
                    max_tilt_deg=round(float(np.degrees(max_tilt)), 1),
                    min_z=round(z_min, 2),
                )
            )
        )
        env.close()
```

## 10. Sources (accessed 2026-09-30)

- gym-pybullet-drones repo + README + pyproject + git history (cloned; `git log`, `git show 29fc5b1^:README.md`): https://github.com/learnsyslab/gym-pybullet-drones
- GitHub repo page (stars, issues, install text): https://github.com/learnsyslab/gym-pybullet-drones
- GitHub releases API (only v1.0.0, 2021-03-09): https://api.github.com/repos/learnsyslab/gym-pybullet-drones/releases
- PyPI JSON for pybullet 3.2.7 (file list: manylinux x86_64 wheels cp36-cp311 + sdist, upload 2025-01-30): https://pypi.org/pypi/pybullet/json
- Old performance claims: README at commit `29fc5b1^` (2023-04-15 parent), same repo.
- Everything numeric in sections 3 and 4 is from the scripts in section 9, run on this machine.

## Decisions recommended

1. **Use gym-pybullet-drones, installed on Python 3.14.2 from a git checkout**, with pybullet built from source using `CFLAGS="-Dfdopen=fdopen"`; cache the built wheel. No separate 3.10-3.12 environment is required for wheel reasons. Keep the sim in **worker processes** behind a narrow interface for crash isolation and to quarantine the library's `print()/exit()/KMP_DUPLICATE_LIB_OK` side effects. (High confidence.)
2. **Do not rely on the "20 kHz / 80 drones / 4 envs" claim or quote it.** It is pre-2023, aggregate, Intel-era, and not in the current README. State instead our own measured figures: ~130-170k raw steps/s and ~37k with PID per core; ~4.5-7k control steps/s/core at 240/48 with PID. (High.)
3. **Subclass `CtrlAviary` (or `BaseAviary`)**, not the RL classes or `VelocityAviary`, and own the control stack: our `DSLPIDControl` wrapper, `ctrl.reset()` in `_housekeeping`, our 11-action mapper in `_preprocessAction`, a FIFO for command latency/actuator lag, wind in `_physics`, collisions from `getContactPoints`, obstacles from `createMultiBody` (seeded, spawned at event ticks). Set `obstacles=False`. (High.)
4. **Sim time is driven purely by `env.step()`; never call `sync()`**. Default `pyb_freq=240, ctrl_freq=48` (20.8 ms tick); reserve `ctrl=240` (4.2 ms tick) for fine latency sweeps; avoid pyb >= 1000 unless needed. Reported decisions/sec and sim_speed must be computed from sim ticks and CPU-time for the sim, separately from provider latency. (High for mechanism; medium for the exact frequency, pending latency-design file 08.)
5. **Cruise speed 1-2 m/s** (raise `SPEED_LIMIT`; default 0.25 m/s is useless for avoidance). At 2 m/s brake distance is 0.54 m / 0.42 s at 48 Hz, which puts the interesting latency range at ~100-500 ms. Avoid >= 4 m/s (tilt > 70 deg). (Medium-high; one controller, one drone model, no wind.)
6. **Perception = rayTestBatch**, not camera images (0.18 ms for 360 rays vs ~10 ms per 64x48 image with default scene, 2 ms with an empty scene). Images only for an optional multimodal arm. (High.)
7. **Pool size 6-8 worker processes** (4 P + up to 4 E cores), `spawn` start method, one PyBullet client per process, leave 1-2 cores for the harness. Expected sim-only capacity ~20k control steps/s aggregate; sim is never the bottleneck vs LLM calls. (Medium; contaminated by shared-machine load.)
8. **Plan for wind to be a weak stressor**: a hovering PID cancels steady wind in seconds; use time-varying gusts while translating and size forces to 0.05-0.3 N (20-110% of the 0.265 N weight; >~1 N lateral exceeds authority). (Medium.)
9. Dependency note for file 14: `uv pip install --no-deps -e` plus explicit numpy/scipy/gymnasium/transforms3d/pybullet (+ control/matplotlib if imports demand) for the sim worker env; torch/SB3 only if an RL arm is built (they do install on 3.14 arm64).

## Open questions

1. Cross-process determinism: does the same scenario+controller produce bit-identical trajectories in two different processes / after interpreter restart / at different times of day? (Expected yes; unverified.)
2. Does the headline per-core throughput hold on an idle machine? My final numbers were taken at load 6-10 on a 10-core box shared with other agents; parallel scaling results especially are noisy (10-proc case collapsed in run B).
3. Memory: no growth in 1000 resets by RSS, but macOS RSS is unreliable (it decreased); need a 10k-reset soak using `footprint`/`vmmap` to rule out a slow leak.
4. `DSLPIDControl` tuning at 2 m/s with wind: how does it behave under 0.3 N gusts while cruising (altitude sag, saturation)? Not tested (only hover-with-wind).
5. Is the ~100 µs PID cost reducible enough to matter? Probably not worth it; decide after measuring where Jev/LLM arms spend time.
6. Thread safety of one PyBullet client used from an asyncio thread pool (relevant if the harness ever steps the sim from a thread): untested; the recommendation is processes.
7. GUI/OpenGL on the current macOS for the demo recording path (file 12): untested here.
8. Exact compiler diagnostic for the plain-pybullet failure (I only identified the failing file `zutil.c` and that the README flag fixes it); matters if Xcode CLT versions change.
9. Whether 2-drone scenarios (other drone enters trajectory) need `Physics.PYB_DW` downwash or whether simple kinematic second drone is enough (file 06).

## Follow-up spikes (small, runnable experiments)

1. **Idle-machine rerun**: `uv run python bench1_steps.py` and `bench3_parallel.py` (section 9) on a quiet machine (`caffeinate -i`, Activity Monitor empty); record `sysctl vm.loadavg` before/after; target: reproduce >=25k ctrl steps/s (raw 240/48) and >=7k (PID 240/48) per core.
2. **Cross-process determinism**: run the `determinism` hash in `bench4_semantics.py` in two spawned processes and across two interpreter launches; assert equal digests.
3. **Reset-soak**: 10,000 resets with a primitive-obstacle scenario; sample `footprint -p <pid>` every 500 resets; accept if slope < 0.1 MB/1000 resets.
4. **Latency-injection prototype on this sim**: FIFO of N ticks in `_preprocessAction`; sweep N = 0, 2, 5, 14, 24, 48 ticks (0-1 s at 48 Hz) on a "box appears 1.2 m ahead at 2 m/s" scenario with a PID+TTC-brake arm; plot collision rate vs N. (Feeds file 08 and confirms the feasibility window.)
5. **Gust test**: Ornstein-Uhlenbeck wind (sigma 0.1 N, tau 1 s) while cruising at 1-2 m/s; record max cross-track error, tilt, and saturation of RPM.
6. **Minimal-deps worker env**: create a venv with only `pybullet numpy scipy gymnasium transforms3d` (+`control`, `matplotlib` if import fails) and import `CtrlAviary` and `DSLPIDControl`; list what is actually required to drop torch/SB3.
7. **EGL/OpenGL check on macOS**: `p.connect(p.GUI)` smoke test and one `getCameraImage` with `ER_BULLET_HARDWARE_OPENGL` to see whether a faster render path exists for demo recording.
8. **PID cost reduction**: profile `computeControl` (cProfile) and test a numpy-only quaternion path; only pursue if the PID share exceeds 50% of a flight's CPU in the full harness.
