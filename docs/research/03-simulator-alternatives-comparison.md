# 03 — Simulator alternatives comparison

Agent 03 of 14. Access date for all web/GitHub facts: **2026-09-30**. Hardware for spikes: Apple M4, 10 cores, 16 GB, macOS 27.0 (Darwin 27), uv 0.9.18, CPython 3.14.2.
Scratch (not in repo): `/private/tmp/claude-501/-Users-subhajithait-Documents-testing-drone-jev/0ec4827c-5206-473e-b48d-298a2822fdb4/scratchpad/03/` (`bench.py`, `rp.py`, `score.py`).

## 0. Bottom line

1. **Do not use one simulator.** Use a **tiered design**: Tier 0 = a small in-house, deterministic, vectorisable kinematic/point-mass+first-order-lag sim for the thousands-of-flights latency sweeps; Tier 1 = gym-pybullet-drones (PyBullet, DIRECT, `DSLPIDControl`) for a calibrated high-fidelity subset (~5-10 % of flights) that must reproduce Tier 0's *ranking*, not its absolute numbers.
2. **Primary recommendation:** T0 custom + T1 gym-pybullet-drones (gpd). **Fallback if PyBullet cannot be built on the M4 (real risk, see 2.1):** T1 = MuJoCo with a hand-written quadrotor MJCF (installs from a `cp314` arm64 wheel, measured 110 k steps/s), with RotorPy as a pure-Python cross-check.
3. **Contradicts `docs/initial-plan.txt` in three places** (Section 1): the "~20 kHz / 80 drones / 4 envs" number is a legacy-README cherry-pick (and is not claimed for 1 drone); PyBullet has **no macOS wheel at all** and none above cp311, so "just works on Apple Silicon" needs a source build; the project moved to `learnsyslab/`.
4. Scored matrix (Section 4) ranks a single simulator: custom 84, MuJoCo 81, RotorPy 77, gpd 72, Gazebo+PX4 67, Webots 66 ... Isaac/Aerial Gym cannot run on a Mac. gpd is 4th on raw score but is the best-value *validation* tier (Section 6).

## 1. Verified facts about the plan's primary pick (gym-pybullet-drones)

| Fact | Evidence (accessed 2026-09-30) | Confidence |
|---|---|---|
| Repo is now `learnsyslab/gym-pybullet-drones` (old `utiasDSL/...` redirects). MIT. 2,149 stars, 111 open issues, not archived. Default branch `main`; `master` and `paper` branches hold the legacy IROS-2021 code. | `gh api repos/learnsyslab/gym-pybullet-drones`, branches list | High |
| Last commit on `main`: 2026-09-06 ("typos", "pybullet installation"). **Only GitHub release is v1.0.0 (2021-03-09)**, so there is no versioned release of the refactor; pin a commit SHA. `pyproject` says version 2.2.0 (poetry), Python `^3.12`, numpy ^2.5, torch ^2.13, pybullet ^3.2.7, gymnasium ^1.3, SB3 ^2.9. | `gh api .../commits`, `.../releases/latest`, `pyproject.toml` | High |
| README (`main`) says "Tested on Intel x64/Ubuntu 24.04 and Apple Silicon/macOS 26." So the plan's claim is **true for `main`**. The **legacy `master` README** says macOS 10.15/12. | `raw.githubusercontent.com/learnsyslab/gym-pybullet-drones/main/README.md` | High |
| README (`main`): "Beyond Python 3.10, `pybullet` has no pre-built wheel ... On macOS, build and install `pybullet` with `CFLAGS="-Dfdopen=fdopen" pip install pybullet --no-cache-dir`". PyPI `pybullet 3.2.7` (2025-01-30) ships only `manylinux x86_64` wheels cp36-cp311 + sdist: **no macOS wheel of any kind**. So on an M4 + Python 3.14.2 pybullet is always a source build. | README + `pypi.org/pypi/pybullet/json` file list | High |
| The ~20 kHz claim exists only in the **legacy** README (`master`/`paper`): "multiple (80) drones in multiple (4) environments ... ~20kHz PyBullet physics updates", next to "~5kHz PyBullet physics (CPU-only)" for the baseline. The current `main` README contains no throughput number. The same table lists "80 drones in 4 env, no vision" at 0.8x-0.95x *real-time factor* in the legacy table, which is a different (paced, GUI-era) metric. | legacy README lines 61-67 | High |
| gpd targets RL training (Gymnasium/SB3) and Betaflight SITL, not PX4. | README | High |

**Inference:** 20 kHz aggregate over 80 drones means ~250 Hz per drone-env; it says nothing about 1 drone on an M4, where PyBullet's ~5 kHz single-env figure is the relevant scale. Agent 02 owns the measured number; this file only uses it as a tier-sizing input.

## 2. Candidate-by-candidate findings

GitHub status pulled via `gh api` (pushed_at = last push to any branch; releases = latest GitHub release).

| Candidate | Last push | Latest release | Archived | License | Stars | Verdict in one line |
|---|---|---|---|---|---|---|
| gym-pybullet-drones | 2026-09-06 | v1.0.0 (2021) | no | MIT | 2.1k | Active, unreleased refactor; pybullet source-build on Mac |
| Webots | 2026-09-29 | R2025a (2025-02-04) | no | Apache-2.0 | 4.7k | Very active, native macOS app, physics headless-capable but slow and not built for 1000s of runs |
| AirSim (Microsoft) | 2026-09-15 (CI pin commits only) | v1.8.1 (2022-07-18) | no, but **legacy** | "NOASSERTION" (MIT historically) | 18.5k | Functionally dead; successor Project AirSim |
| Colosseum (fork) | 2026-07-08 | v2.1.0 (2023-06) | **yes** | - | 669 | Archived fork |
| Isaac Sim | 2026-09-28 | v6.1.0 (2026-09-10) | no | custom/NOASSERTION | 4.2k | Windows/Linux + NVIDIA RTX only |
| Isaac Lab | 2026-09-30 | v3.0.0-EA (2026-09-16) | no | BSD-3 | 8.3k | Same NVIDIA constraint |
| PX4-Autopilot | 2026-09-30 | v1.17.0 (2026-05-13) | no | BSD-3 | 12.7k | Active; SITL is the gold standard for flight-stack fidelity |
| ArduPilot | 2026-09-30 | Plane-4.7.1 (2026-09-03) | no | GPL-3.0 | 16k | Active; GPL |
| gz-sim (Gazebo) | 2026-09-30 | - | no | Apache-2.0 | 1.5k | Active |
| Flightmare | **2024-06-14 push; last real commits 2023-05-15** | none | no | NOASSERTION | 1.4k | Abandoned |
| Aerial Gym | 2026-06-28 | 2.0.1 (2026-06-28) | no | BSD-3 | 777 | Active, NVIDIA-only (Isaac Gym / Isaac Sim WIP) |
| MuJoCo | 2026-09-30 | 3.14.0 (2026-09-22) | no | Apache-2.0 | 15.4k | Very active, arm64 cp314 wheel |
| Genesis | 2026-09-30 | v1.4.3 (2026-09-30) | no | Apache-2.0 | 30k | Very active, **`requires-python <3.14`** |
| RotorPy | 2026-09-07 | v3.0.0 (2026-09-07) | no | MIT | 311 | Active, pure Python, installs on 3.14.2 |
| CrazyS | **2022-08-11** | none | no | Apache-2.0 | 177 | Abandoned (ROS1/Gazebo) |
| Unity ML-Agents | 2026-09-29 | release_23 (2025-09-02) | no | "NOASSERTION" (Apache-2.0 historically) | 19.7k | Active; Python pkg pinned `<=3.10.12` |
| safe-control-gym (gpd sibling) | 2026-04-29 | - | no | MIT | 918 | Relevant only if MPC/safe-RL on PyBullet is wanted; also under `learnsyslab` |

### 2.1 gym-pybullet-drones (PyBullet)
- **Apple Silicon / no Docker:** works, but pybullet needs a C++ source build (Xcode CLT; the README's `CFLAGS` workaround). Python 3.14 build not verified by this agent (agent 02 owns it). The fix is to install pybullet in a separate 3.11/3.12 env behind a process boundary (brief, global Q4) if 3.14 build fails. Confidence medium-high that a source build on 3.12 works; medium on 3.14.
- **Headless throughput:** DIRECT mode, CPU only, single-threaded per client. ~5 kHz physics per env (legacy README). Good enough: 1 drone at 240 Hz physics = ~20x real-time on one core.
- **Sensors:** kinematic state exact; RGB/depth/segmentation via `getCameraImage` (CPU TinyRenderer in DIRECT; slow; avoid in sweeps). No lidar natively, but `p.rayTestBatch` gives one (agent 07).
- **Adversarial events:** wind/gusts = `applyExternalForce` (easy); obstacles = `loadURDF` of boxes/spheres at runtime (native, cheap); battery/GPS/sensor noise/moving target/second drone = custom wrappers around the observation/action path (all easy, because Python owns the loop).
- **Multi-drone:** native (`num_drones`, same client), collision between drones supported by the physics.
- **PX4/MPC:** no PX4; Betaflight SITL only. MPC must be in Python (agent 05).
- **Determinism:** PyBullet is deterministic for a fixed build + fixed step order in one process (agent 08/02 to verify across processes).

### 2.2 Webots (R2025a)
- Native macOS app (Apache-2.0); repo is very active (commits 2026-09-26/27 include macOS asset fixes). Stock Mavic 2 Pro demo has GPS/IMU/gyro/compass/camera and a PID controller in C and Python. Wind is not a built-in physics feature of the Mavic world (unverified; it can be injected through the Supervisor API `Node.addForce`).
- **Throughput:** real-time-factor simulation with optional fast mode; no vectorisation; one Webots process per simulation and a separate controller process. I did not find a measured headless drone benchmark (web search returned none); treat "fast" as O(10-100x) real time for a single drone without rendering, **unverified**. Startup per world is seconds, so thousands of resets are expensive unless you reset in place.
- **Integration:** controller-process IPC adds latency jitter that would confound decision-latency measurement unless the Webots clock is the only clock (it can be: Webots is synchronous by default).
- **Verdict:** best visuals/sensors on Mac, wrong tool for 10^3-10^4 flights. Plan's "don't start with Webots" is correct.

### 2.3 AirSim / Colosseum / Project AirSim
- Microsoft's AirSim repo is not flagged archived but the last release is v1.8.1 (2022-07-18) and the docs describe it as legacy/archived with no new features (AirStack docs, Register 2022, issue #5023). 2026 commits are CI action pinning only. Colosseum (community fork) is **archived**. Successor "Project AirSim" (MIT, UE5) targets Linux/Windows + NVIDIA.
- Unreal Engine on a 16 GB Mac is exactly what the plan wanted to avoid; 1 kHz physics; no headless throughput for thousands of runs. **Reject.**

### 2.4 Isaac Sim / Isaac Lab / OmniDrones / Aerial Gym
- Isaac Sim README: "Operating System: Windows 11 or Linux (Ubuntu 22.04/24.04)" and RTX-class NVIDIA GPUs; no macOS. Isaac Lab and Aerial Gym inherit the constraint (Aerial Gym 2.0: GPU ray-cast lidar/camera, "thousands of MAVs", Isaac Sim support "under development"). Outstanding throughput on NVIDIA, **zero** on an M4. **Reject for local Mac**; mention as optional cloud-GPU future, out of scope (no Docker, local-only).

### 2.5 Gazebo + PX4 / ArduPilot SITL
- PX4 docs: Gazebo support is documented for Ubuntu (22.04/24.04); macOS is **not** documented as supported for the current Gazebo (Harmonic+) path. `HEADLESS=1` and `PX4_SIM_SPEED_FACTOR` supported; docs say powerful desktops reach ~6-10x real time. Plan-relevant: PX4 SITL uses **lockstep**, so sim time is controlled by the simulator, which is helpful for the clock model (agent 08).
- **PX4 SIH** (Simulation-In-Hardware, PX4 docs): headless, zero external dependencies, physics inside PX4 over uORB, wind params (`SIH_WIND_N/E`), multi-vehicle via PX4 instances, lockstep deterministic, faster-than-real-time (demonstrated 10x). macOS is not explicitly documented; PX4 has a macOS build toolchain (inference, medium-low confidence that SIH runs on macOS without issues). Sensor model is IMU/GPS/baro/mag/airspeed only (no rangefinder/camera).
- **Fidelity win:** real flight-stack cascaded controllers, EKF with GPS-loss behaviour. **Cost:** C++ build, MAVLink/uXRCE-DDS, per-flight restart (seconds), hard to do 10^4 flights/hour on 10 cores. Discrete action -> setpoint mapping via offboard mode is additional integration surface.
- ArduPilot SITL: GPL-3.0 and Python-friendly (MAVProxy/pymavlink), same throughput class. Licence is fine for private local experiments; note copyleft if ever redistributed.
- **Verdict:** excellent if the claim must be "works with a real autopilot"; wrong for the main sweep. Keep as a "Tier 2 (optional, stretch)" for a 50-flight sanity run, not in the core plan.

### 2.6 Flightmare, CrazyS
- Flightmare: last commits 2023-05-15, Unity render engine + ROS + custom C++ quadrotor model (35 kHz simple single-drone dynamics per gpd README), no macOS support path. **Abandoned.** CrazyS: ROS1/Gazebo, last commit 2022-08-11. **Abandoned.**

### 2.7 MuJoCo (+ MJX, MuJoCo Warp)
- **Verified in a scratch venv (Python 3.14.2, arm64):** `uv pip install mujoco` -> mujoco 3.14.0 from a `cp314` wheel; PyPI lists `macosx_11_0_arm64` wheels for cp310-cp315. numpy 2.5.3, scipy 1.18.1 also have cp314 arm64 wheels. **This is the only serious physics engine with a first-class Python 3.14 arm64 wheel.**
- MJX-JAX: runs on XLA backends; Apple GPU via `jax-metal` has "missing a lot of features" (GitHub issue #40 in `hello-robot/stretch_mujoco`, JAX discussion #9847). MuJoCo Warp requires NVIDIA ("designed for NVIDIA hardware"; CPU only for dev). **Conclusion: on M4, use plain CPU MuJoCo, not MJX/Warp.** Batched GPU simulation is unavailable, so MJX adds nothing here.
- No ready-made quadrotor env in core MuJoCo; write a 40-line MJCF (free joint + thrust actuators or a body-wrench; MuJoCo Menagerie has quadrotor models, unverified in this pass). Wind via `xfrc_applied` or fluid drag (`<option wind=...>` is built in, inference from MuJoCo docs, medium confidence), native `mj_ray` for lidar, native `rangefinder` sensors, noisy-sensor plugins are custom.
- **Measured (this M4, Py 3.14.2):** free-body 27 g quad + box obstacle, dt=4 ms: **110,251 steps/s** (single thread), i.e. 440x real time per drone. With 16 python-level `mj_ray` calls per step (naive loop, numpy allocs each call): 7,361 steps/s; expect >10x better if rays are vectorised (`mj_multiRay`) — untested.

### 2.8 Genesis
- v1.4.3 released today; Apache-2.0; 30 k stars; Quadrants compiler targets CUDA/ROCm/Metal/Vulkan/CPU; has a Drone hover example. **`requires-python = ">=3.10,<3.14"`** in `pyproject.toml` means it does **not** install on 3.14.2 (needs a 3.12/3.13 env). Its strengths (GPU-batched physics, Metal backend) are real on M4 in principle, but I did not run it. Risk: young API, fast-moving, heavy deps (torch, open3d optional). Not a good basis for a reproducible benchmark that other people/agents re-run for months. Confidence medium.

### 2.9 RotorPy (UPenn)
- v3.0.0 2026-09-07; MIT; pure Python. **Verified:** `uv pip install rotorpy` on Python 3.14.2 installs and imports. Features (README): wind profiles, IMU + motion-capture + new pinhole camera sensor, batched multi-drone (25x speedup at >1000 drones on CPU vs sequential), SE3 controller, gymnasium env, optional PX4 extra.
- **Measured:** `Multirotor.step` with Crazyflie params, dt=4 ms, default integrator: **~250 steps/s** single drone (3000 steps, this M4). That is 1x real time — about 400x slower than MuJoCo, 13x slower than the PyBullet ~5 kHz. Caveat: untuned quick test (default scipy ODE-based integrator; a fixed-step RK/Euler mode or the batched path would be faster). So, as shipped, it cannot be the throughput tier; it is good as a **dynamics cross-check** (rotor-speed-level model with aero drag) for calibrating T0.

### 2.10 Unity ML-Agents
- Repo active (2026-09-29), last release_23 (2025-09-02). PyPI `mlagents 1.1.0` `requires_python <=3.10.12,>=3.10.1`, last upload 2024-10-05 -> incompatible with 3.14.2 and pinned old. Requires Unity editor (GUI, several GB), no quadrotor physics out of the box, no PX4. **Reject.**

### 2.11 Custom minimal simulator (Tier 0)
**Measured on this M4 / Py 3.14.2 (bench.py):**

| Implementation | Result |
|---|---|
| Pure-Python floats, 1-axis first-order-lag point mass | 3.33 M steps/s |
| NumPy 3-vector per step (python loop, tiny arrays) | 114 k steps/s (overhead-dominated) |
| NumPy batched, 1,000 drones per step | 66.3 M drone-steps/s (= ~66 k batched steps/s x 1000 drones) |
| MuJoCo quad (dt 4 ms) | 110 k steps/s |
| RotorPy quad | 250 steps/s |

A batched NumPy kinematic tier can push 10^7-10^8 drone-steps/s, so **10,000 flights of 60 s at 100 Hz (6x10^7 steps) cost about 1-10 s of physics**; everything else (decision logic, LLM calls) dominates. That is what makes latency sweeps at 7+ latency levels x N controllers x 10^4 scenarios practical.

## 3. Criteria and weights (sum = 100)

| # | Criterion | Weight | Justification |
|---|---|---|---|
| 1 | Install friction on M4/no-Docker/Py 3.14.2 | 15 | Hard constraint from brief; a sim that needs a VM or GPU has score 0 regardless of quality |
| 2 | Headless throughput (1 drone, flights/hour) | 20 | The experiment's core is "thousands of flights x latency levels"; throughput is what enables statistical power (agent 08) |
| 3 | Adversarial-event support/ease of injection | 15 | Seven event types, 6 of which are custom in every candidate; ease of writing them in Python and seeding them matters |
| 4 | Sensor fidelity | 10 | LLM prompts are text state; cameras are an optional arm. Important but not central |
| 5 | Multi-drone | 5 | Only one of seven events needs a second drone |
| 6 | PX4/MPC integration | 5 | Plan says PID/MPC is low-level; PX4 is optional; MPC is done in Python |
| 7 | Python loop control/determinism (own the clock) | 10 | Need lockstep/latency injection and seeded paired scenarios (agent 08) |
| 8 | Maintenance health (2026-09-30 data) | 10 | Experiment lifetime is months; abandoned = risk |
| 9 | Licence | 5 | All fine except GPL (ArduPilot) and custom NVIDIA terms |
| 10 | Time-to-first-result | 5 | YAGNI; prefer an env that already has a quadrotor |

Scores 0-5. Weighted total = sum(weight x score)/5, so max = 100.

| Simulator | Install (15) | Thru-put (20) | Events (15) | Sensors (10) | Multi (5) | PX4/MPC (5) | Loop/det. (10) | Maint. (10) | Lic. (5) | TTFR (5) | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Custom minimal (numpy)** | 5 | 5 | 5 | 2 | 5 | 0 | 5 | 3 | 5 | 4 | **84** |
| **MuJoCo custom quad** | 5 | 4 | 4 | 3 | 4 | 1 | 5 | 5 | 5 | 2 | **81** |
| **RotorPy** | 5 | 2 | 4 | 3 | 4 | 3 | 5 | 5 | 5 | 4 | **77** |
| **gym-pybullet-drones** | 3 | 3 | 4 | 3 | 4 | 1 | 5 | 4 | 5 | 5 | **72** |
| Gazebo + PX4/ArduPilot | 2 | 3 | 4 | 4 | 4 | 5 | 2 | 5 | 5 | 1 | 67 |
| Webots | 4 | 2 | 3 | 5 | 3 | 2 | 2 | 5 | 5 | 3 | 66 |
| PX4 SIH (headless) | 2 | 3 | 3 | 2 | 3 | 5 | 2 | 5 | 5 | 2 | 60 |
| Genesis | 1 | 3 | 3 | 3 | 4 | 1 | 4 | 5 | 5 | 2 | 60 |
| Unity ML-Agents | 1 | 3 | 3 | 4 | 4 | 0 | 3 | 4 | 4 | 1 | 55 |
| Isaac Sim/Lab (OmniDrones) | 0 | 0 | 4 | 5 | 5 | 3 | 3 | 5 | 3 | 0 | 49 |
| Flightmare | 1 | 4 | 2 | 4 | 4 | 2 | 2 | 0 | 3 | 1 | 47 |
| AirSim (legacy) | 1 | 1 | 3 | 5 | 3 | 5 | 3 | 1 | 4 | 1 | 47 |
| Aerial Gym | 0 | 0 | 3 | 4 | 5 | 1 | 3 | 4 | 5 | 0 | 42 |
| CrazyS | 0 | 1 | 2 | 3 | 3 | 1 | 1 | 0 | 5 | 0 | 27 |

Scoring notes (subjective, inference): gpd install = 3 (source build of pybullet); PyBullet throughput = 3 (~5 kHz per env, legacy figure, below MuJoCo's measured 110 kHz here though PyBullet includes a richer quadrotor stack incl. drag/ground effect/downwash models, which is worth something as "fidelity"); Isaac/Aerial Gym throughput = 0 *on this hardware*. The custom sim wins points because it is tuned to this experiment, and scores 0 on PX4/MPC and 2 on sensors; the score does not mean it is a credible standalone sim (hence tiering).

**Sensitivity (runs of `score.py`):**
- Fidelity-heavy weights (sensors 20, PX4 10, throughput 10): top four = RotorPy 68, MuJoCo 66, Gazebo+PX4 66, Webots 65. gpd drops out of the top four.
- Throughput-heavy (throughput 35, PX4 0, TTFR 10): custom 91, MuJoCo 82, gpd 75, RotorPy 72.
- Stable across both: custom and MuJoCo near the top; AirSim/Flightmare/Isaac/Aerial Gym/CrazyS at the bottom. gpd is **never first** on raw score; the case for it is validation-tier credibility and ready-made PID/quadrotor stack (a judgment, not a score).

## 4. Recommendation and fallback

**Primary: tiered T0 custom + T1 gym-pybullet-drones.**
- Reason to keep gpd despite the score: the user's plan is built on it; it ships `DSLPIDControl` (a real cascaded PID with attitude loop), 4-rotor RPM dynamics, ground effect/drag/downwash options, and obstacles via URDF — so that "a normal PID handles stabilisation" is demonstrated with a real controller, not a first-order-lag lie. It also means the "demo" (agent 12) uses a recognisable environment.
- Conditions to keep it primary: Gate G1 passes (see Follow-ups): pybullet builds on M4 and gpd steps headless at >= 2 kHz/drone-env on one core under Python 3.12 (or 3.14.2 if the build works).
- **Fallback (automatic if G1 fails):** T1 = MuJoCo quad MJCF driven by a small Python PID cascade written in the repo. It has a verified cp314 arm64 wheel and 110 kHz raw stepping, so the whole repo stays in a single Python 3.14.2 env with no process boundary. Cost: write and validate your own quadrotor+attitude controller (~2-3 days). RotorPy stays available as the independent dynamics reference.
- Explicit non-choices: Isaac/Aerial Gym (no macOS), AirSim/Flightmare/CrazyS (dead), Unity ML-Agents (Py <= 3.10.12), Genesis (Py < 3.14; revisit only as an optional GPU-batched tier), Webots/Gazebo (per-run process overhead, poor for 10^3-10^4 flights; Webots is fine for a one-off glossy video).

## 5. Tiered-fidelity design (concrete)

```
Tier 0  "kin"  : numpy-batched, B scenarios per step, dt = 10 ms (100 Hz).
                 State: p, v (3D); command = velocity setpoint from action mapping (agent 11).
                 Dynamics: v += clip(a_cmd(v_cmd - v)/tau, a_max_xy, a_max_z) dt ; p += v dt
                          + wind drift term (OU gust added to v) ; battery derates a_max and v_max
                 Obstacles = spheres/AABBs, analytic ray/segment intersection; sensors = state + noise.
                 Used for: latency sweep (0..3000 ms), 10^4 scenarios x 7 controllers, power analysis.
Tier 1  "pb"   : gym-pybullet-drones DIRECT, DSLPIDControl at 240/48 Hz, same Scenario JSON.
                 Used for: 5-10 % stratified subset (e.g. 500-1000 flights), GUI demo replay, calibration.
Tier 2  (opt)  : PX4 SIH or Gazebo+PX4 (Linux box/VM, out of scope for the no-Docker Mac plan).
```

**Shared interface** (single `Scenario`, `EventInjector`, `Controller`, `DecisionProvider`, `MetricCollector` protocols; owner agent 14):

```python
class SimBackend(Protocol):
    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation: ...
    def step(
        self, setpoint: Setpoint, n_ticks: int = 1
    ) -> StepResult: ...  # physics+events
    @property
    def sim_time_s(self) -> float: ...
    def close(self) -> None: ...


# KinematicBackend(batched) and PyBulletBackend both implement it; events are data, not code paths.
```

**Calibration (T0 <- T1), fit before any T0 results are trusted:**
1. Step-response identification in T1: command +1 m/s (and 3 m/s) velocity steps, -3 m/s braking, climb/descend steps, log v(t); fit `tau` and `a_max_{xy,z}` for T0 (first-order lag with accel/jerk limits). Include PID overshoot and settling time as extra T0 knobs if fit error > 10 %.
2. Braking-distance check: `d_brake(v0)` in T0 must agree with T1 within 10 % for v0 in {1,2,3,5} m/s. The latency "knee" (agent 13) is determined by this curve, so this is the most important calibration.
3. Wind response: constant 3 m/s wind step -> steady-state drift of the PID under T1 vs T0's drift term.

**Validation (is the tiering valid?) — rank-preservation, not absolute equality:**
- Run the same 500-1000 paired `ScenarioSpec` seeds through both tiers for all arms at 3 latencies {0, 100, 1000 ms}.
- Acceptance: (a) Spearman rho >= 0.9 over arm ranking by collision rate at each latency; (b) collision-vs-latency curve knee location within one latency grid step; (c) per-scenario collision agreement (Cohen's kappa) >= 0.6; (d) absolute collision-rate gap reported with CI, **never hidden**. If (a) or (b) fails, T0 results are labelled "screening only" and headline claims are restricted to T1.
- Report both tiers in every figure; headline numbers use T1 only unless the validation passes, and the claim wording says "on the kinematic tier, validated against PyBullet on N flights".

**Why this is better than T1 only:** T1 at ~5 kHz physics and 100 Hz control per drone ~= 50x real time; 10^4 flights x 7 arms x 7 latencies x 30 s = 1.5 x 10^7 sim-seconds, i.e. ~85 core-hours at 50x. On 6-8 worker cores (agent 09) that is ~11-14 hours, too slow to iterate, but fine for a 5-10 % subset. T0 does the same volume in minutes. (The arithmetic is illustrative, inference; the API-arm wall time will dominate anyway and is agent 09's domain.)

## 6. Contradictions with `docs/initial-plan.txt`

1. "~20 kHz ... 80 drones across 4 envs" is from the **legacy** README only, is an aggregate over 80 drones x 4 envs, sits beside an explicit "cherry pick" warning, and is absent from the current README. For the experiment's 1-drone case the relevant baseline is the ~5 kHz single-env figure (legacy README) and whatever agent 02 measures. The plan states it "reports" this without qualification; moderate misstatement.
2. "Apple Silicon/macOS 26 tested": **true for the current `main` README**, but the same README says pybullet has no prebuilt wheel beyond Python 3.10 and must be source-built with `CFLAGS="-Dfdopen=fdopen"`. PyPI confirms no macOS pybullet wheels exist at all. "Explicit Apple Silicon tested" does not mean "pip install works on Python 3.14.2".
3. "Current maintained version" — maintained (commits 2026-09-06) but **only release is v1.0.0 from 2021**; pin a SHA. The repo moved from `utiasDSL` to `learnsyslab`.
4. Plan's Webots description (GPS/IMU/gyro/compass/camera/wind) matches Webots' demo, but nothing verified here supports "wind" as a built-in drone effect; treat as custom.
5. "Killer demo on thousands of flights" is achievable only via the tiered design; thousands of flights on PyBullet alone is viable but slower (Section 5).

## 7. Sources (accessed 2026-09-30)
- gym-pybullet-drones: https://github.com/learnsyslab/gym-pybullet-drones (README `main`, `pyproject.toml`, commits/branches/releases via `gh api`); legacy README https://raw.githubusercontent.com/learnsyslab/gym-pybullet-drones/master/README.md and `.../paper/README.md`
- pybullet wheels: https://pypi.org/pypi/pybullet/json ; mujoco: https://pypi.org/pypi/mujoco/json ; numpy, scipy, torch, genesis-world, mlagents, rotorpy, gymnasium JSON via same pattern
- Webots: https://github.com/cyberbotics/webots ; AirSim: https://github.com/microsoft/AirSim , https://github.com/microsoft/AirSim/issues/5023 , https://docs.theairlab.org/0.20/docs/simulation/ms-airsim/ , https://www.theregister.com/2022/07/19/microsofts_metaverse_is_for_training ; Colosseum https://github.com/CodexLabsLLC/Colosseum
- Isaac Sim README: https://github.com/isaac-sim/IsaacSim ; Isaac Lab https://github.com/isaac-sim/IsaacLab ; Aerial Gym https://github.com/ntnu-arl/aerial_gym_simulator
- PX4 Gazebo: https://docs.px4.io/main/en/sim_gazebo_gz/ ; PX4 SIH: https://docs.px4.io/main/en/sim_sih/ ; https://github.com/PX4/PX4-Autopilot ; ArduPilot https://github.com/ArduPilot/ardupilot ; gz-sim https://github.com/gazebosim/gz-sim
- Flightmare https://github.com/uzh-rpg/flightmare ; CrazyS https://github.com/gsilano/CrazyS ; Genesis https://github.com/Genesis-Embodied-AI/Genesis ; RotorPy https://github.com/spencerfolk/rotorpy ; ML-Agents https://github.com/Unity-Technologies/ml-agents
- MJX / Apple: https://mujoco.readthedocs.io/en/stable/mjx.html , https://github.com/hello-robot/stretch_mujoco/issues/40 , https://github.com/jax-ml/jax/discussions/9847 , https://mujoco.readthedocs.io/en/latest/mjwarp/

Unverified/inference items are flagged inline; main low-confidence items: Webots fast-mode throughput, PX4 SIH on macOS, MuJoCo Menagerie quadrotor, RotorPy batched throughput, Genesis on M4.

## Decisions recommended
1. Adopt the **two-tier** architecture: T0 batched-NumPy kinematic sim for sweeps, T1 gym-pybullet-drones (DIRECT, DSLPID) for a 5-10 % validation subset and the demo. Pin gpd to a commit SHA on `learnsyslab/gym-pybullet-drones@main`.
2. Fallback T1 = custom MuJoCo quad (cp314 arm64 wheel verified, 110 k steps/s); cross-check dynamics with RotorPy. Trigger: Gate G1 below fails.
3. One `SimBackend` protocol + a serialisable `ScenarioSpec`; events are data injected by a backend-agnostic `EventInjector`, so both tiers see identical seeds (paired design).
4. Require T0 calibration (step/brake/wind responses) and the rank-preservation acceptance test (Spearman >= 0.9, knee within one latency step, kappa >= 0.6) before publishing any T0-only claim.
5. Explicitly reject: Isaac Sim/Lab, Aerial Gym (no macOS), AirSim/Colosseum/Flightmare/CrazyS (unmaintained/archived), Unity ML-Agents (Py <= 3.10.12). Defer Genesis (Py < 3.14) and Gazebo/PX4 (Linux-only path) to optional stretch.
6. Correct the plan's wording: drop the "20 kHz" claim from any deliverable; report measured steps/s for 1 drone instead.

## Open questions
1. Does `pybullet` build from source on macOS 27 / arm64 under CPython 3.14.2 (README workaround was written for macOS 26/older Python)? Owner: 02. If not, which Python (3.12/3.13) for the sim-worker process?
2. Measured 1-drone DIRECT steps/s for gpd on this M4 (legacy claim ~5 kHz)? Owner: 02.
3. Does PyBullet reproduce bit-identical trajectories across processes/restarts with the same seed? Owner: 02/08.
4. Is `mujoco.mj_multiRay` fast enough for a 16-64 ray lidar in a 100 Hz control loop (estimated >50 k rays/s; not measured)? Owner: 07.
5. Is RotorPy 250 steps/s an artifact of its default integrator? Can a fixed-step RK4 be enabled for 10x+? (Cross-check role only; low priority.)
6. Does PX4 SIH build and run on macOS 27/arm64? Only matters if a "real autopilot" claim is required.
7. Is the T0 -> T1 ranking preserved for the **slow-LLM** arm (seconds of latency), where T1's PID transient may matter less but braking distance dominates? Needs the validation run.
8. Does the user actually need a real-autopilot claim (PX4)? If yes, Tier 2 needs Linux hardware or a VM, which conflicts with the local/no-Docker constraint.

## Follow-up spikes (small, runnable experiments)
1. **Gate G1 (1-2 h):** in scratch: `uv venv --python 3.12`, `CFLAGS="-Dfdopen=fdopen" uv pip install pybullet --no-cache`, `uv pip install -e` gpd at SHA; run a 10-s `CtrlAviary` DIRECT hover with `DSLPIDControl`; accept if steps/s >= 2000 (1 drone, `pyb_freq=240`) and memory flat over 200 resets. Repeat on 3.14.2; record which versions work.
2. **Calibration spike (half day):** script the velocity-step/braking/wind responses in gpd, fit `tau`, `a_max`, compare with a T0 implementation using the model in Section 5; output fit residuals.
3. **MuJoCo fallback spike (half day):** write the quad MJCF + PID cascade, check hover RMS < 5 cm, braking distance vs gpd within 10 %, `mj_multiRay` 32 rays @100 Hz timing; keep `bench.py` as baseline (110 k steps/s).
4. **Rank-preservation pilot (1 day):** 200 paired scenarios x 3 latencies x 4 cheap arms (hover, PID-no-avoid, reactive-brake, FSM) on T0 and T1; compute Spearman/kappa/knee; decide whether T0 passes.
5. **RotorPy integrator check (1 h):** switch to a fixed-step integrator and rerun `rp.py`; compare step response with gpd `DSLPID` as an independent dynamics reference.
6. **Webots reality check (optional, 1 h):** run the stock Mavic world with `--mode=fast --no-rendering --batch`, measure real-time factor; only if a pretty demo video is wanted.
