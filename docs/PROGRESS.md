# Implementation progress

Plan: `docs/MASTER-PLAN.md`. Work happens on `main`, one commit per milestone, nothing pushed.
If a usage limit interrupts work: wait for the reset, read this file, continue at the first unchecked item.

## Milestones
- [x] M0 Foundations: requirements split, Docker removed, `scripts/setup_sim_env.sh`, domain types, SQLite schema/store, settings
- [x] M1 Vertical slice: PyBullet world, executor, shield, sensors, latency injection, flight runner
- [x] M2 Scenarios: budget-first generator, oracle calibration, certification, labeller
- [x] M3 Classical arms + first collision-vs-latency curve (pilot: 6 arms x 8 latencies x shield off/on, ~13k flights)
- [x] M4 Providers + live run (8 arms, 960 flights, 6,891 API calls, $0.45; Jev p50 ~600 ms from India)
- [x] M5 API + dashboard (HTML) + replay + report (dashboard verified visually against real data)
- [x] M6 Visual verification, fixes, README, reports

## Log
- 2026-10-17 M0: venv on Python 3.14.2, pybullet 3.2.7 built with `CFLAGS=-Dfdopen=fdopen`, gym-pybullet-drones 2.2.0 @ 7ebad1e.
- 2026-10-17 M1/M2: sim core (~6k ticks/s), carrot-based ActionExecutor, forward-corridor shield, 5 classical arms + oracle(+error), latency policies (lockstep/fixed/sampled/measured), SQLite worker/queue/supervisor, calibration (static d_min0=1.14 m, approach 1.6-2.1 m, cross 1.07-1.2 m), generator + oracle certification. 70+ unit tests.
- Fixed along the way: GPS freeze captured at first observe instead of loss time; deadline events counted pre-event decisions applied after onset; shield cancelled in-progress sidesteps (stall) -> now only stops forward motion; shield looked at whole +-45 deg arc -> now a forward corridor.
- Providers (`app/providers`) and stats/report (`app/analysis`) built by sub-agents with 67 and 34 tests; live smoke of Jev/gpt-4o-mini/Gemini OK.
- Pilot experiment (classical arms, 8-point latency sweep, shield off/on, ~13k flights) running -> `data/drone_jev.db`.
- 2026-10-18 Pilot analysis: oracle collision rate 0/1/10/42/54/68/89/99% at 0/50/100/200/300/500/1000/2000 ms (knee 100-200 ms). Shield flattens the curve.
- Bugs found by running real data (all fixed + regression tests): carrot setpoint dragged down by a sagging drone (ratchet into the ground); shield cancelling in-progress sidesteps; fast repeated dodges (2.5 m/s lateral) destabilising the stock PID in wind (crashes 40/40 -> 0/40 at <=1.5 m/s; recalibrated + retuned); oracle_err errors were per-decision not per-time; dashboard log axis broke on 0 ms; headline curve mixed in impossible controls (added scope selector).
- Equal-budget tuning of the rule-based arms (60 random trials each, held-out `tune` suite, seeds 50000+): both converge to ~0.41-0.42 objective.
- Live run (2026-10-18): 0 failed flights. Without shield collisions track measured latency (Jev 43%, mercury 45%, 4o-mini 61%, Gemini 66%, Luna 71%) while accuracy is ~equal (92-96% acceptable); reflex/FSM 9%. With shield all models 12-27%, no significant differences.
- Reports: `reports/{pilot,pilot-controls,accuracy_x_latency,live}.html` via `scripts/make_report.py`.
- 2026-10-18 Docs: `docs/LIVE-RESULTS.md` (live model findings), open-source README (badges, demo GIF/MP4, screenshots, mermaid diagram), LICENSE (MIT), CONTRIBUTING. Media regenerated with `scripts/capture_docs_media.py` (Playwright + ffmpeg).
- 2026-10-18 Mission mode: stations/pad tracker, several obstacles per world, mission prompt block, mission-only executor hold semantics, rule-based mission pilot (certifies missions), audition of 4 live models (12 missions x shield on/off, $0.28), 3D renderer (`app/render/`, PyBullet + Pillow HUD) and `scripts/render_mission.py`. Winner by pre-declared rule: Mercury Decide + shield (5/12), Jev + shield 4/12 (tie). Write-up: `docs/MISSION-DEMO.md`.
- Bugs found while building missions (all fixed + tests): HERE phase wider than the scan tolerance (drone stopped just outside and held forever); 1.0 m sidestep vs a 0.9 m station window (endless ping-pong); HOVER cancelled an in-progress sidestep (mission-only option now keeps it); pad had no braking zone; mission obstacle spawned on the pad.
- 2026-10-18 Group A simulations (all ten, see `docs/SIMULATIONS.md`): task trackers (`app/sim/tasks.py`: follow target, no-fly zones, landing sites), a steering (proportional-navigation) intruder, static walls and a gust burst in the world, gate mode in the mission tracker, rule-based pilots (`app/controllers/sim_arms.py`), generators and certification per family (`app/scenarios/sims.py`), prompt blocks, sampled (human-like) latency cells, a predictive wrapper, a generic renderer (`app/render/sim_scene.py`, `sim_video.py`), sweeps (`scripts/run_sims.py`, 7 experiments), human and ghost experiments, live runs on four tasks ($0.72), side-by-side videos and screenshots under `assets/sims/<slug>/`. A7 (two drones) and A10 (camera + VLM) were built by sub-agents and checked here (tests, lint, keys-not-in-repo scan, raw-data spot checks).
- Result: 4 of 11 predictions held, 5 falsified, 2 in part. Bugs found by looking at the output (all fixed): delivery success ignored zone breaches (zone-blind pilot "succeeded" 24/24; now requires < 2 cm breach); the gate video said ALL GATES PASSED on a flight that missed three; the outcome banner appeared too early on short flights; `flight-near-miss.mp4` was a collision (renamed); pairing videos across suites needed seeds, not scenario ids.
- Honest limits: the canyon gust control is too easy (peak deviation 0.19 m vs 1.0 m margin); the "human" is the FSM rule with a synthetic reaction time; thresholds and geometry were tuned until the zero-latency pilot succeeded, so latency knees are specific to them.
- 2026-10-18 Orb Dodge game (docs/sims/orbs.md): homing orbs with a radar sense, external orb brains with sim-time latency (rules or gpt-4o-mini), reflex dodge layer, rival-chaser skin, 3 videos. Part B live spend $0.07. Findings: 0 ms pilot 24/24 vs 0/24 at 300 ms; small-model orbs weaker than rule pursuit; n=8 live runs cannot separate the fliers.
- 2026-10-18 Orb Dodge extensions: six-orb swarms (draw_swarm, certified separately), role matrix (rule or model as chaser and dodger), rule vs Mercury vs Jev side-by-sides, rival drone swarm skin. Models failed every swarm (0/8); rule pilot 24/24 only by certification.
- 2026-10-18 Orb Dodge tournament: randomised round robin (4 dodgers x 4 chasers x 12 scenarios, $0.17), subfolders under assets/sims/orbs/. Rule pilot 24/48 survive, Mercury 7/48, Jev 4/48; rule chasers beat gpt-4o-mini as chaser.
- 2026-10-18 Docs: docs/FINDINGS.md (all findings, rule vs LLM vs oracle per task, AI vs rules vs ML comment) and docs/ARCHITECTURE.md (module guide with 14 Mermaid diagrams), linked from the README.
