# Simulation ideas: what to build next

Two studies back this document: a prior-art search ([17a](research/17a-simulation-ideas-prior-art.md): 20 ideas from the literature) and a feasibility study against this codebase ([17b](research/17b-simulation-ideas-feasibility.md): 17 ideas with measured spikes). This is the merged, ranked version. Dates and sources are in those files; web evidence there is mostly abstract- or snippet-level and is labelled as such.

## TL;DR

- **Best new videos:** a *follow-me* chase of a weaving target, a *delivery run* that threads restricted zones, an *intruder that steers into you*, an *engine-out landing-site choice*, and *human vs Jev vs reflex* on the same scenario.
- **Best new science:** tell the model its own latency (and measure whether that helps), sweep decision cadence, show that latency *tails* matter more than the mean, and test whether extrapolating the world to "where it will be when the answer lands" removes the latency penalty.
- **A real gap in the literature:** the search found no benchmark that injects decision latency into a physically simulated drone and sweeps it (the closest, DroneCATS, blames latency for lost moving targets but never sweeps it). That is this project's niche; the time-model trick itself is not new (ReflexBench, Bench2Drive-Robust and RealtimeGym use it). "To the best of a finite search" applies.
- **A flaw to fix first:** `decision_age_ms` was never wired in, so every live prompt said `age=0`. See below.

## A flaw found along the way

`flight_runner._issue` calls `world.observe(...)` without `decision_age_ms`, so the encoder always printed `age=0`, while the system prompt describes `age` as milliseconds since the data was measured. This does **not** change the injected-latency curves (latency is applied by the harness), but all live results (benchmark, mission audition) are "models were not told how stale their data is" runs. It is also the control condition for idea 1 below. Fix and re-run, labelling the old runs as "uninformed age".

## Facts measured for this study

These constrain every design below (spikes in 17b, run on this machine):

| Fact | Number |
|---|---|
| Lateral agility limit of the stock PID | commanded lateral accel above about **3 m/s²** crashes it; forward speeds of 2, 3 and 4 m/s were fine |
| Tracking a moving target | steady-state error equals **target speed × latency** within about 5% (the controller adds almost nothing) |
| Multi-drone (gym-pybullet-drones) | works headless at about 7.4k drone-ticks/s, contact detected; the downwash mode (`PYB_DW`) crashed a drone in every run, so use plain `PYB` |
| Yaw | a yaw-capable controller works (up to 360°/s), but rays, bearings and the oracle assume world axes, so it is a separate 4–5 day project |
| Camera frames | 5 ms at 224×224, 37 ms at 640×480 (an image prompt costs more than a text one) |
| State save/restore for rollouts | a 1 s physics rollout is about 7 ms, but restoring executor and PID state is untested |

## The ideas

Effort is solo-developer days on top of the existing runner, executor and shield. "Prediction" is a falsifiable claim, stated before running.

### Group A: new simulations that make strong videos

| # | Idea | What the video shows | Question and prediction | Days | First experiment |
|---|---|---|---|---|---|
| A1 | **Follow-me: moving-target following** | The drone trailing a weaving target, run at 0, 300 and 1000 ms, with the lag distance drawn as a line | How does tracking error scale with latency? Prediction: error ≥ 1.5 × v × L plus a floor from the 1 m sidestep quantisation; lost-lock rate crosses 50% when v × L is about half the allowed range band | 5 | A scripted tracker vs a 1 m/s target at 0 to 1 s latency; compare to the v × L line (2 days) |
| A2 | **Delivery with restricted zones** | A drone threading red no-fly zones to a pad; a breach meter; with and without a geofence layer | Graded outcome (breach depth in metres) has far more statistical power than binary collisions. Prediction: breach depth rises about linearly with latency, slope near cruise speed | 4 | One zone across the straight path, breach depth vs latency over 30 seeds |
| A3 | **Intruder that steers into you** (proportional navigation) | A red sphere curving into the drone's path; the dodge, then the intruder correcting | Is the knee latency independent of how adaptive the intruder is? Prediction: success collapses at least 2× earlier than for the existing constant-velocity intruder | 3 | Oracle and FSM at 0 to 800 ms on 30 seeds vs the existing approacher |
| A4 | **Engine-out: landing-site choice** | Battery bar falls, three patches (safe/hazard), the drone commits to one | Where does smartness beat speed? Prediction: a crossover latency L* (a few hundred ms to about 1.5 s) above which "nearest site" beats "best site" | 3.5 | Two scripted policies across latencies, no LLM, find L* |
| A5 | **Human vs Jev vs reflex** | Picture-in-picture of a person, Jev and a reflex arm on the same scenario | Where does a human sit on the curve? Prediction: a 250 ms "human" running the FSM rule collides between the 0 ms FSM (9%) and Jev at 600 ms (43%), nearer the FSM | 0.5 (synthetic), +4.5 (playable) | A synthetic human latency trace on the FSM rule, 56 scenarios, no API |
| A6 | **Ghost drone: predictive decisions** | A translucent ghost drone and obstacle drawn where the model *thinks* things will be when its answer lands | Can the latency penalty be engineered away? Prediction: extrapolating by the expected latency recovers at least 50% of the collision gap for constant-velocity hazards, under 25% against a steering intruder | 4 | Feed `range − closing × L` to the reflex arm; pair with idea B1 |
| A7 | **Two-drone airspace** (flagship) | Two coloured drones, one Jev and one reflex or another model, negotiating a crossing | Does risk depend on the *sum* of latencies? Prediction: (L, 0) and (L/2, L/2) give equal collision rates; equal-latency symmetric drones deadlock (both pick the same side) in at least 10% of head-on cases at 400 ms or more | 7 | A 100-line standalone script sweeping (L_A, L_B) on a 5×5 grid, no harness changes |
| A8 | **Gate slalom** | The drone threading rings, misses glowing orange | Pass rate vs latency at fixed speed. Prediction: pass rate falls from about 100% to about 0% between roughly 0.2 and 0.8 s (for 1.2 m/s, ±0.5 m rings). This is "slalom at walking pace", not racing | 3 | Existing mission pilot with zero-dwell stations at 0 to 0.8 s |
| A9 | **Urban canyon with gusts** | A drone shoved toward a wall by a gust | A deliberate *negative control*: prediction that latency does **not** matter for pure gust rejection (the PID handles it) but does for branching corridors | 4 | One wall plus a 1.5 m/s² gust, FSM at four latencies; confirm flatness first |
| A10 | **Structure inspection with a camera + VLM** | Picture-in-picture of what the drone sees, defect list filling in, cost ticker | Is pixel resolution a latency tax? Prediction: accuracy saturates near 128×128 while latency keeps rising with image size | 6 | Not a flight: 100 stills at three resolutions through three VLMs (about $0.20, 1 day) |

### Group B: cheap experiments that sharpen the existing results

| # | Idea | Question and prediction | Days |
|---|---|---|---|
| B1 | **Latency-aware prompting** (and fix `age`) | Does telling the model its latency help? Prediction: for the slow models, giving the *derived* number (`ttc_at_apply = ttc − age`) cuts shield-off collisions by at least 10 points, while just stating the age changes nothing | 1 |
| B2 | **Cadence sweep (pipelined decisions)** | How much does issuing a request every Δ ticks help? Prediction: reaction time improves by about 0.5 × (L − Δ), flattening below Δ of about 100 ms. Risk: AVOID flip-flops may destabilise the PID | 2 |
| B3 | **Latency tails at equal mean** | Is p50 the wrong headline? Prediction: at equal mean, collisions rise with p99; a heavy-tail trace is at least 10 points worse than a constant one | 1.5 |
| B4 | **Zero-latency competence suite** | Separates "instruction following" from latency: run every model at injected 0 ms. Explains the audition's finding that two models failed on the rules, not on speed | 1–2 |
| B5 | **Shield-only floor and override rate** | How much of the score is the shield? A shield-only arm and the shield's override rate beside collisions | 1–2 |
| B6 | **Mission-length survival analysis** | Is per-decision error constant or drifting? Rescues power from the n = 12 audition using the per-decision outcomes already stored | 0.5–1 |
| B7 | **Phantom obstacles and false-alarm cost** | Safe versus merely stopped. Prediction: reacting to every phantom costs more as latency rises, because the model cannot revisit its decision | 1–3.5 |

### Group C: stretch

| # | Idea | Note |
|---|---|---|
| C1 | **Hybrid three tiers** (slow planner + fast model + reflex) | The most open-ended; start with a rule-based planner at 3 s feeding the mission pilot (no API). Prediction: the reflex tier carries collisions, the planner carries mission completion (at least 2× more missions than fast + reflex alone) |
| C2 | **Convoy / formation** | Needs A7; run a continuous-control control group first, because 11-action quantisation can mimic string-instability |
| C3 | **Area coverage with a limited battery** | Mostly a negative result unless drain is raised about 10×; ranked last |
| C4 | **Hidden-target search with memory** | Needs a new sensor; likely a project of its own |

## Recommended build order

About 24.5 working days for the first five, in this order (from 17b, adjusted by 17a's priorities):

| Order | Package | Ideas | Days | Why now |
|---|---|---|---|---|
| 1 | Prompt-honesty and cadence pack | B1 + B2 + B3 | 4.5 | Fixes a real flaw in the data and produces three falsifiable results, mostly with no API cost |
| 2 | Moving things | A3 + A1 | 8 | Needs a generalised moving-entity type; the follow-me chase is the best single video, and the pursuer is cheap once entities exist |
| 3 | Restricted zones | A2 | 4 | Graded metric, no new physics, strong visual, extends missions |
| 4 | Predictive decisions | A6 | 4 | Tests whether latency can be mitigated rather than only measured; pairs with A3 |
| 5 | Human and landing-site baselines | A5 + A4 | 4 | Tells the speed vs smartness vs human story |

Stretch afterwards: C1 (hybrid tiers, 6 days) and A7 (two drones, 7 days; do the standalone heat-map script first). Do B4 to B6 alongside: they are cheap and fix this project's main statistical weaknesses.

## Design rules learned

1. Keep the 11-action contract. Handle follow, zones and landing within it; only fork an action set if the follow-me spike shows quantisation dominating.
2. Cap commanded lateral acceleration at 3 m/s² in every new motion generator.
3. Prefer scenarios with a graded outcome metric (distance, depth, fraction of time) over binary collisions.
4. Do not add yaw casually; it is its own 4–5 day project plus relabelling.
5. Build one generalised moving-entity type first, so zones, targets, buildings and drones share a mechanism.
6. Run multi-drone work in plain `PYB`, never `PYB_DW`, until the downwash crash is explained.

## Claim discipline

The time model (decoupling the simulator from inference and applying a default or previous action while waiting) follows ReflexBench, Bench2Drive-Robust and RealtimeGym. Claim only the combination of a physically simulated drone, a reaction-budget parametrisation and typed-decision models as new, and only as "to the best of a finite search".
