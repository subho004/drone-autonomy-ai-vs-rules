# Mission demo: the best decision model flies a real task

The benchmark measures *how fast* a decision layer has to be. This demo asks a different question: given a multi-step task, which decision model should you actually put in the loop, and what does its flight look like? Four live models audition on the same certified missions; the winner (chosen by a rule fixed beforehand) is rendered flying one in 3D.

Everything here is real: the models run live against their APIs, the drone is simulated in PyBullet, and the video is a replay of a recorded flight drawn with PyBullet's own renderer.

## The mission

Each mission is a short delivery-style task:

1. **Scan three stations**, in order. A station is a ring at a given lateral offset and altitude. To scan it the drone has to line up with it, then hold still inside the ring for 1 s.
2. **Dodge three hazards** that appear along the way: a static block, an object drifting toward the drone, and a crossing intruder just before the pad.
3. **Land on the pad** at the end.

The model never touches the motors. Every 0.5 s or so it picks one of 11 discrete actions (`FORWARD`, `LEFT`, `RIGHT`, `ASCEND`, `DESCEND`, `BRAKE`, `HOVER`, `AVOID_LEFT`, `AVOID_RIGHT`, `RETURN_HOME`, `EMERGENCY_LAND`) and the same PID controller as in the benchmark does the flying. A mission counts as a success only if **all three stations are scanned and the drone lands on the pad without a collision**.

What the model is told, each decision: its sensor state (a 16-ray range ring, speed, battery) plus a compact mission block, for example `station 2/3 phase=FAR moving=NO`, `target: dx=3.4 lat=LEFT alt=DOWN speed=0.8`, and a short rule list (align with LEFT/RIGHT/ASCEND/DESCEND, brake when the station is near, hold to scan, land at the pad). The mission geometry is pre-digested into buckets, because these models are weak at raw numbers. So this tests **instruction following under latency**, not open-ended planning.

## How the winner was chosen

12 missions (seeds 3000+), identical for every model, each **certified feasible** by flying it with a rule-based mission pilot at zero latency. Each model flew each mission with the reflex shield off and on, at its own measured latency. Total cost: $0.28.

The selection rule was fixed before the results: highest mission success rate, then most stations scanned, then fewest collisions.

| arm | shield | missions completed | stations scanned (mean of 3) | collisions | mean decision latency |
|---|---|---|---|---|---|
| rule-based pilot (reference) | off | 12 / 12 | 3.00 | 0% | about 0 ms |
| rule-based pilot (reference) | on | 9 / 12 | 2.83 | 25% | about 0 ms |
| **Mercury Decide** | **on** | **5 / 12** | **2.42** | 58% | 548 ms |
| Jev 1.13 | on | 4 / 12 | 2.00 | 42% | 554 ms |
| Mercury Decide | off | 1 / 12 | 1.92 | 92% | 580 ms |
| Jev 1.13 | off | 1 / 12 | 1.83 | 58% | 551 ms |
| Gemini 3.5 Flash-Lite | on | 0 / 12 | 0.42 | 58% | 1007 ms |
| Gemini 3.5 Flash-Lite | off | 0 / 12 | 0.25 | 83% | 1060 ms |
| GPT-4o-mini | on | 0 / 12 | 0.00 | 100% | 673 ms |
| GPT-4o-mini | off | 0 / 12 | 0.00 | 100% | 707 ms |

The rule-based pilot's 12 / 12 is by construction: the missions were certified with it, so it is the ceiling, not a competitor. (With the shield on it drops to 9 / 12; I did not investigate why, but a plausible cause is the shield braking while the pilot is deliberately approaching a station.)

### Findings

1. **Mercury Decide with the shield won (5 of 12), with Jev a statistical tie (4 of 12).** They disagreed on only 5 missions (mercury alone completed 3, Jev alone completed 2; exact McNemar p = 1.0). Mercury scanned all three stations in 8 of 12 missions, Jev in 6 of 12. With 12 missions this cannot separate them.
2. **Both live leaders trailed the rule-based pilot.** The best live model completed 5 missions to the pilot's 12. A mission needs about 60 consecutive sensible decisions at roughly half-second spacing, and a single bad one (overshooting a station, or a late dodge) usually ends it.
3. **The shield mattered, but at n = 12 it is only suggestive.** Jev completed 4 missions with it and 1 without (3 gained, none lost, exact McNemar p = 0.25); mercury 5 against 1 (5 gained, 1 lost, p = 0.22). Collisions are still common even with the shield, because it only brakes forward motion and cannot stop a crossing intruder.
4. **GPT-4o-mini never completed a station.** It issued no `HOVER` at all (its mix is 33% `FORWARD`, 32% `AVOID_LEFT`, 17% each `LEFT` and `RIGHT`), so it never stopped to scan, and all 12 flights ended at x of about 22.5 m, past the pad and out of the arena (recorded as a crash). It over-applies avoid and sidestep actions and ignores the "hold to scan" rule. Gemini 3.5 Flash-Lite scanned a few stations (0.42 on average) but was the slowest at about 1.0 s per decision and mostly collided or left the arena.
5. **Rule-following, not speed, is what separates these models here.** Jev and mercury have the same latency (about 550 ms) and similar action mixes (about 45% `FORWARD`, 25% `HOVER`, the rest alignment steps). The two other models fail on the rules themselves: one never holds still to scan, the other is slow and collides.

## The rendered flights

`scripts/render_mission.py` picks, for the winning (model, shield) cell, its **median-duration successful run**, not its best one, and renders it. It does the same for Jev, since this project is about Jev and it is a near-tie.

| flight | model | outcome | media |
|---|---|---|---|
| winner | Mercury Decide + shield | all 3 stations scanned, landed on pad | `assets/mission/winner/flight.mp4`, `highlight.gif`, 8 screenshots |
| Jev | Jev 1.13 + shield | all 3 stations scanned, landed on pad | `assets/mission/jev/flight.mp4`, 8 screenshots |

The HUD shows the model's current action, a bar while a request is in flight (so you can see the drone coasting on its previous decision while the model "thinks"), the mission checklist, event toasts, and a top view with the trail.

What the images do and don't show:

- It is a **replay** of a recorded flight rendered offline, not a live feed.
- The drone is drawn **4× larger** than real so it is visible, and stations are shown as flat discs on poles.
- Hazards appear at a fixed offset ahead of the drone, so the same mission looks the same for every model; the camera is a smoothed chase camera.

## Caveats

- **12 missions per cell.** Differences under about 25 points are not resolved; the Jev/mercury ordering is a coin flip.
- **Missions are certified by the rule-based pilot**, so they are biased toward ones a rule-based pilot can finish; the models face the same set.
- **The prompt carries the mission rules.** A model with a different prompt, more examples, or reasoning enabled could do differently; the encoder and rules are one fixed design, not tuned per model.
- **Latency is each model's measured wall-clock time on this machine and network (India)**, converted to simulator ticks. From elsewhere the models are faster and the gaps shift.
- **The prompt's `age` field was always 0** (the staleness of the data was not passed in), so models were not told how old their data was; see [SIMULATION-IDEAS.md](SIMULATION-IDEAS.md).
- Mercury Decide ran on a free tier; Gemini's price is an assumption.
- Simulation only (a 27 g Crazyflie model); nothing here says anything about real hardware.

## Reproduce

```bash
.venv/bin/python scripts/run_missions.py     # audition: about 11 min, about $0.28 with the default keys
.venv/bin/python scripts/render_mission.py   # pick the winner, render video + screenshots (about 3 min)
```

The audition lives in `data/drone_jev.db` as experiment `mission_audition` (viewable in the dashboard); the selection and every audition row are in `assets/mission/selection.json`.
