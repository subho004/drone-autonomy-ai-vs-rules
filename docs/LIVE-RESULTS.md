# Live model results and findings

Real API calls to Jev and four other models, flown on identical certified scenarios in the simulator. Run date: 2026-10-18. Reproduce with `.venv/bin/python scripts/run_live.py` (config: [`configs/experiments/live.yaml`](../configs/experiments/live.yaml)).

## TL;DR

1. **Speed, not smartness, separated the models.** Every model chose an acceptable action 92–96% of the time. Without a reflex shield, collision rate rose in the same order as measured decision latency: Jev (601 ms) 43%, mercury-decide (611 ms) 45%, GPT-4o-mini (718 ms) 61%, Gemini 3.5 flash-lite (976 ms) 66%, GPT-6 Luna (1080 ms) 71%.
2. **Jev is the fastest and safest of the models tested on this network, but it is a tie with mercury-decide**, and the gap to GPT-4o-mini, Gemini and Luna is statistically significant (paired McNemar, Holm-corrected p = 0.04, 0.007, 0.0006).
3. **Nothing here out-reacts a plain reflex controller.** The tuned rule-based arms collide 9% of the time with a decision latency of about 0 ms; Jev collides 43% (difference +34 points, p < 0.0001). A decision model's value is choosing *what* to do, not reacting faster than a reflex.
4. **With a reflex shield on, the models are not distinguishable** (12–27% collisions, all corrected p ≥ 0.3, n = 56). The shield does the safety work, so a fast reflex layer matters more than which model sits above it.
5. **Cost is a non-issue:** 6,891 API calls cost $0.45 in total.

## Setup

| | |
|---|---|
| Scenarios | 60 paired scenarios (`live-range` suite, seeds 20000+), reaction budgets 0.4–4 s (log-uniform), every one certified avoidable by the zero-latency oracle |
| Event mix | obstacle static 40%, obstacle approaching 30%, crossing intruder 20%, battery critical 5%, GPS loss 5%. Headline numbers below use the 56 obstacle/intruder scenarios; battery and GPS are reported separately |
| Conditions | shield off and on; sequential decision cadence; each model at its own **measured** wall-clock latency (converted to simulator ticks, applied at `t + ceil(L/dt)`) |
| Models | `typesafe/jev-1.13` and `inception/mercury-decide:free` (OpenRouter Decisions endpoint); `gpt-4o-mini` (logprobs); `gpt-6-luna` (reasoning off, top-5 logprobs); `gemini-3.5-flash-lite` (thinking minimal) |
| References | tuned reflex arm, tuned rule-based FSM, privileged oracle |
| Prompt | one shared observation encoder (hybrid terse format with explicit battery, GPS, threat and goal flags) and one action glossary for every model |
| Machine / network | Apple M4, client in India (Cloudflare edges MAA/HYD), HTTP/2 persistent connections, 4 concurrent flights |
| Volume | 960 flights, 6,891 API calls, 0 failed flights, 0–1.7% invalid outputs |

## Measured decision latency

Per-call, all decisions in the run (request sent to parsed action):

| model | calls | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| Jev 1.13 | 1,648 | 510 ms | 608 | 725 | 1,185 | 3,003 |
| mercury-decide | 1,604 | 504 ms | 740 | 1,030 | 1,714 | 3,470 |
| GPT-4o-mini | 1,252 | 640 ms | 874 | 1,069 | 1,429 | 3,830 |
| Gemini 3.5 flash-lite | 955 | 782 ms | 1,266 | 1,344 | 1,619 | 2,279 |
| GPT-6 Luna | 832 | 952 ms | 1,348 | 1,543 | 2,011 | 3,505 |

Jev's published figure is about 200–330 ms p50 from the US; from this location it is about 510 ms. The tail matters more than the median: a 1.2 s p99 is a long time for a drone moving at 2 m/s.

## Results

Collision rate on obstacle and intruder scenarios (n = 56 per arm and condition; a collision also counts a ground crash):

| arm | mean latency | shield off: collisions | shield off: success | shield on: collisions | shield on: success | acceptable actions | first evasive action after the event |
|---|---|---|---|---|---|---|---|
| Jev 1.13 | 601 ms | **43%** | 57% | 18% | 82% | 95% | 1.03 s |
| mercury-decide | 611 ms | 45% | 55% | 12% | 88% | 96% | 1.05 s |
| GPT-4o-mini | 718 ms | 61% | 39% | 16% | 84% | 92% | 1.50 s |
| Gemini 3.5 flash-lite | 976 ms | 66% | 34% | 27% | 73% | 93% | 1.67 s |
| GPT-6 Luna | 1080 ms | 71% | 29% | 12% | 88% | 95% | 1.67 s |
| tuned reflex | about 0 ms | 9% | 91% | 11% | 89% | 97% | |
| tuned FSM | about 0 ms | 9% | 91% | 9% | 91% | 96% | |
| oracle (privileged) | about 0 ms | 0% | 100% | 0% | 100% | 100% | |

"Acceptable actions" is the share of decisions in the oracle labeller's acceptable set (`label_v1`). "First evasive action" is the mean sim time from the obstacle appearing to the first applied non-FORWARD action.

### Paired significance (same scenarios, exact McNemar, Holm-corrected)

Shield off, measured latency, Jev against each arm:

| comparison | Jev | other | difference | discordant (Jev-only / other-only) | p (Holm) |
|---|---|---|---|---|---|
| Jev vs mercury | 43% | 45% | −2 pts [−14, +11] | 6 / 7 | 1.0 |
| Jev vs GPT-4o-mini | 43% | 61% | −18 pts [−31, −5] | 3 / 13 | 0.043 |
| Jev vs Gemini flash-lite | 43% | 66% | −23 pts [−36, −10] | 2 / 15 | 0.007 |
| Jev vs GPT-6 Luna | 43% | 71% | −29 pts [−41, −16] | 1 / 17 | 0.0006 |
| Jev vs reflex | 43% | 9% | **+34 pts** [+22, +46] | 19 / 0 | < 0.0001 |
| Jev vs FSM | 43% | 9% | **+34 pts** [+22, +46] | 19 / 0 | < 0.0001 |

With the shield on, no Jev comparison reaches significance (Jev vs FSM +9 pts, corrected p = 0.38; all others p ≥ 1.0 after correction).

### By difficulty tier (reaction budget), shield off

Budget = how much time a decision layer has beyond the physical minimum before avoidance becomes impossible.

| tier (budget) | n | Jev | mercury | GPT-4o-mini | Gemini | Luna | reflex |
|---|---|---|---|---|---|---|---|
| T0 (≥ 2 s) | 15 | 27% | 20% | 33% | 33% | 33% | 7% |
| T1 (1–2 s) | 19 | **16%** | 26% | 53% | 58% | 68% | 0% |
| T2 (0.5–1 s) | 18 | 72% | 72% | 83% | 94% | 100% | 17% |
| T3 (0.25–0.5 s) | 4 | 100% | 100% | 100% | 100% | 100% | 25% |

This is where the models separate. With 1–2 s of slack, a 0.5 s decision layer mostly makes it and a 1 s one often doesn't. Below 1 s of slack nobody can cope: the effective sense-to-act delay is about 1.5× the call latency, so a 0.5 s model reacts in roughly 0.75–1 s. The T0 failures (20–33% even with 2+ s to spare) are not latency failures; see "Failure modes" below.

### By event type, collision rate

| event | n | Jev | Jev + shield | Luna | Luna + shield | reflex |
|---|---|---|---|---|---|---|
| obstacle static | 24 | 33% | 8% | 67% | **0%** | 0% |
| obstacle approaching | 22 | 27% | 14% | 64% | 9% | 0% |
| crossing intruder | 10 | **100%** | 50% | 100% | 50% | 50% |
| battery critical | 4 | 0% (handled) | 0% | 0% | 0% | 0% |

Static obstacles are where a shield helps most: a model that is too slow still stops in time. Crossing intruders defeat every decision model without a shield, and the shield only halves it because it brakes forward motion, not the intruder's lateral approach. Battery-critical events are easy at these deadlines: every arm lands in time.

## Cost and volume

| model | calls | cost |
|---|---|---|
| Jev 1.13 | 1,768 | $0.102 |
| GPT-4o-mini | 1,372 | $0.171 |
| Gemini 3.5 flash-lite | 1,075 | $0.095 (assumed price) |
| GPT-6 Luna | 952 | $0.080 |
| mercury-decide (free tier) | 1,724 | $0.000 |
| **total** | **6,891** | **$0.448** |

Decisions per flight (shield off): Jev 11.2, mercury 11.4, GPT-4o-mini 8.3, Gemini 5.7, Luna 4.8. A slower model makes fewer decisions during the same flight, which is itself a cost of latency.

## Findings

1. **Latency dominates in this regime.** The five models are about equally good at picking an acceptable action, yet their collision rates span 43–71% in the order of their measured latency. Sub-second decision models are physically limited by braking distance, not by judgement.
2. **The reaction budget explains the curve.** A collision is largely determined by whether the effective latency exceeds the scenario's budget. This is why the tier table has a sharp edge at about 1 s, and why a single aggregate collision rate is misleading without the budget distribution.
3. **A fast reflex layer beats a slower decision layer on safety; the decision layer's job is semantic.** Reflex and FSM arms avoided 91% of scenarios with essentially zero latency. The practical architecture is a reflex shield for safety plus a decision model for choosing maneuvers and handling situations a rule cannot.
4. **Choice of model matters much less once a shield is on.** With the shield, even the slowest model reaches 12% collisions on par with Jev. Spend effort on the shield's design first.
5. **Jev's advantage is real but modest and mostly latency.** It is significantly better than the three slower models, indistinguishable from the free mercury-decide, and far behind a reflex arm.
6. **Tails differ even where medians match.** Mercury's p95 (1.03 s) is 40% above Jev's (0.73 s) at the same median, and its p99 is 1.7 s against 1.2 s. At n = 56 this did not show up as a collision difference.
7. **A noisy policy can out-perform a perfect one at high latency.** In the injected-latency sweep (see `reports/accuracy_x_latency.html`), a deliberately error-prone oracle collides less than the perfect one above about 200 ms, because stale precise dodges fail where braking is robust. This is consistent with the staleness explanation, not a model quality claim.

## Failure modes seen

- **Crossing intruders** account for 10 of the 22 model collisions at generous budgets (T0, no shield) although they are only 10 of the 75 flights there: every model collided on every one. The intruder's lateral approach is sensed through a forward corridor and a derivative filter, which lags.
- **Stale decisions:** a model that decided about an obstacle 1 s ago acts on a world that has moved on. The harness applies results at `t + L` rather than discarding them, so stale dodges can steer into the obstacle's new position.
- **Invalid outputs** were rare: GPT-6 Luna 1.7% (shield off), Jev 0.8% (shield off), everything else at most 0.3%. The harness falls back to holding the last action and then HOVER.

## Caveats (read before quoting)

- **Network-specific.** Latencies are from India. From the US the same models would be faster, shrinking the gaps. The conclusion "latency dominates" is robust; the specific collision rates are not portable.
- **Small n.** 56 scenarios per cell: differences smaller than about 15 points are not resolved; per-tier and per-event cells (n = 4–24) are indicative only.
- **Single run.** One repetition per scenario; non-determinism in the APIs (no seeds, documented for Jev) means repeats would differ.
- **Latency is injected, not real-time.** The simulator advances in ticks and a model's measured wall-clock time is converted to ticks. A real-time validation set (about 300 flights at 1× speed) is not part of this run, so agreement between injected and real-time accounting is not verified.
- **Scenario distribution matters.** Collision rate equals roughly the share of scenarios whose budget is shorter than the effective latency. These scenarios were drawn from a 0.4–4 s budget band; a different band shifts every number.
- **Simulation only.** A 27 g Crazyflie model in PyBullet. No claim about real drones.
- **The prompt's `age` field was always 0.** The harness never passed the data's staleness into the observation, so models were not told how old their data was (the system prompt describes `age` as that staleness). Latency itself is applied by the harness, so the curves are unaffected, but these are "uninformed age" runs; whether telling the model helps is an open experiment (see [SIMULATION-IDEAS.md](SIMULATION-IDEAS.md), idea B1).
- **Prompt and state encoding are a factor.** One shared encoding was used; no prompt-format ablation was run. Jev is documented to be weak at raw numbers, so the encoding supplies pre-computed buckets and flags; a different encoding could change its accuracy.
- **Pricing:** Gemini's price is an assumption; mercury-decide was on a free tier.
- **Jev is an alpha endpoint** (`POST /api/alpha/decisions` on OpenRouter); behaviour and limits may change.

## Reproduce and inspect

```bash
.venv/bin/python scripts/run_live.py        # about 21 min on this setup, about $0.45
.venv/bin/python scripts/make_report.py     # reports/live.html (static report with paired tests)
.venv/bin/uvicorn main:app --port 8765      # dashboard: choose experiment "live"
```

Raw data lives in `data/drone_jev.db` (flights, per-decision latencies, labels, costs) and `data/trajectories/` (Parquet, replayable in the dashboard). Related reading: [`MASTER-PLAN.md`](MASTER-PLAN.md), [`research/16c-live-probe.md`](research/16c-live-probe.md) (the earlier 8-model, 24-call probe).
