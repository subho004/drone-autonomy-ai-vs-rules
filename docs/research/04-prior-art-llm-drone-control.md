# 04 - Prior art: LLM/VLM-driven drone control and latency-hybrid architectures

Agent 04. Access date for all web sources: 2026-09-30. Method: WebSearch + WebFetch of arXiv abstracts/HTML, GitHub READMEs, project pages. Caveat: WebFetch returns a small-model summary of each page, not raw text; numbers below are as reported by those summaries and were cross-checked against search snippets where possible. Items marked **[bg]** come from background knowledge and were not re-fetched (lower confidence). Confidence tags: H/M/L.

## 0. Headline findings (read this first)

1. **Jev is real and the exact experiment is partly done already.** (Contradicts the brief's "unverified".) Jev is TypeSafe's hosted "System One" decision model: state + up to 3 typed questions in, typed answers with probabilities out, non-autoregressive/parallel, ~100 ms typical (70-500 ms), $0.042/M input tokens, output free ([flaviocopes.com/jev](https://flaviocopes.com/jev/), H for the vendor-described claims). Agent 01 owns identity; I report only what bears on prior art. A UTS Sydney paper (arXiv 2609.22753) calls it "Jev-1.13.0 through OpenRouter's Decisions endpoint" (M-H).
2. **`RomanSlack/jev-drone` (MIT) is the closest prior art**: MuJoCo Skydio X2, camera-only, Jev at ~2.5 Hz choosing {hold_course, gap_left, gap_right, climb, brake, reacquire} + risk score + target_truly_lost, 500 Hz geometric controller, 50 Hz "code keeps the veto" reflex. It compares Jev vs a no-Jev greedy heuristic, NOT vs a slower LLM, and has no latency sweep, no adversarial-event schedule, no thousands-of-flights statistics ([repo](https://github.com/RomanSlack/jev-drone), [site](https://jev-drone.vercel.app/)).
3. **Latency-injection by converting inference time to sim frames has precedent**: the Time Conversion Mechanism (F_inf = T_inf x FPS) in "Reflex First, Reflect Later" (arXiv 2506.07223), in a non-drone disaster sim. Latency as a first-class variable in real-time decision benchmarks also exists for Street Fighter / HFT (arXiv 2505.19481) and Gaia2 (asynchronous agent env).
4. **Fast/slow dual-system and async planner + reactive layer is the standard pattern** (Helix, DriveVLM-Dual, AsyncDriver, Fast-in-Slow, RRARA, MonTA, PEACE, SkySim, Sinha et al. anomaly MPC). Almost none report a controlled collision-rate-vs-latency curve.
5. **I found no paper/repo combining: drones + closed-loop latency sweep + collision rate + thousands of paired flights + Jev-class vs LLM vs PID/MPC arms.** I also found no LLM-in-the-loop work built on gym-pybullet-drones with latency measurement. Absence of evidence from a finite search, not proof (confidence M).

## 1. Table of works

Columns: Work | What built | Sim/real | LLM/model | Reported latency | How latency handled | Results | Link

### 1A. Direct prior art for this experiment

| Work | What built | Sim/real | Model | Latency | Latency handling | Results | Link |
|---|---|---|---|---|---|---|---|
| jev-drone (R. Slack) | Camera-only quadrotor on 5-station course; Jev = tactical judgment (maneuver Choice, risk Score, target_truly_lost) | MuJoCo sim, Skydio X2, real geometric ctrl | TypeSafe Jev (hosted) | median 0.118 s, p90 0.164 s (tunnel expt); ~21 decisions/s pipelined; ~110 calls per 65 s flight (~80 calls/96k tokens in another run) | 4-rate stack 500/50/15/~2.5 Hz; 50 Hz reflex overrides Jev; scene fingerprint reuses prior judgments | No-Jev greedy heuristic stops at 17.7 m; Jev completes 77.5 m, 0 collisions both, reflex activation 65-71% -> 9%, target visibility 19% -> 82%. Insight: "the state has to contain the answer" (climb chosen only after obstruction height added; p=0.93) | [GitHub](https://github.com/RomanSlack/jev-drone) |
| Jev decision models for edge service orchestration (UTS Sydney) | Jev vs LLMs as interpreter for service admission | Live 3-node OCR service + modeled exec | Jev-1.13.0 (via OpenRouter "Decisions" endpoint) vs DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash, self-hosted 4B | Base-length median/p95: Jev 0.274/0.381 s; DeepSeek 0.378/1.054; GLM 0.646/2.685; Qwen 1.774/3.668 | Deadline-based "on time" metric under offered load 1-16 req/s | Jev 0.91-0.95 exact+on-time where LLMs <0.1; median latency 22.7-64.5% lower than fastest LLM; cost/correct decision 59.7-80.9% lower. Limits: accuracy drops with 8-field contracts (0.53-0.58 vs DeepSeek 0.90-0.93); single domain | [arXiv 2609.22753](https://arxiv.org/abs/2609.22753) |
| RRARA "Reflex First, Reflect Later" | Rule-based reflex acts immediately, LLM reflects asynchronously and intervenes (~60% of steps) | Sim (HAZARD fire/flood/wind, 30 FPS) | GPT-3.5/4/4.1, Llama-2-7b | RL: GPT-3.5 2.35 s, GPT-4 4.11 s, Llama-2-7b 15.6 s | **Time Conversion Mechanism** F_inf = T_inf x FPS: inference time advances the sim | LLM-only agents lose to simple rule/greedy baselines because of latency; RRARA best. Direct precedent for latency-injection design | [arXiv 2506.07223](https://arxiv.org/html/2506.07223v1) |
| Win Fast or Lose Slow (FPX) | Adaptive model-size/quantization selection; HFTBench + StreetFighter | Sim/game | Qwen-family 1.5-14B, various quant | e.g. 3B@6.8-bit 195 ms vs 7B@7.2-bit 354 ms (repo table, RTX 5090) | Choose smaller/quantized model; win-rate rises as latency falls to a Pareto point then drops | Win rate up to +80% (SF), daily yield +26.52% (HFT); 20% lower response time beats slower agents. Latency-quality Pareto curve = the shape this project wants for drones | [arXiv 2505.19481](https://arxiv.org/abs/2505.19481), [repo](https://github.com/HaoKang-Timmy/LatencySensitiveBench) (repo page 403'd on second fetch; table came from first fetch) |
| Gaia2 / ARE (Meta) | Async agent benchmark, 1,120 scenarios, time passes while agent thinks | Sim (phone-like env) | GPT-5, Gemini 2.5 Pro, Claude-4 Sonnet etc. | not per-ms | Wall-clock coupling | Inverse scaling under time limits: GPT-5 0.0% on Time-constrained split in default mode. Not robotics | [arXiv 2509.17158](https://arxiv.org/pdf/2509.17158), [2602.11964](https://awesomepapers.io/ai-agents/papers/2602.11964) (M) |
| SauerkrautLM-Doom | 1.3M-param specialised model vs big LLMs in DOOM | Game sim | small custom vs large LLMs | 31 ms/decision | Fast small model | 178 frags/10 eps vs 13 combined for all large LLMs. Shows "small fast task-trained beats big slow"; does not say if game pauses | [arXiv 2604.07385](https://arxiv.org/abs/2604.07385) |
| Sinha et al. real-time anomaly detection + reactive planning | Fast embedding-space classifier + slow LLM fallback chooser | Sim + real quadrotors/vehicles | small LMs, GPT | not numeric in abstract | **MPC keeps multiple fallback plans jointly feasible for the K steps the slow reasoner needs** | Fast classifier beats autoregressive GPT reasoning | [arXiv 2407.08735](https://arxiv.org/abs/2407.08735) |

### 1B. LLM/VLM drone control systems

| Work | What built | Sim/real | Model | Latency | Handling | Results | Link |
|---|---|---|---|---|---|---|---|
| TypeFly | English task -> MiniSpec plan, streamed and interpreted while LLM is still generating; edge vision encoder to text | Real Tello drone | GPT-4 | ~1.2-1.3 s response (query -> first action) | Token-efficient DSL + stream interpreting | Response time -62% (stream) / -32% (MiniSpec only) vs Python; 10/10 on most tasks; weak on obstacle size estimation; IEEE TMC 2025 | [arXiv 2312.14950](https://arxiv.org/abs/2312.14950) |
| PEACE (planner-executor UAV) | LLM single-pass planner -> ROS2 tool calls -> MAVLink; constraint layer for altitude/geofence; bounded replanning | PX4 SITL, Gazebo | LLM + YOLO/VLM | not quantified | Decouple LLM from control loop | Fewer LLM calls, constraint enforcement | [arXiv 2606.00104](https://arxiv.org/abs/2606.00104) |
| SkySim | Gemini translates commands to waypoints; APF safety filter at 20 Hz | ROS2/Gazebo swarm | Gemini | not quantified | "High-level planner has variable latency; reactive APF filter handles safety" | Feasible, constraint-respecting execution | [arXiv 2602.01226](https://arxiv.org/abs/2602.01226) |
| DroneCATS | MLLM-in-loop benchmark: approach, track, search, multi-drone | not stated in abstract | 2B to frontier MLLMs | none | - | Small models navigate but mis-declare arrival / fail to terminate; frontier models struggle to navigate; failure is protocol discipline and termination action | [arXiv 2609.01404](https://arxiv.org/abs/2609.01404) |
| FSD-VLN | Fast DiT stream + slow VLM semantic stream for aerial VLN | Sim (low-altitude) | VLM + DiT | per-action delay and runtime cut >50% | Two async branches | Up to 2x success on unseen scenes | [arXiv 2607.08359](https://arxiv.org/abs/2607.08359) |
| Language-to-Action | Fine-tuned Qwen2.5-7B-Drone, templated data from A*/Dijkstra/RRT, LTA_Interpreter DSL | Uses **gym-pybullet-drones** | Qwen2.5-7B (LoRA) | none documented | none | No collision/latency metrics in README; open-loop language-to-skills | [GitHub](https://github.com/Tuprott991/Language-to-Action-How-LLMs-Efficiently-Control-Drones) |
| autonomous-drone-sim (MichaelFowler1) | Camera-only city exploration, mapping, English orders | PyBullet (own cascaded controller, **not** gym-pybullet-drones as far as seen) | local LLM via Ollama | not checked | not checked | Only seen via search snippet (L) | github.com/MichaelFowler1/autonomous-drone-sim |
| LLM voice UAV control | Vosk STT + Gemini code gen, EN/AR | Quadrotor (sim/real mix) | Gemini | 300-500 ms end-to-end command execution (their measure) | edge-cloud hybrid | 95% STT accuracy | [MDPI Electronics 14(21):4312](https://doi.org/10.3390/electronics14214312) |
| Closed-loop LLM UAV w/ semantic observations | Generator + evaluator LLMs refine UAV code from semantic trajectory descriptions | Sim | LLMs | none | Refinement loop (offline between runs, not in-flight) | Higher success/completeness with task complexity | [arXiv 2507.01930](https://arxiv.org/abs/2507.01930) |
| Aerial VLN: AerialVLN, OpenFly, CityNav, NavAgent, FlightGPT | Long-horizon instruction navigation, mostly discrete actions | Sim (UE/AirSim/GS) | VLM/LLM agents | not a focus | - | AerialVLN test-unseen SR 7.4/4.5/1.8% (easy/normal/hard); OpenFly-Agent 34.3% seen, 22.6% unseen | [survey 2604.13654](https://arxiv.org/html/2604.13654v1), [OpenFly 2502.18041](https://arxiv.org/html/2502.18041v7) |
| PromptCraft / ChatGPT for Robotics; SayCan; Code-as-Policies; RT-2; OpenVLA; NavGPT; UAV-VLA **[bg]** | Language-to-skill/code/action foundations | Mostly real manipulators; UAV-VLA for aerial mission planning | GPT-x / PaLM / VLA | seconds-scale, not optimised | Open-loop or per-step planner; RT-2 and OpenVLA run 3-10 Hz class | - | Not re-fetched; cite from memory only (L-M) |

### 1C. Dual-system / latency-hybrid architectures (non-drone)

| Work | Fast layer | Slow layer | Numbers | Link |
|---|---|---|---|---|
| Helix (Figure) | S1 80M visuomotor policy, 200 Hz | S2 7B VLM, 7-9 Hz; latent vector passed asynchronously; training deliberately injects deployment-matching temporal offsets | fetched from vendor page (H for description, vendor claim) | [figure.ai/news/helix](https://www.figure.ai/news/helix) |
| Fast-in-Slow | Execution module inside VLM | VLM System 2 | 117.7 Hz control, chunk=8; +8% sim, +11% real | [arXiv 2506.01953](https://arxiv.org/abs/2506.01953) |
| Real-Time Chunking | Async execution of action chunks with inpainting frozen prefix | - | Tested with +100/+200 ms injected latency; best throughput at all delays | [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) |
| DriveVLM-Dual | Traditional 3D perception + motion planner | VLM async, coarse waypoints refined by fast planner | Deployed on production vehicle; latency numbers not retrieved | [arXiv 2402.12289](https://arxiv.org/abs/2402.12289) |
| AsyncDriver | Real-time planner | Llama2-13B features, LLM every N frames | LLM every 3 frames: ~-40% inference time for ~1% loss; nuPlan Hard20 65.00 | [arXiv 2406.14556](https://arxiv.org/html/2406.14556v1) |
| ASSCG | Fast planner | Gated Query/Cache/Drop of slow LLM | AsyncDriver nuPlan 67.28 with 60% latency reduction | [arXiv 2606.25509](https://arxiv.org/abs/2606.25509) |
| MonTA | 7 Hz monitor | Adapter LLMs | Overcooked-AI | [arXiv 2412.00435](https://arxiv.org/abs/2412.00435) |
| LLMs are General Asynchronous Agents; Asynchronous Reasoning; Second Thought | - | concurrent think/act coroutines | no numbers retrieved | [2609.35427](https://arxiv.org/abs/2609.35427), [2512.10931](https://arxiv.org/pdf/2512.10931), [2608.13667](https://arxiv.org/pdf/2608.13667) |

Note: several of the above have 2026 arXiv IDs; I could only read abstracts/summaries. Treat their numbers as "as stated in abstract".

## 2. How latency was handled in the literature (taxonomy)

1. **Decouple by rate** (Helix, jev-drone, SkySim, PEACE): slow planner never in the stability loop; fast layer has veto. Latency is hidden, not measured against safety.
2. **Latency budgeting with guarantees** (Sinha et al.): plan K-step windows where all fallbacks remain feasible. This is the formal version of the "feasibility guarantee" agent 06 must build.
3. **Latency-to-time conversion** (RRARA TCM; Helix's trained-in offsets; RTC's injected delay): inference time advances the world. Direct model for option (c)/(d) in the brief's Q1.
4. **Reduce latency** (TypeFly streaming DSL; FPX smaller/quantized; Jev non-autoregressive): attack the delay.
5. **Query less often / gate** (AsyncDriver every-N-frames; ASSCG Query/Cache/Drop; jev-drone scene-fingerprint reuse).
6. **Smaller specialised model** (DOOM 1.3M, Jev, anomaly embedding classifier).

Observed pattern: nearly every paper argues "fast layer + reflex beats LLM-only" but reports task score, not an explicit collision-rate-vs-injected-latency curve with CIs.

## 3. Discrete-action LLM policies: known failure modes

- Missing information in state: Jev never chose climb until height/visibility was added (jev-drone). Implication: prompt/state design (agent 07) is an experimental factor, not a constant.
- Termination/protocol failures: DroneCATS small models declare arrival early or never stop. Implication: HOVER/RETURN_HOME/EMERGENCY_LAND overuse and "no-op" bias; measure invalid-output and stuck rates.
- Latency collapse: RRARA LLM-only below greedy; Gaia2 inverse scaling with reasoning models. Implication: the "slow LLM" arm must be reported with and without reasoning/thinking tokens.
- Wide-contract degradation: Jev accuracy fell at 8 fields, and self-hosted decision models failed beyond 16 or 254 options (UTS paper). Implication: an 11-way Choice is plausible but untested; Jev limit is 3 questions per call; verify max options (agent 01).
- Geometry weakness: TypeFly fails at obstacle dimension estimation; LLMs reason poorly over raw numbers. Implication: precompute closing speed, time-to-collision, sector clearances.
- Non-determinism and version drift of hosted models (UTS paper pins "Jev-1.13.0"): log model version per call.

## 4. gym-pybullet-drones specifically

- Found: Language-to-Action (Qwen2.5-7B-Drone, builds on it; no latency/collision metric), several RL forks (vkotaru, XHR-ZJU, jackyoung96, trangml multi-task), MuJoCo-Drones-Gym (arXiv 2606.08039, a GPU MuJoCo alternative citing it). Base paper: Panerati et al., arXiv 2103.02142 (H).
- Not found: any gym-pybullet-drones repo/blog with an LLM or decision-API in the loop under injected latency. Searches tried: "LLM drone gym-pybullet-drones", "ChatGPT PyBullet quadrotor", "latency collision". (Confidence M; search engine is US-only and shallow on GitHub.)
- Nearest full-stack precedent for this project's architecture is jev-drone on MuJoCo, not PyBullet.

## 5. Direct answers to the specific questions asked

| Question | Answer |
|---|---|
| Closed-loop latency-vs-collision experiments? | Closest: RRARA (latency -> frames, HAZARD, task score), FPX (latency-win-rate Pareto, games/HFT), RTC (+100/200 ms injected, manipulation throughput), AsyncDriver (LLM interval vs score). None on drones with collision rate; none sweep 0 to 3000 ms. |
| Fast/slow dual-system agents? | Many (Helix, FiS, DriveVLM-Dual, FSD-VLN, ASSCG, MonTA, RRARA). Only FSD-VLN is aerial, and is VLN, not collision avoidance. |
| Async LLM planners with reactive controllers? | PEACE, SkySim, jev-drone (drone); AsyncDriver (driving); Sinha (quadrotor MPC). |
| LLM-controlled drones in PyBullet / gym-pybullet-drones? | Language-to-Action (training-centric); autonomous-drone-sim (plain PyBullet). No latency study. |
| "Reaction time" of LLM agents? | Framed as "response latency" (RRARA), "latency-sensitive decisions" (FPX), Gaia2 temporal windows, HRC guideline 100-500 ms fluent vs 2-5 s deliberate. No standard reaction-time metric defined sense->act in drones. |
| Exact experiment done by anyone? | No full match found. jev-drone does Jev-in-loop with a reflex, no LLM arm, no sweep, no stats. |

## 6. Gap analysis

### What appears novel (given my search)
1. **Explicit collision-rate-vs-decision-latency curves for an LLM-class decision layer on a drone**, with injected latency spanning Jev-like (~100 ms) to LLM-like (1-3+ s), and a knee analysis tied to braking distance.
2. **Matched-scenario paired design across arms** (PID-only, PID+reactive avoidance, MPC, RL, FSM over same 11 actions, slow LLM, Jev) over thousands of flights with CIs. jev-drone has n=1-ish demos.
3. **Separation of "speed" from "smartness"**: same decision model run at several injected latencies (and a fast model at slow latency) to decompose effect. RRARA/FPX conflate size and latency.
4. **Adversarial event suite** (sudden obstacle, wind, GPS loss, battery, second drone) with seeded, feasibility-checked scenarios.
5. A reproducible, open harness for a **vendor-neutral `DecisionProvider`** so Jev is one arm.

### Overlap to acknowledge (do not claim first)
- Jev-in-a-drone-loop with a 50 Hz veto reflex: done by jev-drone (Aug-Sep 2026 timeframe, per site/search; exact date not verified).
- Fast/slow architecture as a concept: Helix, DriveVLM-Dual, etc.
- Latency-to-sim-frame conversion: RRARA TCM.
- "Small fast model beats big slow LLM in real-time control": DOOM paper, FPX, Gaia2.

### Safe vs overclaiming

| Safe claims | Overclaiming (avoid) |
|---|---|
| "In our simulator, with a modelled latency of X ms, collision rate for arm A is Y (95% CI) vs arm B." | "Jev is faster than LLMs at drone control" (latency advantage is vendor/UTS-measured on other tasks; must measure ourselves, and Jev is a hosted API with its own tails, 70-500 ms, UTS p95 0.38 s). |
| "Collision rate rises with decision latency; knee near ~N ms for our dynamics/scenarios." | "First/only study of LLM latency in drones" (only "to our knowledge, we did not find..." hedged; searched 2026-09-30, limited). |
| "With reflex layer, latency matters less" (only if measured; jev-drone reports reflex activation fell 65-71% -> 9% with Jev). | "Jev out-reacts PID/MPC." A PID/MPC + reactive avoidance runs at 50-500 Hz and will have lower reaction latency; Jev's claim is semantic judgment, not raw reflex. The plan's killer-demo bars (PID/MPC < Jev < LLM) invert the truth unless the PID arm lacks avoidance. |
| Results on gym-pybullet-drones with our controller mapping and obstacle geometry. | Sim-to-real or general "drone safety" claims; TypeFly is the only real-hardware LLM drone here. |
| "Results hold at injected-latency, not wall-clock, reproduced from recorded latency distributions." | Reporting wall-clock decisions/sec or sim speed that includes API time as if it were a model property. |
| Latency-vs-quality Pareto for a fixed prompt/state design. | Conclusions about Jev quality generally; its state design matters (jev-drone "state has to contain the answer"). |

### Design implications drawn from prior art
- Include jev-drone-style arms: Jev-with-reflex and Jev-without-reflex, LLM-with-reflex and LLM-without-reflex. The reflex/shield ablation is the most informative single factor and is missing from the literature as a controlled comparison.
- Use TCM-style latency injection as primary, with real-latency replay and a real-time validation subset (cite RRARA; matches option c/d).
- Make decision cadence a factor (AsyncDriver every-N, jev-drone 2.5 Hz), not a constant.
- Report invalid-output/timeouts as first-class (DroneCATS, Gaia2).

## 7. Source reliability notes
- TypeSafe/Jev performance numbers: vendor blog/explainer (flaviocopes.com, DataCamp, CloudRaft are third-party summaries of vendor claims) plus the UTS paper (independent academics, but they pin a vendor-provided endpoint; results limited to one domain). Medium-high.
- jev-drone: single-author MIT repo, demo-grade; numbers self-reported; author affiliation with TypeSafe unknown (not verified).
- WebFetch summaries may omit details (e.g., DriveVLM-Dual latency, MDPI page 403, LatencySensitiveBench second fetch 403). Not re-verified from PDFs.

## Decisions recommended
1. Treat Jev as a verified, hosted, OpenRouter-reachable (via Decisions endpoint per UTS paper; agent 01 to confirm) typed-decision API; build `DecisionProvider` to support its state + questions -> probability schema, including probability output for logging calibration.
2. Cite jev-drone, RRARA (TCM), FPX, Helix/DriveVLM-Dual, Sinha et al. in related work; position as the controlled latency-vs-collision, paired-scenario study they lack.
3. Primary latency method: TCM-style injection (sim at max speed, latency in control ticks), with recorded real-latency distributions replayed and a small real-time validation set.
4. Always include reflex-on/off ablations and a decision-cadence factor.
5. Frame the headline as a latency-collision curve and knee, not "Jev beats PID". Reframe the plan's demo bars: a PID+reactive layer is faster at reflex; Jev's value is semantic maneuver choice (e.g., climb vs gap) at ~100 ms.
6. Use hedged novelty language ("to our knowledge, as of Sept 2026").

## Open questions
1. Max options per Choice question and max questions per call in Jev (3 per vendor page); can 11 actions fit in one Choice, or must it split (e.g., maneuver + urgency)? Latency and accuracy of wide choices (UTS shows degradation with wide contracts).
2. Is jev-drone's author affiliated with TypeSafe; does its code contain reusable state-encoding/prompt specs (yes likely; not read in depth)?
3. Exact latency decomposition of Jev via OpenRouter vs direct (network vs inference); p99 tails.
4. Did any 2026 work on arXiv cover drone latency sweeps that my searches missed (search was US-only, limited depth)? A targeted Google Scholar/Semantic Scholar sweep for "inference latency" + "quadrotor" + "LLM" is pending.
5. DriveVLM-Dual exact latency numbers and Helix's offset-injection details (only vendor/abstract level read).
6. Whether RRARA's TCM accounts for the state-staleness (observation age) as well as action delay; relevant to agent 08.

## Follow-up spikes (small, runnable experiments)
1. Clone `RomanSlack/jev-drone` into scratchpad, read its state-to-JSON and question definitions, and record how it handles in-flight calls (pipelined 21 dec/s vs 2.5 Hz). Output: a reusable prompt/state schema and which behaviours depended on the reflex. (Read-only; do not run with keys.)
2. Point-mass or gym-pybullet-drones toy: one obstacle-appears scenario, reflex on/off, decision delay swept over {0, 50, 100, 300, 1000, 3000} ms with a stub policy having oracle-correct actions; plot collision vs latency; identifies the physical knee before any API is used (feeds agent 08's spike).
3. Literature sweep script (Semantic Scholar API, no key) for queries "LLM drone inference latency collision", "reaction time LLM agent", saved to scratchpad with dates; repeat before publishing novelty claims.
4. Replicate TCM on a trivial environment: verify that F_inf = ceil(T_inf x ctrl_hz) with observation-age tracking changes outcomes vs lockstep, to demonstrate the artifact the brief warns about.
