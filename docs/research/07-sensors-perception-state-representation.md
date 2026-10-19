# 07 — Sensors, Perception and State Representation for LLM Prompts

Agent 07. Date of research: 2026-09-30. Hardware for all measurements: Apple M4 (MacBook Pro, 16 GB), macOS, other research agents were running concurrently (so timings are noisy; p95 is inflated, p50 is the trustworthy figure).
Scratch work: `/private/tmp/claude-501/-Users-subhajithait-Documents-testing-drone-jev/0ec4827c-5206-473e-b48d-298a2822fdb4/scratchpad/07/` (`bench_ray.py`, `bench_ray2.py`, `enc.py`, `sys_prompt.txt`, `parse.py`).

Confidence legend: **[V]** verified by measurement or a fetched/searched source, **[I]** inference, **[R]** recalled from background knowledge (not re-verified today).

---

## 1. What prior work does (and what is known about LLM spatial/numeric weakness)

### 1.1 How prior LLM-robotics systems encode spatial state in text

| System | State encoding given to the LLM | Lesson for us |
|---|---|---|
| TypeFly (drones) [arXiv 2312.14950](https://arxiv.org/pdf/2312.14950), [typefly.github.io](https://typefly.github.io/) (searched 2026-09-30) | YOLOv8 vision encoder produces a *text scene description*: object labels with coarse position info (e.g. `chair_314`), plus a skill list/syntax and few-shot examples. Output language (MiniSpec) designed for token efficiency and streaming. [V] | Prior drone work feeds **symbolic, coarse, label-centric** scenes, not raw floats, and minimises *output* tokens, which is what drives latency. |
| SayCan / Inner Monologue / Code-as-Policies ([CaP](https://arxiv.org/pdf/2209.07753), [Inner Monologue](https://proceedings.mlr.press/v205/huang23c/huang23c.pdf)) | Natural-language scene/success-detector feedback appended to the prompt; CaP lets the model emit code that does the arithmetic (numbers computed by a program, not by the LLM). [V] | Keep arithmetic out of the LLM: precompute derived quantities (distance, bearing, ttc) and hand them over as facts. |
| GPT-Driver [arXiv 2310.01415](https://arxiv.org/pdf/2310.01415) | Ego state and perception/prediction as text; trajectories as coordinates tokenized by the GPT tokenizer (`23.17` -> `23`, `.`, `17`); the paper notes this yields coarse-to-fine token classification rather than regression. [V] | Fine-tuned for that; a *prompted* (non-fine-tuned) model should not be asked to output coordinates. We output an enum. |
| Driving with LLMs / LLM-Driver [arXiv 2310.01957](https://arxiv.org/abs/2310.01957) | Object-level *vector modality* fused via adapters (Vector Former, 64 queries) because raw numeric text is weak. [V] | Training-time solution; out of scope for prompted Jev, but evidence that numerics-as-text is a known bottleneck. |

### 1.2 Known LLM weaknesses relevant to us

1. **Text-only spatial gap.** SiT-Bench (3,800+ questions, 17 subtasks) reports a significant "spatial gap" in text-only LLMs ([arXiv 2601.03590](https://arxiv.org/html/2601.03590)); GRASP and Grid Spatial Understanding study coordinate/grid encodings ([2407.01892](https://arxiv.org/pdf/2407.01892), [2603.17333](https://arxiv.org/pdf/2603.17333)); GeoGramBench reports <50% at the highest abstraction level ([2505.17653](https://arxiv.org/html/2505.17653v1)). [V for existence of findings; I have not validated specific numbers beyond these search summaries.]
2. **Frame/perspective errors.** Models revert to egocentric reasoning and confuse left/right under frame changes; text-only models fail relative-direction tasks when orientation is unusual, pairing "left" with "west" without rotating ([SymPL 2602.19117](https://arxiv.org/html/2602.19117), [AlloEgo-VLM 2608.15605](https://arxiv.org/html/2608.15605), [Orientation-heuristic gap, 2509.18200](https://arxiv.org/html/2509.18200)). [V] **Implication:** give everything in ONE body frame (f/l/u) and never ask the model to rotate between world and body frames. Sign of "left" is a prime error source, so use words (`f`, `l`) in the field names rather than bare signed numbers.
3. **Tokenization of numbers.** Number tokenization fragments place value, and accuracy shifts substantially with digit grouping ([2402.14903](https://www.alphaxiv.org/abs/2402.14903), [2601.14658](https://arxiv.org/pdf/2601.14658)). [V] **Implication:** round aggressively (0.1 m or integer metres/degrees), avoid many-digit floats, and precompute comparisons (the model should read `ttc=1.0`, not derive it from range/closing speed).
4. **Prompt-format sensitivity.** FormatSpread: up to 76 accuracy-point swings from trivial formatting changes (LLaMA-2-13B), persists with model size and instruction tuning; format performance correlates only weakly across models ([arXiv 2310.11324](https://arxiv.org/abs/2310.11324)). [V] **Implication:** the prompt format is a *first-class experimental factor* and must be ablated (Section 8) and the headline claim must be stated per-format (or as a range).

**Inference [I]:** With a prompted tiny/fast model, the decision is a pattern-match from salient features (ttc small -> BRAKE/AVOID; battery low -> RETURN_HOME). Formats that expose the salient feature as a pre-digested token (`ttc=1.0`, `near=2.1m@0`) will beat formats requiring multi-step geometry. Fine-tuned/distilled models are less format-sensitive but more brittle to distribution shift.

---

## 2. Sensor modalities, cost and recommended use

All costs measured on M4, pybullet 202010061, Python 3.14.2, numpy 2.5.3, DIRECT mode, scene = plane + 20 static boxes, one sphere body (`bench_ray.py`, `bench_ray2.py`). Note: pybullet could **not** be built from source in my fresh Python 3.12 venv on this machine (zlib `fdopen` macro clashes with the macOS CLT SDK header; `uv pip install pybullet` fails, and only a `cp35m` wheel exists), so I reused the working Python 3.14 venv that agent 02 had built under the scratchpad. **[V]** This is a real install-friction finding for 02/09/13: pybullet has to be compiled on arm64 macOS and the build can break on SDK updates.

| Modality | Source in sim | Cost on M4 (p50) | Noise model | Use in prompt |
|---|---|---|---|---|
| Kinematic state (pos, vel, rpy, rates) | `getBasePositionAndOrientation`, `getBaseVelocity`, or gym-pybullet-drones `_getDroneStateVector` | ~1-2 us per call [I, sub-10 us] | Gaussian + bias drift (owner 06) | velocity in body frame (f/l/u); absolute position only if GPS "ok" |
| Relative target vector | target - pos rotated to body frame | pure numpy, <5 us | GPS noise/loss propagates here | `tgt=14f,-3l,0u d=14 brg=-10` |
| Raycast "lidar", horizontal | `p.rayTestBatch` | see table below | range noise sigma ~ 0.05 m + 1-2% dropout to max range | 8-sector min-range array |
| Raycast, 3D (16 az x 3 el = 48 rays) | `rayTestBatch` | ~31 us per call (single measurement in first run, noisy) [V] | as above | only for multi-arm (PID/MPC/RL) projections; too many tokens for text |
| Depth image | `getCameraImage` TinyRenderer, rgb+depth+seg returned together | 32x32 **3.2 ms**, 64x64 **5.0 ms**, 128x128 **15.4 ms** (single run, 20 reps) [V] | depth noise, dropout | multimodal arm only; also dominates sim step cost (an env step is ~8-60 us) so depth-heavy runs are ~100x slower than ray-only |
| Battery | internal model (owner 06) | free | SoC reading noise +/-2% | `bat=43` (percent, integer) |
| GPS status | internal flag + drift | free | dropout windows | `gps=ok/lost/drift` |
| Nearest obstacle range/closing speed/ttc | derived from rays across consecutive observations (finite difference) or analytic from ground truth for the oracle | free | inherits ray noise; ttc = range/max(closing, eps), capped at 9.9 | `near=2.1m@0 clos=2.0 ttc=1.0` |
| Other drone | simulator ground truth through a limited-FoV/range "radar" filter + noise | free | range/bearing noise, dropout | `other=6.2m@35 closing=1` |

### 2.1 rayTestBatch cost (measured)

Scene: 20 random static boxes, rays length 10 m at constant altitude, all sectors. Median (p50) and 95th percentile over 400 calls, after warm-up. Only "threads=0 / 1" are meaningful for production; `numThreads=4` spin-waits and showed worse small-batch latency (22 us for 8 rays vs 3.7 us).

| Rays (horizontal ring) | numThreads=1 p50 | numThreads=0 (default/auto) p50 | p95 (auto) | us per ray (auto) |
|---|---|---|---|---|
| 8 | 3.7 us | 3.8 us | 3.8 us | 0.5 |
| 16 | 9.1 us | 8.5 us | 20.7 us | 0.5 |
| 36 | 47.6 us | 45.5 us | 189 us | 1.3 |
| 72 | 90.8 us | 72.6 us | 163 us | 1.0 |
| 180 | 111.4 us | 207.5 us | 832 us | 1.2 |
| 360 | 224.4 us | 329.8 us | 827 us | 0.9 |

Other measured numbers: `stepSimulation` for this near-empty world 7.6 us. [V]

**Conclusions.**
- A 16-ray ring costs ~10 us (p50) versus the ~1-2 ms a sim tick budget allows at ~1 kHz-ish wall pacing: negligible (<1-2% of a physics step in gym-pybullet-drones, which per agent 02 is tens of microseconds to a few hundred microseconds per step). Even 360 rays at ~0.3 ms is cheap if evaluated at decision rate only (e.g. 10-50 Hz), not every physics tick.
- **Cast rays at decision/observation rate, not physics rate.** Observation rate 20-50 Hz is enough; cost then <0.1% of run time.
- Tail risk: p95 is 10-50x p50 under machine load (other agents running), and occasional multi-ms spikes appeared (p95 1.7-3.9 ms at numThreads=1/4). This matters for timing accounting (owner 08): record observation-build time separately from decision latency, and run final benchmarks on an idle machine. [V for spikes, I for cause: scheduler contention].
- Scene complexity scales roughly with obstacle count (BVH), not measured here; repeat with 100 obstacles (spike S1).
- Limits from recall [R]: PyBullet docs cap a batch at 16,384 rays; `rayTestBatch` returns `(objectUid, linkIndex, hitFraction, hitPosition, hitNormal)`; hitFraction=1.0 means no hit -> map to max range. `getCameraImage` with `ER_BULLET_HARDWARE_OPENGL` needs GUI/OpenGL and is not usable headless on macOS in DIRECT mode; use TinyRenderer.
- Raycasts need a collision shape on obstacles; an "obstacle suddenly appears" event must create the body (`createMultiBody`, cost ~50-200 us [I]) at event time, so it is visible to rays only from that tick on. Exclude the drone's own body: start rays 0.1 m outside the body radius or filter `objectUniqueId != drone_id`.

### 2.2 Recommended sensing stack

- Text arms: 16-ray horizontal ring at 22.5 deg, reduced to 8 sector minima (45 deg wide) for the prompt; plus one upward and one downward ray for altitude/ceiling margins (optional).
- Non-text arms (PID+avoidance, MPC, RL): full 16-ray ring + 3D extras, same underlying data (same noise realisation) so sensing is paired across arms.
- Depth image: multimodal arm only (optional, 64x64, ~5 ms), never in the main comparison.

---

## 3. `Observation` dataclass sketch with per-arm projections

Design: one **ground-truth-free, noise-applied** `Observation` produced by a `SensorSuite` once per decision-eligible tick (seeded RNG stream separate from the scenario's RNG so events do not shift sensor noise). Projections are pure functions (`utils/`-style), so every arm sees the *same* information.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class GpsStatus(StrEnum):
    OK = "ok"
    DEGRADED = "drift"
    LOST = "lost"


N_RAYS = 16  # 22.5 deg spacing, index 0 = straight ahead, increasing toward LEFT (CCW)
MAX_RANGE_M = 10.0


@dataclass(frozen=True, slots=True)
class Threat:
    """Nearest obstacle or drone, all in the body frame."""

    range_m: float
    bearing_deg: float  # 0 ahead, +left, -right
    closing_speed_mps: float  # >0 means approaching
    ttc_s: float  # capped at 9.9


@dataclass(frozen=True, slots=True)
class Observation:
    # --- timing (critical for latency studies) ---
    sim_time_s: float  # sim time at which the data was MEASURED
    age_ms: float  # staleness injected by latency model at time of decision (filled by harness)
    decision_index: int
    # --- own state, body frame (x fwd, y left, z up) ---
    vel_body_mps: NDArray[np.float64]  # (3,)
    yaw_rate_dps: float
    altitude_m: float | None  # baro, always available
    # --- mission ---
    target_body_m: NDArray[np.float64] | None  # None when GPS lost and no fallback
    home_body_m: NDArray[np.float64] | None
    gps: GpsStatus
    battery_frac: float  # 0..1, noisy
    wind_est_body_mps: NDArray[np.float64] | None  # (2,) or None (estimator optional)
    # --- perception ---
    rays_m: NDArray[np.float64]  # (N_RAYS,), MAX_RANGE_M = no hit
    nearest_obstacle: Threat | None  # derived from rays, finite differences
    other_drone: Threat | None  # radar-style, may be None (dropout / out of range)
    # --- context ---
    last_action: str
    last_action_age_ms: float  # how long that action has been executing
    # --- optional ---
    depth_64: NDArray[np.float32] | None = None  # multimodal arm only


class ObservationProjection(Protocol):
    """Pure mapping Observation -> arm-specific input. No state, no I/O."""

    def __call__(self, obs: Observation) -> object: ...


# Per-arm projections (all pure functions in utils/ or app/harness/projections.py)
def to_text_prompt(
    obs: Observation, fmt: str
) -> str: ...  # Jev / slow LLM arms (fmt in A..F)
def to_fsm_features(
    obs: Observation,
) -> "FsmFeatures": ...  # rule-based arm: sectors, ttc, battery, gps
def to_pid_avoidance_input(
    obs: Observation,
) -> "AvoidanceInput": ...  # full 16 rays + target vector (APF/VFH)
def to_mpc_input(
    obs: Observation,
) -> "MpcInput": ...  # state vector + obstacle list (range/bearing -> xyz)
def to_rl_vector(
    obs: Observation,
) -> NDArray[np.float32]: ...  # fixed-length normalised vector, see below
```

RL vector normalisation (for arm D): `vel/5`, `tgt/20` clipped to [-1,1], `rays/MAX_RANGE`, `battery`, one-hot gps, `ttc/10`, `age_ms/1000`, last_action one-hot (11). ~55 dims. [I]

Rules:
- **Units SI**, **body frame**, **bearing 0 = ahead, positive = left**, **degrees** for angles in text, radians internally.
- Ground truth (true positions, true events) lives in a separate `GroundTruth` object used only by the oracle labeller and metrics, never in any projection.
- `age_ms` is filled by the harness from the latency model (owner 08). Whether to show it to the model is a prompt-ablation factor (Section 8).
- Sensor failure semantics: GPS lost -> `target_body_m=None`, prompt shows `tgt=unknown`; `gps=lost`. Missing data is explicit, never silently zero.

---

## 4. Candidate prompt encodings with real token counts

Same scenario in all encodings: drone flying forward at 2.3 m/s; target 14.2 m ahead, 2.6 m to the right, 0.3 m up; battery 43%; GPS ok; obstacle 3.4 m away at -20 deg closing 2.1 m/s (ttc 1.6 s); other drone at 6.2 m, +35 deg; wind 0.8f/-0.4l; 16 rays; `age=180 ms`; last action FORWARD. Tokens counted with tiktoken 0.14.0 (`cl100k_base` and `o200k_base`). **Caveat:** Jev's real tokenizer is unknown (and Llama/Qwen/Gemma/Mistral tokenizers differ; expect +-20%), so absolute counts are a proxy; the *ranking* is more robust. Observation-only counts (no system prompt). Source: `enc.py`; raw encodings in `enc_*.txt` in the scratch dir.

| ID | Encoding | chars | cl100k | o200k | Notes |
|---|---|---|---|---|---|
| A | Compact JSON (16 raw rays, full floats) | 269 | **171** | 175 | Most verbose per information; quotes/braces/commas each cost tokens; raw 16-float array invites misreading |
| B | `key=value` lines with verbose names | 297 | **138** | 142 | Readable; explicit frame words (`fwd`, `left`) |
| C | Sector array, rounded ints, one-line groups | 229 | **137** | 137 | Positional array `ranges_m(F,FL,L,BL,B,BR,R,FR)=3,10,...` |
| D | ASCII top-down grid (11x11, 1.5 m/cell) + status line | 264 | **89** | 88 | Cheapest, but ceiling on info (quantised 1.5 m; ~0.75 m error) and relies on the model's 2D visual-in-text parsing, the class of task shown to be hard ([GRASP](https://arxiv.org/pdf/2407.01892)) |
| E | Natural language | 503 | **144** | 145 | Low chars-per-token efficiency; lossless-looking but ambiguity ("a bit to the right"); longest to generate and read |
| F | **Hybrid terse**: abbreviations + 8-sector ranges + precomputed `near/ttc/brg/d` | 200 | **134** | 136 | Short header fields, derived quantities up front; recommended default |

Findings:
- Differences are small in absolute terms (**89-171 tokens**; B, C, E, F all within ~10 tokens). Observation size is not what drives latency at this scale; the **fixed system prompt (~545 tokens) dominates**, so the lever is prefix caching, not shaving 20 tokens. [V numbers; I for the latency conclusion, owner 01 should confirm with provider TTFT data: typical prefill is ~ linear in tokens but tiny at <1k]
- Numbers with a decimal point cost 2-3 tokens (`14`, `.`, `2`); integers cost 1; 1-3 digit integers are one token under both tokenizers. [V: digits 0-9 are all 1 token, measured] Rounding to integers roughly halves numeric-token cost.
- Output tokens (all 11 action names): `FORWARD`=2, `LEFT`=1, `RIGHT`=1, `ASCEND`=2, `DESCEND`=2, `BRAKE`=2-3 depending on tokenizer, `HOVER`=2, `AVOID_LEFT`=3, `AVOID_RIGHT`=3, `RETURN_HOME`=2, `EMERGENCY_LAND`=5. **Uneven action lengths cost generation time and create a latency bias toward short names** (EMERGENCY_LAND takes 5 sequential decode steps vs 1 for LEFT). Recommendation: use **single-token aliases** at the wire level (digits `0`-`9` plus a letter, or single letters `F L R U D B H a b X E`; every single letter/digit measured as 1 token) mapped back in the parser, with `max_tokens=1-2`. Check against the real tokenizer before freezing. Trade-off: semantically meaningless aliases lose some zero-shot competence; ablate (Section 8, factor OUT).
- Sectors as a fixed-length positional list are error-prone for small models (they mis-index). Mitigation, as done in F: explicit order in the field name `rng[F,FL,L,BL,B,BR,R,FR]` and, for the sectors that matter, ALSO give `near=range@bearing`. The `near` field is redundant by design (robustness).

Full example of F (recommended):

```
age=180 last=FWD v=2.3 bat=43 gps=ok
tgt=14.2f,-2.6l,0.3u d=14 brg=-10
rng[F,FL,L,BL,B,BR,R,FR]=3.4,7.8,9.9,9.9,9.9,9.9,3.6,3.4
near=3.4m@-20 clos=2.1 ttc=1.6
other=6.2m@35 closing=1
wind=0.8f,-0.4l
```

(Note: the ASCII arm D in my script drew the obstacle cluster near the front-right; the same info is conveyed, but it demonstrates how cell quantisation loses the 3.4 m vs 3.6 m distinction. Keep D as a low-token ablation arm only.)

### 4.1 History

Options: (a) stateless, current frame only, plus derived `clos`/`ttc` (finite differences done by the harness) [recommended default]; (b) last-k compressed deltas (e.g., `hist=[F:9.9,9.9,9.1,7.4,5.2]` = forward range over last 5 obs), ~+15-25 tokens; (c) action history `acts=FWD,FWD,FWD`, ~+6 tokens, helps oscillation. Recommend (a) + last action, then ablate (b). Rationale: the LLM is bad at differencing numbers [1.2]; the harness can do it for free, and giving history raises the risk of the model echoing earlier actions.

---

## 5. Full example system prompt, action glossary, output schema

Measured size of the prompt below: **544 tokens (cl100k), 548 (o200k)**. Static prefix (cache-friendly; put it first, put the changing observation last). File: `sys_prompt.txt` in the scratch dir.

```
You are the tactical decision module of a quadrotor. A PID controller handles stabilisation; you only choose ONE discrete action every decision cycle. Reply with exactly one action token and nothing else.

FRAME: body frame. f=forward, l=left (negative = right), u=up. Bearings in degrees: 0=ahead, +left, -right. Ranges in metres, speeds m/s. rng[...] = min free range per sector (F,FL,L,BL,B,BR,R,FR), 10 = clear. "age" = milliseconds since this data was measured; the drone keeps executing your previous action meanwhile, so assume the situation has moved on by age*speed.

ACTIONS:
FORWARD        fly toward the target at cruise speed
LEFT / RIGHT   sidestep 1 m/s laterally while holding heading
ASCEND / DESCEND  change altitude 1 m/s
BRAKE          decelerate to zero velocity (use when closing fast: ttc < 1.5 s)
HOVER          hold position
AVOID_LEFT / AVOID_RIGHT  swerve around the nearest threat to that side, then resume
RETURN_HOME    abort mission, fly to home (battery < 25%, or GPS lost and target unreliable)
EMERGENCY_LAND descend and land now (battery < 8%, or unavoidable collision)

PRIORITY: 1) imminent collision (ttc<1.5 or F range<3): BRAKE, or AVOID toward the side with the larger free range; 2) battery/GPS; 3) other drone on converging course: slow/avoid; 4) progress to target.
If target is within 1 m and speed < 0.5: HOVER.

EXAMPLES
age=120 last=FWD v=2.0 bat=80 gps=ok
tgt=12f,0l,0u d=12 brg=0
rng[F,FL,L,BL,B,BR,R,FR]=2.1,4,10,10,10,10,10,2.5
near=2.1m@0 clos=2.0 ttc=1.0
=> AVOID_LEFT

age=90 last=FWD v=2.0 bat=20 gps=ok
tgt=30f,2l,0u d=30 brg=4
rng[F,FL,L,BL,B,BR,R,FR]=10,10,10,10,10,10,10,10
=> RETURN_HOME

Output format: one of the 11 action names, uppercase.
```

Design choices and cautions:
- **Thresholds in the prompt (battery<25, ttc<1.5) are a leak of the rule-based baseline.** For fairness (owner 05, tuning-effort parity), use the *same* thresholds as the FSM arm or withhold them and let the model use judgment; decide once, document, and ablate (factor HINTS: with / without numeric thresholds). Also the glossary semantic must exactly match the controller mapping of owner 11 (e.g. what "swerve then resume" means is a setpoint macro with TTL).
- Few-shot examples bias. Two examples above are deliberately different classes; production should use 6-8 examples, balanced per action (prior probability leakage: if 5/6 shots are AVOID_LEFT, the model over-produces it) and left/right mirrored. Examples are a factor (0, 2, 8 shots).
- Examples must not come from the evaluation scenarios (leak).
- Left/right: examples must include mirror pairs to counter left/right bias [1.2].
- No chain-of-thought: any reasoning tokens multiply latency (inference: 50-200 extra tokens = seconds on slow providers). A "slow LLM" arm may use reasoning as a legitimately different arm, but the Jev arm must be output-constrained.
- Note the slow-LLM arm should receive the *same prompt* (paired design) so differences come from the model, not the prompt; optionally give it a variant asking for a short rationale as a separate arm.

### 5.1 Action glossary (single source of truth; shared with owner 11)

| Wire alias | Name | Family | Semantics for the LLM | Latched / one-shot (owner 11 decides) |
|---|---|---|---|---|
| F | FORWARD | navigate | fly toward target at cruise | latched |
| L / R | LEFT / RIGHT | navigate | sidestep | TTL |
| U / D | ASCEND / DESCEND | navigate | altitude change | TTL |
| B | BRAKE | safety | zero velocity | latched until next action |
| H | HOVER | safety | hold position | latched |
| a / b | AVOID_LEFT / AVOID_RIGHT | reactive | swerve then resume | macro with TTL |
| G | RETURN_HOME | mission | abort and go home | latched, sticky |
| X | EMERGENCY_LAND | emergency | land now | terminal |

Aliases are an experimental factor (Section 8); the canonical names remain the DB/enums. (Letters `a`/`b` are a trap: confusable; in the final design use digits 0-9 and `A`, `E`, or verify tokenization.)

### 5.2 Output schema

Primary (lowest latency, recommended for Jev): **plain single token**, `max_tokens=2`, `temperature=0`, optional `stop=["\n"]`. Enforcement layers, strongest first:
1. If the provider supports `logit_bias`, `allowed_tokens`/grammar, or `response_format` enum: use it (owner 01 to confirm what OpenRouter passes through; constrained decoding can *add* latency at some providers [I]).
2. Otherwise the parser in Section 6.

Structured alternative (slower, for slow-LLM arm or when a provider insists on JSON):

```json
{"type":"object","properties":{"action":{"type":"string","enum":["FORWARD","LEFT","RIGHT","ASCEND","DESCEND","BRAKE","HOVER","AVOID_LEFT","AVOID_RIGHT","RETURN_HOME","EMERGENCY_LAND"]}},"required":["action"],"additionalProperties":false}
```

The JSON wrapper costs ~8-10 extra output tokens; measure the latency penalty (spike S3). Ask for no confidence/rationale in the Jev arm: every field is decode steps. A separate *offline* evaluation can ask for a confidence to study calibration, not in the timed arms.

---

## 6. Parse-failure and fallback policy

Principle: **never stall the control loop on a bad reply; never mask failures in the metrics.**

Parser (tested in `parse.py`):

```python
ACTIONS = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "ASCEND",
    "DESCEND",
    "BRAKE",
    "HOVER",
    "AVOID_LEFT",
    "AVOID_RIGHT",
    "RETURN_HOME",
    "EMERGENCY_LAND",
]
PAT = re.compile(r"\b(" + "|".join(sorted(ACTIONS, key=len, reverse=True)) + r")\b")


def parse(txt: str) -> tuple[str | None, str]:
    t = txt.strip().upper().replace(" ", "_").replace("-", "_")
    if t in ACTIONS:
        return t, "exact"
    u = re.sub(
        r"\b(AVOID|RETURN|EMERGENCY)[ -](LEFT|RIGHT|HOME|LAND)\b", r"\1_\2", txt.upper()
    )
    m = PAT.findall(u)
    if len(set(m)) == 1:
        return m[0], "extracted"
    return None, "fail"  # empty, ambiguous, or multiple distinct actions
```

Verified outcomes: `BRAKE`/`avoid left` -> exact; `brake.`, `**AVOID_LEFT**`, `The answer: HOVER` -> extracted; `I think LEFT or RIGHT`, `AVOID_LEFT is better than LEFT`, empty -> fail (ambiguity is a failure, deliberately: don't guess a direction). Do not use "first match wins".

Failure taxonomy (each logged as a distinct `decision.status`): `ok_exact`, `ok_extracted`, `invalid_output` (unparseable), `ambiguous`, `timeout` (exceeds the decision deadline, counted even if reply arrives later), `transport_error` (HTTP 429/5xx/connection), `refused` (content policy).

Fallback policy (per failure, in order):
1. **Retry: none inside the control loop** (a retry doubles latency and moves the state). Retries may be used in an offline "eval reliability" mode only.
2. **Hold last action** for up to `TTL` (e.g., 0.5 s sim time) if the last valid action was safe and not terminal (FORWARD/LEFT/RIGHT/ASCEND/DESCEND/HOVER).
3. **Reflex safe action** if the situation is threatening (local rule on the same observation: `ttc<1.0 -> BRAKE`, `battery<5% -> EMERGENCY_LAND`) — this is the *safety shield* of owner 11; it must be identical across all arms that have a shield, or off for all in the "raw" comparison.
4. After `N=3` consecutive failures: HOVER and mark `degraded`; after `M=10`: end flight as `provider_failure` (excluded from collision-rate denominators but *reported*).
5. Report invalid-output rate, timeout rate and "fallback-activated fraction of ticks" per arm next to collision rate; an arm with 10% invalid outputs is a different result than a perfect parser. Two headline numbers: *raw* (no shield/fallback) and *shielded*.

Non-determinism: temperature 0 does not guarantee determinism across providers/batching [R]; store the raw reply text, model id/version/provider in the decision row (owner 10) so flights can be re-parsed.

---

## 7. Units, frames, normalisation, decision-age: rules summary

1. One frame (body, f/l/u), one unit system (m, m/s, deg, %). Name axes with letters attached to numbers (`14f,-3l,0u`), not bare tuples: measured as cheap and unambiguous. [I, motivated by 1.2]
2. Round at emission: distances 0.1 m under 10 m else 1 m; speeds 0.1; bearings 5 deg (integer, multiples of 5 lose nothing for 45-degree sectors); battery integer percent. Reduces numeric tokens and false precision.
3. Cap sentinel values: "no hit" = 10 (MAX_RANGE) in rays, `ttc` capped at 9.9. Never `inf`/`nan` in the text.
4. **Decision age** (`age`): include as an explicit field (ms). Hypothesis: a model given `age` can lead the target (e.g., brake earlier). Expect small models to mostly ignore it; make the field an ablation factor (present / absent / *pre-compensated*, where the harness linearly extrapolates the state to decision time so the model sees "current estimate"). Pre-compensation is also a legitimate baseline for the classical arms.
5. Do not put the *simulated* clock in the text (no `t=12.4`): it breaks prefix-cache opportunities, encourages time-based cheating, and adds tokens. (I dropped `t` in F.)
6. Consistent field order: rare fields (other drone, wind) can be omitted when absent; but omitting shifts positions; prefer stable `other=none`. 4 tokens per always-present field is the price of stability. Decide by ablation.

---

## 8. Prompt-format ablation design

Motivation: format swings of tens of accuracy points have been documented ([FormatSpread](https://arxiv.org/abs/2310.11324)); the claim "Jev beats X" has to hold across formats, or be reported with a range over formats.

### 8.1 Two-stage design (cheap first, expensive second)

**Stage 1: offline decision-accuracy ablation (no sim, no latency, cheap).**
- Dataset: N = 2,000 frozen observations generated by the sim (balanced across event types and severity, including near-threshold cases), each with an **oracle label** from the labeller (owner 11/06: the set of *acceptable* actions, not one action; e.g. for an obstacle dead ahead with clear sides: {BRAKE, AVOID_LEFT, AVOID_RIGHT}; for "benign" states: {FORWARD}). Metrics: acceptable-action rate, safety-critical-miss rate (chose FORWARD when a threat required action), invalid-output rate, left/right-mirror consistency (mirror the observation; the chosen action must mirror: a cheap symmetry test that directly detects the frame bug of Section 1.2), input tokens, TTFT/latency p50.
- Factors (full factorial is too large; use a fractional/Latin design):
  - FMT: {A JSON, B key-value, C sector array, D ASCII, E natural language, F hybrid} (6)
  - AGE: {absent, present, pre-compensated} (3)
  - SHOTS: {0, 2, 8} (3)
  - OUT: {name, single-token alias} (2)
  - HINTS: {no thresholds in prompt, thresholds} (2)
  - SEED/prompt-order perturbations: example order shuffled x3
  Full = 6x3x3x2x2 = 216 cells x 3 orders = 648 prompt variants x 2,000 obs = 1.3 M calls; too many. Use **main-effects screening**: one-factor-at-a-time around the baseline (F, present, 2-shot, name, no-hints): 6+3+3+2+2-4 = 12 cells x 3 orders x 2,000 = 72,000 calls. Then full factorial only on the top 2 levels of each factor on a 500-obs subset: 2^5 x 3 x 500 = 48,000 calls. Against a local model (MLX/llama.cpp) this is free (owner 01/09); against paid APIs, it costs real money, so run stage 1 on the stand-in local model and confirm on a 300-observation subset via the API.
- Analysis: mixed-effects logistic regression (acceptable ~ fmt + age + shots + out + hints + (1|observation)), report per-factor effects with CIs, plus FormatSpread-style range (min/max over formats) per model. Paired by observation, so use McNemar/paired bootstrap for pairwise format comparison with Holm correction.

**Stage 2: closed-loop confirmation (expensive).** Take the best and worst format from Stage 1 plus the "naive" one (E natural language) and run them in the full flight benchmark on the same paired scenarios as the other arms (owner 08). Report collision rate and time_to_target per format. Only the closed-loop results can go in the headline.

### 8.2 Controls and confounds
- Token count vs information content: formats differ in tokens, so latency differences partly come from format. Always report (accuracy, tokens, latency) jointly; plot accuracy vs tokens as Pareto.
- Language-model-specific: run on >= 2 models (stand-in small local + one hosted fast model) to check that "best format" transfers; FormatSpread says it often does not.
- Seed control: temperature 0; note repeated calls may still differ; sample each (obs, variant) twice for a stability estimate (disagreement rate).
- Positional artefacts: shuffle sector order variants (F,FL,L,... vs clockwise) in a subfactor to test index-misreading.
- Mirror-symmetry test (as above) and "frame perturbation" test (rotate the world so the obstacle is at +20 instead of -20; action must mirror).

---

## 9. Multimodal (optional arm)

- Input: 64x64 depth (5 ms render on M4, measured) or 128x128 (15 ms) plus the same text status line. Tokenization of an image by hosted VLMs is typically hundreds of tokens per image [R], so slower; TypeFly uses a separate vision encoder precisely to keep the planner text-only [V].
- Verdict: **exclude from the main comparison; include only as a small validation arm** ("does a picture beat 16 rays?") because (i) depth images cost 3-15 ms per frame vs ~10 us for rays, slowing sim by 2-3 orders of magnitude; (ii) latency for VLM calls is larger and variable, so the arm measures a different thing; (iii) no fair classical-arm counterpart besides the raw depth-to-rays reduction, which is exactly what the text arm already gets.
- If built: render at decision rate only; cache the image in the trajectory store for replay (owner 12).

---

## 10. Contradictions with `docs/initial-plan.txt` relevant to this subtask

- The plan says "RGB/depth/segmentation sensors" are provided [plan]. True of gym-pybullet-drones (camera API), but the measured cost (3-15 ms per small frame vs ~10 us for a ray ring) makes images incompatible with the "thousands of flights" goal, so perception must be ray-based in the main experiment. [V measurement, consistent with plan's own goal of maximising throughput]
- "Jev gets the state and has to react": the plan never defines *what state*. With prompted text models the representation is an independent variable that can move results by large margins (FormatSpread), so "Jev vs others" claims are conditional on format unless ablated. [V/I]
- The plan's ~20 kHz / 80 drones claim is out of scope for me, but note: sensing at physics rate would add ~10-300 us per step of ray casting on a step that costs only tens of us, so raycasting at physics rate would itself become the bottleneck; throughput claims need sensing at decision rate only. [I based on measured 7.6 us step for a trivial world and ~10 us for 16 rays; agent 02 has the real step cost.]

---

## Decisions recommended

1. **Sensing**: 16-ray horizontal ring at 22.5 deg (range 10 m, Gaussian noise + dropout), cast at observation rate (20-50 Hz), not physics rate; depth images excluded from the main comparison (multimodal = optional side arm at 64x64).
2. **Shared `Observation` dataclass** as in Section 3; pure per-arm projections; ground truth isolated; explicit `None`s for lost sensors; separate RNG stream for sensor noise.
3. **Default text format = F (hybrid terse)**: body frame f/l/u, rounded numbers, 8-sector minima + precomputed `near/ttc/brg/d` (134 tokens, cl100k). Keep B (key=value, 138) as the readable fallback and A (JSON, 171) as the structured-provider variant; E (NL) and D (ASCII) as ablation arms only.
4. **Fixed static system prompt (~545 tokens) first, observation last** to maximise prefix-cache hits; no chain-of-thought in the Jev arm; `temperature=0`, `max_tokens=1-2`.
5. **Single-token wire aliases** for actions (or test if the real tokenizer makes names cheap), canonical names in DB. Verify with the actual tokenizer before freezing.
6. **Parser**: exact -> extracted-if-unique -> fail; never guess on ambiguity; no in-loop retries; fallback = hold-last-safe-action (TTL) -> reflex shield -> HOVER -> abort after M failures; log the failure taxonomy, report invalid/timeout/fallback rates and both raw and shielded collision rates.
7. **Format ablation is mandatory** (two-stage: offline labelled-observation screening with a mirror-symmetry test, then closed-loop on best/worst formats), and headline claims are stated per format or as a min/max range.
8. **Harness, not LLM, does arithmetic**: closing speed, ttc, bearing, distance, sector minima, extrapolation to decision time.
9. Put `age` in the prompt as an ablation factor; pre-compensated state as a fairness baseline for the classical arms.

## Open questions

1. What tokenizer/model is Jev (owner 01)? Token counts, single-token aliases, and whether `logit_bias`/grammar constrained decoding exists all depend on this.
2. Is the oracle label a single action or a set of acceptable actions per state (owner 11/06)? The ablation needs the set form for fair scoring.
3. Should the classical arms (FSM, PID+avoidance) see the same noisy rays, or ground truth? Recommended: same noisy rays (paired sensing), but confirm with 05.
4. Does the prompt expose numeric thresholds (battery < 25%, ttc < 1.5)? This interacts with tuning-effort parity against the FSM baseline.
5. How much does ray-casting cost with 100+ obstacles and the second drone present? Measured only 20 static boxes.
6. Can pybullet be reliably built on the arm64/macOS SDK for the user's environment without the zlib `fdopen` failure seen in a fresh 3.12 venv (and is a wheel available for 3.14)? Owner 02 succeeded in a 3.14 venv, presumably with a workaround; capture the exact recipe.
7. Where is decision age pre-compensation implemented (sensor layer vs provider wrapper), and does it count as giving the LLM an unfair "physics engine" in the prompt? Needs a ruling from 08.
8. Do hosted fast providers support `stop`, `logit_bias`, JSON-schema enums at negligible overhead via OpenRouter (owner 01)?

## Follow-up spikes (small, runnable experiments)

- **S1: Ray cost at scale.** Extend `bench_ray2.py` to 100/500 obstacles, moving bodies, and a 2-drone world; record p50/p95 for 16/72/360 rays; target: <50 us for 16 rays at 100 obstacles or document the knee. (30 min)
- **S2: Mirror-symmetry and frame test.** Script generating 500 observations + their left/right mirrored twins; run a local small model (MLX/llama.cpp Qwen 1.5-3B or Llama 3.2 1B) over formats A-F; report mirror-consistency and acceptable-action rate per format. Acceptance: any format with mirror inconsistency > 15% is dropped. (2-3 h, free)
- **S3: Output-format latency.** With the real provider (when a key exists), 300 calls each: plain name, single-token alias, JSON schema enum; report p50/p95 TTFT and total; decides Section 5.2. Reuse owner 01's script.
- **S4: Token-count re-measurement with the real tokenizer** (HF `tokenizers` for Qwen/Llama/Gemma) for encodings A-F and the 11 action names; replace the tiktoken proxy numbers.
- **S5: Parser fuzz test.** 10k synthetic model-output strings (markdown, punctuation, lowercase, multiple actions, JSON wrapper, refusal text); assert zero false-exact parses; keep as regression fixtures in `tests/` (only if tests are requested per repo conventions).
- **S6: Observation-rate sweep.** With gym-pybullet-drones, compute collision rate of the FSM arm as sensing rate varies {5, 10, 20, 50 Hz} to show whether 20 Hz suffices (affects ray cost accounting and the latency floor).
- **S7: Depth-image cost in the real env.** Confirm `getCameraImage` cost inside gym-pybullet-drones' own camera API (its `_getDroneImages`) vs bare pybullet; confirm memory stability over 1,000 frames.
