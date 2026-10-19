# 15c - GPT-family decision provider vs TypeSafe Jev: head-to-head evidence and decision matrix

Date: 2026-10-04. Author: research agent 15c. Scope: the question "how to use the GPT decisions API, and will it be a better alternative to Jev?" for the drone-decision experiment (11 discrete actions, harness-owned latency, closed-loop collision-vs-latency study). Another agent is pinning down what "GPT decisions API" literally is; this file only does the head-to-head. No API key exists and no account was created, so **nothing below is a live measurement by us**. Only unauthenticated public metadata (OpenRouter model/endpoint JSON) was queried directly.

Evidence tags: **[V]** fetched from a primary source (vendor docs, arXiv, OpenRouter API); **[3P]** third-party measurement or write-up; **[claim]** vendor claim, unverified; **[I]** my inference or arithmetic. Source URLs are in section 10.

---

## 0. Verdict up front

| Question | Verdict | One-line reason |
|---|---|---|
| (a) Real-time drone decisions | **Jev is better** (GPT-family is worse by roughly 1.4x to 5x on p50, 2x to 10x+ on p95) | Fastest verifiable GPT tiers sit at p50 about 0.6-1.1 s, p95 about 0.9-2.4 s with a 7.6 s tail seen in one report; Jev sits at p50 about 0.2-0.4 s, p95 about 0.45-0.7 s. At 2 m/s that is a d_min of 2.9-4.3 m vs 1.3-1.9 m (section 3). |
| (a') The **OpenAI Decisions API** specifically | **Cannot be assessed today; not a usable alternative** | Limited preview, no docs, no schema, no price, no published latency distribution. The only latency number is a vendor slide (150 ms vs 1.6 s). A standard key reportedly gets HTTP 403. |
| (b) The research comparison | **Complementary, not a replacement** | The headline is "Jev-class decision layer (0.1-0.4 s) vs LLM-class layer (1-3 s)". A GPT-family arm is exactly the LLM-class comparator (master plan arm F). Replacing Jev with GPT deletes the typed-decision arm and the claim. |
| (c) As a fallback | **Campaign-continuity fallback only; never an in-loop fallback** | In-loop fallback must be hold/shield/HOVER (a second 0.7 s+ API call after a timeout is already too late). As a "Jev API is down for the week" substitute it is fine, but its results are a different arm and must be labelled so. |

Decision matrix (section 6): Jev direct 7.4, Jev via OpenRouter 6.8, GPT with logprobs 5.9, GPT-6 Luna chat 4.9, OpenAI Decisions API 1.1 (out of 10, base weights). Jev wins in 97% of 20,000 random weight vectors and in 97% of runs with scores perturbed by plus or minus 1.5 points. The ranking is robust; the *absolute* gap is driven by latency evidence we have not yet reproduced.

---

## 1. What "GPT decisions API" can mean (head-to-head candidates)

| ID | Candidate | Status as of 2026-10-04 | Usable in a drone loop? |
|---|---|---|---|
| A | Jev via OpenRouter (`typesafe/jev-1.13`, `/api/alpha/decisions`) | Live; OR route labelled alpha | Yes (current plan, arm G) |
| B | Jev direct (`api.typesafe.ai/v1/systemone`, `jev-1.13.0`) | TypeSafe docs now say **Generally Available**, 80 req/s [V] | Yes |
| C | GPT-6 Luna chat completions, reasoning effort none, 1-token letter or strict enum | GA since 2026-09-22 [V] | Yes, as "fast LLM" |
| D | A GPT-family model that exposes logprobs: `gpt-4o-mini`, `gpt-4.1-nano` (OpenAI), `gpt-oss-120b/20b` (OpenAI open weights on Cerebras etc.) | Listed on OpenRouter with `logprobs`, `top_logprobs`, `logit_bias` [V] (gpt-4.1-nano shows no logprobs on OR but the Responses API reportedly supports it) | Yes, as "fast LLM with probabilities" |
| E | **OpenAI Decisions API** (constrained GPT-6 Luna) | Announced at DevDay 2026-09-29; "limited preview, selected customers"; no docs, schema, endpoint path, pricing, or rate limits published; `POST /v1/decisions` reportedly 403 "not enabled for this user" [V-sec] | **No** (cannot be called) |

How to "use" GPT today without the Decisions API: add a `ChatDecisionProvider` (already designed in doc 01 section 8) pointing at Luna or a logprob-capable model. Settings: reasoning effort `none`, `max_tokens` 1-4 (more if the model spends hidden reasoning tokens), static action-glossary prompt first and observation last, one-letter action codes A-K, `provider.order` pinned if via OpenRouter. No special "decision" parameter is needed. For the Decisions API, reserve a provider slot (`OpenAIDecisionsProvider`) and implement only when documentation and a key exist.

---

## 2. Head-to-head evidence

### 2.1 Latency (client-side wall time)

| Source | Setup | Jev p50 / p95 | GPT-family p50 / p95 | Tag |
|---|---|---|---|---|
| OpenRouter blog, "Jev vs LLM" (2026-09-19) | 60 tickets, 1 request each, via OR | 194 / 633 ms (injection test: 194 / 688) | **GPT Luna 1,106 / 2,395 ms** (inj. test 805 / 1,491); Claude Opus 1,957 / 2,594 | [V] vendor-adjacent |
| UTS paper arXiv 2609.22753 (Li, Wang, Gong, Lang, Yu) | Edge service orchestration, 8,280 requests, 33 conditions, 3 hosted LLMs + 2 self-hosted decision models | Jev median latency **22.7-64.5% lower than the fastest LLM**; barely moves with input size | n/a | [V] abstract only; full per-model table not read. Doc 01 earlier recorded 315-321 / 529-624 ms for Jev vs 382-435 / 686-836 ms for DeepSeek flash (10-37% gap) |
| navyaai.com first-party benchmark (client in India) | AG News, 200 cases, 1 constrained token | 357.7 / 454.9 ms | gpt-4.1-nano 621 / 893; gpt-4o-mini 675 / 870 | [3P] Jev 1.7-1.9x faster |
| wotai.co (Jev vs Haiku and others) | 150 prose passages, noul question | p50 455 ms | GPT-5.4-mini p50 934 ms; Haiku 4.5 631 ms | [3P] 1.4-2.1x |
| dchristopoulos/jev-aita | 770 AITA posts | median 0.39 s | GPT-5 nano minimal median 1.53 s (3.9x); Sonnet 2.46 s | [3P] |
| Firecrawl comparison blog | OR production traffic (provenance unclear) | P50 0.21 s, P95 0.34 s, P99 0.75 s | n/a | [3P], optimistic |
| OrcaRouter (2026-09-22 to 09-30) | GPT-6 Luna chat via their router | n/a | **median 699 ms, p95 7.6 s** | [3P] |
| eesel.ai author test (laptop) | 20 tickets, Luna and Sol | n/a | Luna 1.5-2.3 s | [3P] |
| Artificial Analysis, OpenAI first-party | Default 10k-token input, TTFT median | n/a | GPT-6 Luna non-reasoning TTFT **0.67 s** (0.74 on another AA page); GPT-5.4 nano non-reasoning 0.59 s; GPT-4.1 nano 0.62 s; **Luna low 1.59 s; Luna medium 5.28 s; xhigh 28.5 s** | [V] (10k prompt, so a 700-token prompt should be somewhat lower; unmeasured) |
| OpenAI DevDay slide | Decisions API vs standard Luna | 150 ms vs 1.6 s | n/a | [claim], no distribution, region, or conditions |
| RomanSlack/jev-drone | Direct, MuJoCo drone | p50 110-118 ms, p90 164 ms | n/a | [3P] best case |

Reading: a verifiable floor for "fast GPT" is **p50 about 0.6-0.7 s** (1-token output, non-reasoning, direct), with p95 about 0.9 s. With default or low reasoning, or via a router, p50 rises to 0.8-2.3 s and p95 to 1.5-7.6 s. Jev is **1.4-2x faster than the cheapest non-reasoning GPT tiers and 4-6x faster than GPT-6 Luna at default settings**. The Jev tail is shorter in absolute terms but still 3x its median.

Caveats: none of these are drone prompts; client regions differ (India vs US); several are vendor-adjacent; and Jev has no server-timing field, so no network/compute split exists. The ratio is robust (every independent source says Jev is faster), the absolute values are not.

### 2.2 Accuracy on typed-decision and classification tasks

| Task / source | Jev | GPT-family | Others | Tag |
|---|---|---|---|---|
| AG News, 4 classes, n=200 (navyaai) | **87.5-88%** | gpt-4.1-nano 1-token 71.0%, strict JSON 83.0%; gpt-4o-mini 83.0% / 82.0%; **gpt-5.6-luna strict JSON 82.0%** | self-hosted 400M model 75.5% | [3P], CI about plus or minus 6 pts |
| Prose-voice noul, n=150 (wotai) | 66.0% | GPT-5.4-mini 67.3% | Haiku 4.5 66.0%, Sonnet 5 71.3% | [3P] |
| Business category, n=149 (wotai) | 79.9% | n/a | Haiku 83.2% | [3P] |
| Commit type, n=100 (wotai) | 50.0% | n/a | Haiku 42.0% | [3P] |
| AITA verdicts, n=770 (jev-aita) | 75.4% top-1; Brier 0.369 | GPT-5 nano (low) Brier **0.480** (worse than base-rate 0.415) | Sonnet 5 76.9%, Brier 0.344 | [3P] |
| TypeSafe 4-workflow benchmark | 67.8% agreement | GPT-5.6 Terra 67.9%, Sol 74.1% | Opus 5 73.1% | [claim] workflows designed by TypeSafe, references from OpenAI/Anthropic models |
| OR blog support triage / injection | 98.3% / 100% | GPT Luna 98.3% / 97.5% | Opus 100% / 98.3% | [V] n=60 / 40 |
| eesel routing test, n=20 | n/a | Luna 40/40 queue accuracy | | [3P] |
| UTS paper live admission path | 0.91-0.95 of requests exact and on time at loads where LLMs fall below 0.1 | | | [V] abstract; it is a latency-deadline effect |

Reading: **accuracy is at parity within roughly plus or minus 5 points, with no stable rank**, and the rank flips by task (Jev wins AG News and commit type, Haiku wins category). None of these are numeric or spatial. **There is no published evidence for either family on drone-like state, so accuracy cannot be called from literature.** It must be measured (section 8).

### 2.3 Probability quality (logprob-derived vs native)

| Provider | Probability source | Evidence | Tag |
|---|---|---|---|
| Jev | Native per-option distribution; `confidence = (p_max - 1/n) / (1 - 1/n)`, derived from it; docs make **no calibration claim** | Raw ECE 0.080 on AG News, 0.037 after temperature scaling (T=5.17); ECE 0.121 on prose voice (Haiku 0.122, GPT-5.4-mini 0.192, Sonnet 0.062); on AITA, confidence correlated only r=0.06-0.24 with commenter agreement; coarse plateaus (e.g. 0.78/0.22/0) reported | [V] docs, [3P] numbers |
| GPT-4.1-nano / 4o-mini, letter logprobs | Token logprobs over A-K | Raw ECE 0.25-0.27 (nano) and 0.175 (4o-mini); extreme overconfidence (4o-mini 99.5% mean confidence at 82% accuracy); temperature-scaled best 0.055 and **0.026**, so after recalibration GPT-with-logprobs matched or beat Jev. Logprobs returned before logit_bias applies; near-duplicate tokens appear | [3P] |
| **GPT-5/6 family (incl. Luna)** | **No logprobs**: OpenRouter `supported_parameters` for every `gpt-5.x` and `gpt-6.x` model lists `seed`, `structured_outputs`, `reasoning_effort` but **not** `logprobs`, `top_logprobs` or `logit_bias` (I queried the public API); OpenAI forums say logprobs are unsupported on reasoning models and `top_logprobs` errors on GPT-5.4 | Only self-reported ("verbalised") probabilities are possible, costing about 10 extra output tokens and adding latency | [V] OR API, [3P] forums |
| OpenAI Decisions API | Unspecified. Press says "confidence score"; one article says probabilities are self-reported by the model and not calibrated | The OpenAI announcement does not say whether it returns option probabilities | [claim] / [3P] |

Reading: Jev's native probabilities are better out of the box than GPT logprobs (ECE 0.08 vs 0.18-0.27 raw) but not clearly better after post-hoc calibration. For this project calibration is done on our own labelled rollouts anyway (doc 01: thresholds must be calibrated on own data), so the gap narrows. The practical gap is that **the newest GPT tiers expose no probabilities at all**, which kills the confidence-gated shield fallback for Luna-class arms unless a legacy or open-weight GPT model is used.

### 2.4 Determinism and reproducibility

- Jev: docs state nothing on determinism [V]. Digitalapplied/others call it non-deterministic by design (returns distributions). TypeSafe documents **first-option bias** ("leans toward the option that comes first") and sensitivity to rewording [V jaggedness doc]. No temperature or seed exists.
- GPT: temperature 0 and `seed` are best effort. In jev-aita, a restarted GPT-5 nano temp-0 run reproduced only **64 of 103 answers (62%)** [3P]. Reasoning models do not expose temperature. OpenRouter lists `seed` for Luna but not logprobs [V].
- Neither is bitwise deterministic, so the plan's **model-call cache keyed on (provider, snapshot, prompt hash) with the recorded latency** is the real reproducibility mechanism for both. Provider-level determinism differences matter only for live runs.

### 2.5 Rate limits and throughput

| Provider | Limit | Tag |
|---|---|---|
| Jev (TypeSafe docs now) | 100K tokens/s, **80 req/s**, `429` on excess, "adjusting dynamically" (doc 01 recorded 40 req/s in docs and 20 req/s in secondary sources; the docs have been updated, plan on 16-20 via OR until we measure) | [V] |
| GPT-6 Luna (OpenAI docs) | Tier 1: 500 RPM / 500K TPM (8.3 req/s); Tier 2: 5K RPM; Tier 3: 5K RPM; Tier 4: 10K RPM; Tier 5: 30K RPM. Tier thresholds not verified | [V] |
| Decisions API | Unpublished | [V-sec] |

Little's law consequence (with L about 0.8 s for Luna): Tier 1 allows about 6.7 concurrent flights and about 1,200 flights/h at 25 calls per flight, which is no worse than Jev via OR (about 860-1,075 flights/h, about 5-6 concurrent flights). From Tier 2 up, GPT is CPU-bound, not rate-bound. Because the GPT arm is slower it makes fewer calls per flight (about 25 vs about 67), which offsets the stricter per-call limit. The measurement-quality cap of about 6 concurrent API flights (doc 13 B5) binds both.

### 2.6 Cost per decision (planning, 700-token prompt)

| Provider | $ / call [I] | $ per 1,000 flights at native cadence | $ per 10,000 flights | Tag |
|---|---|---|---|---|
| Jev ($0.042/M in, $0 out) | 0.00003 | about $2 (67 calls) | about $20 | [V] price |
| Luna chat ($0.10/M in, $0.50/M out, 5 output tokens) | 0.00007 | about $1.8 (25 calls) | about $18 | [V] price |
| Luna "fast" service tier (2x) | 0.00014 | about $3.6 | about $36 | [V-sec] |
| gpt-4o-mini, 1 token | 0.00011 | about $3.3 (31 calls) | about $32 | [V] OR price |
| Decisions API | unknown | unknown | unknown | not published |

Master plan's $8 per 1,000 flights assumed 200 calls per flight (L at or below 100 ms). **Cost is a non-factor** at either provider (under $50 for a 10k-flight arm). TypeSafe states it "cannot prove the price is not subsidised" [3P]; budget a 5-10x price increase and it is still under $300.

### 2.7 Structured-output reliability

- Jev: schema-valid by construction. The "0% structured output error rate" figure is vendor-reported and TypeSafe itself calls it a design guarantee, "not empirical" [claim]. The same vendor benchmark reports 45.5% structured-output errors for Haiku 4.5 and 17% tool-call failures for GPT-5.6 Sol; **treat as marketing until reproduced**.
- GPT: Luna supports `structured_outputs` (strict enum) [V]; navyaai measured zero unparseable responses in 600 strict-JSON calls [3P]. One-token letter outputs are trivially parseable. Reasoning models can return empty content if `max_tokens` is too small.
- Both: "a type-valid answer can be wrong, and that failure is silent" [3P]. Parity for our purposes; the parser and fallback chain in doc 07 cover both.

### 2.8 Numeric and spatial weakness

- Jev's own jaggedness doc [V]: does not count reliably, cannot judge proximity between numeric values, worse with hex/RGB-style numeric encodings, **accuracy falls as irrelevant state grows**, first-option bias. This matches the master-plan rule that the harness precomputes TTC, bearing, sector minima and sends buckets.
- GPT-family: no head-to-head numeric/spatial benchmark exists for the two. General expectation [I] is that a larger non-reasoning LLM tolerates raw numbers better than Jev, but with reasoning off (required for latency) the advantage is unproven. The doc 07 format ablation is the instrument that settles this; it must run on **both** providers.

### 2.9 Maturity, lock-in, terms

| Item | Jev | GPT chat (Luna) | Decisions API |
|---|---|---|---|
| Age | 19 days (launched 2026-09-15); 1.13 snapshot `jev-1.13-20260917` | Luna 12 days (2026-09-22); GPT family multi-year platform | Announced 5 days ago |
| Status | Direct: GA per docs; OpenRouter route: alpha; earlier sources still say waitlist or early access (contradiction) | GA, SLA-style tier system | Limited preview |
| Pinning | Pin `jev-1.13.0` / dated snapshot; `jev-latest` and `jev-preview` aliases move | Dated snapshot on OR (`gpt-6-luna-20260922`); OpenAI deprecates older models (AA notes gpt-4.1-nano deprecated) | n/a |
| Lock-in | **Low**: System-One request shape now served by Ollama (`nimble`, `tev1`), vLLM/Red Hat; local stand-in possible [V-sec] | **Lowest**: chat-completions is a de facto standard | Highest (undocumented proprietary endpoint) |
| Pricing risk | Subsidy acknowledged | Stable list price | Unknown |
| Terms for benchmark publication | **Unverified** (TypeSafe contracts unreadable per doc 01) | **Unverified** (OpenAI terms page returned 403 to my fetcher; I could not read any benchmarking clause) | Unverified |

Do not use or cite: decisions-api.net, opendecisionsapi.com, jevtypesafeai.com, thejevai.com, pasqualepillitteri.it and similar SEO-style pages as evidence beyond "the preview exists"; several are affiliate or lead-generation pages and contradict each other on details (e.g. whether a confidence score is returned).

---

## 3. Reaction-budget impact (d_min table from doc 13 section 2.3, extended)

Formula (doc 13): `d_min(L) = d_min0(v) + v x L_eff`, `d_min0` = 0.50 / 0.95 / 1.66 m at 1 / 2 / 3 m/s, `L_eff = 1.5 x call latency` (sequential cadence). Reflex row uses 0.03 s. Computed by me from the latency evidence above.

| Reactor | p50 / p95 call latency (s) | L_eff p50 / p95 (s) | d_min at 1 m/s (p50 / p95) | **at 2 m/s (p50 / p95)** | at 3 m/s (p50 / p95) |
|---|---|---|---|---|---|
| Reflex (PID/FSM, 1-2 ticks) | - | 0.03 | 0.53 | **1.01** | 1.75 |
| Jev direct, jev-drone best case | 0.115 / 0.20 | 0.17 / 0.30 | 0.67 / 0.80 | **1.29 / 1.55** | 2.18 / 2.56 |
| Jev via OR, planning (doc 13) | 0.25 / 0.60 | 0.375 / 0.90 | 0.88 / 1.40 | **1.70 / 2.75** | 2.79 / 4.36 |
| Jev via OR, UTS paper | 0.318 / 0.560 | 0.48 / 0.84 | 0.98 / 1.34 | **1.90 / 2.63** | 3.09 / 4.18 |
| Jev, India client | 0.358 / 0.455 | 0.54 / 0.68 | 1.04 / 1.18 | **2.02 / 2.31** | 3.27 / 3.71 |
| OpenAI Decisions API, **vendor claim only** (p95 0.40 assumed by me) | 0.15 / 0.40 | 0.23 / 0.60 | 0.72 / 1.10 | **1.40 / 2.15** | 2.33 / 3.46 |
| gpt-4.1-nano / 4o-mini, 1 token (navyaai) | 0.65 / 0.88 | 0.98 / 1.32 | 1.48 / 1.82 | **2.90 / 3.59** | 4.58 / 5.62 |
| GPT-6 Luna chat, planning (none effort) | 0.80 / 1.8 | 1.20 / 2.70 | 1.70 / 3.20 | **3.35 / 6.35** | 5.26 / 9.76 |
| GPT-6 Luna chat, OR blog | 1.106 / 2.395 | 1.66 / 3.59 | 2.16 / 4.09 | **4.27 / 8.13** | 6.64 / 12.44 |
| GPT-6 Luna chat, OrcaRouter tail | 0.699 / 7.6 | 1.05 / 11.4 | 1.55 / 11.9 | **3.05 / 23.8** | 4.81 / 35.9 |
| Slow LLM, p50 about 2 s (doc 13) | 2.0 | 3.0 | 3.50 | **6.95** | 10.66 |

What this changes in the doc 13 reading:

1. The fastest realistic GPT tier needs **1.2-1.9 m more spawn distance than Jev at 2 m/s at p50** (2.9-3.35 vs 1.7) and **about 1 m to 5 m more at p95**. That is a wider Jev-vs-fast-LLM window than the master plan assumed. The plan's fast-LLM row used 400 ms p50 (L_eff 0.6, d_min 2.15 m); the best verifiable GPT is 0.65 s, and Luna at default is slower still. If instead the "fast LLM" is a Groq or Cerebras gpt-oss model at 250-600 ms [I, unmeasured], the plan's row stands and the gap stays small.
2. The window where Jev survives but a fast GPT does not is **about 1.7-3.4 m at 2 m/s**, comfortably inside the 10 m sensing range. This is the region where the paper-style claim has signal.
3. The p95 tail dominates for GPT. A Luna chat tail of 7.6 s would put d_min above 20 m at 2 m/s; those calls should be counted as timeouts under the harness's fixed deadline rule (doc 07/08), not retried.
4. If the Decisions API claim (150 ms) were true and stable, it would sit between Jev direct and Jev via OR and would be a real competitor on latency. That is the single thing most worth checking when access opens (section 8, step 1).

---

## 4. Does the existing plan already contain a GPT arm?

Yes. Master plan section 5.1 arm F is "Fast chat LLM (hundreds of ms) and slow LLM (1-3 s)", and doc 09 models the slow arm as Haiku 4.5 or Sonnet 5.5. The Phase 0 benchmark in doc 01 section 9 already contains fast chat targets. What the new evidence changes:

- **Name the GPT family explicitly** as the F family: it gives one vendor's full latency ladder (Luna none about 0.7-1.1 s, Luna low about 1.6 s TTFT, Luna medium about 5.3 s TTFT) and removes the "different vendors" confound for the fast/slow comparison.
- **Probabilities are not available on the newest GPT tiers**, so the F-fast arm either runs argmax-only (no confidence gating, a documented limitation) or uses a logprob-capable GPT-family model.
- **A Decisions API slot is the cleanest test of the master plan's biggest methodological weakness** (risk: Jev vs LLM are two API shapes, not two models). If access opens, Luna chat vs Luna Decisions on the same model isolates "decision-API architecture" from "model", which no Jev-vs-X comparison can do. High scientific value, zero cost to reserve the adapter, but do not schedule work against it.

---

## 5. Where GPT is genuinely better

- Setup effort: standard chat-completions, `structured_outputs`, documented, no alpha path to verify.
- Maturity, SLA tiers, a stable list price, dated snapshots, wider ecosystem; no subsidy risk.
- Throughput headroom at Tier 2+ (5K+ RPM) once spend tiers are reached.
- Multimodal input (images) for the optional vision side arm; Jev is text-only.
- Tolerance of richer or noisier state (unproven but likely).
- Possible numeric competence with reasoning on, which is useless at drone latency.

## 6. Decision matrix for this project

Scores 0-10 (higher is better). Evidence-based, scored from the tables above; subjective scores are explained in the notes column. "Reaction-budget impact" is computed: `score = 10 x (1 - (d_min_2m/s - 1.0)/8)` clamped, averaged over p50 and p95 (values: Jev OR 8.4, Jev direct 9.0 averaging two latency sources, Luna 4.0 blending three Luna sources, GPT-logprob models 7.2, Decisions API 0 on evidence).

Base weights sum to 100. Latency and reaction-budget are correlated by construction; sensitivity below treats them separately and together.

| Criterion | Weight | A Jev via OR | B Jev direct | C GPT-6 Luna chat | D GPT with logprobs (4o-mini, 4.1-nano, gpt-oss) | E OpenAI Decisions API (evidence today) | Notes |
|---|---|---|---|---|---|---|---|
| Latency p50/p95 | 10 | 7 | 9 | 3 | 5 | 1 | Section 2.1. E would be about 9 if the 150 ms claim held |
| Reaction-budget impact (d_min) | 15 | 8.4 | 9.0 | 4.0 | 7.2 | 0 | Section 3 |
| Probability quality | 10 | 6 | 6 | 3 | 5 | 1 | Native vs logprob vs none; section 2.3 |
| Determinism / reproducibility | 10 | 5 | 5 | 3 | 5 | 1 | Neither deterministic; cache is the mechanism |
| Cost over campaign sizes (doc 09) | 5 | 9 | 9 | 8 | 8 | 4 | All under $50 per 10k flights |
| Rate limit / throughput | 10 | 6 | 8 | 7 | 7 | 0 | Section 2.5 |
| Setup effort | 10 | 5 | 5 | 8 | 7 | 0 | A and B need an unrun alpha path and a second key |
| Maturity / risk | 10 | 4 | 5 | 7 | 5 | 0 | 19 days old, subsidy, OR route alpha |
| Scientific value as the sole decision provider for the paper-style claim | 20 | 9 | 9 | 4 | 5 | 3 | Replacing Jev removes the typed-decision arm; E would be 7-8 if usable |
| **Weighted total (out of 10)** | 100 | **6.81** | **7.40** | **4.90** | **5.88** | **1.10** | |

Ranking: B > A > D > C > E.

Interpretation caveat: B scores at or above A on every criterion because B's inputs (jev-drone direct latency, docs-stated 80 req/s) are the most optimistic and least reproduced; B beating A in 91% of score-noise runs reflects my assumption, not evidence. The firm result is Jev (either route) over GPT.

### 6.1 Sensitivity analysis

Scripts were run in scratch (Python, numpy; not saved to the repo).

| Test | Result |
|---|---|
| Alternative weight scenarios | Real-time (latency and reaction doubled, sci 5): A 6.97, B 7.80, C 4.45, D 6.11. Research (prob/det/sci heavy): A 7.19, B 7.50, C 4.35, D 5.57. **Fallback/ops (setup, maturity, throughput, cost, determinism heavy, sci 0): A 5.62, B 6.30, C 6.15, D 6.11 - GPT chat is within 0.2 of Jev direct and overtakes Jev via OR.** Equal weights: A 6.60, B 7.22, C 5.22, D 6.02. No sci value: A 6.26, B 7.00, C 5.12, D 6.10 |
| One-at-a-time, each weight x0.5 and x1.5 | Ranking B, A, D, C, E in all 18 cases |
| 20,000 random Dirichlet weight vectors | Winner share: B 97.3%, C 1.8%, D 0.9%, A 0%, E 0%. P(A beats C) 92%, P(A beats D) 85%, P(B beats A) 100% (reflects my score assumptions) |
| Weights randomised x0.5-1.5 and every score perturbed by plus or minus 1.5 (20,000 runs) | P(A beats D) 97%, P(A beats C) 99.995%, P(D beats A) 3% |
| Pessimistic for Jev (A latency 5, reaction 6) and optimistic for GPT-logprob (latency 6, reaction 8) | A 6.25, D 6.10 (near tie) |
| Same plus D sci value 7 and A probability 5 | A 6.15, **D 6.50 (D wins)** |
| Decisions API if the vendor claims hold and it is accessible (latency 9, reaction 9, probability 3, determinism 3, cost 4, throughput 3, setup 2, maturity 2, sci 7) | 5.15, still below A 6.81 and B 7.40, mainly because maturity, setup and rate limits are unknown. It would win on latency alone |

Conclusion: Jev leads under every weighting except the pure operations view, where GPT chat ties; D overtakes A only when the evidence on Jev latency is degraded and D's scientific value is raised, i.e. precisely the two things the 1-hour protocol measures. The score uncertainty is dominated by latency evidence (section 7), not by weights.

---

## 7. What cannot be known without live calls

1. Actual p50/p95/p99 from **this machine, this region, this time**, for Jev (OR and direct) and each GPT target with our 700-token prompt. All published numbers differ by 2-3x between sources (Jev p50 0.11-0.46 s; Luna 0.7-2.3 s).
2. Whether Jev handles an 11-option Choice plus auxiliary questions with our encoding and how accurate it is on drone states. No published evidence exists for numeric or spatial state on either family.
3. Whether the OR/direct gap is real (doc 01 estimates 50-200 ms).
4. Real rate-limit behaviour (TypeSafe's 80 req/s docs vs 20 req/s secondary) and OpenAI's tier thresholds.
5. Determinism: repeat-call agreement and the size of Jev's option-order effect on our 11 actions.
6. Whether logprobs work on OpenAI direct for Luna or GPT-5.4 nano (OR says no, community reports mixed).
7. Whether the OpenAI Decisions API can be called at all, its schema, and whether it returns probabilities.
8. Terms for publishing benchmark results (TypeSafe, OpenAI): needs a human reading the contracts.
9. Quality of Jev calibration on our data: only reliable after the labelled set.

## 8. Exact 1-hour measurement protocol (needs keys; none exist today)

Prerequisites (before the hour, 10 min, outside the budget): `OPENROUTER_API_KEY` and `TYPESAFE_API_KEY` and an `OPENAI_API_KEY` in `.env`, Python 3.14.2 env, the doc 01 section 9 script extended with the targets below, laptop on AC, load average under 2, `caffeinate -i`. The same 700-token prompt for every target: 544-token static system prompt (action glossary, priority rules, 2 examples) first, 134-token observation last, one-letter action codes for chat models, 11-option Choice question for Jev. Record region and UTC time in a manifest. Do not use `:free` models. Cache disabled. No retries inside timed calls. Timeout 15 s counted as an outcome.

Targets (all pinned, no silent fallbacks):
T1 Jev via OR (`typesafe/jev-1.13`); T2 Jev direct (`jev-1.13.0`); T3 GPT-6 Luna reasoning none, 1-token letter, OpenAI direct; T4 same via OR; T5 gpt-4o-mini (or gpt-4.1-nano) with `logprobs`/`top_logprobs`, 1 token; T6 gpt-oss-120b on Cerebras via OR with logprobs; T7 (optional) Luna reasoning low as the slow-arm probe; T8 Decisions API probe: one `POST /v1/decisions` with the key, record the status and body only (do not retry, do not sign up for anything).

| Minute | Step | Calls and output |
|---|---|---|
| 0-5 | **Smoke and shape.** One call per target; save raw responses; assert pinned snapshot id on every response; check whether T3 and T4 reject `logprobs` (expected 400); T8 expected 403 | 8 calls; pass/fail table |
| 5-25 | **Latency, closed loop.** Per target 5 warm-up plus 200 sequential calls, one in flight, targets interleaved round-robin in random order each round, persistent HTTP/2 client in a separate process from nothing else running. Then 200 calls at concurrency 4 | About 1,600 sequential plus 800 concurrent calls, under $0.50. Report p50/p95/p99/max with bootstrap CIs, error and 429 counts, ECCDF |
| 25-30 | **Rate-limit ramp.** For T1/T2 and T3 only: 1 to 32 req/s over 3 minutes, stop at the first sustained 429 | Max sustained req/s per target |
| 30-48 | **Accuracy and calibration on labelled scenes.** 300 scenes labelled by the doc 06/11 point-mass viability labeller (safe-action set per scene), balanced across tiers T0-T3, with the mirror-symmetry transform for half | 300 calls per target, 4 concurrent. Metrics: acceptable-action rate with Wilson CI (plus or minus about 5.5 points at n=300), BRAKE/HOVER/AVOID confusion, Brier and ECE of provider probability against the event "chosen action is in the safe set" (for T3/T4/T8 with no probabilities, report argmax accuracy only), always-hover and random floors |
| 48-55 | **Determinism and order bias.** 20 scenes x 25 repeats per target (500 calls): argmax agreement rate, per-action probability standard deviation. For Jev, plus 20 scenes x 4 option shuffles | Agreement %, option-order effect size |
| 55-60 | **Wrap-up.** Parse failures, empty outputs, 4xx/5xx, tokens and `usage.cost` per call, extrapolate the campaign cost; write the manifest | One summary table |

Pre-declared decision rules:

- **GPT-fast is a valid arm F** if p95 is at or below 2 s and parse failures under 1%.
- **Jev is the latency-class arm** if its p50 is at most 0.5x the best GPT p50 AND p95 at most 0.8x. If the ratio is above 0.77 (less than 1.3x faster), the "Jev vs fast LLM" claim must be reframed as a small-gap result.
- **Gate G3** (doc 13): Jev p50 at most 400 ms, p95 at most 1 s, error rate under 1%, at least 10 req/s sustained. If failed, run Jev at recorded latency only, local stand-in for sweeps.
- **Gate G4**: acceptable-action rate at least 0.7 on clear-threat states, with CI above always-hover and random. Applies to every provider; a failing GPT arm is reported as a measured weak point, not tuned in the hour.
- **Decisions API**: if T8 returns anything but 403 or 404, run steps 5-25 and 30-48 on it only after reading whatever schema it returns; otherwise log "not accessible 2026-10-xx" and move on.
- Run the protocol a second time on another day or time-of-day block (20 min, latency step only) before publishing any latency number, as in doc 01.

Falsifiable predictions (so the hour can embarrass this document): H1 best verifiable GPT p50 at least 0.6 s; H2 Jev p95 at most 0.7 s via OR; H3 Luna rejects logprobs on both routes; H4 Decisions API returns 403; H5 GPT acceptable-action rate within plus or minus 10 points of Jev. If H1 or H2 fail, section 3 and the matrix change.

---

## 9. Recommended arm lineup (master plan section 5.1 amended)

| Arm | Provider | Role | Change vs master plan |
|---|---|---|---|
| G | Jev via OpenRouter, pinned `typesafe/jev-1.13` | Primary decision-layer arm | none |
| G2 | Jev direct, `jev-1.13.0` | **OR-vs-direct latency control**, optional, cheap | now documented GA, 80 req/s; promote from "optional" to "run if G3 shows OR p50 above 250 ms" |
| F-fast | **GPT-6 Luna, reasoning none, 1-token letter** (argmax only) | Fast LLM comparator, probably 0.6-1.1 s | name the family; label "no probabilities, confidence gating disabled for this arm" |
| F-fast-p | gpt-oss-120b (Cerebras) or gpt-4o-mini with logprobs | Fast LLM with probabilities, calibratable | add as an ablation, only if the hour shows p95 at or below 1 s |
| F-slow | **GPT-6 Luna reasoning low** (about 1.6 s TTFT) and optionally Haiku 4.5 / Sonnet 5.5 as in doc 09 | Slow LLM, 1.6-3 s regime | single-vendor ladder; Luna medium (5 s) belongs only in a "far too slow" tier |
| X | OpenAI Decisions API | Same-model constrained vs unconstrained test | **placeholder adapter slot only; no work until docs and a key exist** |
| Local | Ollama `/v1/systemone` stand-in | Insurance vs Jev API changes | none |

Everything else (A, A', B1, E, Z0-Z3, O/O2, shield S0/S1, latency injection, real-time validation) is unchanged. Claims ledger updates: replace "Jev is 10-37% faster than a flash LLM" by the arm-specific figures ("1.4-1.9x vs cheap non-reasoning GPT tiers, about 4-6x vs GPT-6 Luna at default settings, per third-party tests, to be replaced by our own measurement"); never write "GPT is slower than Jev" without naming the model, effort and route.

---

## Decisions recommended

1. Keep Jev as the primary decision-layer arm; **do not replace it with a GPT-family provider**. The head-to-head favours Jev on latency, reaction budget, native probabilities and scientific fit, and ties on cost and structured-output reliability.
2. Adopt GPT-6 Luna (reasoning none) as the F-fast arm and GPT-6 Luna low as F-slow, giving a one-vendor latency ladder; keep Haiku as an optional cross-vendor check.
3. Treat the OpenAI Decisions API as **not available**: reserve a provider slot and ignore it until it has docs, a schema, a price and a key. Do not cite the 150 ms figure as anything but a vendor claim.
4. Do not use any GPT model as an in-loop fallback after a Jev timeout; use hold-last, shield, HOVER (doc 07). A GPT substitute is allowed only for campaign continuity and must be reported as a different arm.
5. Run the 1-hour protocol (section 8) before Phase 0 gates G3/G4 are declared; update the d_min table in doc 13 with the measured numbers.
6. Update the master plan facts: TypeSafe docs now say GA and 80 req/s (the OR route is still alpha); newest GPT tiers expose no logprobs; Luna latency evidence (0.7-2.3 s, tail 7.6 s).

## Open questions

1. Which provider does the user mean by "GPT decisions API" (OpenAI Decisions API, or "GPT-class chat as a decision provider")? The other agent is resolving this; the answer changes only whether slot X is wanted.
2. Is a text-only action selection from a GPT-family model with no probabilities acceptable for arm F, or must F expose probabilities (forcing gpt-oss/4o-mini)?
3. Which client region will the experiment run from? It moves every number by 150-250 ms and affects Jev (US West origin) and OpenAI differently.
4. Do TypeSafe's or OpenAI's terms permit publishing benchmark results? Unread contracts; needs a person.
5. What OpenAI usage tier will the account have? Tier 1 (500 RPM) is adequate for the planned 500-1,000 paired flights but not for sweeps.
6. Does Luna on OpenAI direct accept `logprobs` at reasoning effort none, and does `top_logprobs` work?
7. Is the Decisions API's output a probability vector, a single label with a self-reported confidence, or only a label?
8. Does the UTS paper's full table (which three hosted LLMs, per-model p50/p95) contradict the 22.7-64.5% abstract figure, and is the 10-37% DeepSeek-flash gap from doc 01 the same dataset?
9. Is the Jev OR route still alpha after TypeSafe's docs moved to GA?

## Follow-up spikes

1. **S1 (1 h, needs keys):** the section 8 protocol, producing a single results table for T1-T8.
2. **S2 (0.5 day):** extend the doc 01 section 9 script with T3-T8 and a logprob-extraction path; unit-test against the existing mock server.
3. **S3 (1-2 h):** read the full text of arXiv 2609.22753 (tables for the three hosted LLMs) and the "Jev in the Wild" action-selection section for any control-like workloads.
4. **S4 (2 h):** option-order and rewording sensitivity of Jev on the 11-action question using the doc 07 encodings; decide the canonical action order and whether to rotate it as a factor.
5. **S5 (1 h):** read TypeSafe and OpenAI terms on benchmarking and publication; record the answer in the claims ledger.
6. **S6 (watch):** poll the OpenAI API reference and changelog weekly for the Decisions API; when it opens, run the protocol on it only (steps 5-25, 30-48).
7. **S7 (0.5 day):** re-derive the d_min table with the measured latency distributions and the real executor's brake lookup, and re-run the sensitivity script with the measured scores.

---

## 10. Sources (accessed 2026-10-04)

- UTS paper abstract, arXiv 2609.22753: https://arxiv.org/abs/2609.22753
- OpenRouter blog, "Jev vs LLM": https://openrouter.ai/blog/tutorials/jev-vs-llm-when-to-use-each/
- OpenRouter public model and endpoint APIs (queried directly, no key): https://openrouter.ai/api/v1/models, https://openrouter.ai/api/v1/models/openai/gpt-6-luna/endpoints, https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints (both report `latency_last_30m` null when unauthenticated, so OR provider latency stats could not be read)
- TypeSafe docs: https://docs.typesafe.ai/models.md, https://docs.typesafe.ai/confidence.md, https://docs.typesafe.ai/model-jaggedness/jev-1.13.md
- Artificial Analysis: https://artificialanalysis.ai/models/gpt-6-luna-non-reasoning/providers, https://artificialanalysis.ai/models/gpt-6-luna-non-reasoning (and search snippets for gpt-6-luna medium/low/xhigh, gpt-5-4-nano-non-reasoning, gpt-4-1-nano)
- OpenAI GPT-6 Luna model doc: https://developers.openai.com/api/docs/models/gpt-6-luna.md
- OpenAI Decisions API coverage: https://modelsystem.one/runtimes/openai-decisions-api/, https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev, https://www.eesel.ai/blog/openai-decisions-api-pricing, https://www.orcarouter.ai/de/blog/openai-decisions-api-gpt-6-luna, https://alphasignal.ai/news/openai-s-decisions-api-gives-developers-a-constrained-gpt-6-luna-router, https://ai-tldr.dev/releases/openai-decisions-api/, https://huggingface.co/blog/a2aprotocol/openai-decisions-api-vs-jev-a-practical-guide, search result for the OpenAI Developers post https://x.com/OpenAIDevs/status/2105003318917697873 (fetch returned 402)
- Independent Jev benchmarks: https://www.navyaai.com/blog/jev-typesafe-limitations-production, https://wotai.co/blog/typesafe-jev-vs-claude-haiku-tested, https://github.com/dchristopoulos/jev-aita, https://www.digitalapplied.com/blog/typesafe-jev-system-one-model-typed-decisions, https://www.datacamp.com/blog/system-one-models-jev, https://rywalker.com/research/typesafe
- OpenAI logprobs availability discussion: https://community.openai.com/t/logprobs-deprecated-for-gpt-5-models/1355427 and related threads
- Not retrievable: OpenAI terms of use (403), Medium Laya article (403), OpenAI Developers post (402)
- Repo context: docs/MASTER-PLAN.md, docs/research/01, 04, 07, 08, 09, 13
