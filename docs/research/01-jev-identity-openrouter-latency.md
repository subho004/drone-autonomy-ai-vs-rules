# 01 — Jev identity + OpenRouter/API latency realities

Research agent 01. Access date for all web sources: **2026-09-30** (today). Hardware for local measurements: this M4 MacBook Pro, macOS (Darwin 27.0.0), Python 3.14.2 in a scratch venv, httpx 0.28.1.
Legend: **[V]** verified from a primary source I fetched; **[V-secondary]** reported by a vendor/third party I could fetch but not independently reproduce; **[I]** my inference; **[U]** unverified.

---

## 0. TL;DR

1. **"Jev" is almost certainly TypeSafe AI's Jev 1.13, a "System One" typed-decision model** (confidence ~0.9 that this is what the user means). It is live on OpenRouter as `typesafe/jev-1.13` (alias `~typesafe/jev-latest`). I confirmed this in OpenRouter's own public model API, not just in blog posts. **The initial plan's "Jev" was an unspecified black box; the real Jev is not a chat LLM and does not fit the chat-completions API** (details in section 1).
2. **Jev's API shape changes the design.** You send a JSON `state` plus typed `questions` (Choice with up to 255 options, Score 2-10 levels, Noul = yes/no probability). You get back a chosen option plus a **probability per option**, in one parallel pass, with no text generation. There is no `max_tokens`, `temperature`, `seed`, `stop`, `logprobs`, or streaming. The 11 drone actions map naturally onto one Choice question, and side questions (risk score, "obstacle imminent?") ride along in parallel.
3. **Latency reality (the numbers to plan with).** Vendor claim: 70-500 ms end to end, most about 100 ms from US West [V-secondary]. Independent measurements through OpenRouter: median about 315-320 ms, p95 about 530-625 ms (arXiv 2609.22753); OpenRouter's own blog: p50 194 ms, p95 633-688 ms. A public jev-drone project reports median 110-118 ms, p90 164 ms. **Plan on p50 200-320 ms and p95 530-700 ms via OpenRouter; about 110-120 ms p50 is the best published case.** Not the "tens of ms" that a simulator at 240 Hz makes you want. Geography (if you are in Asia) can add 150-250 ms [I].
4. The **10 Hz to 20 Hz closed-loop ambition is out of reach for a single sequential Jev call chain** (7.4 Hz sequential ceiling reported by jev-drone). Pipelined/concurrent calls reach 21 decisions/s but each decision is still about 120-300 ms stale when applied.
5. **"Jev vs slow LLM" gap is real but smaller than the plan implies when the LLM arm is a fast small model.** Big LLMs via API: 1-2.6 s p50 (OpenRouter blog). A tiny/fast chat model on Groq/Cerebras with `max_tokens` of a few tokens can plausibly land at 150-500 ms [I, unmeasured]. The latency "story" must pick its LLM arms deliberately (see section 6).
6. **The plan works whether or not Jev is resolved.** Section 8 gives a `DecisionProvider` Protocol (mypy-clean sketch), a latency-injecting decorator, and stand-ins; section 9 is a ready-to-run benchmark script (dry-run validated against a local mock; needs only `OPENROUTER_API_KEY`). No key was used or exists here.

---

## 1. What is Jev?

### 1.1 Verdict

| Claim | Status | Confidence |
|---|---|---|
| "Jev" = **Jev 1.13 by TypeSafe AI**, first "System One" model; typed decisions with probabilities, non-autoregressive, API-only | [V] (OpenRouter public API + TypeSafe docs + arXiv) | **0.9** that this is the Jev the user means |
| Available on **OpenRouter** as `typesafe/jev-1.13` (provider "TypeSafe"; response model id `typesafe/jev-1.13-20260917`) | [V] `GET https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints` returned it (I called it myself, no key) | 0.99 |
| User's "Jev" might be something else (a private model, a different "Jev") | [U] | 0.1; no evidence of any other AI named Jev beyond community clones |

Why 0.9 and not 1.0: the initial plan is chat-derived and never says who makes Jev. The description "fast, low-latency decision model reachable via OpenRouter or directly from Jev" matches TypeSafe's Jev on every count (low latency, discrete decisions, OpenRouter + direct API). Nothing in the repo says otherwise. **Action: ask the user to confirm it is TypeSafe's Jev.**

### 1.2 Verified facts

- **Maker**: TypeSafe AI (San Francisco), founders Diogo Almeida (ex-OpenAI, credited as RLHF/ChatGPT co-creator), Erik Gafni, Sasha Sheng. Launched **Sept 15, 2026** in early access; $40M seed reported [V-secondary: OpenRouter blog https://openrouter.ai/blog/insights/what-is-jev/; flaviocopes.com/jev; mindstudio.ai blog]. The plan's premise that Jev is a recent, hyped model is consistent.
- **Product nature**: a **System One** model. "Jev evaluates typed *questions* against a *state* and returns structured results directly" [V: https://docs.typesafe.ai/introduction.md]. It **does not generate text, reasoning, or explanations**. Text-only input (no images). Proprietary; **no published weights** [V: OpenRouter blog]; API-only, no self-hosting [V-secondary].
- **Architecture claim**: non-autoregressive, "samples all the answers in parallel, each question evaluated independently", trained with RL for calibrated decisions [V-secondary: flaviocopes.com]. Not independently verifiable.
- **On OpenRouter** [V, my own API calls, 2026-09-30]:
  - `typesafe/jev-1.13`: modality `text->decisions`, created 2026-09-18 00:01 UTC, context 32,000, single provider "TypeSafe", input **$0.042 per M tokens**, output $0, `max_completion_tokens` 28,800, uptime 100% (30m/5m/1d), `supported_parameters: []` (no sampling parameters), `latency_last_30m`/`throughput_last_30m` null when unauthenticated.
  - `~typesafe/jev-latest`: alias, no endpoints listed itself. Docs say the alias moves when a new release ships; **pin `typesafe/jev-1.13`** (it resolves to the dated snapshot `jev-1.13-20260917`; log the `model` field of every response).
  - **It does not appear in the normal `/api/v1/models` list** (464 models, none matching "jev" except the router below). It appears only when querying `output_modalities=all` (637 models). Anyone scanning the normal chat model list will miss it. This is why "check OpenRouter's model list" can look like a dead end.
  - **`typesafe/jev-router` is a different product**: an LLM router (chooses which chat model and reasoning effort to use), `text+image+file+audio+video->text`, pricing "-1" (variable). Do not confuse it with Jev itself. It is not a low-latency decision model [V: model API + https://x.com/OpenRouter/status/2103610898690855161 title].
- **Two first-party surfaces**:
  - OpenRouter: `POST https://openrouter.ai/api/alpha/decisions` (status "alpha"; the docs also list `POST /api/v1/systemone`) [V: https://openrouter.ai/docs/guides/community/jev]. Note the reference page prints the path as `/api/v1/api/alpha/decisions`; this is almost certainly a docs typo [I]. The script in section 9 uses `/api/alpha/decisions` per the tutorial/curl example, **unrun against the live service**.
  - TypeSafe direct: `POST https://api.typesafe.ai/v1/systemone`, model id `jev-latest` / `jev-preview` / pinned `jev-1.13.0`, Bearer key; Python and JS SDKs with retries [V: https://docs.typesafe.ai/api.md, /models.md, /sdk.md]. Also reported on Cloudflare Workers AI (`typesafe/jev`) and Vercel AI Gateway [V-secondary].
- **Independent papers/projects that use Jev** (all within 2 weeks of launch; treat as early evidence):
  - arXiv 2609.22753 "Replacing Large Language Models with Jev Decision Models for Low-Latency Edge Service Orchestration" (Li, Wang, Gong, Lang, Yu; submitted 2026-09-19, revised 09-26; no TypeSafe affiliation mentioned). Measured through OpenRouter's Decisions endpoint on 2026-09-18.
  - arXiv 2609.30216 "Jev in the Wild" (Ling, Xue, Ye). PDF was binary-only via my fetcher; **not read** [U].
  - **`github.com/RomanSlack/jev-drone`** (MIT, about 229 stars): a MuJoCo quadrotor (Skydio X2) hierarchical stack, Jev at about 2.5 Hz, 50 Hz safety reflex with veto, 500 Hz flight controller. **Closest prior art to this project; hand it to agent 04.** Caveats stated by the author: single run, not seed-matched; does not reliably clear the harder tunnel.

### 1.3 Variants, clones, and alternatives (and why they matter)

| Name | What it is | Relevance |
|---|---|---|
| **SemIf** (TheoLeeCJ/SemIf) | Open "semantic ifs": read option probabilities from a small LLM's next-token logits in one forward pass (letter-choice prompt, one token per option, softmax) | Local/any-provider way to get Jev-style probabilities [V via JEV-CPU README] |
| **JEV-CPU** (HF `Meanblock/JEV-CPU`) | Independent CPU port of SemIf; **explicitly not affiliated with TypeSafe/Jev**; Qwen3-0.6B float32, about 1 s/decision on CPU; MIT | Shows the pattern; too slow as-is. Name collision with "Jev" [V] |
| **Ollama 0.35 `/v1/systemone`** | Ollama blog (2026-09-29, official) adds a Jev-compatible endpoint serving open "decision models": `nimble` (9B, Bespoke Labs), `tev1` (4B) and `tev1:0.8b` (Together AI, experimental). "Nimble 9B averaged 91 ms per decision ... on an M5 Max" (not M4) | **Best local stand-in candidate**: same request/response schema as Jev, no API cost [V-secondary blog; I have not run it] |
| **Red Hat / vLLM DiffusionGemma 26B-A4B** | vLLM server exposing a Jev-compatible `/v1/systemone`; quoted 35-60 ms single step on Google Cloud Run; 162 decisions/s on DGX Spark | Not for a 16 GB Mac; shows the schema is becoming a de-facto standard [V-secondary] |
| **`jaredpalmer/kev-4b` on OpenRouter** | 4B model, SiliconFlow, $0.042 in/$0 out, 8k ctx, same price as Jev | Probably a System One-style model; **unexamined** [U] |
| "djev", "OpenJev", "Laya" | Named in a Hugging Face community blog title comparing "decision models" | SEO-ish; **not checked**; ignore unless the user names one [U] |
| **Jev Router** | TypeSafe/OpenRouter LLM router | Not the same thing (see above) |
| Other spellings tried | "Jeev", "JEV" (Japanese encephalitis virus and unrelated), "Jetson" variants | Nothing relevant; JEV-CPU is the only hit and it is a clone |

### 1.4 What Jev does badly, and how that hits this project (from https://docs.typesafe.ai/model-jaggedness/jev-1.13.md [V])

| Documented weakness | Consequence for drone state prompts |
|---|---|
| Does not count reliably; "reads dates as text"; not a calculator; poor with raw numeric encodings (hex, RGB, binary); cannot interpolate exact numbers between score levels | **Do not hand it raw floats and expect it to compute time-to-collision or braking distance.** Pre-compute derived quantities in code (range, closing speed, TTC, bearing) and pass **semantic buckets** ("F: 2.3 m, closing 1.8 m/s, TTC 1.3 s, risk: high"). Feed the numbers it must read, not derive |
| Accuracy drops as **irrelevant state** grows | Keep state compact; no full lidar dumps; 5-7 sectors, not 360 rays |
| Literal reading of questions; fails on double negatives and contradictions between instructions and criteria | Action glossary must be unambiguous; one canonical phrasing per action |
| Adversarial content in state can shift answers | Not a concern for physical sensors, but **sensor-noise adversarial events will behave like noisy text** and must be tested |
| "Can't hallucinate" is overstated: constrained to valid options, still confidently wrong | Safety shield and confidence gating are mandatory (see agent 11) |
| Probabilities for different question types are not comparable | Do not mix Noul and Choice probabilities in one gating rule |
| Output examples show coarse probabilities (for example 0.78/0.22/0) [I from example responses] | Expect plateaus and ties; do not build threshold logic that needs fine resolution |

And per TypeSafe docs, **determinism across repeated identical calls is not documented** [V: /confidence.md, /models.md silent]. Treat it as non-deterministic until measured (spike F3).

### 1.5 Contradictions and gaps in the sources (resolve before trusting numbers)

| Item | Source A | Source B | My handling |
|---|---|---|---|
| Rate limit | docs.typesafe.ai/models.md: 100K tokens/s, **40 req/s** | flaviocopes/eesel/others: 250K tok/s, **1,200 req/min (20 req/s)** | Plan on **20 req/s** (the lower); both "dynamic, changing without notice" |
| OpenRouter path | tutorial: `/api/alpha/decisions` | docs reference prints `/api/v1/api/alpha/decisions`; docs also list `/api/v1/systemone` | Script uses the tutorial path (one string in `build_targets`, easy to change) |
| Context | OpenRouter page 32,000 | TypeSafe: 64k per request, 32k for state + longest question | Irrelevant at about 500-1,200 token prompts |
| Latency | vendor "70-500 ms, most about 100" (US West) | arXiv via OpenRouter median 315-320 ms | Both true in different conditions; see section 3 |
| "$2.50 per hour of drone flying" [search-result snippet] | not found on the source page (mindstudio states about 10 cents per 15-minute sim and "$42 per billion" tokens) | | Do not cite the $2.50 number [U]. Cost math below uses token pricing |
| Model page `openrouter.ai/typesafe/jev-1.13` | returned 404 to my fetcher | model API returns it fine | Use the API, not the HTML page |

### 1.6 Terms of service (partially verified)

- TypeSafe: states it does **not train on user data**; ZDR for enterprise; legal pages are an index only, the contracts themselves were not readable; **benchmark/publish/safety-critical terms unknown** [U]. Early access with launch pricing ("sustainability unproven", eesel review). Ask sales@typesafe.ai if results will be published.
- OpenRouter Terms: no benchmarking clause found; prohibits illegal use, unauthorised "red teaming" (prompt injection/jailbreaking) without written approval; disclaims fitness for safety-critical use; **drones/vehicles not mentioned** [V: https://openrouter.ai/terms]. Simulation-only use is unlikely to conflict; **the "adversarial events" in this project are physical-sim events, not prompt attacks**, so keep that wording distinct in any documentation to avoid looking like red-teaming.
- OpenRouter prompt logging is opt-in; provider-level data policies are filterable via `provider.data_collection: "deny"` [V].

---

## 2. OpenRouter: what it supports, per API surface

### 2.1 Jev (Decisions API) parameters [V: docs/api-reference/alphadecisions]

| Field | Notes |
|---|---|
| `model` | `typesafe/jev-1.13` (pin) |
| `state` | string / object / array, text only |
| `questions` | map of name to `{type: "choice"|"noul"|"score", instructions, criteria}`; `choice` criteria = `{option_key: description}` (max 255 options); `score` criteria = list of 2-10 level descriptions; `noul` criteria `{true, false}` |
| `provider` | `order`, `only`, `ignore`, `allow_fallbacks`, `data_collection`, `zdr`, `max_price`, `sort` (price/throughput/latency). Only one provider exists today, so routing is moot |
| `session_id`, `user`, `trace` | bookkeeping; `session_id` max 256 chars |
| **Absent** | `max_tokens`, `temperature`, `seed`, `stop`, `logprobs`, `logit_bias`, `tools`, `stream`. None needed: output is already typed and fixed-size |

Response: `answers[q]` with `choice`, `confidence`, `probabilities{option}`, or `noul` float, or `score` + `legend` + `probabilities`; plus `usage{input_tokens, output_tokens, cost}` and `model` (dated snapshot), `provider`, `id`. Errors: 400, 401, 402 (credits), 413, 429, 500, 502, 503, **524 (timeout)**, **529 (overloaded)**; retry 429/529 with exponential backoff.

The decision API exposes **no server-side timing field** (only `usage`). Network/TTFT/generation split is impossible for Jev; we can only measure client total and compare against a tiny no-op endpoint for the network floor (script section 9 does this for chat models via streaming; for Jev use the OpenRouter edge RTT as a floor, about 25-30 ms here).

### 2.2 Chat-completions parameters for the LLM/stand-in arms [V: openrouter.ai/docs/api-reference/parameters, /features/provider-routing, /features/structured-outputs]

| Feature | Support reality | Use here |
|---|---|---|
| `max_tokens` | universal | 1-4 for enum answers. **Reasoning models (gpt-oss, gpt-5*) spend hidden reasoning tokens**: set `reasoning.effort` low/minimal and give a bigger `max_tokens` (about 64), else you get empty or truncated output |
| `temperature: 0`, `seed` | accepted by most; **temperature 0 is not bitwise-deterministic** across GPU batching, and some providers ignore `seed` | Record, do not rely; see reproducibility (section 7) |
| `stop` | supported broadly | Stop at newline |
| `logprobs`/`top_logprobs` | **provider-specific.** From live endpoint metadata today: Cerebras (gpt-oss-120b), CoreWeave, Parasail, Novita, Fireworks, DeepInfra (some) list them; **Groq endpoints for gpt-oss-20b/120b and Llama 3.1 8B do not list `logprobs`** | Needed for Jev-style probability output from chat models. Requires `provider.require_parameters: true` |
| `logit_bias` | present on Cerebras, Parasail, DeepInfra, Together, Fireworks endpoints; absent on Groq/Mistral/OpenAI | Constrain first token to the action-letter tokens (A-K) |
| `response_format` json_schema / `structured_outputs` | widespread, enforcement varies by provider; adds constrained-decoding setup cost [I]; no official latency guidance | **Avoid for the latency path.** One-token enum via `logit_bias` + `max_tokens: 1` is cheaper and faster |
| `tools` / tool calls | supported; adds prompt tokens, typically slower | Avoid |
| streaming (`stream: true`) | SSE, occasional comment lines to ignore | Use **only to measure TTFT**, not needed in production for 1-4 tokens |
| `provider.order/only/allow_fallbacks:false` | pins the host; without `allow_fallbacks:false` you can be silently re-routed, which destroys latency comparability | Always pin in benchmarks |
| `provider.sort: "latency"`, `:nitro` (highest throughput), `:floor` (cheapest), `preferred_max_latency` (p50-p99, soft) | Soft preferences based on a rolling 5-minute window | `:nitro` helps long outputs, not 1-token ones [I]. For 1-token decisions, **TTFT/queueing dominates, so sort by latency, not throughput** |
| `session_id` | sticky routing to preserve provider KV/prefix cache; OpenRouter says up to 80-90% prefill savings for multi-turn | Useful for the LLM arm (shared system prompt + action glossary); changing the model slug mid-session breaks stickiness |
| Prompt/prefix caching | provider-dependent, "implicit caching" flag exists per endpoint (Jev: `supports_implicit_caching: false`) | Put the static glossary first, varying state last |
| Rate limits | Paid models: **no platform-level request cap**; 429s come from providers/Cloudflare DDoS; free `:free` models 20 rpm / 50-1000 per day; `GET /api/v1/key` shows quota; in-flight budget reserves estimated cost (402 if a single request's estimate exceeds budget) [V: docs/api-reference/limits] | Never benchmark on `:free`. Keep credit balance healthy: **docs say low balances cause cache expiry and extra latency** |

---

## 3. Real latency components

### 3.1 Decomposition of one remote decision (sense to actuator)

```
sim obs ready
 |- (a) client build + serialize prompt/JSON (0.1-1 ms Python; 300-1,000 tokens)
 |- (b) queue for a free connection / event-loop wake (0-few ms; more under GIL contention in the sim process)
 |- (c) DNS + TCP + TLS (cold only): measured here 4 ms + 13 ms + 16 ms; first request TTFB 67-96 ms; keep-alive removes this
 |- (d) network RTT client -> OpenRouter edge (Cloudflare Workers): measured about 25-30 ms TTFB on a warm HTTP/2 connection to a trivial cached endpoint
 |- (e) OpenRouter processing: auth/credit/route lookup (vendor says about 25-40 ms; one 200-call study measured OpenRouter 70 ms faster than OpenAI direct for TTFT, another reports 80-250 ms extra; inconsistent)
 |- (f) OpenRouter edge -> provider hop (same region: a few ms; cross-continent 70-150 ms)
 |- (g) provider queueing + prefill (for chat LLM: scales with input tokens; TTFT) / single forward pass (Jev)
 |- (h) decode (chat LLM: 1-4 tokens x ms/token; Jev: none, answers sampled in parallel)
 |- (i) response return + JSON parse (1-2 ms)
 '- (j) harness: apply action at the next control tick (up to 1 control period: 4 ms at 240 Hz, 20 ms at 50 Hz, 33 ms at 30 Hz)
```

Measured on this Mac (curl, HTTP/2, 20 sequential requests to `https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints`, no key, small JSON): DNS 4 ms, connect 17 ms, TLS done at 33 ms, first response 67-96 ms; subsequent requests on the same connection 25-30 ms total. This is **edge + trivial metadata**, not inference; it shows the network floor from this location is small, but the **TypeSafe origin is reportedly in US West** [V-secondary: flaviocopes/testmuai], so a user outside North America should expect far more than 25 ms to reach it [I].

### 3.2 Published numbers (nothing here was measured by me except the network floor)

| Source | Setup | p50 | p95 | Notes |
|---|---|---|---|---|
| TypeSafe via flaviocopes/mindstudio/eesel | vendor claim, US West | "most about 100 ms" | range 70-500 ms "end to end" | Vendor numbers; "40x-200x faster than frontier LLMs" on classification |
| OpenRouter blog, Jev vs LLM (https://openrouter.ai/blog/tutorials/jev-vs-llm-when-to-use-each/) 2026-09-19, 1 req/example, 60 tickets | Jev 1.13 | **194 ms** | **633 ms** | GPT-class model "GPT Luna": 1,106 / 2,395 ms; Claude Opus: 1,957 / 2,594 ms |
| same, 40 injection messages | Jev 1.13 | 194 ms | 688 ms | GPT Luna 805 / 1,491; Claude Opus 2,099 / 4,377 |
| arXiv 2609.22753 (2026-09-18), Jev via OpenRouter vs DeepSeek v4.1-flash via OpenRouter, 648 measurements each, 4 concurrent, 15 s timeout | Jev | **315-321 ms** (3 blocks) | **529 / 558 / 624 ms** | DeepSeek flash: 382-435 ms / 686-836 ms. **The Jev advantage over a flash LLM is only 10-37% at median; it is not 40-200x.** Study also used 20-60 ms simulated edge/cloud propagation; unclear if included in these numbers [U]. No rate-limit errors seen |
| jev-drone README (RomanSlack), MuJoCo tunnel | Jev, 3 questions, about 1.2k tokens/call | **110-118 ms** | p90 164 ms | Sequential ceiling 7.4 Hz; 4-6 workers pipelined: 21 decisions/s over about 2,000 calls, zero errors. Endpoint/region not stated [U]. Single machine |
| Artificial Analysis, Llama 3.1 8B, 10k-token input, 72 h median | Groq | TTFT **0.90 s** | n/a | That is for a **10k-token** prompt; our prompt is about 300-400 tokens so TTFT would be far lower. Output 645 tok/s. CoreWeave 0.70 s, Bedrock 0.74 s |
| Generic claims: Groq/Cerebras TTFT "under 100 ms" for small prompts | secondary blog roundups | | | Low-quality sources; treat as optimistic. **Measure** |

### 3.3 Planning numbers (inference from the above, flagged [I])

| Arm | Transport | Expected p50 | Expected p95 | p99 / tail | Basis |
|---|---|---|---|---|---|
| Jev, direct to TypeSafe, US West client | HTTP/2 keep-alive | 110-150 ms | 200-330 ms | 500 ms+ | jev-drone, vendor claim |
| **Jev via OpenRouter** | HTTP/2 keep-alive | **200-320 ms** | **530-700 ms** | 0.8-1.5 s (timeouts/529s possible) | OpenRouter blog + arXiv |
| Jev from Asia or another distant region | | +150-250 ms | | | RTT physics [I]; unmeasured |
| Fast small chat LLM (gpt-oss-20b on Groq, Llama 3.1 8B, Ministral 3B, gpt-5-nano), 300-token prompt, `max_tokens` 1-4 | via OpenRouter | **250-600 ms** | 600-1,500 ms | 2 s+ | [I]: OpenRouter overhead 40-100 ms + provider TTFT 100-400 ms; gpt-5-nano/gemini-flash-lite reasoning-class models can be slower. **Must measure** |
| Mid LLM with short reasoning | via OpenRouter | 1-2.6 s | 2.5-4.4 s | 10 s+ | OpenRouter blog (GPT-class, Opus) |
| Big reasoning LLM | | 3-30 s | | | published ranges; out of scope for a control loop |
| Local `tev1:0.8b` / `nimble` 9B on Ollama, M4 16 GB | localhost | 20-150 ms / 150-500 ms | | | [I]; Ollama's 91 ms is on **M5 Max**; base M4 (about 120 GB/s memory bandwidth) has materially less bandwidth than an M5 Max [I], 9B at 4-bit needs about 5.5 GB. **Unmeasured**; spike F4 |
| Local Qwen-0.5B-1.5B via MLX/llama.cpp, one forward pass, letter-logits (SemIf pattern) | in-process | 15-80 ms | | | [I] prompt-processing bound; spike F4 |
| Configurable mock | in-process | you set it | | | For latency sweeps (agent 08) |

**Key qualitative conclusions**
1. **Tail, not median, decides collisions**: p95/p99 events are the ones where the obstacle appears. Report full CDFs. With 4-6x p50-to-p95 ratios (Jev via OpenRouter: 194 to 633 ms), a mean-latency model understates risk.
2. **Jev's speed advantage is mostly "no decode + no reasoning"**, not a magic network path. Against a *fast non-reasoning small chat model with one output token*, expect a modest gap (tens to low hundreds of ms); against reasoning or large models, a gap of 5-20x. The demo should say which.
3. **A single sequential chain cannot exceed 1/latency**: 1/0.2 s = 5 Hz; 1/0.12 s = 8 Hz (jev-drone: 7.4 Hz). **Decisions/sec** scales only by overlapping calls, and each overlapped decision is stale by its own latency.
4. **The physical knee**: a 0.3 s delay at 2 m/s costs 0.6 m of travel; at 5 m/s, 1.5 m. Stopping distance for deceleration `a`: `d = v^2 / (2a)`; with `a` = 3 m/s^2, `v` = 5 m/s gives 4.2 m, `v` = 2 m/s gives 0.67 m. When obstacles appear at distances of several braking distances, **going from 300 ms to 100 ms barely changes outcomes; going from 3 s to 300 ms changes everything.** This is exactly the "latency knee" agent 13 should plot; scenario feasibility (agent 06) must be defined against braking distance + reaction budget.
5. **Staleness is the variable, not latency alone**: the state sent left at `t_obs`; the action arrives at `t_obs + L` and acts on a world that moved. Add `decision_age_ms` to the state (the benchmark payload does) so Jev can reason about it, and have the harness record `applied_at - observed_at` separately from the API wall latency (agent 08 owns the clock model).

---

## 4. Minimising latency (ranked by expected gain)

| # | Lever | Expected effect | Applies to |
|---|---|---|---|
| 1 | **Persistent HTTP/2 client**, one `httpx.AsyncClient` per process, pre-warmed before timing | Removes 50-100 ms cold connect/TLS on each call [V measured: 67-96 ms cold vs 25-30 ms warm to the edge] | All remote arms |
| 2 | **Region**: run near the provider (US West for Jev). If you are in Asia, a cloud VM in us-west for the API-arm runner is a legitimate experimental control (but hurts "runs on a laptop") | up to 150-250 ms [I] | All |
| 3 | **Direct vs OpenRouter**: direct saves OpenRouter's hop, vendor 25-40 ms, reported 40-250 ms. Jev direct sits at about 110-120 ms p50 (jev-drone) vs about 200-320 ms via OpenRouter (blog/paper), although those are different tests and networks | 50-200 ms [I] | Jev |
| 4 | **Pin the provider** (`order`, `allow_fallbacks: false`) and pin the model snapshot | Removes routing variance and silent model swaps | Chat LLMs |
| 5 | **Fewer, shorter tokens**: prompt 300 tokens, not 3,000. Jev prices and probably scales with input; chat TTFT scales roughly linearly with input on a cold prefix [I] | 10-40% [I] | All |
| 6 | **Static-prefix first, varying state last** and sticky `session_id` so providers can reuse prefix cache | Up to 80-90% prefill saving claimed by OpenRouter on multi-turn [V-secondary]; small at 300 tokens | Chat LLMs |
| 7 | **One-token outputs**: `max_tokens: 1` with letter codes A-K (distinct single tokens) + `logit_bias` to restrict; `stop: ["\n"]` | Removes decode time, parse failures, and gives `top_logprobs` probabilities where supported | Chat LLMs |
| 8 | Turn reasoning off (`reasoning.effort` low/minimal, or pick a non-reasoning model) | Seconds to hundreds of ms | Chat LLMs |
| 9 | **Do not use** `response_format: json_schema`, tools, `:nitro` for 1-token outputs | Adds overhead or does nothing | Chat LLMs |
| 10 | **Pipelining with a speculative/stale-tolerant loop**: issue call N+1 before N returns | Raises decisions/s (21/s reported) without lowering per-decision latency | All |
| 11 | Hedged requests (send to two providers, take the first) | Cuts p99 materially at 2x cost | Optional arm, changes the question being asked; do not mix into the main comparison |
| 12 | Jev-specific: batch several questions per call (they are evaluated in parallel; cost is input tokens once) | Extra signals at no extra latency | Jev |
| 13 | Local model in-process | Removes network entirely; limited by prefill on M4 | Local arm |

Python/client gotchas (for agent 08/09, stated here because they distort latency measurements): time with `time.perf_counter_ns()`; the event loop in the same process as PyBullet stepping adds multi-ms jitter, so **run the sim in a worker process and the API client in another** or at minimum keep sim steps off the loop; TLS handshakes and first-request DNS on a cold pool inflate the first calls, so discard warm-up (the script discards 5).

---

## 5. Rate limits, cost per decision, scale

Token assumptions: Jev 500-1,200 input tokens per call (476 in the tutorial's 3-question example; about 1.2k in jev-drone with 3 questions); chat LLM 350-450 input, 1-4 output. Prices from live OpenRouter endpoint metadata today (per M tokens, in/out): Jev $0.042/$0; gpt-oss-20b Groq $0.075/$0.30; gpt-oss-120b Groq $0.15/$0.60, Cerebras $0.35/$0.75; Llama 3.1 8B Groq $0.05/$0.08; gpt-5-nano $0.05/$0.40 (OpenAI $0.025-0.05 tiers also listed); Ministral 3B $0.10/$0.10; Claude Haiku 4.5 $1/$5.

| Arm | Cost per decision | 200 decisions/flight (20 s at 10 Hz) | 1,000 flights | 10,000 flights |
|---|---|---|---|---|
| Jev (1k tokens) | $0.000042 | $0.0084 | **$8** | **$84** |
| gpt-oss-20b Groq (400 in, about 40 incl. reasoning out) | about $0.00004 | $0.008 | $8 | $80 |
| gpt-oss-120b Cerebras | about $0.00017 | $0.034 | $34 | $340 |
| Claude Haiku 4.5 (400 in, 4 out) | about $0.0004 | $0.08 | $80 | $800 |
| Big reasoning LLM (400 in, 500 out at $5/M) | about $0.003 | $0.6 | $600 | $6,000 |

At 2.5 Hz (50 decisions/flight) divide by 4. So **Jev and fast small LLMs are cheap enough for the full matrix; the slow big LLM arm must be sub-sampled** (for example 100-300 flights, or decision-level caching of oracle labels plus latency injection, agent 08/09).

**Rate limit and time budgets** (Jev: plan 20 req/s; OpenRouter paid models have no platform cap, provider limits apply and are undocumented per endpoint):

- Per-flight decisions are an inherently **sequential chain** (next state depends on prior action), so parallelism comes from running many flights at once. One flight at 10 Hz wants 10 req/s if latency were 0; in lockstep (pause-the-world) with 300 ms calls one flight needs 200 x 0.3 s = 60 s wall time.
- 10,000 Jev flights x 200 decisions = 2M calls. At the 20 req/s cap: **28 h**; at 40 req/s: 14 h. With the 120 ms-best-case latency you still need at least 6-8 flights in flight to saturate 20 req/s; with 300 ms latency, 12-24 concurrent flights. Concurrency limit on this M4 comes from the sim workers (agent 09), not from the API.
- **Recommendation**: API arms run on a **sub-sampled paired set** (for example 500-1,000 flights per API arm); fast, mock/local/PID arms run the full 10k. Budget at that scale is under $100 total for Jev + two fast LLMs; a slow-LLM arm with reasoning costs tens to a few hundred dollars.
- 429/529 handling: exponential backoff with jitter for **data collection**, but in the **latency-measurement path record a timeout/error event as a real outcome** (harness falls back to the last action or HOVER), never silently retry and report the retried latency as if it were a clean decision.
- Early-access risk: limits "adjust dynamically"; uptime/latency can regress during the run. Record timestamp + provider + snapshot id per call and run each arm **interleaved** in time (not arm-by-arm) to avoid confounding arm with time-of-day.

## 6. Fastest providers for tiny outputs: what to pick (and what not to claim)

- Published TTFT data is mostly for long prompts (Artificial Analysis uses 10k input tokens; Groq 0.90 s for Llama 3.1 8B), so **do not transfer them to a 300-token prompt** [I]. Real ranking must come from the script.
- From live OpenRouter metadata, the host/parameter combos that matter for a 1-token arm:
  - **Cerebras, gpt-oss-120b**: `logprobs`, `top_logprobs`, `logit_bias`, `structured_outputs`, `seed` all listed: the **best fit for a Jev-style probability output** from a chat model. $0.35/$0.75.
  - **Groq**: gpt-oss-20b/120b, gpt-oss-safeguard-20b, Llama 3.1 8B. Fast, but **no `logprobs`/`logit_bias` listed** today, so the arm must parse one token of text.
  - **Parasail / CoreWeave / Novita**: `logprobs` + `logit_bias` on small models (Llama 3.2 3B, gpt-oss-20b, Llama 3.1 8B on CoreWeave).
  - **SambaNova, Fireworks**: present for other models; I did not find a small model with useful parameters in my shortlist. [U]
  - Reasoning-class defaults (gpt-5-nano, gpt-oss) can be slower than a non-reasoning 8B unless effort is minimal/low.
- **Recommended API arms (benchmark-first)**: (1) Jev via OpenRouter, (2) Jev direct (if the user wants TypeSafe key), (3) fast small chat LLM: gpt-oss-20b on Groq or Llama 3.1 8B on Groq, (4) fast small chat LLM with logprobs: gpt-oss-120b on Cerebras, (5) "slow LLM" arm: a mid/large model (Haiku 4.5-class or bigger) to get the 1-3 s regime honestly, (6) local `tev1:0.8b`/`nimble` via Ollama `/v1/systemone`, (7) in-process MLX/llama.cpp letter-logit model.
- **Local on M4 16 GB**: model choices Qwen 0.5-3B, Llama 3.2 1B/3B, Gemma/Phi small: prefill-bound; expect tens to low hundreds of ms for ~300 tokens with MLX/llama.cpp [I]. The Ollama `nimble` 9B figure of 91 ms is from an M5 Max and is not transferable [V-secondary]. **Every local latency claim here is unmeasured**; spike F4 measures it. Local has zero network variance and no cost, which makes it the cleanest controlled baseline for separating "model smartness" from "network latency".

---

## 7. Reproducibility and drift

- **Version drift**: alias `~typesafe/jev-latest` moves; pin `typesafe/jev-1.13`; log `response.model` (snapshot `jev-1.13-20260917`); abort the run if the snapshot string changes mid-experiment. OpenRouter chat models behind one slug can be served by different quantizations/providers: pin `provider.order` + `quantizations` and log the provider returned.
- **Non-determinism**: Jev documents none; temperature/seed do not exist. LLM `temperature: 0` + `seed` is best effort only. So: (a) **cache decision outputs keyed by (provider_id, pinned snapshot, canonical rendered input hash)** to make replays reproducible, (b) separately record the **latency distribution** per provider/time window and **resample it** in fast runs (agent 08's hybrid option d), and (c) repeat a **consistency probe** (same input x 50) to quantify output variance (spike F3).
- **Caching caveat** (also arXiv 2609.22753: "with caching enabled on repeated text, latency advantage nearly disappeared"): if any layer caches, latency measured on repeated identical inputs is not representative. Ensure near-duplicate drone states do not hit a provider prefix cache unevenly between arms; vary a nonce in the **tail** of the state, not the head, only if you want cold-path numbers; otherwise state the policy explicitly.
- **Time-of-day/region**: record local UTC time, client region (IP-derived), and OS/Python/httpx versions in a run manifest (agent 08).

---

## 8. DecisionProvider design (model-agnostic; works with or without Jev)

### 8.1 Design rules (consistent with `.github/copilot-instructions.md`)

- Lives in `app/providers/` (Protocol + implementations), DTOs in `app/schemas/decision.py`; class-based, dependencies injected via `__init__` (HTTP client, settings), **no inline instantiation of clients**, no global state, no `print`, `%s` lazy logging, `logger.exception` on failures, full type annotations.
- Errors are typed and client-safe and subclass the repo's `AppError` family from `app/core/exceptions.py` (for example `ProviderTimeoutError(ServiceUnavailableError)`, `ProviderRateLimitedError(TooManyRequestsError)`, `InvalidDecisionOutputError(UnprocessableEntityError)`). **Note: the harness/sim code path is not an HTTP path, so these are domain exceptions first; they only need `AppError` if surfaced through the API.** Flag for agent 14 to decide; the sketch uses stand-in bases.
- **Providers never choose fallbacks** and never know about simulated time. They return wall-clock timing; the harness owns staleness, deadlines, and the safety shield. This keeps the provider testable and the clock model (agent 08) replaceable.
- **Settings** (pydantic-settings, `.env` for keys): `OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, base URLs, model ids, timeouts. Import only via `from app.core.config import settings`.
- Every provider has a **stable `provider_id` including the pinned model snapshot**, stored with each decision row (agent 10).

### 8.2 Protocol and DTO sketch (type-checked: `mypy` clean and `ruff` clean in a Python 3.14.2 scratch venv)

```python
"""Sketch: app/schemas/decision.py + app/providers/base.py (merged for type-checking)."""

import math
import random
from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


# --- stand-ins for app/core/exceptions.py (real ones subclass AppError) ---------------------
class ServiceUnavailableError(Exception): ...


class TooManyRequestsError(Exception): ...


class UnprocessableEntityError(Exception): ...


class ProviderTimeoutError(ServiceUnavailableError): ...


class ProviderRateLimitedError(TooManyRequestsError): ...


class InvalidDecisionOutputError(UnprocessableEntityError): ...


# --- app/schemas/decision.py -----------------------------------------------------------------
class Action(StrEnum):
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ASCEND = "ASCEND"
    DESCEND = "DESCEND"
    BRAKE = "BRAKE"
    HOVER = "HOVER"
    AVOID_LEFT = "AVOID_LEFT"
    AVOID_RIGHT = "AVOID_RIGHT"
    RETURN_HOME = "RETURN_HOME"
    EMERGENCY_LAND = "EMERGENCY_LAND"


class DecisionRequest(BaseModel):
    """Provider-agnostic input. `state` is the per-arm projection of Observation (see 07)."""

    model_config = ConfigDict(frozen=True)
    request_id: str
    state: Mapping[str, object]  # JSON-serialisable projection of the Observation
    allowed_actions: tuple[Action, ...] = tuple(Action)
    sim_time_s: (
        float  # sim time the observation was taken (set by harness, not provider)
    )
    deadline_ms: float = 1000.0  # harness gives up and falls back after this wall time


class DecisionTiming(BaseModel):
    """Wall-clock facts only; the harness converts to sim-time staleness (see 08)."""

    wall_total_ms: float  # perf_counter_ns around the whole call (incl. serialisation)
    wall_ttfb_ms: float | None = None  # first byte/headers
    wall_first_token_ms: float | None = None  # streaming LLMs only
    server_ms: float | None = None  # if the vendor reports it (None for Jev today)
    sim_latency_ms: float | None = None  # set only by LatencyInjectingProvider
    connection_reused: bool | None = None
    retries: int = 0


class DecisionResult(BaseModel):
    action: Action
    probabilities: dict[Action, float] = Field(
        default_factory=dict
    )  # empty if provider has none
    confidence: float | None = (
        None  # in [0, 1]; provider-defined, calibrate per provider
    )
    aux: dict[str, float] = Field(
        default_factory=dict
    )  # e.g. {"risk": 1.7, "obstacle_imminent": 0.93}
    timing: DecisionTiming
    provider_id: str  # stable id incl. pinned model version, e.g. "jev:typesafe/jev-1.13-20260917"
    cache_hit: bool = False
    input_tokens: int | None = None
    cost_usd: float | None = None


class ProviderCapabilities(BaseModel):
    returns_probabilities: bool
    supports_aux_questions: bool  # Jev-style parallel noul/score
    deterministic: (
        bool  # True only if identical input => identical output is guaranteed/tested
    )
    max_state_tokens: int
    is_remote: bool


# --- app/providers/base.py ---------------------------------------------------------------------
@runtime_checkable
class DecisionProvider(Protocol):
    """One decision per call. Must not block the event loop; must honour request.deadline_ms.

    Raises ProviderTimeoutError / ProviderRateLimitedError / InvalidDecisionOutputError.
    Never applies a fallback action itself: fallback + staleness policy belong to the harness.
    """

    @property
    def provider_id(self) -> str: ...

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def decide(self, request: DecisionRequest) -> DecisionResult: ...

    async def aclose(self) -> None: ...


class LatencyModel(Protocol):
    """Pure sampler, fed from recorded JSONL (empirical) or a fitted lognormal."""

    def sample_ms(self, rng: random.Random) -> float: ...


class LogNormalLatency:
    def __init__(self, median_ms: float, sigma: float) -> None:
        self._mu = math.log(median_ms)
        self._sigma = sigma

    def sample_ms(self, rng: random.Random) -> float:
        return rng.lognormvariate(self._mu, self._sigma)


class LatencyInjectingProvider:
    """Decorator: wraps any DecisionProvider (e.g. an oracle/cached one) and stamps sim latency."""

    def __init__(
        self, inner: DecisionProvider, latency: LatencyModel, seed: int
    ) -> None:
        self._inner = inner
        self._latency = latency
        self._rng = random.Random(seed)

    @property
    def provider_id(self) -> str:
        return f"{self._inner.provider_id}+lat"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._inner.capabilities

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        result = await self._inner.decide(request)
        result.timing.sim_latency_ms = self._latency.sample_ms(self._rng)
        return result

    async def aclose(self) -> None:
        await self._inner.aclose()
```

### 8.3 Implementations (each is a thin class; build order in agent 14)

| Class | Backend | Returns probabilities | Notes |
|---|---|---|---|
| `JevDecisionProvider` | `POST /api/alpha/decisions` (OpenRouter) or `api.typesafe.ai/v1/systemone` (direct). Injected `httpx.AsyncClient(http2=True)` | yes (Choice `probabilities`) | One `choice` question `action` with the 11 criteria from the action glossary; optional parallel `noul` (`obstacle_imminent`, `gps_trustworthy`) and `score` (`risk` 0-3) into `aux`. Timing: client-only (no server fields). Pinned `typesafe/jev-1.13` |
| `ChatDecisionProvider` | OpenRouter `/api/v1/chat/completions`, configurable model/provider pin; "fast" and "slow" are just config | optional via `top_logprobs` when the pinned endpoint supports it, else empty | Letter code A-K, `max_tokens` 1-4 (64 for reasoning models with effort low), `temperature 0`, `stop`. Parse failure raises `InvalidDecisionOutputError` (harness counts `invalid_output_rate`) |
| `SystemOneLocalProvider` | Ollama `/v1/systemone` (`nimble`, `tev1`) | yes | Same request body as Jev, swap base URL. Stand-in that preserves Jev semantics offline |
| `LogitsLocalProvider` | in-process MLX/llama.cpp, SemIf pattern (letter choice, one forward pass, softmax over option logits) | yes | Needs a process/thread to avoid blocking the loop; the truly zero-network control |
| `MockLatencyProvider` | no backend; delegates to a policy/oracle callable | configurable | Takes a `LatencyModel` and sleeps or (preferred) returns `sim_latency_ms` only; seeded |
| `LatencyInjectingProvider` | decorator (above) | inherits | Use with cached/oracle decisions for speed-without-network latency sweeps (agent 08 option c/d) |
| `ReplayProvider` | reads `(provider_id, snapshot, input_hash) -> decision` from the model-call cache (agent 10) | as cached | Reproducible re-runs; `cache_hit=True`; latency resampled, not replayed |

**Jev adapter core (illustrative, untested against the live API; field names from the tutorial/reference pages):**

```python
body = {
    "model": self._model,                      # "typesafe/jev-1.13"
    "state": dict(request.state),
    "questions": {
        "action": {"type": "choice", "instructions": ACTION_QUESTION, "criteria": ACTION_CRITERIA},
        "imminent": {"type": "noul", "instructions": "Is a collision imminent within 1 s?",
                     "criteria": {"true": "...", "false": "..."}},
    },
}
resp = await self._client.post(self._url, json=body, headers=self._headers)
# 429/529 -> ProviderRateLimitedError; 524/timeouts -> ProviderTimeoutError
ans = resp.json()["answers"]["action"]
return DecisionResult(action=Action(ans["choice"]),
                      probabilities={Action(k): v for k, v in ans["probabilities"].items()},
                      confidence=ans["confidence"], ...)
```

Design choices worth defending:
- **Probabilities are first-class** (Jev and logit-based arms expose them); the safety shield uses `confidence < threshold => fall back`, which is the vendor's recommended pattern ("confidence-gated routing", thresholds >0.9 critical / <0.5 escalate). **Thresholds must be calibrated on your own rollouts** (vendor says so); pin model version when tuned.
- **Two separate notions of latency** are stored: `wall_*` (real API) and `sim_latency_ms` (injected); the harness, not the provider, turns either into control-tick delay.
- `deadline_ms` is an input so the harness can enforce a per-decision timeout and count a timeout as a first-class outcome.
- **The Protocol is stateless per call** (no conversation history) so arms are comparable and cacheable.

---

## 9. Ready-to-run latency benchmark (run later, once the user has a key)

What it does: pre-warmed HTTP/2 `httpx.AsyncClient`; per target, 5 warm-up + 200 sequential calls (latency) then 400 calls at concurrency 4 and 16 (queueing/rate-limit behaviour); reports p50/p95/p99/max/mean, error and 429 counts, throughput; streaming variants measure first-token latency for chat models; writes every record to JSONL (no prompts, no keys logged). Targets: Jev via OpenRouter Decisions API; six fast chat models pinned to a named provider with `allow_fallbacks: false`. Payload is a realistic about 300-token drone state + 11-action glossary. **Expected spend: about 13 targets x about 800 calls, about $0.50-2 total.** Edit the `fast` list to taste; add a `typesafe` direct target by copying the Jev `Target` with `base = https://api.typesafe.ai` and path `/v1/systemone` and model `jev-1.13.0`, and a local Ollama target at `http://localhost:11434/v1/systemone` (`model: "tev1:0.8b"`, no key).

Validation performed: `py_compile` + `ruff` clean on Python 3.14.2; **dry-run against a local stdlib mock server** (`BENCH_MOCK=1 BENCH_BASE_URL=http://127.0.0.1:8765`) produced correct summaries and a 2,010-record JSONL. **It has not been run against any real endpoint (no key available).** Things likely to need a tweak on first run: the exact OpenRouter Jev path (see section 1.5), `reasoning.effort` acceptance by a given provider, and whether some providers reject `max_tokens` below a threshold. The script prints results (it is a standalone script outside `app/`, so the repo's "no print in app code" rule does not apply; adapt to the logger if it moves into `scripts/`).

Usage:
```bash
uv venv --python 3.14.2 && uv pip install "httpx[http2]"
export OPENROUTER_API_KEY=...            # do not paste in chat or commit
uv run python bench_decision_latency.py  # about 15-25 min
```
Run it **(a)** from the actual machine/network that will run the experiment, **(b)** at 2-3 different times of day, **(c)** with the laptop on power and not thermally loaded (agent 08/09).

```python
"""Decision-provider latency benchmark (Jev decisions API vs fast chat LLMs vs local).

Run later with a key:  OPENROUTER_API_KEY=... uv run python bench_decision_latency.py
Dry-run (no key, local mock server):  BENCH_MOCK=1 uv run python bench_decision_latency.py
Requires: httpx[http2]  (uv pip install "httpx[http2]")
Logs: per-request JSONL written to RESULTS_PATH. No prompt/key is logged.
"""

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# ---------------------------------------------------------------- constants
MOCK = os.environ.get("BENCH_MOCK") == "1"
OPENROUTER_BASE = os.environ.get("BENCH_BASE_URL", "https://openrouter.ai")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "mock" if MOCK else "")
WARMUP = 5  # discarded requests per (target, concurrency)
N_SEQUENTIAL = 200  # closed-loop, one in flight (measures latency)
N_CONCURRENT = 400  # open throughput test (measures queueing / rate limit)
CONCURRENCY_LEVELS = (4, 16)
TIMEOUT_S = 15.0
RESULTS_PATH = ROOT_DIR / "results" / f"latency_{int(time.time())}.jsonl"

ACTIONS = {
    "FORWARD": "Path ahead is clear; continue toward the waypoint.",
    "LEFT": "Shift left toward the target bearing.",
    "RIGHT": "Shift right toward the target bearing.",
    "ASCEND": "Climb to clear an obstacle or regain altitude.",
    "DESCEND": "Lower altitude toward the target altitude.",
    "BRAKE": "Decelerate now; obstacle is close ahead.",
    "HOVER": "Hold position; nothing safe to do but wait.",
    "AVOID_LEFT": "Obstacle ahead/right; dodge left.",
    "AVOID_RIGHT": "Obstacle ahead/left; dodge right.",
    "RETURN_HOME": "Battery low or GPS lost; fly back to home.",
    "EMERGENCY_LAND": "Critical failure; land immediately.",
}

# ~300-token realistic drone state (compact JSON).
STATE: dict[str, Any] = {
    "t_s": 12.34,
    "decision_age_ms": 40,
    "pos_m": [3.1, -1.2, 1.5],
    "vel_mps": [1.8, 0.1, 0.0],
    "yaw_deg": 12.0,
    "target_rel_m": [6.2, 1.1, 0.0],
    "lidar_m_by_sector": {"L90": 9.0, "L45": 7.4, "F": 2.3, "R45": 6.8, "R90": 9.0},
    "closing_speed_mps": 1.8,
    "battery_pct": 61,
    "gps": {"ok": True, "hdop": 0.9},
    "wind_est_mps": [0.5, -0.2, 0.0],
    "other_drone": None,
    "home_rel_m": [-3.1, 1.2, -1.5],
    "last_action": "FORWARD",
}
QUESTION = "Pick the single best discrete flight action for the next 100 ms."


@dataclass
class Target:
    name: str
    url: str
    body: dict[str, Any]
    kind: str  # "decisions" | "chat" | "chat_stream"
    headers: dict[str, str] = field(default_factory=dict)


def chat_body(
    model: str, stream: bool, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    prompt = (
        "Drone state JSON:\n"
        + json.dumps(STATE, separators=(",", ":"))
        + "\nActions: "
        + ",".join(ACTIONS)
        + "\nReply with exactly one action name."
    )
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 4,
        "temperature": 0,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
    }
    body.update(extra or {})
    return body


def build_targets() -> list[Target]:
    base = OPENROUTER_BASE
    decisions = {
        "model": "typesafe/jev-1.13",
        "state": STATE,
        "questions": {
            "action": {"type": "choice", "instructions": QUESTION, "criteria": ACTIONS}
        },
    }
    t = [
        Target("jev_openrouter", f"{base}/api/alpha/decisions", decisions, "decisions")
    ]
    if MOCK:
        return t + [
            Target(
                "chat_mock",
                f"{base}/api/v1/chat/completions",
                chat_body("mock/model", False),
                "chat",
            )
        ]
    # Edit freely. provider.order pins the host; allow_fallbacks=False stops silent re-routing.
    fast = [
        ("openai/gpt-oss-20b", "Groq"),
        ("meta-llama/llama-3.1-8b-instruct", "Groq"),
        ("openai/gpt-oss-120b", "Cerebras"),
        ("openai/gpt-5-nano", None),
        ("google/gemini-2.5-flash-lite", None),
        ("mistralai/ministral-3b-2512", None),
    ]
    for model, prov in fast:
        extra: dict[str, Any] = {
            "provider": {"allow_fallbacks": False, "sort": "latency"}
        }
        if prov:
            extra["provider"] = {"order": [prov], "allow_fallbacks": False}
        if "gpt-oss" in model or "gpt-5" in model:
            # avoid hidden reasoning tokens: gpt-oss supports low (not minimal); gpt-5 supports minimal
            extra["reasoning"] = {"effort": "low" if "gpt-oss" in model else "minimal"}
            extra["max_tokens"] = 64  # reasoning models reject tiny budgets
        t.append(
            Target(
                f"{model}@{prov or 'auto'}",
                f"{base}/api/v1/chat/completions",
                chat_body(model, False, extra),
                "chat",
            )
        )
        t.append(
            Target(
                f"{model}@{prov or 'auto'}#stream",
                f"{base}/api/v1/chat/completions",
                chat_body(model, True, extra),
                "chat_stream",
            )
        )
    return t


async def one_call(client: httpx.AsyncClient, tg: Target) -> dict[str, Any]:
    hdr = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        **tg.headers,
    }
    t0 = time.perf_counter_ns()
    rec: dict[str, Any] = {
        "target": tg.name,
        "ok": False,
        "status": None,
        "ttfb_ms": None,
        "total_ms": None,
        "err": None,
    }
    try:
        if tg.kind == "chat_stream":
            async with client.stream("POST", tg.url, json=tg.body, headers=hdr) as r:
                rec["status"] = r.status_code
                rec["ttfb_ms"] = (time.perf_counter_ns() - t0) / 1e6  # headers received
                first = None
                async for line in r.aiter_lines():
                    if (
                        line.startswith("data:")
                        and '"content"' in line
                        and first is None
                    ):
                        first = (
                            time.perf_counter_ns() - t0
                        ) / 1e6  # first content token
                rec["first_token_ms"] = first
                rec["ok"] = r.status_code == 200
        else:
            r = await client.post(tg.url, json=tg.body, headers=hdr)
            rec["status"] = r.status_code
            rec["ttfb_ms"] = (time.perf_counter_ns() - t0) / 1e6
            rec["ok"] = r.status_code == 200
            if rec["ok"] and tg.kind == "decisions":
                ans = r.json()["answers"]["action"]
                rec["choice"], rec["conf"] = ans.get("choice"), ans.get("confidence")
                rec["cost"] = r.json().get("usage", {}).get("cost")
            elif r.status_code != 200:
                rec["err"] = r.text[:160]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        rec["err"] = f"{type(exc).__name__}: {exc}"[:160]
    rec["total_ms"] = (time.perf_counter_ns() - t0) / 1e6
    return rec


def pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else float("nan")


def summarize(label: str, recs: list[dict[str, Any]]) -> str:
    ok = [r["total_ms"] for r in recs if r["ok"]]
    ft = [r["first_token_ms"] for r in recs if r.get("first_token_ms")]
    errs = len(recs) - len(ok)
    line = (
        f"{label:<46} n={len(recs):<4} err={errs:<3} p50={pct(ok, 50):7.1f} p95={pct(ok, 95):7.1f} "
        f"p99={pct(ok, 99):7.1f} max={max(ok, default=float('nan')):7.1f} "
        f"mean={statistics.fmean(ok) if ok else float('nan'):7.1f} ms"
    )
    if ft:
        line += f" | first-token p50={pct(ft, 50):.1f} p95={pct(ft, 95):.1f}"
    return line


async def run_sequential(
    client: httpx.AsyncClient, tg: Target, sink: list[dict[str, Any]]
) -> None:
    for i in range(WARMUP + N_SEQUENTIAL):
        rec = await one_call(client, tg)
        rec.update(mode="seq", idx=i, warmup=i < WARMUP)
        sink.append(rec)
    print(
        summarize(
            f"{tg.name} [seq]",
            [
                r
                for r in sink
                if r["target"] == tg.name and r["mode"] == "seq" and not r["warmup"]
            ],
        )
    )


async def run_concurrent(
    client: httpx.AsyncClient, tg: Target, conc: int, sink: list[dict[str, Any]]
) -> None:
    sem = asyncio.Semaphore(conc)
    out: list[dict[str, Any]] = []

    async def worker(i: int) -> None:
        async with sem:
            rec = await one_call(client, tg)
            rec.update(mode=f"conc{conc}", idx=i, warmup=False)
            out.append(rec)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(N_CONCURRENT)))
    wall = time.perf_counter() - t0
    sink.extend(out)
    print(
        summarize(f"{tg.name} [conc={conc}]", out)
        + f" | {N_CONCURRENT / wall:.1f} req/s, "
        f"429s={sum(r['status'] == 429 for r in out)}"
    )


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("bench")
    if not API_KEY:
        log.error("OPENROUTER_API_KEY not set (or use BENCH_MOCK=1)")
        return 2
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log.info(
        "start; python=%s httpx=%s mock=%s",
        sys.version.split()[0],
        httpx.__version__,
        MOCK,
    )
    sink: list[dict[str, Any]] = []
    limits = httpx.Limits(
        max_keepalive_connections=32, max_connections=64, keepalive_expiry=60
    )
    async with httpx.AsyncClient(
        http2=True, timeout=TIMEOUT_S, limits=limits
    ) as client:
        for tg in build_targets():
            await run_sequential(client, tg, sink)
            for conc in CONCURRENCY_LEVELS:
                await run_concurrent(client, tg, conc, sink)
    with RESULTS_PATH.open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in sink)
    log.info(
        "done in %.1fs; %d records -> %s",
        time.perf_counter() - start,
        len(sink),
        RESULTS_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

---

## 10. Contradictions with `docs/initial-plan.txt`

1. The plan treats Jev as an undefined fast model that just "gets the state and has to react". **The real Jev cannot be used as a free-form LLM** (no text, no tools, no sampling params) and **needs a typed question/criteria framing**; its output is probabilistic choices. The plan's "LLM" comparison arm is a different API shape, so the comparison is **two different interfaces**, which the provider abstraction has to hide.
2. "Jev low latency ... killer demo: PID/MPC ~ms, Jev short bar, LLM long bar": true in direction only. Jev is **roughly 100-300 ms at best, not single-digit ms**; it is **slower than PID by 1-2 orders of magnitude** and in the same order as a *fast* small LLM under good conditions. The demo's honest framing is "a discrete, semantic decision layer that arrives in the same 10th of a second or so as a human reflex, vs seconds for an LLM", not "as fast as control".
3. The plan implies Jev can drive many decisions per second across thousands of flights. Rate limits (20-40 req/s), cost (0.004-0.01 cents/call), and latency (5-8 Hz sequential) mean API arms need **sub-sampling** unless a local stand-in is used. (Throughput claim "~20 kHz/80 drones" is agent 02's topic.)
4. Jev and "Jev Router" (an LLM router) are different things; a search for "Jev OpenRouter" can easily surface the router page first.

---

## 11. Sources (accessed 2026-09-30)

- OpenRouter model API (primary): https://openrouter.ai/api/v1/models (464 models), `?output_modalities=all` (637), `/models/typesafe/jev-1.13/endpoints`, `/models/~typesafe/jev-latest/endpoints`, `/models/typesafe/jev-router/endpoints`; chat-model endpoint metadata for the shortlisted models.
- OpenRouter docs: https://openrouter.ai/docs/guides/community/jev ; https://openrouter.ai/docs/guides/community/jev-tutorial ; https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request ; https://openrouter.ai/docs/api-reference/parameters ; https://openrouter.ai/docs/features/provider-routing ; https://openrouter.ai/docs/features/structured-outputs ; https://openrouter.ai/docs/guides/best-practices/latency-and-performance ; https://openrouter.ai/docs/api-reference/limits ; https://openrouter.ai/docs/api-reference/overview ; https://openrouter.ai/terms ; https://openrouter.ai/docs/guides/privacy/logging
- OpenRouter blog: https://openrouter.ai/blog/insights/what-is-jev/ ; https://openrouter.ai/blog/tutorials/jev-vs-llm-when-to-use-each/
- TypeSafe docs: https://docs.typesafe.ai/introduction.md ; /api.md ; /models.md ; /concepts/state.md ; /confidence.md ; /model-jaggedness/jev-1.13.md ; /legal.md ; index https://docs.typesafe.ai/llms.txt
- Papers: https://arxiv.org/abs/2609.22753 and https://arxiv.org/html/2609.22753v1 ; https://arxiv.org/pdf/2609.30216 (not read)
- Prior art: https://github.com/RomanSlack/jev-drone ; https://awesomejev.vercel.app/p/jev-drone/
- Secondary: https://flaviocopes.com/jev/ ; https://www.mindstudio.ai/blog/jev-system-one-model-launch ; https://www.eesel.ai/blog/typesafe-jev-review ; https://ollama.com/blog/ollama-now-supports-jev-style-decision-models ; https://developers.redhat.com/articles/2026/09/28/run-decision-model-vllm-and-red-hat-ai ; https://huggingface.co/Meanblock/JEV-CPU ; https://opper.ai/blog/llm-router-latency-benchmark-2026 ; https://markaicode.com/benchmarks/openrouter-production-benchmark-latency/ ; https://artificialanalysis.ai/models/llama-3-1-instruct-8b/providers
- Own measurements: curl HTTP/2 timings to openrouter.ai from this M4 (section 3.1); mock-server dry run of the benchmark script.
- Scratch artifacts: `/private/tmp/claude-501/-Users-subhajithait-Documents-testing-drone-jev/0ec4827c-5206-473e-b48d-298a2822fdb4/scratchpad/01/` (`bench_decision_latency.py`, `provider_sketch.py`, `mock_server.py`, `models.json`, `all.json`).

---

## Decisions recommended

1. **Adopt TypeSafe Jev 1.13 as the Jev arm** (`typesafe/jev-1.13`, pinned; never `~typesafe/jev-latest`), and ask the user to confirm this identification. Record the returned snapshot id on every decision.
2. **Make Jev-style probabilistic output the contract of `DecisionProvider`** (action + per-action probabilities + confidence + optional aux scores) and treat text-LLM arms as adapters onto that contract (one-token letter code + `top_logprobs` where available).
3. **Adopt the `DecisionProvider` Protocol + `LatencyInjectingProvider` from section 8** as the integration point; keep fallback/staleness in the harness, never in providers.
4. **Planning latencies**: Jev via OpenRouter p50 about 250 ms, p95 about 600 ms; direct about 120 ms / 250 ms; fast chat LLM about 400 ms / 1 s; slow LLM 1-3 s; local unknown until spike F4. All to be replaced by measurements from the script.
5. **Always use one persistent HTTP/2 client, pinned provider, pinned model, warm-up discard, perf_counter_ns**; run API clients out of the sim process.
6. **Arm set for the API-side comparison**: Jev (OpenRouter and, if available, direct), two fast chat LLMs (one with logprobs: Cerebras gpt-oss-120b; one Groq), one slow LLM, one local System-One stand-in (Ollama `tev1:0.8b`/`nimble` and/or in-process logit model), plus the mock for latency sweeps.
7. **Sub-sample API arms** (about 500-1,000 paired flights) and use latency-injection with cached decisions for the full matrix; estimated API spend stays under about $100 (excluding a reasoning-LLM arm).
8. **State encoding for Jev**: precomputed semantic/bucketed fields, compact, about 300-600 tokens; do not ask it to compute; include `decision_age_ms`. Hand this to agent 07.
9. **Do not claim "Jev is N times faster than LLM" publicly** without the arm definition; the independent paper found 10-37% at median vs a flash LLM.
10. **Treat TypeSafe alpha/early-access status as a project risk**: pin, cache, and keep a local stand-in so the experiment survives API changes.

## Open questions

1. Is the user's "Jev" indeed TypeSafe Jev 1.13? Do they have (or want) a TypeSafe direct key versus OpenRouter only?
2. Where will the experiment run (region/ISP)? This changes all latency numbers by up to 150-250 ms.
3. What are TypeSafe's actual benchmarking/publication terms and their real rate limit (40 req/s vs 20 req/s)? Is early-access throttling applied to OpenRouter traffic differently?
4. Is Jev output deterministic for identical inputs (docs silent)? Does it have server-side caching?
5. Does Jev's latency depend on number of options/questions/state size (vendor and paper say stable; one doc says 255 options)? Does the 11-option Choice + 2 aux questions change p50?
6. Is the `/api/alpha/decisions` path stable (status: alpha)? What is the real path (`/api/v1/systemone` vs alpha)?
7. What does arXiv 2609.30216 ("Jev in the Wild") say about ecosystem/latency? (PDF not readable by my fetcher.)
8. Does `jaredpalmer/kev-4b` (same price as Jev) behave as a System One model, and could it be a second "decision model" arm?
9. Does the user want a Jev-vs-LLM demo that includes a **reasoning-class** LLM (seconds) or only fast LLMs (hundreds of ms)? This determines the "story".
10. What is the real local latency of `tev1:0.8b`/`nimble` and of Qwen-0.5B logits on a base M4 16 GB?

## Follow-up spikes (small, runnable experiments)

- **F1 (30 min, needs key)**: run `bench_decision_latency.py` as is; save JSONL; produce CDFs per arm. Acceptance: Jev p50/p95 within the planning bands or an explanation. Record region, time, and snapshot id.
- **F2 (15 min, key)**: direct TypeSafe vs OpenRouter: same 11-option Choice payload, 200 sequential calls each, same minute. Gives the OpenRouter overhead for Jev specifically.
- **F3 (15 min, key)**: determinism/consistency probe: same state x 100; count distinct chosen options and probability drift; check whether repeats are faster (server cache). Also vary option order and state size (300 vs 1,500 tokens) and number of questions (1 vs 4) to get latency-vs-input and accuracy sensitivity.
- **F4 (1-2 h, no key)**: local stand-ins on this M4: install Ollama >= 0.35, pull `tev1:0.8b` and `nimble`, hit `/v1/systemone` with the same payload, 200 calls; and an MLX/llama.cpp Qwen2.5-0.5B/1.5B letter-logit forward pass. Report p50/p95, RAM, and tokens/s; compare to Jev.
- **F5 (30 min, key)**: sanity-test the action mapping: take 100 scripted scenes with oracle labels (from agent 11), measure Jev top-1 accuracy, confidence calibration (reliability curve), and accuracy vs bucketed-vs-raw-number encodings (feeds agent 07).
- **F6 (15 min, key)**: rate-limit probe: ramp concurrency 1, 4, 16, 32, 64; find the 429/529 onset for Jev and the chosen chat hosts; confirm 20 vs 40 req/s.
- **F7 (15 min, no key)**: read `arxiv.org/pdf/2609.30216` locally (I saved a copy into the session's tool-results dir) and `docs.typesafe.ai/primitives/choice.md`, `/patterns/*` for undocumented limits.

---
