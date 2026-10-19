# 15b — Using a GPT model as a drone DecisionProvider (how-to)

Access date for all web sources: **2026-10-04**. Scratch (code, mock, venv): `/private/tmp/claude-501/-Users-subhajithait-Documents-testing-drone-jev/0ec4827c-5206-473e-b48d-298a2822fdb4/scratchpad/15b/`. No API key was used or exists; **nothing here was run against OpenAI or OpenRouter**.
Legend: **[V]** read on a primary page this session (OpenAI docs/pricing, OpenRouter public model API); **[V-sec]** secondary source (Artificial Analysis, news); **[R]** recalled, not re-verified; **[I]** inference; **[U]** unknown.
Scope note: another agent (15a) decides whether a product literally named "GPT decisions API" exists. This file covers the practical path regardless: **a chat/responses-API GPT model, forced to a one-token or enum answer, optionally with logprobs, as a Jev stand-in** behind the `DecisionProvider` Protocol from `docs/research/01` section 8.2.

---

## 0. TL;DR

1. **There is no decision-typed endpoint in the OpenAI API; you make one** with a tiny non-reasoning/`none`-effort model, a ~600-token static system prompt, `max_completion_tokens: 1` and a **single-token letter alias per action (A..K)**. Verified: all 11 letters are exactly 1 token in `o200k_base` and `cl100k_base` (tiktoken 0.14, scratch).
2. **Probabilities are the catch.** Per-action probabilities need `logprobs`+`top_logprobs`. On OpenRouter's live metadata today, **only the older `gpt-4o-mini` (and gpt-4o/4-turbo/3.5, gpt-oss, image models) list `logprobs`/`top_logprobs`; every GPT-4.1, GPT-5.x and GPT-6 text model drops them** [V, my own call to `GET /api/v1/models`, 466 models]. OpenAI's own GPT-6 guide says `top_logprobs`/`temperature`/`top_p` are legal on Sol/Luna **only when `reasoning.effort = none`** [V], which conflicts with OpenRouter's metadata for `openai/gpt-6-luna` (no `logprobs`). **Whether `gpt-6-luna` + `effort: none` returns logprobs on the live API is the #1 spike (F1).**
3. **Fastest current GPT tier = `gpt-6-luna` with `reasoning_effort: "none"`** ($0.10 in / $0.01 cached / $0.50 out per 1M) [V], then `gpt-4.1-nano` ($0.10/$0.025/$0.40, no reasoning, non-reasoning TTFT 0.63 s on a 10k-token prompt) and `gpt-4o-mini` (the only one with logprobs, TTFT 0.86 s). Anything with reasoning on is out: Luna low = 1.59 s, medium = 5.28 s TTFT [V-sec, Artificial Analysis].
4. **Expect 0.3-0.9 s p50, not Jev's 0.11-0.32 s.** No published number exists for a 700-token prompt with one output token; published AA numbers are for 10k-token inputs (0.63-0.86 s TTFT for the nano/mini class). Plan: p50 400-800 ms, p95 1-2 s via OpenAI direct [I]; OpenRouter adds a hop (25-250 ms, sources disagree, doc 01). **Measure (spike F2).**
5. **Cost is trivial: about $0.00007 per decision (730 input tokens, 1 output) on Luna uncached; about $0.000024 with a prompt cache hit** (but the cache needs a prefix of at least 1,024 tokens and our static prefix is only about 595, so pad it with the 6-8 few-shot examples doc 07 already wants). 2M calls (10k flights x 200) = $147 uncached, $48 cached.
6. **Code** in section 6 (class `GptDecisionProvider`, httpx HTTP/2, `perf_counter_ns`, async, two modes, parse+fallback, OpenRouter routing config) passes `ruff`, `mypy --strict`, `py_compile` on Python 3.14.2 and a mock-server dry run (section 7).

---

## 1. Which GPT models, which tier (as of 2026-10-04)

All prices per 1M tokens, standard tier, from https://developers.openai.com/api/docs/pricing and https://developers.openai.com/api/docs/models [V] unless noted. Context/limits from each model page [V].

| Model | In / cached / out | Reasoning | logprobs (OpenRouter `supported_parameters`) | Endpoints | Notes |
|---|---|---|---|---|---|
| **gpt-6-luna** | $0.10 / $0.01 / $0.50 | none, low, medium (default!), high, xhigh, max | **no** (OpenAI guide says allowed with `none`) | Responses + Chat Completions; **Realtime not supported** | 1.05M ctx, 128k out, cutoff 2026-05-18. Released 2026-09-22 [V-sec]. Default effort is medium: **you must send `none`** |
| gpt-6.1-sol / gpt-6-sol | $2 / $0.10 / $10 | `none` supported on gpt-6-sol and gpt-6-luna; **not** on gpt-6-astra or 6.1-sol (use low) | no | Responses (Chat Completions for tools discouraged) | Too expensive/slow for a 1-token loop |
| gpt-6-astra | $10 / $1 / $50 | no `none` | no | | Flagship, irrelevant here |
| gpt-5.6-luna (older gen) | $0.20 / - / $1.20 (after the 2026-07-30 80% cut) [V-sec: runtimewire] | | | | **Naming caveat**: the pricing page lists both `gpt-5.6-*` and `gpt-6-*`; I could not find a primary page tying them together [U]. Use `gpt-6-luna` |
| gpt-5.4-nano | $0.20 / $0.02 / $1.25 | yes (hidden reasoning tokens) | no | | Dearer than Luna, older |
| gpt-5-nano | $0.05 / $0.005 / $0.40 | `minimal` | no | | Cheapest; reasoning model, can burn hidden tokens |
| **gpt-4.1-nano** | $0.10 / $0.025 / $0.40 | none (non-reasoning) | **no on OpenRouter**; OpenAI model page does not list it; I recall the OpenAI API accepting it [R] | Chat, Responses, Batch; **Realtime not supported** | 1M ctx. AA TTFT 0.63 s (OpenAI) vs 1.44 s (Azure) at 10k input [V-sec]. Rate limit T1 500 RPM/200k TPM |
| **gpt-4o-mini** | $0.15 / $0.075 / $0.60 | none | **yes** (+`logit_bias`, `prediction`) | Chat | Only mainstream cheap model with logprobs; single snapshot `gpt-4o-mini-2024-07-18`, no deprecation date shown. AA TTFT 0.86 s [V-sec] |
| gpt-oss-20b (via OpenRouter hosts) | $0.018-0.075 / out $0.09-0.30 | yes | yes on Darkbloom, AkashML, DekaLLM, CoreWeave, Parasail, Novita; **not** Groq/Bedrock/Vertex/DeepInfra | | "GPT-family" but open weights; the logprobs-capable cheap fallback |

**Which is "lowest latency"?** There is no OpenAI-published TTFT table. Artificial Analysis (10k-token input, 72 h median) [V-sec]: Luna non-reasoning 0.67-0.74 s (124-131 tok/s); Luna low 1.59 s; Luna medium 5.28 s; high 6.69 s; xhigh 28.5 s; gpt-4.1-nano 0.63 s; gpt-4o-mini 0.86 s. At 700 tokens the prefill part shrinks; the remaining floor is routing/queueing, so expect the non-reasoning small class to be close to each other, with Luna `none` and 4.1-nano the best candidates [I].
OpenAI's own latency guide [V]: output-token count is the dominant lever ("cutting 50% of output tokens may cut about 50% of latency"), input-token reduction only "1-5%" unless huge, so **the single-token answer is worth far more than prompt trimming** (this confirms doc 07's point that the prefix is a caching problem, not a trimming problem).

**Reasoning effort**: `none` on Luna removes hidden reasoning (the cause of the 1.6-28 s TTFTs above). With effort above `none`, OpenAI says to **remove `temperature`, `top_p`, `top_logprobs` and Chat Completions `logprobs`** [V]. gpt-5-family: `minimal`; gpt-5.5 guidance: start at `low`, then `none` [V-sec via search summary].

---

## 2. API surface choices

### 2.1 Chat Completions vs Responses

| Question | Chat Completions (`POST /v1/chat/completions`) | Responses (`POST /v1/responses`) |
|---|---|---|
| Works on OpenRouter | yes, the stable path | OpenRouter has it but I did not verify parity [U] |
| 1-token cap | `max_completion_tokens: 1` allowed | param is `max_output_tokens`; **I recall a minimum of 16** [R, unverified]. A one-letter answer still costs 1 generated token, so a cap of 16 is harmless |
| logprobs | `logprobs: true, top_logprobs: N` | `top_logprobs: N` + `include: ["message.output_text.logprobs"]` [V: include value in reference; N=20 in an example from search results] |
| Structured outputs | `response_format: {type:"json_schema", json_schema:{name, strict:true, schema}}` | `text: {format: {type:"json_schema", name, strict:true, schema}}` [V] |
| Forced tool | `tool_choice: {"type":"function","function":{"name":"x"}}` [R] | `tool_choice: {"type":"function","name":"x"}` [V] |
| Persistent socket | not available | **WebSocket mode** `wss://api.openai.com/v1/responses` (section 2.6) |
| OpenAI recommendation | still supported | recommended for new work with tools on Astra/Sol [V] |

**Recommendation: Chat Completions**, because (a) `max_completion_tokens: 1` and (b) the same body works on OpenAI direct, Azure and OpenRouter, so one provider class serves both transports. Use Responses only if the WebSocket spike (F5) pays off.

### 2.2 Forced single-choice with logprobs (mode A: "Jev-like probabilities")

Request (Chat Completions; `tools`/`response_format` absent):

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "<static ~600-1100 token prefix: role, frame, glossary, alias table A..K, 6-8 shots>"},
    {"role": "user", "content": "age=120 last=FWD v=2.0 bat=80 gps=ok\ntgt=12f,0l,0u ...\nAnswer with one letter."}
  ],
  "max_completion_tokens": 1,
  "temperature": 0,
  "logprobs": true,
  "top_logprobs": 20,
  "stream": false
}
```

How the probabilities are read: `choices[0].logprobs.content[0].top_logprobs` is a list of `{token, logprob}` for the **first generated token**. Take the entries whose `token.strip().upper()` is one of A..K, `exp(logprob)`, sum duplicates (`"F"`, `" F"`, `"f"`), renormalise over the 11. Also keep the **un-renormalised alias mass** (`alias_mass`): if it is well below 1, the model wanted to say something that is not a valid letter (a free quality signal Jev does not give you).
Rules that matter:
- **`top_logprobs` must be at least 11, so set 20**. The older OpenAI cookbook says 0-5 [V: cookbook page], but the current Chat Completions limit is 20 [R] and a Responses example uses 20 [V-sec, search result]. If a host caps lower, probabilities are truncated: detect via `alias_mass`.
- **Do not combine with `logit_bias` to force the alphabet** unless tested: I could not find documentation on whether returned logprobs are pre- or post-bias [U]. Without bias, mass outside A..K is simply measured, which is better data.
- `temperature: 0` does **not** change the logprobs (they are the model distribution), only which token is sampled; we take argmax of the renormalised distribution ourselves, so the sampled token is irrelevant (the provider ignores `message.content` when logprobs exist).
- Probabilities from instruction-tuned models are **not calibrated** (tend to be overconfident); calibrate thresholds on your own rollouts, same as doc 01 says for Jev.
- Alias bias: letters carry no semantics; position bias (A first) is a known LLM effect. Mitigate by shuffling nothing (breaks caching) but **balancing few-shots across letters** and ablating (doc 07 factor OUT).

### 2.3 Structured Outputs with an enum (mode B)

```json
"response_format": {"type": "json_schema", "json_schema": {"name": "drone_action", "strict": true,
  "schema": {"type":"object","properties":{"action":{"type":"string","enum":["FORWARD","LEFT",...]}},
             "required":["action"],"additionalProperties":false}}},
"max_completion_tokens": 32
```
- Guarantees a valid enum member (or a `refusal` field, which the provider must check) [V]. `additionalProperties:false` is mandatory [V]. **First request with a new schema pays extra latency, later ones do not [V]**, so warm up with the real schema (the provider's `warm_up` only opens the socket; send one real decision before timing).
- Output is about 7-8 tokens (`{"action":"AVOID_LEFT"}` = 7 tokens in o200k) versus 1: by the OpenAI rule of thumb that is several times more decode time, but at 100+ tok/s it is ~60 ms more [I]. **No probabilities.** Works on every model including reasoning ones and OpenRouter hosts that implement structured outputs.
- Giving the enum single-letter values (`"enum":["A",...,"K"]`) cuts tokens to about 5; not done by default (keeps names greppable).

### 2.4 Function calling with forced `tool_choice`

Same guarantee as structured outputs but extra prompt tokens (tool schema) and more output tokens (tool-call envelope). `allowed_tools` exists to keep the tool list stable for caching [V]. **Not recommended; strictly dominated by mode B** for a single choice. Only use it if a provider supports tools but not `response_format` (rare).

### 2.5 `logit_bias`, `seed`, `temperature`, determinism

- `logit_bias` (-100..100, token-id keyed) [R]: OpenRouter lists it for `gpt-4o-mini` only among GPT models; **not for Luna/4.1-nano/5.x** [V]. Skip.
- `seed`: listed on all GPT models in OpenRouter metadata [V]; OpenAI documents it as best effort. `system_fingerprint`: returned by Chat Completions on older models; **I could not confirm GPT-6 returns it** [U]; the provider records it when present (`provider_id` ends `@<fingerprint|na>`). Log `response.model` (dated snapshot) on every call; GPT-6 pages say to use alias `gpt-6-luna` with no dated snapshot shown [V], so **drift cannot be pinned**: the consistency probe (same input x 50) belongs in spike F3.
- Logprob values at `temperature 0` are still run-to-run noisy (GPU batching) [I]; do not use thresholds that need 3-decimal precision.

### 2.6 Prompt caching, streaming, predicted outputs, WebSocket, Realtime

| Feature | Facts | Use here |
|---|---|---|
| **Prompt caching** [V: prompt-caching guide] | min **1,024** visible input tokens (GPT-5.6+); older models: depends on settings, cached token counts round down to 128 increments; hit = `usage.prompt_tokens_details.cached_tokens` (Chat) / `usage.input_tokens_details.cached_tokens` (Responses). Discount: cached input = 0.1x (Luna: $0.01 vs $0.10) and 0.05x on 6.1 Sol. **GPT-5.6+ also bills cache writes at 1.25x** (unclear whether gpt-6-luna follows [U]). >15 requests/min on one prefix may overflow-route (lower hit rate). `prompt_cache_key` (stable id) recommended; GPT-6 replaces `prompt_cache_retention` with `prompt_cache_options.ttl` (e.g. `"30m"`) | **Our ~595-token static prefix is below 1,024: no cache.** Pad it to >=1,100 with the balanced 6-8 shots from doc 07 section 5 (also improves accuracy). Latency gain is **not quantified** by OpenAI ("reduces time spent processing input") [V]; on a 1k prefix expect small, tens of ms [I]. Static first, observation last (already the layout) |
| **Streaming / TTFT** | SSE; for a 1-token answer TTFT = total | Do not stream in production. Streaming only for timing studies |
| **Predicted outputs** | Listed on 4.1-nano/4o-mini, absent on Luna [V] | Useless for 1 token. Skip |
| **Responses WebSocket mode** | `wss://api.openai.com/v1/responses`, events `response.create`; up to 60 min per connection, 16 in-flight responses, 32 `stream_id`s; "up to ~40% faster end-to-end" for **rollouts with 20+ tool calls** [V]; examples use gpt-6-astra; says nothing about stateless one-shot calls [V] | A persistent HTTP/2 connection already removes TLS/TCP setup; WS likely saves little (tens of ms) for stateless calls [I]. Spike F5 |
| **Realtime API** | gpt-realtime/gpt-live are audio/voice models; **Luna and 4.1-nano pages list Realtime as not supported** [V] | Not applicable |
| **Batch** | 50% off, asynchronous | Offline oracle labelling for the 2,000-observation ablation (doc 07), not the control loop |

### 2.7 Service tiers (speed vs price)

| Tier | How | Price | Latency effect | Verdict |
|---|---|---|---|---|
| Standard | default | table above | baseline | default |
| **Fast mode (formerly Priority)**, renamed 2026-07-30 [V-sec: OpenAI page via search] | `"service_tier": "fast"` (or legacy `"priority"`) | premium; **2x** on Luna per OpenRouter's `openai/fast` endpoint ($0.20/$1.00) [V: endpoint metadata]; gpt-6-astra $20/$100 [V: pricing page] | OpenAI quantifies "up to 2.5x faster" **token generation for gpt-5.6-sol**, plus 99.9% uptime SLA, "more consistent" speed [V-sec]; nothing quantified for TTFT/1-token loops. Ramp limit: >=1M TPM and >50% rise in 15 min can downgrade requests to standard (response shows `service_tier: "default"`) [V-sec] | Worth one A/B in spike F2: for a 1-token output, generation speed is irrelevant, so the gain is only less queueing/jitter [I] |
| **Flex** | `"service_tier": "flex"` | = Batch price (Luna $0.05/$0.25) [V] | slower, may be rate-limited/unavailable [R] | Offline arms only (prompt ablation) |
| Batch | Batch API | 50% | up to 24 h | Offline only |

**Check `response.service_tier`** in the logs of every call to see which tier actually served it.

### 2.8 Rate limits and spend tiers [V: model pages, rate-limits guide]

| Model | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| gpt-6-luna RPM / TPM | 500 / 500k | 5,000 / 2M | 5,000 / 4M | 10,000 / 10M | 30,000 / 180M |
| gpt-4.1-nano RPM / TPM | 500 / 200k | 5,000 / 2M | 5,000 / 4M | 10,000 / 10M | 30,000 / 150M |
| gpt-4o-mini RPM / TPM | 500 / 200k | 5,000 / 2M | 5,000 / 4M | 10,000 / 10M | 30,000 / 150M |

Tier qualification: T1 $5 paid, T2 $50, T3 $100, T4 $250, T5 $1,000. Limits count the **maximum of `max_tokens` and an estimated count from request characters**, so keep `max_completion_tokens` at 1 [V]. Ramp rule: from 1M TPM up, increase no more than 50% per 15 min or get a `slow_down` error [V].
Arithmetic for the plan: 730 tokens x 5,000 RPM = 3.65M TPM: **at Tier 2-3 the RPM (5,000) binds, not TPM**; Tier 1's 500 RPM = 8.3 req/s which is fine for 1-4 concurrent flights only. 2M calls at 5,000 RPM = 6.7 h; at 10,000 RPM = 3.3 h (vs Jev's 20 req/s cap, doc 01: 28 h).

### 2.9 Region and data residency

Regional processing endpoints carry a **10% uplift** for models released on/after 2026-03-05 [V]; FedRAMP also +10%; **Fast mode is unavailable with EU data residency for the 6.x models** [V: GPT-6 guide]. On OpenRouter, Luna has endpoints: OpenAI, `openai/flex`, `openai/fast`, Azure, `azure/us`, `azure/eu` (+10%), Amazon Bedrock `us-east-1` (+10%) [V: `GET /api/v1/models/openai/gpt-6-luna/endpoints`]. Azure latency was 2.3x worse than OpenAI direct for 4.1-nano in AA's data (1.44 s vs 0.63 s) [V-sec], so **pin OpenAI**, not Azure, for the latency arm. The user's region matters (doc 01: US West origin for Jev; OpenAI front doors are US/EU edge-routed, not measured here [U]).

---

## 3. Published latency evidence (all secondary; none is for our shape)

| Source | Setup | Number | Applicability |
|---|---|---|---|
| Artificial Analysis, `gpt-6-luna-non-reasoning/providers` | 10k input, median 72 h, OpenAI only | TTFT 0.67 s, 124 tok/s; blended $0.08/M | upper bound for our 0.7k prompt [V-sec] |
| AA, other effort levels | same | low 1.59 s, medium 5.28 s, high 6.69 s, xhigh 28.5 s | why `none` is mandatory [V-sec] |
| AA gpt-4.1-nano | 10k input | OpenAI 0.63 s, Azure 1.44 s | [V-sec] |
| AA gpt-4o-mini | 10k input | OpenAI 0.86 s, Azure 1.86 s | [V-sec] |
| OpenRouter blog (doc 01 section 3.2) "GPT Luna" | 1 req/example, ticket classification | p50 805-1,106 ms, p95 1,491-2,395 ms | likely default effort, model gen unclear; Jev p50 194 ms in the same test [V-sec, from doc 01] |
| OpenRouter model pages / endpoints API | unauthenticated | `latency_last_30m` = null for every Luna endpoint | live OpenRouter latency stats require a key [V] |
| OpenAI | no published TTFT table; guide only says output tokens dominate | | |

**Planning numbers [I]**: OpenAI direct, Luna `none` or 4.1-nano, 730-token prompt, 1 output token: p50 **0.35-0.8 s**, p95 **0.8-2 s**, p99 **1.5-4 s**; Fast mode may tighten p95. OpenRouter adds 25-250 ms (doc 01). That is **roughly 2-4x Jev's p50**, partly offset by finer probability resolution (continuous logprobs vs Jev's coarse plateaus), an honest trade the demo should show: "GPT-as-decision-provider" is the *fast small LLM* arm in doc 01's framing, not a Jev-class one.

---

## 4. OpenRouter variant

Endpoint `POST https://openrouter.ai/api/v1/chat/completions`, `Authorization: Bearer $OPENROUTER_API_KEY`; optional `HTTP-Referer`/`X-Title` headers. Differences from OpenAI direct:

| Item | Direct OpenAI | OpenRouter |
|---|---|---|
| Model id | `gpt-6-luna` | `openai/gpt-6-luna` (`:batch` suffix exists; `:flex`/`fast` are endpoint tags, see below) |
| Reasoning | `"reasoning_effort": "none"` | `"reasoning": {"effort": "none"}` (OpenRouter-normalised; its model page lists `reasoning`, `reasoning_effort`, `verbosity`) [V] |
| Routing | n/a | `"provider": {"order": ["OpenAI"], "allow_fallbacks": false, "require_parameters": true, "data_collection": "deny"}`; also `only`, `ignore`, `sort` (`price`/`throughput`/`latency`), `zdr`, `max_price`, `preferred_max_latency` (p50 or per-percentile, soft) [V]. **Setting `order` or `sort` disables load balancing** [V]. Endpoint tags from the endpoints API: `openai`, `openai/flex`, `openai/fast`, `azure`, `azure/us`, `azure/eu`, `amazon-bedrock/us-east-1` [V] |
| Service tier | `"service_tier": "fast"` | pick tag `openai/fast` in `provider.order` (2x price) [V] |
| `prompt_cache_key` | yes | do not send (unverified pass-through) [U]; use sticky `session_id` if wanted (doc 01 section 2.2) |
| logprobs | per model | **`require_parameters: true` + `logprobs` + Luna = no eligible endpoint** (no Luna endpoint lists it [V]); for logprobs use `openai/gpt-4o-mini` (OpenAI + Azure list `logprobs`, `top_logprobs`, `logit_bias`) or `openai/gpt-oss-20b` on `CoreWeave`/`Parasail`/`Novita` [V] |
| Extra hop | 0 | 25-250 ms, inconsistent evidence (doc 01 section 3.1) |
| Credits | prepaid | low credit balance can add latency (doc 01 section 2.2) |

`openrouter_gpt()` in the code is the ready-made config.

---

## 5. Token and cost estimate for the drone prompt

Measured with tiktoken 0.14 `o200k_base` (scratch): doc 07's system prompt = **535 tokens**; the 11-line alias glossary = **60**; so **static prefix = 595**; observation (format F) = **135** (doc 07); total **730 input**; output **1** token (mode A) or **7-8** (mode B). Prefix is **below the 1,024 cache minimum**; padded variant = 1,024 cached + 135 fresh.

| Model | Per call, uncached | Per call, cache hit (prefix padded to 1,024) | 200 calls (one flight) | 1,000 flights (uncached / cached) | 10,000 flights (uncached / cached) |
|---|---|---|---|---|---|
| gpt-6-luna (std) | $0.0000735 | $0.0000242 | $0.0147 / $0.0048 | $14.7 / $4.8 | $147 / $48 |
| gpt-6-luna flex or batch | $0.0000367 | $0.0000121 | $0.0073 / $0.0024 | $7.3 / $2.4 | $74 / $24 |
| gpt-6-luna fast (2x) | $0.000147 | $0.0000485 | $0.029 / $0.0097 | $29 / $9.7 | $294 / $97 |
| gpt-4.1-nano | $0.0000734 | $0.0000395 | $0.0147 / $0.0079 | $14.7 / $7.9 | $147 / $79 |
| gpt-4o-mini (logprobs) | $0.000110 | $0.0000976 | $0.022 / $0.0195 | $22 / $19.5 | $220 / $195 |
| gpt-5.4-nano | $0.000147 | $0.0000487 | $0.0295 / $0.0097 | $29 / $9.7 | $294 / $97 |
| gpt-5-nano (+ hidden reasoning tokens, not counted) | $0.0000369 | $0.0000123 | $0.0074 / $0.0025 | $7.4 / $2.5 | $74 / $25 |
| gpt-6.1-sol (reference, too slow) | $0.00147 | $0.00038 | $0.29 / $0.077 | $294 / $77 | $2,940 / $765 |

Structured mode (+7 output tokens) adds $0.0000035 per Luna call (negligible). The cache-write 1.25x surcharge (if it applies to Luna) is not modelled; it hits only the first call per 30 min/prefix. Compare doc 01: Jev ~$0.000042 per call (1k tokens): **Luna uncached is about 1.7x Jev, cached about 0.6x**. Budget for all API arms at 1,000 flights each stays under about $50.

---

## 6. Provider implementation (copy-paste)

Target: `app/providers/gpt.py` (+ DTOs in `app/schemas/decision.py`). Conforms to `DecisionProvider` (`provider_id`, `decide`, `aclose`; `capabilities` is trivial to add: `returns_probabilities = mode is LOGPROBS`, `deterministic=False`, `is_remote=True`). Matches repo rules: class-based with injected client, full type hints, `%s` lazy logging, `logger.exception` on unexpected transport failure, no `print`, no raw dicts out, no fallback decisions inside the provider (it raises typed errors; hold-last/reflex/HOVER is the harness's job per doc 07 section 6). Design points:
- **Persistent client** (`make_client`: `http2=True`, 32 keep-alive connections) is injected; call `await provider.warm_up()` and one throwaway decision before timing.
- **Timing**: `perf_counter_ns` around the whole call; `client.stream()` is used so `wall_ttfb_ms` = time to response headers; `server_ms` = OpenAI's `openai-processing-ms` header if present.
- **Deadline**: `asyncio.timeout(request.deadline_ms/1000)` -> `ProviderTimeoutError`. No retries in the loop.
- **Parse + fallback ladder inside the provider**: (A) top_logprobs -> renormalised dict; if the host returned no logprobs, (B) parse `message.content` (alias letter, exact name, or *unique* name; ambiguity raises `InvalidDecisionOutputError`); (C) structured mode: JSON enum, then the same tolerant text parse. Refusal or `finish_reason`-truncated empties raise `InvalidDecisionOutputError`.
- Pricing is config, so `cost_usd` per decision is exact for the configured tier; `cache_hit` reads `cached_tokens`.
- Probabilities are only populated when more than one alias came back; `aux["alias_mass"]` is the un-renormalised mass on valid aliases (a gating signal).

```python
"""GPT-family DecisionProvider (OpenAI direct or OpenRouter) over Chat Completions.

Sketch for app/providers/gpt.py. Stand-ins for the repo types (see docs/research/01 section 8.2)
are in the marked block so this file type-checks alone; delete the block and import the real ones.

Two modes:
  LOGPROBS    one output token (letter alias A..K) + top_logprobs=20 -> per-action probabilities.
  STRUCTURED  json_schema strict enum -> one action, no probabilities (works on reasoning models).
"""

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# --- stand-ins (replace with app.schemas.decision / app.core.exceptions imports) ---------------
class ProviderTimeoutError(Exception): ...


class ProviderRateLimitedError(Exception): ...


class ProviderUnavailableError(Exception): ...


class InvalidDecisionOutputError(Exception): ...


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
    model_config = ConfigDict(frozen=True)
    request_id: str
    state: Mapping[str, object]
    allowed_actions: tuple[Action, ...] = tuple(Action)
    sim_time_s: float
    deadline_ms: float = 1000.0


class DecisionTiming(BaseModel):
    wall_total_ms: float
    wall_ttfb_ms: float | None = None
    wall_first_token_ms: float | None = None
    server_ms: float | None = None
    sim_latency_ms: float | None = None
    connection_reused: bool | None = None
    retries: int = 0


class DecisionResult(BaseModel):
    action: Action
    probabilities: dict[Action, float] = Field(default_factory=dict)
    confidence: float | None = None
    aux: dict[str, float] = Field(default_factory=dict)
    timing: DecisionTiming
    provider_id: str
    cache_hit: bool = False
    input_tokens: int | None = None
    cost_usd: float | None = None


# --- wire constants -----------------------------------------------------------------------------
class Mode(StrEnum):
    LOGPROBS = "logprobs"
    STRUCTURED = "structured"


# Single-token (o200k and cl100k verified) aliases. Order is the glossary order in the prompt.
ALIASES: Final[dict[Action, str]] = {
    action: letter for action, letter in zip(Action, "ABCDEFGHIJK", strict=True)
}
ACTION_BY_ALIAS: Final[dict[str, Action]] = {v: k for k, v in ALIASES.items()}
ACTION_SCHEMA: Final[dict[str, Any]] = {
    "name": "drone_action",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": [a.value for a in Action]}},
        "required": ["action"],
        "additionalProperties": False,
    },
}
TOP_LOGPROBS: Final[int] = 20  # chat completions max; 11 aliases need > 5 slots
_NAME_PAT: Final[re.Pattern[str]] = re.compile(
    r"\b(" + "|".join(sorted((a.value for a in Action), key=len, reverse=True)) + r")\b"
)


@dataclass(frozen=True)
class GptConfig:
    base_url: str  # "https://api.openai.com/v1" or "https://openrouter.ai/api/v1"
    api_key: str  # injected from settings; never logged
    model: str  # "gpt-6-luna" | "openai/gpt-6-luna" | "gpt-4o-mini" ...
    system_prompt: str  # static prefix (>= 1024 tokens to be cacheable on most models)
    mode: Mode = Mode.LOGPROBS
    # Merged into the JSON body last: reasoning_effort, service_tier, provider routing, ...
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    price_in_per_m: float = 0.0
    price_cached_per_m: float = 0.0
    price_out_per_m: float = 0.0
    cache_key: str | None = (
        None  # prompt_cache_key (OpenAI direct); stable per experiment arm
    )


# --- response DTOs (only what we read; everything else ignored) ---------------------------------
class _TopLogprob(BaseModel):
    token: str
    logprob: float


class _TokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: list[_TopLogprob] = Field(default_factory=list)


class _Logprobs(BaseModel):
    content: list[_TokenLogprob] | None = None


class _Message(BaseModel):
    content: str | None = None
    refusal: str | None = None


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None
    logprobs: _Logprobs | None = None


class _PromptDetails(BaseModel):
    cached_tokens: int = 0


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: _PromptDetails | None = None


class _ChatResponse(BaseModel):
    model: str = ""
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choices: list[_Choice]
    usage: _Usage = Field(default_factory=_Usage)


# --- pure helpers (move to utils/ if reused) ----------------------------------------------------
def glossary(prefix: str = "") -> str:
    """Alias glossary appended to the static system prompt (built from the enum: DRY)."""
    return prefix + "\n".join(f"{ALIASES[a]} = {a.value}" for a in Action)


def render_observation(state: Mapping[str, object]) -> str:
    text = state.get(
        "text"
    )  # harness-rendered format F (docs/research/07) when present
    return text if isinstance(text, str) else json.dumps(state, separators=(",", ":"))


def probabilities_from_top_logprobs(
    top: list[_TopLogprob], allowed: tuple[Action, ...]
) -> tuple[dict[Action, float], float]:
    """Renormalise first-token top_logprobs over the allowed aliases.

    Returns (probabilities, alias_mass). alias_mass is the raw probability the model put on valid
    aliases; a low value means it wanted to say something else (a quality signal, not an error).
    """
    mass: dict[Action, float] = {}
    for item in top:
        action = ACTION_BY_ALIAS.get(item.token.strip().upper())
        if action is not None and action in allowed:
            mass[action] = mass.get(action, 0.0) + math.exp(item.logprob)
    total = sum(mass.values())
    if total <= 0.0:
        raise InvalidDecisionOutputError("no allowed alias among top_logprobs")
    return {a: p / total for a, p in mass.items()}, total


def parse_text_action(text: str) -> Action:
    """Fallback parser: alias letter, exact name, or a UNIQUE name inside the text. Never guess."""
    t = text.strip().strip("`*.\"' ").upper().replace(" ", "_").replace("-", "_")
    if t in ACTION_BY_ALIAS:
        return ACTION_BY_ALIAS[t]
    if t in Action.__members__:
        return Action(t)
    found = set(_NAME_PAT.findall(text.upper().replace(" ", "_").replace("-", "_")))
    if len(found) == 1:
        return Action(found.pop())
    raise InvalidDecisionOutputError("unparseable or ambiguous action text")


def build_body(cfg: GptConfig, request: DecisionRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {
                "role": "system",
                "content": cfg.system_prompt,
            },  # static prefix FIRST (caching)
            {"role": "user", "content": render_observation(request.state)},
        ],
        "stream": False,
    }
    if cfg.mode is Mode.LOGPROBS:
        body |= {
            "max_completion_tokens": 1,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "temperature": 0,
        }
    else:
        body |= {
            "max_completion_tokens": 32,
            "response_format": {"type": "json_schema", "json_schema": ACTION_SCHEMA},
        }
    if cfg.cache_key:
        body["prompt_cache_key"] = cfg.cache_key
    merged = body | dict(cfg.extra_body)  # extra_body wins; a None value DROPS the key
    return {k: v for k, v in merged.items() if v is not None}


class GptDecisionProvider:
    """DecisionProvider over Chat Completions. Client is injected (persistent, HTTP/2)."""

    def __init__(self, client: httpx.AsyncClient, cfg: GptConfig) -> None:
        self._client = client
        self._cfg = cfg
        self._headers = {"Authorization": f"Bearer {cfg.api_key}"}
        self._url = f"{cfg.base_url.rstrip('/')}/chat/completions"

    @property
    def provider_id(self) -> str:
        return f"gpt-{self._cfg.mode}:{self._cfg.model}"

    async def warm_up(self) -> None:
        """Open the TLS/HTTP2 connection before timing starts (cheap GET, result ignored)."""
        try:
            await self._client.get(
                f"{self._cfg.base_url.rstrip('/')}/models",
                headers=self._headers,
                timeout=5.0,
            )
        except httpx.HTTPError:
            logger.warning("Warm-up request failed for %s", self.provider_id)

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        body = build_body(self._cfg, request)
        t0 = time.perf_counter_ns()
        try:
            async with asyncio.timeout(request.deadline_ms / 1000.0):
                async with self._client.stream(
                    "POST", self._url, json=body, headers=self._headers
                ) as resp:
                    ttfb_ms = (time.perf_counter_ns() - t0) / 1e6  # headers received
                    raw = await resp.aread()
        except TimeoutError as exc:
            logger.warning(
                "Decision timeout after %.0f ms (%s)",
                request.deadline_ms,
                self.provider_id,
            )
            raise ProviderTimeoutError(self.provider_id) from exc
        except httpx.HTTPError as exc:
            logger.exception("Transport error (%s)", self.provider_id)
            raise ProviderUnavailableError(self.provider_id) from exc
        total_ms = (time.perf_counter_ns() - t0) / 1e6
        self._raise_for_status(resp.status_code)
        server_ms = _float_header(resp.headers.get("openai-processing-ms"))
        timing = DecisionTiming(
            wall_total_ms=total_ms, wall_ttfb_ms=ttfb_ms, server_ms=server_ms
        )
        return self._to_result(raw, request, timing)

    def _raise_for_status(self, status: int) -> None:
        if status < 400:
            return
        logger.warning(
            "HTTP %s from %s", status, self.provider_id
        )  # never log body/keys
        if status in (429, 529):
            raise ProviderRateLimitedError(self.provider_id)
        if status in (408, 504, 524):
            raise ProviderTimeoutError(self.provider_id)
        raise ProviderUnavailableError(f"{self.provider_id} status={status}")

    def _to_result(
        self, raw: bytes, request: DecisionRequest, timing: DecisionTiming
    ) -> DecisionResult:
        try:
            data = _ChatResponse.model_validate_json(raw)
            choice = data.choices[0]
        except (ValueError, IndexError) as exc:
            raise InvalidDecisionOutputError("malformed completion payload") from exc
        if choice.message.refusal:
            raise InvalidDecisionOutputError("model refused")
        probs, mass = self._extract(choice, request.allowed_actions)
        action = max(probs, key=lambda a: probs[a])
        used = data.usage
        cached = (
            used.prompt_tokens_details.cached_tokens
            if used.prompt_tokens_details
            else 0
        )
        aux = {"alias_mass": mass} if mass is not None else {}
        return DecisionResult(
            action=action,
            probabilities=probs if len(probs) > 1 else {},
            confidence=probs[action] if len(probs) > 1 else None,
            aux=aux,
            timing=timing,
            provider_id=f"gpt-{self._cfg.mode}:{data.model or self._cfg.model}"
            f"@{data.system_fingerprint or 'na'}",
            cache_hit=cached > 0,
            input_tokens=used.prompt_tokens,
            cost_usd=self._cost(used.prompt_tokens, cached, used.completion_tokens),
        )

    def _extract(
        self, choice: _Choice, allowed: tuple[Action, ...]
    ) -> tuple[dict[Action, float], float | None]:
        if self._cfg.mode is Mode.LOGPROBS:
            tokens = choice.logprobs.content if choice.logprobs else None
            if tokens and tokens[0].top_logprobs:
                return probabilities_from_top_logprobs(tokens[0].top_logprobs, allowed)
            logger.warning(
                "No logprobs returned by %s; falling back to text parse",
                self.provider_id,
            )
            return {parse_text_action(choice.message.content or ""): 1.0}, None
        content = choice.message.content or ""
        try:
            return {Action(json.loads(content)["action"]): 1.0}, None
        except ValueError, KeyError, TypeError:
            return {
                parse_text_action(content): 1.0
            }, None  # truncated/odd JSON: tolerant parse

    def _cost(self, prompt: int, cached: int, completion: int) -> float:
        c = self._cfg
        return (
            (prompt - cached) * c.price_in_per_m
            + cached * c.price_cached_per_m
            + completion * c.price_out_per_m
        ) / 1e6

    async def aclose(self) -> None:
        await self._client.aclose()


def _float_header(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def make_client() -> httpx.AsyncClient:
    """One per process. HTTP/2 + keep-alive; deadline is enforced per call, not here."""
    return httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(10.0, connect=3.0),
        limits=httpx.Limits(
            max_connections=32, max_keepalive_connections=32, keepalive_expiry=120.0
        ),
    )


# --- example configs ----------------------------------------------------------------------------
def openai_direct_luna(api_key: str, system_prompt: str, mode: Mode) -> GptConfig:
    # gpt-6-luna: reasoning_effort "none" is required for speed; sampling params are only legal
    # with "none". logprobs on Luna is documented as allowed with none but NOT yet confirmed live.
    return GptConfig(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model="gpt-6-luna",
        system_prompt=system_prompt,
        mode=mode,
        cache_key="drone-armA",
        extra_body={
            "reasoning_effort": "none"
        },  # + {"service_tier": "fast"} to buy speed (2x)
        price_in_per_m=0.10,
        price_cached_per_m=0.01,
        price_out_per_m=0.50,
    )


def openrouter_gpt(
    api_key: str, system_prompt: str, mode: Mode, model: str
) -> GptConfig:
    routing = {
        "provider": {
            "order": ["OpenAI"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
    }
    reasoning = {"reasoning": {"effort": "none"}} if "luna" in model else {}
    return GptConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        mode=mode,
        extra_body=routing | reasoning,
    )
```

Usage sketch:

```python
cfg = openai_direct_luna(
    settings.openai_api_key, SYSTEM_PROMPT_PLUS_GLOSSARY, Mode.STRUCTURED
)
provider = GptDecisionProvider(
    make_client(), cfg
)  # one per process, injected via Depends()/factory
await provider.warm_up()
result = await provider.decide(
    DecisionRequest(
        request_id="r1", sim_time_s=t, state={"text": obs_text}, deadline_ms=1500
    )
)
```
For `Mode.LOGPROBS` on Luna add nothing (the body already carries `logprobs`/`top_logprobs`); if the API rejects them with `reasoning_effort: none` the 400 surfaces as `ProviderUnavailableError`: fall back to `gpt-4o-mini` (direct `openai_direct_luna` with `model="gpt-4o-mini"` and `extra_body={}`) or STRUCTURED. Settings fields to add (`app/core/config.py`): `openai_api_key`, `openrouter_api_key` (+ base URLs/model ids). Not added here (task said touch no repo file).

Known sharp edges in the sketch:
1. `temperature: 0` is sent in LOGPROBS mode; a reasoning-capable model called without `none` rejects it. `extra_body` is merged last and **a `None` value drops the key**, so pass `{"temperature": None}` for such models.
2. `prompt_cache_key` is set by `openai_direct_luna` only; not sent to OpenRouter.
3. `warm_up` does `GET {base}/models`; on OpenAI that lists models (cheap, but requires a valid key); harmless if it fails.
4. `http2=True` was only constructed, not exercised: the mock is plaintext HTTP/1.1 (httpx does not do h2c). HTTP/2 over TLS to OpenAI/OpenRouter is untested here.

---

## 7. Validation performed

In scratch (`.venv`, Python 3.14.2, httpx 0.28.x with `h2`, pydantic, ruff, mypy):
- `ruff check` clean (rules E,F,I,UP,B,ASYNC; line length 100), `ruff format` applied, **`mypy --strict` clean**, `py_compile` clean on the provider; the dry-run script is also ruff/mypy clean after autofix (E501 ignored there).
- `dryrun.py` against a local stdlib mock server (`ThreadingHTTPServer`), all assertions pass:
  - LOGPROBS body: `max_completion_tokens == 1`, `logprobs`, `top_logprobs == 20`, system message first, bearer header present; probabilities renormalised over aliases (`F` plus `" f"` summed, non-alias tokens `The` ignored), `alias_mass`, `cache_hit` from `cached_tokens`, `server_ms` from header, `cost_usd` exact to 1e-12.
  - STRUCTURED body: strict json_schema, `reasoning_effort` pass-through via `extra_body`; returns `AVOID_LEFT`, no probabilities.
  - Missing logprobs -> text fallback (`"brake."` -> BRAKE); 429 -> `ProviderRateLimitedError`; refusal and ambiguous text (`"LEFT or RIGHT"`) -> `InvalidDecisionOutputError`; deadline 100 ms vs 500 ms server -> `ProviderTimeoutError`.
  - 50 sequential keep-alive calls: loopback p50 0.95 ms, p95 1.44 ms, i.e. client-side overhead (build, serialise, parse) is **about 1 ms**, negligible next to the 300+ ms network/model time.
- **Not validated**: any real OpenAI/OpenRouter response shape (the mock mirrors the documented Chat Completions shape from memory [R]); HTTP/2 over TLS; GPT-6 behaviour with `reasoning_effort: none` and logprobs.

---

## 8. Decisions recommended

1. **Treat "GPT decisions API" as "fast small GPT via Chat Completions with a one-token answer"**; implement as one `GptDecisionProvider` with a `Mode` switch, not as separate classes. Keep Jev as the primary arm; this is an additional arm labelled "gpt-6-luna(none)" / "gpt-4o-mini(logprobs)".
2. **Default arm config**: `gpt-6-luna`, `reasoning_effort: "none"`, STRUCTURED mode, `max_completion_tokens` 32, OpenAI direct (not OpenRouter, not Azure), standard tier. If F1 shows Luna returns logprobs with `none`, switch to LOGPROBS mode (1 token) as default.
3. **Probability-bearing arm** (for Jev-comparable confidence gating): `gpt-4o-mini` + LOGPROBS today (only mainstream GPT with logprobs); note it is an older, slower (AA 0.86 s) model and a different intelligence tier, so label it as a separate arm.
4. **Pad the static prefix to >=1,100 tokens with the 6-8 balanced few-shots** so prompt caching engages (and set `prompt_cache_key` per arm); static first, observation last.
5. **Always set reasoning to `none`** on 6.x/5.x models; refuse to run an arm whose response reports reasoning tokens > 0 (check `usage.completion_tokens_details.reasoning_tokens` [R]).
6. **Use letters A..K as wire aliases** (verified 1 token each); keep names in the DB; ablate letters vs names (doc 07 factor OUT).
7. **Never fall back inside the provider**; raise typed errors, harness applies hold-last/reflex/HOVER; report timeout/invalid rates (doc 07 section 6).
8. **Budget**: API-GPT arms at 1,000 flights each = about $15-30 per arm (uncached); fine without sub-sampling on cost, only on RPM (Tier 2 needs $50 paid, Tier 4 $250 for 10k RPM).
9. **Record per call**: `response.model`, `system_fingerprint` (if any), `service_tier`, `cached_tokens`, `alias_mass`, wall timings, request id header; interleave arms in time (doc 01 section 5).

## 9. Open questions

1. Does the product the user calls "GPT decisions API" exist (15a)? If it is a Jev-style typed endpoint, this whole file is only the fallback.
2. **Does `gpt-6-luna` with `reasoning_effort: none` actually return `logprobs`/`top_logprobs` on OpenAI direct?** OpenAI's guide says allowed [V], OpenRouter's metadata says not listed [V]. Also `logit_bias`, `seed`, `system_fingerprint` on Luna: unknown.
3. Is `gpt-6-luna` the same family as `gpt-5.6-luna` (price $0.20/$1.20) or a successor? Which snapshot will the alias move to, and can drift be pinned?
4. Does the 1.25x cache-write surcharge and the 1,024-token minimum apply to `gpt-6-luna` (documented for "GPT-5.6+")? What is the real hit rate at ~1 request per 100-300 ms per arm (the >15 rpm per-prefix overflow note)?
5. Does Chat Completions accept `max_completion_tokens: 1` on Luna, and what is Responses' `max_output_tokens` minimum (recall: 16)?
6. Does Fast mode (`service_tier: fast`) help p50/p95 for 1-token outputs, or only generation throughput? Is it enabled for Luna on direct (OpenRouter lists `openai/fast` for Luna)?
7. Is the `top_logprobs` cap 20 on the live Chat Completions API for these models (older cookbook says 5)?
8. Does returned logprob mass reflect `logit_bias` (pre/post)? Terms: any OpenAI clause on benchmarking/publishing latency or on use in autonomous vehicle control (simulation only here) [U, not read].

## 10. Follow-up spikes (all need a key; about 1-2 h each; total spend under $2)

- **F1 capability probe (15 min)**: one call each to `gpt-6-luna` (effort none), `gpt-4.1-nano`, `gpt-4o-mini`, `gpt-5-nano` (minimal) with `logprobs: true, top_logprobs: 20, max_completion_tokens: 1`; record accept/reject, number of `top_logprobs` returned, `system_fingerprint`, `reasoning_tokens`, `service_tier`, `max_completion_tokens: 1` acceptance; same via OpenRouter `require_parameters`.
- **F2 latency (1 h)**: reuse the doc 01 benchmark harness with this provider: 200 sequential + concurrency 4/16, for Luna std vs Luna fast vs 4.1-nano vs 4o-mini (mode A) and Luna mode B, OpenAI direct vs OpenRouter pinned `order:["OpenAI"]`, padded vs unpadded prefix (cache effect), at 2-3 times of day. Report p50/p95/p99 and cache hit rate.
- **F3 consistency (30 min)**: same input x 50, per model: argmax flip rate and std of the BRAKE probability; checks whether logprobs are stable enough for thresholds, and whether the alias drifts.
- **F4 accuracy vs Jev (2-3 h, ties to doc 07 S3)**: 300 frozen observations with oracle acceptable-action sets; compare Jev (choice), Luna mode B, 4o-mini mode A; also letters vs names, mirror-symmetry test, calibration curve of alias probabilities (reliability diagram).
- **F5 WebSocket mode (1 h)**: Responses WebSocket with `store: false` for 200 stateless decisions vs HTTP/2 Chat; keep only if p50 improves by more than 50 ms.
- **F6 key-free fallback**: run the same provider against Ollama/vLLM `localhost` (`base_url` swap) with a small open model to confirm the letter-logit pattern end to end offline.

---

## Sources

- OpenAI: [models](https://developers.openai.com/api/docs/models), [pricing](https://developers.openai.com/api/docs/pricing), [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [gpt-4.1-nano](https://developers.openai.com/api/docs/models/gpt-4.1-nano), [gpt-4o-mini](https://developers.openai.com/api/docs/models/gpt-4o-mini), [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model), [latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization), [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [function calling](https://developers.openai.com/api/docs/guides/function-calling), [rate limits](https://developers.openai.com/api/docs/guides/rate-limits), [WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode), [logprobs cookbook](https://developers.openai.com/cookbook/examples/using_logprobs), [Responses reference](https://developers.openai.com/api/reference/resources/responses/methods/create). Priority/Fast mode pages returned 404/403 to my fetcher; those facts come from search summaries and the pricing page's Fast table.
- OpenRouter public API (no key): `GET /api/v1/models` (466 models) and `/api/v1/models/openai/{gpt-6-luna,gpt-4.1-nano,gpt-4o-mini,gpt-oss-20b}/endpoints`; [provider routing](https://openrouter.ai/docs/provider-routing).
- Artificial Analysis: [Luna non-reasoning providers](https://artificialanalysis.ai/models/gpt-6-luna-non-reasoning/providers), [gpt-4.1-nano](https://artificialanalysis.ai/models/gpt-4-1-nano/providers), [gpt-4o-mini](https://artificialanalysis.ai/models/gpt-4o-mini/providers).
- [runtimewire: GPT-5.6 Luna price cut, Sol fast mode](https://runtimewire.com/article/openai-cuts-gpt-5-6-luna-prices-adds-sol-fast-mode); OpenAI community thread on Responses logprobs; docs/research/01 and 07 (this repo).
