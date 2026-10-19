# 16b — Fast single-choice decision models: Gemini, OpenAI, OpenRouter (as of 2026-10-17)

Legend: **[L]** = live call to a read-only listing endpoint with the user's keys today; **[V]** = primary doc fetched today; **[V-sec]** = third-party page; **[I]** = inference; **[U]** = unverified. WebFetch summaries come from a small model, so doc quotes are summaries, not byte-exact. No keys are recorded here. No inference calls were made (another agent does the live probe). OpenRouter's per-endpoint latency is a rolling 30-minute window taken at one moment, on OpenRouter's own traffic mix (not our 685-token prompt), so use it for ranking, not as a promise.

---

## 0. TL;DR

1. **Listing calls all succeeded.** Gemini `GET /v1beta/models` returned 200, OpenAI `GET /v1/models` returned 200, OpenRouter `GET /api/v1/models` returned 200 (470 models).
2. **The Gemini key sees the newest tiers**: `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, and `gemini-3.5/3.6/3.7/3.8-flash`, plus `gemini-4-argon`. A listing does not prove billing is active or the tier's rate limit [U].
3. **The OpenAI key sees `gpt-6-luna`, `gpt-6-sol`, `gpt-6-astra`, `gpt-6.1-sol`, `gpt-5.6-*`, `gpt-4.1-nano`, `gpt-4o-mini`, `gpt-5.4-nano`, `gpt-5-nano`.** There is **no decisions model id** in `/v1/models`. That is expected: Decisions is an endpoint (`POST /v1/decisions`), not a listed model, so the listing cannot tell us whether the key is enabled. Only a request can (403 vs 200), and that is the other agent's probe.
4. **The OpenAI Decisions API is no longer limited preview.** OpenAI's guide now says public beta, `POST /v1/decisions`, model `gpt-6-luna` only, `choice` questions return a `probabilities` array plus `confidence`, $0.10 per 1M input tokens, **no charge for output tokens or cache reads/writes** [V]. This overturns doc 15a's "not usable today". It is also on OpenRouter as hidden `openai/gpt-6-luna-decisions` (OpenAI endpoint, p50 193 ms / p90 408 ms / p99 1,125 ms right now) [L]. It is the single most relevant candidate: it returns per-action probabilities natively.
5. **For Gemini, probabilities are the weak spot.** No Gemini 3.x/3.5 doc page mentions logprobs [V]; the new Interactions API explicitly drops them [V-sec forum]; user reports say logprobs 400 on some 2.5/3.x models [V-sec]. Whether `gemini-2.5-flash-lite` or `gemini-3.5-flash-lite` returns `logprobsResult` via `generateContent` is **unresolved without a live call** (spike F1).
6. **Fastest measured non-decision LLM routes for constrained choice are open-weight on specialist hardware via OpenRouter**: `openai/gpt-oss-20b` on CoreWeave (p50 125 ms, logprobs yes), `meta-llama/llama-3.1-8b-instruct` on Groq (p50 146 ms, no logprobs) or CoreWeave (240 ms, logprobs yes), `gpt-oss-120b` on Groq 168 ms / Cerebras 222 ms (Cerebras logprobs yes) [L].

---

## 1. What each key can list [L]

All HTTP 200. Chat/decision-relevant ids only (the rest are TTS, image, audio, embedding, deep research).

### 1.1 Gemini key (`generateContent`-capable)

| Family | Ids |
|---|---|
| Flash-Lite | `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-3.5-flash-lite`, `gemini-flash-lite-latest` (alias) |
| Flash | `gemini-2.5-flash`, `gemini-3-flash-preview`, `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`, `gemini-flash-latest` (alias) |
| Pro / other | `gemini-2.5-pro`, `gemini-3.1-pro-preview`, `gemini-pro-latest`, `gemini-4-argon`, `gemini-robotics-er-2-preview`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `gemini-omni-flash-preview`, `gemini-omni-1.1-flash` |

All Flash/Flash-Lite entries report 1,048,576 input and 65,536 output limits and `thinking: true` in the model metadata. `gemma-4-*` is served here too (open model, may be a lower-latency route [U]).

### 1.2 OpenAI key

Decision-relevant: `gpt-6-luna`, `gpt-6-sol`, `gpt-6-astra`, `gpt-6.1-sol`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini`, `chat-latest`. No `decisions`-named id.

### 1.3 OpenRouter key

470 models. `openai/gpt-6-luna-decisions` and `typesafe/jev-1.14` are **not** in the key's default listing, but they are returned by the public `?output_modalities=decisions` query and have working `/endpoints` pages [L]. Treat them as hidden-but-routable (same pattern as Jev in doc 01). Listing 27 `->decisions` models today (table in section 4).

---

## 2. Gemini candidates

### 2.1 Facts

| Item | Finding |
|---|---|
| Prices (paid, per 1M) | `2.5-flash-lite` $0.10 in / $0.40 out; `3.1-flash-lite` $0.25 / $1.50; `3.5-flash-lite` $0.30 / $2.50 (cache read $0.03); `3.6/3.8-flash` $0.75 / $3.75 until 2026-12-31, then $1.50 / $7.50 [V: ai.google.dev/pricing] |
| Thinking control | New models use `thinking_level` (3.5-flash-lite: `minimal`/`low`/`medium`/`high`, **default "On (minimal)"**; 3.6-flash: default medium, levels minimal..high; 3.8-flash: default medium, levels low..high) [V: /thinking]. **No 3.x/3.5 model documents a fully-off setting.** Only `gemini-2.5-flash-lite` is listed "Off" by default [V]. A third-party guide says `thinking_budget` and `thinking_level` together return 400, and that 3.1-flash-lite defaults to High [V-sec; conflicts with the Google doc for 3.5, so set it explicitly]. In `generateContent` the field is `generationConfig.thinkingConfig.thinkingLevel` [I from SDK naming; the Google page shows the Interactions-API snake_case form `generation_config.thinking_level`] |
| Structured output / enum | Supported on 3.5-flash-lite [V]. `responseMimeType: "text/x.enum"` + `responseSchema: {type: STRING, enum: [...]}` is the lowest-overhead single-choice form in `generateContent` [I from older docs; re-verify in spike] |
| Logprobs | Not mentioned on the 3.5-flash-lite model page [V]. `responseLogprobs`/`logprobs` (1-20) exist on `generateContent` [V-sec: Google blog/forum, Vertex-centric]. Interactions API has no logprobs by design (Google forum reply, May 2026) [V-sec]. Forum reports of "logprobs not enabled for model" 400s on 2.5-flash and 3.x Pro [V-sec]. **Status for the user's key: unknown until probed** |
| Context caching | Implicit caching minimum is 4,096 tokens for 3.6/3.8 Flash, 2,048 for 2.5 Flash; **no minimum is listed for Flash-Lite** [V]. 3.5-flash-lite shows cache read $0.03/M (10% of input) plus $1.00/h storage on the pricing page, but the free tier says "caching not available" [V]. Our 685-token prompt is below every stated minimum: **assume no caching** [I]. Padding to 4k+ tokens would add prefill cost for little latency benefit [I] |
| Latency | Artificial Analysis lists `gemini-3.5-flash-lite` TTFT **6.85 s (6.1-7.9 s across snapshots)** at 10k input, but "time to first answer token" **includes thinking time** and only the default reasoning mode is benchmarked; 449 tok/s output [V-sec]. There is no published sub-second figure for any Gemini Flash-Lite with minimal thinking. Anecdote: "sub-2-second completions" for 3.1 Flash-Lite [V-sec, weak] |
| Rate limits | Docs list no per-model RPM/TPM; they point to the AI Studio page for the key [V]. Tier 1 = linked active billing account, auto-upgrade [V]. Batch enqueued tokens at Tier 1: 10M for Flash-Lite, 3M for 3.6/3.8 Flash [V]. Read the real RPM in AI Studio (not available to me) [U] |

### 2.2 Gemini request for lowest latency (single choice, 11 actions)

Model order to try: `gemini-2.5-flash-lite` (thinking off by default, cheapest) then `gemini-3.5-flash-lite` with `minimal`, then `gemini-3.1-flash-lite` with `minimal`.

```
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent
Header: x-goog-api-key: <from settings>        (never ?key=)
{
  "systemInstruction": {"parts":[{"text": "<static ~550-token prompt>"}]},
  "contents": [{"role":"user","parts":[{"text":"<~135-token observation>"}]}],
  "generationConfig": {
    "temperature": 0,
    "maxOutputTokens": 4,
    "responseMimeType": "text/x.enum",
    "responseSchema": {"type":"STRING","enum":["A","B","C","D","E","F","G","H","I","J","K"]},
    "responseLogprobs": true,
    "logprobs": 11,
    "thinkingConfig": {"thinkingBudget": 0}
  }
}
```

For `gemini-3.5-flash-lite` / `3.1-flash-lite`: replace `thinkingBudget` with `"thinkingLevel": "minimal"` and **never send both**. Use letter aliases (A..K) as in doc 15b so each action is one token. If the enum mime type rejects `logprobs`, fall back to a plain text answer with `maxOutputTokens: 2` and the same logprobs flags. Reuse one HTTP/2 client; time with `perf_counter_ns` around the full request.

### 2.3 Gemini verdict

Cheapest and probably fastest Gemini: `gemini-2.5-flash-lite` (thinking already off). The newest `3.5-flash-lite` is better quality but its documented minimum is "minimal", not off, and the only published TTFT (6-8 s) is badly misleading for us because it is the default mode on a 10k prompt. Rank on measured p50/p95 from our own probe, not on AA.

---

## 3. OpenAI candidates

Prior detail is in doc 15b (tiers, caching, fast mode). New or changed today:

| Item | Finding |
|---|---|
| **Decisions API** | **Public beta**, GA "in the coming weeks"; `POST /v1/decisions`; `model: "gpt-6-luna"` (only); body `input` (string or message parts) + `questions[]`; `choice` question => `{choice, probabilities[], confidence}`; `predicate` => `probability`; `score` => weighted score; refusal possible; $0.10/M input, **no output/cache charges**; ZDR/HIPAA eligible; US and EEA data residency [V: developers.openai.com/api/docs/guides/decisions]. Vendor claim at DevDay: ~150 ms vs ~1.6 s for plain Luna [V-sec]. A 3rd-party source dated 2026-10-02 saw 403 "not enabled" [V-sec]; this key's access is untested. Logprobs: not applicable, probabilities are a first-class field |
| OpenRouter route | `openai/gpt-6-luna-decisions` (text+image->decisions, $0.10/M, one endpoint "OpenAI"): **p50 193 ms, p90 408 ms, p99 1,125 ms, 100% uptime** in the last 30 min [L]. For comparison the same snapshot shows `typesafe/jev-1.14` p50 136 / p90 284 / p99 965 and `typesafe/jev-1.13` p50 129 / p90 253 / p99 355 [L] |
| `gpt-6-luna` as chat | Reasoning `none` is supported, default is medium; one snapshot id `gpt-6-luna`; the model page does not mention logprobs [V]. OpenRouter endpoints show no logprobs for any Luna route and slow current latency: OpenAI direct p50 1,973 ms / p90 6,324 ms; `openai/fast` p50 897 ms; Azure EU p50 841 ms; Bedrock us-east-1 p50 1,512 ms [L]. Those include whatever effort callers use, so treat as an upper bound (doc 15b planning numbers: 0.35-0.8 s p50 with effort none) |
| Rate limits | Luna tiers are now named Build 5,000 RPM / 2M TPM, Launch 10,000 / 10M, Grow 30,000 / 180M [V]. The Tier 1-5 names in doc 15b are older; map by value. The key's actual tier is [U] |
| `gpt-4.1-nano`, `gpt-4o-mini` | Both listed on the key. `openai/gpt-4o-mini` direct on OpenRouter: p50 492 ms / p90 695 ms / p99 1,752 ms, logprobs + top_logprobs + structured outputs all true; Azure routes 767-1,474 ms [L]. 4.1-nano does not list logprobs on OpenRouter (doc 15b) |
| Caching | Prompt-cache minimum 1,024 tokens; our 685-token prompt is below it (doc 15b) |

### 3.1 OpenAI request for lowest latency

A. Decisions API (confirm field names against the guide in the spike; I read a summary, not the raw schema):

```
POST https://api.openai.com/v1/decisions
{
  "model": "gpt-6-luna",
  "input": "<~135-token observation, preceded by static context if the API has no instructions field>",
  "questions": [{
    "name": "action", "type": "choice",
    "instructions": "<static ~550-token prompt>",
    "choices": [{"value":"A","description":"..."}, ... 11 entries]
  }]
}
```

B. Chat Completions (logprobs candidate, `gpt-4o-mini`):

```
POST https://api.openai.com/v1/chat/completions
{"model":"gpt-4o-mini","messages":[{"role":"system","content":"<static>"},{"role":"user","content":"<obs>"}],
 "max_completion_tokens":1,"temperature":0,"logprobs":true,"top_logprobs":11,
 "logit_bias":{"<token ids of A..K>":100}}   // optional; verify token ids first
```

C. `gpt-6-luna` chat: `{"model":"gpt-6-luna","reasoning_effort":"none","max_completion_tokens":1,...}` + `service_tier:"fast"` as the A/B arm. Expect no logprobs; log whether the API rejects the parameter.

---

## 4. Other decision-specific models on OpenRouter (hidden `->decisions` list) [L]

All are `POST /api/alpha/decisions`-style typed-question models (same shape as Jev), 27 listed. Latency is the last 30 min per endpoint.

| Model id | Provider | p50 ms | p90 ms | p99 ms | $/M in |
|---|---|---|---|---|---|
| `typesafe/jev-1.14` | TypeSafe | 136 | 284 | 965 | 0.042 |
| `typesafe/jev-1.13` | TypeSafe | 129 | 253 | 355 | 0.042 |
| `liquid/d1` | Liquid | 163 | 312 | 793 | 0.02 |
| `cloudflare/clef-flash` | PrimeIntellect | 179 | 312 | 607 | 0.021 |
| `perplexity/pplx-decider-flash-v1.2` | Perplexity | 182 | 1,505 | 1,966 | 0.01 |
| `openai/gpt-6-luna-decisions` | OpenAI | 193 | 408 | 1,125 | 0.10 |
| `microsoft/microsoft-decision-1` | Azure | 259 | 518 | 1,004 | 0.042 |
| `inception/mercury-decide` | Inception | 277 | 452 | 1,502 | 0.02 |
| `nace-ai/drex-v2.0` | DeepInfra | 332 | 642 | 1,514 | 0.06 |
| `respan/span-02` | Respan | 367 | 2,148 | 2,467 | 0.01 |
| `upstage/solar-decide-flash` | Upstage | 496 | 607 | 980 | 0.05 |

Others listed but not stat-checked: `liquid/d1-ultra`, `scaledown/classify-1`, `cloudflare/clef`/`clef-omni`, `togethercomputer/tev1-4b-experimental`, `jaredpalmer/kev-4b`, `perplexity/pplx-decider-v1.2`, `respan/span-01(-lite)`, `upstage/solar-decide`, `nace-ai/drex-v1.5`, `inception/mercury-decide:free`. Whether each exposes a full probability vector over 11 choices is not verified here [U]; Jev does (doc 01).

## 5. Fast open-weight chat routes with logprobs on OpenRouter [L]

`lp` = endpoint lists `logprobs`+`top_logprobs`; `so` = structured outputs. Latency = OpenRouter "latency" (time to first token, ms), last 30 min.

| Model | Provider | p50 | p90 | p99 | tok/s p50 | lp | so | $/M in |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-20b` | CoreWeave fp4 | 125 | 259 | 968 | 135 | yes | yes | 0.03 |
| `openai/gpt-oss-20b` | Groq | 256 | 643 | 1,659 | 456 | **no** | yes | 0.075 |
| `openai/gpt-oss-20b` | Bedrock | 345 | 703 | 2,714 | 404 | no | no | 0.07 |
| `openai/gpt-oss-120b` | Groq | 168 | 437 | 1,150 | 312 | no | no | 0.15 |
| `openai/gpt-oss-120b` | Together | 217 | 361 | 1,196 | 168 | no | yes | 0.15 |
| `openai/gpt-oss-120b` | Cerebras fp16 | 222 | 508 | 1,285 | 792 | **yes** | yes | 0.35 |
| `openai/gpt-oss-120b` | Nebius fp4 | 271 | 505 | 1,136 | 299 | no | yes | 0.15 |
| `openai/gpt-oss-120b` | CoreWeave fp4 | 268 | 512 | 1,992 | 77 | yes | yes | 0.03 |
| `openai/gpt-oss-120b` | BaseTen fp4 | 238 | 401 | 809 | 193 | no | yes | 0.10 |
| `meta-llama/llama-3.1-8b-instruct` | Groq | 146 | 317 | 624 | 309 | no | no | 0.05 |
| `meta-llama/llama-3.1-8b-instruct` | CoreWeave | 240 | 424 | 704 | 138 | **yes** | yes | 0.22 |
| `meta-llama/llama-3.3-70b-instruct` | CoreWeave fp16 | 158 | 361 | 635 | 83 | yes | yes | 0.71 |
| `nvidia/nemotron-3.5-lightning` | CoreWeave | 285 | 401 | 637 | 259 | yes | yes | 0.07 |
| `google/gemma-4-26b-a4b-it` | CoreWeave | 356 | 697 | 1,872 | 51 | yes | yes | 0.10 |
| `google/gemma-4-26b-a4b-it` | Makora | 387 | 855 | 1,988 | 158 | yes | no | 0.10 |
| `openai/gpt-4o-mini` | OpenAI | 492 | 695 | 1,752 | 91 | yes | yes | 0.15 |

Observations: Groq does not expose logprobs for any model checked (consistent with Groq's API historically); SambaNova's gpt-oss and Llama-70B-turbo are slow with huge tails right now (gpt-oss-120b p99 45 s) [L]. CoreWeave is the only host that is both fast and logprob-capable across several models. Reasoning models (gpt-oss) must be told to use low or no reasoning, else hidden reasoning tokens dominate TTFT; OpenRouter `reasoning: {effort: "low"}` or `{enabled:false}` where supported [I; verify per provider].

---

## 6. Ranked shortlist for live probing

Rank is by expected (latency x probability usefulness x risk), not by quality.

| # | Vendor / model id / route | Why | Lowest-latency parameters |
|---|---|---|---|
| 1 | **OpenAI `gpt-6-luna` Decisions API**, direct `POST /v1/decisions` (OpenRouter fallback `openai/gpt-6-luna-decisions`) | Purpose-built, returns per-choice `probabilities` + `confidence`, p50 193 ms on OpenRouter, no output-token cost; same API shape as Jev (direct apples-to-apples). Risk: key may get 403 "not enabled"; schema read from a summary | Section 3.1 A; single `choice` question with 11 choices; HTTP/2 keep-alive; warm with the real schema once |
| 2 | **OpenRouter `typesafe/jev-1.14`** (reference, not a candidate) | The incumbent. Included in every probe as the control; p50 136 ms right now | per doc 01 |
| 3 | **Open-weight `openai/gpt-oss-20b` via OpenRouter, provider `CoreWeave`** | Fastest measured chat route (p50 125 ms, p99 968 ms), logprobs + structured outputs, $0.03/M. Reasoning must be minimized | `provider: {order:["CoreWeave"], allow_fallbacks:false, require_parameters:true}`, `max_tokens` 1-2 (plus any reasoning tokens), `temperature:0`, `logprobs:true`, `top_logprobs:11`, `reasoning:{effort:"low"}` [verify], letters A..K |
| 4 | **Google `gemini-2.5-flash-lite`**, direct `generateContent` | Thinking off by default, $0.10/$0.40, stable GA id on the key; logprob status unknown but the best chance among Gemini models (older generation) | Section 2.2 with `thinkingBudget:0` |
| 5 | **OpenAI `gpt-4o-mini`, direct** | Best-known logprobs + structured outputs path; p50 492 ms / p90 695 ms on OpenRouter; stable, single snapshot | Section 3.1 B; `max_completion_tokens:1`, `top_logprobs:11` |
| 6 | **`openai/gpt-oss-120b` via OpenRouter `Cerebras`** (or `meta-llama/llama-3.1-8b-instruct` on `CoreWeave` for a non-reasoning arm) | Cerebras has logprobs, 792 tok/s, p50 222 ms; Llama-8B/CoreWeave p50 240 ms, non-reasoning, logprobs. Gives a quality ladder (8B vs 120B) at similar latency | provider pinned, `logprobs`+`top_logprobs`, `max_tokens` 1-3, reasoning minimal for gpt-oss |
| 7 | **Google `gemini-3.5-flash-lite`**, direct, `thinkingLevel: minimal` | Newest Lite tier, cheap, structured output supported; thinking cannot be fully off; TTFT unknown (AA 6-8 s is default-mode, 10k prompt). Probe to settle the 3.x TTFT and logprobs questions | Section 2.2 with `thinkingConfig.thinkingLevel:"minimal"` |
| 8 | **OpenAI `gpt-6-luna` chat, `reasoning_effort:"none"`, `service_tier:"fast"`** (A/B vs standard) | The "GPT as plain LLM" arm for the comparison story; expect no logprobs, slower than #1 | Section 3.1 C |

Honorable mentions to add only if cheap: `liquid/d1` (p50 163 ms) and `cloudflare/clef-flash`/PrimeIntellect (179 ms) as additional decision-API arms; `meta-llama/llama-3.1-8b-instruct` on Groq (146 ms but no logprobs, so text-only argmax).

---

## Decisions recommended

1. **Make the OpenAI Decisions API the primary "OpenAI-native decision" arm (rank 1)**, with the OpenRouter `openai/gpt-6-luna-decisions` route as the backup if the key gets 403. This replaces doc 15a's verdict that it is unusable.
2. **Keep Jev as the control in every probe**, and compare against it on identical observations.
3. **Use letter aliases A..K and `max_tokens` 1-2 everywhere**; judge probabilities from `top_logprobs` or the Decisions `probabilities` array, never from generated text.
4. **Do not rank Gemini on Artificial Analysis numbers.** The only published TTFT is thinking-inclusive on a 10k prompt; measure ourselves.
5. **Rank gpt-oss-20b on CoreWeave as the open-weight latency reference**, and pin providers with `allow_fallbacks:false` so latency stats are not mixed across hosts.
6. **Do not rely on context caching for any candidate**: all stated minimums (1,024 OpenAI; 2,048-4,096 Gemini) exceed our ~685-token prompt, and Decisions API has no cache charge anyway.

## Open questions

1. Is this OpenAI key enabled for `POST /v1/decisions`? (listing cannot show it.) What are the exact request/response field names and does `probabilities` carry all 11 choice values with their labels?
2. Does the Decisions API accept a long static instruction/system field or only per-question `instructions`, and does repeating 550 tokens per call cost latency (cache is free but may not apply under 1,024 tokens)?
3. Does `gemini-2.5-flash-lite` (or any 3.x Flash-Lite) return `logprobsResult` on `generateContent` with an enum response schema? Does `text/x.enum` conflict with `responseLogprobs`?
4. Can any Gemini 3.x/3.5 model reach sub-500 ms p50 at `minimal` thinking for ~700 input tokens and 1 output token?
5. What are the real RPM/TPM limits of the Gemini key (AI Studio shows them; I could not read them) and the OpenAI key (Build/Launch/Grow tier)?
6. Does gpt-oss-20b on CoreWeave keep p50 ~125 ms with reasoning minimized, and does OpenRouter honor `reasoning` low/off there?
7. How representative is OpenRouter's 30-min latency window versus our client location (US West origin per doc 01)? Several of these numbers will move.
8. Do the other `->decisions` models (Liquid d1, Cloudflare clef-flash) return a full probability vector for an 11-way choice, and what does OpenRouter's alpha decisions endpoint require for them?

## Follow-up spikes

- **F1 (Gemini logprobs, ~30 min)**: three requests each to `gemini-2.5-flash-lite`, `3.1-flash-lite`, `3.5-flash-lite` with and without `responseLogprobs`, with and without `text/x.enum`; record HTTP status, presence of `logprobsResult.topCandidates`, and `usageMetadata.thoughtsTokenCount`. Confirms thinking is actually 0 tokens.
- **F2 (Decisions API access, ~15 min)**: one real `choice` request to `/v1/decisions` on the OpenAI key (403 vs 200), then to `openai/gpt-6-luna-decisions` on OpenRouter. Save the raw schema to docs.
- **F3 (latency ladder, ~1 h)**: 200 sequential requests each (after 5 warmups, same HTTP/2 connection, 685-token prompt) across the 8 shortlisted routes, interleaved in time, recording p50/p90/p99 plus TTFB vs total; compare to the OpenRouter 30-min figures.
- **F4 (open-weight reasoning off)**: gpt-oss-20b/CoreWeave with `reasoning` effort low versus disabled; measure hidden-token count and whether logprob 1st token is still the answer letter.
- **F5 (probability quality)**: on 100 labelled scenario observations, compute agreement with Jev's argmax and calibration (Brier) of Decisions `probabilities` vs `top_logprobs` softmax.
- **F6 (rate-limit read)**: read the AI Studio and OpenAI usage-limits pages in the browser (user session) for the real RPM/TPM caps before planning the 500-1,000 paired-scenario runs.

---

Sources used today: Google AI pages (pricing, thinking, caching, rate-limits, gemini-3.5-flash-lite model page), OpenAI Decisions guide and gpt-6-luna model page (developers.openai.com), Artificial Analysis Gemini 3.5 Flash-Lite providers page, Google AI Developers Forum threads on logprobs, ProPakistani/eesel/Digg write-ups of the Decisions API launch, OpenRouter `GET /api/v1/models`, `/models?output_modalities=decisions`, and `/models/{id}/endpoints` (live), plus the key-authenticated listing calls for Gemini and OpenAI. Related repo docs: `MASTER-PLAN.md`, `15a`, `15b`, `15c`.
