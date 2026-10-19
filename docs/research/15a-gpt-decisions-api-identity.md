# 15a — "The GPT decisions API": what exists (as of 2026-10-04)

Access date for every source below: **2026-10-04** unless stated. No API key was used; nothing was signed up for. Legend: **[V]** fetched from a primary source by me; **[V-sec]** third-party report I could fetch; **[I]** inference; **[U]** unverified. Note: WebFetch summarises pages with a small model, so quoted details from fetched pages are summaries, not byte-exact; the OpenRouter model-list data (curl) is exact.

---

## 0. TL;DR

1. **A real product with this name now exists: OpenAI "Decisions API", powered by GPT-6 Luna.** Announced at OpenAI DevDay on **2026-09-29** (5 days ago), "limited preview", "broad release planned in the coming days". Confidence it is what the user means: **~0.85**.
2. **It is not usable today and has no public schema.** No docs page, no API reference, no changelog entry, no pricing, no SDK method [V]. Third parties report a standard key gets HTTP 403 "Decision API is not enabled for this user" (2026-10-02) [V-sec, single source].
3. **It is not on OpenRouter.** OpenRouter's Decisions endpoint (`POST /api/alpha/decisions`) lists 13 `->decisions` models today; **none are OpenAI/GPT** [V, my own curl of the public model API]. OpenAI's Luna is only listed as an ordinary `text+image+file->text` chat model there.
4. **TypeSafe has no GPT-backed variant**; Jev is a proprietary single model, no custom base models [V: docs.typesafe.ai/models.md].
5. **Verdict: not a usable alternative to Jev for this experiment right now.** It is a candidate arm to add the day docs/access arrive. Jev stays primary; widen the "decision model" arm set to the other OpenRouter `->decisions` models, which are real and callable today (section 3).

---

## 1. Interpretations and verdicts

| # | Interpretation | Verdict | Confidence |
|---|---|---|---|
| 1 | A real product named "Decisions API" from OpenAI, on GPT-6 Luna | **EXISTS, but limited preview; undocumented; not accessible without OpenAI whitelisting.** Almost certainly what the user heard about (news cycle is 5 days old and explicitly "vs Jev") | 0.85 that this is the user's meaning; 0.95 that the product exists |
| 2 | OpenRouter's Decisions endpoint serving a GPT-family model | **FALSE today.** 13 decision models, zero OpenAI. The only "GPT" on OpenRouter are chat models (`openai/gpt-6-luna`, `-pro`, `~openai/gpt-luna-latest`, `gpt-5.6-luna`...). Third-party article says explicitly it is "not available on OpenRouter" [V-sec] | 0.95 |
| 3 | OpenAI Chat/Responses API as a decision classifier (structured outputs enum / logprobs / `max_tokens`) | **WORKS TODAY, is the actual fallback.** GPT-6 Luna supports Responses and Chat Completions, `structured_outputs`, `reasoning.effort: none`, streaming; no Fine-tuning, no Realtime [V: https://developers.openai.com/api/docs/models/gpt-6-luna.md]. Measured by a third party: ~1.46 s median with no reasoning and strict JSON schema [V-sec: eesel]. That is slow vs Jev's ~0.2-0.3 s; this is the gap OpenAI's new product claims to close. I did not verify logprobs support for Luna (page lists none in "Supported features") [U] | 0.9 |
| 4a | "GPT-4/5 Decisions" in some other sense (fine-tuned small models, predicted outputs, realtime tiers) | Nothing found under this name. Luna page: fine-tuning "Not supported", Realtime "Not supported" [V] | 0.8 (absence of evidence) |
| 4b | Independent clones using the name | **decisions-api.net**: a self-declared independent service ("not affiliated with ... OpenAI or TypeSafe AI"), `POST /api/v1/decisions`, model `decisions-1` [V-sec]. A "Decisions API" explainer ecosystem (eesel, Hugging Face community blogs, firecrawl, modelsystem.one, orcarouter, Digg, Vercel i/ pages, explainx) is mostly SEO re-reporting of one OpenAI announcement. Treat as noise; none gives an official schema | 0.9 |
| 4c | The user conflating "GPT" with Jev's lineage | Possible: Jev's founder is ex-OpenAI (see doc 01); OpenRouter's own Jev-vs-LLM blog used a "GPT Luna" model as the slow LLM comparator | 0.3 |

---

## 2. Primary-source evidence for the OpenAI product

| Claim | Source | Status |
|---|---|---|
| Announced 2026-09-29 at DevDay: "enables real-time decision-making by focusing Luna's intelligence on a specific set of user-defined questions with finite pre-defined answers"; text or image context in, answers out; "limited preview" | OpenAI Developers X post https://x.com/OpenAIDevs/status/2105003318917697873 (my fetch got **HTTP 402**; wording taken from the OpenAI community thread, which reproduces it), and https://community.openai.com/t/devday-2026-announcements-and-developer-resources/1402006 | [V] via community thread (OpenAI-run forum) |
| Recap page exists | https://openai.com/index/devday-2026-recap/ | My fetch **403**; wording seen only via search snippets/community thread [V-sec] |
| Decisions API **absent from OpenAI API changelog** | https://developers.openai.com/api/docs/changelog (Sep-Oct 2026 entries: only Sep 22 GPT-6 Sol/Luna release and Sep 25 image fix) | [V] |
| **No docs guide** | `https://developers.openai.com/api/docs/guides/decisions` and `platform.openai.com/docs/guides/decisions` return **404** (curl, 2026-10-04) | [V] |
| **No API reference entry** | `https://developers.openai.com/api/docs/api-reference/decisions` returns HTTP 200 but serves the generic "API Overview" page with zero occurrences of "decisions" (soft fallback) | [V] |
| **Absent from OpenAI's docs index** | `https://developers.openai.com/llms.txt`: 0 matches for "decision" | [V] |
| **Absent from Luna model page endpoint table** (Chat Completions, Responses, Batch supported; no Decisions row) | https://developers.openai.com/api/docs/models/gpt-6-luna.md | [V] |
| Latency claim "~150 ms vs ~1.6 s for regular Luna" (10x) from a DevDay slide | modelsystem.one, firecrawl, orcarouter | [V-sec]; vendor claim, unreproduced |
| Price, rate limits, answer-count limit, fine-tuning, region, schema | "not published" per three independent write-ups | [V-sec] |
| Route `POST https://api.openai.com/v1/decisions`; key gets 403 "Decision API is not enabled for this user" (2026-10-02) | eesel.ai https://www.eesel.ai/blog/openai-decisions-api (single source; other articles hedge that no endpoint is published) | [V-sec], **treat route as unconfirmed** |
| Request shapes shown in articles (e.g. `DecisionRequest` with question/answers/context) | explainx.ai itself says "illustrative ... not a guaranteed field list" | **Not a schema. Do not code against it** |

**Request/response schema:** none exists in any primary source. I will not invent one. If asked "what does it look like", the honest answer is "unpublished".

Underlying model facts [V: Luna model page and OpenRouter list]: `gpt-6-luna` released 2026-09-22; text+image in, text out; 1,050,000 context; $0.10 / $0.50 per M tokens in/out (cached in $0.01; >272K prompts 2x); OpenRouter lists the same prices. Rate limits (chat path): Tier 1 500 RPM / 500K TPM up to Tier 5 30,000 RPM.

---

## 3. What OpenRouter's Decisions endpoint actually serves (live, exact)

Source: `GET https://openrouter.ai/api/v1/models?output_modalities=decisions` (public, no key), fetched 2026-10-04T04:47Z, 13 models. Prices are per token as returned; output is $0 for all.

| Model id | Modality | Input $/M tok | Note |
|---|---|---|---|
| `typesafe/jev-1.13` | text->decisions | 0.042 | primary (doc 01) |
| `~typesafe/jev-latest` | text->decisions | 0.042 | alias, do not pin |
| `jaredpalmer/kev-4b` | text->decisions | 0.042 | 4B; previously "unexamined" |
| `togethercomputer/tev1-4b-experimental` | text->decisions | 0.042 | Together AI, experimental |
| `liquid/d1` | text->decisions | 0.04 (cache read 0.04) | Liquid; 65,536 ctx; single provider "Liquid" |
| `perplexity/pplx-decider-v1-27b` | **text+image**->decisions | 0.04 | Perplexity; 262,144 ctx; images |
| `cloudflare/clef-flash` | text+image->decisions | 0.09 | |
| `cloudflare/clef` | text+image->decisions | 0.24 | |
| `inception/mercury-decide:free` | text->decisions | **0 (free)** | Inception; 32,768 ctx; diffusion-LLM vendor |
| `upstage/solar-decide` | text->decisions | 0.05 | |
| `respan/span-01` | text->decisions | 0.02 | |
| `respan/span-01-lite`, `respan/span-01-lite:free` | text->decisions | 0 | |

These were not in doc 01 (it knew only Jev, kev-4b and tev1). Consequence: the "System One / decision-model" category already has at least 9 vendors on one schema; Jev is not unique. None is GPT-family. Creation timestamps run from Sep 18 (Jev) to about Oct 1 (pplx, liquid, clef) [V, `created` field].

**Decisions request/response schema (OpenRouter), from docs** (https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md, plus https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README; WebFetch summary, not byte-for-byte; consistent with doc 01 section 2.1):

- `POST /api/alpha/decisions` (docs print base `https://openrouter.ai/api/v1`; doc 01 flagged the path as likely a docs typo). Auth: Bearer key.
- Request: required `model` (string, e.g. `typesafe/jev-1.13`), `state` (string | object | array, "the content to evaluate"), `questions` (map). Optional `provider` (routing prefs), `session_id` (<=256 chars), `user` (<=256 chars), `trace` (trace_id, trace_name, span_name, ...).
- Question types: `noul` (`instructions`; `criteria` {"true","false"}), `choice` (`criteria` map label -> description), `score` (`criteria` array of >=1 level descriptions).
- Response: `id`, `model`, `provider`, `answers` (map): noul -> `noul` 0-1; choice -> `choice`, `confidence`, `probabilities`; score -> `score`, `confidence`, `probabilities`, `legend`; plus `usage` (tokens, cost).
- Errors: 200, 400, 401, 402, 403, 404, 413, 429, 500, 502, 503, 524, 529.
- Docs state "the set of decision models changes, so never pick from memory" and point to `GET /api/v1/models?output_modalities=decisions` (https://openrouter.ai/skills/openrouter-decisions).
- **The endpoint accepts any listed decision model**, so non-Jev models (kev, liquid, perplexity, ...) are swappable by changing `model` only. Image input only on the `text+image` ones. Per-model quirks (limits, accuracy) are untested [U].
- OpenAI's endpoint is a different, unrelated route with no published schema; the identical-sounding names ("Decisions API", `/decisions`) are the collision to guard against in any code or docs. Note `decisions-api.net` and OpenRouter both use `/api/.../decisions` with `state`+`questions` bodies, which is the OpenRouter/Jev shape, not OpenAI's.

---

## 4. TypeSafe check

https://docs.typesafe.ai/models.md [V]: single model `jev-1.13.0`, aliases `jev-latest` and `jev-preview`; no OpenAI/GPT variant mentioned; "Custom base models are not supported"; same weights for every account; no customer fine-tuning. So "System One accepts GPT-backed models" is **false**.

---

## 5. Is it a better alternative to Jev? (assessment, mostly [I])

| Dimension | Jev (OpenRouter) | OpenAI Decisions API (as announced) | Luna via Responses API + enum (works today) |
|---|---|---|---|
| Access today | Yes, any OpenRouter key | **No** (limited preview, 403 for most) | Yes, OpenAI key |
| Documented schema | Yes | **No** | Yes |
| Latency | measured p50 ~194-320 ms (doc 01) | **vendor slide ~150 ms, unmeasured** | ~1.5 s median, third party |
| Probabilities per option | Yes | "confidence" reportedly; unconfirmed | Only if logprobs available [U] |
| Cost | $0.042/M in, $0 out | unpublished (Luna base $0.10/M) | $0.10/$0.50 per M |
| Image input | No | Yes | Yes |
| Reproducibility/pinning | pinned snapshot id | unknown | `gpt-6-luna` snapshot only |

Honest read: if OpenAI's 150 ms holds, it would be latency-competitive with Jev but would be a **different network/geography/rate-limit regime**, and it brings no advantage for a text-only drone state beyond maybe stronger reasoning on ambiguous cases. Everything numeric about it is a vendor slide. It cannot be called better until measured.

---

## Decisions recommended

1. **Keep Jev (`typesafe/jev-1.13` via OpenRouter) as the primary decision-model arm.** Do not wait for OpenAI.
2. **Reframe the arm taxonomy:** "System One / decision-model" arm family (Jev, plus 1-2 of `liquid/d1`, `perplexity/pplx-decider-v1-27b`, `inception/mercury-decide:free`, `jaredpalmer/kev-4b`) vs "chat LLM as classifier" family. The `DecisionProvider` from doc 01 needs no change; `JevDecisionProvider` just takes a `model` id (rename to `OpenRouterDecisionsProvider`).
3. **Add the OpenAI Decisions API as an optional, deferred arm** behind a flag, implemented only when OpenAI publishes docs and the user's key is enabled. Do not code a speculative schema now (YAGNI).
4. **For a GPT arm today, use Luna via Responses API** (`gpt-6-luna`, `reasoning.effort: none`, strict json_schema enum) as the "GPT as classifier" baseline; expect ~1.5 s and label it as such. This is also exactly the "slow-LLM" arm the latency story needs.
5. In all write-ups use unambiguous names: "OpenRouter Decisions API (Jev et al.)" vs "OpenAI Decisions API (GPT-6 Luna, preview)".
6. Ask the user which one they meant (see open questions) before spending effort.

## Open questions

1. Did the user mean OpenAI's DevDay "Decisions API", or something else (e.g. a GPT model through OpenRouter's Decisions endpoint, which does not exist)?
2. Does the user have, or can they request, OpenAI preview access? (Needs a key; I did not and will not try.)
3. Is the "broad release in the coming days" real, and what are the final schema, price, and rate limits? Unpublished as of today; the 403 report is single-source.
4. Does OpenAI's Decisions API return per-option probabilities (needed for confidence gating like Jev's), or just a label plus a "confidence"?
5. Is Luna's `logprobs` supported on Responses/Chat Completions? Luna page's "Supported features" does not list it [U].
6. Do the other OpenRouter decision models match Jev's quality/latency on the drone prompts? No data.
7. Are TypeSafe/OpenAI/each vendor's ToS compatible with publishing latency benchmarks? (carried from doc 01)
8. Is `output_modalities=all` the 650-model list and is `decisions` filter stable? The `?output_modalities=decisions` filter works today [V].

## Follow-up spikes

- **S1 (15 min, needs key):** `GET /api/v1/models/{id}/endpoints` with a key for each of the 13 decision models to read `latency_last_30m`/`throughput_last_30m` (null unauthenticated), then run the doc-01 benchmark script against 3-4 of them with the same drone payload. Output: p50/p95/p99 and agreement matrix on a fixed set of 200 labelled drone states.
- **S2 (user-gated):** if the user has OpenAI preview access, probe `POST https://api.openai.com/v1/decisions` with the eesel-reported route; record the real schema and latency verbatim, 200 sequential calls, from the experiment machine.
- **S3 (30 min, needs OpenAI key):** Luna Responses API enum classification with `reasoning.effort: none`: measure TTFT/total, check whether `logprobs`/`top_logprobs` are accepted, and compare with `max_output_tokens` small + `text.format` enum vs plain letter output.
- **S4 (watch):** re-fetch https://developers.openai.com/api/docs/changelog, `.../guides/decisions` and `llms.txt` weekly for the broad release; re-run the `output_modalities=decisions` listing to see if any `openai/*` decision model appears on OpenRouter.
- **S5 (small):** consistency probe (same input x 50) across Jev, `liquid/d1`, `pplx-decider-v1-27b` to compare determinism, extending doc 01's F3.

---

## Source list (all accessed 2026-10-04 unless noted)

- OpenRouter model API (exact): https://openrouter.ai/api/v1/models?output_modalities=decisions and `=all` (fetched by curl 2026-10-04T04:47Z); per-model `/endpoints` for liquid/d1, inception/mercury-decide:free, perplexity/pplx-decider-v1-27b
- OpenRouter docs: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md ; https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README ; https://openrouter.ai/skills/openrouter-decisions ; https://openrouter.ai/docs/llms.txt ; https://openrouter.ai/docs/guides/community/jev.md
- TypeSafe: https://docs.typesafe.ai/models.md
- OpenAI: https://developers.openai.com/api/docs/models/gpt-6-luna.md ; https://developers.openai.com/api/docs/changelog ; https://developers.openai.com/llms.txt ; guides/decisions (404) ; api-reference/decisions (generic overview page) ; https://openai.com/index/devday-2026-recap/ (403 to my fetcher) ; https://community.openai.com/t/devday-2026-announcements-and-developer-resources/1402006 ; https://x.com/OpenAIDevs/status/2105003318917697873 (402 to my fetcher)
- Third-party (secondary): https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev ; https://modelsystem.one/news/openai-decisions-api-preview/ ; https://www.orcarouter.ai/blog/openai-decisions-api-gpt-6-luna ; https://www.eesel.ai/blog/openai-decisions-api ; https://explainx.ai/blog/openai-decisions-api-gpt-6-luna-devday-2026 ; https://decisions-api.net/ (independent, unaffiliated)
