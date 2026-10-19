# 16a - Decision-model survey for the 11-action drone experiment (as of 2026-10-17)

Access date for every source: **2026-10-17** unless stated. Legend: **[V]** = verified by me against a primary source (OpenRouter model API via authenticated read-only GET, or a vendor doc I fetched); **[V-sec]** = reported by a third party I could fetch (blogs, benchmark sites; many are vendor-run or tiny samples); **[I]** = my inference; **[U]** = unverified. WebFetch summarises pages with a small model, so page-derived numbers are summaries, not byte-exact. No paid inference calls were made. The only calls were read-only `GET /api/v1/models?output_modalities=...` and `GET /api/v1/models/{slug}/endpoints`. Raw JSON is in the scratchpad (`.../scratchpad/16a`), no key is stored anywhere.

---

## 0. TL;DR

1. **The catalogue grew from 13 (Oct 4, doc 15a) to 27 `->decisions` entries (Oct 17)** [V]. Counting aliases, `:free` duplicates and dated pairs, that is about **17 distinct model families** from **13 vendors**. Output is `$0` for every one; billing is input tokens only (see section 2).
2. **A GPT-family decision model is now on OpenRouter**: `openai/gpt-6-luna-decisions` (created 2026-10-06) [V]. This **supersedes doc 15a's claim** ("none are OpenAI/GPT"). OpenAI's own endpoint went public beta around 2026-10-06 (`POST /v1/decisions`, fetched docs) [V], but its native schema differs from the OpenRouter/System One schema (section 3).
3. **Every model on OpenRouter's Decisions endpoint takes the same `POST /api/alpha/decisions` body (`model`, `state`, `questions`)** [V: OpenRouter API reference]. OpenRouter adapts OpenAI to it. So swapping `model` is a one-line change. Per-model differences are limits (options per question, context, image support, `box`), not the schema.
4. **Latency on OpenRouter is a tight cluster at p50 ~120-300 ms for about ten models** [V, endpoint stats snapshot]. The fastest p50 are `cloudflare/clef-omni` 117 ms, `typesafe/jev-1.13` 118 ms, `typesafe/jev-1.14` 135 ms. Nothing hosted beats ~100 ms through OpenRouter; claims of 38.8 ms (Clef-flash) or 67 ms (Celeris-1) are vendor numbers measured direct from a nearby region [V-sec].
5. **No hosted model is clearly better than Jev for this use.** Independent small benchmarks put Jev, OpenAI Decisions, Perplexity Decider, Microsoft Decision-1 and Clef in a statistical tie on accuracy, with Jev leading on speed-per-accuracy. The real latency step change is **self-hosting a 3-4B decision model** (8-45 ms reported), which is a different project decision (section 8).
6. **Keep Jev as primary**, add `openai/gpt-6-luna-decisions`, `cloudflare/clef-omni` (or `clef`), `perplexity/pplx-decider-v1.2` and one of `liquid/d1-ultra` / `microsoft/microsoft-decision-1` as live-probe arms (section 9).

---

## 1. Catalogue (OpenRouter `output_modalities=decisions`), 27 entries [V]

Source: `GET https://openrouter.ai/api/v1/models?output_modalities=decisions` (2026-10-17 ~17:45Z). `output_modalities=all` returns 685 models and contains exactly the same 27 decision entries [V]. `supported_parameters` is **empty (`[]`) for all 27** [V]: there are no chat-style parameters (no `temperature`, `max_tokens`, `response_format`). Routing is done with the request `provider` object instead. `throughput_last_30m` is 0 for every model (no generated tokens), so it carries no information [V].

Price is input $/M tokens; output $0 everywhere. "Ctx" is the model-level `context_length`; the per-provider value can be lower (noted). Latency is `latency_last_30m` (ms, p50/p75/p90/p99) from the `/endpoints` call, a rolling window with unknown, probably small, n; treat as a snapshot, not a distribution [V for the numbers, I for the caveat]. Created = OpenRouter `created`.

| Model id (created) | Vendor / provider route | Params | Ctx | $/M in | Inputs | Latency p50/p75/p90/p99 ms | Notes |
|---|---|---|---|---|---|---|---|
| `typesafe/jev-1.13` (09-18) | TypeSafe | undisclosed | 262,144 | 0.042 | text,image | 118 / 213 / 246 / 353 | Stable. Dated snapshot `jev-1.13-20260917` [V]. |
| `typesafe/jev-1.14` (10-14) | TypeSafe | undisclosed | 262,144 | 0.042 | text,image | 135 / 205 / 243 / 590 | Released 10-14. Adds `box` type, image composition, long context, but TypeSafe flags image + >64k context as experimental; to become default `jev-latest` about 10-21 [V-sec: remio, ecosistemastartup]. |
| `~typesafe/jev-latest` | TypeSafe alias | | | 0.042 | | (no own stats) | Alias; will move to 1.14. Never pin. |
| `openai/gpt-6-luna-decisions` (10-06) | OpenAI | undisclosed (GPT-6 Luna) | 1,050,000 | 0.10 | text,image | 190 / 245 / 352 / 945 | `is_moderated: true` [V]. Public beta. |
| `cloudflare/clef-omni` (10-09) | Cloudflare (Workers AI) | 30B MoE, 3B active (Qwen3-Omni-30B-A3B fine-tune) | 65,536 | 0.15 | text,image,audio,video | 117 / 134 / 207 / 656 | Single provider; fastest p50 of the set at snapshot. |
| `cloudflare/clef` (10-01) | PrimeIntellect / DeepInfra(fp8) / Cloudflare | 27B (Qwen3.8-27B) | 65,536 (PrimeIntellect only 16,384) | 0.042 (PI) / 0.24 | text,image | PI 242/296/359/590; DI 295/492/611/1468; CF 517/669/895/1408 | Three routes; price and latency differ a lot by route. |
| `cloudflare/clef-flash` (10-01) | PrimeIntellect / DeepInfra(bf16) / Cloudflare | 9B (Qwen3.5-9B) | 24,576 (PI 16,384) | 0.021 (PI) / 0.038 | text,image | PI 177/225/311/577; DI 229/269/467/1509; CF 441/665/947/2667 | |
| `perplexity/pplx-decider-v1.2` (10-16) | Perplexity | 27B | 262,144 | 0.02 | text,image | 273 / 405 / 695 / 3085 | Successor of v1.1 27B; same API contract [V description]. |
| `perplexity/pplx-decider-flash-v1.2` (10-16) | Perplexity | 4B | 262,144 | 0.01 | text,image | 182 / 717 / 1505 / 1966 | Cheapest; heavy tail. |
| `~perplexity/pplx-decider-latest`, `~...-flash-latest` | aliases | | | | | (no stats) | Never pin. |
| `liquid/d1` (10-01) | Liquid | undisclosed (hosted; open d1-3B / d1-omni-600M exist) | 65,536 | 0.02 | text,image | 231 / 386 / 640 / 1307 | |
| `liquid/d1-ultra` (10-15) | Liquid | "largest d1", undisclosed | 32,768 | 0.042 | text,image | 173 / 327 / 756 / 2354 | 2 days old. |
| `microsoft/microsoft-decision-1` (10-09) | Azure | 9B (Qwen3.5-9B post-trained) [V-sec] | 32,768 | 0.042 | text | 259 / 402 / 518 / 1004 | Reported 429s under load (about 2% in one test) [V-sec]. |
| `inception/mercury-decide` (09-30) | Inception | diffusion LM, undisclosed | 65,536 | 0.02 | text,image | 277 / 334 / 452 / 1502 | Dev-friendly. |
| `inception/mercury-decide:free` | Inception | same | 65,536 | **0** per list; endpoint row still shows 0.02 with discount 0.5 [V] | text,image | 280 / 333 / 444 / 1397 | Free tier. Free rate limits not published in what I could read [U]. |
| `upstage/solar-decide-flash` (10-08) | Upstage (zdr + default) | on Solar Mini 4 | 524,288 | 0.05 | text | 495 / 512 / 587 / 923 (zdr); 562 single sample (default) | Max **26 candidates per choice question** [V-sec: rbeckner]. |
| `upstage/solar-decide` (09-28) | Upstage | Solar Mini 4 | 524,288 | 0.05 | text | **8,624 / 9,739 / 11,995 / 15,328** | Anomalous multi-second latency at snapshot; unusable. |
| `togethercomputer/tev1-4b-experimental` (09-30) | Together | 4B (Qwen3.5-4B SFT) | 32,768 | 0.042 | text | 191 / 198 / 323 / 373 | Single question, **2-24 options**, `max_completion_tokens: 8`; served on chat-completions style runtime [V-sec: Eden AI/OpenRouter listing]. Tightest tail of all, but experimental. |
| `jaredpalmer/kev-4b` (09-25) | Phala / SiliconFlow(fp8) | 4B (LoRA + pointer head on Qwen3.5-4B-Base) | 8,192 | 0.042 | text | Phala 217/282/395/705; SF 413/465/499/702 | Research preview, weakest hosted accuracy in tests. |
| `nace-ai/drex-v1.5` (10-09) | DeepInfra | 9B | 131,072 | 0.04 | text | 189 / 304 / 546 / 1367 | |
| `nace-ai/drex-v2.0` (10-12) | DeepInfra | 35B MoE (3B active, Qwen3.6-35B-A3B) | 131,072 | 0.06 | text | 332 / 461 / 642 / 1514 | Nace Open RAIL-M license; self-claimed DI 61.25 [V-sec: HF]. |
| `scaledown/classify-1` (10-13) | ScaleDown | undisclosed | 1,000,000 | 0.04 | text,image,file | 303 / 424 / 523 / 2692 | Document-oriented classifier; description says "calibrated probabilities". |
| `respan/span-02` (10-16) | Respan | open-weight, undisclosed | 131,072 | 0.01 | text,image | 264 / 520 / 2130 / 2350 | |
| `respan/span-01` (09-26) | Respan | undisclosed | **0 (listed)** | 0.02 | text | 176 / 196 / 317 / 704 | **"Behavior scoring" over conversation spans** [V description]; wrong task shape. Context listed 0. |
| `respan/span-01-lite`, `:free` | Respan | | 0 | 0 | text | 274 / 619 / 693 / 1370 | Same, free lite tier. |

Not on OpenRouter but real [V-sec]: Celeris-1 Decision (hosted, $0.04/M, vendor-measured 67 ms for 1 question, 81 ms for 50), Fastino GLiDE ($0.15/M), H2O-Lightning-4B, open d1-3B / d1-omni-600M, Decision 3.0 family, Jebadiah/Mapika 4B (see section 8).

Dedupe for counting: Jev x2 (+alias), Clef x3, Decider x2 (+2 aliases), d1 x2, Solar x2, Drex x2, Span x3 (+free), Mercury x1 (+free), Luna-Decisions, Decision-1, Classify-1, Tev1, Kev 4B.

---

## 2. Pricing and limits (what the catalogue and docs say)

- **Billing is input-token only** for all 27; output $0 [V]. Price range $0.01-$0.24/M. At the project's likely payload (a bucketed state of ~100-300 tokens plus an 11-option glossary of perhaps 150-400 tokens), that is under **$0.0001 per decision** [I]. Even 100,000 decisions is of the order of a few dollars; **cost is irrelevant to model choice** [I].
- **Token counting differs a lot between vendors**: on the same compact set Jev billed ~77k input tokens vs Microsoft Decision-1 ~26k, so list price per M tokens does not predict the bill [V-sec: akitaonrails; rbeckner says the same]. Measure `usage.input_tokens` and `usage.cost` in the live probe.
- **Rate limits**: OpenRouter docs state none for the Decisions endpoint [V: API ref says "does not state limits on questions, options, state size, rate"]. Jev/TypeSafe: one secondary source says 1,200 RPM / 250k tokens per second during early access, another says unpublished [V-sec, conflicting, U]. Microsoft Decision-1 returned about 10/504 HTTP 429s in a burst test [V-sec]. **Plan for a retry-once policy and count 429s as fallback events.**
- **Options per choice question**: not documented by OpenRouter. Known caps: Solar Decide Flash 26; Tev1 2-24; Clef up to 64 questions per request [V-sec]. 11 actions is well within all of them [I]. Option-order bias exists (reordering changes 7.0% of Jev's answers and 11.5% for JEV-9B on 16-option questions [V-sec: redreamality]), so **randomise option order across a probe set and keep the glossary fixed in the real runs**.
- **Context** is irrelevant for a bucketed state (every model has at least 8k; Kev 4B's 8,192 is the smallest) [I].
- Dated snapshots: OpenRouter exposes canonical slugs with dates (e.g. `perplexity/pplx-decider-flash-v1.2-20261016`); use these for reproducibility, never the `~...-latest` aliases [V: model `canonical_slug`].

---

## 3. Request / response schema: same or different?

**OpenRouter Decisions (all 27)** [V: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md, accessed 2026-10-17]:

- `POST /api/alpha/decisions`. Required `model`, `state` (string | object | array), `questions` (map id -> question). Optional `provider` routing prefs (`order`, `only`, `ignore`, `allow_fallbacks`, `sort: price|throughput|latency|exacto`, `preferred_max_latency` with p50/p75/p90/p99 cutoffs, `quantizations`, `zdr`, `data_collection`, `max_price`), `session_id`, `user`, `trace`.
- Question types: `noul` (yes/no), `choice` (map option -> guidance), `score` (ordered levels), and now `box` (image bounding box, Jev 1.14 only per release coverage [V-sec]).
- Response: `id` (`gen-dec-...`), `model` (served snapshot), `provider`, `answers` (map keyed by question id; `choice` + `confidence` + `probabilities`), `usage` {input_tokens, output_tokens, cost}.
- **`provider.sort: "latency"` and `preferred_max_latency`** are directly useful for multi-route models (Clef has 3 routes, Clef-flash 3, Kev 2, Solar 2). To force a route for a clean probe, use `provider.only` or `order` with `allow_fallbacks: false` [V schema, behaviour I].
- Per-model quirks that break "just swap `model`":
  - `togethercomputer/tev1-4b-experimental`: **one question only**, 2-24 options, `max_completion_tokens: 8` [V-sec]. OK for the project (one 11-way question) but cannot add auxiliary `noul` questions.
  - `respan/span-01*`: scores *behaviours* in a conversation span, not a general choice [V description]; do not use.
  - `upstage/solar-decide*`: 26 candidates/choice cap [V-sec]; fine for 11.
  - Image/audio/video inputs only on some (Clef-omni accepts all four; text-only: Decision-1, Drex, Solar, Tev1, Kev, Span-01) [V]. Irrelevant for a bucketed text state.
- Confidence/probability semantics are **not guaranteed to be comparable**: Jev is overconfident (ECE 0.074 on one index, mean confidence 0.81 vs accuracy 0.74), Perplexity has temperature stored in weights, OpenAI returned 94/6 on a fair coin [V-sec]. **Calibrate per model on a held-out set before thresholding.**

**OpenAI native `POST /v1/decisions`** [V: https://developers.openai.com/api/docs/guides/decisions, fetched 2026-10-17; the page that returned 404 on Oct 4 now exists]:
- Request `model` (only `gpt-6-luna`), `input` (string or message parts, images inline base64 only), `questions` = **array** of `{name, type: predicate|choice|score, ...}`, choice options as `choices: [{value, description}]`.
- Response `answers` = **array** in request order; `predicate -> probability`, `choice -> choice, probabilities[] (array), confidence`, `score`, plus a `refusal` variant.
- Billing $0.10/M input only; public beta, GA "in coming weeks"; ZDR/HIPAA eligible; US/EU data residency; "up to 10x faster than Responses API", no absolute figure [V]. Temperature/streaming/tools not supported [V-sec: apidog].
- **Different from System One schema** (array vs map, `predicate` vs `noul`, `input` vs `state`). Going through OpenRouter (`openai/gpt-6-luna-decisions`) normalises it to the OpenRouter schema [V: description says it is "served through OpenAI's Decisions API"; schema normalisation is I]. For the project, prefer the OpenRouter route for one code path; use the native route only to measure OpenRouter overhead.

**Cloudflare direct** `POST https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/@cf/cloudflare/clef-flash` with System One-compatible body (`state` + up to 64 typed questions) [V-sec: developersdigest; I did not find or fetch Cloudflare's own page, treat as U until the first call]. 4 images max, 4 MiB each, 13 MiB body [V-sec].

**TypeSafe direct** (`api.typesafe.ai` / `/v1/systemone`): I could not confirm the endpoint, limits or a latency SLA from any primary page. Secondary sources say direct API is early-access/waitlisted and `jev-latest` / `jev-preview` aliases exist [V-sec]. Marked **U**.

---

## 4. Published benchmarks (all vendor-run or small independent sets; no primary peer review)

| Source | What | Result | Caveat |
|---|---|---|---|
| Perplexity model card via BenchLM (https://benchlm.ai/benchmarks/perplexity-decision-panel) [V-sec] | 7,210 rows, 11 task families, accuracy | Decider v1 27B 85.71%, Jev 1.13 84.51%, Qwen3.8-27B 74.76% | Vendor run, not rerun, no calibration measure |
| akitaonrails (https://akitaonrails.com/en/2026/10/10/which-jev-like-ai-decision-classifier-model-is-best/) [V-sec] | 168 compact EN/PT-BR cases + 130 v1, hosted client latency | Compact acc: Decider 1.1 local 99.4, **Jev 97.6 (255 ms)**, **Microsoft Decision-1 97.6 (422 ms)**, Clef 27B 95.8 (302 ms), **OpenAI Decisions 90.5 (222 ms)**, Clef-flash 87.5 (394 ms), Kev 4B 87.5 | AI-written labels; 12 scenarios per cell; OpenAI weak on numeric thresholds (8/12), 94/6 on fair coin |
| rbeckner (https://rbeckner.com/articles/decision-model-benchmark/) [V-sec] | 24 frozen cases, 21 questions, 56-option function choice, serial via gateway | Luna Decisions 24/24 **181 ms med / 307 p95**; Jev 24/24 243 / 324; Decider V1 24/24 276 / 429; Decision-1 24/24 430 / 487; Decider V1.1 24/24 435 / 616; Liquid d1 23/24 472 / 962; Drex v1.5 22/24 582 / 745; Kev 4B 22/24 1980; Clef-flash 21/24 952 / 1168 | n=24, p95 on 24 samples, labels by agent. Solar Decide Flash excluded for 26-option cap |
| Arize (https://arize.com/blog/decision-model-benchmark/) | Jev vs Liquid d1 | Cited via search summary only: acc 91.1 vs 87.6, p50 0.08 s vs 0.20 s, ECE 0.023 vs 0.051 | **Page returned HTTP 403; numbers from a search summary, treat as U** |
| redreamality survey (https://redreamality.com/blog/open-decision-models-2026-survey-benchmarks/) [V-sec] | Aggregates Jev Decision Index 0.3.1 and JevBench | Jev 1.13 DI 60.11 (#3), pplx-decider v1.1 DI 62.75 (#1), Clef DI 53.08 (#19) with new-domain score ~20 pts lower than public, Clef-flash 47.61, d1-3B 40.08 at **16 ms median**, H2O-Lightning-4B 29 ms | "Decision Index" is run by the Jev ecosystem; Clef generalisation gap is a warning |
| BenchLM JevBench v1.5.4 (https://benchlm.ai/decision-models) [V-sec, retrieved 10-15] | 1,624 decisions, composite index, 106 ranked | Jev 1.13.0 **#3 72.13**; GPT-6 Luna (low effort) 40.48; Gemini 3.1 Flash-Lite 19.58; kev 4B 38.07; DeepSeek V4.1 Flash 6.65 | Index, not accuracy. Hosted Clef/Decider/Decision-1/OpenAI-Decisions not ranked in this list. Shows general chat LLMs well below decision models |
| Cloudflare vendor run (via https://www.developersdigest.tech/blog/cloudflare-clef-decision-models-2026) [V-sec] | Jev Decision Index 0.2.1 | Clef BFCL 98.5 vs Jev 95.8; BANKING77 94.2 vs 79.7; Jev wins GPQA, MMLU-Pro, When2Call. Latency median/p95: Clef 209/239 ms, **Clef-flash 38.8/122 ms**, Jev 524/536 ms | Vendor, direct region, "Jev 524 ms" contradicts every independent Jev measurement (118-255 ms). Treat as marketing |
| Liquid d1 launch (aiweekly/alphasignal summaries) [V-sec] | d1-3B open weights | 8 ms on RTX 4090, 26 ms Jetson AGX Orin, 50 ms Orin Nano per single question; DI 0.2.1 48.57 | Vendor-reported; 2 days old at time of reporting |
| Jev vs Decider (https://jevstation.com/compare/jev-vs-perplexity-decider) | Banking77, AG News | Acc tied; ECE 0.052 vs 0.086 favouring Decider; Jev 412 ms vs Decider 493 ms | Third-party site, via search summary only (U) |

**Nothing here benchmarks control/action-selection tasks.** All are classification / routing / function-selection. No source tests a 11-way short-horizon control choice, and none tests determinism under identical input. [I] Hence the project's own probe is the only evidence that matters.

Calibration summary: Jev overconfident but usable after temperature fit; Perplexity Decider best-calibrated in two independent tests (Brier 0.012, ECE 0.052); OpenAI Decisions showed bad probability behaviour on coin probes (94/6); Microsoft Decision-1 99.5% confident errors on long Portuguese context [V-sec]. For bucketed English-like state these may not transfer [I].

---

## 5. Latency evidence, side by side

| Model | OpenRouter endpoint p50 / p99 (10-17) [V] | Independent client median / p95 [V-sec] | Vendor claim [V-sec] |
|---|---|---|---|
| jev-1.13 | 118 / 353 | 243 / 324 (rbeckner); 255 (akita); 135 (Celeris-measured, us-east-1) | 114 ms demo |
| jev-1.14 | 135 / 590 | none yet | |
| gpt-6-luna-decisions | 190 / 945 | **181 / 307** (rbeckner); 222 (akita) | "up to 10x faster than Luna" |
| clef-omni | **117** / 656 | none | |
| clef (best route PI) | 242 / 590 | 302 (akita) | 209 / 239 direct |
| clef-flash (PI) | 177 / 577 | 394 (akita), 952 / 1168 (rbeckner) | **38.8 / 122 direct** |
| pplx-decider-v1.2 | 273 / 3085 | v1: 276 / 429; v1.1: 435 / 616 | |
| pplx-decider-flash-v1.2 | 182 / 1966 | none | |
| liquid/d1-ultra | 173 / 2354 | none | |
| liquid/d1 | 231 / 1307 | 472 / 962 | d1-3B 8-50 ms on-device |
| microsoft-decision-1 | 259 / 1004 | 422-430 / 487 | "2.5x faster than H2O-Lightning-4B" |
| tev1-4b | 191 / 373 | 0.26 s p50, 0.38 s p95 (Eden AI, third-party aggregator) | |
| mercury-decide(:free) | 277 / 1502 | none | "up to 14 decisions/s" (throughput, not latency) |

Reading: through OpenRouter, ~120-280 ms p50 and p99 above 1 s for most. For a 48 Hz control loop (20.8 ms tick), every hosted option costs 6-14 ticks p50 and up to 30-150 ticks at the tail [I, from MASTER-PLAN control rate]. The latency-sweep design already handles that. The difference between 118 ms and 270 ms matters only on the reaction-budget curve in the 0.1-0.3 s region [I].

The snapshot numbers differ run-to-run (doc 01 reported Jev p50 194-320 ms; today 118 ms): single rolling window, region unknown, so these are a **ranking aid, not a measurement** [I].

---

## 6. Per-model verdict for an 11-way action choice on bucketed state

Criteria: pinned/immutable id, accuracy on short structured state, tail latency, option cap, maturity.

| Model | Maturity | Verdict |
|---|---|---|
| typesafe/jev-1.13 | GA-like, alpha endpoint, stable snapshot | **Primary.** Lowest p50, lowest p99 among established models, best Jev-bench rank among hosted, overconfident -> calibrate |
| typesafe/jev-1.14 | New, experimental extras, unproven | Probe as a drift/upgrade arm; pin `jev-1.14-20261014` |
| openai/gpt-6-luna-decisions | Public beta (via OpenRouter and native) | **Strong second.** Independent p95 307 ms, 24/24 on routing test; but moderated flag, weak numeric thresholds (bucketing mitigates), bad coin calibration, GA pending. Native schema differs |
| cloudflare/clef-omni | New, vendor-claimed best-in-class | **Probe**: best p50 at snapshot; unknown quality; modest generalisation gap warning for Clef family |
| cloudflare/clef (PrimeIntellect route) | New | Probe; pick route explicitly (route spread 242 vs 517 ms) |
| cloudflare/clef-flash | New | Vendor latency leader; independent accuracy 87.5 / 21 of 24, confidence gating rejected 9 of 12 valid requests. Cheap arm, not first choice |
| perplexity/pplx-decider-v1.2 | New (10-16), v1.1 well tested | **Probe**: best accuracy/calibration in independent tests; p99 3 s is a concern |
| perplexity/pplx-decider-flash-v1.2 | New | Skip until a tail exists; p75 717 ms |
| liquid/d1-ultra, liquid/d1 | New, 23/24 for d1 | Probe d1-ultra if budget allows; p99 2.3 s |
| microsoft/microsoft-decision-1 | New, Foundry-backed | Good accuracy, slower (~430 ms client), 429s under burst; optional arm |
| inception/mercury-decide(:free) | New | **Free dev arm** for harness debugging; accuracy unknown |
| togethercomputer/tev1-4b-experimental | Experimental | Tight tail (p99 373 ms) and 11 options fits 2-24; but 22/24 accuracy and one-question limit. Worth a quick probe for tail |
| nace-ai/drex-v1.5 / v2.0 | New | Mid pack; v1.5 22/24, 582 ms client; skip unless time |
| scaledown/classify-1 | New | Document classifier; skip |
| upstage/solar-decide-flash | New | 495 ms p50 and 26-option cap; skip |
| upstage/solar-decide | | **Exclude**: 8.4 s p50 |
| jaredpalmer/kev-4b | Research preview | **Exclude** (8k ctx, 1.98 s client latency, 22/24, gating accepted 0/12) |
| respan/span-01*, span-02 | span-01 is behaviour scoring | **Exclude span-01 family**; span-02 skip unless asked (p90 2.1 s) |

---

## 7. Alternatives outside the OpenRouter Decisions endpoint

- **OpenAI native Decisions API**: covered above; same model as the OpenRouter listing. Only worth calling directly to quantify OpenRouter overhead.
- **TypeSafe direct**: no primary docs found; waitlist reported [V-sec]; the OpenRouter `provider` for Jev is "TypeSafe" itself, so OpenRouter is proxying TypeSafe, adding some overhead of unknown size [I]. A direct comparison needs access (U).
- **Cloudflare Workers AI direct** (`@cf/cloudflare/clef`, `clef-flash`): real, System One-compatible [V-sec]. It is the only route where I would expect the vendor-claimed ~40-200 ms. On OpenRouter the Cloudflare route is the **slowest** of the three Clef/Clef-flash routes (441-517 ms p50) [V], so the vendor claim does not survive the OpenRouter path. Worth one direct probe if the user has a Cloudflare account (the user would need to create/own the key, I did not).
- **Celeris-1 Decision** (hosted, direct, $0.04/M, claimed 67 ms for 1 question) and **Fastino GLiDE** [V-sec: BenchLM page]: not on OpenRouter; accuracy unestablished; Celeris's numbers are self-measured. Candidate only if the user wants to chase latency beyond OpenRouter.
- **General chat LLM with structured output** (Gemini 3.1 Flash-Lite enum; Groq/Cerebras small models): I found **no primary or independent measurements** of single-token classification latency for Groq or Cerebras [searches returned generic throughput blogs]. JevBench ranks Gemini 3.1 Flash-Lite 19.58 and GPT-6 Luna (low effort) 40.48 vs Jev 72.13 [V-sec]. A Google AI forum thread reports Flash-Lite classification calls sometimes taking 15-20 s about 25% of the time [V-sec, one user]. Verdict: **not beating Jev on latency or accuracy**; keep Luna Responses-API as the "slow LLM" arm, as doc 15a already recommends.
- **Self-hosted open decision heads**: d1-3B 8 ms (RTX 4090) to 26 ms (Jetson Orin) [V-sec vendor]; H2O-Lightning-4B 29 ms; Decision 3.0 Edge 0.6B 45 ms, 61% accuracy; Jebadiah 4B v2 205 ms and 90.5% [V-sec]. These skip the network but add GPU/serving work and a different "arm" definition (local, not API). See section 8 for the decision.

---

## 8. Ranking: top 5 for this project

Ranking considers (1) tail-aware latency, (2) independent evidence of accuracy on short structured state, (3) schema compatibility with the existing Jev provider, (4) maturity, (5) reproducibility (dated snapshot).

| # | Model | Why | Confidence |
|---|---|---|---|
| 1 | `typesafe/jev-1.13` (pin `jev-1.13-20260917`) | Lowest p50/p99 combination of an established model, strongest rank among hosted in JevBench, already the design target | high |
| 2 | `openai/gpt-6-luna-decisions` | Independent p95 307 ms, 24/24 routing, public beta, same endpoint; OpenAI-scale infra | medium (beta, undocumented limits, moderated) |
| 3 | `cloudflare/clef-omni` (and `clef` via PrimeIntellect route) | Fastest p50 at snapshot; open weights, multiple routes; accuracy is the unknown and the Clef family generalises worse out of domain | medium-low |
| 4 | `perplexity/pplx-decider-v1.2` | Top independent accuracy/calibration; Apache-2.0 weights allow later self-hosting; tail risk (p99 3 s) | medium |
| 5 | `liquid/d1-ultra` (or `microsoft/microsoft-decision-1`) | d1-ultra p50 173 ms but 2 days old; Decision-1 has independent evidence (24/24, 97.6% compact) but ~430 ms client | low-medium |

Worth a live latency probe (in priority order): **Jev 1.13, Luna-Decisions, Clef-omni, Decider v1.2, d1-ultra, Decision-1, Jev 1.14, Tev1-4B**. Optional smoke arm: `mercury-decide:free` (free). Do not probe: Solar Decide, Kev 4B, Span-01*, ScaleDown Classify-1.

Non-API option, **local decision head** (d1-3B, H2O-Lightning-4B, Decision 3.0 Edge): is the only credible path to <50 ms but requires a GPU/Apple-silicon runtime and a separate harness arm. Recommended as a labelled "local" arm only if the user wants to push reaction latency below the hosted floor; not a replacement for the hosted arm [I].

---

## Decisions recommended

1. **Keep `typesafe/jev-1.13` pinned (dated snapshot) as the primary decision-model arm.** Do not use `~typesafe/jev-latest` (moves to 1.14 on about 10-21) or any `~perplexity/*-latest` alias.
2. **Generalise the provider to `OpenRouterDecisionsProvider(model_id)`** (already suggested in 15a). All the shortlisted models share the schema, so one client covers them. Add **one** question of type `choice` with 11 options; do not use auxiliary questions initially (Tev1 cannot).
3. **Add `openai/gpt-6-luna-decisions` as the second arm** via OpenRouter. Supersede doc 15a's "defer OpenAI" decision: the OpenAI product is now public beta and reachable through OpenRouter with the same schema.
4. **Run the latency/accuracy probe on the 8 models above** and choose the "decision-model family" for the main experiment from probe data, not from vendor claims. Make `provider.only=[x]` and `allow_fallbacks=false` standard in probe requests so the route is explicit and recorded (`provider` and `model` fields in the response, dated snapshot).
5. **Treat all vendor and Jev-ecosystem benchmarks as non-transferable.** Build the project's own 200-300 labelled drone states (bucketed) and measure accuracy, calibration (ECE/Brier), option-order sensitivity, and run-to-run determinism per model.
6. **Exclude** `upstage/solar-decide`, `jaredpalmer/kev-4b`, `respan/span-01*` from experiments.
7. **Record route-level latency, not just model-level**, because Clef and Clef-flash latency varies 2-3x by route.
8. **Describe the latency arm honestly in write-ups**: the hosted floor through OpenRouter is ~120 ms p50 (snapshot); cite per-model measured percentiles from the project's own probe.

## Open questions

1. Which of the models actually handle a **short, bucketed numeric/ordinal state** well? Existing benchmarks are text classification. Numeric thresholds are a named weak spot (OpenAI Decisions 8/12; "decision models weak at quantities" per OpenRouter's own skill page [V]). Will bucket labels as words (e.g. `OBSTACLE_CLOSE`) help?
2. How much latency does **OpenRouter add** vs direct vendor calls (Cloudflare direct, OpenAI direct, TypeSafe direct)? Unmeasured. Cloudflare's own 38.8 ms vs 441 ms through OpenRouter's Cloudflare route is an upper bound on routing + region effects but not a clean measure.
3. **Is Jev 1.14 an accuracy/latency regression or improvement** vs 1.13 on short English state? Only press coverage so far; becomes default on about 10-21.
4. **Do any of these models give deterministic output** for identical input and identical option order? Not reported anywhere I found.
5. What are the **free-tier rate limits** for `mercury-decide:free` and `span-01-lite:free`? Not in anything I could read.
6. Does `provider.preferred_max_latency` / `sort: latency` actually steer to a lower-latency route for Clef, or only for ranking? Docs say it exists [V]; behaviour untested.
7. OpenAI native vs OpenRouter-proxied Luna Decisions: same model weights/behaviour? Description says "served through OpenAI's Decisions API" [V], so presumed yes [I].
8. Moderation flag (`is_moderated: true` for Luna Decisions): could a drone-collision prompt trigger refusals? A `refusal` answer variant exists in OpenAI's native schema [V]; check the OpenRouter normalisation.
9. Do ToS of each vendor allow publishing latency/accuracy comparisons? (carried from docs 01, 15a)
10. **Direct TypeSafe API access and real SLA**: unconfirmed.

## Follow-up spikes

- **S1 (30 min, key already available to the probe agent)**: for each shortlisted model, 200 sequential calls with a fixed bucketed state, `provider.only` set, `allow_fallbacks=false`; record client-side p50/p95/p99, HTTP status mix, `usage.input_tokens`, `usage.cost`, served `model` snapshot id, `provider`. Interleave models to cancel drift; include a canary request repeated every 20 calls. Output: ECCDF per model.
- **S2 (1-2 h)**: 250 labelled drone states generated from the viability-set labels in MASTER-PLAN section 7; run all shortlisted models; report accuracy, top-2 accuracy, ECE, Brier, option-order sensitivity (shuffle option order), invalid/refusal rate.
- **S3 (30 min)**: determinism test: same input x 50 per model, count answer flips and probability spread.
- **S4 (30 min, user-gated)**: Cloudflare Workers AI direct call to `@cf/cloudflare/clef-flash` and `clef` from the experiment machine, to check the 38.8 ms claim and OpenRouter overhead. Needs a Cloudflare account/token owned by the user.
- **S5 (30 min)**: OpenAI native `/v1/decisions` with `gpt-6-luna` vs `openai/gpt-6-luna-decisions` via OpenRouter, same payload; quantifies proxy overhead and checks schema normalisation, moderation behaviour.
- **S6 (optional, 2 h)**: local decision head (d1-3B or H2O-Lightning-4B) on the user's hardware, to establish whether a sub-50 ms arm is feasible; only if the project wants a "local System One" arm.
- **S7 (watch)**: re-run the catalogue listing weekly (it grew 13 -> 27 in 13 days); re-pin when Jev 1.14 becomes stable and when OpenAI Decisions reaches GA.

---

## Sources (accessed 2026-10-17)

Primary / exact:
- OpenRouter model API: `https://openrouter.ai/api/v1/models?output_modalities=decisions` and `=all`; `https://openrouter.ai/api/v1/models/{slug}/endpoints` for each of 27 entries (read-only GET).
- OpenRouter Decisions API reference: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md
- OpenRouter Decisions skill: https://openrouter.ai/skills/openrouter-decisions ; index https://openrouter.ai/docs/llms.txt
- OpenAI Decisions API guide: https://developers.openai.com/api/docs/guides/decisions

Secondary (third party, mostly small-n or vendor-run):
- https://akitaonrails.com/en/2026/10/10/which-jev-like-ai-decision-classifier-model-is-best/
- https://rbeckner.com/articles/decision-model-benchmark/
- https://redreamality.com/blog/open-decision-models-2026-survey-benchmarks/
- https://benchlm.ai/decision-models ; https://benchlm.ai/md/decision-models.md ; https://benchlm.ai/benchmarks/perplexity-decision-panel
- https://www.developersdigest.tech/blog/cloudflare-clef-decision-models-2026
- https://arize.com/blog/decision-model-benchmark/ (403 to my fetcher; numbers from search snippet only)
- https://jevstation.com/compare/jev-vs-perplexity-decider (search snippet only)
- Search-result summaries (not fetched pages): Liquid d1 launch (aiweekly.co, alphasignal.ai, orcarouter.ai), Jev 1.14 (remio.ai, ecosystemastartup.com, zeli.app), Microsoft Decision-1 (infoworld.com, letsdatascience.com, testingcatalog.com), Nace Drex (huggingface.co/nace-ai/drex-v2.0, techjacksolutions.com), Tev1 (edenai.co, pricingsaas.com), Mercury Decide (OpenRouter listings), OpenAI Decisions beta coverage (letsdatascience.com, propakistani.pk, apidog.com, aicatchup.com), Jev pricing/limits (flaviocopes.com, layer3labs.io, opentweet.io, eesel.ai), Gemini Flash-Lite (discuss.ai.google.dev thread, emelia.io), Groq/Cerebras (generic blogs, no usable measurement).
- Prior repo research: docs/research/01-jev-identity-openrouter-latency.md, docs/research/15a-gpt-decisions-api-identity.md, docs/MASTER-PLAN.md (11 actions, 48 Hz control, latency injection design).
