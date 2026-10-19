# 16c - Live feasibility probe (Jev + other decision models + OpenAI + Gemini)

Date: 2026-10-17. Client: user's Mac in India (IST). Region inferred from Cloudflare `cf-ray` suffix on responses (MAA = Chennai edge; HYD seen on some OpenRouter calls), not from any IP-lookup service. One persistent httpx 0.28.1 HTTP/2 client per host, `perf_counter_ns`, strictly sequential calls with a 350 ms pause, first 3 calls per model discarded as warm-up, 24 measured calls per model (8 scenarios x 3 rounds). Raw numbers: `16c-live-probe-results.json` (no keys, no prompts, headers filtered to rate-limit/ray ids). Scratch code: `/private/tmp/claude-501/-Users-subhajithait-Documents-testing-drone-jev/0ec4827c-5206-473e-b48d-298a2822fdb4/scratchpad/16c`.

Key handling: keys were read from `.env` inside the Python process only and sent only in headers (Gemini via `x-goog-api-key`). No key was printed, logged, or saved. **No exposure incident.**

## 0. Step 0 (free auth check, HTTP status only)
OpenRouter `GET /api/v1/key` 200 · OpenAI `GET /v1/models` 200 · Gemini `GET /v1beta/models` 200. All three keys authenticate. No 401/402/403/429 occurred in the whole run.

## 1. Setup
- Prompt: doc 07 system prompt style (frame, priority rules, 11 actions) + hybrid terse observation (`age= last= v= bat= gps=`, `tgt=`, `rng[F,FL,L,BL,B,BR,R,FR]`, `near= clos= ttc=`). About 490-850 tokens depending on tokenizer. No few-shot examples.
- Decisions endpoint (Jev and peers): `POST https://openrouter.ai/api/alpha/decisions`, one `choice` question with 11 criteria (action name -> description), state = system text + observation. **The path `/api/alpha/decisions` works live** (the `/api/v1/api/alpha/...` print in docs is a typo). Bearer auth.
- OpenAI: Chat Completions, letter aliases A-K (single token), `temperature` default for Luna, `logprobs`+`top_logprobs`, 1-token answer.
- Gemini: `generateContent`, plain 1-letter output, `thinkingConfig.thinkingLevel=minimal`.
- 8 labelled states (acceptable set in brackets): clear path {FORWARD}; obstacle dead ahead ttc 0.5 {AVOID_L, AVOID_R, BRAKE}; obstacle right, left free {AVOID_LEFT, BRAKE}; obstacle left, right free {AVOID_RIGHT, BRAKE}; battery 5% {RETURN_HOME, EMERGENCY_LAND}; battery 18% {RETURN_HOME}; GPS lost {HOVER, BRAKE, RETURN_HOME}; at target {HOVER, BRAKE}.
- **n = 24 per model (8 states x 3 repeats; repeats are near-deterministic, so effectively 8 independent states). Accuracy numbers are anecdotal, not statistics.**

## 2. Response schema observed (OpenRouter Decisions)
Top-level keys: `model`, `answers`, `usage`, `id`, `provider`. `answers.action` = `{type:"choice", choice, probabilities{11 keys, rounded to 2 dp}, confidence}`. `usage` = `{input_tokens, output_tokens, cost}` (cost in USD, direct). Snapshot ids returned in `model`: `typesafe/jev-1.13-20260917`, `inception/mercury-decide-20260930`, `togethercomputer/tev1-4b-experimental-20260923`, `jaredpalmer/kev-4b-20260924`, `cloudflare/clef-flash` (undated; served by "Cloudflare" in the run, by "PrimeIntellect" on one earlier call, so provider routing varies). **No rate-limit headers** came back from OpenRouter on the decisions endpoint (only `cf-ray`). Jev usage: ~849 input tokens, `output_tokens` 110 (billed $0 per docs; cost $0.0000357/call = input only).
Probabilities are coarse (0.00/0.01/0.97) for Jev; very peaked.

## 3. Results

| Model (route) | n | p50 ms | p95 ms | max ms | err | Acc (n=24) | Probs | Est. cost/call |
|---|---|---|---|---|---|---|---|---|
| typesafe/jev-1.13 (OR Decisions) | 24 | **531** | **961** | 2502 | 0 | 0.75 (18/24) | yes (2dp) | $0.000036 |
| inception/mercury-decide:free | 24 | 453 | 811 | 836 | 0 | **1.00** (24/24) | yes | $0 |
| cloudflare/clef-flash | 24 | 620 | 1111 | 1470 | 0 | 0.625 | yes | $0.000015 |
| togethercomputer/tev1-4b-experimental | 24 | 521 | 608 | 626 | 0 | 0.50 | yes | $0.000026 |
| openai gpt-4o-mini (Chat, logprobs top20) | 24 | 688 | 1028 | 2566 | 0 | 0.75 | yes (logprobs) | $0.000082 |
| openai gpt-4.1-nano (Chat, logprobs top20) | 24 | 628 | 706 | 948 | 0 | 0.625 | yes (logprobs) | $0.000055 |
| openai gpt-6-luna, reasoning_effort=none | 24 | 973 | 1225 | 1696 | 0 | 0.83 (20/24) | yes, top5 only | $0.000056 |
| gemini-3.5-flash-lite, thinking minimal | 24 | 801 | 1104 | 1198 | 0 | 0.92 (22/24) | no | ~$0.000057 (price assumed) |
| conc=4, gpt-4o-mini (16 calls) | 16 | 1263 | 1521 | 1951 | 0 | n/a | | 3.3 req/s, no 429 |

Notes
- p95 on n=24 is essentially the 2nd-largest value; treat as indicative. Jev max 2502 ms was one outlier (1 in 24).
- Gemini first attempt with `maxOutputTokens=4` returned empty content (finishReason MAX_TOKENS even with thinking minimal); 16 works. Those 27 calls are invalid (accuracy 0) and kept only in the JSON under `main` with an INVALID note; the table uses the rerun (`extra.gemini`).
- Luna needs `max_completion_tokens` >2 (2 gave HTTP 400 "limit reached", 16 works; it emitted 4 completion tokens). `top_logprobs` max is **5** for Luna (20 gives 400). **Luna with `reasoning_effort=none` accepts `logprobs`**, resolving doc 15b's #1 open question, but only 5 alternatives, so 11-way distributions are truncated (alias mass ~0.99 observed).
- gpt-4o-mini/4.1-nano accepted top_logprobs 20 and returned proper 11-letter distributions (alias mass ~1.0).
- Rate-limit headers: OpenAI exposes them: gpt-4o-mini/4.1-nano 30,000 RPM, 150M TPM; Luna 30,000 RPM, 180M TPM (Tier 5-like). OpenRouter decisions: none exposed. Gemini: none (only `server-timing`, e.g. `dur=1028` ms server-side for the Gemini call, so ~all of Gemini latency is server-side plus ~230 ms network).
- Concurrency-4 test only on gpt-4o-mini (highest RPM): no 429, 3.3 req/s, per-call latency rose from p50 688 to 1263 ms under 4 parallel (client-side/HTTP2 or provider queueing; not investigated). OpenRouter/Jev concurrency was NOT tested (the instruction limited the concurrency test to the highest-limit model, and the Jev limit is unknown/unexposed). The master plan's "at least 10 req/s sustained" in G3 is therefore **unverified**.

### Network floor from this machine (8 free GETs each, warm connection)
OpenRouter edge metadata GET: **52 ms** p50 (first 470 ms incl. TLS). Gemini metadata GET: 229 ms p50. OpenAI metadata GET: **865 ms** p50 (OpenAI API appears to be served from far away, or the metadata endpoint is slow; the earlier `openai-processing-ms` header said only 267-668 ms server time, so ~300-500 ms of the OpenAI latency is path/network).
So for Jev: ~531 ms total = ~50 ms to OpenRouter edge + ~480 ms OpenRouter-to-TypeSafe + inference. The published 200-330 ms p50 (doc 01) was presumably measured from the US near the TypeSafe origin (US West); from India, expect roughly +200-300 ms. That is consistent with doc 01's warning.

## 4. Comparison against published numbers and master-plan gates
- Published Jev p50: ~200-330 ms (jev-drone 110-118 ms). Measured here: **531 ms p50, 961 ms p95**. About 1.6-2.7x slower, plausibly due to region (India to US).
- **G3 (p50 <= 400 ms, p95 <= 1 s, error < 1%, >= 10 req/s): FAILS the p50 from this location (531 > 400); p95 passes marginally (961 < 1000, n=24 so fragile); error 0/24 + 0/27 warm-up passes; throughput untested.** Not passable as-is from India unless the runner is moved to a US region (or a cloud VM) or the gate is relaxed to ~600 ms p50. The honest framing: it is a location problem more than a model problem; a US-hosted runner is the cheap fix to test (Follow-up F1).
- Best latency among all tested: mercury-decide:free p50 453 ms (free), then tev1 521 ms. **No tested model meets 400 ms p50 from India.** The fastest *network* path is OpenRouter (52 ms floor); OpenAI direct and Gemini direct were slower (688-973 ms).
- **G4 (>= 0.7 acceptable-action rate): Jev 0.75 on this tiny set, borderline pass, with a clear, explainable failure.** All 6 Jev misses were the two battery states (FORWARD at bat=5 and bat=18; 6/6 deterministic). With an explicit state flag `batt_state=CRITICAL(<8%)` / `LOW(<25%)` appended, Jev picked EMERGENCY_LAND (3/3) and RETURN_HOME (3/3) at ~480-560 ms. So the failure is encoding (a raw `bat=5` number is not salient), not capability; precomputed buckets (already the master plan's mitigation) matter. Jev got all obstacle, clear-path, GPS-lost and at-target states right. The 0.7 G4 check applies to "clear-threat states" (obstacle ones here: Jev 9/9). With flags added, effective Jev ~8/8 on these states. Still n is tiny and scenarios are easy; do not claim competence beyond "not obviously broken".
- Other models: mercury-decide:free perfect 24/24 (n small; also best latency and free, a serious arm candidate); gemini-3.5-flash-lite 22/24; Luna 20/24 (misses: dead-ahead sometimes EMERGENCY_LAND/RIGHT, left-obstacle AVOID_LEFT = the wrong side, a symmetry error worth probing); tev1-4b 12/24 and clef-flash 15/24 mostly ignore battery/GPS/at-target states (default FORWARD). Left/right mistakes: Luna chose AVOID_LEFT with an obstacle on the left (twice), which is the kind of failure a mirror test would catch.
- gpt-4.1-nano and gpt-4o-mini are poor on battery and at-target states too (FORWARD), same salience issue.
- Positional/alias effect: LLM models were given letter aliases without semantic names; accuracy can differ with names (factor OUT in doc 07). Not tested.

## 5. Cost
Total approx. 305 paid calls (incl. ~16 exploratory) across 8 models; **estimated total spend about USD 0.011** (Jev 3 x ~$0.001, OpenAI ~$0.006, others fractions of a cent; Gemini priced at an assumed $0.10/$0.40 per M since no price was fetched). Well under the USD 3 cap. No credit or billing warnings.

## 6. Other findings
- `jaredpalmer/kev-4b` call: works, but `output_tokens` 166 and 2.3 s on the single call (reasoning-like output; not benchmarked, 1 call only).
- OpenRouter Cloudflare edge colo seen: MAA and HYD (India); provider routing for `cloudflare/clef-flash` varied between PrimeIntellect and Cloudflare.
- Latency spread: Jev 446-693 ms except one 961 and one 2502 ms; tev1 is remarkably tight (506-626 ms, 4B model, p95/p50 = 1.17); mercury has occasional 800 ms spikes.
- Probabilities: all decision models and both OpenAI chat models returned usable per-action probabilities; Gemini did not (no logprobs requested/available in this simple path).

## Decisions recommended
1. Keep `typesafe/jev-1.13` via OpenRouter as the primary arm, but **stop treating 200-330 ms as the expected latency**; plan for ~530 ms p50 / ~1 s p95 from India, and run the real experiment from a US-region machine (or inject latency from recorded numbers) if G3 as written is to be met.
2. Re-state G3 as region-specific: "p50 <= 400 ms measured from the experiment host; if run from India, p50 <= 600 ms, p95 <= 1.2 s". Mark ">= 10 req/s" as unverified.
3. Add explicit precomputed buckets for battery and GPS (`batt_state=CRITICAL/LOW/OK`, `gps=LOST`) and the nearest-obstacle ttc, as the master plan intends; the probe shows raw numeric battery is ignored by every small model.
4. Add `inception/mercury-decide:free` as a second decision-model arm (fastest, free, 24/24 here); add gemini-3.5-flash-lite and gpt-4o-mini as the "fast LLM" arms; gpt-6-luna as a slower, higher-capability LLM arm. Drop tev1 and clef-flash as arms for now (poor on non-obstacle states, small n).
5. For OpenAI logprobs: gpt-4o-mini/4.1-nano with `top_logprobs=20`; Luna requires `reasoning_effort=none`, `top_logprobs<=5`, `max_completion_tokens>=~8`.
6. Gemini: use `maxOutputTokens >= 16` with `thinkingLevel=minimal` (4 returns empty).

## Open questions
1. Is Jev's latency dominated by region (India to US-West) or by load? Needs the same test from a US VM, and repeated at other times of day (this was one ~5 minute window).
2. Real rate limit of OpenRouter Decisions/Jev (no headers returned) and whether 10 req/s is sustainable; not tested.
3. Does the 110 ms output_tokens=110 on Jev (usage field) reflect internal reasoning or accounting? Billed at $0 output.
4. Why did conc=4 roughly double latency on gpt-4o-mini (688 to 1263 ms)? Client HTTP/2 stream behaviour or provider queueing.
5. Luna's left/right error (AVOID_LEFT with a left obstacle): real bias or alias confusion?
6. Does clef-flash/tev1's poor non-obstacle accuracy persist with the battery flag?

## Follow-up spikes
- F1 (30 min): repeat Jev + mercury from a US-region VM (e.g. cheap cloud shell) to separate region from model latency; 3 time-of-day windows.
- F2 (20 min): Jev sustained throughput probe: 10 req/s for 30 s via OpenRouter (watch 429s), only after the user approves a burst.
- F3 (30 min): 200-state labelled set with mirror pairs, battery/GPS buckets, and 3 prompt encodings; compare Jev, mercury, Gemini, gpt-4o-mini with bootstrap CIs (needed for a real G4 CI).
- F4 (15 min): same states with name aliases instead of letters for the chat LLMs.
- F5: determinism probe (same state x 50) on Jev vs mercury.
