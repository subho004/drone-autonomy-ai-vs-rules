# 09 - Orchestration, parallelism, throughput on M4 16 GB (no Docker)

Author: research agent 09. Date of research: 2026-09-30. Scratch work and raw outputs: `/private/tmp/claude-501/-Users-subhajithait-Documents-testing-drone-jev/0ec4827c-5206-473e-b48d-298a2822fdb4/scratchpad/09/` (scripts `scale2.py`, `httpx_bench.py`, `mock_server.py`, `sqlq.py`, `crash.py`, `model.py`).

Legend: **[V]** verified fact (measured here or read from a primary source), **[I]** inference, **[A]** assumption used as a model input. Confidence: H/M/L.

---

## 0. TL;DR

1. The machine is an **Apple M4, 10 cores = 4 P + 6 E, 16 GiB** [V]. E cores run scalar Python at **~0.57-0.6x** of a P core (measured, QoS-clamp test) [V], so the box is worth **~7.5 P-core equivalents** at most, ~6.5 after reserving one for orchestrator/SQLite/event loop [I].
2. **Only classical arms are CPU-bound.** Every LLM arm is bound by something else: local LLM by the single GPU (~0.5k flights/h), fast API by the provider **requests-per-minute cap** (~0.9k flights/h at an assumed 1000 RPM), slow API by concurrency/RPM (~7k flights/h). The in-flight concurrency needed is small (5-45 requests); concurrency is **not** the binding constraint, rate limits and GPU are.
3. **Recommendation: runner CLI as execution engine, FastAPI as thin control/monitoring plane over SQLite.** Do not run the pool inside uvicorn. **Ray: against.**
4. **SQLite as job queue works and is not a bottleneck**: 13-18k claim+complete ops/s with 1-10 spawn workers, zero duplicate claims, crash-recovery via leases verified (SIGKILL test) [V]. One trap found: an `OR` in the claim query defeats the index and made it 36x slower (369/s). Use partial indexes and two statements.
5. **Measurement caveat (important):** during all my runs the host had load average **45-95** (other research agents compiling pybullet with ~30 concurrent `clang`), so all wall-clock numbers below are contended and labelled as such. I compensated by measuring CPU-time-normalised rates, but *absolute scaling on 1..10 workers could not be demonstrated cleanly*. A clean re-run script is listed under follow-ups.

---

## 1. Machine facts (measured)

| Fact | Value | Source |
|---|---|---|
| Chip | Apple M4 (`machdep.cpu.brand_string`) | `sysctl` [V] |
| Cores | `hw.ncpu=10`, `hw.physicalcpu=10`, `hw.perflevel0.physicalcpu=4` (Performance), `hw.perflevel1.physicalcpu=6` (Efficiency) | `sysctl` [V] |
| RAM | `hw.memsize=17179869184` (16 GiB) | `sysctl` [V] |
| OS | `sw_vers`: ProductName macOS, ProductVersion 27.0, BuildVersion 26A428 | `sw_vers` [V]. Contradicts nothing directly but note: the plan says "macOS 26 tested"; this box reports 27.0 (build string 26A428). Agent 02 must check whether gym-pybullet-drones' "tested on macOS 26" holds on 27. |
| Python | uv-managed CPython 3.14.2 (matches repo); also 3.14.6 via Homebrew. `sys._is_gil_enabled()` = True (default GIL build) | [V] |
| Free-threaded 3.14 build | not installed; `uv python list` shows `3.14.2+freethreaded` "download available" | [V] |
| SQLite | 3.50.4 (Python's `sqlite3`) | [V] |
| Packages resolvable on 3.14.2 arm64 | numpy 2.5.3, httpx 0.28.1, aiohttp, uvloop installed; `pybullet 3.2.7` and `ray 2.58.0` resolve (uv dry-run, **not** installed/run) | [V] for resolve, M confidence that they run |
| Power | AC, battery 100%; `pmset -g therm`: no thermal/performance warning recorded | [V] |
| Bare worker cost | python+numpy+httpx+sqlite3 import: 0.47 s, 49 MB RSS; `spawn` of a trivial process ~0.07 s | [V] (under load) |

Host contention: `vm.loadavg` was 45-57 (1-min) during scaling runs and 52-95 later; `ps` showed dozens of `clang` processes at ~20% CPU each. Everything measured wall-clock is therefore a **pessimistic lower bound** for a quiet machine.

---

## 2. Experiments run

### 2.1 Multiprocessing `spawn` scaling, N = 1..10 workers

Workload: stand-in for one PyBullet control tick, pure-Python scalar loop (`py`, 2000 float ops/step) and small-matrix numpy (`np`, 20x `tanh(A@x)` with 12x12). Each worker runs 3 s after a `Barrier`; I record the **wall rate** and the **CPU-time-normalised rate** (iterations per CPU-second), best of 3 repetitions.

| N workers | py wall total it/s | py per-CPU-second rate per worker | np wall total it/s | np per-CPU-s rate | load avg (1 min) at run |
|---|---|---|---|---|---|
| 1 | 1,398 | 7,166 | 10,474 | 54,246 | 45 / 58 |
| 2 | 2,994 | 7,552 | 21,702 | 50,272 | 45 / 55 |
| 4 | 7,099 | 7,913 | 58,671 | 55,277 | 49 / 47 |
| 6 | 11,769 | 8,191 | 71,060 | 55,234 | 52 / 44 |
| 8 | 15,230 | 8,078 | 76,341 | 54,243 | 51 / 52 |
| 10 | 13,193 | 7,194 | 94,686 | 53,589 | 57 / 60 |

[V] measured. Interpretation:

- **Wall-clock "scaling" here is an artefact**: total throughput rose ~9.4x from 1 to 10 workers only because each extra process grabs another share of an oversubscribed scheduler (load 45-60 on 10 cores). Per-worker wall rate stayed at ~1.4-1.9k it/s, i.e. each worker got ~20-25% of a core. This does **not** demonstrate P/E scaling. [V for the numbers, the interpretation is I].
- The CPU-normalised rate is flat (7.2-8.2k it/s per CPU-second for `py`, ~54k for `np`) up to 10 workers: per-CPU-second speed does not collapse when all 10 cores are used, only a ~12% dip at N=10 (7,194 vs 8,191 at N=6) consistent with some time landing on E cores. Low confidence on that reading.
- **E-core vs P-core speed, direct test** using `taskpolicy -c background` (which confines a process to E cores; Apple: background QoS threads never run on P cores): default QoS `py` = 7,773 and 7,056 it/CPU-s; background clamp = 4,305 and 4,327 it/CPU-s. **E core ≈ 0.56-0.61x a P core** for scalar Python [V]. Source for the QoS mechanism: Eclectic Light Co., https://eclecticlight.co/2022/11/28/can-you-game-core-allocation-on-apple-silicon/ and https://eclecticlight.co/2025/05/09/what-is-quality-of-service-and-how-does-it-matter/ (accessed 2026-09-30). Consequence: `taskpolicy` is "a brake, not an accelerator": you can push work **onto** E cores, never pin it onto P cores; there is **no core pinning** on macOS (no `sched_setaffinity`).
- Effective capacity model used below: `C = 4 + 6*0.58 = 7.5 P-eq`, minus 1 P-eq reserved for orchestrator/event loops/SQLite writer = **6.5 P-eq** [I].

### 2.2 asyncio + httpx against a local mock server with injected latency

Setup: aiohttp mock server (`/v1/chat`, 100 ms `asyncio.sleep`, tiny JSON reply), httpx 0.28.1 `AsyncClient(Limits(max_connections=C, max_keepalive_connections=C))`, HTTP/1.1 keep-alive, 1.2 KB request body (~300-token prompt), C concurrent looping workers for 4 s, plus a 1 ms ticker coroutine measuring event-loop lag. Machine under load ~50.

| C in-flight | req/s | overhead p50 (ms over 100 ms) | overhead p99 (ms) | loop lag p99 (ms) | loop lag max (ms) |
|---|---|---|---|---|---|
| 1 | 9.5 | 4.5 | 17 | 5.0 | 15 |
| 4 | 37 | 5.4 | 13 | 3.7 | 17 |
| 16 | 60 (ideal 160) | 124 | 844 | 92 | 136 |
| 64 | 25 | 1,526 | 5,924 | 246 | 278 |
| 128 | 33 | 3,499 | 6,546 | 276 | 283 |
| 256 | 26 | 7,213 | 13,536 | 416 | 1,542 |
| 512 | 20 | 14,301 | 31,164 | 777 | 2,719 |

[V] but **heavily contaminated** by the saturated host (mock server is a separate single-threaded process that also got starved; ideal throughput at C=64 would be 640 req/s). What survives as evidence:

- At C<=4 the client+server stack adds **~4-5 ms p50 / 13-17 ms p99** on loopback even under load: a floor on measurement error when the loop is otherwise idle.
- Once CPU is oversubscribed, **event-loop lag and therefore measured "latency" blow up by 1-2 orders of magnitude** (p99 lag 92 ms at C=16, 0.78 s at C=512). Measuring API latency from inside a process that is also running sims, on a busy machine, is invalid. This is the quantitative reason for the dedicated latency probe in Section 6.
- I could not obtain the clean capacity number (expected: several hundred req/s per process for a mock with 100 ms latency). **Follow-up spike F1.** Planning figure used in the model: 1-3 ms CPU per request for client overhead (httpx is known to be slower than aiohttp; I did not measure clean) [A].

### 2.3 SQLite job-queue claiming (spawn workers, WAL, `synchronous=NORMAL`, `busy_timeout=30000`)

20,000 jobs, each worker loops `claim -> (optional sleep) -> mark done`, autocommit, one atomic `UPDATE ... WHERE id=(SELECT ...) RETURNING id` per claim.

| Workers | simulated work | claim+complete ops/s | claim p50 | worst-worker claim p99 | duplicates |
|---|---|---|---|---|---|
| 1 | 0 | 12,999 | 0.015 ms | 0.10 ms | 0 |
| 4 | 0 | 13,282 | 0.015 ms | 20.9 ms | 0 |
| 8 | 0 | 18,458 | 0.016 ms | 999 ms | 0 |
| 10 | 0 | 15,759 | 0.016 ms | 1,092 ms | 0 |
| 10 | 5 ms | 1,327 (= 10/0.0075) | 0.069 ms | 3.9 ms | 0 |

[V], SQLite 3.50.4, contended host. Findings:

- **Queue overhead is irrelevant**: even 20k ops/s is >4 orders of magnitude above the need (a flight costs ~0.1-20 s).
- **Trap:** my first version used `WHERE status='pending' OR (status='running' AND lease_until<?)`; SQLite cannot use one index for the `OR`, every claim scanned all completed rows: **369 claims/s for 20k jobs (54 s)**, degrading with table size. Fixed by a **partial index** `ON job(id) WHERE status='pending'`, a second partial index on leases, and **two statements** (pending first, then expired lease). 36x faster.
- Tail (p99 ~1 s at 8-10 workers with zero work) is writer-lock convoying through `busy_timeout` sleep/retry under CPU starvation; with realistic per-job work it disappears (3.9 ms p99). Use `BEGIN IMMEDIATE` for any multi-statement write to avoid upgrade deadlocks (`SQLITE_BUSY` is returned immediately when a read txn tries to upgrade).
- **Crash test (`crash.py`)**: 4 workers, lease 1 s, 40 jobs at 200 ms each; `SIGKILL` worker 0 mid-job. Result: 40/40 done, 0 stuck in `running`, exactly one job (id 19) reclaimed with `attempts=2` [V]. Lease-based resumability works.
- Agent 10 owns schema and the 8-writer result-write test; this section is only the *queue* path.

---

## 3. Process architecture (no Docker)

```
                          ┌──────────────────────────────────────────────┐
  uvicorn (FastAPI)       │ SQLite file  (WAL)  data/experiments.db      │
  read/ctl plane  ───────►│ experiments, scenarios, job, flights, ...    │◄─── results
  (POST experiment,       └───────▲─────────────▲──────────────▲─────────┘
   GET progress, SSE)             │ claim/lease │              │
                                  │             │              │
  scripts/run_experiment.py  (supervisor, NOT in uvicorn)
    ├─ pool "cpu"   : W_cpu spawn procs, 1 flight at a time, sync loop     (classical arms)
    ├─ pool "api"   : W_api spawn procs, asyncio, K flights interleaved     (API-fast / API-slow arms)
    ├─ pool "local" : 1-2 spawn procs + 1 local LLM server (MLX/llama.cpp)  (local-LLM arm)
    └─ probe        : 1 process, no sim, measures API latency distributions  (quiet-loop, Sect. 6)
```

Rules (derived from measurements and platform facts):

1. **`multiprocessing.get_context("spawn")` always.** macOS default is already `spawn` (3.14 changed the Linux default to `forkserver`, macOS/Windows unchanged; Python 3.14 "What's New", https://docs.python.org/3.14/whatsnew/3.14.html, accessed 2026-09-30). `fork` after importing PyBullet/Metal/Accelerate is unsafe. Spawn costs 0.07 s start + 0.47 s imports [V] once per worker, not per flight: workers are long-lived.
2. **One PyBullet physics client per process at a time in `cpu` pool** (`p.connect(p.DIRECT)`); never share a client across processes. [I from PyBullet model; agent 02 verifies multi-client-per-process and memory growth.] Reuse the env object across flights with `reset()` if agent 02 shows `reset` is cheaper than re-construct; recycle the worker every N flights (e.g. 500) as a leak guard (`max_tasks_per_child`-style, implemented by the supervisor respawning).
3. **Pool sizes on 4P+6E** [I]:
   - Classical sweep: `W_cpu = 9` (10 cores minus 1 for supervisor/SQLite/FastAPI). Going to 10 gives ~+0% (E-cores are already in use) and hurts tail latencies. Do not exceed 10 processes; oversubscription only adds context-switch and cache loss (and every extra process costs ~50-150 MB).
   - When API/local arms run concurrently with classical ones: `W_cpu = 6`, `W_api = 2`, `W_local = 1`, supervisor 1. Because macOS will place the hottest threads on P cores and spill the rest to E, a mixed pool gets automatically sorted by the scheduler; you cannot (and should not) pin.
   - Latency-sensitive real-time validation runs (Section 6) run **alone** (nothing else on the box), `W=1..2`.
4. **API pool = asyncio with K interleaved flights per process**, not one process per in-flight request. Reason: the API arm needs only ~0.17 core of CPU to drive 900 flights/h (Section 5), but each request waits 0.3-8 s. Per process: `K` flight coroutines each own a sim client; the sim step between awaits is a synchronous burst of ~3-10 ms. Rate-limit budget is split statically across processes (`rpm_per_proc = RPM_total / W_api`) with a token bucket + `Retry-After` honouring + jittered exponential backoff; this avoids any cross-process limiter. **Needs verification**: multiple DIRECT clients per process with gym-pybullet-drones (spike F3); fallback is `W_api = K` small processes (each 1 flight at a time, ~100 MB), which at K<=24 still fits in memory (~2.4 GB).
5. **Local LLM**: a single `llama.cpp`/`mlx-lm` server process (one GPU; decode is memory-bandwidth-bound, M4 base = 120 GB/s [I, Apple spec, unverified here]); clients call it over loopback HTTP, server batches concurrent requests. Keep **`W_local <= 2` sim processes**: more clients do not raise throughput, they only steal CPU that the server's Metal dispatch thread needs.
6. **Loopback HTTP keep-alive**, `httpx.AsyncClient` (or aiohttp, see 2.2) created once per process, `http2=False` for loopback, `http2=True` for real providers only if the provider benefits (agent 01 owns).

### Memory budget (16 GiB)

| Item | Estimate | Basis |
|---|---|---|
| macOS + desktop apps (browser, Claude, editor) | 4-6 GB | [A] |
| 9 sim workers x ~150 MB (python+numpy+pybullet+gym+torch-free) | ~1.4 GB | bare worker measured at 49 MB [V]; pybullet/gym adds ~50-100 MB [A] |
| SQLite + page cache + supervisor + uvicorn | 0.3 GB | [A] |
| Local LLM 1B Q4 / 3B Q4 + KV | 1 GB / 2.5 GB | [A] |
| Headroom for file cache, Parquet/NPZ writes | 2+ GB | |
| **Total** | **~9-11 GB** -> fits, but **no torch/SB3 in sim workers**, and do not run the RL arm's training in parallel with a 3B LLM + 9 workers | [I] |

Acceptance: `memory_pressure` stays green; swap used (`sysctl vm.swapusage`) < 1 GB at steady state.

---

## 4. Throughput model

### 4.1 Definitions

- `T_f` flight length in sim seconds [A: 20 s]; `f_c` control rate [A: 30 Hz] -> `N_tick = T_f * f_c = 600`; physics substeps are inside `t_tick`.
- `t_tick` CPU seconds per control tick (physics substeps + controller + obs + raycasts) [A: 1.0 ms; plausible range 0.5-3 ms; **agent 02's measurement replaces this**]; `t_reset` per flight [A: 0.10 s].
- `C_eff = 6.5` P-equivalents (Section 2.1).
- CPU per flight: `c = N_tick*(t_tick + t_ctrl_extra) + t_reset + D*t_req`, with `t_req` = client CPU per LLM request [A: 1.5 ms] and `t_ctrl_extra` = MPC solve time if any.
- **Calls per flight** under the async latency model (global Q1 options b/c/d): a new request is issued when the previous answer arrives, capped by the decision rate `f_d` [A: 10 Hz]:
  `D = T_f * min(f_d, 1 / L)` where `L` is the (real or sampled) decision latency. Faster model -> *more* calls per flight. This is the non-obvious budget driver.
- Bounds (flights/hour):
  - CPU: `F_cpu = 3600 * C_eff / c`
  - Rate limit: `F_api = 3600 * R_calls_per_s / D`   (R = RPM/60)
  - GPU (local): `F_gpu = 3600 * B / (D * t_call)` with `B` = effective batch speed-up [A: 3x], `t_call` single-stream call time.
  - **Expected `F = min(F_cpu, F_api, F_gpu)`**. (Also bounded by wall clock if flights are paced in real time, see 4.3.)
- In-flight requests needed (Little's law): `N_inflight = lambda_calls * L_real = (F/3600) * D * L_real`.
- Cost: `$/flight = D * (tok_in*p_in + tok_out*p_out)/1e6`.

### 4.2 Assumed prices and limits (all **[A]**, sources given, treat as order-of-magnitude)

| Item | Value used | Source / confidence |
|---|---|---|
| Fast provider 8B-class (Cerebras/Groq) | $0.10/M in, $0.10/M out | Cerebras Llama 3.1 8B $0.10/$0.10 (June 2026) per secondary source https://www.morphllm.com/cerebras-pricing ; Groq 8B $0.05/$0.08 historically, but the same search reports Llama 3.1 8B Instant moved to "contact sales" on 2026-08-26 (https://www.eesel.ai/blog/groq-pricing, https://www.cloudzero.com/blog/groq-pricing/). Accessed 2026-09-30. **L confidence**: secondary blogs, contradictory; agent 01 must verify from provider pages. |
| Fast provider rate limit | 1000 RPM / 250-300k TPM (Groq "developer" tier); free tier 30 RPM | https://tokenmix.ai/blog/groq-free-tier-limits-2026 , https://fast.io/resources/groq-rate-limit/ (secondary, L-M) |
| OpenRouter free models | 20 req/min; 50/day (<10 credits) or 1000/day (>=10 credits purchased); limits are global per account, extra keys do not help; negative balance blocks even free models; in-flight estimated cost is reserved against balance; Cloudflare may throttle abusive patterns; 429s carry `X-RateLimit-*`/`Retry-After` | **[V] official docs** https://openrouter.ai/docs/api-reference/limits (accessed 2026-09-30). Consequence: free tier = a smoke-test tool (1000 calls/day = 5-15 flights/day). |
| Slow LLM | Claude Haiku 4.5 $1/$5 per MTok; Sonnet 5.5 $2/$10; Batch API -50%; cache reads 0.1x | **[V]** https://platform.claude.com/docs/en/about-claude/pricing (accessed 2026-09-30). Note newer tokenizer ~+30% tokens (Claude >=4.7). Batch API is **not usable** for in-loop latency but is usable for cache pre-fill / offline labelling. |
| Slow-LLM rate limit | 1000 RPM assumed (provider tier dependent) | [A], L |
| Tokens per call | fast arm: 300 in / 3 out; slow arm: 500 in / 60 out (reasoning variant 500 / 400) | [A] (agent 07 owns prompt size) |
| Local LLM | 1B Q4 ~0.15 s/call (300-token prompt w/ prefix cache, 3 tokens out); 3B Q4 ~0.40 s/call; batching gives ~3x aggregate | [A], **L**: no local benchmark done here; agent 01 |

### 4.3 Flights-per-hour and budget table

`T_f=20 s, f_c=30 Hz, f_d=10 Hz, t_tick=1 ms, C_eff=6.5`. Numbers are from `model.py` (scratch).

| Arm | Calls/flight D | CPU-bound fl/h | API/GPU-bound fl/h | **Expected fl/h** | In-flight requests | $/flight | **1k flights: time, $** | **10k flights: time, $** |
|---|---|---|---|---|---|---|---|---|
| Classical PID + reactive | 0 | 33,300 | - | **~33k** | 0 | 0 | 2 min, $0 | 18 min, $0 |
| Classical MPC (5 ms solve/tick) | 0 | 6,300 | - | **~6.3k** | 0 | 0 | 10 min, $0 | 1.6 h, $0 |
| Local LLM 1B (0.15 s/call, batch x3) | 133 | 25,900 | 540 (GPU) | **~540** | 3 | 0 | 1.9 h, $0 | 18.5 h, $0 |
| Local LLM 3B (0.40 s/call, batch x3) | 50 | 30,100 | 540 (GPU) | **~540** (coincidence of D*t_call) | 3 | 0 | 1.9 h, $0 | 18.5 h, $0 |
| API-fast (L=0.3 s, 1000 RPM) | 67 | 29,200 | 900 (RPM) | **~900** | 5 | $0.0020 | 1.1 h, **$2** | 11 h, **$20** |
| API-slow Haiku 4.5 (L=2.5 s) | 8 | 32,800 | 7,500 (RPM) | **~7.5k** | 42 | $0.0064 | 8 min, **$6** | 1.3 h, **$64** |
| API-slow Sonnet 5.5 (L=2.5 s) | 8 | 32,800 | 7,500 | **~7.5k** | 42 | $0.0128 | 8 min, **$13** | 1.3 h, **$128** |
| API-slow reasoning, Sonnet 5.5, 400 out tok (L=8 s) | 2 | 33,100 | 24,000 | **~24k** | 133 | $0.0125 | 2.5 min, **$12.5** | 25 min, **$125** |

Reading guide:

- **Classical numbers are the least certain** (factor ~3 either way) because `t_tick` is assumed; replace with agent 02's measured steps/s. The shape is robust: classical flights are cheap, **LLM arms are 1-2 orders of magnitude slower per flight, and all of that slowness is external to the CPU.**
- **API-fast is the wall-clock bottleneck, not slow-API.** A fast model with L=0.3 s generates 67 calls/flight vs 8 for a 2.5 s model, so an RPM cap hurts it 8x more. At 1000 RPM: 900 flights/h. On the free OpenRouter tier (20 RPM, 1000/day): **~0.3 flights/min, 15/day** - useless beyond smoke tests. [I from V limits]
- **Dollar budgets are small** at these token sizes: the **whole 10k-flight campaign across all four arms is roughly $150-$300 of API money** (API-fast $20 + slow $64-128 + margin for retries/parse failures ~1.3x). The real costs are wall-clock time and rate-limit tier, not dollars. A 10x sensitivity (prompt 3k tokens with few-shot examples, 30 Hz decisions, no caching) moves this to ~$1-3k: prompt size and decision rate are the knobs.
- **Sub-sampling rule for the LLM arms** (global Q9): run classical arms on all N flights (cheap), LLM arms on a paired subset `n_llm` chosen by power analysis (agent 08), e.g. 1k flights per LLM arm rather than 10k. Paired design means classical results on the same 1k scenarios are reused for the paired test.
- **Local LLM caveats**: the GPU is shared with the display/compositor and the LLM competes with sim workers for memory bandwidth and thermals [I]. `t_call` and batching gain are placeholders: spike F5.
- **Real-time validation set** (global Q1 option d): flights paced at 1x wall clock take `T_f` = 20 s each; with K=16 interleaved flights per process x 2 processes, 1k flights = `1000*20/32` ≈ 10.4 min wall of pure waiting if API allowed it, but for the fast arm the RPM cap still binds (67 calls/flight * 1000 flights / 16.7 rps = 67 min). Budget ~1-2 h per 1k-flight real-time validation of a fast arm.

### 4.4 Sensitivity (what moves the answer)

| Parameter | Change | Effect |
|---|---|---|
| `t_tick` 1 ms -> 3 ms | x3 | classical PID 33k -> ~11k fl/h; LLM arms unchanged (they are not CPU-bound) |
| `f_d` 10 -> 5 Hz | halves D for fast arms | API-fast 900 -> ~1.8k fl/h, cost halves; but changes the experiment (decision rate is a factor, not a nuisance) |
| RPM 1000 -> 10,000 (higher tier) | x10 | API-fast then becomes CPU/concurrency bound at ~9k fl/h, in-flight ≈ 50 |
| `T_f` 20 -> 60 s | x3 D | fast arms 3x slower and costlier |
| Local LLM batching 3x -> 1x | 3x slower | 1B arm ~180 fl/h |
| Thermal throttling | -10-25% sustained on fanless-ish loads [I, L] | applies to every CPU-bound row; MacBook Pro has a fan, sustained all-core loads are OK but check with spike F6 |

---

## 5. API-arm bottleneck analysis (in-flight concurrency, pooling, rate limits)

- **Little's law**: `N_inflight = RPS * L`. At 1000 RPM (16.7 rps): L=0.3 s -> **5** in flight; L=2.5 s -> **42**; L=8 s -> **133**. A single asyncio process holding 64-128 keep-alive connections is enough; in my (contaminated) test the client scaled to C=512 without errors but with huge lag, so the limiting factor is CPU availability of the event loop, not connection count [V/I].
- **httpx pooling**: one `AsyncClient` per process, `Limits(max_connections=K*1.5, max_keepalive_connections=K*1.5)`, `Timeout(connect=2, read=per-arm deadline)`. Pool exhaustion is otherwise silent: requests queue inside httpx and **their waiting time shows up in your measured latency**. Always record `t_acquire` vs `t_request` separately or (simpler) assert `K <= max_connections`.
- **Backoff**: on 429/5xx honour `Retry-After`, else `min(cap, base*2^n) * U(0.5,1.5)`. **A retry is a latency event**: record `n_retries` and total elapsed; do not hide retries from the latency stats (agent 08's invalid-output / timeout definitions). Per-decision hard deadline (e.g. 2x the arm's p99) -> fallback action (global Q: HOVER/BRAKE by agent 11) and count as timeout.
- **OpenRouter specifics** [V from official limits doc]: limits are global to the account; in-flight budget is reserved against balance, so keep a positive balance >= `K * max_cost_per_call * safety`; free variants share upstream pools; 429 during streaming arrive as SSE events. Use `max_tokens` small and no streaming for enum outputs (agent 01).
- **Do not use more than one process per provider key** without splitting RPM statically; there is no cross-process limiter by design (KISS).

---

## 6. Time model, caching, and the latency-measurement caveat

- Execution of API flights is **decoupled from measured latency**: primary method is (d) hybrid (agent 08 confirms): sim runs at max speed, sampled or recorded latency becomes N control-tick delay. Real wall-clock latency is still *recorded* per call.
- **Measurement hygiene (evidence in 2.2)**: latency recorded inside a busy worker is polluted by event-loop lag (tens to hundreds of ms under load; here p99 lag 92 ms at C=16 even before tuning). Therefore:
  1. A **dedicated latency probe** process (no sim, `K<=8`, no other heavy load, same prompts) produces the latency distribution used for sampled-latency injection and for reporting. Record `perf_counter_ns` around `await client.post`, plus `loop_lag` ticker alongside.
  2. In worker processes, log latency too, but tag it `contaminated=loop_lag_p99>2ms`.
  3. Real-time validation flights run **alone** on the machine (`caffeinate -dimsu`, plugged in, no other pool running).
- **LLM response cache** (table `model_call_cache`, agent 10 owns schema): key = `sha256(provider, model, version/route, params, normalised prompt)`; value = response text, token counts, **original measured latency**, `recorded_at`. Uses: (a) resume after crash without re-paying, (b) re-run analysis/bugfix reruns free, (c) reproducibility under temperature 0 with provider non-determinism (replay returns the recorded answer). Expected hit rate within one campaign is low (states differ across seeds) unless the state is quantised; it is ~100% for exact re-runs. **Caveat**: a cache hit has ~0 real latency. The stored original latency must be what feeds the delay model (mode d), and runs must carry `cache_hit` flag; **never compute latency statistics or decisions/sec from hit rows**; never use the cache in real-time validation or latency-probe runs. Also exclude cached rows from "cost" totals but keep a `cost_if_uncached` column.
- **Cost guard**: supervisor keeps a running `$` ledger from token counts per call; hard stop at `BUDGET_USD` constant (scripts convention: constants at top); abort the pool (not crash) when exceeded; since jobs are resumable, raising the cap and restarting continues.

---

## 7. Resumable job queue design (SQLite)

### 7.1 Schema (queue part; fits agent 10's schema, `experiments`/`scenarios`/`arms`/`flights` are theirs)

```sql
PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=30000; PRAGMA foreign_keys=ON;

CREATE TABLE job (
  id            INTEGER PRIMARY KEY,
  experiment_id INTEGER NOT NULL REFERENCES experiments(id),
  scenario_id   TEXT    NOT NULL,
  arm_id        TEXT    NOT NULL,           -- includes config hash: name@sha8(config)
  seed          INTEGER NOT NULL,
  rep           INTEGER NOT NULL DEFAULT 0,
  pool          TEXT    NOT NULL,           -- 'cpu' | 'api' | 'local'
  mode          TEXT    NOT NULL,           -- 'fast' | 'realtime_validation'
  priority      INTEGER NOT NULL,           -- scenario block index: all arms of a scenario adjacent
  status        TEXT    NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
  worker_id     TEXT,
  lease_until   REAL,
  attempts      INTEGER NOT NULL DEFAULT 0,
  error         TEXT,
  created_at    REAL NOT NULL, started_at REAL, finished_at REAL,
  CONSTRAINT uq_job_run UNIQUE (experiment_id, scenario_id, arm_id, seed, rep, mode)
);
CREATE INDEX ix_job_pending ON job(pool, priority, id) WHERE status='pending';
CREATE INDEX ix_job_lease   ON job(pool, lease_until)  WHERE status='running';
```

(Naming follows the repo convention: plural table names in the ORM layer, `ix_/uq_/fk_` prefixes, `id` primary key. Table is `jobs` in the ORM; shown singular here for brevity.)

### 7.2 Protocol

```python
CLAIM_PENDING = """UPDATE job SET status='running', worker_id=:w, lease_until=:lease,
                   attempts=attempts+1, started_at=:now
                   WHERE id=(SELECT id FROM job WHERE status='pending' AND pool=:pool
                             ORDER BY priority, id LIMIT 1) RETURNING id, scenario_id, arm_id, seed, rep, mode"""
CLAIM_EXPIRED = """... WHERE id=(SELECT id FROM job WHERE status='running' AND pool=:pool
                   AND lease_until<:now ORDER BY lease_until LIMIT 1) RETURNING ..."""
```

1. **Claim**: one atomic statement (SQLite serialises writers; `UPDATE ... RETURNING` needs SQLite >= 3.35; have 3.50.4 [V]). Try pending first, then expired leases (two statements; see the `OR` trap in 2.3). Claim batches of `K` rows for API workers (one statement per row in a short `BEGIN IMMEDIATE`, or `LIMIT K` loop).
2. **Lease** = `max(60 s, 10 x p99 flight wall time of that pool)`; renew `lease_until` every ~10 s via a heartbeat coroutine/thread (real-time flights last 20+ s). Expired lease + `attempts < MAX_ATTEMPTS(3)` -> reclaimable; else mark `failed` with `error` (poison-pill isolation; do not retry forever).
3. **Idempotency**: the natural key is `(experiment_id, scenario_id, arm_id, seed, rep, mode)`; `arm_id` embeds a config hash so changing a prompt/model version creates *new* jobs rather than silently reusing old results. `flights` has the same `UNIQUE` key; write with `INSERT ... ON CONFLICT DO NOTHING` so a double execution (lease expired on a slow-but-alive worker) produces one row.
4. **Atomic completion**: worker writes large artifacts (trajectory NPZ/Parquet, decision log) to `data/tmp/<job>.part`, `os.replace` to final path, **then** in one transaction inserts `flights` + `decisions` batch and sets `job.status='done'` (guard `WHERE id=? AND worker_id=?` so a zombie cannot complete a reclaimed job). Startup sweep removes orphan `.part` files and orphan final files without a `flights` row.
5. **Determinism**: `seed` fully determines scenario event schedule (agent 06) and sim noise; latency sampling uses a separate RNG stream `seed_lat = hash(seed, arm_id)`; real API non-determinism is captured through the cache (Section 6). Reruns of a failed job produce the same scenario.
6. **Ordering for partial results**: `priority = scenario block`, so all arms of a scenario are adjacent and an interrupted campaign still yields **complete paired blocks** (McNemar/Wilcoxon-ready), and the supervisor can stop at any n.
7. **Supervisor** (`scripts/run_experiment.py`): constants at top (`ROOT_DIR` sys.path trick, per repo Scripts rule), logs start/end/elapsed, spawns pools, restarts dead workers (`exitcode != 0`), handles SIGINT by letting workers finish the current flight (graceful) or SIGTERM -> immediate; `kill -9` anywhere is safe by design (tested 2.3). On start: `UPDATE job SET status='pending' WHERE status='running' AND lease_until<now` is unnecessary (claim handles it) but a summary query prints resume state.
8. **Progress**: `SELECT pool, status, count(*) FROM job GROUP BY 1,2` (partial indexes keep it cheap); FastAPI exposes it (Section 9).
9. **Writers**: per-flight results are one transaction per flight (hundreds/s at most), far below SQLite limits; the queue claims are 13-18k/s [V]. No dedicated writer process needed unless agent 10's decisions-table spike shows otherwise.

---

## 8. macOS specifics

| Topic | Guidance | Status |
|---|---|---|
| Sleep | Run campaigns as `caffeinate -dimsu -w <supervisor_pid>` (or `caffeinate -i`), plugged in. Closing the lid sleeps the Mac unless external display + clamshell. Battery at 100% AC [V]. | standard macOS tool, M |
| Thermals | `pmset -g therm` reports none recorded [V]. Sustained all-core loads on a MacBook Pro with fan should hold, but check throughput decay over a 30-min burn (spike F6); `sudo powermetrics --samplers cpu_power,thermal` needs sudo (not run). | L |
| P/E cores | No pinning. `taskpolicy -c background` forces E-cores (measured E ≈ 0.58 P [V]); QoS `utility/user-initiated` are scheduled mostly to P first then E. Default processes started from a terminal run at `user-interactive/initiated` QoS. **Do not clamp sim workers to background** (my clamped run: wall rate 20-278 it/s vs 4,300 CPU-normalised, i.e. starved by any other load). Use `taskpolicy -b` only for the *analysis/compression* jobs you want out of the way. `nice` has limited effect on Apple Silicon. | [V for E/P ratio], Eclectic Light docs M |
| Workers count | 9 for CPU pool (Section 3); more does not help. Hyper-threading does not exist on Apple Silicon. | [I] |
| Python 3.14 free-threading | Officially supported in 3.14 (PEP 779), single-thread penalty ~5-10%; but it needs a separate `3.14t` build, and **native wheels (pybullet, numpy, gymnasium deps) must be cp314t-compatible**; pybullet is a C extension that re-enables the GIL unless it declares support [I, L]. Multi-process spawn gives the same parallelism without the risk. **Decision: do not use free-threading.** Source: https://docs.python.org/3.14/whatsnew/3.14.html (accessed 2026-09-30). | V for docs, L for wheel status |
| 3.14 JIT | "Experimental, not production-ready" in macOS binaries (same doc); PyBullet stepping runs in C++, JIT gains would only touch the Python glue. Not worth enabling. | V |
| Subinterpreters (`concurrent.interpreters`, `InterpreterPoolExecutor`) | New in 3.14 (PEP 734); C-extension support is uneven; pybullet likely not subinterpreter-safe. Not recommended. | V docs, I for pybullet |
| I/O | APFS fsync is cheap; `synchronous=NORMAL` WAL is safe against app crashes, loses last txn on OS crash (acceptable, jobs re-run). Put `data/` on the internal SSD, not iCloud/Dropbox-synced `~/Documents` (this repo lives under `~/Documents`: confirm no iCloud Desktop&Documents sync, SQLite files + WAL under sync cause corruption/conflicts) | I, M |
| Spotlight/Time Machine | Exclude the `data/` dir to avoid `mds` CPU spikes during millions of small writes | I |
| GUI | Headless DIRECT only; GUI/OpenGL workers would add a WindowServer connection per process and cost a core | I (agent 02) |

---

## 9. Runner CLI vs FastAPI-managed jobs - recommendation

**Runner CLI for execution; FastAPI is a control/monitor plane over the same SQLite file.**

| Criterion | Runner CLI (`scripts/run_experiment.py`) | Jobs inside FastAPI (BackgroundTasks / in-app pool) |
|---|---|---|
| Crash isolation | `kill -9`/OOM of a worker or the supervisor never touches the API | uvicorn reload/crash kills all jobs |
| Latency measurement | No unrelated event loop sharing the GIL; quiet | API event loop + request handling adds jitter to recorded latency (Section 2.2 evidence) |
| Process model | Clean `spawn` from a `__main__` guarded script | `spawn` from uvicorn workers re-imports the app in every child (DB engine, settings), `--reload` spawns a watcher; fragile |
| Long runs (hours) | Runs under `caffeinate`, survives terminal/HTTP disconnects with `nohup`/launchd | Tied to server lifetime |
| Repo conventions | Fits the "Scripts" rule (constants at top, ROOT_DIR sys.path, log start/end/elapsed) and layered services reused by both | Routes must not hold business logic; services would be shared anyway |
| Interactivity | Needs a separate way to watch progress | Native |

Design: `app/services/experiment_service.py` (class-based, DI) holds enqueue/progress/cancel logic; both the CLI and the API use it via repositories. FastAPI endpoints (plural-noun, no trailing id per repo URL rule): `POST /experiments` (create + enqueue jobs), `GET /experiments/{experiment_id}/progress`, `POST /experiments/{experiment_id}/pause` (sets a flag row the supervisor polls), `GET /experiments/{experiment_id}/events` (SSE tailing job counts). Starting the runner from the API, if wanted: `subprocess.Popen([...], start_new_session=True)` with a PID file, **never** a multiprocessing pool in the uvicorn process. SQLite access: sync `sqlite3` (or SQLAlchemy sync engine) in workers; `aiosqlite`/async SQLAlchemy in the API (agent 10). This keeps the template's async stack for the API without forcing it on hot paths.

---

## 10. Ray: for / against

**Against, for this project. Recommendation: do not adopt Ray.**

Against:
- Single 10-core laptop, one host: Ray's value (multi-node scheduling, autoscaling, object store, fault tolerance across nodes) is unused. `multiprocessing` + SQLite queue already gives resumability, measured at 13-18k ops/s.
- Platform support: Ray docs list Python 3.10-3.12 stable, 3.13 beta and do not mention 3.14; Apple Silicon is supported but "multi-node clusters are untested" on macOS (https://docs.ray.io/en/latest/ray-overview/installation.html, accessed 2026-09-30). `uv pip install --dry-run ray` does resolve `ray==2.58.0` on this 3.14.2 venv, so it may work, but I did **not** run it (confidence M/L). Adds a heavy dependency (14 packages in dry-run) to a template that is trying to shed asyncpg/psycopg2/boto3.
- Overheads: GCS + raylet + dashboard + object-store processes and memory on a 16 GB machine [I, not measured]; per-task scheduling latency of ~ms is fine for 1 s flights but the extra daemons compete for the 1 core we reserve.
- Measurement purity: Ray actors/worker processes add scheduler threads and IPC to the machine we are trying to characterise for latency; more variance to explain.
- Resumability is still ours: Ray does not give idempotent `(scenario, arm, seed)` semantics or a durable queue; you would build the SQLite table anyway.
- Hard to debug pybullet global state inside Ray workers; multiprocessing workers are plain processes with `py-spy`/`lldb`.

For (steelman):
- If the experiment ever moves to a multi-machine or cloud box, Ray (or plain `concurrent.futures` + the same SQLite/Postgres queue over a network FS, or `dask`) is the upgrade path. Keep `Runner` behind a small interface (`claim_next`, `complete`) so swapping is possible; this is YAGNI-compatible because the claim/complete functions are the seam.
- Ray Tune/RLlib for the RL arm (agent 05): SB3 already covers it without Ray.

Also rejected: `concurrent.futures.ProcessPoolExecutor` as the primary mechanism (no lease/resume, a crashed child poisons the pool -> `BrokenProcessPool`); fine inside a worker for sub-tasks only.

---

## 11. Micro-benchmarks to run first, with acceptance thresholds

Run on a **quiet** machine (check `sysctl vm.loadavg` < 2 before each; quit browsers/IDE builds), AC power.

| # | Benchmark | Acceptance / decision rule |
|---|---|---|
| B1 | 1 flight, 1 worker, classical PID, DIRECT, 600 ticks, 100 reps: `t_tick`, `t_reset`, RSS growth over 1000 resets | Fills model inputs; **RSS growth < 50 MB/1000 flights** else recycle every N flights; `t_tick` decides whether classical 10k is minutes or hours |
| B2 | Same on N=1..10 spawn workers, `taskpolicy` default, 60 s each | Aggregate flights/s at N=9 >= **0.75 * (4 + 5*0.58) * single-worker P rate** (~4.5-5x of N=1) ; per-worker p99 flight time <= 1.5x p50; if N=10 < N=9, use 9 |
| B3 | httpx vs aiohttp mock-server at K = 1, 16, 64, 128 (this doc's `httpx_bench.py`, quiet machine) | Loop lag p99 <= **2 ms at K=16 with no sim**, <= 10 ms with K=16 sims interleaved; if violated, move to 1 flight/process or use a probe process for latency stats; req/s >= 400 at K=64 for 100 ms server latency |
| B4 | Queue bench (this doc's `sqlq.py`) with real job durations + crash test | 0 duplicates, all jobs done, stuck `running` = 0 after kill; claim p99 < 50 ms with 10 workers doing real work |
| B5 | Local LLM: 300-token prompt, 3 tokens out, single stream vs batch 4/8 on MLX and llama.cpp, with 6 sim workers running | Report p50/p95 latency and calls/s; decides `W_local`, and whether local arm is 500 or 2,000 fl/h; if p95 > 2x p50 under sim load -> run local arm alone |
| B6 | Sustained 30-min all-core burn, log flights/min per 5-min window | Throughput decay < 15% -> no mitigation; otherwise run 8 workers |
| B7 | API latency probe against the real provider (needs key, later; agent 01 script) | p50/p95/p99, 429 rate at target RPM, failure rate < 0.5% |
| B8 | End-to-end dry run: 200 scenarios x 4 arms (mock provider with lognormal latency) with `kill -9` at random times | Final `flights` count exactly 800, byte-identical metrics vs an uninterrupted run |

---

## 12. Contradictions / notes relative to `docs/initial-plan.txt`

- The plan's "~20 kHz with 80 drones across 4 environments" is an **aggregate**, multi-drone, multi-env figure. For this project the relevant quantity is **1 drone, 1 process, DIRECT**; per-process throughput matters because parallelism here is across flights. Even if the aggregate claim were accurate, it does not transfer to "thousands of flights" as a headline; Section 4.3 shows classical flights scale with cores while LLM arms are bounded by GPU/RPM, so **simulation speed is not the limiting factor for the experiment the plan describes; API rate limits and GPU are**. (Verification of the 20 kHz claim itself belongs to agent 02; I did not verify the README.)
- "macOS 26 tested": this machine reports macOS 27.0 (build 26A428). Not verified by me.
- The plan frames "simulation_speed x decisions x environmental changes" as what to maximise; with LLM arms the maximisable quantity is `decisions per API-rate-limit-minute`, which is an entirely different bottleneck.

---

## Decisions recommended

1. **Runner CLI (`scripts/run_experiment.py`, `spawn` pools) executes; FastAPI only enqueues, reports progress (SSE) and pauses.** No pools in uvicorn.
2. **SQLite job table with leases, partial indexes, two-statement claim, unique `(experiment, scenario, arm_id, seed, rep, mode)` key, atomic artifact-then-row completion**, priority by scenario block so partial campaigns stay paired. Measured: 13-18k claims/s, 0 duplicates, crash recovery OK.
3. **Pool sizing**: 9 CPU workers for classical-only campaigns; mixed: 6 cpu + 2 api (asyncio, K~16 flights each) + 1 local; never > 10 processes; no pinning, no `taskpolicy -c background` for sim workers.
4. **Expected fl/h planning numbers** (to be replaced by B1/B2): classical ~33k (PID) / ~6k (MPC 5 ms); local LLM ~0.5k; API-fast ~0.9k at 1000 RPM; API-slow ~7.5k. **Run LLM arms on a paired subset (~1k flights), classical arms on all.**
5. **Budget**: API money is ~$20 (fast) and $64-$128 (slow, 10k flights) under stated assumptions; total campaign < ~$300 with margin; **wall-clock and rate-limit tier are the real constraints.** Build a `$` ledger and hard `BUDGET_USD` stop.
6. **Latency measurement**: dedicated quiet probe process for latency distributions; worker-recorded latencies flagged by loop lag; real-time validation runs alone under `caffeinate`.
7. **Response cache** yes, keyed by (provider, model, route/version, params, prompt), storing original latency; cache hits excluded from latency and decisions/sec statistics; never used in probe/validation runs.
8. **Ray: no.** Keep a thin `claim/complete` seam for future swap. **Free-threading/JIT/subinterpreters: no.**
9. Reserve ~1 P-eq for supervisor/SQLite/FastAPI; budget 16 GiB as ~9-11 GiB used; keep torch/SB3 out of sim workers.
10. Store `data/` on a non-synced local path (not iCloud Documents) and exclude from Spotlight.

## Open questions

1. True `t_tick` and `t_reset` for gym-pybullet-drones 1 drone DIRECT on this M4 (agent 02); can one process host multiple DIRECT clients / envs, and what RSS per client?
2. Decision rate `f_d` and flight length `T_f`: fixed by the experiment design (agent 08/11), because `D = T_f * min(f_d, 1/L)` drives cost and time of the fast arms by up to 10x.
3. What is Jev and what are its real RPM/pricing (agent 01)? If Jev is hardware-backed or a private API, rate limits could be much higher or lower than 1000 RPM; the 0.9k fl/h number is RPM-driven.
4. Do the secondary-source price/limit claims hold (Groq 8B moved to contact-sales 2026-08-26; Cerebras 8B $0.10/$0.10)? Only Anthropic pricing and OpenRouter limits were verified from primary docs.
5. Local LLM actual latency/batching on M4 base (120 GB/s) under concurrent sim load; MLX vs llama.cpp server batching behaviour.
6. Is `pybullet` 3.2.7 (resolved for 3.14.2) a native cp314 wheel or an sdist build (other agents were running ~30 `clang` processes, suggesting a source build)? This affects install friction and Python version decision (agent 02/14).
7. Is `~/Documents` synced by iCloud on this machine (SQLite WAL + sync hazard)?
8. Will a provider permit this traffic pattern (thousands of sequential 1-3 token calls) under its ToS / abuse heuristics (OpenRouter notes Cloudflare protection for "unreasonable usage")?

## Follow-up spikes (small, runnable experiments)

- **F1 (clean capacity + jitter):** re-run `scratchpad/09/httpx_bench.py` and `scale2.py` when `sysctl vm.loadavg` < 2; add aiohttp and uvloop client variants; add a variant where each of K coroutines also performs a 5 ms synchronous numpy burst between awaits to measure added latency error. (I tried waiting for a quiet host via `clean_driver.sh`; load never dropped below ~23, so the driver was stopped and no clean numbers exist.)
- **F2 (P/E scaling proper):** with the real gym-pybullet-drones flight loop (from agent 02), N=1..10 workers, 60 s each, report flights/s and per-worker p50/p99; fit `C_eff` and confirm the 0.58 E/P ratio on the actual workload (PyBullet is C++-heavy, the ratio may differ from scalar Python).
- **F3 (multi-client per process):** `p.connect(p.DIRECT)` x16 in one process, step them interleaved; measure RSS and ticks/s vs 16 processes; decides the API-pool layout.
- **F4 (queue under real load):** `sqlq.py` with 10 workers running real flights + 1 writer of decision rows (agent 10's spike) to confirm claim p99 < 50 ms.
- **F5 (local LLM):** single-stream vs batched throughput for 1B/3B Q4 on MLX and llama.cpp-server with 6 sim workers running.
- **F6 (thermal/soak):** 30-min 9-worker burn, log flights/min per window and `pmset -g therm`.
- **F7 (mock-provider chaos):** B8 with a mock provider injecting lognormal latency, 429s with `Retry-After`, timeouts, and malformed outputs; verify exact-once results and budget ledger.
- **F8 (free-threaded 3.14t):** `uv python install 3.14t`, test whether pybullet/numpy import and keep the GIL disabled; only to close the question, expected outcome "not worth it".
