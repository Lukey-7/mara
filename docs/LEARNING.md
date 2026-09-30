# Learning log

One section per phase: what was built, how it works, why, trade-offs, interview Q&A.

---

## Phase 1 — Skeleton

### What exists now

```
mara/core/config.py     Settings (pydantic-settings): every tunable, read from .env
mara/core/schema.py     Chunk: the one schema LlamaIndex writes and Haystack reads
mara/core/cache.py      make_cache_key() + RedisCache (fail-open get/set, mget, pipeline)
mara/llm/base.py        LLMProvider Protocol, LLMResponse, LLMError / RetryableLLMError
mara/llm/gemini.py      GeminiProvider (google-genai async client, JSON-schema output)
mara/llm/openai_provider.py  OpenAIProvider (openai async client, response_format json_schema)
mara/llm/resilience.py  AsyncRateLimiter (token bucket), retry_async (backoff), ResilientLLM
mara/llm/cached.py      CachedLLM: Redis cache for generate() and per-text for embed()
mara/llm/factory.py     build_llm(): Cached(Resilient(Gemini|OpenAI))
mara/api/main.py        create_app(), lifespan wiring, GET /health
tests/                  29 tests, no API key, fakeredis instead of Redis
Dockerfile, docker-compose.yml (api + redis + chroma), Makefile, CI
```

### How a call flows

```
agent.generate(prompt, json_schema=Plan)
  └─ CachedLLM.generate
       key = mara:v1:llm:gemini:gemini-2.5-flash:sha256({prompt, system, temperature, ...})
       hit?  → return LLMResponse(cached=True)                      (~1 ms)
       miss → ResilientLLM.generate
                └─ limiter.acquire()          wait for a token bucket token
                └─ GeminiProvider.generate    HTTP call; 429/5xx → RetryableLLMError
                └─ on RetryableLLMError: sleep base·2^n (capped, jittered), retry ≤ 4×
              cache.set(key, response, ttl=7d)
```

### Why it is built this way

- **Protocol + decorators** so cross-cutting policy (cache, retry, rate limit) is written once
  and every vendor gets it. Adding Anthropic = one new file implementing two methods.
- **Cache key = hash of everything that affects output.** Miss one input (say, the JSON
  schema) and two different requests collide and you serve a wrong cached answer. That is the
  single most important property of a cache key.
- **Fail-open cache**: Redis outage → slower, not down.
- **Token bucket, not fixed window**: fixed windows allow 2× the limit across a boundary.
- **Vendor retries off** so we do not get retry-inside-retry (4 × 4 = 16 attempts).
- **Errors translated at the boundary**: providers turn SDK exceptions into
  `RetryableLLMError` / `LLMError`, so the retry loop knows nothing about vendors.

### Trade-offs to own in an interview

- In-process rate limiter: correct for one API replica; multiple replicas need Redis
  (`INCR` + `EXPIRE`, or a Lua token bucket).
- Caching temperature>0 outputs means "regenerate" returns the same answer until TTL.
- Cache stores the full response JSON; fine for text, would want compression for big outputs.
- `/health` pings Chroma with a 1 s timeout on every call; cache the result if it were polled
  hard.

### 5 likely interview questions

**Q1. What goes into your LLM cache key and why hash it?**
Provider, model, prompt, system prompt, temperature, max_tokens, and the output JSON schema
(as canonical JSON, sorted keys). Every one of those changes the output, so every one must be
in the key or two different requests would share an entry. Hashing gives a fixed-length key
(prompts can be 10 KB) and avoids Redis key-size issues; the readable prefix
(`mara:v1:llm:gemini:model:`) keeps `SCAN` and bulk invalidation practical. A `version`
segment lets me invalidate everything by changing one config value.

**Q2. What happens if Redis goes down mid-run?**
`RedisCache` catches `RedisError` on both reads and writes: reads become misses, writes are
skipped, a warning is logged. The run continues, just slower and with more API calls. The
`/health` endpoint reports `redis: error` with `status: degraded` but still returns 200, so an
orchestrator does not kill the API process over a cache outage.

**Q3. Why a token bucket instead of just sleeping 6 seconds between calls at 10 RPM?**
Sleeping serializes everything: five parallel researcher calls would take 30 s even if the
API would happily accept a burst. A token bucket allows a burst up to capacity, then throttles
to the refill rate, which matches how providers actually enforce limits (per-minute windows).
The `asyncio.Lock` makes waiters queue fairly instead of thundering-herd on refill.

**Q4. Why exponential backoff with jitter, and why cap it?**
Exponential: a 429 means the server is overloaded, so back off progressively instead of
hammering. Jitter: if five parallel calls all fail at once and all retry after exactly 2 s,
they collide again ("retry storm"); randomizing in [50%, 100%] spreads them. Cap: without it
attempt 6 waits 64 s, which is longer than the user's patience; 30 s is the ceiling and the
retry count (4) bounds total wait.

**Q5. Why a `Protocol` rather than an abstract base class?**
Structural typing: any object with `generate` and `embed` of the right shape *is* an
`LLMProvider`, no inheritance needed. Tests use a `FakeLLM` that never imports production
classes; decorators like `CachedLLM` wrap and re-expose the interface without subclassing.
`@runtime_checkable` keeps `isinstance` working for the factory tests. An ABC would work too;
Protocol is the more Pythonic fit for "interface as a contract, not a hierarchy".

### 3 questions for you to answer before "next"

1. Two calls differ only in `temperature` (0.0 vs 0.7). Do they share a cache entry? Should
   they? What would happen if they did?
2. The rate limiter is set to 10 RPM with burst 10. At t=0 you fire 15 concurrent requests.
   Sketch when each one is sent. Now the API is restarted at t=30 s: what is the bucket state
   and is that a problem?
3. `ResilientLLM` wraps the provider; `CachedLLM` wraps `ResilientLLM`. Swap them. What
   changes about (a) cost on a cache hit, (b) what gets cached when a call fails after
   retries?
