# Design decisions

A running log. Each entry: the decision, why, what we gave up.

## D1. One clear job per framework

| Component | Job | Why this tool |
|---|---|---|
| **LlamaIndex** | Ingestion: load PDFs / web pages / KB markdown, semantic chunking (`SemanticSplitterNodeParser`), metadata extraction, write nodes to Chroma | Best-in-class loaders and node parsers |
| **Haystack 2.x** | Query-time retrieval pipeline: BM25 + embedding retrievers → `DocumentJoiner` (RRF) → cross-encoder ranker, with metadata filters | Explicit, inspectable pipeline graph of components |
| **ChromaDB** | Persistent vector store shared by both | Simple, local, metadata filtering |
| **Redis** | LLM + embedding cache, research-job status, rate limiting | Fast in-memory store with TTLs |
| **FastAPI** | HTTP API + SSE streaming of agent progress | Async, typed, auto docs |
| **Docker Compose** | api + redis + chroma | One-command setup |

The two frameworks never meet: LlamaIndex writes chunks, Haystack reads them, and the contract
between them is `mara/core/schema.py::Chunk`. Trade-off: two dependency trees to keep in sync.
Alternative rejected: doing everything in one framework. Both *can*, but LlamaIndex's retrieval
pipelines are less explicit than Haystack's, and Haystack's loaders/chunkers are thinner than
LlamaIndex's. Using each for its strength is the honest reason both are on the resume.

## D2. Plain-Python orchestration, no agent framework

The orchestrator (Phase 4) is an explicit `ResearchState` object passed through explicit steps.
No LangGraph / CrewAI / AutoGen. Why: it has to be whiteboard-able and every control-flow
decision (parallel research, the single critic loop, timeouts) should be visible in ~40 lines.
Trade-off: we hand-write things a framework gives for free (checkpointing, visual graphs).

## D3. `LLMProvider` as a Protocol + decorators (Phase 1)

```
CachedLLM( ResilientLLM( GeminiProvider | OpenAIProvider ) )
```

- **Strategy**: `GeminiProvider` and `OpenAIProvider` are interchangeable behind
  `LLMProvider` (`mara/llm/base.py`). Agents never import a vendor SDK.
- **Factory**: `mara/llm/factory.py::build_llm` picks the strategy from config and stacks the
  decorators, so wiring lives in one place.
- **Decorator**: `ResilientLLM` (rate limit + retry) and `CachedLLM` (Redis cache) each wrap
  any `LLMProvider` and are themselves `LLMProvider`s. Policy is written once, not per vendor.
- `Protocol` rather than an ABC: structural typing means `FakeLLM` in tests needs no import
  from production code, and `isinstance(x, LLMProvider)` still works (`runtime_checkable`).

Order matters: cache is outermost so a hit never consumes a rate-limit token or risks a retry.

Vendor SDK retries are disabled (`max_retries=0` for OpenAI; the genai client's default retry
options are left off) so retry policy is not applied twice.

## D4. Cache key design (Phase 1)

`mara:<version>:<namespace>:<provider>:<model>:<sha256(canonical JSON of inputs)>`

- Inputs hashed: prompt, system prompt, temperature, max_tokens, output JSON schema. Anything
  that can change the output must be in the key, or you get a wrong answer served as a hit.
- Canonical JSON (sorted keys, fixed separators) so kwarg order does not change the hash.
- Readable prefix, hashed suffix: keys stay short and fixed-length, but `SCAN mara:v1:llm:gemini:*`
  still works for inspection and targeted invalidation.
- `version` in the key: bumping `CACHE_KEY_VERSION` invalidates every entry without a `FLUSHALL`.
- Embeddings are cached **per text**, not per batch, so re-ingesting a document where one
  paragraph changed only embeds that paragraph. Vectors are stored as packed float32
  (4 bytes/dim vs ~20 for JSON).
- TTLs: LLM 7 days, embeddings 30 days (embeddings only change if the model changes, and the
  model is in the key). Redis runs `volatile-lru`, so under memory pressure only TTL'd cache
  keys are evicted, never job state.
- **Fail-open**: if Redis is down, reads are misses and writes are skipped. The system gets
  slower and pricier, not broken.

Trade-off: temperature > 0 outputs are cached too, so a "regenerate" gives the same answer
until TTL. For a research tool that is desirable (reproducible runs); a bypass flag can be
added if needed.

## D5. Rate limiting: token bucket in-process (Phase 1)

`AsyncRateLimiter` is a token bucket (burst = RPM, refill = RPM/60 per second) guarded by an
`asyncio.Lock`, so concurrent agents queue in order. Chosen over a fixed window because a
burst at the boundary of two windows can exceed the limit 2×. It is per-process; a
Redis-backed limiter (INCR + EXPIRE) is the upgrade if the API runs multiple replicas, and
Phase 4 adds a Redis limiter per client for the public endpoints.

Retry: exponential backoff `base·2^attempt`, capped, with jitter in [50%, 100%] so parallel
researchers that all hit a 429 do not retry in lockstep. Only `RetryableLLMError` (429, 5xx,
timeouts) is retried; 4xx client errors fail fast.

## D6. `/health` returns 200 with `status: degraded`

A liveness probe that fails when Redis blips would make the orchestrator restart a perfectly
good API process. So `/health` is always 200 while the process is up, and reports each
dependency separately. A readiness probe that requires `status == ok` can be added for k8s.

## D7. Pin everything

Exact versions in `pyproject.toml`, `uv.lock` committed, Docker image tags pinned
(`redis:8.8.3-alpine`, `chromadb/chroma:1.5.9` matching the `chromadb` client, `uv:0.12.3`).
LlamaIndex and Haystack change APIs between minors; a green build today should be green in a
month. Versions were checked against PyPI / Docker Hub on 2026-09-30.
