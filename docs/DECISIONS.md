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

## D8. Chroma access goes through our own `ChunkStore`, not LlamaIndex's `ChromaVectorStore` (Phase 2)

The plan assumed `llama-index-vector-stores-chroma` would write chunks that Haystack reads.
Inspecting `ChromaVectorStore.add` (v0.6.0) showed it serialises the *entire node* as a JSON
string in a `_node_content` metadata field (duplicating the text), uses `collection.add`
(not upsert, so re-ingestion raises on existing ids), and adds `_node_type`, `document_id`,
`ref_doc_id` fields that Haystack would surface as junk metadata. Both frameworks *can* read
it, but neither cleanly.

Decision: one 150-line module, `mara/core/chunk_store.py`, owns the collection. It writes the
plain layout `ids = chunk_id, documents = text, metadatas = Chunk.to_chroma_metadata()` and
records the embedding model + dimensions in the collection metadata, refusing to open a
collection built with a different embedding (`EmbeddingMismatchError`). LlamaIndex's job is
loaders + node parsing (`PDFReader`, `SemanticSplitterNodeParser`, `SentenceSplitter`);
Haystack (Phase 3) reads the same plain layout. The integration package was removed from
`pyproject.toml` because we do not use it. Resume wording stays true: LlamaIndex does
ingestion; ChromaDB is the shared store.

## D9. Citable unit = PDF page / markdown section / web page (Phase 2)

Chunking runs per *source unit*, never across them. A chunk therefore always has an exact
page number or section heading, which the Writer needs for citations like "[3] Raft notes,
Leader election". Cost: a sentence that straddles a page break is split. Accepted.

## D10. Idempotent ingestion via content hashes (Phase 2)

`doc_id = sha256(source_type + whitespace-normalised text of all units)[:16]`. Same bytes under
a new filename ⇒ same id ⇒ `skipped_duplicate` (no embedding calls). `force=True` deletes
the old chunks and re-ingests (`replaced`). `chunk_id = doc_id + "-" + sha256(doc_id, index,
text)[:16]`, so identical splits produce identical ids and Chroma `upsert` is a no-op.
Documents are *derived* from chunk metadata (`GET /documents` groups by `doc_id`) rather than
kept in a second table that could drift. Fine to ~100k chunks; beyond that, a documents table.

## D11. Embeddings: task types and 768 dims (Phase 2)

`LLMProvider.embed(texts, kind="document" | "query")`. Gemini's embedding model is asymmetric
(`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` task types); OpenAI's is symmetric and ignores
`kind`. Both support Matryoshka truncation, so `EMBEDDING_DIMENSIONS=768` cuts storage and
search cost 4× versus Gemini's native 3072 for a small quality loss. The dims and kind are part
of the cache key. Embedding calls get their own rate limiter (`EMBEDDING_REQUESTS_PER_MINUTE`)
because providers limit them separately from generation.

## D12. Metadata encoding for filters (Phase 2)

Chroma metadata is flat scalars or lists. `tags` is stored as a list (Chroma ≥1.1 supports
`{"tags": {"$contains": "raft"}}`); `published_date` as int `YYYYMMDD` so `$gte/$lte` range
filters work; `None` fields are omitted. `mara/core/filters.py::build_where` is the only code
that knows the grammar (single condition bare, several wrapped in `$and`, tags any-of via
`$or`). The same `MetadataFilter` model is what the Planner emits in Phase 4.

## D13. Semantic chunks are capped (Phase 2)

`SemanticSplitterNodeParser` cuts where the cosine distance between neighbouring sentence
windows exceeds the Nth percentile *within the document*. A single-topic page therefore yields
one giant chunk. Giant chunks hurt retrieval precision and overflow the reranker's 512-token
window, so anything over `MAX_CHUNK_CHARS` is re-packed sentence by sentence
(`pack_sentences`). The percentile default is 90 (LlamaIndex's default 95 produced too few
splits on short notes).

## D14. Query pipeline = Haystack components, one small graph per configuration (Phase 3)

`mara/retrieval/hybrid.py` wires `InMemoryBM25Retriever`, a custom `ProviderQueryEmbedder`
(so query embeddings go through our cache/rate-limit stack), chroma-haystack's
`ChromaEmbeddingRetriever`, `DocumentJoiner(join_mode="reciprocal_rank_fusion")` and the
`SentenceTransformersSimilarityRanker` (cross-encoder `ms-marco-MiniLM-L-6-v2`, from the
`sentence-transformers-haystack` package: Haystack 3 moved it out of core). Each
(mode, rerank) combination is its own `Pipeline` built lazily and cached: "BM25 only" is a
2-node graph, "hybrid + rerank" a 5-node graph. Why not one graph with stages switched off:
the eval must report the latency of each configuration honestly, and a graph that always
runs both retrievers cannot do that. Cost: the cross-encoder is instantiated once (only the
rerank graph has it); retriever components are cheap wrappers over shared stores.

## D15. BM25 copy lives in Haystack's in-memory store, rebuilt from Chroma (Phase 3)

Chroma is the source of truth. `BM25Index` (InMemoryDocumentStore, BM25L) is filled from
`ChunkStore.all_chunks()` at startup and kept in sync by the ingestion pipeline's dual write
(`IngestionPipeline(indexes=[bm25])`, delete propagates too). The two writes are not atomic;
a crash between them leaves the BM25 copy behind until the next restart heals it. Chosen over
Elasticsearch/OpenSearch because the goal is one process on a laptop; the upgrade path is
swapping `BM25Index` for an OpenSearch document store behind the same two methods.

BM25L rather than textbook BM25Okapi: Okapi's IDF `log((N-n+0.5)/(n+0.5))` is ≤ 0 for any
term in at least half the documents, which on a small or filtered corpus zeroes out perfectly
good matches (found by a test: every BM25 score was 0.0 on a 4-document corpus).

## D16. Local embeddings by default; API embeddings optional (Phase 3)

`EMBEDDING_PROVIDER=local` runs `BAAI/bge-small-en-v1.5` (33M params, 384 dims) on the CPU
via sentence-transformers. Ingestion and search then need no API key and no rate limiting,
`make demo` works offline after the first model download, and the retrieval eval is
reproducible. Gemini / OpenAI embeddings remain one config switch away (with task types and
Matryoshka truncation, D11). The `CompositeProvider` pairs any generator with any embedder;
with no LLM key configured it still embeds, so only `/research` (Phase 4) needs a key.
Rate limiting is skipped for local embeddings (`embedding_limiter=None`).

## D17. Filters written once, evaluated by both retrievers (Phase 3)

`MetadataFilter.to_haystack()` produces Haystack's filter dict; chroma-haystack translates it
to Chroma's `where`, and the in-memory store evaluates it in Python, so one filter reaches
both legs of the hybrid search. Tags became boolean flags (`tag:raft = True`) next to the
`tags` list because Haystack's in-memory filters have no list-membership operator; equality
on a flag works in both stores. Chroma also rejects empty lists as metadata values, so `tags`
is omitted when a chunk has none.

## D18. Embedded Chroma option (Phase 3)

`CHROMA_PERSIST_PATH=./data/chroma` uses an on-disk Chroma inside the API process instead of
the server container: no Docker needed for local development, tests and evals. chroma-haystack
only implements `run_async` for remote clients, so `ChromaDenseRetriever` wraps it and runs
the sync method in a thread when embedded.

## D19. Orchestration is one plain-Python function over an explicit state object (Phase 4)

`mara/agents/orchestrator.py::Orchestrator.run` is ~30 lines: Planner → [Researcher →
Summarizer → Critic] × (1 + ≤ `max_loops`) → Writer. `ResearchState` (Pydantic) carries
everything: plan, evidence per sub-question, verified notes, critique, sources, answer,
warnings and the trace. Each agent is `run(state, step) -> state`. No LangGraph / CrewAI /
AutoGen: the control flow must be whiteboard-able and every decision (parallel research,
the single loop, timeouts, degradation) visible in one place. What we give up: built-in
checkpointing, graph visualisation and human-in-the-loop hooks; the trace + job store cover
the first, the SSE stream the last.

## D20. Structured output: provider JSON mode + Pydantic validation + one retry (Phase 4)

Every LLM-facing agent has a Pydantic output model. The provider is asked for JSON matching
its schema (Gemini `response_json_schema`, OpenAI `response_format`), the reply is validated
with `model_validate_json`, and on failure the call is repeated once with the validation
error quoted back. Nested `$ref`s are inlined (`mara/llm/schema_utils.py`) because provider
support for references is uneven. LLM-facing models use only simple JSON types; dates are
strings converted in code.

## D21. Evidence must be quotable, and code checks it (Phase 4)

The Summarizer's notes are `{claim, supporting_quote, chunk_id}`; `verify_quote` requires
the quote to be a substring of the chunk (verbatim modulo whitespace, case-sensitive) and
drops the note otherwise. The Writer only ever sees verified notes, and its citation numbers
are assigned by code from those notes before the prompt is built; after generation, citations
to unknown numbers are stripped and `citation_coverage` (share of factual sentences with a
citation) is recorded. So a hallucinated fact cannot acquire a citation: it would need a
verbatim quote that does not exist. Cost: the model sometimes fails to copy a quote exactly
and a true claim is dropped; the drop count is in the trace so this is measurable.

## D22. The Critic can trigger at most one extra loop, and code decides coverage (Phase 4)

The Critic proposes up to 3 new sub-questions; the orchestrator loops only while
`state.loop < max_loops` (default 1). Coverage is computed in code (a sub-question is covered
iff it has verified notes); the model's job is judging *quality* of coverage, conflicts, and
proposing better queries. A hard cap rather than "loop until satisfied" bounds cost and
latency and makes the run reproducible.

## D23. Graceful degradation per step (Phase 4)

Each agent runs under `asyncio.wait_for(timeout)`. Planner failure → the question becomes the
single sub-question; a sub-question's retrieval failure → empty evidence + warning; web search
failure → warning, internal corpus only; Summarizer/Critic failure → the run continues with
what it has; only Writer failure marks the job failed. Warnings are handed to the Writer so
the answer states its limitations.

## D24. Jobs: Redis for live state, JSON files for history (Phase 4)

`POST /research` returns a job id at once; the run is an `asyncio.Task` in the API process.
State and an append-only event list live in Redis (`RedisJobStore`, TTL) or in memory when
Redis is down (`MemoryJobStore`), and each finished run is archived as
`data/traces/<job_id>.json`. `GET /research/{id}/events` is server-sent events implemented by
polling the event list, so it works with either store and resumes from `Last-Event-ID`.
Trade-off: in-process tasks die with the process; a queue (Celery/RQ/Arq) is the upgrade when
runs must survive restarts or scale across workers.

## D25. Web search: DuckDuckGo by default-available, Tavily by key, no-op by default (Phase 4)

`WebSearchProvider` has three implementations: `NoopWebSearch` (default: the internal corpus
only), `DuckDuckGoWebSearch` (free, no key, via `ddgs`; rate-limited) and `TavilyWebSearch`
(API key). The Researcher fetches each result, cleans it with trafilatura and ingests it
through the normal pipeline (tagged `web-search`), so web evidence is chunked, embedded and
cited exactly like everything else.

## D26. UI: one HTML file, server-sent events (Phase 6)

`mara/static/index.html` is the entire front end: inline CSS and JS, no framework, no build
step, served by `mara/api/ui.py`. Progress uses `EventSource` on
`GET /research/{id}/events`; the final state comes from `GET /research/{id}`. SSE over
WebSockets because the channel is one-way and SSE reconnects and resumes by itself. Model
output is HTML-escaped before the 15-line markdown renderer touches it. Trade-off: no
components or type checking, acceptable at this size.

## D27. Two cache shapes: a decorator object and a decorator function (Phase 6)

`CachedLLM` (Phase 1) is the Decorator *pattern*: an object wrapping an `LLMProvider`, with
per-text keys for embeddings and hit counters for the trace. `redis_cached` in
`mara/core/cache.py` is a Python *function decorator* for any async function with
JSON-serialisable results; `CachedWebSearch` uses it so a repeated query does not hit the
rate-limited search API again. Both build keys with `make_cache_key` and both fail open.

## D28. Per-client API rate limiting: fixed window in Redis (Phase 6)

`mara/api/rate_limit.py`: `INCR` a key per (client IP, minute), `EXPIRE` on first use,
429 with `Retry-After` above the limit, applied to POST `/research` and `/search` only.
Chosen over the in-process token bucket used for outbound LLM calls because the counter must
be shared by every API replica. Known weaknesses: up to 2x the limit across a window
boundary, identity by IP, fail-open when Redis is down.

## D29. `Reranker` Protocol (Phase 6)

The pipeline's last stage is typed as `Reranker` (`mara/retrieval/components.py`): anything
with `run(query, documents, top_k)`. Haystack's cross-encoder ranker and our
`PassthroughRanker` both satisfy it structurally, so swapping rerankers is a factory change.

## D30. Install mode and formatter scope, learned from CI (Phase 6)

`[tool.uv] link-mode = "copy"`: on Linux uv hard-links packages from its cache, NLTK 3.10
refuses to open multiply-linked data files, and LlamaIndex ships its tokenizer data inside
the wheel, so semantic chunking failed on the CI runner only. Also: `ruff format --check`
covers Python code blocks in Markdown, so docs must go through the formatter too. CI was red
from Phase 2 to Phase 6 because it was not checked after each push; it is now part of the
definition of done for every commit.

## D31. Local-first: no Docker required (Phase 6)

The supported way to run MARA is local: `make run-local` uses an embedded on-disk Chroma
(`CHROMA_PERSIST_PATH`), local sentence-transformers models, and the in-memory job store when
Redis is absent. `Dockerfile` and `docker-compose.yml` are kept as optional deliverables but
have never been built or run; the README and `docs/INTERVIEW.md` say so.
