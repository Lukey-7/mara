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

---

## Phase 2 — Ingestion (LlamaIndex)

### What exists now

```
mara/core/schema.py        Chunk.to_chroma_metadata() / from_chroma(): the ONE storage layout
mara/core/filters.py       MetadataFilter + build_where(): the only code that knows Chroma's grammar
mara/core/chunk_store.py   ChunkStore: upsert / query / list_documents / delete, embedding guard
mara/ingest/loaders.py     load_pdf (per page), load_html (trafilatura), load_markdown (per section)
mara/ingest/chunking.py    SemanticChunker (LlamaIndex SemanticSplitterNodeParser) + FixedChunker
mara/ingest/embeddings.py  ProviderEmbedding: LlamaIndex BaseEmbedding -> our LLMProvider (cached)
mara/ingest/pipeline.py    IngestionPipeline.ingest(): hash -> skip | chunk -> embed -> upsert
mara/ingest/web.py         fetch_html with timeout + size cap
mara/api/ingest_routes.py  POST /ingest/pdf | /ingest/url | /ingest/kb, GET /documents, DELETE
knowledge_base/*.md        10 notes on distributed databases (the "internal KB")
sample_corpus/             3 Wikipedia PDFs (CC BY-SA) + 5 URLs; scripts/ingest_sample_corpus.py
scripts/compare_chunking.py  semantic vs fixed side-by-side on any file
```

### How one PDF flows

```
POST /ingest/pdf (multipart)
  └─ load_pdf(bytes)                 LlamaIndex PDFReader -> one SourceDocument per page (page kept)
  └─ IngestionPipeline.ingest(pages)
       doc_id = sha256(normalised text of all pages)[:16]
       exists?  -> "skipped_duplicate" (0 API calls)          <- idempotent
       └─ SemanticChunker.chunk(doc_id, pages)               per page:
            SemanticSplitterNodeParser.aget_nodes_from_documents
              1. sentence-split (NLTK punkt, bundled with llama-index-core)
              2. embed every sentence window (buffer_size=1 -> 3 sentences) via ProviderEmbedding
                 -> CachedLLM -> ResilientLLM(embedding limiter) -> Gemini RETRIEVAL_DOCUMENT
              3. cosine distance between consecutive windows
              4. cut where distance > 90th percentile
            pack_sentences(): re-split anything > 2000 chars
            -> Chunk(chunk_id, doc_id, text, page, title, tags, published_date, ...)
       └─ llm.embed([chunk texts], kind="document")          one vector per chunk (cached per text)
       └─ ChunkStore.upsert(chunks, vectors)                 Chroma: ids / documents / metadatas
```

### Why it is built this way

- **Semantic chunking answers "where does one idea end?"** Fixed-size windows cut mid-thought
  and pad chunks with neighbouring text; retrieval then matches a chunk on words that belong
  to a different idea. Splitting at embedding-distance spikes keeps each chunk about one thing,
  so its vector represents it faithfully and a quote pulled from it is coherent.
- **Same embedding stack for chunking and storage.** The sentence embeddings used to *find*
  boundaries go through the same cache / rate-limit / retry decorators as everything else, via
  a tiny LlamaIndex `BaseEmbedding` adapter. Re-running ingestion on an unchanged file costs
  zero API calls: the doc hash short-circuits first, and even a `force` re-run hits the
  embedding cache.
- **Own the storage layout.** See D8: LlamaIndex's Chroma integration stores its private node
  JSON in metadata. A 150-line `ChunkStore` gives both frameworks a clean, documented layout
  and lets us refuse to mix embedding models.
- **Citable units.** Chunk per page / section (D9) so every chunk can be cited precisely.
- **Filters designed for the Planner.** `MetadataFilter` (source types, tags, doc ids, date
  range) is the same object the Planner will emit in Phase 4; `build_where` is 20 lines and
  tested against Chroma's real grammar.

### Trade-offs to own in an interview

- Semantic chunking costs one embedding call per sentence window: roughly 3x the embedding
  volume of fixed chunking. Cached, so paid once; still the wrong choice for a corpus that
  changes constantly.
- The percentile threshold is *relative to the document*: a one-topic page still gets a cut at
  its single largest distance if that is above the 90th percentile. That is why the size cap
  and fallback exist; an absolute threshold would need calibration per embedding model.
- Per-page chunking splits sentences at page breaks and cannot merge a heading on one page
  with its paragraph on the next.
- `GET /documents` scans all chunk metadata; fine to ~100k chunks, then it needs a documents
  table or a materialised view.
- Ingestion runs inside the request (a 20-page PDF is tens of seconds on a free-tier key).
  Phase 4's job runner will move it to the background.
- Wikipedia PDFs carry the *render* date as CreationDate, not the article date; the API
  accepts an explicit `published_date` override for that reason.

### 5 likely interview questions

**Q1. How does semantic chunking actually decide where to cut?**
Split the text into sentences. For each sentence, build a window of it plus `buffer_size`
neighbours on each side and embed the window. Compute cosine distance between consecutive
window embeddings. Take the Nth percentile (we use 90) of all those distances in the document
as the threshold; every position whose distance exceeds it becomes a boundary. Sentences
between boundaries form one chunk. Intuition: neighbouring sentences on the same topic have
similar embeddings; a jump in distance is a topic shift. The buffer smooths out single odd
sentences.

**Q2. Why not just use fixed-size chunks with overlap? What did you measure?**
Fixed windows are cheaper (no embedding calls to chunk) and predictable in size, but they cut
mid-idea and blend neighbouring topics, which hurts precision and makes quotes incoherent.
Overlap patches the boundary problem at the cost of duplicated text in the index. I kept both
strategies behind one `Chunker` Protocol and a config switch, plus `scripts/compare_chunking.py`
to show boundaries side by side, and Phase 3's retrieval eval reports Recall@5 / MRR for each,
so the choice is data-driven rather than a slogan.

**Q3. How do you make ingestion idempotent?**
Content-addressed ids. `doc_id` is a hash of the normalised text of every unit, so the same
PDF uploaded twice, under any filename, is recognised and skipped before any API call.
`chunk_id` hashes `doc_id + index + text`, so a forced re-ingest that produces the same split
upserts in place rather than duplicating. Embeddings are cached per text, so even a forced
re-run of an unchanged document costs nothing. Only the `ingested_at` metadata changes.

**Q4. What is stored in Chroma for one chunk, and why that shape?**
`id = chunk_id`, `document = text`, `embedding = 768-d vector`, `metadata = {doc_id,
source_type, title, url_or_path, page?, section?, published_date as int YYYYMMDD, tags as
list, ingested_at iso}`. Flat scalars / lists only, because that is what Chroma filters on.
Dates are ints so range filters work; tags a list so `$contains` works; `None` is omitted. The
collection metadata records the embedding model + dimensions, and the store refuses to open a
collection built with a different one: mixing embedding spaces silently returns garbage.

**Q5. Where does trafilatura fit and why not BeautifulSoup?**
Web pages are mostly boilerplate: navigation, cookie banners, footers, related links. Indexing
that pollutes retrieval and citations. trafilatura is a boilerplate-removal library that
extracts the main article text plus metadata (title, date) using heuristics tuned on large
corpora; BeautifulSoup gives you a DOM and leaves the "which part is the article" problem to
you. `favor_precision=True` prefers dropping borderline content over keeping junk.

### 3 questions for you to answer before "next"

1. You ingest a 30-page PDF with `breakpoint_percentile=90`. Roughly how many chunks per page
   do you expect, and why does the answer barely depend on the page's content?
2. Two KB notes share a paragraph verbatim. Do they get the same `doc_id`? The same
   `chunk_id` for that paragraph? What happens on `upsert`?
3. A teammate changes `EMBEDDING_DIMENSIONS` from 768 to 1536 and restarts the API. What does
   `/health` show, and what is the fix? Why is refusing better than continuing?

---

## Phase 3 — Hybrid retrieval (Haystack)

### What exists now

```
mara/retrieval/hybrid.py      HaystackHybridRetriever: one Haystack Pipeline per (mode, rerank)
mara/retrieval/components.py  ProviderQueryEmbedder, ChromaDenseRetriever, PassthroughRanker
mara/retrieval/bm25_index.py  BM25Index: Haystack InMemoryDocumentStore (BM25L), synced with Chroma
mara/retrieval/rrf.py         our own 10-line reciprocal_rank_fusion (tested against Haystack's)
mara/retrieval/factory.py     build_retriever(): chroma-haystack store + cross-encoder ranker
mara/core/filters.py          MetadataFilter.to_haystack(): one filter for both retrievers
mara/llm/local_embeddings.py  LocalEmbeddingProvider (bge-small-en-v1.5, 384 dims, CPU)
mara/llm/composite.py         CompositeProvider: generator (Gemini/OpenAI/None) + embedder
mara/api/search_routes.py     POST /search
eval/retrieval_eval.json      35 questions → relevant sections / documents
eval/run_retrieval_eval.py    Recall@5, MRR@10, latency for 4 configurations
```

### The pipeline

```
POST /search {"query": "...", "filters": {...}, "config": {"mode": "hybrid", "rerank": true}}
  └─ HaystackHybridRetriever.retrieve → Pipeline.run_async
       ┌─ bm25:     InMemoryBM25Retriever(top_k=20, filters)           ~4 ms
       ├─ embedder: ProviderQueryEmbedder → llm.embed([q], kind="query")
       ├─ dense:    ChromaEmbeddingRetriever(top_k=20, filters)        ~35 ms incl. embedding
       ├─ joiner:   DocumentJoiner(reciprocal_rank_fusion)  20+20 → ~30 unique
       └─ ranker:   SentenceTransformersSimilarityRanker(ms-marco-MiniLM-L-6-v2, top_k=8)
                    cross-encoder reads (query, chunk) pairs together  ~1.7 s on CPU
  → RetrievalResult(chunks[rank, score, chunk], stage_counts, score_kind, latency_ms)
```

### Eval results (real numbers, `make eval-retrieval`, 2026-09-30)

Corpus: 18 documents / 317 chunks (10 KB notes, 3 Wikipedia PDFs, 5 Wikipedia pages),
semantic chunking, `BAAI/bge-small-en-v1.5` embeddings, 35 questions graded at the level of
the relevant markdown section (PDFs: document level). Laptop CPU.

| Configuration | Recall@5 | MRR@10 | mean latency (ms) | p50 latency (ms) |
|---|---|---|---|---|
| BM25 only | 0.871 | 0.797 | 4 | 4 |
| Dense only | 0.857 | 0.809 | 36 | 35 |
| Hybrid (RRF) | 0.914 | 0.790 | 53 | 53 |
| Hybrid + rerank | 0.929 | 0.871 | 1790 | 1835 |

Reading it honestly:
- Hybrid beats either retriever alone on recall (+4–6 points): BM25 and dense fail on
  *different* questions, and RRF keeps whatever either found. RRF alone does not improve MRR:
  fusion promotes items both lists agree on, not necessarily the best one.
- The cross-encoder is what fixes the ordering (MRR 0.79 → 0.87) because it reads query and
  chunk together instead of comparing two independent vectors.
- The price is ~1.8 s per query for ~30 candidate pairs on a CPU: 30× the rest of the
  pipeline. Knobs: `RETRIEVAL_CANDIDATES` (fewer pairs), `MAX_CHUNK_CHARS` (shorter pairs),
  the ONNX backend of the ranker, or a GPU.
- Two questions were missed by every configuration (q03 Raft membership changes, q04 Paxos
  prepare phase): the top results were on-topic chunks from the Raft / Paxos *PDFs and web
  pages*, which outranked the labelled KB section. That is a limitation of the labels (only one
  relevant unit listed) more than of retrieval; the eval file is where to fix it.
- The corpus is small, so absolute numbers are optimistic. The *ordering* of configurations is
  the result that generalises.

### Semantic vs fixed chunking, side by side (`make compare-chunking`)

`knowledge_base/01-raft.md` (6 sections, 2.5k chars): fixed 256-token windows → 6 chunks of
233–658 chars, cut by budget (chunk [2] ends mid-argument in "Leader election"). Semantic
(p90) → 10 chunks of 67–567 chars, cut at topic shifts: the intro splits into "what Raft is"
and "one leader at a time; all writes go through the leader", and the election section
splits between the timeout mechanism and the vote-granting rule.
`raft_algorithm.pdf` page 2 (3.4k chars): fixed → 4 chunks of 467–1184 chars, boundaries at
arbitrary sentences with 32 tokens of duplicated overlap; semantic → 4 chunks of 284–1575
chars: term start / election mechanics / split vote → log replication. Same count, different
boundaries, no duplication.

### Why it is built this way

- **Two retrievers because they fail differently.** BM25 needs the query's words to appear
  ("hinted handoff" → exact); dense embeddings match paraphrases ("who is allowed to become
  leader" → "vote only if the candidate's log is at least as up to date"). Fusing rank lists
  with RRF needs no score calibration between the two, which is why it is the standard first
  choice.
- **Rank fusion, then rerank.** Cheap retrievers over-fetch (20 + 20), fusion dedupes and
  orders, the expensive cross-encoder only sees ~30 candidates instead of 317 chunks.
- **Haystack for the graph.** Each stage is a component with typed inputs/outputs; the
  pipeline object can be printed, drawn and inspected (`include_outputs_from` gives every
  stage's output, which is what `stage_counts` and the trace use).
- **Local embeddings by default** (D16): retrieval evals must be reproducible and free.
- **One filter grammar for both legs** (D17): a Planner-emitted filter reaches BM25 and
  Chroma identically.

### Trade-offs to own in an interview

- In-memory BM25 is rebuilt at startup (O(N)) and is per process: two API replicas would
  each hold a copy. Past ~100k chunks: OpenSearch behind the same interface.
- Dual writes (Chroma, then BM25) are not atomic; a restart heals drift.
- The cross-encoder dominates latency; batch size and candidate count are the levers.
- bge-small (384 dims) is a small model; on harder corpora a larger bi-encoder or API
  embeddings would lift the dense leg.
- RRF's k=60 is a convention, not tuned; the DocumentJoiner also supports weighting the legs.

### 5 likely interview questions

**Q1. Walk me through reciprocal rank fusion. Why rank, not score?**
Each retriever returns a ranked list. Every item gets `Σ 1/(k + rank_i)` over the lists it
appears in (k=60). Items ranked well by both lists accumulate the most; an item only one
retriever found still gets a vote. BM25 scores and cosine similarities live on different
scales, so summing or averaging scores needs calibration and breaks when one leg returns
nothing; ranks are scale-free. k flattens the curve so rank 1 vs rank 3 is not a landslide.
It is 10 lines (`mara/retrieval/rrf.py`) and my test checks it orders identically to
Haystack's implementation.

**Q2. Why a cross-encoder reranker if you already have embeddings?**
A bi-encoder embeds query and passage *independently* and compares vectors: fast, indexable,
but it cannot model interactions between specific query terms and passage terms. A
cross-encoder feeds `[query, passage]` through one transformer and outputs a relevance score,
attending across both: far more accurate, but O(candidates) forward passes per query, so it
can only run on a short list. The pattern is retrieve-cheap-then-rerank-expensive. In my eval
it raised MRR from 0.79 to 0.87 at a cost of ~1.7 s on CPU.

**Q3. How do metadata filters interact with vector search?**
Chroma applies the `where` filter during the HNSW search (pre-filtering with candidate
expansion), so `top_k` is filled with matching items, not filtered afterwards (post-filtering
can return fewer than k or nothing). I keep tags as boolean flags and dates as ints so both
Chroma and the in-memory BM25 store can evaluate the same filter; `MetadataFilter` is built
once by the Planner and pushed to both retrievers.

**Q4. What is BM25 and why did BM25L matter?**
BM25 scores a document by summing, per query term, IDF × a saturating term-frequency term
normalised by document length. Textbook Okapi IDF is `log((N − n + 0.5)/(n + 0.5))`, which is
≤ 0 when a term appears in at least half the documents; on a small or filtered corpus that
zeroed every score in a test. BM25L / BM25Plus use a strictly positive IDF and add a small
constant so long documents are not over-penalised.

**Q5. How do you keep the keyword index and the vector index consistent?**
Chroma is the source of truth. Ingestion writes Chroma first, then the BM25 index; deletes
propagate the same way. The BM25 index is rebuilt from Chroma on every startup, so any drift
from a crash between the two writes lasts until the next restart. For strict consistency I
would write both under one transactional outbox or move BM25 into the same engine
(OpenSearch does both keyword and vectors).

### 3 questions for you to answer before "next"

1. Two retrievers return lists A = [x, y, z] and B = [y, w]. Compute the RRF scores with k=60
   and give the fused order. Now put w at the top of B: does it beat x?
2. The reranker takes 1.8 s for 30 candidates. You are asked to get under 500 ms without a
   GPU. Name three changes, and what each one costs in recall.
3. A user filters by `tags: ["raft"]` and `source_types: ["web"]`. Trace the filter from the
   request body to the Chroma `where` clause and to the in-memory BM25 store.

---

## Phase 4 — Agents + orchestrator

### What exists now

```
mara/agents/state.py         ResearchState + every agent schema (Plan, EvidenceNote, Critique, ...)
mara/agents/base.py          StructuredLLM (JSON mode → Pydantic → one retry), prompt rendering
mara/agents/planner.py       Planner: question → 2-5 sub-questions with sources/filters
mara/agents/researcher.py    Researcher: per sub-question, concurrently: [web search → ingest] → retrieve
mara/agents/summarizer.py    Summarizer: chunks → notes; verify_quote() drops non-verbatim quotes
mara/agents/critic.py        Critic: coverage (code) + gaps/conflicts/new sub-questions (model)
mara/agents/writer.py        Writer: numbered sources → cited answer; citation validation + coverage
mara/agents/orchestrator.py  the control flow, timeouts, degradation, trace
mara/agents/web_search.py    WebSearchProvider: Noop | DuckDuckGo (ddgs) | Tavily
mara/agents/jobs.py          RedisJobStore / MemoryJobStore, TraceArchive (JSON per run)
mara/agents/runner.py        ResearchService: background asyncio task per job, emits events
mara/api/research_routes.py  POST /research, GET /research, GET /research/{id}, /events (SSE)
prompts/planner.md, summarizer.md, critic.md, writer.md
```

### The run, end to end

```
POST /research {"question": "...", "options": {"web_search": false, "max_loops": 1}}
  → 202 {"job_id": "..."}; ResearchService.start() → asyncio.create_task(orchestrator.run)

Orchestrator.run(state)
  planner     LLM → Plan{sub_questions[q1..qn]} (sources, tags, dates)        1 call
  ┌─ researcher  asyncio.gather over sub-questions:                              0 calls
  │               [web: search → fetch → trafilatura → ingest (tag web-search)]
  │               retriever.retrieve(q, filters, top_k) → evidence[q]
  │  summarizer  gather: LLM per sub-question → notes; verify_quote() drops     n calls
  │  critic      LLM → gaps / conflicts / new sub-questions                     1 call
  └─ loop once if needs_more_research and loop < max_loops (new sub-questions only)
  writer      sources = numbered verified chunks; LLM → answer [n]; validate    1 call
  → status done; trace[AgentStep...], warnings, citation_coverage; archived to data/traces/
GET /research/{id}/events  → SSE: "planner: 3 sub-questions", "researcher: q1: 6 chunks", ...
```

Typical cost: 3 + n LLM calls (n = sub-questions), ~2n with one loop. At Gemini free-tier
10 RPM the rate limiter stretches a 4-sub-question run to roughly 1-2 minutes.

### Why it is built this way

- **Explicit state, explicit steps.** `ResearchState` is the only memory; agents are
  `run(state, step) -> state`. You can print the state between any two steps, replay a
  step, or unit-test an agent with a hand-built state. (D19)
- **Evidence is retrieval's job, not the model's.** The Researcher has no prompt. The model
  only ever *summarises* and *judges* text that retrieval found; every claim has to be
  backed by an exact quote, checked in code. (D21)
- **Code owns the invariants, the model owns judgement.** Coverage, citation numbering,
  citation validity, loop count, timeouts: code. Sub-question phrasing, claim extraction,
  conflict detection, prose: model. (D22)
- **One loop, hard cap.** Bounded latency and cost; a reproducible trace. (D22)
- **Degrade, don't abort.** Each step has a timeout and a fallback; the Writer is told about
  every warning and says so in the answer. (D23)
- **Jobs outlive the request.** 202 + job id, Redis/in-memory state, SSE progress from an
  append-only event list. (D24)

### Trade-offs to own in an interview

- Runs are in-process `asyncio.Task`s: an API restart kills them. A queue (Arq/Celery) is
  the upgrade when that matters.
- `verify_quote` is strict; models sometimes fail to copy exactly and a *true* claim gets
  dropped. The drop count is in the trace, so the cost is measurable (Phase 5 eval).
- Coverage of a sub-question is "has ≥1 verified note", which is necessary, not sufficient;
  the Critic judges sufficiency but its gap list is advisory (it drives the loop, not the
  numbering).
- Web pages are ingested permanently into the corpus (tagged `web-search`); a research run
  therefore changes the corpus. Simple and cacheable, but a wrong page pollutes later runs
  until deleted.
- The SSE endpoint polls the store every 300 ms instead of using Redis pub/sub: simpler,
  works with the in-memory store, costs one small read per client per 300 ms.
- No real LLM run happened in this environment (no API key); the flow is verified by 100
  tests with a scripted fake LLM and real retrieval. Phase 5's eval is the first thing to
  run with a key.

### 5 likely interview questions

**Q1. Why not one big prompt with the whole corpus?**
Three reasons. Context: even a small corpus (317 chunks ≈ 150k tokens) exceeds sensible
prompt sizes, and cost scales with it per question. Faithfulness: a single prompt cannot show
*where* a claim came from; my design ties each claim to a verbatim quote from a retrieved
chunk, checked in code. Debuggability: when the answer is wrong I can see which step failed:
bad plan, bad retrieval, bad summary or bad writing, each with inputs and outputs in the
trace. The cost is more calls (3 + n) and more moving parts.

**Q2. How do you stop hallucinated citations?**
Citations are never free text the model invents. (1) The Summarizer must return a verbatim
quote with each claim, and `verify_quote` drops any note whose quote is not a substring of
the chunk. (2) Source numbers are assigned by code from the surviving notes *before* the
Writer runs, so `[3]` means one specific chunk. (3) After writing, citations to numbers that
do not exist are stripped and logged, and `citation_coverage` records how many factual
sentences carry a citation. A fabricated fact would need a fabricated quote that matches the
chunk text character for character.

**Q3. What happens when the web search provider is down mid-run?**
The Researcher catches `WebSearchError` (and fetch errors per page), records a warning
"web search failed; using the internal corpus only", and still runs retrieval over the
internal corpus for that sub-question. The Writer receives the warnings and adds a
limitation sentence. The job finishes as `done`, not `failed`; the trace shows the warning.
Same pattern for timeouts: a step that exceeds `AGENT_TIMEOUT_S` records an error and the
run continues with what it has.

**Q4. How does the Critic loop terminate?**
Two independent guards. The Critic may propose at most 3 new sub-questions and sets
`needs_more_research`; the orchestrator loops only while `state.loop < max_loops` (default 1,
max 2 by validation). The loop researches *only* the new sub-questions (existing evidence is
kept), so the second round is cheaper than the first. Worst case is therefore
Planner + 2 × (Researcher, Summarizer, Critic) + Writer, known before the run starts.

**Q5. How would you scale this to many concurrent users?**
Today: one process, jobs as asyncio tasks, in-memory BM25, Redis for cache and job state.
Steps: (1) move jobs to a queue with worker processes; the API only enqueues and serves
state; (2) replace the in-memory BM25 with OpenSearch (keyword + vector in one engine) or
keep Chroma and add a shared BM25 service; (3) the LLM rate limiter becomes Redis-based so
all workers share the quota; (4) per-tenant caches keyed on provider/model/prompt already
exist. Latency per run is dominated by LLM calls; batching sub-question summaries into one
call is the first cost cut.

### 3 questions for you to answer before "next"

1. A Summarizer note has a correct claim but its quote has one wrong character. What happens
   to it, and where in the trace can you see that it happened? Is that the right trade-off?
2. `max_loops=1`. The Critic proposes 3 new sub-questions in round 1 and 3 more in round 2.
   How many times does each agent run, and which sub-questions get researched in round 2?
3. Redis goes down after a job started. What happens to (a) the LLM cache, (b) the job's
   status endpoint, (c) the SSE stream, (d) the archived trace?

---

## Phase 5 — Answer quality eval

### What exists now

```
mara/agents/metrics.py     citation_validity(), answer_metrics(), judge_faithfulness()
prompts/judge.md           LLM-as-judge prompt: grade sentences against the cited excerpts ONLY
eval/answer_eval.json      15 research questions (one deliberately unanswerable: expect_gap)
eval/run_answer_eval.py    drives the API, scores each finished run, prints a markdown table
```

### The three automatic checks

| Check | Question it answers | How |
|---|---|---|
| Citation coverage | Did the writer cite what it claims? | share of factual sentences (≥25 chars, not headings) containing `[n]`; a sentence that says "No evidence was found" counts as covered |
| Citation validity | Does `[n]` point at real evidence? | the source's excerpt must occur verbatim (modulo whitespace) in the cited chunk, re-checked against the evidence in the state; unknown numbers are counted |
| Faithfulness (LLM-as-judge) | Are the sentences actually supported by those excerpts? | a second model call sees only the answer and the cited excerpts, never the full chunks or its own knowledge, and classifies each sentence supported / partial / unsupported |

Plus operational numbers per run: sources, notes dropped by the quote check, loops, gaps,
whether the answer admits a gap, LLM calls, tokens, latency.

### Status: not yet run

The eval needs an LLM key and this environment had none, so **there are no answer-quality
numbers yet**. Everything else is in place: `make run-local`, `make ingest-sample`, then

```bash
uv run python eval/run_answer_eval.py --judge
```

prints the table to paste into the README. The metric code is unit-tested with a
hand-built finished state (`tests/test_answer_metrics.py`). Until the eval has run, the
README says exactly that rather than showing invented numbers.

### What to expect and how to read it

- Validity should be ~1.0 by construction (the Writer only sees verified sources); anything
  lower means a source excerpt changed between summarising and writing, or a bug.
- Coverage below 1.0 is the interesting metric: it is the share of sentences the model
  wrote *without* pointing at evidence. The Writer prompt forbids that; the metric measures
  how well the instruction holds.
- Faithfulness catches the subtler failure: a cited sentence whose excerpt does not actually
  say that. Judge scores are themselves model outputs (noisy, prompt-sensitive), so report
  them next to the deterministic checks, never alone.
- `notes_dropped` shows how often the Summarizer failed to copy a quote exactly: the cost of
  the strict verbatim rule. If it is high, loosen the check (e.g. normalise quotes) or
  shorten `SUMMARIZER_MAX_QUOTE_CHARS`.
- a15 (throughput numbers) is designed to have no evidence; the right answer admits it.

### 5 likely interview questions

**Q1. How do you evaluate an answer when there is no gold answer?**
Split the question into properties you can check without a reference. Grounding is
checkable: did every claim cite a source, and does that source really contain the quoted
support (coverage, validity: deterministic). Faithfulness is judgeable: does the cited text
support the sentence (LLM-as-judge with the evidence in the prompt and nothing else).
Completeness needs a reference or a human; I approximate it with the Critic's gap list and a
question designed to have no answer, checking that the system says so.

**Q2. Isn't LLM-as-judge circular?**
It would be if the judge saw the same inputs and could use world knowledge. Here the judge
sees only the answer and the excerpts behind its citations, and is told to classify
sentences by whether the excerpt supports them, not by whether they are true. That turns it
into an entailment check on short texts, which models do reliably. It is still a model
output: I report it beside the deterministic metrics and keep the examples of unsupported
sentences so a human can spot-check.

**Q3. Why report a question that is designed to fail?**
Because "I don't know" is a feature. A research assistant that answers every question is
hallucinating on some of them. a15 asks for a number the corpus does not contain; the correct
behaviour is a sentence starting "No evidence was found for ...", which the coverage metric
credits and the `admits_gap` flag records.

**Q4. What would you do if coverage were 0.7?**
Look at the uncited sentences in the archived traces. Usually three kinds: connective prose
that is not factual (fix the metric: exclude it), true facts the writer added from memory
(fix the prompt, lower temperature, or drop them in post-processing), or facts that were in
the evidence but the Summarizer did not extract as notes (improve summarisation or raise
`RESEARCH_TOP_K`).

**Q5. How much does a run cost and where would you cut it?**
Cost = (3 + n) calls, roughly (planner ~1k tokens) + n × (summarizer ~3k) + critic ~2k +
writer ~3k input tokens; the trace records exact counts per step. Cuts, in order: cache hits
on repeated questions (already free), one summarizer call for all sub-questions (fewer calls,
one big prompt), a smaller model for planner/summarizer, fewer chunks per sub-question.

### 3 questions for you to answer before "next"

1. Validity checks the excerpt against the chunk; coverage checks for a `[n]`. Construct an
   answer with coverage 1.0 and validity 1.0 that is still wrong. Which check catches it?
2. The judge prompt says "do not use your own knowledge". Why is that instruction load-
   bearing, and what would a score of 1.0 on a nonsense answer tell you?
3. Which of the reported numbers would you put on a resume, and how would you phrase them
   so they cannot be called invented?

---

## Phase 6 — UI + polish

### What exists now

```
mara/static/index.html     the whole UI: one file, plain HTML + JS, no build step
mara/api/ui.py             GET / serves it
mara/api/rate_limit.py     per-client fixed-window limiter in Redis, POST routes only
mara/core/cache.py         redis_cached: the generic cache function decorator
mara/agents/web_search.py  CachedWebSearch uses that decorator
scripts/demo.py            make demo: ingest sample corpus, run one question, print trace
docs/ARCHITECTURE.md       one query end to end, file and function at each step
docs/INTERVIEW.md          resume map, 3-minute script, 15 questions, whiteboard kit
.github/workflows/ci.yml   ruff check + ruff format --check + pytest, no keys, no models
```

### How the UI works

```
page load      GET /health      -> chips: LLM key present? corpus size, reranker, web search
               GET /research    -> recent runs list
"Research"     POST /research   -> 202 {job_id}
               new EventSource("/research/{id}/events")
                 event: step    -> append "planner: 3 sub-questions ..." to the live list
                 event: done    -> close stream, GET /research/{id}, render
render         answer: tiny markdown renderer; [n] becomes a link that highlights source n
               sources: title, page/section, the verified quote
               trace: plan + evidence counts, critic summary, per-step latency/calls/tokens
"Search only"  POST /search     -> ranked chunks with scores; works with no LLM key
```

### Why it is built this way

- **No build step.** One HTML file with inline CSS and JS is the smallest thing that shows
  the system working: no node, no bundler, nothing to break before a demo. The cost is no
  components and no type checking; fine at 250 lines, wrong at 2,500.
- **SSE, not WebSockets.** Progress flows one way (server to browser). `EventSource` is
  built into browsers, reconnects by itself and sends `Last-Event-ID`, and the server side is
  a plain streaming HTTP response. WebSockets would add a protocol upgrade and a connection
  manager for a channel we never write to.
- **The UI reads the same endpoints as the evals.** Everything it shows comes from
  `GET /research/{id}`, so the trace in the browser is exactly what is archived on disk.
- **Rate limiting in Redis, in front of the expensive routes only.** A research run costs
  LLM calls; polling and the SSE stream cost almost nothing, so only POSTs are counted.
- **CI needs nothing external.** Fake LLM, fake embeddings, fakeredis, embedded Chroma: the
  suite runs in about 15 seconds on a laptop and the same way on a clean Linux runner.

### Two bugs that only CI found (worth telling in an interview)

1. **Hard links vs NLTK.** On Linux, `uv` installs packages as hard links into its cache.
   NLTK 3.10 refuses to open data files with more than one hard link (a guard against
   link-following attacks), and LlamaIndex bundles its sentence-tokenizer data inside the
   wheel. Result: semantic chunking raised `PermissionError` on the CI runner and worked on
   Windows. Fix: `link-mode = "copy"` in `[tool.uv]`. Lesson: "works on my machine" included
   the package installer's behaviour.
2. **The formatter also formats Markdown.** `ruff format --check` covers Python code blocks
   inside `.md` files, so a doc commit failed CI. Lesson: run the same commands CI runs
   before pushing, on every commit, not only code commits.

### Trade-offs to own in an interview

- The markdown renderer is 15 lines and handles headings, bullets, bold, code and citations
  only. HTML is escaped first, so model output cannot inject markup.
- SSE here is implemented by polling the job store every 300 ms per connected client.
  Simple and store-agnostic; Redis pub/sub would remove the polling at the cost of a second
  code path for the in-memory store.
- The fixed-window limiter allows up to twice the limit across a window boundary, and it
  identifies clients by IP, which is wrong behind a shared proxy. An API key per client is
  the fix.
- CI was red from Phase 2 until Phase 6 because it was only checked after Phase 1.

### 5 likely interview questions

**Q1. Why server-sent events rather than WebSockets or polling?**
The data flows one way and is a sequence of small text events, which is exactly what SSE is
for: a long-lived HTTP response with `text/event-stream`, automatic reconnection and
resumption through `Last-Event-ID`, and it passes through ordinary HTTP infrastructure.
WebSockets are for two-way, low-latency traffic. Client polling would work but wastes
requests and delays updates; I do keep a polling fallback in the page for when the stream
drops.

**Q2. How does the rate limiter work and what are its weaknesses?**
One Redis key per client per minute: `INCR`, and on the first increment `EXPIRE` it for the
window. Above the limit the API returns 429 with `Retry-After`. It is atomic and shared by
every API replica. Weaknesses: a burst at a window boundary can reach twice the limit
(a sliding-window log or token bucket in Lua fixes that), client identity is the IP, and if
Redis is down it fails open by design.

**Q3. How do you keep model output from breaking the page?**
The answer is untrusted text. The renderer escapes HTML before applying any formatting, and
only then turns a small, fixed set of patterns (headings, bullets, bold, code, `[n]`) into
tags. Citation links are generated from digits only. Nothing from the model is ever
inserted as raw HTML.

**Q4. What does your CI prove, and what does it not?**
It proves the deterministic shell: orchestration order, loop cap, timeouts, degradation,
quote verification, citation checks, caching, retries, filters, ingestion idempotency and
real Haystack pipelines with fake embeddings, on Linux. It does not prove model quality,
the live provider adapters, a real Redis server or the Docker image. Those need a key and
running services, and the README lists them as unverified.

**Q5. Tell me about a bug that was hard to find.**
Semantic chunking passed locally and failed in CI with a `PermissionError` from NLTK. The
traceback pointed at a security check refusing a "multiply-linked file". The file was
tokenizer data shipped inside the LlamaIndex wheel, and the extra link came from the package
installer hard-linking files out of its cache on Linux. Two independent, reasonable
behaviours combined into a failure. The fix was one config line; finding it meant reading
the traceback literally instead of suspecting my own code.

### 3 questions for you to answer

1. A user opens the page while a job is half finished and the connection drops. Trace what
   the browser and the server each do so that no step is shown twice or lost.
2. Two API replicas sit behind a load balancer. Which of these still work correctly without
   changes, and which do not: the LLM cache, the API rate limiter, the outbound LLM rate
   limiter, research jobs, the BM25 index?
3. Which three claims about this project can you not make yet, and what exactly would you
   run to be able to make them?
