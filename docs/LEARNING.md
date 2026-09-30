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
