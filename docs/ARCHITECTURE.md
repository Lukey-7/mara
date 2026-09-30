# Architecture: one research query, end to end

Every step below names the file and function that does it, in the order it happens.
Start the API (`make run-local`), open `http://localhost:8080/`, and follow along.

## 0. Startup wiring — `mara/api/main.py::create_app` (the `lifespan` function)

| What is built | Where |
|---|---|
| Settings from `.env` | `mara/core/config.py::Settings` |
| LLM stack `CachedLLM(ResilientLLM(CompositeProvider))` | `mara/llm/factory.py::build_llm` |
| Chroma client + `ChunkStore` (embedding-model guard) | `mara/core/chunk_store.py::make_chroma_client`, `mara/ingest/factory.py::build_store` |
| BM25 copy, filled from Chroma | `mara/retrieval/bm25_index.py::BM25Index.reload` |
| Ingestion pipeline (semantic chunker, dual write) | `mara/ingest/factory.py::build_pipeline` |
| Haystack retriever + cross-encoder warm-up | `mara/retrieval/factory.py::build_retriever` |
| Job store, orchestrator, research service | `mara/agents/factory.py::build_job_store`, `build_orchestrator`, `build_research_service` |

Each block is wrapped in `try/except`: a missing key or a down dependency makes `/health`
report `degraded` with the reason, and only the endpoints that need it return 503.

## 1. The request

1. Browser: `mara/static/index.html::research()` → `POST /research {question, options}`.
2. `mara/api/research_routes.py::start_research` → `ResearchService.start`
   (`mara/agents/runner.py`): builds a `ResearchState` (`mara/agents/state.py`), saves it
   (`JobStore.save`), appends the event `queued`, and schedules
   `asyncio.create_task(self._run(state))`. The HTTP response is `202 {job_id}`.
3. Browser opens `EventSource("/research/{id}/events")` →
   `research_routes.py::stream_events` polls `JobStore.events(job_id, after_seq)` every
   300 ms and emits one SSE `step` event per agent message, then a final `done`.

## 2. The run — `mara/agents/orchestrator.py::Orchestrator.run`

Each agent call goes through `Orchestrator._step`: `asyncio.wait_for(agent.run(state, step),
timeout)`, record an `AgentStep` (latency, LLM calls, tokens, cache hits, error) in
`state.trace`, emit an event.

### 2a. Planner — `mara/agents/planner.py::Planner.run`

- Prompt: `prompts/planner.md` rendered by `mara/agents/base.py::render` with the question,
  the allowed sources and the corpus's known tags (`mara/api/main.py::_known_tags`).
- `mara/agents/base.py::StructuredLLM.call(prompt, Plan, step)`:
  - `CachedLLM.generate` (`mara/llm/cached.py`) builds the key with
    `mara/core/cache.py::make_cache_key` (provider, model, prompt, temperature, schema).
    Hit → return with `cached=True`.
  - Miss → `ResilientLLM.generate` (`mara/llm/resilience.py`): `AsyncRateLimiter.acquire`
    (token bucket) inside `retry_async` (exponential backoff + jitter on
    `RetryableLLMError`).
  - → `CompositeProvider.generate` (`mara/llm/composite.py`) → `GeminiProvider.generate`
    (`mara/llm/gemini.py`) with `response_json_schema =
    inline_refs(Plan.model_json_schema())` (`mara/llm/schema_utils.py`).
  - `Plan.model_validate_json`; on a validation error the call is repeated once with the
    error quoted back.
- `normalise_sub_questions` renumbers ids and clamps sources. If the planner fails,
  `Orchestrator.run` uses `planner.py::fallback_plan`.

### 2b. Researcher — `mara/agents/researcher.py::Researcher.run`

`asyncio.gather` over sub-questions → `_research_one`:

1. Optional web evidence (`_web_ingest`): `WebSearchProvider.search`
   (`mara/agents/web_search.py`, cached by `CachedWebSearch` via
   `mara/core/cache.py::redis_cached`) → `mara/ingest/web.py::fetch_html` →
   `mara/ingest/loaders.py::load_html` (trafilatura) → `IngestionPipeline.ingest`. Any
   failure becomes a warning.
2. Filters: `SubQuestion.to_filter` (`mara/agents/state.py`) merges the planner's sources /
   tags / dates with the request-level `MetadataFilter`.
3. Retrieval: `mara/retrieval/hybrid.py::HaystackHybridRetriever.retrieve`:
   - `mara/core/filters.py::to_haystack` → one filter dict for both legs;
   - `Pipeline.run_async` on the graph built by `HaystackHybridRetriever._build`:
     - `bm25`: Haystack `InMemoryBM25Retriever` over `BM25Index.store`;
     - `embedder`: `mara/retrieval/components.py::ProviderQueryEmbedder.run_async` →
       `llm.embed([query], kind="query")` → `mara/llm/local_embeddings.py::LocalEmbeddingProvider.embed`;
     - `dense`: `components.py::ChromaDenseRetriever` → chroma-haystack
       `ChromaEmbeddingRetriever`;
     - `joiner`: Haystack `DocumentJoiner(join_mode="reciprocal_rank_fusion")` (same
       ordering as our `mara/retrieval/rrf.py::reciprocal_rank_fusion`);
     - `ranker`: `SentenceTransformersSimilarityRanker` (cross-encoder) built by
       `mara/retrieval/factory.py::ranker_factory`;
   - `mara/retrieval/bm25_index.py::document_to_chunk` turns Haystack Documents back into
     `Chunk`s → `RetrievalResult`.
4. `state.evidence[sub_question_id] = hits`.

### 2c. Summarizer — `mara/agents/summarizer.py::Summarizer.run`

`asyncio.gather` over sub-questions with evidence → `_summarize_one`: prompt
`prompts/summarizer.md` with the chunks (`<chunk id=...>`), `StructuredLLM.call(...,
SummarizerOutput)`, then **`verify_quote(note.supporting_quote, chunk.text)`** for every
note; failures are dropped and counted in the step summary.

### 2d. Critic — `mara/agents/critic.py::Critic.run`

Prompt `prompts/critic.md` → `Critique`. Coverage and gaps are recomputed in code from the
verified notes; new sub-questions go through `normalise_sub_questions` (max 3).
Back in `Orchestrator.run`: if `critique.needs_more_research and state.loop <
state.options.max_loops`, extend the plan and repeat 2b–2d for the new sub-questions only.

### 2e. Writer — `mara/agents/writer.py::Writer.run`

`build_sources(state)` numbers the distinct chunks behind the verified notes; prompt
`prompts/writer.md` gets the numbered sources, gaps, conflicts and warnings;
`StructuredLLM.call(..., WriterOutput)`; then `drop_invalid_citations` and
`citation_coverage`.

## 3. Finishing

- `Orchestrator.run` sets `status` (`done` unless the writer produced no answer) and
  `finished_at`.
- `ResearchService._run` saves the final state (`RedisJobStore.save` / `MemoryJobStore.save`)
  and archives it: `mara/agents/jobs.py::TraceArchive.write` → `data/traces/<job_id>.json`.
- The SSE stream sees the terminal status and sends `done`; the page calls `loadJob()` →
  `GET /research/{id}` (`research_routes.py::get_research`) and renders the answer
  (`md()`), sources (clickable `[n]`) and trace (`renderTrace()`).

## Ingestion, for reference

`POST /ingest/pdf` → `mara/api/ingest_routes.py::ingest_pdf` → `loaders.py::load_pdf`
(LlamaIndex `PDFReader`, one unit per page) → `mara/ingest/pipeline.py::IngestionPipeline.ingest`:
`loaders.py::content_hash` → `ChunkStore.has_document` (skip duplicates) →
`mara/ingest/chunking.py::SemanticChunker.chunk` (LlamaIndex `SemanticSplitterNodeParser`
through `mara/ingest/embeddings.py::ProviderEmbedding`, then `pack_sentences` for oversized
chunks) → `llm.embed(texts, kind="document")` → `ChunkStore.upsert`
(`Chunk.to_chroma_metadata`) → `BM25Index.add`.

## Interfaces (Protocols) and the patterns behind them

| Protocol | File | Implementations | Pattern |
|---|---|---|---|
| `LLMProvider` | `mara/llm/base.py` | `GeminiProvider`, `OpenAIProvider`, `CompositeProvider`; wrapped by `ResilientLLM`, `CachedLLM` | Strategy + Decorator |
| `Chunker` | `mara/ingest/chunking.py` | `SemanticChunker`, `FixedChunker` | Strategy |
| `SecondaryIndex` | `mara/ingest/pipeline.py` | `BM25Index` | Observer-style dual write |
| `Retriever` | `mara/retrieval/hybrid.py` | `HaystackHybridRetriever` (tests: `FakeRetriever`) | Strategy / Pipeline |
| `Reranker` | `mara/retrieval/components.py` | Haystack `SentenceTransformersSimilarityRanker`, `PassthroughRanker` | Strategy |
| `WebSearchProvider` | `mara/agents/web_search.py` | `NoopWebSearch`, `DuckDuckGoWebSearch`, `TavilyWebSearch`, `CachedWebSearch` | Strategy + Decorator |
| `Agent` | `mara/agents/base.py` | `Planner`, `Researcher`, `Summarizer`, `Critic`, `Writer` | Pipeline over shared state |
| `JobStore` | `mara/agents/jobs.py` | `RedisJobStore`, `MemoryJobStore` | Strategy |

Factories (`mara/llm/factory.py`, `mara/ingest/factory.py`, `mara/retrieval/factory.py`,
`mara/agents/factory.py`) are the only places that read settings to choose implementations.
