# Interview kit

1. [Resume → code map](#1-resume--code-map) (and what still needs a live run)
2. [Explain MARA in 3 minutes](#2-explain-mara-in-3-minutes)
3. [The 15 hardest questions](#3-the-15-hardest-questions)
4. [Whiteboard kit: 5 functions from memory](#4-whiteboard-kit)

---

## 1. Resume → code map

### Bullet 1
> Built a multi-agent AI research assistant that plans tasks, retrieves evidence, summarizes
> findings, and generates citation-backed answers using LlamaIndex and Haystack

| Claim | Where it is true |
|---|---|
| multi-agent | `mara/agents/orchestrator.py::Orchestrator.run` drives five agents over one `ResearchState` (`mara/agents/state.py`); the Critic can trigger one extra loop |
| plans tasks | `mara/agents/planner.py::Planner.run` + `prompts/planner.md` → `Plan` (2–5 `SubQuestion`s with sources and filters) |
| retrieves evidence | `mara/agents/researcher.py::Researcher.run` (`asyncio.gather`) → `mara/retrieval/hybrid.py::HaystackHybridRetriever.retrieve` |
| summarizes findings | `mara/agents/summarizer.py::Summarizer.run` → `EvidenceNote{claim, supporting_quote, chunk_id}`, checked by `verify_quote` |
| citation-backed answers | `mara/agents/writer.py::Writer.run`, `build_sources`, `drop_invalid_citations`, `citation_coverage` |
| using LlamaIndex | ingestion: `mara/ingest/loaders.py::load_pdf` (`PDFReader`), `mara/ingest/chunking.py` (`SemanticSplitterNodeParser`, `SentenceSplitter`), `mara/ingest/embeddings.py::ProviderEmbedding` (`BaseEmbedding`) |
| using Haystack | query pipeline: `mara/retrieval/hybrid.py::_build` (`Pipeline`, `InMemoryBM25Retriever`, `DocumentJoiner`, `SentenceTransformersSimilarityRanker`, chroma-haystack `ChromaEmbeddingRetriever`), `mara/retrieval/bm25_index.py` (`InMemoryDocumentStore`) |

### Bullet 2
> Implemented hybrid retrieval with semantic chunking, metadata filtering, and reranking over
> PDFs, web pages, and internal knowledge bases

| Claim | Where it is true |
|---|---|
| hybrid retrieval | BM25 + dense fused with RRF: `hybrid.py::_build`; own implementation `mara/retrieval/rrf.py::reciprocal_rank_fusion` (parity-tested against Haystack) |
| semantic chunking | `mara/ingest/chunking.py::SemanticChunker` (+ `pack_sentences` cap); compare with `scripts/compare_chunking.py` |
| metadata filtering | `mara/core/filters.py::MetadataFilter`, `to_haystack`, `build_where`; storage layout `mara/core/schema.py::Chunk.to_chroma_metadata` |
| reranking | cross-encoder `ms-marco-MiniLM-L-6-v2`: `mara/retrieval/factory.py::ranker_factory` |
| PDFs | `loaders.py::load_pdf`, `POST /ingest/pdf` (page numbers kept) |
| web pages | `mara/ingest/web.py::fetch_html` + `loaders.py::load_html` (trafilatura), `POST /ingest/url`, and on the fly in `researcher.py::_web_ingest` |
| internal knowledge bases | `loaders.py::load_markdown` / `load_kb_dir`, `POST /ingest/kb`, `knowledge_base/*.md` |
| measured | `eval/run_retrieval_eval.py`: Recall@5 0.929 / MRR@10 0.871 for hybrid + rerank vs 0.871 / 0.797 for BM25 (35 questions, 317 chunks) |

### Tech line
| Tech | Where |
|---|---|
| Python | everything; 3.12, `uv`, ruff, pytest (112 tests) |
| FastAPI | `mara/api/main.py`, `ingest_routes.py`, `search_routes.py`, `research_routes.py` (SSE), `ui.py`, `rate_limit.py` |
| LlamaIndex | `llama-index-core` 0.14.25, `llama-index-readers-file` 0.7.0 — see bullet 1 |
| Haystack | `haystack-ai` 3.2.0, `chroma-haystack` 4.5.0, `sentence-transformers-haystack` 0.1.2 — see bullet 1 |
| ChromaDB | `mara/core/chunk_store.py::ChunkStore` (writes, admin, embedding-model guard); read at query time through chroma-haystack |
| Redis | cache: `mara/core/cache.py` (`RedisCache`, `redis_cached`), `mara/llm/cached.py::CachedLLM`; jobs: `mara/agents/jobs.py::RedisJobStore`; per-client rate limiting: `mara/api/rate_limit.py::RateLimiter` |
| Docker | `Dockerfile` (multi-stage), `docker-compose.yml` (api + redis + chroma) — written, **never run** |
| OpenAI/Gemini APIs | `mara/llm/gemini.py::GeminiProvider` (`google-genai`), `mara/llm/openai_provider.py::OpenAIProvider` (`openai`), selected by `mara/llm/factory.py` |

### Honest status (updated 2026-10-01)

| Item | Status |
|---|---|
| **Gemini** | Exercised live: 16 real research runs, including the 15-question answer eval (coverage 0.96, validity 1.00, judge faithfulness 0.70). |
| **OpenAI** | Adapter tested only against a stub SDK client; never called live. Say "OpenAI-compatible adapter" if pressed. |
| **Redis** | Exercised for real against a native Windows Redis 3.0 service (cache, job store, rate limiter), using the RESP2 protocol setting. |
| **Docker** | `Dockerfile` and `docker-compose.yml` were written but never built or run. Run them once, or drop "Docker" from the resume tech line. |

Be ready to discuss the 0.70 faithfulness score and the over-eager critic; both are in the
README's evaluation section with their likely causes.

### Wording suggestions (defensible today)

- "…hybrid retrieval (BM25 + dense, reciprocal rank fusion, cross-encoder reranking):
  Recall@5 0.93 / MRR 0.87 vs 0.87 / 0.80 for BM25 alone on a 35-question eval set."
  Say "small sample corpus" when asked.
- "…citation-backed answers: every claim must carry a verbatim quote from its source,
  verified in code."
- Haystack: the installed major version is 3.x (`haystack-ai`), successor of 2.x with the
  same component/pipeline model. Say "Haystack", not "Haystack 2".

---

## 2. Explain MARA in 3 minutes

> **What it is (20 s).** MARA is a research assistant over your own documents: PDFs, web
> pages and a markdown knowledge base. You ask a compound question and get an answer where
> every factual sentence has a citation, plus a trace of how it was produced.
>
> **Why multi-agent (30 s).** A single RAG call retrieves once with the user's wording and
> cannot tell you where a claim came from. I split the work into five small steps over one
> explicit state object: a Planner breaks the question into two to five sub-questions and
> says which sources to search; a Researcher runs retrieval for each sub-question in
> parallel; a Summarizer turns the retrieved chunks into evidence notes; a Critic checks
> coverage and conflicts and can trigger exactly one more research round; a Writer produces
> the cited answer. The orchestration is thirty lines of plain Python, no agent framework,
> because I wanted to be able to draw it.
>
> **The part I'm proudest of (40 s).** Citations cannot be hallucinated. Every note the
> Summarizer produces must include a verbatim quote from the chunk, and code checks that the
> quote really is a substring; otherwise the note is dropped. Citation numbers are assigned
> by code from the surviving notes before the Writer runs, and afterwards citations to
> numbers that do not exist are stripped. The model does judgement and prose; code owns the
> invariants.
>
> **Retrieval (40 s).** Ingestion uses LlamaIndex: loaders plus semantic chunking, which
> splits where the embedding similarity between neighbouring sentences drops, so a chunk is
> about one thing. Chunks go to Chroma with metadata. At query time a Haystack pipeline runs
> BM25 and dense retrieval, fuses the two rank lists with reciprocal rank fusion, and a
> cross-encoder reranks the top thirty. On my 35-question eval that took Recall@5 from 0.87
> with BM25 alone to 0.93, and MRR from 0.80 to 0.87; the price is about 1.8 seconds of
> reranking on a CPU.
>
> **Engineering (30 s).** FastAPI with background jobs and server-sent events for live agent
> steps. The LLM sits behind one interface with Gemini and OpenAI adapters, wrapped by a
> rate limiter, retries with backoff, and a Redis cache keyed on a hash of model, prompt and
> parameters. Everything degrades instead of failing: if web search or Redis is down, the run
> continues and the answer says what was missing. 112 tests run with a fake LLM, so CI needs
> no keys.
>
> **What I'd do next (20 s).** Move jobs to a queue, replace the in-memory BM25 with
> OpenSearch for scale, and cut reranking latency with an ONNX model.

---

## 3. The 15 hardest questions

**1. Why both LlamaIndex and Haystack? Isn't one enough?**
Either could do the whole job; I use each where it is strongest and keep them decoupled.
LlamaIndex has the better ingestion toolbox: loaders and node parsers, in particular
`SemanticSplitterNodeParser`. Haystack has the more explicit query-time model: a pipeline is
a typed graph of components I can inspect stage by stage, which my trace and eval need. They
never call each other; the contract is one chunk schema (`mara/core/schema.py`) and one
Chroma layout. The honest cost is two dependency trees to pin. If I had to drop one I would
keep Haystack and re-implement the semantic splitter (about 40 lines).

**2. Why not just one big prompt?**
Context, faithfulness, debuggability. The sample corpus alone is ~150k tokens and cost
scales per question. One prompt cannot show where a claim came from; my pipeline ties each
claim to a verbatim quote from a retrieved chunk and verifies it in code. And when an answer
is wrong, the trace shows which step failed. The cost: 3 + n LLM calls instead of one.

**3. How do you stop hallucinated citations?**
Three mechanisms, all in code. (1) `verify_quote`: each note must carry a quote that is a
substring of its chunk (whitespace-normalised, case-sensitive); failures are dropped and
counted. (2) `build_sources`: citation numbers are assigned from verified notes before the
Writer's prompt is built, so `[3]` has a fixed meaning. (3) `drop_invalid_citations` and
`citation_coverage` after writing. What this does not catch: a sentence citing a real source
for something the source does not say. The LLM-as-judge faithfulness check measures that.

**4. How would this scale to 10 million documents?**
That is 100M+ chunks. What breaks, in order: (a) in-memory BM25 → OpenSearch/Elasticsearch,
or one engine for keyword and vectors; (b) single-node Chroma → a distributed vector store
(Qdrant, Milvus, pgvector) with quantisation, sharded by tenant or time; (c) ingestion → a
queue with workers and batched GPU embedding, idempotent by content hash (already true);
(d) `GET /documents` derived from chunk metadata → a documents table; (e) reranking → a GPU
service or ONNX model; candidates per query stay constant; (f) jobs → a task queue, a shared
Redis rate limiter, more API replicas. The agent design, chunk schema, filters and evals do
not change.

**5. What is the latency and cost per query, and how would you cut it?**
Retrieval is measured: 4 ms BM25, ~36 ms dense, ~53 ms hybrid, ~1.8 s with CPU reranking. A
research run is 3 + n LLM calls for n sub-questions, roughly double with one loop. Wall-clock
is dominated by LLM latency and, on a free tier, by the 10-requests-per-minute limiter. Exact
tokens and latency per step are in every trace, so I quote measured numbers from a run, not
estimates. Cuts, by payoff: cache hits, one summarizer call for all sub-questions, a smaller
model for planner/summarizer, fewer chunks per sub-question, skipping the critic for simple
questions, an ONNX reranker.

**6. Walk me through reciprocal rank fusion. Why not combine scores?**
Each list contributes `1/(k + rank)` per item (k=60); sum; sort. BM25 scores and cosine
similarities are on different, query-dependent scales, so combining them needs normalisation
that breaks when one list is empty or skewed. Ranks are scale-free. k damps the head so that
agreement between lists beats a single first place. My 10-line version is tested to order
exactly like Haystack's `DocumentJoiner`.

**7. Bi-encoder vs cross-encoder?**
A bi-encoder embeds query and passage separately, so passages can be indexed and searched in
milliseconds, but the two never see each other. A cross-encoder runs the pair through one
transformer with attention between them: far more accurate, one forward pass per candidate,
so it only reranks a short list. Retrieve cheaply for recall, rerank expensively for
precision.

**8. How does semantic chunking decide where to cut, and when is it worse than fixed-size?**
Sentence-split; embed each sentence with a window of neighbours; cosine distance between
consecutive windows; cut where the distance exceeds the 90th percentile of the document's
distances. Worse when documents have no topical structure (logs, tables), when the corpus
changes constantly (~3× the embedding calls), or when chunks need a hard token budget (I cap
oversized chunks with a sentence packer).

**9. What goes into the LLM cache key, and how can caching LLM output go wrong?**
Provider, model, prompt, system prompt, temperature, max tokens and the output schema, as
canonical JSON hashed with SHA-256, plus a version prefix for mass invalidation. Failure
modes: a missing input in the key (a wrong answer served as a hit); stale answers after the
corpus changes (safe here: the prompt contains the retrieved chunks, so new evidence changes
the key); non-deterministic outputs frozen until the TTL; caching errors (only successes are
cached).

**10. What happens when Redis is down?**
Cache reads become misses and writes are skipped; the API rate limiter fails open; the job
store falls back to memory at startup; `/health` reports `degraded` but stays 200. Slower and
more expensive, not down. Lost: job state across restarts and the shared rate limit.

**11. Why plain Python instead of LangGraph?**
The control flow is a sequence with one bounded loop; a graph framework adds abstractions
without removing complexity I actually have. Plain Python keeps state explicit, makes every
agent unit-testable with a hand-built state, and fits on a whiteboard. I would reach for a
framework for durable checkpoints, human-in-the-loop interrupts or dynamic graphs.

**12. The Critic is an LLM. Why trust it to decide whether to loop?**
I don't, fully. Coverage is computed in code (covered iff a verified note exists); the model
contributes judgement on thin evidence, conflicts and better queries. The loop is capped
(`max_loops`, default 1) and the second round researches only the new sub-questions.

**13. How do you test a system whose core is non-deterministic?**
Separate the deterministic shell from the model. Orchestration order, loop cap, timeouts,
degradation, quote verification, citation validation, caching and retries are tested with a
scripted `FakeLLM`. Retrieval is tested with real Haystack pipelines and fake topic-axis
embeddings. Model-dependent quality is measured, not unit-tested: the two evals.

**14. How do metadata filters work across two different stores?**
One `MetadataFilter`; `to_haystack()` emits Haystack's grammar, which chroma-haystack
translates to Chroma's `where` and the in-memory store evaluates in Python. Stored metadata
is kept simple for that: dates as `YYYYMMDD` ints, tags as boolean flags (`tag:raft = True`)
because the in-memory store has no list-membership operator. Filters apply during retrieval,
so `top_k` is filled with matching items.

**15. What is the weakest part, and what surprised you in the eval?**
Weakest: answer quality is not yet measured with a live model, and the Redis and Docker
paths have only been tested with fakes or not at all. Retrieval labels list one relevant
unit per question, so two "misses" were on-topic PDF chunks outranking the labelled KB
section. Surprises: RRF improved recall but not MRR (0.790 vs 0.797–0.809 for the single
retrievers) — fusion rewards agreement, not correctness, and only the cross-encoder fixed
the ordering; and textbook BM25Okapi scored every document 0.0 on a four-document corpus
because its IDF is zero for terms in half the documents, hence BM25L.

---

## 4. Whiteboard kit

Five functions to write from memory. Each is under 25 lines and matches the repo.

### 4.1 Reciprocal rank fusion — `mara/retrieval/rrf.py`

```python
from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: one list of ids per retriever, best first."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
```

Say: ranks not scores (scale-free); k=60 damps the head; ties broken by id for determinism.

### 4.2 The orchestrator loop — `mara/agents/orchestrator.py`

```python
async def run(self, state: ResearchState) -> ResearchState:
    state = await self._step(self.planner, state)
    if not state.plan:  # planner failed: degrade
        state.plan = fallback_plan(state)
    while True:
        state = await self._step(self.researcher, state)  # parallel per sub-question
        state = await self._step(self.summarizer, state)  # verified notes only
        state = await self._step(self.critic, state)
        c = state.critique
        if c and c.needs_more_research and state.loop < state.options.max_loops:
            state.loop += 1
            state.plan.extend(c.new_sub_questions)  # only these are researched next
            continue
        break
    state = await self._step(self.writer, state)
    state.status = "done" if state.answer else "failed"
    return state


async def _step(self, agent, state):
    step = AgentStep(agent=agent.name, loop=state.loop, started_at=utc_now())
    try:
        state = await asyncio.wait_for(agent.run(state, step), timeout=self._timeout)
    except (TimeoutError, AgentError, LLMError) as e:
        step.error = str(e) or "timed out"
        state.warnings.append(f"{agent.name}: {step.error}")
    state.trace.append(step)
    return state
```

Say: one state object; hard loop cap; every step timed, traced, and allowed to fail.

### 4.3 The cache decorator — `mara/core/cache.py`

```python
def redis_cached(cache: RedisCache, namespace: str, ttl_s: int, version: str = "v1"):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            key = make_cache_key(
                version, namespace, "fn", fn.__qualname__, args=args, kwargs=kwargs
            )
            if (raw := await cache.get(key)) is not None:  # hit
                return json.loads(raw)
            result = await fn(*args, **kwargs)  # miss: call through
            await cache.set(key, json.dumps(result, default=str).encode(), ttl_s)
            return result

        return wrapper

    return decorator


def make_cache_key(version, namespace, provider, model, **parts) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"mara:{version}:{namespace}:{provider}:{model}:{digest}"
```

Say: key = hash of everything that changes the output; canonical JSON so argument order
does not matter; TTL; failures are not cached; `RedisCache.get/set` swallow Redis errors
(fail-open). `CachedLLM` is the same idea with per-text keys for embeddings.

### 4.4 Quote verification — `mara/agents/summarizer.py`

```python
_WS = re.compile(r"\s+")


def verify_quote(quote: str, chunk_text: str) -> bool:
    q = _WS.sub(" ", quote).strip()
    return len(q) >= 10 and q in _WS.sub(" ", chunk_text)


def keep_verified(notes, chunks_by_id):
    kept, dropped = [], 0
    for note in notes:
        chunk = chunks_by_id.get(note.chunk_id)
        if chunk is None or not verify_quote(note.supporting_quote, chunk.text):
            dropped += 1
            continue
        kept.append(note)
    return kept, dropped
```

Say: verbatim modulo whitespace (PDF line wraps), case-sensitive, a minimum length so one
word is not "evidence"; unknown chunk ids are dropped too; the drop count goes in the trace.

### 4.5 The metadata filter builder — `mara/core/filters.py`

```python
def build_where(f: MetadataFilter | None) -> dict | None:
    if f is None:
        return None
    conds = []
    if f.source_types:
        conds.append({"source_type": {"$in": list(f.source_types)}})
    if f.doc_ids:
        conds.append({"doc_id": {"$in": list(f.doc_ids)}})
    if f.tags:  # any-of
        tags = [{f"tag:{t.strip().lower()}": {"$eq": True}} for t in f.tags]
        conds.append(tags[0] if len(tags) == 1 else {"$or": tags})
    if f.date_from:
        conds.append({"published_date": {"$gte": date_to_int(f.date_from)}})
    if f.date_to:
        conds.append({"published_date": {"$lte": date_to_int(f.date_to)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}
```

Say: Chroma wants a bare condition or an explicit `$and`; dates are ints so ranges work;
tags are boolean flags so the same filter also works in the in-memory BM25 store
(`to_haystack` is this function's twin in Haystack's grammar).
