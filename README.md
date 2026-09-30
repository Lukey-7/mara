# MARA — Multi-Agent Research Assistant

A research assistant that **plans** a question into sub-questions, **retrieves** evidence with
hybrid search (BM25 + dense + reranking) over PDFs, web pages and an internal knowledge base,
**summarizes** it into verifiable evidence notes, **checks** coverage, and **writes** a
citation-backed answer. Every run produces a trace of what each agent did.

> Status: **Phase 3 of 6** (skeleton, ingestion, hybrid retrieval). Agents, answer evals and
> UI follow. Numbers in this README come only from `eval/`.

## Stack

| Layer | Tool | Job |
|---|---|---|
| API | FastAPI | HTTP endpoints, SSE progress streaming |
| Ingestion | LlamaIndex | loaders, semantic chunking, metadata → Chroma |
| Retrieval | Haystack 2.x | BM25 + embedding retrievers → RRF → cross-encoder reranker |
| Vector store | ChromaDB | persistent, metadata-filterable |
| Cache / state | Redis | LLM + embedding cache, job status, rate limiting |
| LLM | Gemini (default) / OpenAI | behind one `LLMProvider` interface |
| Embeddings / reranking | sentence-transformers (local, CPU) or Gemini / OpenAI embeddings | `bge-small-en-v1.5`, `ms-marco-MiniLM-L-6-v2` |
| Packaging | Docker Compose | `api` + `redis` + `chroma` |

## Run

```bash
cp .env.example .env          # add GEMINI_API_KEY
docker compose up --build     # http://localhost:8080/health, docs at /docs
```

Local development:

```bash
uv sync                       # creates .venv from uv.lock
docker compose up redis chroma -d
make run                      # uvicorn with reload on :8080
make test                     # pytest, no API key needed
make lint
```

Load the sample corpus (10 KB notes, 3 Wikipedia PDFs, 5 URLs — see `sample_corpus/README.md`):

```bash
make ingest-sample
curl "http://localhost:8080/documents?source_type=kb&tag=consensus"
```

## Ingestion

| Endpoint | What it does |
|---|---|
| `POST /ingest/pdf` | multipart upload; one citable unit per page; tags / title / published_date optional |
| `POST /ingest/url` | fetch + boilerplate removal (trafilatura); per-URL errors don't fail the batch |
| `POST /ingest/kb` | re-scan `knowledge_base/`; one unit per markdown section; frontmatter = metadata |
| `GET /documents` | list with filters: `source_type`, `tag`, `date_from`, `date_to`, `q` |
| `DELETE /documents/{doc_id}` | remove all chunks of a document |

Ingestion is idempotent (content-hashed `doc_id`), chunks are produced by LlamaIndex's
`SemanticSplitterNodeParser` (split where sentence-embedding similarity drops), with a
fixed-size `SentenceSplitter` available via `CHUNKING_STRATEGY=fixed`. Compare them on a file:

```bash
make compare-chunking
```

Embeddings default to a local model (`BAAI/bge-small-en-v1.5`, CPU) so ingestion and search
need no API key; set `EMBEDDING_PROVIDER=gemini|openai` to use API embeddings instead.

## Hybrid retrieval

```
query ─┬─ BM25 (Haystack InMemoryBM25Retriever) ───────────────┐
       └─ query embedding → ChromaEmbeddingRetriever ──────────┴→ DocumentJoiner (RRF) → cross-encoder → top_k
```

`POST /search` runs the pipeline alone, with metadata filters (`source_types`, `tags`,
`doc_ids`, `date_from`/`date_to`) applied to both legs, and `config.mode`
(`bm25` | `dense` | `hybrid`) / `config.rerank` to pick a configuration.

Eval (`make eval-retrieval`; 35 questions over the sample corpus, 317 chunks, graded at the
level of the relevant markdown section / PDF document; laptop CPU; 2026-09-30):

| Configuration | Recall@5 | MRR@10 | mean latency (ms) | p50 latency (ms) |
|---|---|---|---|---|
| BM25 only | 0.871 | 0.797 | 4 | 4 |
| Dense only (bge-small) | 0.857 | 0.809 | 36 | 35 |
| Hybrid (RRF) | 0.914 | 0.790 | 53 | 53 |
| Hybrid + rerank (ms-marco-MiniLM-L-6-v2) | 0.929 | 0.871 | 1790 | 1835 |

Hybrid retrieval recovers what either leg misses; the cross-encoder fixes the ordering but
costs ~1.8 s per query on a CPU. Two of 35 questions were missed by every configuration
(on-topic PDF/web chunks outranked the labelled KB section). Small corpus: treat the ordering
of configurations, not the absolute numbers, as the result. Details in
[docs/LEARNING.md](docs/LEARNING.md#phase-3--hybrid-retrieval-haystack).

Run without Docker at all (embedded Chroma, local models):

```bash
make run-local
```

## Layout

```
mara/core       config, shared Chunk schema + Chroma layout, metadata filters, ChunkStore, cache
mara/llm        LLMProvider protocol, Gemini/OpenAI adapters, retry+rate-limit, cache decorator
mara/ingest     LlamaIndex loaders (PDF / HTML / markdown), semantic + fixed chunkers, pipeline
mara/api        FastAPI app + ingestion routes
mara/retrieval  Haystack pipelines: BM25 + dense → RRF → cross-encoder; own rrf.py; BM25 index
mara/agents     (phase 4) Planner, Researcher, Summarizer, Critic, Writer + orchestrator
knowledge_base/ internal KB: markdown notes with frontmatter (title, tags, published_date)
sample_corpus/  public-domain-ish PDFs + URL list for demos and evals
prompts/        one prompt file per agent
eval/           retrieval and answer-quality evals
scripts/        ingest_sample_corpus.py, compare_chunking.py
docs/           DECISIONS.md, LEARNING.md, ARCHITECTURE.md
```

See [docs/DECISIONS.md](docs/DECISIONS.md) for the why behind each choice and
[docs/LEARNING.md](docs/LEARNING.md) for a phase-by-phase walkthrough.
