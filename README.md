# MARA — Multi-Agent Research Assistant

A research assistant that **plans** a question into sub-questions, **retrieves** evidence with
hybrid search (BM25 + dense + reranking) over PDFs, web pages and an internal knowledge base,
**summarizes** it into verifiable evidence notes, **checks** coverage, and **writes** a
citation-backed answer. Every run produces a trace of what each agent did.

> Status: **Phase 1 of 6** (skeleton). Ingestion, retrieval, agents, evals and UI follow.
> Numbers in this README will only ever come from `eval/`.

## Stack

| Layer | Tool | Job |
|---|---|---|
| API | FastAPI | HTTP endpoints, SSE progress streaming |
| Ingestion | LlamaIndex | loaders, semantic chunking, metadata → Chroma |
| Retrieval | Haystack 2.x | BM25 + embedding retrievers → RRF → cross-encoder reranker |
| Vector store | ChromaDB | persistent, metadata-filterable |
| Cache / state | Redis | LLM + embedding cache, job status, rate limiting |
| LLM | Gemini (default) / OpenAI | behind one `LLMProvider` interface |
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

## Layout

```
mara/core       config, shared chunk schema, Redis cache primitives
mara/llm        LLMProvider protocol, Gemini/OpenAI adapters, retry+rate-limit, cache decorator
mara/api        FastAPI app
mara/ingest     (phase 2) LlamaIndex loaders + semantic chunking
mara/retrieval  (phase 3) Haystack hybrid retrieval pipeline
mara/agents     (phase 4) Planner, Researcher, Summarizer, Critic, Writer + orchestrator
prompts/        one prompt file per agent
eval/           retrieval and answer-quality evals
docs/           DECISIONS.md, LEARNING.md, ARCHITECTURE.md
```

See [docs/DECISIONS.md](docs/DECISIONS.md) for the why behind each choice and
[docs/LEARNING.md](docs/LEARNING.md) for a phase-by-phase walkthrough.
