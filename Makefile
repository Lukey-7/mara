.PHONY: install test lint fmt run up down logs ingest-sample compare-chunking

install:        ## create .venv with all deps (incl. dev) from uv.lock
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

run:            ## API on the host (needs redis + chroma: `make up` or `docker compose up redis chroma`)
	uv run uvicorn mara.api.main:app --reload --port 8080

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

ingest-sample:  ## load knowledge_base/ + sample_corpus/ through the running API
	uv run python scripts/ingest_sample_corpus.py

compare-chunking:  ## semantic vs fixed-size chunking on one KB note (needs an API key)
	uv run python scripts/compare_chunking.py knowledge_base/01-raft.md
