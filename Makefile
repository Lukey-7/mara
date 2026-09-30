.PHONY: install test lint fmt run up down logs

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
