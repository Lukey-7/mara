# ---- Stage 1: build the virtualenv with uv (build tools stay in this stage) ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first, source second: editing code does not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY mara ./mara
RUN uv sync --frozen --no-dev --no-editable

# ---- Stage 2: slim runtime with only the venv + runtime files ----
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY prompts ./prompts
COPY knowledge_base ./knowledge_base
COPY sample_corpus ./sample_corpus
COPY scripts ./scripts
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=2)"
CMD ["uvicorn", "mara.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
