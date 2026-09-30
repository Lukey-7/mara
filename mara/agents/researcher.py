"""Researcher: for every sub-question (concurrently) run hybrid retrieval over the internal
corpus, optionally after a live web search whose pages are fetched, cleaned and ingested on
the fly. The one agent with no prompt: evidence comes from retrieval, never from the model."""

import asyncio
import logging

from mara.agents.state import AgentStep, ResearchState, SubQuestion
from mara.agents.web_search import WebSearchError, WebSearchProvider
from mara.ingest.loaders import load_html
from mara.ingest.pipeline import IngestionPipeline
from mara.ingest.web import FetchError, fetch_html
from mara.retrieval.hybrid import RetrievalConfig, RetrievedChunk, Retriever

log = logging.getLogger(__name__)
WEB_SEARCH_TAG = "web-search"


class Researcher:
    name = "researcher"

    def __init__(
        self,
        retriever: Retriever,
        web_search: WebSearchProvider,
        ingest: IngestionPipeline | None,
        web_results: int = 3,
        fetch_timeout_s: float = 15.0,
        fetch_max_bytes: int = 5_000_000,
    ) -> None:
        self._retriever, self._web, self._ingest = retriever, web_search, ingest
        self._web_results = web_results
        self._fetch_timeout, self._fetch_max = fetch_timeout_s, fetch_max_bytes

    async def run(self, state: ResearchState, step: AgentStep) -> ResearchState:
        todo = [sq for sq in state.plan if sq.id not in state.evidence]  # loop: new ones only
        step.input_summary = f"{len(todo)} sub-questions, web_search={state.options.web_search}"
        results = await asyncio.gather(
            *(self._research_one(state, sq) for sq in todo), return_exceptions=True
        )
        summary = []
        for sq, res in zip(todo, results, strict=True):
            if isinstance(res, BaseException):
                state.evidence[sq.id] = []
                state.warnings.append(f"researcher: {sq.id} failed: {res}")
                summary.append(f"{sq.id}: error")
                continue
            hits, warnings = res
            state.evidence[sq.id] = hits
            state.warnings.extend(warnings)
            summary.append(f"{sq.id}: {len(hits)} chunks")
        step.output_summary = ", ".join(summary)
        return state

    async def _research_one(
        self, state: ResearchState, sq: SubQuestion
    ) -> tuple[list[RetrievedChunk], list[str]]:
        warnings: list[str] = []
        allowed = state.options.allowed_sources
        if "web" in sq.sources and "web" in allowed and state.options.web_search:
            warnings.extend(await self._web_ingest(sq))
        filters = sq.to_filter(state.options.filters, allowed)
        result = await self._retriever.retrieve(
            sq.text, filters, RetrievalConfig(top_k=state.options.top_k)
        )
        return result.chunks, warnings

    async def _web_ingest(self, sq: SubQuestion) -> list[str]:
        """search -> fetch -> trafilatura -> ingest. Every failure is a warning: the run
        continues on the internal corpus (graceful degradation)."""
        if self._ingest is None:
            return [f"{sq.id}: web search requested but ingestion is unavailable"]
        try:
            hits = await self._web.search(sq.text, self._web_results)
        except WebSearchError as e:
            return [f"{sq.id}: web search failed ({e}); using the internal corpus only"]
        if not hits:
            return [f"{sq.id}: web search returned no results"]

        async def one(url: str) -> str | None:
            try:
                html = await fetch_html(url, self._fetch_timeout, self._fetch_max)
                doc = await asyncio.to_thread(load_html, html, url, None, [WEB_SEARCH_TAG])
                if doc is None:
                    return f"{sq.id}: no article text at {url}"
                await self._ingest.ingest([doc])
                return None
            except FetchError as e:
                return f"{sq.id}: {e}"

        outcomes = await asyncio.gather(*(one(h.url) for h in hits))
        return [w for w in outcomes if w]
