"""WebSearchProvider: find candidate pages for a query. Fetching + cleaning + ingestion happen
in the Researcher, so a provider only has to return (title, url, snippet)."""

import asyncio
import logging
from typing import Protocol

import httpx
from pydantic import BaseModel

log = logging.getLogger(__name__)


class WebResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchError(Exception):
    pass


class WebSearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[WebResult]: ...


class NoopWebSearch:
    """Default: the internal corpus only."""

    name = "none"

    async def search(self, query: str, max_results: int) -> list[WebResult]:
        return []


class DuckDuckGoWebSearch:
    """Free, no key (the `ddgs` package). Rate-limited and occasionally blocked; the
    Researcher treats failures as a warning, never an error."""

    name = "duckduckgo"

    async def search(self, query: str, max_results: int) -> list[WebResult]:
        from ddgs import DDGS

        def _run() -> list[dict]:
            return DDGS().text(query, max_results=max_results) or []

        try:
            rows = await asyncio.wait_for(asyncio.to_thread(_run), timeout=20)
        except Exception as e:  # noqa: BLE001 - library raises many ad-hoc exception types
            raise WebSearchError(f"duckduckgo: {type(e).__name__}: {e}") from e
        return [
            WebResult(title=r.get("title", ""), url=r["href"], snippet=r.get("body", ""))
            for r in rows
            if r.get("href")
        ]


class TavilyWebSearch:
    """Tavily search API (free tier available). Plain httpx: the endpoint is one POST."""

    name = "tavily"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    async def search(self, query: str, max_results: int) -> list[WebResult]:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": self._key, "query": query, "max_results": max_results},
                )
                resp.raise_for_status()
        except httpx.HTTPError as e:
            raise WebSearchError(f"tavily: {type(e).__name__}: {e}") from e
        return [
            WebResult(title=r.get("title", ""), url=r["url"], snippet=r.get("content", ""))
            for r in resp.json().get("results", [])
            if r.get("url")
        ]
