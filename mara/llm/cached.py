"""CachedLLM: an LLMProvider decorator that caches completions and embeddings in Redis.

It sits OUTSIDE ResilientLLM, so a cache hit never waits for a rate-limit token.
"""

from array import array

from pydantic import BaseModel

from mara.core.cache import RedisCache, make_cache_key
from mara.llm.base import EmbedKind, LLMProvider, LLMResponse


class CachedLLM:
    def __init__(
        self,
        inner: LLMProvider,
        cache: RedisCache,
        llm_ttl_s: int,
        embedding_ttl_s: int,
        key_version: str = "v1",
    ) -> None:
        self.inner = inner
        self.name, self.model, self.embedding_model = inner.name, inner.model, inner.embedding_model
        self.embedding_dimensions = inner.embedding_dimensions
        self._cache = cache
        self._llm_ttl, self._emb_ttl, self._version = llm_ttl_s, embedding_ttl_s, key_version
        # Counters the trace reads (per process; the per-run numbers come from LLMResponse.cached).
        self.hits = self.misses = self.embedding_hits = self.embedding_misses = 0

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        key = make_cache_key(
            self._version, "llm", self.name, self.model,
            prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens,
            schema=json_schema.model_json_schema() if json_schema else None,
        )  # fmt: skip
        if (raw := await self._cache.get(key)) is not None:
            self.hits += 1
            return LLMResponse.model_validate_json(raw).model_copy(update={"cached": True})

        self.misses += 1
        resp = await self.inner.generate(
            prompt, system=system, temperature=temperature,
            max_tokens=max_tokens, json_schema=json_schema,
        )  # fmt: skip
        await self._cache.set(key, resp.model_dump_json().encode(), self._llm_ttl)
        return resp

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        """Per-text caching: a batch where 90 of 100 texts are cached calls the API for 10."""
        keys = [
            make_cache_key(
                self._version,
                "emb",
                self.name,
                self.embedding_model,
                text=t,
                kind=kind,
                dims=self.embedding_dimensions,
            )  # fmt: skip
            for t in texts
        ]
        cached = await self._cache.get_many(keys)
        result: list[list[float] | None] = [_decode(raw) if raw else None for raw in cached]

        missing = [i for i, v in enumerate(result) if v is None]
        self.embedding_hits += len(texts) - len(missing)
        self.embedding_misses += len(missing)
        if missing:
            fresh = await self.inner.embed([texts[i] for i in missing], kind=kind)
            for i, vec in zip(missing, fresh, strict=True):
                result[i] = vec
            await self._cache.set_many(
                {keys[i]: _encode(vec) for i, vec in zip(missing, fresh, strict=True)},
                self._emb_ttl,
            )
        return result  # type: ignore[return-value]  # every slot is filled by now


# Vectors are stored as packed float32 (4 bytes/dim) instead of JSON (~20 bytes/dim).
def _encode(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def _decode(raw: bytes) -> list[float]:
    return array("f", raw).tolist()
