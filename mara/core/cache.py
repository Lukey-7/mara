"""Redis-backed cache primitives: deterministic key building + a fail-open byte cache."""

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as redis

log = logging.getLogger(__name__)


def make_cache_key(version: str, namespace: str, provider: str, model: str, **parts: Any) -> str:
    """`mara:<version>:<namespace>:<provider>:<model>:<sha256 of the other parts>`.

    Everything that can change the output must be in the key (prompt, system prompt,
    temperature, max tokens, output schema). Parts are serialized as canonical JSON (sorted
    keys, fixed separators) so the same inputs always hash to the same key. The readable
    prefix lets us inspect or bulk-delete keys per model; the version lets us invalidate
    everything at once.
    """
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"mara:{version}:{namespace}:{provider}:{model}:{digest}"


class RedisCache:
    """Thin wrapper over redis that never lets a cache failure break a request.

    If Redis is down, reads count as misses and writes are skipped (fail-open): the system
    gets slower and more expensive, but it keeps working.
    """

    def __init__(self, client: redis.Redis) -> None:
        self.client = client

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        if not keys:
            return []
        try:
            return await self.client.mget(keys)
        except redis.RedisError as e:
            log.warning("cache read failed, treating as miss: %s", e)
            return [None] * len(keys)

    async def set_many(self, items: dict[str, bytes], ttl_s: int) -> None:
        if not items:
            return
        try:
            async with self.client.pipeline(transaction=False) as pipe:
                for key, value in items.items():
                    pipe.set(key, value, ex=ttl_s)
                await pipe.execute()
        except redis.RedisError as e:
            log.warning("cache write failed, skipping: %s", e)

    async def get(self, key: str) -> bytes | None:
        return (await self.get_many([key]))[0]

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        await self.set_many({key: value}, ttl_s)
