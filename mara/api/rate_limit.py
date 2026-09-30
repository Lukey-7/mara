"""Per-client API rate limiting in Redis: a fixed-window counter (INCR + EXPIRE).

One key per (client, minute). The first request in a window creates the key with a TTL; every
request increments it; above the limit the API answers 429 with Retry-After. Fail-open: if
Redis is unreachable, requests pass (availability over strictness for a research tool).

Trade-off vs the token bucket used for outbound LLM calls: a fixed window allows up to 2x the
limit across a window boundary, but it is one atomic counter that every API replica shares,
which an in-process bucket is not.
"""

import logging
import time

import redis.asyncio as redis
from fastapi import HTTPException, Request

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, client: redis.Redis, limit: int, window_s: int = 60) -> None:
        self._r, self.limit, self.window_s = client, limit, window_s

    async def check(self, client_id: str) -> tuple[bool, int]:
        """(allowed, seconds until the window resets)"""
        now = time.time()
        key = f"mara:rl:{client_id}:{int(now // self.window_s)}"
        try:
            count = await self._r.incr(key)
            if count == 1:
                await self._r.expire(key, self.window_s)
        except (redis.RedisError, OSError) as e:
            log.warning("rate limiter unavailable, allowing request: %s", e)
            return True, 0
        return count <= self.limit, self.window_s - int(now % self.window_s)


async def rate_limit(request: Request) -> None:
    """FastAPI dependency. Only expensive, state-changing calls (POST) are limited; polling
    a job and the SSE stream are not."""
    limiter: RateLimiter | None = request.app.state.rate_limiter
    if limiter is None or request.method != "POST":
        return
    client_id = request.client.host if request.client else "unknown"
    allowed, retry_after = await limiter.check(client_id)
    if not allowed:
        raise HTTPException(
            429,
            f"rate limit exceeded: {limiter.limit} requests per {limiter.window_s}s",
            headers={"Retry-After": str(retry_after)},
        )
