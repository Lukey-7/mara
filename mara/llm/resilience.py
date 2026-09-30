"""Rate limiting + retry with exponential backoff, applied to every provider call.

`ResilientLLM` wraps any LLMProvider (Decorator pattern) so the policy lives in one place
instead of being copy-pasted into each vendor implementation.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from mara.llm.base import LLMProvider, LLMResponse, RetryableLLMError

log = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Token bucket: holds up to `burst` tokens, refilled at `per_minute / 60` tokens/second.
    Each request takes one token; with an empty bucket, the caller sleeps until one refills."""

    def __init__(
        self,
        per_minute: float,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.rate = per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(1, int(per_minute)))
        self.tokens = self.capacity
        self._clock, self._sleep = clock, sleep
        self._updated = clock()
        self._lock = asyncio.Lock()  # waiters queue on the lock, so they are served in order

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                self.tokens = min(self.capacity, self.tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await self._sleep((1 - self.tokens) / self.rate)


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call `fn`; on RetryableLLMError wait base * 2^attempt (capped, with jitter) and retry.
    Non-retryable errors propagate immediately."""
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except RetryableLLMError as e:
            if attempt == max_retries:
                raise
            # Jitter in [50%, 100%] of the delay, so parallel callers don't retry in lockstep.
            delay = min(max_delay, base_delay * 2**attempt) * (0.5 + rng() / 2)
            log.warning(
                "retryable LLM error (attempt %d): %s; sleeping %.1fs", attempt + 1, e, delay
            )
            await sleep(delay)
    raise AssertionError("unreachable")


class ResilientLLM:
    """LLMProvider decorator: every call acquires a rate-limit token, then retries on
    transient failures. Each retry is a new request, so it takes a new token."""

    def __init__(
        self,
        inner: LLMProvider,
        limiter: AsyncRateLimiter,
        max_retries: int,
        base_delay: float,
        max_delay: float,
    ) -> None:
        self.inner = inner
        self.name, self.model, self.embedding_model = inner.name, inner.model, inner.embedding_model
        self._limiter = limiter
        self._retry = dict(max_retries=max_retries, base_delay=base_delay, max_delay=max_delay)

    async def _call[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        async def attempt() -> T:
            await self._limiter.acquire()
            return await fn()

        return await retry_async(attempt, **self._retry)

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        return await self._call(
            lambda: self.inner.generate(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._call(lambda: self.inner.embed(texts))
