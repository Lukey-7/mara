import fakeredis
import pytest

from mara.agents.factory import build_web_search
from mara.agents.web_search import CachedWebSearch, NoopWebSearch, WebResult, WebSearchError
from mara.core.cache import RedisCache, redis_cached
from mara.core.config import Settings


@pytest.fixture
def cache() -> RedisCache:
    return RedisCache(fakeredis.FakeAsyncRedis())


async def test_redis_cached_decorator_caches_by_arguments(cache):
    calls = []

    @redis_cached(cache, namespace="t", ttl_s=60)
    async def slow_square(x: int, *, offset: int = 0) -> dict:
        calls.append((x, offset))
        return {"value": x * x + offset}

    assert await slow_square(3) == {"value": 9}
    assert await slow_square(3) == {"value": 9}  # hit
    assert await slow_square(3, offset=1) == {"value": 10}  # different kwargs: miss
    assert await slow_square(4) == {"value": 16}
    assert calls == [(3, 0), (3, 1), (4, 0)]
    assert slow_square.__name__ == "slow_square"  # functools.wraps


async def test_decorator_sets_a_ttl_and_does_not_cache_failures(cache):
    attempts = 0

    @redis_cached(cache, namespace="t", ttl_s=60)
    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError):
        await flaky()
    assert await flaky() == "ok" and await flaky() == "ok"
    assert attempts == 2
    [key] = await cache.client.keys("mara:v1:t:*")
    assert 0 < await cache.client.ttl(key) <= 60


class CountingSearch:
    name = "counting"

    def __init__(self, fail: bool = False) -> None:
        self.calls, self.fail = 0, fail

    async def search(self, query: str, max_results: int) -> list[WebResult]:
        self.calls += 1
        if self.fail:
            raise WebSearchError("rate limited")
        return [WebResult(title=f"{query} result", url="https://example.org/a", snippet="s")]


async def test_cached_web_search_hits_the_provider_once_per_query(cache):
    inner = CountingSearch()
    search = CachedWebSearch(inner, cache, ttl_s=60)

    first = await search.search("raft", 3)
    second = await search.search("raft", 3)
    await search.search("paxos", 3)

    assert first == second and isinstance(second[0], WebResult)
    assert inner.calls == 2 and search.name == "counting"

    failing = CachedWebSearch(CountingSearch(fail=True), cache, ttl_s=60)
    with pytest.raises(WebSearchError):
        await failing.search("x", 1)


def test_build_web_search_picks_provider_and_wraps_with_cache(cache):
    none = build_web_search(Settings(_env_file=None), cache)
    assert isinstance(none, NoopWebSearch)
    ddg = build_web_search(Settings(_env_file=None, web_search_provider="duckduckgo"), cache)
    assert isinstance(ddg, CachedWebSearch) and ddg.name == "duckduckgo"
    plain = build_web_search(Settings(_env_file=None, web_search_provider="duckduckgo"), None)
    assert plain.name == "duckduckgo" and not isinstance(plain, CachedWebSearch)
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        build_web_search(Settings(_env_file=None, web_search_provider="tavily"), cache)
