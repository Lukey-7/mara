import fakeredis
import pytest
import redis.asyncio as redis
from pydantic import BaseModel

from mara.core.cache import RedisCache, make_cache_key
from mara.llm.base import LLMProvider
from mara.llm.cached import CachedLLM
from tests.fakes import FakeLLM


@pytest.fixture
def cache() -> RedisCache:
    return RedisCache(fakeredis.FakeAsyncRedis())


def make(inner: FakeLLM, cache: RedisCache) -> CachedLLM:
    return CachedLLM(inner, cache, llm_ttl_s=60, embedding_ttl_s=60)


class Answer(BaseModel):
    answer: str


def test_key_is_deterministic_and_order_independent():
    a = make_cache_key("v1", "llm", "gemini", "m", prompt="p", temperature=0.2)
    b = make_cache_key("v1", "llm", "gemini", "m", temperature=0.2, prompt="p")
    assert a == b
    assert a.startswith("mara:v1:llm:gemini:m:")


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "other"},
        {"system": "be terse"},
        {"temperature": 0.9},
        {"max_tokens": 100},
    ],
)
def test_key_changes_when_any_output_affecting_input_changes(change):
    base = dict(prompt="p", system=None, temperature=0.2, max_tokens=None)
    assert make_cache_key("v1", "llm", "g", "m", **base) != make_cache_key(
        "v1", "llm", "g", "m", **{**base, **change}
    )


def test_key_changes_with_model_provider_and_version():
    k = make_cache_key("v1", "llm", "gemini", "m1", prompt="p")
    assert k != make_cache_key("v1", "llm", "gemini", "m2", prompt="p")
    assert k != make_cache_key("v1", "llm", "openai", "m1", prompt="p")
    assert k != make_cache_key("v2", "llm", "gemini", "m1", prompt="p")


def test_cached_llm_satisfies_protocol(cache):
    assert isinstance(make(FakeLLM(), cache), LLMProvider)


async def test_second_identical_call_is_a_cache_hit(cache):
    inner = FakeLLM(replies=["first"])
    llm = make(inner, cache)

    r1 = await llm.generate("what is raft?", temperature=0.0)
    r2 = await llm.generate("what is raft?", temperature=0.0)

    assert len(inner.generate_calls) == 1
    assert (r1.cached, r2.cached) == (False, True)
    assert r2.text == "first" and r2.input_tokens == r1.input_tokens
    assert (llm.hits, llm.misses) == (1, 1)


async def test_different_schema_is_a_different_entry(cache):
    inner = FakeLLM()
    llm = make(inner, cache)
    await llm.generate("q")
    await llm.generate("q", json_schema=Answer)
    assert len(inner.generate_calls) == 2


async def test_embeddings_are_cached_per_text(cache):
    inner = FakeLLM()
    llm = make(inner, cache)

    first = await llm.embed(["alpha", "beta"])
    second = await llm.embed(["beta", "gamma", "alpha"])

    assert inner.embed_calls == [["alpha", "beta"], ["gamma"]]  # only the new text hit the API
    assert second[0] == first[1] and second[2] == first[0]  # order preserved
    assert (llm.embedding_hits, llm.embedding_misses) == (2, 3)


class BrokenRedis(fakeredis.FakeAsyncRedis):
    async def mget(self, *a, **kw):
        raise redis.ConnectionError("down")

    def pipeline(self, *a, **kw):
        raise redis.ConnectionError("down")


async def test_cache_fails_open_when_redis_is_down():
    inner = FakeLLM(replies=["ok", "ok again"])
    llm = make(inner, RedisCache(BrokenRedis()))

    assert (await llm.generate("q")).text == "ok"
    assert (await llm.generate("q")).text == "ok again"  # no crash, just no caching
    assert len(inner.generate_calls) == 2
