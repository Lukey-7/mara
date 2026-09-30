"""Provider adapters tested against stub SDK clients: no network, no API keys."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from google.genai import errors
from pydantic import BaseModel

from mara.core.config import Settings
from mara.llm.base import LLMError, LLMProvider, RetryableLLMError
from mara.llm.cached import CachedLLM
from mara.llm.composite import CompositeProvider
from mara.llm.factory import build_base_provider, build_llm, llm_configuration_error
from mara.llm.gemini import GeminiProvider
from mara.llm.openai_provider import OpenAIProvider
from mara.llm.resilience import ResilientLLM


class Plan(BaseModel):
    sub_questions: list[str]


def async_returning(value=None, exc=None, sink=None):
    async def fn(**kwargs):
        if sink is not None:
            sink.append(kwargs)
        if exc:
            raise exc
        return value

    return fn


# ---------------- Gemini ----------------


def gemini_with(generate=None, embed=None) -> GeminiProvider:
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=generate, embed_content=embed,
    )))  # fmt: skip
    return GeminiProvider("k", "gemini-test", "emb-test", client=client)


async def test_gemini_generate_requests_json_and_reports_tokens():
    sink: list[dict] = []
    resp = SimpleNamespace(
        text='{"sub_questions": ["a"]}',
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
    )
    p = gemini_with(generate=async_returning(resp, sink=sink))

    out = await p.generate("plan this", system="you plan", json_schema=Plan)

    cfg = sink[0]["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema["properties"]["sub_questions"]["type"] == "array"
    assert cfg.system_instruction == "you plan"
    assert (out.input_tokens, out.output_tokens, out.model) == (12, 7, "gemini-test")
    assert Plan.model_validate_json(out.text).sub_questions == ["a"]


@pytest.mark.parametrize("code,expected", [(429, RetryableLLMError), (503, RetryableLLMError),
                                           (400, LLMError)])  # fmt: skip
async def test_gemini_errors_are_translated(code, expected):
    err = errors.APIError(code, {"error": {"message": "boom", "status": "X"}})
    p = gemini_with(generate=async_returning(exc=err))
    with pytest.raises(expected) as info:
        await p.generate("q")
    assert type(info.value) is expected


async def test_gemini_embed_returns_vectors_in_order_with_task_type():
    sink: list[dict] = []
    resp = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0]),
                                       SimpleNamespace(values=[3.0, 4.0])])  # fmt: skip
    p = gemini_with(embed=async_returning(resp, sink=sink))
    p.embedding_dimensions = 2
    assert await p.embed(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert sink[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert sink[0]["config"].output_dimensionality == 2
    await p.embed(["q"], kind="query")
    assert sink[1]["config"].task_type == "RETRIEVAL_QUERY"


# ---------------- OpenAI ----------------


def openai_with(create=None, embed=None) -> OpenAIProvider:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        embeddings=SimpleNamespace(create=embed),
    )
    return OpenAIProvider("k", "gpt-test", "emb-test", client=client)


async def test_openai_generate_builds_messages_and_schema():
    sink: list[dict] = []
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"sub_questions": []}'))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=3),
    )
    p = openai_with(create=async_returning(resp, sink=sink))

    out = await p.generate("q", system="sys", max_tokens=50, json_schema=Plan)

    kw = sink[0]
    assert kw["messages"] == [{"role": "system", "content": "sys"},
                              {"role": "user", "content": "q"}]  # fmt: skip
    assert kw["max_completion_tokens"] == 50
    assert "temperature" not in kw  # omitted unless explicitly set
    assert kw["response_format"]["json_schema"]["name"] == "Plan"
    assert (out.input_tokens, out.output_tokens) == (20, 3)


async def test_openai_rate_limit_is_retryable():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    err = openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)
    p = openai_with(create=async_returning(exc=err))
    with pytest.raises(RetryableLLMError):
        await p.generate("q")


async def test_openai_embed_sorts_by_index_and_passes_dimensions():
    sink: list[dict] = []
    resp = SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[2.0]),
                                 SimpleNamespace(index=0, embedding=[1.0])])  # fmt: skip
    p = openai_with(embed=async_returning(resp, sink=sink))
    assert await p.embed(["a", "b"]) == [[1.0], [2.0]]
    assert "dimensions" not in sink[0]
    p.embedding_dimensions = 256
    await p.embed(["a"], kind="query")
    assert sink[1]["dimensions"] == 256


# ---------------- Factory ----------------


def settings(**kw) -> Settings:
    kw.setdefault("embedding_provider", "gemini")  # never load the local model in unit tests
    return Settings(_env_file=None, **kw)


def test_factory_picks_provider_from_config():
    g = build_base_provider(settings(llm_provider="gemini", gemini_api_key="k"))
    o = build_base_provider(
        settings(llm_provider="openai", openai_api_key="k", embedding_provider="openai")
    )
    assert isinstance(g, CompositeProvider) and isinstance(g, LLMProvider)
    assert isinstance(g._generator, GeminiProvider) and g._embedder is g._generator
    assert isinstance(o._generator, OpenAIProvider) and o.embedding_provider == "openai"


def test_mixed_vendors_build_a_separate_embedder():
    p = build_base_provider(
        settings(llm_provider="gemini", gemini_api_key="k", openai_api_key="k2",
                 embedding_provider="openai")
    )  # fmt: skip
    assert isinstance(p._generator, GeminiProvider)
    assert isinstance(p._embedder, OpenAIProvider)
    assert (p.name, p.embedding_provider) == ("gemini", "openai")


async def test_missing_llm_key_still_embeds_but_cannot_generate():
    s = settings(llm_provider="gemini", gemini_api_key=None, openai_api_key="k",
                 embedding_provider="openai")  # fmt: skip
    assert "GEMINI_API_KEY" in llm_configuration_error(s)
    p = build_base_provider(s)
    assert p.has_generator is False and isinstance(p._embedder, OpenAIProvider)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        await p.generate("q")


def test_missing_embedding_key_fails_clearly():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_base_provider(settings(gemini_api_key="k", embedding_provider="openai"))


def test_factory_stacks_cache_outside_resilience():
    import fakeredis

    llm = build_llm(settings(gemini_api_key="k"), fakeredis.FakeAsyncRedis())
    assert isinstance(llm, CachedLLM)
    assert isinstance(llm.inner, ResilientLLM)
    assert isinstance(llm.inner.inner, CompositeProvider)
    assert llm.inner._embedding_limiter is not None  # API embeddings are rate limited
