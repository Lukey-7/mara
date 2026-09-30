import pytest

from mara.llm.base import LLMError, RetryableLLMError
from mara.llm.resilience import AsyncRateLimiter, ResilientLLM, retry_async
from tests.fakes import FakeLLM


class FakeTime:
    """Deterministic clock: sleep() advances time instantly and records the delay."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


async def test_retry_succeeds_after_transient_failures_with_exponential_backoff():
    t = FakeTime()
    outcomes = [RetryableLLMError("429"), RetryableLLMError("503"), "ok"]

    async def fn():
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    result = await retry_async(
        fn, max_retries=4, base_delay=1.0, max_delay=30, sleep=t.sleep, rng=lambda: 1.0
    )
    assert result == "ok"
    assert t.sleeps == [1.0, 2.0]  # base * 2^attempt (rng=1.0 means no jitter reduction)


async def test_backoff_is_capped_and_jittered():
    t = FakeTime()

    async def always_fail():
        raise RetryableLLMError("429")

    with pytest.raises(RetryableLLMError):
        await retry_async(
            always_fail, max_retries=5, base_delay=1.0, max_delay=4.0, sleep=t.sleep,
            rng=lambda: 0.0,
        )  # fmt: skip
    # rng=0 -> 50% of the capped delay: 1,2,4,4,4 -> 0.5,1,2,2,2
    assert t.sleeps == [0.5, 1.0, 2.0, 2.0, 2.0]


async def test_non_retryable_error_is_raised_immediately():
    t = FakeTime()
    calls = 0

    async def bad_request():
        nonlocal calls
        calls += 1
        raise LLMError("400 invalid argument")

    with pytest.raises(LLMError):
        await retry_async(bad_request, max_retries=4, base_delay=1, max_delay=30, sleep=t.sleep)
    assert calls == 1 and t.sleeps == []


async def test_rate_limiter_allows_burst_then_spaces_requests():
    t = FakeTime()
    limiter = AsyncRateLimiter(per_minute=60, burst=2, clock=t.clock, sleep=t.sleep)  # 1 req/s

    for _ in range(4):
        await limiter.acquire()

    assert t.sleeps == pytest.approx([1.0, 1.0])  # 2 free (burst), then one per second


async def test_resilient_llm_retries_and_takes_a_token_per_attempt():
    t = FakeTime()
    inner = FakeLLM(replies=[RetryableLLMError("429"), "answer"])
    limiter = AsyncRateLimiter(per_minute=600, burst=10, clock=t.clock, sleep=t.sleep)
    llm = ResilientLLM(inner, limiter, max_retries=3, base_delay=0.01, max_delay=1)

    resp = await llm.generate("q")

    assert resp.text == "answer"
    assert len(inner.generate_calls) == 2
    assert limiter.tokens == pytest.approx(8, abs=0.1)  # two attempts consumed two tokens
