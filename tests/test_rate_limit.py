import fakeredis
import redis.asyncio as redis

from mara.api.rate_limit import RateLimiter


async def test_fixed_window_counts_per_client_and_sets_ttl():
    r = fakeredis.FakeAsyncRedis()
    limiter = RateLimiter(r, limit=2, window_s=60)

    assert (await limiter.check("alice"))[0] is True
    assert (await limiter.check("alice"))[0] is True
    allowed, retry_after = await limiter.check("alice")
    assert allowed is False and 0 < retry_after <= 60
    assert (await limiter.check("bob"))[0] is True  # separate counter per client

    [key] = await r.keys("mara:rl:alice:*")
    assert 0 < await r.ttl(key) <= 60  # the window cleans itself up


async def test_limiter_fails_open_when_redis_is_down():
    class Down(fakeredis.FakeAsyncRedis):
        async def incr(self, *a, **kw):
            raise redis.ConnectionError("down")

    assert await RateLimiter(Down(), limit=1).check("alice") == (True, 0)


def test_api_returns_429_with_retry_after_on_posts_only(make_client):
    with make_client(api_rate_limit_per_minute=2) as c:
        body = {"query": "raft"}
        assert c.post("/search", json=body).status_code == 200
        assert c.post("/search", json=body).status_code == 200
        r = c.post("/search", json=body)
        assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
        assert "rate limit exceeded" in r.json()["detail"]
        assert c.get("/health").status_code == 200  # reads are not limited
        assert c.get("/documents").status_code == 200


def test_rate_limit_can_be_disabled(make_client):
    with make_client(api_rate_limit_per_minute=0) as c:
        assert all(c.post("/search", json={"query": "x"}).status_code == 200 for _ in range(5))
