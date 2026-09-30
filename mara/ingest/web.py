"""Fetch a URL as HTML with size/time limits. Extraction lives in loaders.load_html."""

import httpx

USER_AGENT = "MARA research assistant/0.1 (+https://github.com/Lukey-7/mara)"


class FetchError(Exception):
    pass


async def fetch_html(url: str, timeout_s: float, max_bytes: int) -> str:
    if not url.lower().startswith(("http://", "https://")):
        raise FetchError(f"unsupported URL scheme: {url}")
    try:
        async with (
            httpx.AsyncClient(
                follow_redirects=True, timeout=timeout_s, headers={"User-Agent": USER_AGENT}
            ) as client,
            client.stream("GET", url) as resp,
        ):
            if resp.status_code >= 400:
                raise FetchError(f"HTTP {resp.status_code} for {url}")
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "xml" not in ctype and ctype:
                raise FetchError(f"not an HTML page ({ctype}): {url}")
            buf = bytearray()
            async for part in resp.aiter_bytes():
                buf.extend(part)
                if len(buf) > max_bytes:
                    raise FetchError(f"page larger than {max_bytes} bytes: {url}")
            return buf.decode(resp.encoding or "utf-8", errors="replace")
    except httpx.HTTPError as e:
        raise FetchError(f"{type(e).__name__} fetching {url}: {e}") from e
