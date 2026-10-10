"""Web access tools: search and page fetching.

Search providers (``AMY_SEARCH_PROVIDER``):
- ``searxng``     self-hosted SearXNG (recommended); needs ``AMY_SEARXNG_URL`` with JSON enabled
- ``brave``       Brave Search API; needs ``AMY_BRAVE_API_KEY`` (free tier available)
- ``duckduckgo``  no setup; scrapes DuckDuckGo's HTML endpoint, so best-effort
- ``auto``        the first configured of the above (default)

``fetch_url`` refuses private/LAN addresses unless ``AMY_WEB_ALLOW_PRIVATE`` is set, so a
malicious page can't steer the model into probing the local network.
"""

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from app.config import Settings

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36 AmyAI/1.0"
)
MAX_FETCH_BYTES = 5 * 1024 * 1024
MAX_PAGE_CHARS = 15_000


class WebError(Exception):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def provider_name(settings: Settings) -> str:
    choice = settings.search_provider
    if choice != "auto":
        return choice
    if settings.searxng_url:
        return "searxng"
    if settings.brave_api_key:
        return "brave"
    return "duckduckgo"


async def search(
    query: str, settings: Settings, client: httpx.AsyncClient, limit: int = 8
) -> list[SearchResult]:
    provider = provider_name(settings)
    try:
        if provider == "searxng":
            results = await _searxng(query, settings, client)
        elif provider == "brave":
            results = await _brave(query, settings, client)
        else:
            results = await _duckduckgo(query, client)
    except httpx.HTTPError as exc:
        raise WebError(f"Search ({provider}) failed: {exc}") from exc
    return results[:limit]


async def _searxng(query: str, settings: Settings, client: httpx.AsyncClient) -> list[SearchResult]:
    if not settings.searxng_url:
        raise WebError("AMY_SEARXNG_URL is not set")
    resp = await client.get(
        settings.searxng_url.rstrip("/") + "/search",
        params={"q": query, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    if resp.status_code == 403:
        raise WebError("SearXNG refused JSON output; enable 'json' under search.formats")
    resp.raise_for_status()
    return [
        SearchResult(r.get("title", ""), r["url"], r.get("content", ""))
        for r in resp.json().get("results", [])
        if r.get("url")
    ]


async def _brave(query: str, settings: Settings, client: httpx.AsyncClient) -> list[SearchResult]:
    if not settings.brave_api_key:
        raise WebError("AMY_BRAVE_API_KEY is not set")
    resp = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query},
        headers={"X-Subscription-Token": settings.brave_api_key, "Accept": "application/json"},
        timeout=20,
    )
    resp.raise_for_status()
    return [
        SearchResult(r.get("title", ""), r["url"], r.get("description", ""))
        for r in resp.json().get("web", {}).get("results", [])
    ]


async def _duckduckgo(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    resp = await client.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    resp.raise_for_status()
    results = parse_duckduckgo(resp.text)
    if not results and "anomaly" in resp.text.lower():
        raise WebError("DuckDuckGo is rate-limiting this server; configure SearXNG or Brave")
    return results


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._field: str | None = None
        self._buf: list[str] = []
        self._href = ""

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if "result__a" in classes:
            self._field, self._buf, self._href = "title", [], a.get("href") or ""
        elif "result__snippet" in classes and self.results:
            self._field, self._buf = "snippet", []

    def handle_data(self, data):
        if self._field:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag != "a" or not self._field:
            return
        text = " ".join("".join(self._buf).split())
        if self._field == "title":
            self.results.append(SearchResult(text, _ddg_url(self._href), ""))
        else:
            self.results[-1].snippet = text
        self._field = None


def _ddg_url(href: str) -> str:
    # Older markup wraps links as //duckduckgo.com/l/?uddg=<encoded target>.
    parsed = urlparse(href)
    if parsed.path.startswith("/l/") and "uddg" in parse_qs(parsed.query):
        return parse_qs(parsed.query)["uddg"][0]
    return href


def parse_duckduckgo(html: str) -> list[SearchResult]:
    parser = _DDGParser()
    parser.feed(html)
    return [r for r in parser.results if r.url.startswith("http")]


# ---- fetching ----


def is_public_address(host: str) -> bool:
    """True if every address the host resolves to is publicly routable."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise WebError(f"Can't resolve {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            return False
    return True


async def _check_url(url: str, settings: Settings) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise WebError("Only http(s) URLs can be fetched")
    if settings.web_allow_private:
        return
    if not await asyncio.to_thread(is_public_address, parsed.hostname):
        raise WebError(
            f"Refusing to fetch {parsed.hostname}: it is a private/local address "
            "(set AMY_WEB_ALLOW_PRIVATE=true to allow)"
        )


async def fetch(url: str, settings: Settings, client: httpx.AsyncClient) -> str:
    """Fetch a URL and return readable text (main content for HTML pages)."""
    for _ in range(5):  # follow redirects manually so every hop is checked
        await _check_url(url, settings)
        try:
            async with client.stream(
                "GET", url, headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=False
            ) as resp:
                if resp.is_redirect and "location" in resp.headers:
                    url = urljoin(url, resp.headers["location"])
                    continue
                if resp.status_code >= 400:
                    raise WebError(f"HTTP {resp.status_code} from {url}")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_FETCH_BYTES:
                        break
                ctype = resp.headers.get("content-type", "")
                encoding = resp.encoding or "utf-8"
        except httpx.HTTPError as exc:
            raise WebError(f"Couldn't fetch {url}: {exc}") from exc
        return truncate(await asyncio.to_thread(to_text, bytes(body), ctype, encoding), url)
    raise WebError("Too many redirects")


def to_text(body: bytes, content_type: str, encoding: str) -> str:
    if "pdf" in content_type:
        return _pdf_text(body)
    text = body.decode(encoding, errors="replace")
    if "html" not in content_type and not text.lstrip()[:15].lower().startswith(
        ("<!doctype", "<html")
    ):
        return text
    import trafilatura

    extracted = trafilatura.extract(
        text, output_format="markdown", include_links=True, include_tables=True, favor_recall=True
    )
    return extracted or "(No readable text found on this page.)"


def _pdf_text(body: bytes) -> str:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(body))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages[:50])


def truncate(text: str, url: str) -> str:
    if len(text) <= MAX_PAGE_CHARS:
        return text
    return (
        text[:MAX_PAGE_CHARS] + f"\n\n[Truncated: page is {len(text):,} characters. Source: {url}]"
    )


EXCERPT_CHARS = 2500


async def search_with_excerpts(
    query: str, settings: Settings, client: httpx.AsyncClient, read_top: int = 3
) -> str:
    """Search, then read the top results in parallel and include page excerpts.

    Snippets alone are short and often stale, and smaller models tend to answer from them
    instead of calling fetch_url, so the most relevant pages are read up front.
    """
    results = await search(query, settings, client)
    if not results:
        return "No results."

    async def excerpt(r: SearchResult) -> str | None:
        try:
            text = await asyncio.wait_for(fetch(r.url, settings, client), timeout=12)
        except (WebError, TimeoutError):
            return None
        text = text.strip()
        if len(text) > EXCERPT_CHARS:
            text = text[:EXCERPT_CHARS] + " …"
        return text or None

    excerpts = await asyncio.gather(*(excerpt(r) for r in results[:read_top]))
    blocks = []
    for i, r in enumerate(results, 1):
        block = f"[{i}] {r.title}\n{r.url}\n{r.snippet}".rstrip()
        page = excerpts[i - 1] if i <= len(excerpts) else None
        if page:
            block += f"\n--- page excerpt ---\n{page}\n--- end excerpt ---"
        blocks.append(block)
    return "\n\n".join(blocks) + "\n\nUse fetch_url to read any result in full."


def format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results."
    return "\n\n".join(
        f"[{i}] {r.title}\n{r.url}\n{r.snippet}".rstrip() for i, r in enumerate(results, 1)
    )
