"""SearXNG JSON client.

One request per (query, category, time_range), exactly the request the spec
writes out in section 3:

    q=<query>&format=json&categories=<cat>&time_range=<tr>&language=en&safesearch=0&pageno=1

Results are de-duplicated by URL and cut at ``thresholds.max_snippets``.
The raw response is kept separately so a snapshot can store it verbatim.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urldefrag

import httpx

from harness.config import env, thresholds

DEFAULT_URL = "http://127.0.0.1:8080"


def base_url(url: str | None = None) -> str:
    return (url or env("SEARXNG_URL", DEFAULT_URL) or DEFAULT_URL).rstrip("/")


def search_raw(
    query: str,
    category: str = "general",
    time_range: str | None = None,
    url: str | None = None,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Return SearXNG's JSON response untouched."""
    params: dict[str, str] = {
        "q": query,
        "format": "json",
        "categories": category,
        "language": "en",
        "safesearch": "0",
        "pageno": "1",
    }
    if time_range:
        params["time_range"] = time_range
    own_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        response = client.get(base_url(url) + "/search", params=params)
        response.raise_for_status()
        payload = response.json()
    finally:
        if own_client:
            client.close()
    if not isinstance(payload, dict) or "results" not in payload:
        raise RuntimeError("SearXNG did not return a JSON result list; is format=json enabled?")
    return payload


def _canonical(url: str) -> str:
    return urldefrag(url.strip())[0].rstrip("/")


def normalize(raw: dict[str, Any], max_results: int | None = None) -> list[dict[str, Any]]:
    """Turn a raw response into ranked, URL-de-duplicated result dicts."""
    limit = int(max_results or thresholds()["max_snippets"])
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in raw.get("results", []):
        url = item.get("url")
        if not url:
            continue
        key = _canonical(url)
        if key in seen:
            continue
        seen.add(key)
        engines = item.get("engines") or ([item["engine"]] if item.get("engine") else [])
        out.append(
            {
                "rank": len(out) + 1,
                "url": url,
                "title": (item.get("title") or "").strip(),
                "snippet": (item.get("content") or "").strip(),
                "engines": list(engines),
                "published_date": item.get("publishedDate") or item.get("pubdate") or None,
                "score": item.get("score"),
                "category": item.get("category"),
            }
        )
        if len(out) >= limit:
            break
    return out


def search(
    query: str,
    category: str = "general",
    time_range: str | None = None,
    url: str | None = None,
    max_results: int | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Ranked results with rank, url, title, snippet, engines, published_date."""
    return normalize(search_raw(query, category, time_range, url, timeout), max_results)
