"""Fetch a page, extract its main text, and chunk it by tokens.

Policy from the spec's section 4 fetch policy and the thresholds block:
  * 10 second timeout per page, HTML only, capped at 5 MB
  * main text through trafilatura, then code fences and residual tags stripped
  * chunks of ``chunk_tokens`` with ``chunk_overlap`` (tiktoken cl100k_base)
  * at most ``max_chunks_per_page`` chunks, each at most ``passage_max_chars``
"""

from __future__ import annotations

import html as html_lib
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import tiktoken
import trafilatura

from harness.config import thresholds

USER_AGENT = "Mozilla/5.0 (compatible; jev-test/0.1; +https://github.com/clduab11/jev-test)"
MAX_BYTES = 5_000_000
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]{1,200}>")
_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status: int | None = None
    content_type: str = ""
    error: str | None = None
    text: str = ""
    n_bytes: int = 0
    fetched_at: str = ""
    elapsed_s: float = 0.0
    chunks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean_text(text: str) -> str:
    text = _FENCE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def extract_text(body: str, url: str | None = None, content_type: str = "") -> str:
    """Main text of a page. Falls back to a tag strip when trafilatura gives nothing."""
    if content_type.startswith("text/plain"):
        return clean_text(body)
    text = trafilatura.extract(
        body,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
        output_format="txt",
    )
    if not text:
        text = trafilatura.extract(body, url=url, include_comments=False, favor_recall=True)
    if not text:
        text = body
    return clean_text(text)


def fetch(url: str, timeout: float = 10.0, client: httpx.Client | None = None) -> FetchResult:
    """Fetch one URL. Never raises; errors are recorded on the result."""
    result = FetchResult(url=url, fetched_at=now_iso())
    own_client = client is None
    client = client or httpx.Client(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    started = time.perf_counter()
    try:
        with client.stream("GET", url) as response:
            result.status = response.status_code
            result.final_url = str(response.url)
            result.content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if response.status_code >= 400:
                result.error = f"http {response.status_code}"
                return result
            if not result.content_type.startswith(TEXT_TYPES):
                result.error = f"unsupported content type {result.content_type or 'unknown'}"
                return result
            chunks: list[bytes] = []
            size = 0
            for part in response.iter_bytes():
                chunks.append(part)
                size += len(part)
                if size >= MAX_BYTES:
                    break
            raw = b"".join(chunks)[:MAX_BYTES]
            result.n_bytes = len(raw)
            encoding = response.encoding or "utf-8"
        try:
            body = raw.decode(encoding, errors="replace")
        except LookupError:
            body = raw.decode("utf-8", errors="replace")
        result.text = extract_text(body, url=result.final_url or url, content_type=result.content_type)
        if not result.text:
            result.error = "no extractable text"
    except httpx.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:300]
    except Exception as exc:  # noqa: BLE001 - a snapshot must survive any single bad page
        result.error = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        result.elapsed_s = round(time.perf_counter() - started, 3)
        if own_client:
            client.close()
    return result


_ENCODINGS: dict[str, tiktoken.Encoding] = {}


def encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    if name not in _ENCODINGS:
        _ENCODINGS[name] = tiktoken.get_encoding(name)
    return _ENCODINGS[name]


def count_tokens(text: str, name: str = "cl100k_base") -> int:
    return len(encoding(name).encode(text, disallowed_special=()))


def cap_chars(text: str, max_chars: int) -> str:
    """Cut at max_chars on a word boundary when one is near enough."""
    return _cap_chars(text, max_chars)


def _cap_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars * 0.6 else cut).rstrip()


def chunk_text(
    text: str,
    chunk_tokens: int | None = None,
    chunk_overlap: float | None = None,
    max_chunks: int | None = None,
    max_chars: int | None = None,
    min_tokens: int = 20,
) -> list[dict[str, Any]]:
    """Split text into overlapping token windows; returns chunk_index, text, n_tokens."""
    t = thresholds()
    chunk_tokens = int(chunk_tokens or t["chunk_tokens"])
    chunk_overlap = float(t["chunk_overlap"] if chunk_overlap is None else chunk_overlap)
    max_chunks = int(max_chunks or t["max_chunks_per_page"])
    max_chars = int(max_chars or t["passage_max_chars"])
    enc = encoding()
    tokens = enc.encode(text, disallowed_special=())
    if not tokens:
        return []
    overlap = int(round(chunk_tokens * chunk_overlap))
    step = max(1, chunk_tokens - overlap)
    out: list[dict[str, Any]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_tokens]
        if len(window) < min_tokens and out:
            break
        piece = _cap_chars(enc.decode(window).strip(), max_chars)
        if piece:
            out.append({"chunk_index": len(out), "text": piece, "n_tokens": len(window)})
        if len(out) >= max_chunks or start + chunk_tokens >= len(tokens):
            break
    return out


def fetch_and_chunk(url: str, timeout: float = 10.0, client: httpx.Client | None = None) -> FetchResult:
    result = fetch(url, timeout=timeout, client=client)
    if result.ok:
        result.chunks = chunk_text(result.text)
    return result
