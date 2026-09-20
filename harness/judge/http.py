"""HTTP judge: anything that speaks TypeSafe's ``/v1/systemone`` wire format.

Covers Jev at api.typesafe.ai, LitJev and the Laya shim on localhost, and
OpenRouter Decisions (a different ``path`` and extra headers; the request body
is the same). Nothing here knows about a provider's private schema.

Rules from docs/JUDGMENT_SPEC.md that this module enforces:
  * every call is cached on disk under ``cache_dir`` keyed by
    ``cache_key(model, state, questions)`` so re-runs replay for free
  * the responding ``model`` id is stored with every answer set
  * the model id must be a pinned version, never a ``*-latest`` alias
  * 429, 5xx and 529 are retried with exponential backoff and honour Retry-After;
    any other 4xx raises immediately with the server's message. The default budget
    of ten attempts capped at sixty seconds covers a multi-minute endpoint restart,
    because one unlucky request among arm D's twenty-five thousand must not end a
    five-hour run.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Self

import httpx

from .base import JSONValue, Question, SystemOneResponse, cache_key, parse_answers

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


class JudgeRequestError(RuntimeError):
    """A judge request failed for a reason a retry will not fix, or retries ran out."""

    def __init__(self, message: str, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    """Parse a Retry-After header given as seconds or as an HTTP date."""
    value = headers.get("Retry-After")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


class HttpJudge:
    """Judge implementation over HTTP with a disk cache and a small worker pool."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        path: str = "/v1/systemone",
        cache_dir: str | os.PathLike[str] = "json_cache",
        timeout: float = 60.0,
        max_workers: int = 4,
        max_retries: int = 10,
        backoff_base: float = 0.5,
        backoff_cap: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        allow_alias: bool = False,
    ) -> None:
        if not allow_alias and model.endswith("latest"):
            raise ValueError(
                f"judge model {model!r} is a moving alias; pin a version such as jev-1.13.0"
            )
        self.model = model
        self.path = path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(1, int(max_workers))
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    @classmethod
    def from_env(cls, prefix: str = "JEV", **overrides) -> HttpJudge:
        """Build from ``<prefix>_BASE_URL``, ``<prefix>_MODEL`` and the matching key.

        The key variable is ``TYPESAFE_API_KEY`` for the default prefix and
        ``<prefix>_API_KEY`` otherwise (for example ``LOCAL_JUDGE_API_KEY``,
        which may be empty for a local shim).
        """
        base_url = os.getenv(f"{prefix}_BASE_URL", "https://api.typesafe.ai")
        model = os.getenv(f"{prefix}_MODEL", "jev-1.13.0")
        key_var = "TYPESAFE_API_KEY" if prefix == "JEV" else f"{prefix}_API_KEY"
        api_key = os.getenv(key_var, "")
        if prefix == "JEV" and not api_key:
            raise RuntimeError("TYPESAFE_API_KEY is empty; put it in .env")
        kwargs = {"base_url": base_url, "api_key": api_key, "model": model}
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------ cache

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("unreadable cache entry %s; fetching again", path)
            return None

    def _write_cache(self, key: str, entry: dict) -> None:
        path = self._cache_path(key)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _to_response(entry: dict, cached: bool) -> SystemOneResponse:
        usage = entry.get("usage") or {}
        return SystemOneResponse(
            model=str(entry.get("model", "")),
            answers=parse_answers(entry["answers"]),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cached=cached,
            raw=entry,
        )

    # --------------------------------------------------------------- requests

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self.backoff_cap, self.backoff_base * (2**attempt))
        delay *= 0.5 + random.random() / 2  # jitter keeps the pool from retrying in step
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    def _post_with_retry(self, body: dict) -> dict:
        attempt = 0
        while True:
            try:
                resp = self._client.post(self.path, json=body)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise JudgeRequestError(
                        f"judge unreachable after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                self._sleep(self._backoff(attempt, None))
                attempt += 1
                continue
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise JudgeRequestError(
                        "judge returned a non-JSON body", resp.status_code, resp.text[:2000]
                    ) from exc
            if resp.status_code in RETRY_STATUSES:
                if attempt >= self.max_retries:
                    raise JudgeRequestError(
                        f"judge returned HTTP {resp.status_code} after {attempt + 1} attempt(s)",
                        resp.status_code,
                        resp.text[:2000],
                    )
                self._sleep(self._backoff(attempt, _retry_after_seconds(resp.headers)))
                attempt += 1
                continue
            raise JudgeRequestError(
                f"judge returned HTTP {resp.status_code}: {resp.text[:500]}",
                resp.status_code,
                resp.text[:2000],
            )

    # ---------------------------------------------------------------- public

    def system_one(self, state: JSONValue, questions: Mapping[str, Question]) -> SystemOneResponse:
        questions = dict(questions)
        key = cache_key(self.model, state, questions)
        entry = self._read_cache(key)
        if entry is not None:
            return self._to_response(entry, cached=True)
        body = {"model": self.model, "state": state, "questions": questions}
        raw = self._post_with_retry(body)
        if not isinstance(raw, dict) or "answers" not in raw:
            raise JudgeRequestError("judge response has no answers", body=json.dumps(raw)[:2000])
        entry = {
            "model": raw.get("model", self.model),
            "answers": raw["answers"],
            "usage": raw.get("usage", {}),
            "request": body,
        }
        try:
            response = self._to_response(entry, cached=False)
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeRequestError(
                f"malformed judge answers: {exc}", body=json.dumps(raw)[:2000]
            ) from exc
        self._write_cache(key, entry)
        log.debug("judge %s answered %d question(s)", entry["model"], len(response.answers))
        return response

    def system_one_many(
        self, items: Sequence[tuple[JSONValue, Mapping[str, Question]]]
    ) -> list[SystemOneResponse]:
        """Run ``(state, questions)`` pairs through the worker pool, preserving order."""
        items = list(items)
        if not items:
            return []
        if self.max_workers == 1 or len(items) == 1:
            return [self.system_one(state, questions) for state, questions in items]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda item: self.system_one(item[0], item[1]), items))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
