"""Arm C-self: the generator judges itself through TypeSafe's system-one-adapter.

The adapter answers the same questions, in the same wire shapes, with any
OpenAI-compatible LLM. Pointed at the local Gemma endpoint with
``llm_answer_mode="probabilities"`` it makes the small model write a
probability per option instead of a bare verdict. Nothing about those numbers
is calibrated; that is the ablation (spec section 10): does the small model
need the framing, or a real judge?

Same disk cache and cache key as HttpJudge, under a model id prefixed
``self:`` so a Gemma answer can never be mistaken for a Jev answer. The
responding model id from the provider is logged with every answer set.

Local endpoints: the Unsloth Desktop proxy answers 401 to any bearer token it
did not issue and accepts requests that carry no Authorization header. The
OpenAI SDK always attaches one, so with an empty ``api_key`` the header is
removed by an httpx request hook just before the request leaves.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .base import JSONValue, Question, SystemOneResponse, cache_key, parse_answers


def _plain(answer: Any) -> dict[str, Any]:
    """A typesafe_sdk answer model as the wire dict HttpJudge caches."""
    if isinstance(answer, Mapping):
        data = dict(answer)
    elif hasattr(answer, "model_dump"):
        data = answer.model_dump()
    else:
        data = dict(vars(answer))
    if "type" not in data:
        if "noul" in data:
            data["type"] = "noul"
        elif "choice" in data:
            data["type"] = "choice"
        elif "score" in data:
            data["type"] = "score"
    return data


def _strip_authorization(request: Any) -> None:
    request.headers.pop("Authorization", None)


def _client_without_auth(base_url: str, timeout: float):
    """An OpenAI client whose requests carry no Authorization header."""
    import httpx
    import openai

    http_client = httpx.Client(timeout=timeout, event_hooks={"request": [_strip_authorization]})
    return openai.OpenAI(base_url=base_url, api_key="local", max_retries=0, http_client=http_client)


class AdapterJudge:
    """Judge implementation over ``system_one_adapter`` with the HttpJudge disk cache."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        cache_dir: str | os.PathLike[str] = "json_cache",
        llm_answer_mode: str = "probabilities",
        structured_outputs: bool = True,
        n_retry_malformed: int = 1,
        api: str = "chat_completions",
        max_workers: int = 1,
        timeout: float = 180.0,
    ) -> None:
        from system_one_adapter import SystemOneAdapterClient
        from system_one_adapter.providers.openai import OpenAIProvider

        self.provider_model = model
        self.model = f"self:{model}:{llm_answer_mode}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(1, int(max_workers))
        self._provider = OpenAIProvider(model, base_url=base_url, api_key=api_key or "local", api=api)
        if not api_key:
            self._provider._client = _client_without_auth(base_url, timeout)
        self._client = SystemOneAdapterClient(
            structured_outputs=structured_outputs,
            llm_answer_mode=llm_answer_mode,
            n_retry_malformed_structure=int(n_retry_malformed),
            model=self._provider,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> AdapterJudge:
        base_url = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8888/v1")
        model = os.getenv("GENERATOR_MODEL", "")
        if not model:
            raise RuntimeError("GENERATOR_MODEL is empty; put it in .env")
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "model": model,
            "api_key": os.getenv("GENERATOR_API_KEY", ""),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

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

    def system_one(self, state: JSONValue, questions: Mapping[str, Question]) -> SystemOneResponse:
        questions = dict(questions)
        key = cache_key(self.model, state, questions)
        path = self._cache_path(key)
        if path.exists():
            return self._to_response(json.loads(path.read_text(encoding="utf-8")), cached=True)
        response = self._client.system_one(state, questions)
        answers = {qid: _plain(a) for qid, a in dict(response.answers).items()}
        usage = response.usage
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
        entry = {
            "model": self.model,
            "provider_model": getattr(response, "model", None) or self.provider_model,
            "answers": answers,
            "usage": {
                "input_tokens": usage_dict.get("input_tokens", 0),
                "output_tokens": usage_dict.get("output_tokens", 0),
                "raw": usage_dict,
            },
            "request": {"model": self.model, "state": state, "questions": questions},
        }
        out = self._to_response(entry, cached=False)
        self._write_cache(key, entry)
        return out

    def system_one_many(
        self, items: Sequence[tuple[JSONValue, Mapping[str, Question]]]
    ) -> list[SystemOneResponse]:
        items = list(items)
        if self.max_workers == 1 or len(items) <= 1:
            return [self.system_one(state, questions) for state, questions in items]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda item: self.system_one(item[0], item[1]), items))
