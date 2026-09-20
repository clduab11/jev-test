"""Generator client for llama-server or Unsloth Desktop.

Uses the OpenAI-compatible ``/v1/completions`` endpoint with the Gemma 4 turn
format rendered here, instead of ``/v1/chat/completions``. Reason, found on
2026-09-19 against Unsloth Desktop's llama-server build: its chat template
prefills an empty thought channel when thinking is off, and Gemma 4 E2B then
writes its reasoning into the visible answer and never states the answer.
With no ``<|think|>`` token and no prefill the model answers directly and
cites. Thinking on means the ``<|think|>`` token at the head of the system
turn, the official Gemma 4 mechanism.

Sampling comes from ``generator.sampling`` in judge/questions.v1.json. The
seed is passed per request and recorded on every generation.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from harness.config import env, generator_settings

DEFAULT_URL = "http://127.0.0.1:8888/v1"
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
MAX_RETRIES = 6
BACKOFF_BASE = 0.5
BACKOFF_CAP = 30.0

log = logging.getLogger(__name__)
_CHANNEL_RE = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
_MARKER_RE = re.compile(r"<\|?channel\|?>|<\|?turn\|?>|<\|think\|>")


@dataclass
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str
    seed: int | None
    thinking: bool
    raw_text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def base_url(url: str | None = None) -> str:
    return (url or env("LLAMA_SERVER_URL", DEFAULT_URL) or DEFAULT_URL).rstrip("/")


def model_id(model: str | None = None) -> str:
    return model or env("GENERATOR_MODEL", "") or ""


def render_prompt(system: str | None, user: str, thinking: bool = False) -> str:
    """Gemma 4 turns: optional system turn, one user turn, an open model turn."""
    parts = ["<bos>"]
    if system or thinking:
        head = "<|think|>\n" if thinking else ""
        parts.append(f"<|turn>system\n{head}{(system or '').strip()}<turn|>\n")
    parts.append(f"<|turn>user\n{user.strip()}<turn|>\n<|turn>model\n")
    return "".join(parts)


def clean_output(text: str) -> str:
    """Drop any thought channel that leaked into the answer and any stray markers."""
    text = _CHANNEL_RE.sub("", text)
    text = _MARKER_RE.sub("", text)
    return text.strip()


def generate(
    system: str | None,
    user: str,
    seed: int | None = None,
    thinking: bool = False,
    max_tokens: int = 400,
    url: str | None = None,
    model: str | None = None,
    timeout: float = 180.0,
    client: httpx.Client | None = None,
) -> Generation:
    """One completion. Returns the text plus token counts and latency."""
    sampling = generator_settings()["sampling"]
    body: dict[str, Any] = {
        "model": model_id(model),
        "prompt": render_prompt(system, user, thinking=thinking),
        "max_tokens": max_tokens,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "stop": ["<turn|>"],
    }
    if seed is not None:
        body["seed"] = int(seed)
    own_client = client is None
    client = client or httpx.Client(timeout=timeout)
    started = time.perf_counter()
    endpoint = base_url(url) + "/completions"
    try:
        # The generator is a local process that reloads models and drops connections.
        # One blip must not cost an arm that has been running for hours.
        attempt = 0
        while True:
            try:
                response = client.post(endpoint, json=body)
                if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()
                payload = response.json()
                break
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = isinstance(exc, (httpx.TransportError, ValueError)) or status in RETRY_STATUSES
                if not retryable or attempt >= MAX_RETRIES:
                    raise
                delay = min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt)) * (0.5 + random.random() / 2)
                log.warning("generator attempt %d failed (%s); retrying in %.1fs", attempt + 1, exc, delay)
                time.sleep(delay)
                attempt += 1
    finally:
        if own_client:
            client.close()
    latency = time.perf_counter() - started
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    raw_text = choice.get("text") or ""
    return Generation(
        text=clean_output(raw_text),
        model=str(payload.get("model") or body["model"]),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_s=round(latency, 3),
        finish_reason=str(choice.get("finish_reason") or ""),
        seed=seed,
        thinking=thinking,
        raw_text=raw_text,
    )
