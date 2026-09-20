"""SimpleQA grading with the official three-way grader prompt.

The prompt text is committed verbatim at harness/grading/prompts/simpleqa_grader.txt
(simple-evals ``GRADER_TEMPLATE``). The grade is A, B or C, parsed the way the
official code parses it: first A, B or C in the reply, default C.

The grader is a frontier model behind the LiteLLM gateway (GRADER_BASE_URL,
GRADER_API_KEY, GRADER_MODEL). It plays no role in the pipeline (spec rule 9).
Every grading call is cached under json_cache/grading/ keyed by
(grader model, full prompt), the same convention as judge calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from harness.config import ROOT, env, load_env

PROMPT_PATH = ROOT / "harness" / "grading" / "prompts" / "simpleqa_grader.txt"
CACHE_DIR = ROOT / "json_cache" / "grading"
LETTERS = {"A": "correct", "B": "incorrect", "C": "not_attempted"}
GRADER_MAX_TOKENS = 512  # see the note in claim_support.py: the grader deliberates first

log = logging.getLogger(__name__)


def write_cache_entry(path: Path, entry: dict) -> None:
    """Write a cache entry atomically, so a killed run never leaves a torn file that
    poisons every later grading pass."""
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def grader_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(question: str, target: str, predicted_answer: str) -> str:
    return grader_template().format(question=question, target=target, predicted_answer=predicted_answer)


def parse_letter(text: str) -> str:
    match = re.search(r"(A|B|C)", text or "")
    return match.group(0) if match else "C"


def _client():
    from openai import OpenAI

    load_env()
    api_key = env("GRADER_API_KEY", "")
    if not api_key:
        raise RuntimeError("GRADER_API_KEY is empty; put it in .env")
    return OpenAI(base_url=env("GRADER_BASE_URL"), api_key=api_key)


def grade(
    question: str,
    target: str,
    predicted_answer: str,
    model: str | None = None,
    cache_dir: Path = CACHE_DIR,
    client=None,
) -> dict[str, Any]:
    """Grade one answer. Empty answers are not attempted and never reach the grader."""
    load_env()
    model = model or env("GRADER_MODEL", "") or ""
    if not predicted_answer.strip():
        return {"label": "not_attempted", "letter": "C", "model": model, "cached": False, "reply": "", "skipped": "empty answer"}
    prompt = build_prompt(question, target, predicted_answer)
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.exists():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return {"label": entry["label"], "letter": entry["letter"], "model": entry["model"], "cached": True, "reply": entry["reply"]}
        except (OSError, ValueError, KeyError):
            log.warning("unreadable grading cache entry %s; grading again", path.name)
    client = client or _client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=GRADER_MAX_TOKENS,
    )
    reply = (response.choices[0].message.content or "").strip()
    letter = parse_letter(reply)
    entry = {
        "model": str(response.model or model),
        "prompt": prompt,
        "reply": reply,
        "letter": letter,
        "label": LETTERS[letter],
        "usage": {
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
        },
    }
    write_cache_entry(path, entry)
    return {"label": entry["label"], "letter": letter, "model": entry["model"], "cached": False, "reply": reply}


def grade_records(records: list[dict[str, Any]], model: str | None = None, workers: int = 4) -> list[dict[str, Any]]:
    """Grade every record's ``answer`` against its ``gold``; abstentions are not attempted."""
    client = _client()

    def one(record: dict[str, Any]) -> dict[str, Any]:
        answer = "" if record.get("abstained") else (record.get("answer") or "")
        for attempt in range(3):
            try:
                return grade(record["query"], record["gold"], answer, model=model, client=client)
            except Exception as exc:  # noqa: BLE001 - one bad call must not lose the other 499
                log.warning("grading call failed (%d/3) for %s: %s", attempt + 1, record.get("question_id"), exc)
                last = exc
        # A record the grader could not reach is not "not attempted"; mark it so
        # grade_file can report it rather than folding it into the coverage denominator.
        return {"label": None, "letter": None, "model": model, "cached": False, "reply": "", "error": str(last)[:200]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, records))
