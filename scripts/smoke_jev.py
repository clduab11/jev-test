"""One real Jev call on the fixed S0 query, then the same call served from disk.

Usage:  uv run python scripts/smoke_jev.py

Reads .env for JEV_BASE_URL, TYPESAFE_API_KEY and JEV_MODEL, sends the S0
intake questions from judge/questions.v1.json, prints the responding model id,
every answer and the token usage, then repeats the call and checks that the
second response came from json_cache/ rather than the network.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.judge.base import load_questions
from harness.judge.http import HttpJudge

QUERY = "What is the capital of France?"


def main() -> int:
    load_dotenv(ROOT / ".env")
    judge = HttpJudge.from_env(cache_dir=ROOT / "json_cache")
    questions = load_questions(ROOT / "judge" / "questions.v1.json", "S0_intake")
    state = {
        "query": QUERY,
        "today": time.strftime("%Y-%m-%d"),
        "palace_has_verified_hits": False,
    }

    first = judge.system_one(state, questions)
    print(f"model: {first.model}")
    for qid, answer in first.answers.items():
        print(f"  {qid}: {json.dumps(asdict(answer))}")
    print(f"usage: input_tokens={first.input_tokens} output_tokens={first.output_tokens}")
    print(f"cached: {first.cached}")

    second = judge.system_one(state, questions)
    print(f"cached: {second.cached}")
    if not second.cached:
        print("FAIL: second call did not come from the cache", file=sys.stderr)
        return 1
    if second.answers != first.answers or second.model != first.model:
        print("FAIL: cached answers differ from the live answers", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
