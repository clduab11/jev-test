"""One S0 call and one S1 call through system-one-adapter against the local Gemma.

Usage:  uv run --with "system-one-adapter[openai]>=0.2.0" python scripts/smoke_adapter.py [--no-structured]

Prints the answers, the token usage and, on failure, the adapter's attempt
trace, so the C-self wiring can be judged before any arm runs.
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

from harness.judge.adapter import AdapterJudge
from harness.judge.base import load_questions

QUERY = "What is the capital of France?"
PASSAGE = {
    "id": "r01s",
    "title": "Paris - Wikipedia",
    "domain": "en.wikipedia.org",
    "source_type": "encyclopedia",
    "text": "Paris is the capital and largest city of France. With an estimated population of 2.1 million residents, it is the centre of the Ile-de-France region.",
}


def main() -> int:
    load_dotenv(ROOT / ".env")
    structured = "--no-structured" not in sys.argv
    judge = AdapterJudge.from_env(cache_dir=ROOT / "json_cache", structured_outputs=structured)
    print(f"judge model id: {judge.model} (provider model {judge.provider_model}, structured={structured})")
    spec = ROOT / "judge" / "questions.v1.json"

    for stage, state in (
        ("S0_intake", {"query": QUERY, "today": time.strftime("%Y-%m-%d"), "palace_has_verified_hits": False}),
        ("S1_passage_gate", {"query": QUERY, "passage": PASSAGE}),
    ):
        questions = load_questions(spec, stage)
        started = time.perf_counter()
        try:
            response = judge.system_one(state, questions)
        except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not hide
            print(f"{stage}: FAILED {type(exc).__name__}: {str(exc)[:800]}")
            debug = getattr(exc, "debug", None) or getattr(exc, "attempts", None)
            if debug:
                print("debug:", json.dumps(debug, default=str)[:2000])
            return 1
        print(f"{stage}: {time.perf_counter() - started:.1f}s, cached={response.cached}, model={response.model}")
        for qid, answer in response.answers.items():
            print(f"  {qid}: {json.dumps(asdict(answer))}")
        print(f"  usage: input={response.input_tokens} output={response.output_tokens} raw={json.dumps(response.raw.get('usage', {}).get('raw', {}), default=str)[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
