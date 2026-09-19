"""Stage S2: sufficiency (spec section 5), the negative-rejection lever.

One request per query per round. This is the only stage that sends several
passages in one state, because the judgment is about the set; the set is
already bounded by top_evidence_k and max_evidence_tokens.

The refinement query is written by the generator (Gemma, thinking off), never
by the judge and never by a frontier model: a frontier model inside arm D
would break the small-model thesis (spec section 16).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from harness.config import SPEC_PATH, thresholds
from harness.generate.openai_compat import generate
from harness.judge.base import Answer, Judge, Question, load_questions, to_plain

STAGE = "S2_sufficiency"

REFINE_SYSTEM = "You write one web search query. Reply with the query only."
REFINE_USER = (
    "A web search for the question below did not find the answer.\n"
    "Write one different search query that would find the specific fact the "
    "question asks for. Use different words from the question where you can. "
    "Reply with the query only, on one line, with no quotes and no explanation.\n\n"
    "QUESTION: {query}"
)
_QUERY_MAX_CHARS = 200
_LABEL_RE = re.compile(r"^\s*(?:search\s*)?query\s*[:\-]\s*", re.IGNORECASE)


def questions() -> dict[str, Question]:
    return load_questions(SPEC_PATH, STAGE)


def build_state(query: str, evidence: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"query": query, "evidence": [{"id": e["id"], "text": e["text"]} for e in evidence]}


def decide(
    answers: Mapping[str, Answer], round_index: int, t: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """generate, refine (round 0 only) or abstain; plus the conflicts flag."""
    t = t or thresholds()
    sufficient_p = float(answers["evidence_sufficient"].noul)
    conflicts_p = float(answers["evidence_conflicts"].noul)
    sufficient = sufficient_p >= float(t["sufficient_min"])
    if sufficient:
        action = "generate"
    elif round_index < int(t["max_refine_rounds"]):
        action = "refine"
    else:
        action = "abstain"
    return {
        "action": action,
        "round": int(round_index),
        "sufficient": sufficient,
        "sufficient_p": sufficient_p,
        "conflicts": conflicts_p >= float(t["conflicts_min"]),
        "conflicts_p": conflicts_p,
    }


def run(
    judge: Judge, query: str, evidence: list[Mapping[str, Any]], round_index: int
) -> dict[str, Any]:
    response = judge.system_one(build_state(query, evidence), questions())
    decision = decide(response.answers, round_index)
    decision.update(
        {
            "evidence_ids": [e["id"] for e in evidence],
            "answers": to_plain(response.answers),
            "model": response.model,
            "cached": response.cached,
            "n_requests": 1,
            "n_cached": int(response.cached),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
    )
    return decision


def clean_query(text: str) -> str:
    """First non-empty line, labels and quotes removed, whitespace collapsed, capped."""
    for line in (text or "").splitlines():
        line = _LABEL_RE.sub("", line.strip())
        line = line.strip().strip("`\"'").strip()
        if line:
            return " ".join(line.split())[:_QUERY_MAX_CHARS].strip()
    return ""


def refine_query(query: str, seed: int | None = None, thinking: bool = False) -> dict[str, Any]:
    """Ask the generator for exactly one alternative search query."""
    generation = generate(
        REFINE_SYSTEM, REFINE_USER.format(query=query), seed=seed, thinking=thinking, max_tokens=64
    )
    refined = clean_query(generation.text) or query
    return {
        "query": refined,
        "same_as_original": refined.strip().lower() == query.strip().lower(),
        "raw": generation.text,
        "model": generation.model,
        "latency_s": generation.latency_s,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
    }
