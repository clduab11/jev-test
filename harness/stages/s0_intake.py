"""Stage S0: intake (spec section 3).

One request per query. The judge decides whether to search, which lanes to
use, how recent the sources must be, and which SearXNG category to hit. Code
turns the four answers into a retrieval plan with the thresholds from
judge/questions.v1.json and never reads them from anywhere else.

Everything date-shaped stays in code (design rule 5): ``today`` is data the
judge sees, and the recency window is turned into a SearXNG ``time_range``
and a palace ``since`` date here.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

from harness.config import SPEC_PATH, thresholds
from harness.judge.base import Answer, Judge, Question, load_questions, to_plain
from harness.retrieval.palace import VERIFIED, Palace, search_wing

STAGE = "S0_intake"
TIME_RANGE = {"any": None, "year": "year", "month": "month", "week": "week", "day": "day"}
_WINDOW_DAYS = {"year": 365, "month": 30, "week": 7, "day": 1}
PRECHECK_N = 3


def questions() -> dict[str, Question]:
    return load_questions(SPEC_PATH, STAGE)


def build_state(query: str, today: str, palace_has_verified_hits: bool) -> dict[str, Any]:
    return {"query": query, "today": today, "palace_has_verified_hits": bool(palace_has_verified_hits)}


def palace_precheck(palace: Palace, query: str) -> bool:
    """One cheap search of the verified wing before S0; sets palace_has_verified_hits."""
    return bool(search_wing(palace, query, VERIFIED, n=PRECHECK_N))


def since_for(time_range: str | None, today: str) -> str | None:
    """The palace ``authored_at`` floor that matches a SearXNG time range."""
    days = _WINDOW_DAYS.get(time_range or "")
    if days is None:
        return None
    return (date.fromisoformat(today) - timedelta(days=days)).isoformat()


def decide(
    answers: Mapping[str, Answer],
    palace_has_verified_hits: bool,
    *,
    force_search: bool = False,
    force_web: bool = False,
    t: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the section 3 code policy to one set of answers."""
    t = t or thresholds()
    floor = float(t["choice_floor"])
    needs = answers["needs_search"]
    source = answers["source"]
    recency = answers["recency"]
    category = answers["category"]

    retrieve_judged = float(needs.noul) >= float(t["needs_search_min"])
    if source.confidence < floor:
        source_judged = "both"  # widen, never narrow, on uncertainty
    elif source.choice == "palace" and not palace_has_verified_hits:
        source_judged = "both"
    else:
        source_judged = source.choice
    time_range = None if recency.confidence < floor else TIME_RANGE.get(recency.choice)
    category_choice = "general" if category.confidence < floor else category.choice
    return {
        "retrieve": True if force_search else retrieve_judged,
        "retrieve_judged": retrieve_judged,
        "source": "web" if force_web else source_judged,
        "source_judged": source_judged,
        "time_range": time_range,
        "category": category_choice,
        "forced": {"search": bool(force_search), "web": bool(force_web)},
    }


def run(
    judge: Judge,
    query: str,
    today: str,
    palace_has_verified_hits: bool = False,
    *,
    force_search: bool = False,
    force_web: bool = False,
) -> dict[str, Any]:
    """Send the S0 questions once and return the decision plus the raw answers."""
    state = build_state(query, today, palace_has_verified_hits)
    response = judge.system_one(state, questions())
    decision = decide(
        response.answers, palace_has_verified_hits, force_search=force_search, force_web=force_web
    )
    decision.update(
        {
            "since": since_for(decision["time_range"], today),
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
