"""Reciprocal rank fusion (spec section 4 ranking, design rule 10).

    fused(id) = sum over every list that contains id of 1 / (k + rank(id))

``k`` is ``thresholds.rrf_k``. A passage absent from a list gets nothing from
it. Ranks are 1-based and may tie: chunks of one page share that page's
retriever rank because SearXNG ranked pages, not chunks, and equal judge
probabilities share a rank (competition ranking: 1, 2, 2, 4).

The three list builders read passage dicts as the stages produce them:
``engine_rank`` and ``engines`` for the retriever list, ``similarity`` for the
palace list, ``gate.answers.contains_answer_evidence`` for the judge list.
``select`` applies ``top_evidence_k`` and ``max_evidence_tokens``; it stops at
the first passage that would overflow the token budget, the same rule arm B
uses, so the evidence block is always a prefix of the fused order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from harness.config import thresholds
from harness.retrieval.fetch import count_tokens


def competition_ranks(keyed: Mapping[str, Any], descending: bool = True) -> dict[str, int]:
    """1-based ranks; equal keys share a rank and the following rank is skipped."""
    values = list(keyed.values())
    ranks: dict[str, int] = {}
    for pid, value in keyed.items():
        better = sum(1 for other in values if (other > value if descending else other < value))
        ranks[pid] = 1 + better
    return ranks


def retriever_ranks(passages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """SearXNG position, ties broken by how many engines returned the URL."""
    keyed = {}
    for passage in passages:
        if passage.get("engine_rank") is None:
            continue
        keyed[passage["id"]] = (int(passage["engine_rank"]), -len(passage.get("engines") or []))
    return competition_ranks(keyed, descending=False)


def palace_ranks(passages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """MemPalace cosine similarity, palace lane only."""
    keyed = {
        p["id"]: float(p["similarity"]) for p in passages if p.get("similarity") is not None
    }
    return competition_ranks(keyed, descending=True)


def evidence_ranks(passages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """The judge's ``contains_answer_evidence`` probability from stage S1."""
    keyed = {}
    for passage in passages:
        answers = (passage.get("gate") or {}).get("answers") or {}
        value = answers.get("contains_answer_evidence")
        if value is None:
            continue
        keyed[passage["id"]] = float(value)
    return competition_ranks(keyed, descending=True)


def rrf(rank_lists: Sequence[Mapping[str, int]], k: int | None = None) -> list[tuple[str, float]]:
    """Fused (id, score) pairs, best first. Ties: better best-rank, then id."""
    k = int(thresholds()["rrf_k"] if k is None else k)
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranks in rank_lists:
        for pid, rank in ranks.items():
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + int(rank))
            best[pid] = min(best.get(pid, int(rank)), int(rank))
    return sorted(scores.items(), key=lambda item: (-item[1], best[item[0]], item[0]))


def fuse(passages: Sequence[Mapping[str, Any]], k: int | None = None) -> dict[str, Any]:
    """Run all three lists over one candidate set. Returns the order and the lists."""
    lists = {
        "retriever": retriever_ranks(passages),
        "palace": palace_ranks(passages),
        "evidence": evidence_ranks(passages),
    }
    fused = rrf([lists["retriever"], lists["palace"], lists["evidence"]], k)
    return {"order": [pid for pid, _ in fused], "scores": dict(fused), "ranks": lists}


def select(
    order: Sequence[str],
    passages_by_id: Mapping[str, Mapping[str, Any]],
    top_k: int | None = None,
    max_tokens: int | None = None,
    count: Callable[[str], int] = count_tokens,
) -> list[dict[str, Any]]:
    """The top ``top_k`` passages of ``order`` that fit in ``max_tokens``."""
    t = thresholds()
    top_k = int(t["top_evidence_k"] if top_k is None else top_k)
    max_tokens = int(t["max_evidence_tokens"] if max_tokens is None else max_tokens)
    chosen: list[dict[str, Any]] = []
    used = 0
    for pid in order:
        passage = passages_by_id.get(pid)
        if passage is None:
            continue
        n_tokens = int(passage.get("n_tokens") or count(passage["text"]))
        if used + n_tokens > max_tokens:
            break
        chosen.append({**passage, "n_tokens": n_tokens})
        used += n_tokens
        if len(chosen) >= top_k:
            break
    return chosen
