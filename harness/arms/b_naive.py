"""Arm B: naive RAG. Top 8 pages from the frozen snapshot, chunks stuffed in
retriever order up to 12,000 tokens, no gating, no verification. Uncited
claims are kept, fabricated citations are still counted and stripped.

The two numbers come from the arm B description in judge/questions.v1.json
("searxng top 8 pages, chunked, stuffed to 12k tokens, no gate"). A page whose
fetch failed falls back to its search snippet, as the spec's fetch policy says.
"""

from __future__ import annotations

from typing import Any

from harness.retrieval.fetch import count_tokens
from harness.stages import s3_generate

ARM = "B"
TOP_PAGES = 8
MAX_EVIDENCE_TOKENS = 12_000


def build_evidence(replayed: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunks of the top pages in retriever order, cut at the token budget."""
    evidence: list[dict[str, Any]] = []
    used = 0
    for page in replayed["results"][:TOP_PAGES]:
        if page.get("chunks"):
            candidates = [
                {"id": c["cite_id"], "text": c["text"], "drawer_id": c["drawer_id"], "n_tokens": c["n_tokens"]}
                for c in page["chunks"]
            ]
        elif page.get("snippet"):
            snippet = page["snippet"]
            candidates = [
                {
                    "id": f"r{page['rank']:02d}s",
                    "text": snippet,
                    "drawer_id": None,
                    "n_tokens": count_tokens(snippet),
                }
            ]
        else:
            candidates = []
        for item in candidates:
            if used + item["n_tokens"] > MAX_EVIDENCE_TOKENS:
                return evidence
            evidence.append(item)
            used += item["n_tokens"]
    return evidence


def answer(query: str, replayed: dict[str, Any], seed: int | None = None, thinking: bool = False) -> dict[str, Any]:
    evidence = build_evidence(replayed)
    if not evidence:
        return {
            "arm": ARM,
            "answer": "",
            "answer_raw": "",
            "abstained": True,
            "abstain_reason": "no evidence in snapshot",
            "claims": [],
            "citations": [],
            "fabricated_citations": 0,
            "evidence_ids": [],
            "n_evidence_pages": 0,
            "n_evidence_chunks": 0,
            "evidence_tokens": 0,
            "latency_s": 0.0,
            "generator_model": None,
            "seed": seed,
        }
    record = s3_generate.generate_answer(query, evidence, seed=seed, keep_uncited=True, thinking=thinking)
    record.update(
        {
            "arm": ARM,
            "n_evidence_pages": len({e["id"][:3] for e in evidence}),
            "n_evidence_chunks": len(evidence),
            "evidence_tokens": sum(e["n_tokens"] for e in evidence),
            "evidence_drawer_ids": [e["drawer_id"] for e in evidence],
        }
    )
    return record
