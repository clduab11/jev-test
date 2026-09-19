"""Stage S4: verification (spec section 7).

One request per claim against only the passages the claim cites (filter first,
rule 2), then one answer-level request. The support verdict is also the
admission decision for memory: stage M2 reads the kept claims this stage
returns.

    keep      supported   and confidence >= support_accept
    conflict  contradicted and confidence >= support_accept  (stripped, shown as a conflict)
    strip     otherwise
    abstain   if no claim is kept, or answer_addresses_query < addresses_min
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from harness.config import SPEC_PATH, thresholds
from harness.judge.base import ChoiceAnswer, Judge, Question, load_questions, run_many
from harness.stages.s3_generate import split_sentences

STAGE = "S4_verify"


def claim_questions() -> dict[str, Question]:
    return load_questions(SPEC_PATH, STAGE, "claim_questions")


def answer_questions() -> dict[str, Question]:
    return load_questions(SPEC_PATH, STAGE, "answer_questions")


def build_claim_state(
    query: str, claim: str, cited_evidence: list[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "query": query,
        "claim": claim,
        "cited_evidence": [{"id": e["id"], "text": e["text"]} for e in cited_evidence],
    }


def build_answer_state(query: str, answer: str) -> dict[str, Any]:
    return {"query": query, "answer": answer}


def verdict(support: ChoiceAnswer, t: Mapping[str, Any] | None = None) -> str:
    t = t or thresholds()
    if float(support.confidence) >= float(t["support_accept"]):
        if support.choice == "supported":
            return "keep"
        if support.choice == "contradicted":
            return "conflict"
    return "strip"


def _strip_reason(support: ChoiceAnswer) -> str:
    if support.choice == "supported":
        return "supported_low_confidence"
    if support.choice == "contradicted":
        return "contradicted_low_confidence"
    return "unsupported"


def verify(
    judge: Judge,
    query: str,
    claims: list[Mapping[str, Any]],
    evidence_by_id: Mapping[str, str],
    *,
    kept_sentences: list[str] | None = None,
    answer: str | None = None,
    t: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every claim, rebuild the answer from what survives, then check it
    addresses the query. ``claims`` are S3 claims ({text, citations});
    ``evidence_by_id`` maps citation ids to passage text."""
    t = t or thresholds()
    qs = claim_questions()
    items = []
    for claim in claims:
        cited = [{"id": c, "text": evidence_by_id[c]} for c in claim["citations"] if c in evidence_by_id]
        items.append((build_claim_state(query, claim["text"], cited), qs))
    responses = run_many(judge, items) if items else []

    kept: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    stripped: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    for claim, response in zip(claims, responses, strict=True):
        support = response.answers["support"]
        decision = verdict(support, t)
        entry = {
            "text": claim["text"],
            "citations": list(claim["citations"]),
            "support": {
                "choice": support.choice,
                "confidence": float(support.confidence),
                "probabilities": dict(support.probabilities),
            },
            "model": response.model,
            "cached": response.cached,
        }
        verdicts.append({**entry, "verdict": decision})
        if decision == "keep":
            kept.append(entry)
        elif decision == "conflict":
            conflicts.append({**entry, "reason": "contradicted"})
        else:
            stripped.append({**entry, "reason": _strip_reason(support)})

    sentences = list(kept_sentences) if kept_sentences is not None else split_sentences(answer or "")
    removed = {e["text"] for e in conflicts} | {e["text"] for e in stripped}
    final = " ".join(s for s in sentences if s not in removed).strip()

    out: dict[str, Any] = {
        "kept": kept,
        "conflicts": conflicts,
        "stripped": stripped,
        "claim_verdicts": verdicts,
        "n_requests": len(items),
        "n_cached": sum(1 for r in responses if r.cached),
        "input_tokens": sum(r.input_tokens for r in responses),
        "output_tokens": sum(r.output_tokens for r in responses),
        "model": next((r.model for r in responses), None),
    }
    if not kept:
        out.update(
            {"abstained": True, "abstain_reason": "no_claims_kept", "answer": "", "addresses_query": None}
        )
        return out

    check = judge.system_one(build_answer_state(query, final), answer_questions())
    addresses_p = float(check.answers["answer_addresses_query"].noul)
    passed = addresses_p >= float(t["addresses_min"])
    out["n_requests"] += 1
    out["n_cached"] += int(check.cached)
    out["input_tokens"] += check.input_tokens
    out["output_tokens"] += check.output_tokens
    out["model"] = out["model"] or check.model
    out["addresses_query"] = {"noul": addresses_p, "passed": passed, "cached": check.cached}
    if not passed:
        out.update({"abstained": True, "abstain_reason": "answer_off_target", "answer": ""})
    else:
        out.update({"abstained": False, "abstain_reason": None, "answer": final})
    return out
