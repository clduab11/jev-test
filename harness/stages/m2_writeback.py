"""Stage M2: write verified claims back to memory (spec section 8).

No judge request in v1. Every claim S4 kept with support confidence at or
above ``memory_write_min`` becomes one drawer in the verified wing, filed
through ``palace.file_verified`` with its support probability, the judge model
and the evidence drawer ids it rests on. Claims below the bar are not stored;
abstentions store nothing.

The stored text is the claim sentence with its citation tags removed. The
tags are per-question ids that mean nothing on recall; the provenance lives
in ``evidence_drawer_ids``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from harness.config import thresholds
from harness.retrieval.palace import Palace, file_verified
from harness.stages.s3_generate import plain_sentence


def write_back(
    palace: Palace | None,
    kept: list[Mapping[str, Any]],
    *,
    query: str,
    query_id: str,
    arm: str,
    judge_model: str,
    cite_to_drawer: Mapping[str, str | None] | None = None,
    verified_at: str | None = None,
    t: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """File the kept claims that clear memory_write_min. Returns what was written."""
    if palace is None:
        return {"written": [], "skipped": [], "n_written": 0, "disabled": True}
    t = t or thresholds()
    floor = float(t["memory_write_min"])
    cite_to_drawer = cite_to_drawer or {}
    written: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for claim in kept:
        prob = float(claim["support"]["confidence"])
        text = plain_sentence(claim["text"])
        if prob < floor or not text:
            skipped.append({"text": text, "support_prob": prob, "reason": "below_memory_write_min"})
            continue
        evidence_ids = [cite_to_drawer.get(c) or f"cite:{c}" for c in claim["citations"]]
        drawer_id, created = file_verified(
            palace,
            text=text,
            query=query,
            query_id=query_id,
            arm=arm,
            support_prob=prob,
            support_choice=str(claim["support"]["choice"]),
            judge_model=judge_model,
            evidence_drawer_ids=evidence_ids,
            verified_at=verified_at,
        )
        written.append(
            {"drawer_id": drawer_id, "created": created, "text": text, "support_prob": prob}
        )
    return {"written": written, "skipped": skipped, "n_written": len(written), "disabled": False}
