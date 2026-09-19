"""Arm D: the full pipeline with a decision model making every call (spec
sections 2 to 8), replayed from a frozen snapshot.

    S0 intake ─► S1 on the 30 snippets ─► "fetch" the top max_pages survivors
    from the snapshot ─► S1 on their chunks ─► RRF ─► S2 ─► [refine once]
    ─► S3 Gemma ─► S4 per claim + answer level ─► M2 write-back

Two replay decisions the spec leaves open, both recorded in CHANGELOG.md:

* A refinement round cannot run a new SearXNG query offline. It searches the
  snapshot's evidence wing with the refined query instead, restricted to this
  question's own drawers, which is a second look at the same frozen result
  set with different words. Those hits join the ranking with a palace
  similarity, exactly as the spec's palace lane does.
* A survivor snippet whose page is beyond ``max_pages`` is not fetched and is
  not evidence; the spec only lets a snippet stand in for a page that failed
  to fetch. The count of such survivors is logged per question.

The judge is any object with the Judge protocol; arm D passes Jev over HTTP,
the C arms pass a local judge, and the tests pass a fake.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from harness.config import thresholds
from harness.judge.base import Judge
from harness.retrieval import fuse
from harness.retrieval.palace import (
    EVIDENCE,
    VERIFIED,
    Palace,
    domain_of,
    search_evidence_for_question,
    search_wing,
)
from harness.stages import m2_writeback, s0_intake, s1_gate, s2_sufficiency, s3_generate, s4_verify

ARM = "D"
JEV_USD_PER_MTOK_INPUT = 0.042  # output tokens are free (spec section 14)
CONFLICTS_NOTE = (
    "The EVIDENCE passages disagree with each other about the answer. "
    "Say that the sources disagree and give each version with its citation."
)


class _Usage:
    """Judge request and token totals, per stage and overall."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, int]] = {}

    def add(self, name: str, out: Mapping[str, Any]) -> None:
        entry = self.stages.setdefault(
            name, {"n_requests": 0, "n_cached": 0, "input_tokens": 0, "output_tokens": 0}
        )
        for key in entry:
            entry[key] += int(out.get(key) or 0)

    def as_dict(self) -> dict[str, Any]:
        totals = {"n_requests": 0, "n_cached": 0, "input_tokens": 0, "output_tokens": 0}
        for entry in self.stages.values():
            for key in totals:
                totals[key] += entry[key]
        return {
            **totals,
            "cost_usd": round(totals["input_tokens"] * JEV_USD_PER_MTOK_INPUT / 1_000_000, 6),
            "stages": self.stages,
        }


def snippet_id(rank: int) -> str:
    return f"r{rank:02d}s"


def snippet_passages(replayed: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for page in replayed["results"]:
        if not (page.get("snippet") or "").strip():
            continue
        out.append(
            s1_gate.make_passage(
                id=snippet_id(page["rank"]),
                text=page["snippet"],
                kind="snippet",
                lane="web",
                title=page.get("title"),
                domain=domain_of(page["url"]),
                engine_rank=int(page["rank"]),
                engines=list(page.get("engines") or []),
                url=page["url"],
                drawer_id=None,
                chunk_index=None,
                fetched_at=page.get("fetched_at"),
            )
        )
    return out


def chunk_passages(page: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        s1_gate.make_passage(
            id=chunk["cite_id"],
            text=chunk["text"],
            kind="chunk",
            lane="web",
            title=page.get("title"),
            domain=domain_of(page["url"]),
            engine_rank=int(page["rank"]),
            engines=list(page.get("engines") or []),
            url=page["url"],
            drawer_id=chunk.get("drawer_id"),
            chunk_index=chunk.get("chunk_index"),
            n_tokens=chunk.get("n_tokens"),
            fetched_at=page.get("fetched_at"),
        )
        for chunk in page.get("chunks") or []
    ]


def drawer_passage(hit: Mapping[str, Any], id: str, wing: str) -> dict[str, Any]:
    """A memory drawer as an S1 passage: domain 'palace', title = room, typed by wing."""
    return s1_gate.make_passage(
        id=id,
        text=hit["text"],
        kind="drawer",
        lane="palace",
        title=hit.get("room"),
        domain="palace",
        wing=wing,
        drawer_id=hit.get("drawer_id"),
        similarity=hit.get("similarity"),
        authored_at=hit.get("authored_at"),
        engine_rank=None,
        engines=[],
    )


def rank_and_select(
    evidence: list[dict[str, Any]], t: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fusion = fuse.fuse(evidence, k=t["rrf_k"])
    by_id = {p["id"]: p for p in evidence}
    selected = fuse.select(fusion["order"], by_id, t["top_evidence_k"], t["max_evidence_tokens"])
    for passage in selected:
        passage["rrf_score"] = fusion["scores"][passage["id"]]
    return selected, fusion


def _abstain(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "arm": ARM,
        "abstained": True,
        "abstain_reason": reason,
        "answer": "",
        "answer_raw": extra.pop("answer_raw", ""),
        "claims": [],
        "citations": [],
        "fabricated_citations": int(extra.pop("fabricated_citations", 0)),
        **extra,
    }


def _passage_summary(p: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": p["id"],
        "drawer_id": p.get("drawer_id"),
        "kind": p.get("kind"),
        "lane": p.get("lane"),
        "domain": p.get("domain"),
        "source_type": p.get("source_type"),
        "engine_rank": p.get("engine_rank"),
        "similarity": p.get("similarity"),
        "n_tokens": p.get("n_tokens"),
        "rrf_score": p.get("rrf_score"),
        "gate": (p.get("gate") or {}).get("answers"),
        "text": p["text"],
    }


def palace_lane(
    memory_palace: Palace, query: str, since: str | None, t: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Verified wing first, then evidence wing, from the memory palace (spec section 3)."""
    out: list[dict[str, Any]] = []
    for wing in (VERIFIED, EVIDENCE):
        hits = search_wing(memory_palace, query, wing, n=t["palace_top_k"], since=since)
        for hit in hits:
            out.append(drawer_passage(hit, f"m{len(out) + 1:02d}", wing))
    return out


def answer(
    query: str,
    question_id: str,
    replayed: Mapping[str, Any],
    judge: Judge,
    *,
    memory_palace: Palace | None = None,
    snapshot_palace: Palace | None = None,
    seed: int | None = None,
    today: str | None = None,
    force_web: bool = True,
    force_search: bool = True,
    use_palace_lane: bool = False,
    thinking: bool = False,
) -> dict[str, Any]:
    """Answer one question. ``replayed`` is ``snapshot.replay(...)`` output."""
    t = thresholds()
    started = time.perf_counter()
    today = today or time.strftime("%Y-%m-%d", time.gmtime())
    usage = _Usage()
    generator_s = 0.0

    # ---- S0 ---------------------------------------------------------------
    has_hits = bool(
        use_palace_lane and memory_palace is not None and s0_intake.palace_precheck(memory_palace, query)
    )
    s0 = s0_intake.run(judge, query, today, has_hits, force_search=force_search, force_web=force_web)
    usage.add("s0", s0)
    base: dict[str, Any] = {"question_id": question_id, "s0": s0, "today": today, "seed": seed}

    if not s0["retrieve"]:
        from harness.arms import a_plain

        plain = a_plain.answer(query, seed=seed, thinking=thinking)
        plain.update({**base, "arm": ARM, "searched": False, "judge": usage.as_dict()})
        return plain

    # ---- web lane: snippets, then chunks of the top survivors ---------------
    s1: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    judged_ids: set[str] = set()
    pages = {int(page["rank"]): page for page in replayed["results"]}

    if s0["source"] in ("web", "both"):
        snippets = snippet_passages(replayed)
        gate_snippets = s1_gate.gate(judge, query, snippets, quarantine_palace=memory_palace, t=t)
        usage.add("s1_snippets", gate_snippets)
        survivors = sorted(
            gate_snippets["evidence"] + gate_snippets["conflicts"], key=lambda p: p["engine_rank"]
        )
        fetch_ranks = [p["engine_rank"] for p in survivors][: int(t["max_pages"])]
        not_fetched = [p["engine_rank"] for p in survivors][int(t["max_pages"]) :]
        chunk_candidates: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        for survivor in survivors:
            if survivor["engine_rank"] not in fetch_ranks:
                continue
            page = pages[survivor["engine_rank"]]
            if page.get("fetched") and page.get("chunks"):
                chunk_candidates.extend(chunk_passages(page))
            else:
                fallback.append(survivor)  # failed fetch: the judged snippet stands in
        gate_chunks = s1_gate.gate(judge, query, chunk_candidates, quarantine_palace=memory_palace, t=t)
        usage.add("s1_chunks", gate_chunks)
        evidence += gate_chunks["evidence"] + [p for p in fallback if p["gate"]["route"] == "EVIDENCE"]
        conflicts += gate_chunks["conflicts"] + [p for p in fallback if p["gate"]["route"] == "CONFLICT"]
        judged_ids |= {p["id"] for p in gate_chunks["evidence"] + gate_chunks["conflicts"] + gate_chunks["dropped"]}
        judged_ids |= {p["id"] for p in fallback}
        s1["snippets"] = {
            "n": len(snippets),
            "reasons": gate_snippets["reasons"],
            "survivors": [p["engine_rank"] for p in survivors],
            "fetched": fetch_ranks,
            "not_fetched": not_fetched,
            "fallback_snippets": [p["id"] for p in fallback],
        }
        s1["chunks"] = {
            "n": len(chunk_candidates),
            "reasons": gate_chunks["reasons"],
            "dropped": [{"id": p["id"], "reason": p["gate"]["reason"]} for p in gate_chunks["dropped"]],
        }

    # ---- palace lane (memory track pass two) --------------------------------
    if s0["source"] in ("palace", "both") and use_palace_lane and memory_palace is not None:
        drawers = palace_lane(memory_palace, query, s0.get("since"), t)
        gate_drawers = s1_gate.gate(judge, query, drawers, quarantine_palace=memory_palace, t=t)
        usage.add("s1_drawers", gate_drawers)
        evidence += gate_drawers["evidence"]
        conflicts += gate_drawers["conflicts"]
        s1["drawers"] = {"n": len(drawers), "reasons": gate_drawers["reasons"]}

    # ---- rank, judge sufficiency, refine once -----------------------------
    round_index = 0
    s2_rounds: list[dict[str, Any]] = []
    refine: dict[str, Any] | None = None
    selected: list[dict[str, Any]] = []
    fusion: dict[str, Any] = {}
    while True:
        selected, fusion = rank_and_select(evidence, t)
        if selected:
            s2 = s2_sufficiency.run(judge, query, selected, round_index)
            usage.add("s2", s2)
        else:  # nothing to judge; a request would only confirm the obvious
            s2 = {
                "action": "refine" if round_index < int(t["max_refine_rounds"]) else "abstain",
                "round": round_index,
                "sufficient": False,
                "sufficient_p": None,
                "conflicts": False,
                "conflicts_p": None,
                "evidence_ids": [],
                "reason": "no_evidence",
            }
        s2_rounds.append(s2)
        if s2["action"] == "generate":
            break
        if s2["action"] == "abstain":
            return _abstain(
                "insufficient_evidence",
                **base,
                searched=True,
                s1=s1,
                s2=s2_rounds,
                refine=refine,
                evidence=[_passage_summary(p) for p in selected],
                evidence_ids=[p["id"] for p in selected],
                conflicts_block=[_passage_summary(p) for p in conflicts],
                fusion=fusion.get("ranks"),
                judge=usage.as_dict(),
                generator_s=round(generator_s, 3),
                latency_s=round(time.perf_counter() - started, 3),
            )
        # refine: the generator writes one alternative query
        refine = s2_sufficiency.refine_query(query, seed=seed, thinking=thinking)
        generator_s += float(refine.get("latency_s") or 0.0)
        new_passages: list[dict[str, Any]] = []
        refine["hits"] = 0
        if snapshot_palace is not None:
            known = {
                chunk["drawer_id"]: (page, chunk)
                for page in replayed["results"]
                for chunk in page.get("chunks") or []
            }
            hits = search_evidence_for_question(
                snapshot_palace, refine["query"], question_id, n=t["palace_top_k"], max_distance=t["palace_max_distance"]
            )
            refine["hits"] = len(hits)
            by_id = {p["id"]: p for p in evidence + conflicts}
            for index, hit in enumerate(hits, 1):
                if hit["drawer_id"] in known:
                    page, chunk = known[hit["drawer_id"]]
                    pid = chunk["cite_id"]
                    if pid in judged_ids:  # verdict stands; it just gains a palace similarity
                        if pid in by_id:
                            by_id[pid]["similarity"] = hit["similarity"]
                            by_id[pid]["lane"] = "both"
                        continue
                    passage = chunk_passages({**page, "chunks": [chunk]})[0]
                    passage.update({"similarity": hit["similarity"], "lane": "palace"})
                else:
                    passage = drawer_passage(hit, f"p{index:02d}", EVIDENCE)
                    passage.update({"domain": domain_of(hit["url"]) if hit.get("url") else "palace"})
                    passage["source_type"] = s1_gate.source_type_for(passage["domain"], None)
                new_passages.append(passage)
        gate_new = s1_gate.gate(judge, query, new_passages, quarantine_palace=memory_palace, t=t)
        usage.add("s1_refine", gate_new)
        evidence += gate_new["evidence"]
        conflicts += gate_new["conflicts"]
        judged_ids |= {p["id"] for p in new_passages}
        refine["judged"] = len(new_passages)
        refine["reasons"] = gate_new["reasons"]
        round_index += 1

    # ---- S3 -----------------------------------------------------------------
    s2_final = s2_rounds[-1]
    conflict_block = sorted(
        conflicts, key=lambda p: -float(p["gate"]["answers"].get("contradicts_query_premise", 0.0))
    )[: int(t["top_evidence_k"])]
    s3 = s3_generate.generate_answer(
        query,
        [{"id": p["id"], "text": p["text"]} for p in selected],
        conflicts=[{"id": p["id"], "text": p["text"]} for p in conflict_block],
        conflicts_note=CONFLICTS_NOTE if s2_final.get("conflicts") else None,
        seed=seed,
        keep_uncited=False,
        thinking=thinking,
    )
    generator_s += float(s3.get("latency_s") or 0.0)
    common = {
        **base,
        "searched": True,
        "s1": s1,
        "s2": s2_rounds,
        "refine": refine,
        "evidence": [_passage_summary(p) for p in selected],
        "evidence_ids": s3["evidence_ids"],
        "evidence_texts": {p["id"]: p["text"] for p in selected + conflict_block},
        "conflicts_block": [_passage_summary(p) for p in conflict_block],
        "fusion": fusion.get("ranks"),
        "answer_raw": s3["answer_raw"],
        "claims_generated": s3["claims"],
        "uncited_claims": s3["uncited_claims"],
        "fabricated_citations": s3["fabricated_citations"],
        "generator_model": s3["generator_model"],
        "thinking": s3["thinking"],
        "prompt_tokens": s3["prompt_tokens"],
        "completion_tokens": s3["completion_tokens"],
        "finish_reason": s3["finish_reason"],
    }
    if s3["abstained"]:
        return _abstain(
            "generator_insufficient_evidence",
            **common,
            s3_stripped=s3["stripped"],
            judge=usage.as_dict(),
            generator_s=round(generator_s, 3),
            latency_s=round(time.perf_counter() - started, 3),
        )

    # ---- S4 -----------------------------------------------------------------
    evidence_by_id = {p["id"]: p["text"] for p in selected + conflict_block}
    s4 = s4_verify.verify(
        judge, query, s3["claims"], evidence_by_id, kept_sentences=s3["kept_sentences"], answer=s3["answer"], t=t
    )
    usage.add("s4", s4)

    # ---- M2 -----------------------------------------------------------------
    cite_to_drawer = {p["id"]: p.get("drawer_id") for p in selected + conflict_block}
    m2 = m2_writeback.write_back(
        memory_palace,
        s4["kept"],
        query=query,
        query_id=question_id,
        arm=ARM,
        judge_model=str(s4.get("model") or s0.get("model") or getattr(judge, "model", "")),
        cite_to_drawer=cite_to_drawer,
        t=t,
    )

    record = {
        "arm": ARM,
        **common,
        "abstained": s4["abstained"],
        "abstain_reason": s4["abstain_reason"],
        "answer": s4["answer"],
        "claims": s4["kept"],
        "citations": sorted({c for claim in s4["kept"] for c in claim["citations"]}),
        "conflicts": s4["conflicts"],
        "stripped": s3["stripped"] + [{"sentence": e["text"], "reason": e["reason"], "support": e["support"]} for e in s4["stripped"]],
        "addresses_query": s4["addresses_query"],
        "s4": {"n_requests": s4["n_requests"], "verdicts": [{"text": v["text"], "verdict": v["verdict"], "support": v["support"]} for v in s4["claim_verdicts"]]},
        "m2": {"n_written": m2["n_written"], "written": m2["written"], "skipped": m2["skipped"], "disabled": m2.get("disabled", False)},
        "judge": usage.as_dict(),
        "generator_s": round(generator_s, 3),
        "latency_s": round(time.perf_counter() - started, 3),
    }
    return record
