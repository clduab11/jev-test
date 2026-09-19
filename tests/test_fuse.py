"""Reciprocal rank fusion: the arithmetic, ties, absent lists, and the caps."""

from __future__ import annotations

from harness.retrieval import fuse


def test_rrf_arithmetic_with_k_60():
    a = {"x": 1, "y": 2}
    b = {"y": 1, "z": 1}
    fused = dict(fuse.rrf([a, b], k=60))
    assert fused["x"] == 1 / 61
    assert fused["y"] == 1 / 62 + 1 / 61
    assert fused["z"] == 1 / 61
    order = [pid for pid, _ in fuse.rrf([a, b], k=60)]
    assert order[0] == "y", "present in both lists beats top of one list"


def test_absent_from_a_list_gets_nothing_and_ties_break_deterministically():
    fused = fuse.rrf([{"a": 1}, {"b": 1}], k=60)
    assert [pid for pid, _ in fused] == ["a", "b"]  # equal score, equal best rank, id order


def test_competition_ranks_share_rank_on_ties():
    ranks = fuse.competition_ranks({"a": 0.9, "b": 0.9, "c": 0.5}, descending=True)
    assert ranks == {"a": 1, "b": 1, "c": 3}
    ranks = fuse.competition_ranks({"a": (1, 0), "b": (2, -3), "c": (2, -1)}, descending=False)
    assert ranks == {"a": 1, "b": 2, "c": 3}  # same position, more engines wins


def test_list_builders_read_passage_fields():
    passages = [
        {"id": "r01c0", "engine_rank": 1, "engines": ["a", "b"], "gate": {"answers": {"contains_answer_evidence": 0.6}}},
        {"id": "r01c1", "engine_rank": 1, "engines": ["a", "b"], "gate": {"answers": {"contains_answer_evidence": 0.9}}},
        {"id": "r03c0", "engine_rank": 3, "engines": ["a"], "similarity": 0.8, "gate": {"answers": {"contains_answer_evidence": 0.7}}},
        {"id": "p01", "engine_rank": None, "similarity": 0.9, "gate": {"answers": {"contains_answer_evidence": 0.5}}},
    ]
    assert fuse.retriever_ranks(passages) == {"r01c0": 1, "r01c1": 1, "r03c0": 3}
    assert fuse.palace_ranks(passages) == {"r03c0": 2, "p01": 1}
    assert fuse.evidence_ranks(passages) == {"r01c1": 1, "r03c0": 2, "r01c0": 3, "p01": 4}
    out = fuse.fuse(passages, k=60)
    assert out["order"][:2] == ["r03c0", "r01c1"], "three lists beat two; then top page with top evidence"
    assert set(out["ranks"]) == {"retriever", "palace", "evidence"}


def test_select_applies_top_k_then_token_budget():
    by_id = {f"p{i}": {"id": f"p{i}", "text": "t", "n_tokens": 300} for i in range(10)}
    order = [f"p{i}" for i in range(10)]
    assert [p["id"] for p in fuse.select(order, by_id, top_k=6, max_tokens=2500)] == order[:6]
    assert [p["id"] for p in fuse.select(order, by_id, top_k=6, max_tokens=700)] == order[:2]
    assert fuse.select(["missing"], by_id, top_k=6, max_tokens=2500) == []


def test_select_counts_tokens_when_missing():
    by_id = {"a": {"id": "a", "text": "one two three"}}
    chosen = fuse.select(["a"], by_id, top_k=6, max_tokens=2500, count=lambda text: len(text.split()))
    assert chosen[0]["n_tokens"] == 3
