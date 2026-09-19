"""Stage S2 policy: generate, refine once, then abstain; the conflicts flag; query cleaning."""

from __future__ import annotations

from harness.config import thresholds
from harness.stages import s2_sufficiency as s2
from tests.fakes import FakeJudge, noul

T = thresholds()


def answers(sufficient=0.9, conflicts=0.1):
    return {"evidence_sufficient": noul(sufficient), "evidence_conflicts": noul(conflicts)}


def test_state_is_query_plus_id_text_pairs_only():
    evidence = [{"id": "r01c0", "text": "a", "drawer_id": "ev1", "n_tokens": 3}]
    assert s2.build_state("Q?", evidence) == {"query": "Q?", "evidence": [{"id": "r01c0", "text": "a"}]}


def test_decide_generate_refine_abstain():
    assert s2.decide(answers(), 0)["action"] == "generate"
    assert s2.decide(answers(sufficient=T["sufficient_min"]), 0)["action"] == "generate", "at the bar counts"
    low = answers(sufficient=T["sufficient_min"] - 0.01)
    assert s2.decide(low, 0)["action"] == "refine"
    assert s2.decide(low, T["max_refine_rounds"])["action"] == "abstain"


def test_conflicts_flag_uses_conflicts_min():
    assert s2.decide(answers(conflicts=T["conflicts_min"]), 0)["conflicts"] is True
    assert s2.decide(answers(conflicts=T["conflicts_min"] - 0.01), 0)["conflicts"] is False


def test_run_makes_one_request_and_records_ids():
    judge = FakeJudge(lambda state, qs: answers(sufficient=0.2))
    out = s2.run(judge, "Q?", [{"id": "r01c0", "text": "a"}, {"id": "r02c0", "text": "b"}], round_index=0)
    assert out["n_requests"] == 1 and out["action"] == "refine"
    assert out["evidence_ids"] == ["r01c0", "r02c0"]
    assert set(judge.calls[0][1]) == {"evidence_sufficient", "evidence_conflicts"}


def test_clean_query_takes_one_line_and_drops_labels_and_quotes():
    assert s2.clean_query('Query: "Dumoulin Robinson Crusoe engravings count"\nsecond line') == "Dumoulin Robinson Crusoe engravings count"
    assert s2.clean_query("\n\n  `who  painted   it`  ") == "who painted it"
    assert s2.clean_query("") == ""
    assert len(s2.clean_query("x" * 500)) == 200


def test_refine_query_uses_the_generator_and_flags_repeats(monkeypatch):
    class Gen:
        text = "Dumoulin 1810 Robinson Crusoe plates"
        model = "gemma"
        latency_s = 0.1
        prompt_tokens = 30
        completion_tokens = 8

    monkeypatch.setattr(s2, "generate", lambda *a, **k: Gen())
    out = s2.refine_query("How many engravings did Dumoulin publish?", seed=1)
    assert out["query"] == "Dumoulin 1810 Robinson Crusoe plates" and out["same_as_original"] is False
    Gen.text = "how many engravings did dumoulin publish?"
    assert s2.refine_query("How many engravings did Dumoulin publish?")["same_as_original"] is True
    Gen.text = "   "
    assert s2.refine_query("Original?")["query"] == "Original?", "an empty reply falls back to the original"
