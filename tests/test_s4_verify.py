"""Stage S4: verdicts, answer rebuilding, and the abstain paths."""

from __future__ import annotations

from harness.config import thresholds
from harness.stages import s4_verify as s4
from tests.fakes import FakeJudge, choice, noul

T = thresholds()
OPTIONS = ("supported", "contradicted", "unsupported")
EVIDENCE = {"r01c0": "X was born in 1950 in Lyon.", "r02c0": "X moved to Paris in 1970."}


def test_verdict_table():
    accept = T["support_accept"]
    assert s4.verdict(choice("supported", accept, OPTIONS)) == "keep"
    assert s4.verdict(choice("supported", accept - 0.01, OPTIONS)) == "strip"
    assert s4.verdict(choice("contradicted", accept, OPTIONS)) == "conflict"
    assert s4.verdict(choice("contradicted", accept - 0.01, OPTIONS)) == "strip"
    assert s4.verdict(choice("unsupported", 0.99, OPTIONS)) == "strip"


def _judge(claim_verdicts, addresses=0.9):
    def fn(state, qs):
        if "claim" in state:
            name, conf = claim_verdicts[state["claim"]]
            return {"support": choice(name, conf, OPTIONS)}
        return {"answer_addresses_query": noul(addresses)}

    return FakeJudge(fn)


CLAIMS = [
    {"text": "X was born in 1950 [r01c0].", "citations": ["r01c0"]},
    {"text": "X moved to Paris in 1970 [r02c0].", "citations": ["r02c0"]},
    {"text": "X died in 2001 [r02c0].", "citations": ["r02c0"]},
]
SENTENCES = [c["text"] for c in CLAIMS] + ["Sources agree on this."]


def test_keep_strip_conflict_and_rebuild():
    judge = _judge({
        CLAIMS[0]["text"]: ("supported", 0.95),
        CLAIMS[1]["text"]: ("unsupported", 0.9),
        CLAIMS[2]["text"]: ("contradicted", 0.9),
    })
    out = s4.verify(judge, "Q?", CLAIMS, EVIDENCE, kept_sentences=SENTENCES)
    assert out["abstained"] is False
    assert [k["text"] for k in out["kept"]] == [CLAIMS[0]["text"]]
    assert [c["text"] for c in out["conflicts"]] == [CLAIMS[2]["text"]]
    assert [s["reason"] for s in out["stripped"]] == ["unsupported"]
    assert out["answer"] == "X was born in 1950 [r01c0]. Sources agree on this."
    assert out["n_requests"] == 4, "three claims plus one answer-level request"
    assert out["addresses_query"]["passed"] is True
    # the claim state carries only the cited passage, never the whole evidence block
    claim_states = judge.states("s4")
    assert claim_states[0]["cited_evidence"] == [{"id": "r01c0", "text": EVIDENCE["r01c0"]}]
    assert claim_states[1]["cited_evidence"] == [{"id": "r02c0", "text": EVIDENCE["r02c0"]}]


def test_low_confidence_supported_is_stripped_with_its_reason():
    judge = _judge({c["text"]: ("supported", T["support_accept"] - 0.05) for c in CLAIMS})
    out = s4.verify(judge, "Q?", CLAIMS, EVIDENCE, kept_sentences=SENTENCES)
    assert out["abstained"] is True and out["abstain_reason"] == "no_claims_kept"
    assert {s["reason"] for s in out["stripped"]} == {"supported_low_confidence"}
    assert out["n_requests"] == 3, "no answer-level request when nothing is kept"
    assert out["addresses_query"] is None and out["answer"] == ""


def test_answer_off_target_abstains_after_the_answer_level_check():
    judge = _judge({c["text"]: ("supported", 0.95) for c in CLAIMS}, addresses=T["addresses_min"] - 0.01)
    out = s4.verify(judge, "Q?", CLAIMS, EVIDENCE, kept_sentences=SENTENCES)
    assert out["abstained"] is True and out["abstain_reason"] == "answer_off_target"
    assert len(out["kept"]) == 3 and out["n_requests"] == 4
    assert judge.states("s4a")[0]["answer"].startswith("X was born")


def test_no_claims_means_no_requests_and_abstain():
    judge = _judge({})
    out = s4.verify(judge, "Q?", [], EVIDENCE, kept_sentences=["Just prose."])
    assert out["abstained"] and out["n_requests"] == 0 and judge.calls == []


def test_rebuild_falls_back_to_splitting_the_answer():
    judge = _judge({CLAIMS[0]["text"]: ("supported", 0.9)})
    out = s4.verify(judge, "Q?", CLAIMS[:1], EVIDENCE, answer="X was born in 1950 [r01c0]. Plain prose.")
    assert out["answer"] == "X was born in 1950 [r01c0]. Plain prose."
