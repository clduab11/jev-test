"""Section 6 post-processing and arm B evidence stuffing. No generator needed."""

from __future__ import annotations

from harness.arms import b_naive
from harness.stages import s3_generate as s3

EVIDENCE_IDS = {"r01c0", "r02c1", "r03c0"}


def test_abstain_token_detected_with_tolerance():
    assert s3.postprocess("INSUFFICIENT_EVIDENCE", EVIDENCE_IDS, keep_uncited=True)["abstained"]
    assert s3.postprocess("  INSUFFICIENT_EVIDENCE.\n", EVIDENCE_IDS, keep_uncited=True)["abstained"]
    assert not s3.postprocess("Paris is the capital [r01c0].", EVIDENCE_IDS, keep_uncited=True)["abstained"]


def test_claims_and_citations_are_extracted():
    out = s3.postprocess(
        "Paris is the capital of France [r01c0]. It has about 2.1 million residents [r01c0][r02c1].",
        EVIDENCE_IDS,
        keep_uncited=False,
    )
    assert not out["abstained"]
    assert [c["citations"] for c in out["claims"]] == [["r01c0"], ["r01c0", "r02c1"]]
    assert out["citations"] == ["r01c0", "r02c1"]
    assert out["fabricated_citations"] == 0
    assert out["answer"].startswith("Paris is the capital")


def test_citation_after_the_period_stays_with_its_sentence():
    out = s3.postprocess("Paris is the capital of France. [r01c0] Lyon is third. [r02c1]", EVIDENCE_IDS, keep_uncited=False)
    assert len(out["claims"]) == 2
    assert out["claims"][0]["text"].endswith("[r01c0]")


def test_fabricated_citation_strips_sentence_and_counts():
    out = s3.postprocess("Paris is the capital [r01c0]. Lyon has 500,000 people [r99c9].", EVIDENCE_IDS, keep_uncited=True)
    assert out["fabricated_citations"] == 1
    assert len(out["claims"]) == 1
    assert "Lyon" not in out["answer"]
    assert out["stripped"][0]["reason"] == "fabricated_citation"


def test_uncited_claims_kept_in_arm_b_and_stripped_otherwise():
    text = "Paris is the capital [r01c0]. The tower was finished in 1889. Sources agree on this."
    b = s3.postprocess(text, EVIDENCE_IDS, keep_uncited=True)
    d = s3.postprocess(text, EVIDENCE_IDS, keep_uncited=False)
    assert b["uncited_claims"] == ["The tower was finished in 1889."]
    assert "1889" in b["answer"]
    assert "1889" not in d["answer"]
    assert d["stripped"][0]["reason"] == "uncited_claim"
    # a sentence with no digit, date or proper noun is plain prose and is kept in both
    assert "Sources agree on this." in d["answer"]


def test_proper_noun_and_month_count_as_fact_shape():
    assert s3.has_fact_shape("The award went to Michio Sugeno.")
    assert s3.has_fact_shape("It was announced in September.")
    assert not s3.has_fact_shape("The sources disagree about this.")


def test_user_message_layout_matches_spec():
    msg = s3.build_user_message(
        "Q?", [{"id": "r01c0", "text": "one"}, {"id": "r02c1", "text": "two"}], [{"id": "r03c0", "text": "three"}]
    )
    assert msg == "EVIDENCE:\n[r01c0] one\n[r02c1] two\nCONFLICTS:\n[r03c0] three\n\nQUESTION: Q?"


def test_arm_b_stuffs_top_pages_in_order_within_budget(monkeypatch):
    monkeypatch.setattr(b_naive, "MAX_EVIDENCE_TOKENS", 700)
    replayed = {
        "results": [
            {"rank": 1, "snippet": "s1", "chunks": [
                {"cite_id": "r01c0", "text": "a", "drawer_id": "ev1", "n_tokens": 300},
                {"cite_id": "r01c1", "text": "b", "drawer_id": "ev2", "n_tokens": 300},
            ]},
            {"rank": 2, "snippet": "fallback snippet for page two", "chunks": []},
            {"rank": 3, "snippet": "s3", "chunks": [{"cite_id": "r03c0", "text": "c", "drawer_id": "ev3", "n_tokens": 300}]},
        ]
    }
    evidence = b_naive.build_evidence(replayed)
    assert [e["id"] for e in evidence] == ["r01c0", "r01c1", "r02s"]
    assert evidence[2]["drawer_id"] is None


def test_comma_separated_ids_in_one_bracket_are_citations():
    out = s3.postprocess("Metformin was introduced in 1957 [r01c0, r02c1]. See also [see above].", EVIDENCE_IDS, keep_uncited=False)
    assert out["claims"][0]["citations"] == ["r01c0", "r02c1"]
    assert out["fabricated_citations"] == 0
    assert s3.citation_ids("[r01c0,r03c0] and [r99]") == ["r01c0", "r03c0", "r99"]
    assert s3.citation_ids("[see above]") == []
