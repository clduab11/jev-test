"""Citation support grading: the prompt, the label parser, the rate, and the bars."""

from __future__ import annotations

import math

from harness.grading import claim_support, prereg


def test_prompt_lists_only_cited_passages_and_strips_tags():
    prompt = claim_support.build_prompt(
        "When was X born?",
        "X was born in 1950 [r01c0].",
        [{"id": "r01c0", "text": "X was born in 1950 in Lyon."}],
    )
    assert "Sentence to check: X was born in 1950." in prompt
    assert "[r01c0] X was born in 1950 in Lyon." in prompt
    assert "SUPPORTED" in prompt and "UNSUPPORTED" in prompt


def test_parse_label_checks_the_negative_first():
    assert claim_support.parse_label("SUPPORTED") == "supported"
    assert claim_support.parse_label("UNSUPPORTED") == "unsupported"
    assert claim_support.parse_label("Not supported.") == "unsupported"
    assert claim_support.parse_label("") == "unsupported"


def test_claim_evidence_uses_the_record_texts():
    record = {"evidence_texts": {"r01c0": "a", "r02c0": "b"}}
    claim = {"citations": ["r02c0", "r09c9"]}
    assert claim_support.claim_evidence(record, claim) == [{"id": "r02c0", "text": "b"}]


def test_support_rate_pools_claims_and_bootstraps_over_questions():
    out = claim_support.support_rate([["supported", "supported"], ["unsupported"], [], ["supported"]])
    assert out["kept_claims"] == 4 and out["supported"] == 3 and out["rate"] == 0.75
    assert out["n_questions"] == 4 and out["n_questions_with_claims"] == 3
    lo, hi = out["rate_ci"]
    assert 0.0 <= lo <= 0.75 <= hi <= 1.0
    empty = claim_support.support_rate([[], []])
    assert math.isnan(empty["rate"]) and math.isnan(empty["rate_ci"][0])


def test_prereg_bars_evaluate_and_format():
    d = {"coverage": 0.6, "coverage_ci": [0.4, 0.8], "hallucination_rate": 0.05, "hallucination_rate_ci": [0.0, 0.2], "truthfulness": 0.5}
    b = {"truthfulness": 0.4}
    rows = prereg.evaluate(d, b, {"rate": 0.95, "rate_ci": [0.9, 1.0]})
    assert [r["passed"] for r in rows] == [True, True, False, True]
    assert rows[2]["value"] == 0.5 - 0.4
    text = prereg.format_table(rows, "PILOT", n=20)
    assert "not a finding" in text and "3 of 4 bars cleared" in text
    rows = prereg.evaluate({"coverage": 0.0, "hallucination_rate": math.nan, "truthfulness": 0.0}, None, None)
    assert not any(r["passed"] for r in rows)
    assert "n/a" in prereg.format_table(rows, "PILOT")
