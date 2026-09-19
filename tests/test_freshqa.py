"""FreshQA parsing and slicing on a synthetic export. No download."""

from __future__ import annotations

from harness import datasets
from harness.datasets import freshqa

EXPORT = """Warning: Please do not click thumbs up/down,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,
id,split,question,effective_year,next_review,false_premise,num_hops,fact_type,source,answer_0,answer_1,answer_2,answer_3,answer_4,answer_5,answer_6,answer_7,answer_8,answer_9,note
0,TEST,Who is the current mayor of X?,2026,weekly,FALSE,one-hop,fast-changing,https://x.example,Alice,A. Smith,,,,,,,,,
1,TEST,What year was X founded?,before 2022,occasionally,FALSE,one-hop,never-changing,https://x.example,1900,,,,,,,,,,
2,DEV,Who won the latest X cup?,2026,weekly,FALSE,one-hop,fast-changing,https://x.example,Team B,,,,,,,,,,
3,TEST,What did the first X on the moon eat?,before 2022,occasionally,TRUE,one-hop,fast-changing,https://x.example,No X has been on the moon,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,
"""


def test_parse_skips_preamble_and_collects_all_answers():
    rows = freshqa.parse_rows(EXPORT)
    assert [r["question_id"] for r in rows] == ["freshqa-0", "freshqa-1", "freshqa-2", "freshqa-3"]
    assert rows[0]["gold"] == "Alice" and rows[0]["golds"] == ["Alice", "A. Smith"]
    assert rows[0]["metadata"]["fact_type"] == "fast-changing" and rows[0]["metadata"]["false_premise"] is False
    assert rows[3]["metadata"]["false_premise"] is True


def test_select_takes_the_fast_changing_test_slice_and_caps_with_a_seed():
    rows = freshqa.parse_rows(EXPORT)
    picked = freshqa.select(rows)
    assert [r["question_id"] for r in picked] == ["freshqa-0", "freshqa-3"], "DEV and never-changing rows are out"
    assert [r["question_id"] for r in freshqa.select(rows, include_false_premise=False)] == ["freshqa-0"]
    one = freshqa.select(rows, n=1, seed=1)
    assert len(one) == 1 and one == freshqa.select(rows, n=1, seed=1), "seeded and reproducible"
    assert len(freshqa.select(rows, n=None, split=None, fact_types=None)) == 4


def test_registry_knows_both_loaders():
    assert datasets.module("freshqa") is freshqa
    assert datasets.module("simpleqa").__name__ == "harness.datasets.simpleqa"
    try:
        datasets.module("nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("unknown dataset must raise")
