"""Sentence splitting around abbreviations and initials (found on the arm D pilot:
"Dr. X ... Dr. Y ..." split into "Dr." fragments that outlived their claims)."""

from __future__ import annotations

from harness.stages import s3_generate as s3

IDS = {"r08c2", "r11c0", "r11c1", "r01c0", "r02c0"}


def test_titles_stay_attached_to_their_sentences():
    text = "Dr. Abdul Ahad Guru is a pioneer [r11c1]. Dr. Upendra Kaul was the first [r08c2]."
    assert s3.split_sentences(text) == [
        "Dr. Abdul Ahad Guru is a pioneer [r11c1].",
        "Dr. Upendra Kaul was the first [r08c2].",
    ]


def test_initials_and_common_abbreviations_do_not_split():
    assert s3.split_sentences("J. R. R. Tolkien wrote it in 1937 [r01c0]. It sold well [r02c0].") == [
        "J. R. R. Tolkien wrote it in 1937 [r01c0].",
        "It sold well [r02c0].",
    ]
    assert s3.split_sentences("He lived in the U.S. for years [r01c0]. Then he moved.") == [
        "He lived in the U.S. for years [r01c0].",
        "Then he moved.",
    ]
    assert s3.split_sentences("Costs rose, e.g. rent [r01c0]. Wages did not.") == [
        "Costs rose, e.g. rent [r01c0].",
        "Wages did not.",
    ]


def test_ordinary_sentences_still_split():
    assert s3.split_sentences("Paris is the capital. Lyon is third. Done!") == [
        "Paris is the capital.",
        "Lyon is third.",
        "Done!",
    ]


def test_pilot_answer_keeps_no_orphan_fragments():
    raw = (
        "Dr. Abdul Ahad Guru is remembered as a pioneer cardiologist of Kashmir [r11c1]. "
        "Dr. Abdul Ahad Guru (1937-1993) was one of the most respected cardiologists [r11c0]. "
        "Dr. Upendra Kaul was the first qualified cardiologist of Kashmiri origin [r08c2]."
    )
    out = s3.postprocess(raw, IDS, keep_uncited=False)
    assert len(out["claims"]) == 3 and out["kept_sentences"] == [c["text"] for c in out["claims"]]
    assert all(c["text"].startswith("Dr. ") for c in out["claims"])
    # what S4 would rebuild after stripping the first two claims
    survivors = [s for s in out["kept_sentences"] if s == out["claims"][2]["text"]]
    assert " ".join(survivors) == "Dr. Upendra Kaul was the first qualified cardiologist of Kashmiri origin [r08c2]."
