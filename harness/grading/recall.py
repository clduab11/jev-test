"""Gold-answer recall in the frozen evidence: the ceiling every arm is measured against.

The spec asks for this twice (section 11's predicted negatives, and
``prereg_predicted_negatives`` in judge/questions.v1.json): "no arm exceeds the
retriever's recall ceiling; report recall@30 alongside". Nothing computed it until
now, so every number the project has published has been missing the one piece of
context that says how much of the result was ever available to win.

What this measures: whether the gold answer string appears verbatim in the frozen
evidence for a question, after light normalisation. It runs offline against the
manifest and the palace, touches no endpoint, and takes seconds.

What it does not measure, and why the number is a bound rather than a truth:

  * It UNDERCOUNTS. An answer stated in other words ("the fourteenth of March"
    for "March 14, 1916") does not match, so a question can be answerable from
    evidence this counts as a miss. Real recall is at least this number.
  * It OVERCOUNTS on short answers. A gold answer of "150" matches any page
    containing "150" for any reason. Recall for short numeric answers is
    reported separately so the reader can discount it.

Both directions are reported rather than corrected, because a grader call per
question would cost more than the measurement is worth and would put a model in
the loop of a number whose whole purpose is to be model-free.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from harness.config import ROOT

TOP_PAGES_B = 8  # arm B reads this many pages; see harness/arms/b_naive.py
SHORT_ANSWER_CHARS = 4


def normalize(text: str) -> str:
    """Casefold, strip accents, collapse whitespace and drop punctuation runs."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains(haystack: str, gold: str) -> bool:
    """Whether the normalised gold string appears in the normalised haystack."""
    gold_n = normalize(gold)
    if not gold_n:
        return False
    return gold_n in haystack


def palace_documents(palace_dir: Path) -> dict[str, str]:
    """Every drawer's text, read straight from the palace with read-only sqlite.

    Deliberately avoids importing chroma: this must work on a palace another
    process is reading, must not load an embedding model, and must not be able
    to write anything.
    """
    db = palace_dir / "chroma.sqlite3"
    uri = f"file:{db.as_posix()}?mode=ro"
    out: dict[str, str] = {}
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT e.embedding_id, m.string_value FROM embeddings e "
            "JOIN embedding_metadata m ON m.id = e.id AND m.key = 'chroma:document'"
        )
        for drawer_id, document in rows:
            if document:
                out[str(drawer_id)] = str(document)
    return out


def measure(snapshot_id: str, dataset: str = "simpleqa") -> dict[str, Any]:
    """Recall of the gold answer in the frozen evidence, per question and overall."""
    from harness import datasets
    from harness.retrieval.snapshot import load_manifest, palace_path

    manifest = load_manifest(snapshot_id)
    question_ids = [q["question_id"] for q in manifest["questions"]]
    rows = datasets.module(dataset).by_id(question_ids)
    documents = palace_documents(palace_path(snapshot_id))

    per_question: list[dict[str, Any]] = []
    for entry in manifest["questions"]:
        qid = entry["question_id"]
        row = rows.get(qid)
        if row is None:
            continue
        golds = row.get("golds") or [row["gold"]]
        snippets = normalize(" ".join(f"{r.get('title', '')} {r.get('snippet', '')}" for r in entry["results"]))
        pages = entry["pages"]
        chunk_text_all = normalize(
            " ".join(documents.get(c["drawer_id"], "") for p in pages for c in p["chunks"])
        )
        chunk_text_top8 = normalize(
            " ".join(
                documents.get(c["drawer_id"], "")
                for p in pages
                if int(p["rank"]) <= TOP_PAGES_B
                for c in p["chunks"]
            )
        )
        hit = any(contains(chunk_text_all, g) or contains(snippets, g) for g in golds)
        per_question.append(
            {
                "question_id": qid,
                "gold": row["gold"],
                "n_results": entry["n_results"],
                "n_chunks": entry["n_chunks"],
                "in_any_evidence": hit,
                "in_chunks": any(contains(chunk_text_all, g) for g in golds),
                "in_snippets_only": any(contains(snippets, g) for g in golds)
                and not any(contains(chunk_text_all, g) for g in golds),
                "in_top8_pages": any(contains(chunk_text_top8, g) for g in golds),
                "short_answer": len(normalize(row["gold"])) <= SHORT_ANSWER_CHARS,
            }
        )

    n = len(per_question)
    hits = [q for q in per_question if q["in_any_evidence"]]
    short = [q for q in per_question if q["short_answer"]]
    long_answers = [q for q in per_question if not q["short_answer"]]
    empty = [q for q in per_question if q["n_chunks"] == 0]
    return {
        "snapshot": snapshot_id,
        "n": n,
        "recall": len(hits) / n if n else float("nan"),
        "recall_in_chunks": sum(1 for q in per_question if q["in_chunks"]) / n if n else float("nan"),
        "recall_top8_pages": sum(1 for q in per_question if q["in_top8_pages"]) / n if n else float("nan"),
        "recall_long_answers_only": (
            sum(1 for q in long_answers if q["in_any_evidence"]) / len(long_answers) if long_answers else float("nan")
        ),
        "n_short_answers": len(short),
        "n_zero_chunk_questions": len(empty),
        "mean_results": sum(q["n_results"] for q in per_question) / n if n else float("nan"),
        "per_question": per_question,
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [
        f"Gold-answer recall in the frozen evidence, snapshot {result['snapshot']}, n={result['n']}",
        "",
        "This is the ceiling: no arm can answer a question correctly from evidence that",
        "does not contain the answer. It counts a verbatim string match after light",
        "normalisation, so it undercounts paraphrase and overcounts very short answers.",
        "",
        f"  recall, anywhere in the evidence      {result['recall']:.3f}",
        f"  recall, in fetched page text only     {result['recall_in_chunks']:.3f}",
        f"  recall, within the top 8 pages        {result['recall_top8_pages']:.3f}   (what arm B reads)",
        f"  recall, excluding short answers       {result['recall_long_answers_only']:.3f}"
        f"   ({result['n_short_answers']} answers of {SHORT_ANSWER_CHARS} characters or fewer excluded)",
        "",
        f"  questions with no fetched text at all  {result['n_zero_chunk_questions']}",
        f"  mean search results per question       {result['mean_results']:.1f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from harness.config import configure_stdout

    configure_stdout()
    parser = argparse.ArgumentParser(description="gold-answer recall in a frozen snapshot")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--dataset", default="simpleqa")
    parser.add_argument("--out", default=None, help="write the full per-question result to this JSON file")
    args = parser.parse_args(argv)

    result = measure(args.snapshot, args.dataset)
    print(format_report(result))
    if args.out:
        path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
