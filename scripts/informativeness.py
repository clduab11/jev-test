#!/usr/bin/env python3
"""informativeness.py - does a higher judge score mean a better passage?

Gate mobility says whether a threshold on a judge's score can move. It can't say whether
moving it helps, because random noise moves beautifully. This script asks the question
that separates signal from noise, using a label no model produced.

THE LABEL. For each S1 judgment of `contains_answer_evidence`, the passage is labelled
positive when the gold answer string appears in its text, using the normaliser from
harness/grading/recall.py. No grader is involved. Its known biases come from recall.py's
own docstring: it undercounts answers stated in other words, and it overcounts very short
answers, so golds of SHORT_ANSWER_CHARS or fewer are dropped.

THE THREE TESTS
  A  Jev alone, every question: AUROC of its score against the label.
  B  Same passage, both judges (the self-judging local model ran the pilot only).
  C  Inside each bucket of the local model's score, can Jev's score still rank?
     Where the local model says 1.00, it claims certainty. If Jev separates right from
     wrong inside that set, its extra resolution is information, not noise.

Intervals resample whole questions, since passages from one question move together.
Buckets with fewer than MIN_PER_CLASS positives or negatives are reported as too small.

Usage:  python scripts/informativeness.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from harness.audit import auroc, auroc_with_ci, bootstrap_ci, to_grid
from harness.grading.recall import SHORT_ANSWER_CHARS, normalize

QUESTION = "contains_answer_evidence"
MIN_PER_CLASS = 20


def gold_by_query(root: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    for path in glob.glob(str(root / "results" / "*_simpleqa_*.json")):
        for record in json.loads(Path(path).read_text(encoding="utf-8")).get("records", []):
            if record.get("query") and record.get("gold"):
                gold[record["query"]] = record["gold"]
    return gold


def labelled(path: Path, gold: dict[str, str]) -> dict[str, tuple[float, bool, str]]:
    """state_id -> (score, label, query) for every usable S1 evidence judgment."""
    out: dict[str, tuple[float, bool, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["question_id"] != QUESTION or row["value"] is None:
                continue
            state = row["state"] or {}
            query = state.get("query")
            answer = gold.get(query)
            if not answer or len(normalize(answer)) <= SHORT_ANSWER_CHARS:
                continue
            text = (state.get("passage") or {}).get("text", "")
            out[row["state_id"]] = (float(row["value"]), normalize(answer) in normalize(text), query)
    return out


def fmt(result: dict) -> str:
    ci = result.get("ci")
    interval = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    return (
        f"{result['auroc']:.3f}{interval}  "
        f"(pos {result['positives']}, neg {result['negatives']}, questions {result.get('clusters', '-')})"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    gold = gold_by_query(root)
    jev = labelled(root / "corpus/raw/jev-1.13.0_triples.jsonl", gold)
    local = labelled(root / "corpus/raw/gemma-self_triples.jsonl", gold)
    report: dict = {
        "label": "gold answer string appears in passage text (harness/grading/recall.py normalize)",
        "short_answer_chars_excluded": SHORT_ANSWER_CHARS,
        "question": QUESTION,
    }

    # A - Jev alone
    keys = sorted(jev)
    a = auroc_with_ci([jev[k][0] for k in keys], [jev[k][1] for k in keys], [jev[k][2] for k in keys])
    report["A_jev_all_questions"] = a
    print(f"A  Jev, all questions            AUROC {fmt(a)}")

    # B - paired
    keys = sorted(set(jev) & set(local))
    labels = [jev[k][1] for k in keys]
    clusters = [jev[k][2] for k in keys]
    j_scores = [jev[k][0] for k in keys]
    l_scores = [local[k][0] for k in keys]
    b_jev = auroc_with_ci(j_scores, labels, clusters)
    b_local = auroc_with_ci(l_scores, labels, clusters)
    diff_ci = bootstrap_ci(
        lambda idx: auroc([j_scores[i] for i in idx], [labels[i] for i in idx])
        - auroc([l_scores[i] for i in idx], [labels[i] for i in idx]),
        clusters,
    )
    report["B_paired"] = {
        "passages": len(keys),
        "jev": b_jev,
        "local_self_judge": b_local,
        "difference_jev_minus_local": b_jev["auroc"] - b_local["auroc"],
        "difference_ci": diff_ci,
    }
    print(f"B  Paired, Jev                   AUROC {fmt(b_jev)}")
    print(f"   Paired, local self-judge      AUROC {fmt(b_local)}")
    print(
        f"   Difference                    {b_jev['auroc'] - b_local['auroc']:+.3f} "
        f"[{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}]"
    )

    # C - inside each bucket of the local model's score
    report["C_within_local_buckets"] = {}
    for bucket in (1.0, 0.5, 0.0):
        idx = [i for i, s in enumerate(l_scores) if to_grid(s) == to_grid(bucket)]
        sub_labels = [labels[i] for i in idx]
        pos = sum(sub_labels)
        neg = len(sub_labels) - pos
        entry = {
            "passages": len(idx),
            "share_containing_answer": pos / len(idx) if idx else None,
        }
        name = f"{bucket:.2f}"
        if pos < MIN_PER_CLASS or neg < MIN_PER_CLASS:
            entry["too_small"] = True
            entry["positives"], entry["negatives"] = pos, neg
            print(f"C  Local said {name}: {len(idx)} passages, {pos} pos / {neg} neg - too small to report")
        else:
            result = auroc_with_ci(
                [j_scores[i] for i in idx], sub_labels, [clusters[i] for i in idx]
            )
            entry.update(result)
            print(
                f"C  Local said {name}: {len(idx)} passages, "
                f"{pos / len(idx):.0%} contain the answer -> Jev AUROC inside {fmt(result)}"
            )
        report["C_within_local_buckets"][name] = entry

    target = root / "results" / "informativeness.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten {target.relative_to(root)}")
    draw(report, root / "docs" / "assets" / "informativeness.png")


def draw(report: dict, out: Path) -> None:
    """One row per AUROC, with its interval and the 0.5 line random noise would sit on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jev_color, local_color = "#1f3a5f", "#7a4b00"
    a = report["A_jev_all_questions"]
    b = report["B_paired"]
    rows = [
        (f"Jev, every question ({a['clusters']} questions)", a, jev_color),
        (f"Jev, same passages ({b['jev']['clusters']} questions)", b["jev"], jev_color),
        ("Gemma judging itself, same passages", b["local_self_judge"], local_color),
    ]
    certain = report["C_within_local_buckets"].get("1.00", {})
    if not certain.get("too_small") and "auroc" in certain:
        rows.append(
            (f"Jev, only where Gemma said 1.00 ({certain['passages']} passages)", certain, jev_color)
        )

    fig, ax = plt.subplots(figsize=(10, 3.9))
    for y, (label, result, color) in enumerate(reversed(rows)):
        lo, hi = result["ci"]
        ax.plot([lo, hi], [y, y], color=color, linewidth=3, alpha=0.45, solid_capstyle="round")
        ax.plot(result["auroc"], y, "o", color=color, markersize=9)
        ax.text(hi + 0.012, y, f"{result['auroc']:.3f}", va="center", fontsize=10, color=color)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in reversed(rows)], fontsize=10)
    ax.axvline(0.5, color="#aa2222", linestyle="--", linewidth=1.2)
    ax.text(0.508, -0.45, "random noise", color="#aa2222", fontsize=9, va="bottom")
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("AUROC: chance a passage with the answer outscores one without (95% CI by question)")
    ax.set_title("Does a higher score mean the passage actually has the answer?", fontsize=12)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    fig.savefig(out.with_suffix(".svg"))
    print(f"figure  {out}")


if __name__ == "__main__":
    main()
