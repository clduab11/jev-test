#!/usr/bin/env python3
"""export_corpus.py - turn json_cache/ into a corpus, and split it by what may be published.

Every cache file is self-describing: {model, answers, usage, request{model, state, questions}}.
So each answer reconstructs to a (state, typed question, calibrated probability) triple carrying
the verbatim instructions and criteria the model actually saw.

TWO OUTPUTS, AND THE SPLIT IS DELIBERATE.

  corpus/raw/      the full triples. LOCAL ONLY, gitignored, not published.
  corpus/public/   aggregate statistics and the datasheet. No verbatim state, no per-row output.

Why the split. TypeSafe's Master Customer Agreement (https://typesafe.ai/legal/mca) assigns
ownership of Output to the customer at 4.2, so these triples are ours. But 2.3(b) forbids using
Output "to perform model distillation, train a model to imitate the output of the Services, or
develop (or to facilitate the development of) a similar or competing product or service", and
2.3 survives termination under 10.4. Publishing 67k labelled triples hands a third party the
exact input for that, and "facilitate" is the operative word. Separately the state field holds
verbatim text fetched from third-party web pages, which is its own redistribution question under
2.3(l). Aggregate statistics carry neither problem.

Usage:  python scripts/export_corpus.py [--cache json_cache] [--out corpus]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

JEV = "jev-1.13.0"
SELF_PREFIX = "self:"

# Stage is inferred from question id, because the cache does not record which stage made the call.
STAGE_BY_QID = {
    "is_relevant": "s1",
    "contains_answer_evidence": "s1",
    "contradicts_query_premise": "s1",
    "contains_prompt_injection": "s1",
    "is_promotional_or_boilerplate": "s1",
    "sufficient": "s2",
    "conflicts": "s2",
    "support": "s4",
    "addresses_query": "s4",
}


def state_fingerprint(state) -> str:
    """Stable hash of a state, so one state under two models can be paired without storing it."""
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def answer_value(answer: dict):
    """Return (primitive, scalar, distribution) for one typed answer."""
    kind = answer.get("type")
    if kind == "noul" or "noul" in answer:
        return "noul", answer.get("noul"), None
    if kind == "choice" or "choice" in answer:
        return "choice", answer.get("confidence"), answer.get("probabilities")
    if kind == "score" or "score" in answer:
        return "score", answer.get("score"), answer.get("probabilities")
    return "unknown", None, None


def iter_records(cache_dir: Path):
    for path in sorted(cache_dir.rglob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(blob, dict):
            continue
        answers = blob.get("answers")
        request = blob.get("request")
        if isinstance(answers, dict) and isinstance(request, dict):
            yield blob, request, answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="json_cache")
    parser.add_argument("--out", default="corpus")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = root / args.cache
    raw_dir = root / args.out / "raw"
    pub_dir = root / args.out / "public"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model: dict[str, list[dict]] = defaultdict(list)
    models_by_state: dict[str, set[str]] = defaultdict(set)
    tokens: Counter = Counter()
    requests: Counter = Counter()
    skipped = 0

    for blob, request, answers in iter_records(cache_dir):
        model = blob.get("model") or request.get("model") or "unknown"
        if model == JEV:
            family = JEV
        elif model.startswith(SELF_PREFIX):
            family = "gemma-self"
        else:
            skipped += 1
            continue

        state = request.get("state")
        questions = request.get("questions") or {}
        fingerprint = state_fingerprint(state)
        models_by_state[fingerprint].add(family)
        requests[family] += 1
        tokens[family] += int((blob.get("usage") or {}).get("input_tokens") or 0)

        for qid, answer in answers.items():
            if not isinstance(answer, dict):
                continue
            spec = questions.get(qid) or {}
            primitive, scalar, distribution = answer_value(answer)
            rows_by_model[family].append(
                {
                    "state_id": fingerprint,
                    "question_id": qid,
                    "stage": STAGE_BY_QID.get(qid, "other"),
                    "primitive": primitive,
                    "instructions": spec.get("instructions"),
                    "criteria": spec.get("criteria"),
                    "state": state,
                    "value": scalar,
                    "probabilities": distribution,
                    "teacher_model": model,
                }
            )

    # ---- raw, local only -------------------------------------------------
    for family, rows in sorted(rows_by_model.items()):
        target = raw_dir / f"{family}_triples.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"raw   {target.relative_to(root)}  {len(rows):,} triples")

    paired = sorted(fp for fp, models in models_by_state.items() if len(models) > 1)
    (raw_dir / "paired_state_ids.json").write_text(json.dumps(paired, indent=1), encoding="utf-8")

    # ---- public, aggregate only -----------------------------------------
    stats: dict = {
        "generated_from": args.cache,
        "note": "Aggregate statistics only. No verbatim state and no per-row model output.",
        "models": {},
        "paired_states": len(paired),
        "cache_records_skipped": skipped,
    }
    for family, rows in sorted(rows_by_model.items()):
        nouls = [r["value"] for r in rows if r["primitive"] == "noul" and r["value"] is not None]
        hist = Counter(round(float(v), 2) for v in nouls)
        stats["models"][family] = {
            "requests": requests[family],
            "triples": len(rows),
            "input_tokens": tokens[family],
            "primitives": dict(Counter(r["primitive"] for r in rows)),
            "stages": dict(Counter(r["stage"] for r in rows)),
            "question_ids": sorted({r["question_id"] for r in rows}),
            "noul": {
                "n": len(nouls),
                "distinct_values_2dp": len(hist),
                "min": min(nouls) if nouls else None,
                "max": max(nouls) if nouls else None,
                "histogram_2dp": {f"{k:.2f}": v for k, v in sorted(hist.items())},
            },
        }

    # ---- paired: same state AND same question id, answered by both ------
    # The unpaired comparison is not apples to apples: the self-judge arm only ran on the pilot,
    # while Jev ran on all 500 questions. Restricting to identical (state, question) pairs is the
    # comparison that holds up. Still aggregate - histograms, not rows.
    paired_set = set(paired)
    by_key: dict[str, dict] = defaultdict(dict)
    for family, rows in rows_by_model.items():
        for r in rows:
            if r["primitive"] == "noul" and r["value"] is not None and r["state_id"] in paired_set:
                by_key[family][(r["state_id"], r["question_id"])] = float(r["value"])
    families = sorted(by_key)
    shared = sorted(set.intersection(*(set(by_key[f]) for f in families))) if families else []
    queries = {
        (r["state"] or {}).get("query")
        for rows in rows_by_model.values()
        for r in rows
        if r["state_id"] in paired_set and isinstance(r["state"], dict)
    }
    stats["paired"] = {
        "note": "Same state and same question id, answered by both models.",
        "judgments": len(shared),
        "distinct_questions": len(queries - {None}),
        "models": {},
    }
    for family in families:
        hist = Counter(round(by_key[family][k], 2) for k in shared)
        stats["paired"]["models"][family] = {
            "histogram_2dp": {f"{k:.2f}": v for k, v in sorted(hist.items())}
        }
    if len(families) == 2 and shared:
        a, b = families
        agree = sum(1 for k in shared if (by_key[a][k] >= 0.5) == (by_key[b][k] >= 0.5))
        stats["paired"]["binary_agreement_at_0.5"] = round(agree / len(shared), 4)

    (pub_dir / "statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # The viewer table is the PAIRED histogram, since that is the comparison that holds up.
    with (pub_dir / "noul_histogram.csv").open("w", encoding="utf-8") as handle:
        handle.write("model,probability,count\n")
        for family, blob in stats["paired"]["models"].items():
            for value, count in blob["histogram_2dp"].items():
                handle.write(f"{family},{value},{count}\n")

    print(f"pub   {(pub_dir / 'statistics.json').relative_to(root)}")
    print(f"pub   {(pub_dir / 'noul_histogram.csv').relative_to(root)}")
    for family, blob in stats["models"].items():
        noul = blob["noul"]
        print(
            f"      {family:<12} {blob['requests']:>6,} req  {blob['triples']:>7,} triples  "
            f"{noul['distinct_values_2dp']:>4} distinct noul values"
        )
    print(f"      paired states (same state, both models): {len(paired):,}")


if __name__ == "__main__":
    main()
