"""Run one arm over a frozen snapshot, or grade an existing results file.

    uv run python -m harness.run --arm A --dataset simpleqa --snapshot pilot-20260919 --out results/
    uv run python -m harness.run --arm B --dataset simpleqa --snapshot pilot-20260919 --out results/
    uv run python -m harness.run --grade results/A_simpleqa_pilot-20260919.json results/B_simpleqa_pilot-20260919.json

Results go to results/<arm>_<dataset>_<snapshot>.json with one record per
question: query, gold, answer, abstained, claims, citations,
fabricated_citations, latency, generator model, seed. ``--grade`` adds a
``grade`` to every record and a ``metrics`` block to the file, then prints
truthfulness, hallucination rate and coverage with bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from harness.config import ROOT, env, load_env, spec_version
from harness.grading import crag_score

ARMS = ("A", "B")


def results_path(out_dir: Path, arm: str, dataset: str, snapshot: str) -> Path:
    return out_dir / f"{arm}_{dataset}_{snapshot}.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_rows(dataset: str, question_ids: list[str]) -> dict[str, dict[str, Any]]:
    if dataset != "simpleqa":
        raise ValueError(f"unknown dataset {dataset!r}")
    from harness.datasets import simpleqa

    return simpleqa.by_id(question_ids)


def run_arm(arm: str, dataset: str, snapshot: str, out_dir: Path, limit: int | None, seed: int, log=print) -> Path:
    from harness.retrieval.snapshot import load_manifest, open_palace, palace_path, replay

    manifest = load_manifest(snapshot)
    question_ids = manifest["question_ids"][: limit or None]
    rows = _load_rows(dataset, question_ids)
    palace = open_palace(palace_path(snapshot), read_only=True) if arm == "B" else None

    records: list[dict[str, Any]] = []
    for index, qid in enumerate(question_ids, 1):
        row = rows[qid]
        started = time.perf_counter()
        if arm == "A":
            from harness.arms import a_plain

            record = a_plain.answer(row["query"], seed=seed)
        elif arm == "B":
            from harness.arms import b_naive

            replayed = replay(snapshot, qid, palace=palace)
            record = b_naive.answer(row["query"], replayed, seed=seed)
        else:
            raise ValueError(f"arm {arm!r} is not implemented yet")
        record = {"question_id": qid, "query": row["query"], "gold": row["gold"], **record}
        record["wall_s"] = round(time.perf_counter() - started, 3)
        records.append(record)
        preview = "ABSTAIN" if record["abstained"] else (record["answer"][:90] + ("..." if len(record["answer"]) > 90 else ""))
        log(f"[{index}/{len(question_ids)}] {qid} ({record['wall_s']:.1f}s): {preview}")

    payload = {
        "arm": arm,
        "dataset": dataset,
        "snapshot": snapshot,
        "spec_version": spec_version(),
        "generator_model": env("GENERATOR_MODEL"),
        "generator_url": env("LLAMA_SERVER_URL"),
        "seed": seed,
        "thinking": False,
        "n": len(records),
        "created_at": _now(),
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = results_path(out_dir, arm, dataset, snapshot)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {path} ({len(records)} records)")
    return path


def grade_file(path: Path, model: str | None = None, log=print) -> dict[str, Any]:
    from harness.grading import simpleqa as grader

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "simpleqa":
        raise ValueError(f"no grader for dataset {payload.get('dataset')!r}")
    grades = grader.grade_records(payload["records"], model=model)
    for record, grade in zip(payload["records"], grades, strict=True):
        record["grade"] = grade
    labels = [g["label"] for g in grades]
    metrics = crag_score.bootstrap(labels)
    metrics["grader_model"] = next((g["model"] for g in grades if g.get("model")), model or env("GRADER_MODEL"))
    metrics["graded_at"] = _now()
    metrics["fabricated_citations_total"] = sum(int(r.get("fabricated_citations") or 0) for r in payload["records"])
    payload["metrics"] = metrics
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"graded {path.name}: {metrics['correct']} correct, {metrics['incorrect']} incorrect, {metrics['not_attempted']} not attempted (grader {metrics['grader_model']})")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run an arm over a snapshot, or grade results")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--dataset", default="simpleqa")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--seed", type=int, default=None, help="generator seed; default GENERATOR_SEED from .env")
    parser.add_argument("--grade", nargs="*", metavar="RESULTS_JSON", help="grade these results files (or the file this run would write)")
    parser.add_argument("--grader-model", default=None)
    args = parser.parse_args(argv)

    load_env()
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    seed = args.seed if args.seed is not None else int(env("GENERATOR_SEED", "20260919") or 20260919)

    if args.grade is None and not args.arm:
        parser.error("give --arm to run, or --grade to grade")
    if args.arm:
        if not args.snapshot:
            parser.error("--snapshot is required to run an arm")
        run_arm(args.arm, args.dataset, args.snapshot, out_dir, args.limit, seed)

    if args.grade is not None:
        paths = [Path(p) for p in args.grade] or [results_path(out_dir, args.arm, args.dataset, args.snapshot)]
        rows = []
        for path in paths:
            payload = grade_file(path, model=args.grader_model)
            rows.append((payload["arm"], payload["metrics"]))
        print()
        print(crag_score.format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
