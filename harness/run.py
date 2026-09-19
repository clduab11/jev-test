"""Run one arm over a frozen snapshot, or grade an existing results file.

    uv run python -m harness.run --arm A --dataset simpleqa --snapshot pilot-20260919
    uv run python -m harness.run --arm B --dataset simpleqa --snapshot pilot-20260919
    uv run python -m harness.run --arm D --dataset simpleqa --snapshot pilot-20260919 --limit 2
    uv run python -m harness.run --arm D --dataset simpleqa --snapshot simpleqa-500-20260919 --resume
    uv run python -m harness.run --grade results/A_simpleqa_pilot-20260919.json results/B_simpleqa_pilot-20260919.json

Results go to results/<arm>_<dataset>_<snapshot>.json with one record per
question: query, gold, answer, abstained, claims, citations,
fabricated_citations, latency, generator model, seed, and for the judged arms
every stage's decisions plus judge request and token counts. ``--grade`` adds
a ``grade`` to every record and a ``metrics`` block to the file, then prints
truthfulness, hallucination rate and coverage with bootstrap intervals. For a
judged arm it also grades citation support and prints the four pre-registered
bars against arm B, labelled with the snapshot so a pilot reads as a pilot.

``--resume`` continues an interrupted run from its results file: answered
questions are kept, the rest are answered, and the file is checkpointed every
ten questions. A judged arm keeps its memory palace on resume instead of
wiping it.

Arm D and C-self write verified claims to a memory palace of their own under
palace/ (git-ignored), one per snapshot and arm, wiped at the start of a fresh
pass-one run because the spec says pass one starts from an empty verified wing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from harness.config import ROOT, env, load_env, spec_version, thresholds
from harness.grading import crag_score

ARMS = ("A", "B", "D", "C-self")
JUDGED_ARMS = ("D", "C-self")
CHECKPOINT_EVERY = 10


def results_path(out_dir: Path, arm: str, dataset: str, snapshot: str) -> Path:
    return out_dir / f"{arm}_{dataset}_{snapshot}.json"


def default_memory_palace(snapshot: str, arm: str, pass_index: int = 1) -> Path:
    return ROOT / "palace" / f"{snapshot}_{arm}_pass{pass_index}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_rows(dataset: str, question_ids: list[str]) -> dict[str, dict[str, Any]]:
    from harness import datasets

    return datasets.module(dataset).by_id(question_ids)


def build_judge(arm: str):
    """The judge for a judged arm. Arm D is Jev over HTTP; C-self is Gemma through the adapter."""
    if arm == "D":
        from harness.judge.http import HttpJudge

        return HttpJudge.from_env(
            "JEV", cache_dir=ROOT / "json_cache", max_workers=int(thresholds()["judge_pool_workers"])
        )
    if arm == "C-self":
        from harness.judge.adapter import AdapterJudge

        return AdapterJudge.from_env(cache_dir=ROOT / "json_cache", max_workers=1)
    raise ValueError(f"no judge wired for arm {arm!r}")


def _payload(
    arm: str,
    dataset: str,
    snapshot: str,
    seed: int,
    today: str,
    records: list[dict[str, Any]],
    judge,
    memory_path: Path | None,
    memory_palace,
    complete: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "arm": arm,
        "dataset": dataset,
        "snapshot": snapshot,
        "spec_version": spec_version(),
        "generator_model": env("GENERATOR_MODEL"),
        "generator_url": env("LLAMA_SERVER_URL"),
        "seed": seed,
        "thinking": False,
        "today": today,
        "n": len(records),
        "complete": complete,
        "created_at": _now(),
    }
    if judge is not None:
        totals = {"n_requests": 0, "n_cached": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        for record in records:
            for key in totals:
                totals[key] += (record.get("judge") or {}).get(key, 0)
        totals["cost_usd"] = round(totals["cost_usd"], 6)
        totals["per_query_requests"] = round(totals["n_requests"] / len(records), 1) if records else None
        payload["judge"] = {
            "model": next((r.get("s0", {}).get("model") for r in records if r.get("s0")), judge.model),
            "requested_model": judge.model,
            "base_url": env("JEV_BASE_URL") if arm == "D" else env("LLAMA_SERVER_URL"),
            **totals,
        }
        if getattr(judge, "failures", 0):
            payload["judge"]["fallback_answers"] = int(judge.failures)
        payload["memory_palace"] = {
            "path": str(memory_path),
            "drawers": memory_palace.count() if memory_palace else 0,
        }
    payload["records"] = records
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def run_arm(
    arm: str,
    dataset: str,
    snapshot: str,
    out_dir: Path,
    limit: int | None,
    seed: int,
    memory_palace_path: Path | None = None,
    keep_memory: bool = False,
    resume: bool = False,
    log=print,
) -> Path:
    from harness.retrieval.snapshot import load_manifest, open_palace, palace_path, replay

    manifest = load_manifest(snapshot)
    question_ids = manifest["question_ids"][: limit or None]
    rows = _load_rows(dataset, question_ids)
    palace = open_palace(palace_path(snapshot), read_only=True) if arm != "A" else None
    today = str(manifest.get("created_at", ""))[:10] or time.strftime("%Y-%m-%d", time.gmtime())
    path = results_path(out_dir, arm, dataset, snapshot)

    done: dict[str, dict[str, Any]] = {}
    if resume and path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        wanted = set(question_ids)
        done = {r["question_id"]: r for r in prior.get("records", []) if r.get("question_id") in wanted}
        log(f"resuming {path.name}: {len(done)} of {len(question_ids)} questions already answered")

    judge = None
    memory_palace = None
    memory_path: Path | None = None
    if arm in JUDGED_ARMS:
        judge = build_judge(arm)
        memory_path = memory_palace_path or default_memory_palace(snapshot, arm)
        if memory_path.exists() and not keep_memory and memory_palace_path is None and not done:
            shutil.rmtree(memory_path)
            log(f"memory palace {memory_path} wiped: pass one starts from an empty verified wing")
        memory_palace = open_palace(memory_path)
        judge_url = env("JEV_BASE_URL") if arm == "D" else env("LLAMA_SERVER_URL")
        log(f"judge {judge.model} at {judge_url}; memory palace {memory_path} ({memory_palace.count()} drawers)")

    records: list[dict[str, Any]] = []
    answered_now = 0
    for index, qid in enumerate(question_ids, 1):
        if qid in done:
            records.append(done[qid])
            continue
        row = rows[qid]
        started = time.perf_counter()
        if arm == "A":
            from harness.arms import a_plain

            record = a_plain.answer(row["query"], seed=seed)
        elif arm == "B":
            from harness.arms import b_naive

            replayed = replay(snapshot, qid, palace=palace)
            record = b_naive.answer(row["query"], replayed, seed=seed)
        elif arm in JUDGED_ARMS:
            from harness.arms import c_self, d_jev

            module = d_jev if arm == "D" else c_self
            replayed = replay(snapshot, qid, palace=palace)
            record = module.answer(
                row["query"],
                qid,
                replayed,
                judge,
                memory_palace=memory_palace,
                snapshot_palace=palace,
                seed=seed,
                today=today,
            )
        else:
            raise ValueError(f"arm {arm!r} is not implemented yet")
        record = {"question_id": qid, "query": row["query"], "gold": row["gold"], **record}
        record["wall_s"] = round(time.perf_counter() - started, 3)
        records.append(record)
        answered_now += 1
        preview = "ABSTAIN" if record["abstained"] else (record["answer"][:90] + ("..." if len(record["answer"]) > 90 else ""))
        extra = ""
        if record.get("judge"):
            j = record["judge"]
            extra = f", {j['n_requests']} judge requests ({j['n_cached']} cached), ${j['cost_usd']:.4f}"
        if record.get("abstained") and record.get("abstain_reason"):
            preview += f" ({record['abstain_reason']})"
        log(f"[{index}/{len(question_ids)}] {qid} ({record['wall_s']:.1f}s{extra}): {preview}")
        if answered_now % CHECKPOINT_EVERY == 0:
            _write(path, _payload(arm, dataset, snapshot, seed, today, records, judge, memory_path, memory_palace, complete=False))

    payload = _payload(arm, dataset, snapshot, seed, today, records, judge, memory_path, memory_palace, complete=True)
    _write(path, payload)
    log(f"wrote {path} ({len(records)} records, {answered_now} answered in this run)")
    if judge is not None:
        j = payload["judge"]
        log(
            f"judge totals: {j['n_requests']} requests ({j['n_cached']} cached), "
            f"{j['input_tokens']} input tokens, ${j['cost_usd']:.4f}, {j['per_query_requests']} requests per query"
        )
        if getattr(judge, "failures", 0):
            log(f"judge failed to answer {judge.failures} request(s); those got the do-not-know fallback")
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
    if payload.get("arm") in JUDGED_ARMS:
        from harness.grading import claim_support

        per_record = claim_support.grade_records(payload["records"], model=model)
        for record, claim_grades in zip(payload["records"], per_record, strict=True):
            record["claim_support"] = claim_grades
        metrics["claim_support"] = claim_support.support_rate([[g["label"] for g in cg] for cg in per_record])
    payload["metrics"] = metrics
    _write(path, payload)
    log(f"graded {path.name}: {metrics['correct']} correct, {metrics['incorrect']} incorrect, {metrics['not_attempted']} not attempted (grader {metrics['grader_model']})")
    if "claim_support" in metrics:
        cs = metrics["claim_support"]
        log(f"  citation support: {cs['supported']} of {cs['kept_claims']} kept claims supported by the grader")
    return payload


def print_bars(payload: dict[str, Any], out_dir: Path, log=print) -> None:
    """A judged arm against the four pre-registered bars, using arm B on the same snapshot."""
    from harness.grading import prereg

    b_path = results_path(out_dir, "B", payload["dataset"], payload["snapshot"])
    b_metrics = None
    if b_path.exists():
        b_metrics = json.loads(b_path.read_text(encoding="utf-8")).get("metrics")
    rows = prereg.evaluate(payload["metrics"], b_metrics, payload["metrics"].get("claim_support"))
    label = f"PILOT snapshot {payload['snapshot']}" if payload["n"] < 100 else f"snapshot {payload['snapshot']}"
    log("")
    log(prereg.format_table(rows, label, n=payload["n"], arm=payload["arm"]))
    if b_metrics is None:
        log(f"(no graded arm B file at {b_path.name}; the gain bar cannot be evaluated)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run an arm over a snapshot, or grade results")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--dataset", default="simpleqa")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--seed", type=int, default=None, help="generator seed; default GENERATOR_SEED from .env")
    parser.add_argument("--resume", action="store_true", help="continue from the existing results file")
    parser.add_argument("--memory-palace", default=None, help="judged arms: palace directory for verified claims (default palace/<snapshot>_<arm>_pass1, wiped first)")
    parser.add_argument("--keep-memory", action="store_true", help="judged arms: do not wipe the default memory palace first")
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
        run_arm(
            args.arm,
            args.dataset,
            args.snapshot,
            out_dir,
            args.limit,
            seed,
            memory_palace_path=Path(args.memory_palace) if args.memory_palace else None,
            keep_memory=args.keep_memory,
            resume=args.resume,
        )

    if args.grade is not None:
        paths = [Path(p) for p in args.grade] or [results_path(out_dir, args.arm, args.dataset, args.snapshot)]
        rows = []
        judged = []
        for path in paths:
            payload = grade_file(path, model=args.grader_model)
            rows.append((payload["arm"], payload["metrics"]))
            if payload.get("arm") in JUDGED_ARMS:
                judged.append(payload)
        print()
        print(crag_score.format_table(rows))
        for payload in judged:
            print_bars(payload, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
