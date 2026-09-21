#!/usr/bin/env python3
"""gate_replay.py - was 0.80 the wrong place to put the S4 confidence gate?

Arm D lost to arm B by 0.128 truthfulness on the 500-question run, against a pre-registered bar
of +0.15. This asks whether the threshold caused it, and it is careful about which parts of that
question the existing data can actually answer.

WHAT IS COMPUTABLE WITH ZERO NEW GRADER CALLS

  1. RAISING the gate. Every kept claim carries a Sonnet claim-support label, and every kept
     claim has confidence >= 0.80. So the risk-coverage curve is exact on [0.80, 1.00] and tells
     you whether a stricter gate would have been better. It cannot speak below 0.80.

  2. WHAT A LOWER GATE WOULD RESCUE. Stripped claims carry their confidence and their choice
     label, so the rescue set at any threshold is exact. Only claims stripped as
     'supported_low_confidence' can be rescued by moving the threshold - the rest were stripped
     on their label (unsupported, contradicted, fabricated_citation, uncited_claim) and no
     threshold change touches them.

  3. A PROXY CEILING on the question-level gain, using arm B's grade on the same question from
     the same frozen snapshot as a stand-in for whether the rescued answer would have been
     right. This is an UPPER BOUND, not a measurement: B's grade is B's answer, not D's.

WHAT IS NOT COMPUTABLE, AND WHY

  The rigorous version needs Sonnet claim-support labels on the rescued claims, which were never
  graded because they never shipped. This script writes that worklist to
  results/gate_replay_worklist.jsonl so the grading run is a single pass over a known, bounded
  set. Until that runs, treat every number below 0.80 as a ceiling.

Usage:  python scripts/gate_replay.py [--arm-d PATH] [--arm-b PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

RESCUABLE_REASON = "supported_low_confidence"
SHIPPED_GATE = 0.80


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def grade(record: dict) -> str | None:
    return (record.get("grade") or {}).get("label")


def confidence(entry: dict) -> float | None:
    value = (entry.get("support") or {}).get("confidence")
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-d", default="results/D_simpleqa_simpleqa-500-20260919.json")
    parser.add_argument("--arm-b", default="results/B_simpleqa_simpleqa-500-20260919.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    arm_d = load(root / args.arm_d)
    arm_b = {r["question_id"]: r for r in load(root / args.arm_b)}

    # ---- 1. exact risk-coverage above the shipped gate ------------------
    graded: list[tuple[float, bool]] = []
    misaligned = 0
    for record in arm_d:
        claims = record.get("claims") or []
        labels = record.get("claim_support") or []
        if len(claims) != len(labels):
            misaligned += 1
            continue
        for claim, label in zip(claims, labels):
            conf = confidence(claim)
            verdict = (label or {}).get("label")
            if conf is None or verdict not in ("supported", "unsupported"):
                continue
            graded.append((conf, verdict == "supported"))

    print("=" * 72)
    print("1. RAISING THE GATE - exact, uses the Sonnet labels already in results/")
    print("=" * 72)
    print(f"   graded kept claims: {len(graded):,}   (records with misaligned labels: {misaligned})")
    print(f"   {'threshold':>10}{'kept':>8}{'coverage':>10}{'unsupported':>13}{'error rate':>12}")
    for step in range(80, 101, 2):
        threshold = step / 100
        subset = [ok for conf, ok in graded if conf >= threshold]
        if not subset:
            continue
        bad = sum(1 for ok in subset if not ok)
        marker = "  <- shipped" if abs(threshold - SHIPPED_GATE) < 1e-9 else ""
        print(
            f"   {threshold:>10.2f}{len(subset):>8}{len(subset) / len(graded):>10.1%}"
            f"{bad:>13}{bad / len(subset):>12.2%}{marker}"
        )

    # ---- 2. what a lower gate rescues, exactly --------------------------
    rescuable: list[dict] = []
    reasons: Counter = Counter()
    for record in arm_d:
        for entry in record.get("stripped") or []:
            reasons[entry.get("reason")] += 1
            if entry.get("reason") != RESCUABLE_REASON:
                continue
            conf = confidence(entry)
            if conf is None:
                continue
            rescuable.append(
                {
                    "question_id": record["question_id"],
                    "confidence": conf,
                    "sentence": entry.get("sentence"),
                    "abstained": bool(record.get("abstained")),
                    "abstain_reason": record.get("abstain_reason"),
                }
            )

    print()
    print("=" * 72)
    print("2. LOWERING THE GATE - exact rescue set, outcome NOT yet measured")
    print("=" * 72)
    print(f"   stripped claims by reason: {dict(reasons)}")
    print(f"   rescuable by threshold alone ('{RESCUABLE_REASON}'): {len(rescuable)}")
    print("   everything else was stripped on its LABEL; no threshold change touches it.")

    # ---- 3. proxy ceiling on the question-level gain --------------------
    print()
    print("=" * 72)
    print("3. PROXY CEILING on questions recovered   (arm B's grade as a stand-in - UPPER BOUND)")
    print("=" * 72)
    print(f"   {'threshold':>10}{'claims':>8}{'questions':>11}{'B right':>9}{'B wrong':>9}{'ceiling':>10}")
    base_correct, base_wrong = 332, 26
    rows = []
    for step in range(0, 81, 5):
        threshold = step / 100
        hits = [r for r in rescuable if r["confidence"] >= threshold]
        questions = {r["question_id"] for r in hits if r["abstained"]}
        right = sum(1 for q in questions if grade(arm_b.get(q, {})) == "correct")
        wrong = sum(1 for q in questions if grade(arm_b.get(q, {})) == "incorrect")
        score = ((base_correct + right) - (base_wrong + wrong)) / 500
        rows.append(
            {
                "threshold": threshold,
                "claims_rescued": len(hits),
                "questions_unabstained": len(questions),
                "b_correct": right,
                "b_incorrect": wrong,
                "ceiling_truthfulness": round(score, 4),
            }
        )
        print(
            f"   {threshold:>10.2f}{len(hits):>8}{len(questions):>11}{right:>9}{wrong:>9}{score:>10.3f}"
        )
    print("   arm D as shipped: 0.612      arm B: 0.740      pre-registered bar: B + 0.15 = 0.890")

    # ---- worklist for the rigorous version ------------------------------
    worklist = root / "results/gate_replay_worklist.jsonl"
    with worklist.open("w", encoding="utf-8") as handle:
        for row in sorted(rescuable, key=lambda r: -r["confidence"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = root / "results/gate_replay.json"
    summary.write_text(
        json.dumps(
            {
                "shipped_gate": SHIPPED_GATE,
                "graded_kept_claims": len(graded),
                "stripped_by_reason": dict(reasons),
                "rescuable_claims": len(rescuable),
                "raise_gate_curve": [
                    {
                        "threshold": t / 100,
                        "kept": sum(1 for c, _ in graded if c >= t / 100),
                        "unsupported": sum(1 for c, ok in graded if c >= t / 100 and not ok),
                    }
                    for t in range(80, 101, 2)
                ],
                "lower_gate_proxy_ceiling": rows,
                "caveat": (
                    "Rows below 0.80 are an UPPER BOUND computed from arm B's grades on the same "
                    "questions, not a measurement of arm D's rescued answers. Grade "
                    "results/gate_replay_worklist.jsonl to replace them."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"   worklist -> {worklist.relative_to(root)}  ({len(rescuable)} claims to grade)")
    print(f"   summary  -> {summary.relative_to(root)}")


if __name__ == "__main__":
    main()
