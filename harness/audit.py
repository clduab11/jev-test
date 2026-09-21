"""Audit a judge's confidence scores: can a threshold on them move, and does moving it help?

Any LLM-as-judge pipeline eventually puts a threshold on a confidence score. This module
answers two separate questions about that score, and keeps them separate on purpose.

SPREAD - can a threshold move at all? (no labels needed)

    gate_mobility     share of 0.01 threshold steps, 0.00 to 0.99, that change which items
                      pass. By construction this equals the number of distinct occupied
                      0.01 bins below 1.00, divided by 100 - so it is the distinct-value
                      count on a grid, and it inherits that count's weakness: uniformly
                      random scores get close to 100%. tests/test_audit.py pins both facts.
    top_k_mass        share of scores on the k most common values. High means the score
                      has collapsed onto a few settings.

    Spread is necessary for a threshold to do anything and says nothing about whether
    what it does is useful. Random noise has excellent spread.

INFORMATIVENESS - does a higher score mean a better item? (labels needed)

    auroc             probability a random positive outscores a random negative, ties
                      counted half. Random noise scores 0.5 however well it spreads.
    auroc_within      AUROC of one judge's score, restricted to items another judge put in
                      one bucket. The sharpest test of whether extra resolution is real:
                      if a coarse judge calls 189 items "certain" and a finer judge can
                      still rank the right ones above the wrong ones inside that set, the
                      finer judge's extra values carry information. Noise cannot do this.
    bootstrap_ci      interval that resamples whole clusters (questions), because items
                      from one question move together and resampling items alone would
                      report intervals that are far too narrow.

The command line takes a CSV of scores and runs both halves:

    python -m harness.audit scores.csv                        # spread only
    python -m harness.audit scores.csv --label label          # plus AUROC
    python -m harness.audit scores.csv --label label --judge judge --cluster question

No network, no model, standard library plus numpy.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence

import numpy as np

# Match the benchmark's other intervals (harness/grading/crag_score.py).
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260919
GRID = 100  # thresholds live on a 0.01 grid


def to_grid(score: float) -> int:
    """Place a probability on the 0.01 grid as an integer 0..100.

    Rounding to two places first, then scaling, keeps 0.29 at 29 instead of letting float
    noise put it at 28, which would silently move it across a threshold.
    """
    return round(round(float(score), 2) * GRID)


# ---------------------------------------------------------------- spread


def gate_mobility(scores: Sequence[float]) -> float:
    """Share of adjacent 0.01 threshold steps that change the set of items passing."""
    if not scores:
        return float("nan")
    occupied = {to_grid(s) for s in scores}
    return sum(1 for t in range(GRID) if t in occupied) / GRID


def distinct_values(scores: Sequence[float]) -> int:
    return len({to_grid(s) for s in scores})


def top_k_mass(scores: Sequence[float], k: int = 3) -> float:
    if not scores:
        return float("nan")
    counts = Counter(to_grid(s) for s in scores)
    return sum(c for _, c in counts.most_common(k)) / len(scores)


def interior_mass(scores: Sequence[float], lo: float = 0.10, hi: float = 0.90) -> float:
    if not scores:
        return float("nan")
    a, b = to_grid(lo), to_grid(hi)
    return sum(1 for s in scores if a <= to_grid(s) <= b) / len(scores)


def spread(scores: Sequence[float]) -> dict:
    return {
        "n": len(scores),
        "distinct_values": distinct_values(scores),
        "top3_mass": top_k_mass(scores, 3),
        "interior_mass": interior_mass(scores),
        "gate_mobility": gate_mobility(scores),
    }


# ------------------------------------------------------------ informativeness


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann-Whitney AUROC with average ranks for ties. O(n log n)."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_within(
    scores: Sequence[float],
    labels: Sequence[bool],
    bucket_scores: Sequence[float],
    bucket: float,
) -> tuple[float, int, int]:
    """AUROC of `scores`, restricted to items whose `bucket_scores` sit at `bucket`.

    Returns (auroc, positives, negatives) so a caller can refuse to report a bucket that is
    too small to mean anything.
    """
    target = to_grid(bucket)
    keep = [i for i, b in enumerate(bucket_scores) if to_grid(b) == target]
    sub_s = [scores[i] for i in keep]
    sub_l = [labels[i] for i in keep]
    pos = sum(1 for v in sub_l if v)
    return auroc(sub_s, sub_l), pos, len(sub_l) - pos


def bootstrap_ci(
    stat: Callable[[list[int]], float],
    clusters: Sequence[Hashable],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% interval for `stat`, resampling whole clusters with replacement.

    `stat` receives a list of item indices and returns a number. Items that share a cluster
    are kept together, so passages from one question are never split across a resample.
    """
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        groups[cluster].append(index)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        picked = rng.integers(0, len(keys), size=len(keys))
        indices = [i for k in picked for i in groups[keys[k]]]
        value = stat(indices)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def auroc_with_ci(
    scores: Sequence[float],
    labels: Sequence[bool],
    clusters: Sequence[Hashable] | None = None,
) -> dict:
    point = auroc(scores, labels)
    out = {
        "auroc": point,
        "positives": int(sum(1 for v in labels if v)),
        "negatives": int(sum(1 for v in labels if not v)),
    }
    if clusters is not None:
        out["ci"] = bootstrap_ci(
            lambda idx: auroc([scores[i] for i in idx], [labels[i] for i in idx]), clusters
        )
        out["clusters"] = len(set(clusters))
    return out


# ------------------------------------------------------------------ command line


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "t", "supported", "correct"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.audit",
        description="Can a threshold on this judge's confidence move, and does moving it help?",
    )
    parser.add_argument("csv", help="CSV with a header row")
    parser.add_argument("--score", default="score", help="column holding the probability")
    parser.add_argument("--label", help="column holding the truth (1/0, true/false)")
    parser.add_argument("--judge", help="column naming the judge, to compare several")
    parser.add_argument("--cluster", help="column to resample by, usually the question id")
    args = parser.parse_args(argv)

    with open(args.csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[args.judge] if args.judge else "judge"].append(row)

    for name, items in groups.items():
        scores = [float(r[args.score]) for r in items]
        s = spread(scores)
        print(f"\n{name}  (n={s['n']:,})")
        print(f"  distinct values       {s['distinct_values']}")
        print(f"  share on top 3 values {s['top3_mass']:.1%}")
        print(f"  share in 0.10-0.90    {s['interior_mass']:.1%}")
        print(f"  gate mobility         {s['gate_mobility']:.0%}   (spread only; random noise scores high)")
        if args.label:
            labels = [_truthy(r[args.label]) for r in items]
            clusters = [r[args.cluster] for r in items] if args.cluster else None
            info = auroc_with_ci(scores, labels, clusters)
            ci = f" [{info['ci'][0]:.3f}, {info['ci'][1]:.3f}]" if "ci" in info else ""
            print(
                f"  AUROC                 {info['auroc']:.3f}{ci}   "
                f"(pos {info['positives']}, neg {info['negatives']}; random noise scores 0.5)"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
