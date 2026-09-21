#!/usr/bin/env python3
"""calibration_figure.py - the launch figure: what a threshold can and cannot do.

Two panels, both computed from corpus/public/statistics.json, both label-free.

  LEFT   where each judge puts its probability mass. Gemma self-judging piles 99.3% of 5,065
         Nouls onto three values (0.00, 1.00, 0.50). Jev spreads 64,964 across 99 and never
         emits 0.00 or 1.00 at all.

  RIGHT  gate mobility. Sweep an accept threshold from 0 to 1 and plot the share of answers
         accepted. Gemma's curve is a staircase with three risers, so almost every threshold
         you could pick behaves identically to its neighbours - the knob is not connected to
         anything. Jev's curve is smooth, so a threshold is a real control surface.

Distinct-value count is deliberately NOT the headline: uniformly random noise scores well on
it. Mass concentration and gate mobility are the honest measures, and they are label-free, so
neither inherits the benchmark's Sonnet-graded labels.

Usage:  python scripts/calibration_figure.py [--stats corpus/public/statistics.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {
    "jev-1.13.0": "Jev 1.13.0 (calibrated decision model)",
    "gemma-self": "Gemma 4 E2B judging itself (verbalized)",
}
COLOR = {"jev-1.13.0": "#1f3a5f", "gemma-self": "#7a4b00"}


def histogram(stats: dict, model: str) -> dict[float, int]:
    return {float(k): v for k, v in stats["models"][model]["noul"]["histogram_2dp"].items()}


def accept_curve(hist: dict[float, int], steps: int = 101):
    """Share of answers with probability >= t, for t on a 0..1 grid."""
    total = sum(hist.values())
    thresholds = [i / (steps - 1) for i in range(steps)]
    return thresholds, [
        sum(count for value, count in hist.items() if value >= t) / total for t in thresholds
    ]


def mobility(accepted: list[float]) -> float:
    """Share of adjacent threshold steps that actually change the accept set.

    A knob that moves nothing has mobility 0. A perfectly smooth score approaches 1.
    """
    changes = sum(1 for a, b in zip(accepted, accepted[1:]) if abs(a - b) > 1e-9)
    return changes / (len(accepted) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default="corpus/public/statistics.json")
    parser.add_argument("--out", default="docs/assets/calibration.png")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stats = json.loads((root / args.stats).read_text(encoding="utf-8"))
    models = [m for m in ("gemma-self", "jev-1.13.0") if m in stats["models"]]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for model in models:
        hist = histogram(stats, model)
        total = sum(hist.values())
        values = sorted(hist)
        left.bar(
            values,
            [hist[v] / total for v in values],
            width=0.008,
            color=COLOR[model],
            alpha=0.85,
            label=f"{LABEL[model]}  (n={total:,})",
        )

    left.set_yscale("log")
    left.set_xlabel("P(yes) returned for a Noul question")
    left.set_ylabel("share of answers (log scale)")
    left.set_title("Where each judge puts its probability mass")
    left.legend(loc="upper center", fontsize=8.5)
    left.grid(alpha=0.2, linewidth=0.5)

    summary = []
    for model in models:
        hist = histogram(stats, model)
        total = sum(hist.values())
        thresholds, accepted = accept_curve(hist)
        move = mobility(accepted)
        interior = sum(c for v, c in hist.items() if 0.10 <= v <= 0.90) / total
        right.plot(
            thresholds,
            accepted,
            color=COLOR[model],
            linewidth=2.2,
            label=f"{LABEL[model]}  (mobility {move:.0%})",
        )
        summary.append(
            {
                "model": model,
                "n": total,
                "distinct_values_2dp": stats["models"][model]["noul"]["distinct_values_2dp"],
                "mass_interior_0.10_0.90": round(interior, 4),
                "gate_mobility": round(move, 4),
            }
        )

    right.axvline(0.80, color="#aa2222", linestyle="--", linewidth=1.2)
    right.text(
        0.805,
        0.94,
        "the 0.80 gate\nshipped in arm D",
        fontsize=8,
        color="#aa2222",
        va="top",
    )
    right.set_xlabel("accept threshold")
    right.set_ylabel("share of answers accepted")
    right.set_title("Gate mobility: does moving the threshold do anything?")
    right.legend(loc="lower left", fontsize=8.5)
    right.grid(alpha=0.2, linewidth=0.5)
    right.set_ylim(-0.02, 1.02)

    fig.suptitle(
        "A threshold is only a control surface if the score underneath it moves",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    fig.savefig(out.with_suffix(".svg"))

    metrics = root / "corpus/public/calibration_metrics.json"
    metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"figure  {out.relative_to(root)}")
    print(f"figure  {out.with_suffix('.svg').relative_to(root)}")
    print(f"metrics {metrics.relative_to(root)}")
    for row in summary:
        print(
            f"  {row['model']:<12} n={row['n']:>7,}  distinct={row['distinct_values_2dp']:>3}  "
            f"interior={row['mass_interior_0.10_0.90']:.1%}  mobility={row['gate_mobility']:.0%}"
        )


if __name__ == "__main__":
    main()
