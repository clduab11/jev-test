"""The four pre-registered bars for arm D on the web track (spec section 11).

Values come from ``prereg_bar_web_track_arm_D`` in judge/questions.v1.json.
This module only compares numbers and prints; it decides nothing about what
to run. Every printout carries the label it was given (for example PILOT)
because a 20-question pilot is a shape check, not a finding.
"""

from __future__ import annotations

import math
from typing import Any

from harness.config import load_spec


def _fmt(value: float | None, ci: list[float] | None = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    text = f"{value:.3f}"
    if ci and not any(isinstance(c, float) and math.isnan(c) for c in ci):
        text += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return text


def _ok(value: float | None) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def evaluate(
    d_metrics: dict[str, Any],
    b_metrics: dict[str, Any] | None,
    claim_support: dict[str, Any] | None,
    spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    bars = (spec or load_spec())["prereg_bar_web_track_arm_D"]
    coverage = d_metrics.get("coverage")
    halluc = d_metrics.get("hallucination_rate")
    gain = None
    if b_metrics and _ok(d_metrics.get("truthfulness")) and _ok(b_metrics.get("truthfulness")):
        gain = float(d_metrics["truthfulness"]) - float(b_metrics["truthfulness"])
    rate = claim_support.get("rate") if claim_support else None
    return [
        {
            "bar": "coverage",
            "rule": "at least",
            "threshold": bars["coverage_min"],
            "value": coverage,
            "ci": d_metrics.get("coverage_ci"),
            "passed": _ok(coverage) and coverage >= bars["coverage_min"],
        },
        {
            "bar": "hallucination rate when attempting",
            "rule": "at most",
            "threshold": bars["hallucination_rate_when_attempting_max"],
            "value": halluc,
            "ci": d_metrics.get("hallucination_rate_ci"),
            "passed": _ok(halluc) and halluc <= bars["hallucination_rate_when_attempting_max"],
        },
        {
            "bar": "truthfulness gain over arm B",
            "rule": "at least",
            "threshold": bars["truthfulness_gain_over_B_min"],
            "value": gain,
            "ci": None,
            "passed": _ok(gain) and gain >= bars["truthfulness_gain_over_B_min"],
        },
        {
            "bar": "citation support rate on kept claims",
            "rule": "at least",
            "threshold": bars["citation_support_rate_min"],
            "value": rate,
            "ci": claim_support.get("rate_ci") if claim_support else None,
            "passed": _ok(rate) and rate >= bars["citation_support_rate_min"],
        },
    ]


def format_table(rows: list[dict[str, Any]], label: str, n: int | None = None, arm: str = "D") -> str:
    head = f"Pre-registered bars (set for arm D), web track, applied to arm {arm}. {label}"
    if n is not None:
        head += f", n={n}"
    lines = [head + ". A pilot checks shapes and cost; it is not a finding.", ""]
    lines.append(f"{'bar':<40}{'rule':<10}{'bar value':>10}  {'observed [95% CI]':<26}{'result':<8}")
    for row in rows:
        lines.append(
            f"{row['bar']:<40}{row['rule']:<10}{row['threshold']:>10.2f}  "
            f"{_fmt(row['value'], row['ci']):<26}{'clears' if row['passed'] else 'misses':<8}"
        )
    cleared = sum(1 for r in rows if r["passed"])
    lines.append("")
    lines.append(f"{cleared} of {len(rows)} bars cleared ({label}).")
    return "\n".join(lines)
