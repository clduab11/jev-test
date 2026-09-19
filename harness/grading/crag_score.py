"""CRAG-style scoring with bootstrap intervals (spec section 11).

  truthfulness       = (correct - incorrect) / N
  hallucination rate = incorrect / attempted
  coverage           = attempted / N

Each with a bootstrap 95 percent interval over questions: 1,000 resamples,
seed 20260919. Labels are ``correct``, ``incorrect`` or ``not_attempted``;
an abstention counts as ``not_attempted``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

LABELS = ("correct", "incorrect", "not_attempted")
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260919


def _metrics(correct: int, incorrect: int, n: int) -> dict[str, float]:
    attempted = correct + incorrect
    return {
        "truthfulness": (correct - incorrect) / n if n else math.nan,
        "hallucination_rate": incorrect / attempted if attempted else math.nan,
        "coverage": attempted / n if n else math.nan,
    }


def score(labels: list[str]) -> dict[str, Any]:
    """Point estimates plus counts."""
    for label in labels:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
    n = len(labels)
    correct = labels.count("correct")
    incorrect = labels.count("incorrect")
    out: dict[str, Any] = {
        "n": n,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": labels.count("not_attempted"),
        "attempted": correct + incorrect,
    }
    out.update(_metrics(correct, incorrect, n))
    return out


def bootstrap(
    labels: list[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Point estimates with 95 percent percentile intervals over questions."""
    result = score(labels)
    n = len(labels)
    if n == 0:
        for key in ("truthfulness", "hallucination_rate", "coverage"):
            result[f"{key}_ci"] = [math.nan, math.nan]
        return result
    rng = np.random.default_rng(seed)
    codes = np.array([0 if l == "correct" else 1 if l == "incorrect" else 2 for l in labels])
    draws = rng.integers(0, n, size=(resamples, n))
    sampled = codes[draws]
    correct = (sampled == 0).sum(axis=1)
    incorrect = (sampled == 1).sum(axis=1)
    attempted = correct + incorrect
    truth = (correct - incorrect) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        halluc = np.where(attempted > 0, incorrect / np.maximum(attempted, 1), np.nan)
    coverage = attempted / n
    for key, values in (("truthfulness", truth), ("hallucination_rate", halluc), ("coverage", coverage)):
        finite = values[~np.isnan(values)]
        if finite.size:
            lo, hi = np.percentile(finite, [2.5, 97.5])
            result[f"{key}_ci"] = [round(float(lo), 4), round(float(hi), 4)]
        else:
            result[f"{key}_ci"] = [math.nan, math.nan]
    result["bootstrap"] = {"resamples": resamples, "seed": seed}
    return result


def _fmt(value: float, ci: list[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def format_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    """One line per (name, bootstrap result)."""
    header = (
        f"{'arm':<6}{'N':>4}{'corr':>6}{'inc':>5}{'abst':>6}  "
        f"{'truthfulness [95% CI]':<26}{'hallucination [95% CI]':<26}{'coverage [95% CI]':<26}"
    )
    lines = [header]
    for name, m in rows:
        lines.append(
            f"{name:<6}{m['n']:>4}{m['correct']:>6}{m['incorrect']:>5}{m['not_attempted']:>6}  "
            f"{_fmt(m['truthfulness'], m['truthfulness_ci']):<26}"
            f"{_fmt(m['hallucination_rate'], m['hallucination_rate_ci']):<26}"
            f"{_fmt(m['coverage'], m['coverage_ci']):<26}"
        )
    return "\n".join(lines)
