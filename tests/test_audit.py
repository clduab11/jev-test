"""The confidence audit: spread and informativeness are different questions.

The two tests that matter most here pin a mistake this project published and then
corrected. Gate mobility was presented as the honest alternative to counting distinct
values, on the grounds that random noise games a distinct-value count. It doesn't escape
that: mobility IS the distinct-value count on a 0.01 grid. Only a labelled measure tells
signal from noise.
"""

from __future__ import annotations

import math
import random

from harness import audit


def test_gate_mobility_is_the_distinct_value_count_on_a_grid():
    """The accept set only changes between thresholds t and t+0.01 when some score sits
    exactly at t, so mobility counts occupied bins below 1.00. Pinned so nobody presents
    it as independent of the distinct-value count again."""
    rng = random.Random(1)
    for _ in range(50):
        scores = [round(rng.random(), 2) for _ in range(rng.randint(1, 400))]
        below_one = {audit.to_grid(s) for s in scores} - {audit.GRID}
        assert audit.gate_mobility(scores) == len(below_one) / audit.GRID


def test_random_noise_has_high_mobility_and_no_information():
    """The objection a reader will raise, answered in code. Uniform noise spreads well,
    so any spread measure flatters it. AUROC against labels it cannot see sits at 0.5."""
    rng = random.Random(20260919)
    labels = [rng.random() < 0.3 for _ in range(5000)]
    noise = [rng.random() for _ in labels]
    assert audit.gate_mobility(noise) > 0.95
    assert abs(audit.auroc(noise, labels) - 0.5) < 0.03


def test_a_coarse_judge_has_low_mobility_even_when_its_calls_are_right():
    """A judge that only says 0, 0.5 or 1 can rank well and still give a threshold
    nothing to move between. This is the self-judging local model's failure."""
    labels = [True] * 60 + [False] * 140
    coarse = [1.0] * 50 + [0.5] * 10 + [0.0] * 130 + [0.5] * 10
    assert audit.gate_mobility(coarse) == 0.02  # 0.00 and 0.50; 1.00 is off the grid's steps
    assert audit.top_k_mass(coarse, 3) == 1.0
    assert audit.auroc(coarse, labels) > 0.9


def test_auroc_counts_ties_as_half():
    assert audit.auroc([0.5, 0.5], [True, False]) == 0.5
    assert audit.auroc([0.9, 0.1], [True, False]) == 1.0
    assert audit.auroc([0.1, 0.9], [True, False]) == 0.0
    assert math.isnan(audit.auroc([0.3, 0.4], [True, True]))


def test_auroc_within_restricts_to_one_bucket_of_another_judge():
    """Inside the items a coarse judge called certain, a finer judge still ranks."""
    coarse = [1.0, 1.0, 1.0, 1.0, 0.0]
    fine = [0.95, 0.90, 0.30, 0.20, 0.99]
    labels = [True, True, False, False, True]
    value, pos, neg = audit.auroc_within(fine, labels, coarse, 1.0)
    assert (value, pos, neg) == (1.0, 2, 2)


def test_bootstrap_resamples_whole_clusters():
    """Items from one question must travel together, or the interval is too narrow."""
    seen: list[list[int]] = []

    def stat(indices: list[int]) -> float:
        seen.append(indices)
        return float(len(indices))

    clusters = ["q1", "q1", "q1", "q2"]
    audit.bootstrap_ci(stat, clusters, resamples=200)
    for indices in seen:
        q1 = sum(1 for i in indices if i in (0, 1, 2))
        assert q1 % 3 == 0, "a question's three items were split across a resample"


def test_to_grid_is_not_moved_by_float_noise():
    assert audit.to_grid(0.29) == 29
    assert audit.to_grid(0.1 + 0.2) == 30
    assert audit.to_grid(1.0) == 100


def test_command_line_reports_both_halves(tmp_path, capsys):
    path = tmp_path / "scores.csv"
    path.write_text(
        "judge,question,score,label\n"
        "a,q1,0.9,1\na,q1,0.2,0\na,q2,0.8,1\na,q2,0.1,0\n",
        encoding="utf-8",
    )
    assert audit.main([str(path), "--label", "label", "--judge", "judge", "--cluster", "question"]) == 0
    out = capsys.readouterr().out
    assert "gate mobility" in out and "AUROC" in out and "1.000" in out
