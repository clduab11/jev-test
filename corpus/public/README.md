---
license: mit
language:
  - en
tags:
  - calibration
  - llm-as-judge
  - evaluation
  - selective-prediction
pretty_name: "Gate mobility: can a threshold on a judge's confidence do anything?"
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: noul_histogram.csv
---

# Gate mobility: can a threshold on a judge's confidence do anything?

Two judges answered **the same question about the same state, 3,885 times**, across the 21
pilot questions of a retrieval benchmark: a local Gemma 4 E2B judging its own work, and
TypeSafe's `jev-1.13.0`. This release is the confidence distributions they produced.

![calibration](calibration.png)

| Paired: same state, same question (n=3,885, 21 questions) | Gemma 4 E2B (verbalized) | `jev-1.13.0` |
|---|---|---|
| Distinct values @2dp | 12 | 99 |
| Mass on its top 3 values | **99.6%** | 34.4% |
| Mass in interior (0.10–0.90) | 6.3% | **30.3%** |
| **Gate mobility** | **11%** | **99%** |
| Agree on the yes/no call at 0.5 | **82.8%** | |

**Gate mobility** = the share of adjacent threshold settings (0.00→1.00 in 0.01 steps) that
change the accept set at all. At 11%, **89% of the thresholds you could pick on the local
model's score are indistinguishable from their neighbours** — the knob isn't connected to
anything.

The two judges agree on the yes/no call 82.8% of the time. The local model isn't mostly
*wrong* — it just can't say *how sure* it is. **99.6% of its answers land on three numbers:
0.00, 1.00, and 0.50.** That 0.50 is "I don't know" as a literal coin flip.

Only the paired set is a fair comparison. The self-judge arm ran on the 21-question pilot,
while Jev ran all 500; unpaired totals (5,065 vs 64,964 Nouls) are in `statistics.json` but
compare different question sets.

Distinct-value count is deliberately not the headline: uniform random noise scores well on it.
Mobility and mass concentration are the honest measures, and both are label-free.

## Files

`noul_histogram.csv` (the viewer table — paired histograms) · `statistics.json` (per-model
counts, stages, question ids, paired and unpaired histograms) · `calibration_metrics.json` ·
`calibration.png` / `.svg`

## Limitations

- **The local arm is verbalized confidence** — the known-worst way to get a probability out of a
  small model. Constrained-decode logprobs and self-consistency vote fractions are free and
  likely much better, and are **not** tested here. This is not evidence that local models can't
  be calibrated.
- **21 questions.** 3,885 judgments sounds large, but they are many passages per question and
  correlated within a question. Treat it as a 21-question result.
- One dataset, one snapshot, one task family (English web passages judged for relevance and
  evidential support).
- Mobility is a property of the score distribution, not of accuracy. Spread is necessary for a
  threshold to work, not sufficient.

## Not included, and why

The underlying 67,312 `(state, question, probability)` triples stay local. TypeSafe's
[MCA](https://typesafe.ai/legal/mca) §4.2 assigns Output ownership to the customer, so they're
mine — but §2.3(b) bars using Output to "perform model distillation, train a model to imitate
the output of the Services, or ... facilitate the development of a similar or competing product
or service." Publishing them is the input for exactly that. Separately, the `state` field holds
verbatim third-party page text (§2.3(l)). Aggregate statistics carry neither problem.

## Provenance

500 SimpleQA questions, seed 20260919, snapshot frozen 2026-09-19, graded 2026-09-20 by
`claude-sonnet-5` (not humans). Judge pinned to `jev-1.13.0`, never the moving `jev-latest`
alias. Judged arm cost $0.53 for 13,974 requests.

Code and raw per-question results: <https://github.com/clduab11/jev-test>

MIT.
