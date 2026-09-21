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

Two judges answered **byte-identical question sets over identical states** in a 500-question
retrieval benchmark: a local Gemma 4 E2B judging its own work, and TypeSafe's `jev-1.13.0`.
This release is the confidence distributions they produced.

![calibration](calibration.png)

|  | Gemma 4 E2B (verbalized) | `jev-1.13.0` |
|---|---|---|
| Noul answers | 5,065 | 64,964 |
| Distinct values @2dp | 19 | 99 |
| Mass at extremes (≤0.02 or ≥0.98) | **90.5%** | 19.0% |
| Mass in interior (0.10–0.90) | 9.5% | **29.3%** |
| Range | 0.00 – 1.00 | 0.01 – 0.99 |
| **Gate mobility** | **18%** | **99%** |

**Gate mobility** = the share of adjacent threshold settings (0.00→1.00 in 0.01 steps) that
change the accept set at all. At 18%, **82% of the thresholds you could pick on the local
model's score are indistinguishable from their neighbours** — the knob isn't connected to
anything.

"19 distinct values" sounds fine until you see where the mass is: **99.3% of the local model's
answers land on three numbers — 0.00 (70.0%), 1.00 (20.2%), 0.50 (9.1%).** That 0.50 spike is
"I don't know" as a literal coin flip.

Distinct-value count is deliberately not the headline: uniform random noise scores well on it.
Mobility and mass concentration are the honest measures, and both are label-free.

## Files

`noul_histogram.csv` (the viewer table) · `statistics.json` (per-model counts, stages,
question ids, full histograms) · `calibration_metrics.json` · `calibration.png` / `.svg`

## Limitations

- **The local arm is verbalized confidence** — the known-worst way to get a probability out of a
  small model. Constrained-decode logprobs and self-consistency vote fractions are free and
  likely much better, and are **not** tested here. This is not evidence that local models can't
  be calibrated.
- One dataset, one snapshot, one task family (English web passages judged for relevance and
  evidential support).
- Stage mix differs between arms (4,980 vs 63,080 S1 Nouls) — compare distributions, not counts.
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
