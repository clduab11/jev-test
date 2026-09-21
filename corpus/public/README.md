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

Two judges answered the same question about the same passage 3,885 times, across the 21 pilot
questions of a retrieval benchmark. One was a local Gemma 4 E2B judging its own work. The other
was TypeSafe's `jev-1.13.0`. This release is the confidence values they gave.

![calibration](calibration.png)

| Same passage, same question (n=3,885, 21 questions) | Gemma 4 E2B (verbalized) | `jev-1.13.0` |
|---|---|---|
| Distinct values at 2 decimal places | 12 | 99 |
| Share of answers on its top 3 values | 99.6% | 34.4% |
| Share between 0.10 and 0.90 | 6.3% | 30.3% |
| Gate mobility | **11%** | **99%** |
| Agree on the yes/no call at 0.5 | 82.8% | |

Gate mobility is the share of threshold steps that change anything. Move an accept threshold
from 0 to 1 in steps of 0.01 and count how often a step changes which answers get through. At
11%, 89% of the thresholds you could pick on the local model's score give the same result as
the one beside them. Tuning that threshold does nothing.

The two judges agree on the yes/no call 82.8% of the time, so the local model is mostly right.
What it can't do is say how sure it is. It put 99.6% of its answers on three values: 0.00,
1.00 and 0.50, and the 0.50 is how it says "I don't know."

Only the paired set is a fair comparison. The self-judge arm ran on the 21-question pilot and
Jev ran all 500, so the unpaired totals in `statistics.json` (5,065 and 64,964 answers) cover
different questions.

The count of distinct values is not the headline because random noise scores well on it.
Gate mobility and where the answers land are better measures, and neither needs labels.

## Files

`noul_histogram.csv` is the table shown above as paired histograms. `statistics.json` has
per-model counts, stages, question ids, and paired and unpaired histograms.
`calibration_metrics.json` has the numbers in the table. `calibration.png` and `.svg` are the
figure.

## Limitations

- The local arm used verbalized confidence: the model is asked how sure it is and the number
  is read back. That is the worst way to get a probability out of a small model. Constrained
  decoding with token logprobs and self-consistency vote fractions cost nothing and are likely
  much better. Neither is tested here, so this is not evidence that local models can't be
  calibrated.
- 21 questions. The 3,885 judgments are many passages per question and correlated within a
  question.
- One dataset, one snapshot, one task: English web passages judged for relevance and support.
- Gate mobility describes how the scores are spread, not whether they are right. Spread is
  needed for a threshold to work, and it isn't enough on its own.

## What is not included

The 67,312 underlying (state, question, probability) records stay local, for two reasons.
TypeSafe's [customer agreement](https://typesafe.ai/legal/mca) assigns ownership of model
output to the customer in §4.2, so the records are mine, but §2.3(b) bars using output to
"perform model distillation, train a model to imitate the output of the Services, or ...
facilitate the development of a similar or competing product or service." Publishing them
would hand someone the input for that. The `state` field also holds text copied from
third-party web pages (§2.3(l)). Aggregate statistics carry neither problem.

## Provenance

500 SimpleQA questions, seed 20260919, snapshot frozen 2026-09-19, graded 2026-09-20 by
`claude-sonnet-5` (a model, not people). The judge is pinned to `jev-1.13.0`, not the
`jev-latest` alias, which moves. The judged arm cost $0.53 for 13,974 requests.

Code and per-question results: <https://github.com/clduab11/jev-test>

MIT.
