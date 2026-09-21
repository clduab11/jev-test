---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - calibration
  - llm-as-judge
  - evaluation
  - selective-prediction
  - rag
pretty_name: "Jev vs LLM-as-judge: calibration statistics from a 500-question RAG benchmark"
size_categories:
  - n<1K
---

# Calibration statistics: a calibrated decision model vs an LLM judging itself

Aggregate statistics from a pre-registered 500-question retrieval-and-answer benchmark in which
every pipeline decision was made by a typed-question judge. Two judges answered **byte-identical
question sets over identical states**: TypeSafe's `jev-1.13.0`, and a local Gemma 4 E2B judging
its own work through the same interface.

This release answers one narrow question: **when you put a threshold on a judge's confidence,
does moving that threshold do anything?**

## The headline

| | Gemma 4 E2B judging itself | `jev-1.13.0` |
|---|---|---|
| Noul answers | 5,065 | 64,964 |
| Distinct values @2dp | 19 | **99** |
| Probability mass at extremes (≤0.02 or ≥0.98) | **90.5%** | 19.0% |
| Probability mass in interior (0.10–0.90) | 9.5% | **29.3%** |
| Observed range | 0.00 – 1.00 | 0.01 – 0.99 |
| **Gate mobility** | **18%** | **99%** |

**Gate mobility** is the share of adjacent threshold settings (0.00→1.00 in 0.01 steps) that
change the accept set at all. On the local model's score, **82% of thresholds you could pick are
indistinguishable from their neighbours** — the knob is not connected to anything. On the
calibrated model's score, 1% are.

"19 distinct values" sounds non-degenerate until you see where the mass is: **99.3% of the local
model's answers land on exactly three values — 0.00 (70.0%), 1.00 (20.2%), and 0.50 (9.1%).**
That 0.50 spike is verbalized confidence reporting "I don't know" as a literal coin flip.

Distinct-value count is deliberately **not** the headline metric, because uniformly random noise
scores well on it. Mass concentration and gate mobility are the honest measures, and both are
label-free — neither inherits the source benchmark's grader labels.

## What is in here

| File | Contents |
|---|---|
| `statistics.json` | Per-model request counts, token counts, primitive and stage breakdowns, question ids, full 2dp Noul histograms |
| `noul_histogram.csv` | Flat `model,probability,count` — the figure's input |
| `calibration_metrics.json` | Distinct values, interior mass, gate mobility per model |
| `calibration.png` / `.svg` | The two-panel figure |

## What is deliberately NOT in here, and why

The underlying corpus is 67,312 `(state, typed question, calibrated probability)` triples from
14,859 `jev-1.13.0` requests, plus 5,159 from 1,088 local-model requests, including 801 states
where **both models answered the identical question set**. That corpus exists and stays local.

Two reasons, both deliberate:

1. **Contract.** TypeSafe's [Master Customer Agreement](https://typesafe.ai/legal/mca) §4.2
   assigns Output ownership to the customer, so the triples are mine. But §2.3(b) forbids using
   Output "to perform model distillation, train a model to imitate the output of the Services, or
   develop (or to facilitate the development of) a similar or competing product or service," and
   §2.3 survives termination under §10.4. Publishing 67k labelled triples is the exact input a
   third party needs for that, and *facilitate* is the operative word.
2. **Third-party content.** The `state` field contains verbatim text fetched from third-party web
   pages. Redistributing that is a separate question under §2.3(l), independent of the first.

Aggregate statistics carry neither problem. If you want the raw corpus, the path is a licence
conversation with TypeSafe, not a download.

## Provenance

- **Source benchmark**: 500 SimpleQA questions, fixed seed 20260919, search results and page text
  frozen 2026-09-19, run completed and graded 2026-09-20.
- **Judge model**: `jev-1.13.0`, pinned — not the `jev-latest` alias. An alias moving silently
  re-tunes every threshold you published.
- **Local judge**: `unsloth/gemma-4-E2B-it-qat-GGUF` answering the same questions through
  `system-one-adapter`, verbalized confidence.
- **Cost of the judged arm**: $0.53 for 13,974 requests over 500 questions.
- **Questions**: five passage-relevance Nouls (S1), a sufficiency pair (S2), and a verification
  Choice plus an addresses-query Noul (S4). Verbatim instructions and criteria are in
  `judge/questions.v1.json` in the source repository.

## Limitations, stated up front

- **One dataset, one snapshot, one task family.** These are English web-search passages judged for
  relevance and evidential support. Nothing here generalises to other domains without testing.
- **The local-model arm is verbalized confidence**, which is the known-degenerate configuration.
  Constrained-decode logprobs and self-consistency vote fractions are cheaper sources of graded
  confidence and are **not** measured here. Anyone citing this as "local models cannot be
  calibrated" is overreading it; the claim is about verbalized confidence specifically.
- **Stage mix differs between the two arms** (the local arm ran 4,980 S1 Nouls against 63,080),
  so compare distributions, not raw counts.
- **Gate mobility is a property of the score distribution, not of accuracy.** A well-spread score
  can still be wrong. Spread is necessary for a threshold to function, not sufficient for it to
  function *well*.

## Citation

```bibtex
@misc{dukes2026jevcalibration,
  title  = {Calibration statistics: a calibrated decision model vs an LLM judging itself},
  author = {Dukes, Chris},
  year   = {2026},
  url    = {https://huggingface.co/datasets/clduab11/jev-calibration-statistics}
}
```

MIT for these statistics and the code that produced them. The source benchmark datasets keep
their own licences.
