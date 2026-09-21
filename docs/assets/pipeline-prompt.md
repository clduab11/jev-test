# Image prompt for the pipeline figure

Paste the prompt below into an image model to regenerate `pipeline.png`. It describes the pipeline as run on the 500-question SimpleQA track, graded 2026-09-20, with the decline counts from that run on the exits. The Mermaid source in `docs/DESIGN.md` is the ground truth for the boxes and arrows; this prompt exists because image models draw a cleaner poster than Mermaid renders.

Check the result against the arrow list before committing it.

The committed `pipeline.png` came from this prompt on 2026-09-21 and was checked against all 24 arrows. Two errors were corrected by hand in the pixels: an arrow from S0 straight into S1 that does not exist was removed, and the S4 threshold, drawn as 0.90, was changed to 0.80. Three small defects were left: the "survivors" arrow and the "benchmark replay" arrow each carry a second arrowhead, and the arrow from S1 to quarantine has no "injection" label.

## Why the prompt is written this way

Three generations taught the same lesson: the image model connects boxes that sit next to each other, whether or not an arrow exists. The 2026-09-19 figure drew an arrow from quarantine to the evidence wing. The first 2026-09-21 figure placed the four retrieval boxes in one row and chained them left to right, sent evidence straight from S1 to S2, and gave quarantine an outgoing arrow. Shaded group regions made this worse, because they forced unrelated boxes into rows, and the model put boxes in the wrong region both times.

So this prompt fixes the layout on a grid and puts the real chain of stages along the middle row, where the model's habit of joining neighbours draws the true arrows. It drops the shaded regions and lets the colours and legend show the grouping. It asks for one arrowhead per arrow, and it names every false arrow the model has drawn so far.

## What changed from the 2026-09-19 prompt

- Two exits to Abstain that the pipeline always had and the old figure left out: S3, when Gemma says the evidence is not enough (`harness/arms/d_jev.py`), and S4, when the answer misses the question (`harness/stages/s4_verify.py`, `addresses_min` 0.5).
- Decline counts from the 500-question run on each exit to Abstain.
- Rank fusion names the evidence score it uses: Jev's `contains_answer_evidence` probability, which scored AUROC 0.899 against a label no model produced (`results/informativeness.json`).
- Edge labels on the two S0 exits, and the 5-page and 2,500-token caps, from the Mermaid source.

## Prompt

Draw a clean, flat technical diagram of a question-answering pipeline. Wide horizontal format, white background, no gradients, no 3D, no photographs, sans-serif labels, high contrast, generous spacing between boxes. Read the whole description before drawing.

Title at the top, centred: "jev-test pipeline: Jev decides, Gemma writes, MemPalace remembers"

Colours: dark blue rounded boxes are decisions made by a decision model called Jev. Brown rounded boxes are the two places a small language model called Gemma writes text. Green cylinders are a local memory store called MemPalace. Light grey boxes are retrieval. White boxes with a thin outline are inputs and outputs. Put a small legend for these five colours at the bottom left. Do not draw any shaded background regions or region titles.

LAYOUT. Three rows. Place every box exactly where this sketch shows it:

TOP ROW:     .         .    [MemPalace search]  [Fetch and chunk]  (evidence wing)   .     .     (verified wing)   .
MIDDLE ROW:  [Question] [S0]       .               [S1]             [Rank fusion]   [S2]  [S3]      [S4]         [Answer]
BOTTOM ROW:  .         .    [SearXNG]           (quarantine)         .          [Gemma new query]  [Abstain]      .

So: MemPalace search sits above and SearXNG sits below the gap between S0 and S1. Fetch and chunk sits directly above S1, and quarantine directly below S1. The evidence wing sits above Rank fusion. The small Gemma query box sits directly below S2. The verified wing sits directly above S4. Abstain sits below, between S3 and S4.

Boxes, with their exact label text:

- White: "Question"
- Dark blue: "S0 intake. Search or not? Web, memory, or both? How recent? Which category?"
- Dark blue: "S1 gate, one passage per call. Relevant? Usable fact? Contradicts the premise? Prompt injection? Boilerplate?" with a small grey note under it: "source label assigned by code from a short list of known sites"
- Dark blue: "S2 sufficiency. Enough to answer? Do sources disagree?"
- Dark blue: "S4 verify, one claim per call. Supported, contradicted, or unsupported, with confidence. Then: does the answer address the question?"
- Brown: "S3 Gemma writes at most 5 sentences, every fact cited by passage id"
- Brown, smaller: "Gemma writes one new search query"
- Grey: "SearXNG: up to 30 results"
- Grey: "MemPalace search: verified wing, then evidence wing"
- Grey: "Fetch and chunk: 300 tokens, at most 5 pages, stored word for word"
- Grey: "Rank fusion: search rank + memory similarity + Jev's evidence score. Top 6 passages, 2,500 tokens." with a small yellow tag attached: "evidence score: AUROC 0.899"
- Green cylinders: "quarantine", "evidence wing", "verified wing"
- White: "Answer with a probability on every sentence"
- White: "Abstain"

ARROWS. Draw exactly these 24 arrows and no others. Every arrow has exactly one arrowhead, at the second box named. Where two arrows join the same pair of boxes in opposite directions, draw them as two separate parallel arrows.

1. Question to S0
2. S0 to SearXNG, label "web or both"
3. S0 to MemPalace search, label "memory or both"
4. SearXNG to S1
5. MemPalace search to S1
6. S1 up to Fetch and chunk, label "survivors, at most 5 pages"
7. Fetch and chunk back down to S1, label "chunks judged again"
8. Fetch and chunk to evidence wing
9. S1 down to quarantine, label "injection"
10. S1 to Rank fusion, label "evidence"
11. Rank fusion to S2
12. S2 down to Gemma writes one new search query, label "not enough, first time"
13. Gemma writes one new search query to SearXNG, solid line running along the bottom of the picture, label "live"
14. Gemma writes one new search query to MemPalace search, dashed line, label "benchmark replay: a second look at the frozen pages"
15. S2 to Abstain, label "not enough, second time: 60"
16. S2 to S3, label "enough"
17. S3 to Abstain, label "Gemma says the evidence is not enough: 2"
18. S3 to S4
19. S4 to Answer, label "kept, confidence 0.80 or higher"
20. S4 up to verified wing, label "kept claims"
21. S4 to Abstain, label "nothing survives: 44"
22. S4 to Abstain, label "answer misses the question: 29"
23. verified wing to MemPalace search, dashed line running along the top of the picture, label "recalled next time, judged again"
24. evidence wing to MemPalace search, dashed line running along the top of the picture, no label

DO NOT DRAW any of these arrows, which do not exist:
- between MemPalace search and SearXNG
- between MemPalace search and Fetch and chunk
- between Fetch and chunk and Rank fusion
- between Rank fusion and evidence wing
- between S1 and S2 directly (evidence reaches S2 only through Rank fusion)
- between any two cylinders
- from quarantine to anything (quarantine only receives the one arrow from S1)
- between SearXNG and quarantine

Two lines of small text under the diagram:
"The small model touches the process in two places. Every other decision is a typed question with a probability, and thresholds live in code."
"Numbers on the arrows into Abstain are questions declined on the 500-question SimpleQA run, graded 2026-09-20."

No other text, no logos, no icons of people, no watermark.
