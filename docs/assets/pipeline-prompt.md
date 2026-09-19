# Image prompt for the pipeline figure

Paste the prompt below into an image model to regenerate `pipeline.png`. It describes the pipeline as built on 2026-09-19, after arm D ran on the 20-question pilot. The Mermaid source in README.md is the ground truth; this prompt exists because image models draw a cleaner poster than Mermaid renders.

## Prompt

Draw a clean, flat, wide technical diagram (16:9, white background, no gradients, no 3D, no photographs, sans-serif labels, high contrast) of a question-answering pipeline. Read the whole description before drawing; every box below must appear once, with its exact label text, and the arrows must connect exactly as listed.

Colour code, shown in a small legend at the bottom left: dark blue rounded boxes are decisions made by a decision model called Jev; brown rounded boxes are the only two places a small language model called Gemma writes text; green cylinders are a local memory store called MemPalace; grey boxes are retrieval; white boxes with a thin outline are inputs and outputs.

Flow from left to right.

1. A white rounded box on the far left: "Question".
2. An arrow to the first dark blue box: "S0 intake. Search or not? Web, memory, or both? How recent? Which category?"
3. From S0, two arrows: one down to a grey box "SearXNG: up to 30 results" and one to a grey box "MemPalace search: verified wing, then evidence wing".
4. Both grey boxes feed a dark blue box: "S1 gate, one passage per call. Relevant? Usable fact? Contradicts the premise? Prompt injection? Boilerplate?" Under it, a small grey note: "source label assigned by code from a short list of known sites".
5. From S1, three labelled exits: an arrow labelled "survivors, at most 5 pages" to a grey box "Fetch and chunk: 300 tokens, stored word for word"; an arrow labelled "injection" to a green cylinder "quarantine"; and an arrow labelled "evidence" to a grey box "Rank fusion: search rank + memory similarity + judge score, top 6 passages".
6. The "Fetch and chunk" box has an arrow to a green cylinder "evidence wing" and an arrow back into S1 labelled "chunks judged again".
7. Rank fusion feeds a dark blue box: "S2 sufficiency. Enough to answer? Do sources disagree?"
8. From S2, three exits: "enough" to the brown box in step 9; "not enough, first time" to a small brown box "Gemma writes one new search query"; "not enough, second time" to a white rounded box "Abstain".
9. The small brown query box has two arrows: a solid one back to "SearXNG" labelled "live", and a dashed one to "MemPalace search" labelled "benchmark replay: a second look at the frozen pages".
10. The main brown box: "S3 Gemma writes at most 5 sentences, every fact cited by passage id".
11. S3 feeds a dark blue box: "S4 verify, one claim per call. Supported, contradicted, or unsupported, with confidence. Then: does the answer address the question?"
12. From S4, three exits: "kept, confidence 0.80 or higher" to a white rounded box "Answer with a probability on every sentence"; "kept claims" to a green cylinder "verified wing"; "nothing survives" to the same "Abstain" box.
13. A dashed arrow from "verified wing" back to "MemPalace search" labelled "recalled next time, judged again". A dashed arrow from "evidence wing" to "MemPalace search".

Group the four dark blue boxes inside a faint blue region titled "Jev decides: typed answers with probabilities". Group the three green cylinders inside a faint green region titled "MemPalace: local, verbatim". Group SearXNG, Fetch and chunk, MemPalace search and Rank fusion inside a faint grey region titled "Retrieval".

Add one line of small text under the diagram: "The small model touches the process in two places. Every other decision is a typed question with a probability, and thresholds live in code."

No other text, no logos, no icons of people, no watermark.
