# Changelog

## Pilot 2026-09-19 (arms A and B, 20 questions)

A pilot only. Twenty SimpleQA questions drawn with seed 20260919, frozen as snapshot pilot-20260919. The numbers below check that the harness works end to end and carry wide intervals; nothing here is a finding.

| Arm | Correct | Incorrect | Not attempted | Truthfulness [95% CI] | Hallucination rate [95% CI] | Coverage [95% CI] |
|---|---|---|---|---|---|---|
| A, generator alone | 0 | 6 | 14 | -0.30 [-0.50, -0.10] | 1.00 [1.00, 1.00] | 0.30 [0.10, 0.50] |
| B, naive RAG, top 8 pages, 12k tokens | 12 | 4 | 4 | 0.40 [0.05, 0.75] | 0.25 [0.06, 0.47] | 0.80 [0.60, 0.95] |

Grader: claude-sonnet-5 through the LiteLLM gateway with the official SimpleQA prompt, committed under harness/grading/prompts/. Intervals are bootstrap 95 percent over questions, 1,000 resamples, seed 20260919. Generator: unsloth/gemma-4-E2B-it-qat-GGUF (UD-Q4_K_XL) in Unsloth Desktop, thinking off, seed 20260919. Result files: results/A_simpleqa_pilot-20260919.json and results/B_simpleqa_pilot-20260919.json.

Observations to carry into the full run, none of them conclusions:

- Arm B wrote two fabricated citation ids in 20 answers: one id past the last search result, one chunk index past the last chunk of its page. Both sentences were stripped as section 6 requires; one of them held the correct answer.
- The model sometimes puts several ids in one bracket, like [r01c1, r02c0]. The claim parser accepts that form.
- The longest arm B prompt was 13,834 tokens, inside the 16,384 context.
- Arm A answered 6 of 20 and all 6 were wrong; the other 14 were declines.

Decisions made while building, recorded because the spec left them open:

- The snapshot is frozen with category general and no time range, and every result page is fetched instead of the first five, so any arm can pick any page offline. S0 answers are still evaluated and logged per arm.
- Citation ids in prompts are short (r07c2 style). The manifest maps each one to its MemPalace drawer id, so a citation still resolves to one verbatim stored passage.
- The generator is driven through /v1/completions with the Gemma 4 turn format rendered in code. The chat template shipped with Unsloth Desktop prefills an empty thought channel when thinking is off, and Gemma 4 E2B then writes its reasoning into the visible answer without answering.
- Arm A sends the bare question with the system prompt "Answer the question in at most five sentences."
- Arm B falls back to the search snippet for a page that failed to fetch, with citation id rNNs.
- The context length is 16,384 instead of the 8,192 in the generator block, because arm B stuffs up to 12,000 evidence tokens.
- MemPalace 3.10 does not expose the add_drawer signature the spec recorded. Evidence is filed with the metadata conventions of file_conversation_exchange and deterministic drawer ids hashed from (url, chunk_index).

## v1.1 candidates

- A repair pass that lets the generator rewrite using only kept claims (already listed in the spec as a v1.1 option).
- When a sentence cites one bad id next to valid ones, strip the bad id instead of the whole sentence. This changes section 6 and is not applied in v1.
- Log the S0 category and recency choices per question so a later snapshot can be frozen per category.

## v1-prereg (2026-09-19)

Judgment spec v1 written and pre-registered before any benchmark run.

- Five decision stages (S0 intake, S1 passage gate, S2 sufficiency, S4 verification, M2 memory write-back) with question text, criteria, and thresholds fixed in `judge/questions.v1.json`.
- Seven arms: A plain, B naive RAG, C-laya, C-classical, C-self, D Jev, E ceiling.
- Three tracks: web (SimpleQA, FreshQA), controlled (RGB, CRAG), memory (two passes).
- Pre-registered bars for arm D on the web track and for the memory track. Predicted negatives written down.
- MemPalace added as the verbatim evidence store, the frozen snapshot, and the verified-claim memory.
- Repository scaffold: folder layout, Judge interface stub, dependency manifest, environment template.

No code paths run yet. No results exist.
