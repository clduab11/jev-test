# Changelog

## Pilot 2026-09-19 (arm D, 20 questions)

The same twenty questions and the same frozen snapshot as the arms A and B pilot below, now with the full pipeline: S0 intake, S1 gate on snippets and chunks, rank fusion, S2 sufficiency with one refinement round, S3 generation, S4 verification, M2 write-back. Twenty questions is a shape and cost check. Nothing here is a finding.

| Arm | Correct | Incorrect | Not attempted | Truthfulness [95% CI] | Hallucination rate [95% CI] | Coverage [95% CI] |
|---|---|---|---|---|---|---|
| A, generator alone | 0 | 6 | 14 | -0.30 [-0.50, -0.10] | 1.00 [1.00, 1.00] | 0.30 [0.10, 0.50] |
| B, naive RAG, top 8 pages, 12k tokens | 12 | 4 | 4 | 0.40 [0.05, 0.75] | 0.25 [0.06, 0.47] | 0.80 [0.60, 0.95] |
| D, Jev 1.13.0 decides | 15 | 0 | 5 | 0.75 [0.55, 0.95] | 0.00 [0.00, 0.00] | 0.75 [0.55, 0.95] |

Pre-registered bars for arm D, evaluated on the pilot and labelled as such by the grade step: coverage 0.75 (bar 0.50), hallucination rate when attempting 0.00 (bar 0.10), truthfulness gain over arm B 0.35 (bar 0.15), citation support rate 0.91, 20 of 22 kept claims, interval [0.79, 1.00] (bar 0.90). Four of four cleared on twenty questions; fifteen attempts with no error is consistent with a true error rate up to one in five.

Cost and load: 979 Jev requests, 49 per question, 837,000 input tokens, $0.035 in total, about 42,000 input tokens per question. The spec's section 12 budget was 90 requests; pages whose snippet is dropped are never fetched, which is where the difference comes from. Mean wall clock 4.8 s per question, of which 1.8 s is the generator. Judge: jev-1.13.0 at api.typesafe.ai, the responding model id logged on every answer set. Citation support grader: claude-sonnet-5, prompt committed at harness/grading/prompts/claim_support_grader.txt. Result file: results/D_simpleqa_pilot-20260919.json, which carries every stage's decisions, the six evidence passages, and per-stage request counts for each question.

Observations, none of them conclusions:

- Five abstentions: two where S2 found the evidence insufficient after the refinement round, one where S4 stripped every claim, two where the answer-level check said the surviving answer did not address the question. Both of the last kind were cases where the draft answered a neighbouring question (a year for the wrong event; a description of an artwork instead of its title).
- S4 stripped nine claims in total: six supported below the 0.80 confidence gate, two unsupported, one contradicted below the gate. The two claims the outside grader called unsupported were both kept by S4 at 0.92 and above; one is the Kashmir cardiologist answer, where the grader read the passage as saying "of Kashmiri origin" while the claim says "in Kashmir".
- The refinement round ran three times and turned one abstention into a correct answer. For the Kashmir question the first pass had judged the right page's snippet off topic; the refined query surfaced that page's own chunk from the frozen set at evidence probability 0.97, and rank fusion put it first because it appeared in all three rank lists.
- S2 flagged conflicting evidence on two questions. No S1 passage was routed to the conflict block on any question; SimpleQA has no false premises, so that path is exercised only by the fake-judge tests so far.
- S0 chose category general on 17 questions, science on 2, news on 1, and a recency window on 1; the snapshot was frozen with general and no window, so these are logged, not acted on. `needs_search` was at least 0.98 on every question.
- Of the 600 snippets judged, 157 were evidence, 74 boilerplate, 278 off topic, 65 nothing usable. Of the 286 chunks judged, 76 were evidence. On eight questions, survivor snippets beyond the five-page fetch cap were left unfetched; the count is recorded per question.
- Zero fabricated citation ids in arm D against two in arm B on the same questions. M2 wrote 24 verified claims to the pass-one memory palace.
- Arm B was re-run after the sentence-splitter fix below. Its grades did not change.

Decisions made while building, recorded because the spec left them open:

- A refinement round cannot run a new SearXNG query against a frozen snapshot. In replay it searches the snapshot's evidence wing with the refined query, restricted to the question's own drawers, and the hits join the ranking with a palace similarity. A chunk already judged in round one keeps its verdict and gains the similarity; an unjudged chunk is judged like any other passage.
- A survivor snippet whose page is beyond `max_pages` is not fetched and is not evidence. The spec lets a snippet stand in only for a page that failed to fetch. The number of such survivors is logged per question.
- When S2 flags conflicting evidence and no S1 passage sits in the conflict block, the generator's CONFLICTS block carries one note line saying the evidence passages disagree, so the system prompt's conflict instruction still applies.
- Only a fetched chunk is filed to the quarantine wing on an injection verdict, as the spec says; a snippet is dropped and logged, and a recalled drawer is flagged in place.
- `source_type` comes from a domain allowlist in `harness/stages/s1_gate.py`: exact host, then hostname prefix, then registered-domain suffix, with `unknown` for everything else. The list is short on purpose.
- M2 stores each kept claim with its citation tags removed; the tags are per-question ids that mean nothing on recall, and the evidence drawer ids are in the drawer's metadata.
- Arm D keeps its verified claims in a memory palace of its own under palace/, one per snapshot and arm, wiped at the start of a pass-one run. The snapshot's palace is opened read-only.
- The sentence splitter no longer breaks after "Dr.", initials, "U.S." and similar. Found on the Kashmir answer, where three "Dr." fragments outlived the claims they belonged to.
- The FreshQA sheet holds 155 fast-changing questions across both splits, fewer than the 200 the spec asked for, so the loader returns the whole fast-changing test slice and the manifest records the count.
- Arm C-self: the adapter client works against the local model once the bearer token the OpenAI SDK always attaches is removed; the Unsloth Desktop proxy rejects any token it did not issue and accepts requests with none. On the smoke test the small model returned probabilities of exactly 0.0 and 1.0, so the arm has not been run; whether its numbers carry any information is the first thing to check.
- Arm C-self fallback: when the small model returns no valid answer set even after the adapter's retry, the judge answers 0.5 for a yes-or-no question and a uniform spread for a choice, counts the event, and the result file records how many requests fell back. A run is never aborted by one bad judgment. The two-question C-self run started with this in place; its numbers are not in this changelog.
- Arm C-laya: `harness/judge/laya_shim.py` serves Laya on `/v1/systemone` so HttpJudge can use it unchanged. The wire layer is tested with a stand-in model and a real HTTP round trip; Laya itself (torch, transformers, about 421M parameters) is not installed on this machine yet, so the arm has not run.
- The 500-question freeze: the public engines behind the home SearXNG suspend themselves after a few dozen queries (Brave and Google CSE for too many requests, Startpage and at times DuckDuckGo on a CAPTCHA, Wikipedia on access denied). Of the first 236 questions frozen, 42 got no results, 12 fewer than ten, and none the full thirty; the pilot's twenty had 27 on average. The freeze now waits between searches, prints an engine report (`--report`), and can re-search questions below a result count (`--redo-below`) once the engines recover. Ten results from one engine is not the snapshot the spec describes, so nothing is reported from it until the short questions are redone. The first freeze also died at question 236 on a Windows file lock while another process read the manifest; the manifest write now retries.

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
- Log the S0 category and recency choices per question so a later snapshot can be frozen per category. (Logged since the arm D pilot; a per-category freeze is still open.)
- Let a survivor snippet beyond the fetch cap count as evidence on its own. On the pilot, eight questions left such snippets unused.

## v1-prereg (2026-09-19)

Judgment spec v1 written and pre-registered before any benchmark run.

- Five decision stages (S0 intake, S1 passage gate, S2 sufficiency, S4 verification, M2 memory write-back) with question text, criteria, and thresholds fixed in `judge/questions.v1.json`.
- Seven arms: A plain, B naive RAG, C-laya, C-classical, C-self, D Jev, E ceiling.
- Three tracks: web (SimpleQA, FreshQA), controlled (RGB, CRAG), memory (two passes).
- Pre-registered bars for arm D on the web track and for the memory track. Predicted negatives written down.
- MemPalace added as the verbatim evidence store, the frozen snapshot, and the verified-claim memory.
- Repository scaffold: folder layout, Judge interface stub, dependency manifest, environment template.

No code paths run yet. No results exist.
