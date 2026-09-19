# Changelog

## v1-prereg (2026-09-19)

Judgment spec v1 written and pre-registered before any benchmark run.

- Five decision stages (S0 intake, S1 passage gate, S2 sufficiency, S4 verification, M2 memory write-back) with question text, criteria, and thresholds fixed in `judge/questions.v1.json`.
- Seven arms: A plain, B naive RAG, C-laya, C-classical, C-self, D Jev, E ceiling.
- Three tracks: web (SimpleQA, FreshQA), controlled (RGB, CRAG), memory (two passes).
- Pre-registered bars for arm D on the web track and for the memory track. Predicted negatives written down.
- MemPalace added as the verbatim evidence store, the frozen snapshot, and the verified-claim memory.
- Repository scaffold: folder layout, Judge interface stub, dependency manifest, environment template.

No code paths run yet. No results exist.
