# jev-test

A small model on a laptop writes the answers. A separate judge model, TypeSafe's Jev (`jev-1.13.0`), makes every call the small model is bad at: whether to search, which pages count, whether there is enough evidence, and whether each sentence is backed by what it cites. SearXNG finds the pages and MemPalace remembers them.

This repository measures how well that works on 500 SimpleQA questions. Jev's confidence score tracks whether a passage holds the answer (AUROC 0.899), and it ranks passages better than the small model, Gemma 4 E2B, judging itself, including passages Gemma marked as fully certain. The repository also ships `harness/audit.py` for checking any judge's confidence scores.

End to end, the judged pipeline did not beat the no-judge baseline: it scored 0.612 against 0.740, missed its main pre-registered bar, made about the same number of mistakes on the questions both answered, and lost because it declined far more questions. That is written up under [Open problems](#open-problems).

The rules were fixed before the first run, in [docs/JUDGMENT_SPEC.md](docs/JUDGMENT_SPEC.md). The pipeline and the 20-question pilot are in [docs/DESIGN.md](docs/DESIGN.md).

## The problem in one paragraph

Small language models are cheap and private. A two-billion-parameter model fits in about three gigabytes of memory and runs on a normal laptop. It writes clear sentences. But when you ask it to look something up, it makes poor decisions. It searches for the wrong thing. It treats junk pages as evidence. It answers when the evidence is missing. It sometimes cites a source that says no such thing. Those failures are decisions. The writing was never the problem. The idea here is to take every decision away from the small model and give it to a model built only for decisions.

## What it found

Intervals are 95%, from a bootstrap over whole questions (1,000 resamples), because passages from one question move together.

### 1. Jev's confidence tracks whether a passage holds the answer

For each passage, Jev answers "does this passage contain evidence that answers the question?" with a probability. The check uses no model: a passage counts as positive when the gold answer string appears in its text (`harness/grading/recall.py`). Golds of 4 characters or fewer are dropped, because short strings match pages for unrelated reasons.

Across 9,075 passages from 349 questions, the AUROC is 0.899 [0.881, 0.916]. Random scores sit at 0.5. With golds longer than 7 characters only (8,056 passages) it is 0.902, and measured within each question it is 0.925.

![Does a higher score mean the passage has the answer?](docs/assets/informativeness.png)

### 2. Jev ranks better than Gemma judging itself, including where Gemma claims certainty

On the 20-question pilot the same pipeline also ran with Gemma 4 E2B judging its own work, on the same passages. Gemma's confidence is verbalized: it is asked how sure it is and the number is read back. On 473 passages from 12 questions, Jev's AUROC is 0.926 [0.857, 0.977] and Gemma's is 0.804 [0.723, 0.852], a difference of +0.122 [+0.046, +0.199]. The sample is small; see the notes.

Gemma marked 189 of those passages 1.00, fully certain. Only 40% of them (76) contain the gold answer string. Inside that group Jev's score still separates the two kinds, with AUROC 0.835 [0.662, 0.955]. Only 7 of the 12 questions have both kinds there; leaving out one question at a time gives 0.814 to 0.909.

### 3. Gemma judging itself gives a threshold nothing to hold

This finding measures spread only. On 3,885 judgments both models made, they gave the same yes-or-no call at 0.5 on 82.8%. Gemma used 12 distinct values and put 99.6% of its answers on three of them: 0.00, 1.00 and 0.50. Jev used 99 distinct values and put 30.3% of its answers between 0.10 and 0.90, against Gemma's 6.3%.

![Where each judge puts its confidence, and whether a threshold moves](docs/assets/calibration.png)

Gate mobility, the share of 0.01 threshold steps from 0.00 to 0.99 that change which answers pass, is 11% for Gemma and 99% for Jev. It equals the count of distinct values below 1.00 on a 0.01 grid, divided by 100, so uniform random noise scores near 100% on it. It shows only that a threshold can move. Findings 1 and 2 are the evidence; this one is the explanation. `tests/test_audit.py` pins this.

### 4. As a verifier, the judge held up

The last stage checks each drafted sentence against only the passages it cites. On the 500-question run, 612 of the 623 kept sentences were supported: 0.982 [0.972, 0.992], against a pre-registered bar of 0.90. That covers only kept sentences, across 365 of the 500 questions, so it is not a whole-pipeline accuracy.

The practical rule: use the judge's score to verify and to rank, and send low scores somewhere (a second search, a stronger model, a person) instead of discarding them.

## Check your own judge

`harness/audit.py` asks two separate questions about any judge's confidence scores. The first is whether a threshold can move: it reports distinct values, the share on the top three values, and gate mobility, with no labels needed. The second is whether moving it helps: it reports AUROC against your labels, with intervals that resample by question. Random noise passes the first and fails the second.

```bash
uv run python -m harness.audit my_scores.csv --label label --judge judge --cluster question
```

It has no network or model dependencies and 8 tests in `tests/test_audit.py`. `scripts/informativeness.py` is the worked example that produced findings 1 and 2.

## Open problems

The judged pipeline (D) lost to the no-judge baseline (B). These measurements and next steps are open for anyone to take further.

### The end-to-end result

The questions were drawn with seed 20260919, search results and page text were frozen on 2026-09-19, and answers were graded on 2026-09-20 by claude-sonnet-5 with the official SimpleQA grading prompt. Right scores +1, declined 0, wrong -1. D cost $0.53 for 13,974 judge requests.

| Version | Correct | Wrong | Declined | Score [95% CI] | Attempted |
|---|---|---|---|---|---|
| A, Gemma alone, no search | 6 | 153 | 341 | -0.294 [-0.340, -0.254] | 31.8% |
| B, top 8 pages, no judge | 407 | 37 | 56 | 0.740 [0.688, 0.792] | 88.8% |
| D, Jev makes every decision | 332 | 26 | 142 | 0.612 [0.562, 0.662] | 71.6% |

D cleared three of its four pre-registered bars (`harness/grading/prereg.py`): coverage 0.716 against at least 0.50, wrong-when-attempting 0.073 [0.047, 0.101] against at most 0.10, and citation support 0.982 against at least 0.90. It missed the main one, a gain over B of at least +0.15, which is the only bar that measures whether the judge helps. D minus B, paired, is -0.128 [-0.178, -0.078].

### Where the gap comes from

On the 353 questions both answered, D got 330 right and 23 wrong, and B got 331 right and 22 wrong. On D's 142 declines, B got 76 right, 15 wrong and declined 51. B was wrong on 16.5% (15/91) of those it answered, against 6.2% (22/353) where both answered, so D did steer away from harder questions. It gave up about five right answers for every wrong one it avoided.

| D's decline reason | Questions | B right | B wrong | B declined |
|---|---|---|---|---|
| Insufficient evidence | 60 | 11 | 9 | 40 |
| Final check removed every sentence | 44 | 38 | 1 | 5 |
| Answer judged off-target | 29 | 22 | 3 | 4 |

Another 2 came from the generator saying the evidence was insufficient, and 7 have no recorded reason. The insufficient-evidence declines are well aimed. The other two rows are where the right answers were lost.

### The snapshot prediction was not borne out

Before the run, the README predicted that a thin search snapshot would push D's declines up (commit 8507fc6; the text is now in [docs/DESIGN.md](docs/DESIGN.md)). Search did lose engines to rate limits and CAPTCHAs, returning 10.7 results per question against the pilot's 28.7. D's decline rate barely moved: 25% on the pilot (5/20), 28% at scale (142/500). B changed. It was wrong on 25% of its pilot attempts (4/16) and 8% at scale (37/444). The 20-question pilot overstated D's edge.

### Gate replay

`scripts/gate_replay.py` replays the 0.80 gate on the final check. Of 172 removed sentences, 94 could return by moving it; the rest were removed for their label. Moving the gate to 0.00 reopens at most 46 declined questions. With B's grades as a stand-in that scores about 0.688, and 0.704 if all 46 came out right.

### Worth trying

- Send low-confidence questions to a second search or a stronger model instead of declining them. Finding 1 says the score ranks well, which is what routing needs.
- Set how often the judge declines from the no-judge version's error rate instead of from fixed thresholds.
- Test on a harder dataset. The FreshQA loader is written.
- Grade the 94 sentences in `results/gate_replay_worklist.jsonl`.
- Get Gemma's confidence from token logprobs (nearly free) or the vote fraction over 10 samples. Neither has been tried, and `harness/audit.py` can measure both.

## Notes on the measurements

- The answer-in-passage label is approximate in both directions. It misses answers phrased differently from the gold string, and it can fire on a common gold word that appears for unrelated reasons.
- Finding 2 rests on 90 answer-bearing passages across 9 questions, 82 of them in 5 questions. A bootstrap over 12 clusters runs narrow, so treat its intervals as approximate. A jackknife gives about [+0.025, +0.220] for the difference.
- A grader bug was caught and fixed during development. An 8-token output cap cut off the grading model, which reasons for about 120 tokens before its verdict, so early replies came back empty and were scored unsupported. The fix (commit 8507fc6, `tests/test_claim_support.py`) landed at 16:34 UTC on 2026-09-20. The 500-question grading ran from 17:30 to 17:34 UTC and reports 0 unreadable replies.
- Citation support measures careful citing. On the pilot, re-grading moved it from 0.704 to 0.926 for Gemma judging itself and left Jev's at 0.909. That Gemma version answered fewer questions correctly.
- Finding 3 rebuilds from the published `corpus/public/statistics.json` (`scripts/calibration_figure.py`). Finding 4, the tables and the gate replay rebuild from `results/` (`scripts/gate_replay.py`; the tables are direct counts). Findings 1 and 2 come from `scripts/informativeness.py` over `corpus/raw/`, which stays local; the script and `results/informativeness.json` are published. Search depth comes from `snapshots/*/manifest.json`.

## What leaves your computer

Version D sends to TypeSafe's servers: the question, one passage at a time (up to about 1,800 characters each), the chosen evidence block, and each sentence of the draft answer. When memory is consulted, that includes your own stored text. It does not send your identity, whole pages, or anything about your browsing, though every request is tied to your TypeSafe API key and account. On the pilot that was about 50 requests per question. TypeSafe may keep what it receives, so treat anything sent to the judge as shared with them.

Versions A, B, C-laya, C-classical, and C-self send nothing anywhere. Gemma, Laya, MemPalace, and a SearXNG on your own machine is a complete setup. The only outside calls are the ones SearXNG makes to public search engines.

Grading the benchmark sends questions, reference answers, evidence, and answers to the grading model's provider. That is test equipment. It is not part of the product.

## Status

| Item | State |
|---|---|
| Pre-registered rules | [docs/JUDGMENT_SPEC.md](docs/JUDGMENT_SPEC.md), with thresholds in [judge/questions.v1.json](judge/questions.v1.json) |
| Versions A, B and D, 500 SimpleQA questions | run and graded, in `results/` |
| Gate replay | 94 sentences not yet graded |
| Calibration statistics | on Hugging Face; raw judge records stay local under TypeSafe's terms |
| Version C-self | run on the 20-question pilot |
| Versions C-laya, C-classical and E | not run |
| FreshQA track | loader done, not run |
| Memory experiment (second pass) | write-back runs; the second pass has not |

## Running it

```bash
# search engine
docker compose -f docker/compose.yml up -d searxng

# writer: llama-server or Unsloth Desktop on an OpenAI-compatible port
llama-server -hf unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL --spec-type draft-mtp -ngl 999 -fa on --port 8081

# harness
uv sync
cp .env.example .env      # then add TYPESAFE_API_KEY and the grader key
uv run pytest             # no network, no model

# freeze a snapshot, run the arms, grade
uv run python -m harness.retrieval.snapshot --dataset simpleqa --n 20 --seed 20260919 --id pilot-20260919
uv run python -m harness.run --arm A --dataset simpleqa --snapshot pilot-20260919
uv run python -m harness.run --arm B --dataset simpleqa --snapshot pilot-20260919
uv run python -m harness.run --arm D --dataset simpleqa --snapshot pilot-20260919 --limit 2   # about 130 requests, half a cent
uv run python -m harness.run --arm D --dataset simpleqa --snapshot pilot-20260919
uv run python -m harness.run --grade results/A_simpleqa_pilot-20260919.json results/B_simpleqa_pilot-20260919.json results/D_simpleqa_pilot-20260919.json
```

The grade step prints the score table and, for a judged version, the four bars labelled with the snapshot, so a pilot always reads as a pilot. To run the whole web track unattended (finish the freeze, re-search short questions when engines allow it, run A, B and D, grade):

```bash
uv run --with "system-one-adapter[openai]>=0.2.0" python scripts/run_web_track.py --snapshot simpleqa-500-20260919 --n 500 --cself-pilot
```


## More

- [docs/DESIGN.md](docs/DESIGN.md): the pipeline, each version, the pilot and the design sources.
- Aggregate statistics: <https://huggingface.co/datasets/clduab11/jev-calibration-statistics>

## License

MIT for this repository. Gemma 4 is Apache 2.0. Laya is Apache 2.0. MemPalace is MIT. Jev is a hosted service with its own terms. Benchmark datasets keep their own licenses and are downloaded at run time, never copied here.
