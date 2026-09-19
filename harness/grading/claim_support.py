"""Citation support rate: the fourth pre-registered bar (spec section 11).

For every claim S4 kept, a frontier model that plays no role in the pipeline
reads the claim and only the passages it cites and says SUPPORTED or
UNSUPPORTED. The rate is supported claims over kept claims, with a bootstrap
95 percent interval over questions.

The prompt is committed at harness/grading/prompts/claim_support_grader.txt.
The grader comes from the same LiteLLM gateway as the SimpleQA grader and
every call is cached under json_cache/grading/ keyed by (model, full prompt).
Jev never grades Jev (rule 9): this module never imports the judge.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from harness.config import ROOT, env, load_env
from harness.grading.crag_score import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from harness.grading.simpleqa import _client
from harness.stages.s3_generate import plain_sentence

PROMPT_PATH = ROOT / "harness" / "grading" / "prompts" / "claim_support_grader.txt"
CACHE_DIR = ROOT / "json_cache" / "grading"
LABELS = ("supported", "unsupported")


def grader_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(question: str, claim: str, evidence: list[dict[str, Any]]) -> str:
    passages = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence) or "(none)"
    return grader_template().format(question=question, claim=plain_sentence(claim), evidence=passages)


def parse_label(text: str) -> str:
    """UNSUPPORTED contains SUPPORTED, so look for the negative first. Default unsupported."""
    upper = (text or "").upper()
    if "UNSUPPORTED" in upper or "NOT SUPPORTED" in upper:
        return "unsupported"
    if "SUPPORTED" in upper:
        return "supported"
    return "unsupported"


def grade_claim(
    question: str,
    claim: str,
    evidence: list[dict[str, Any]],
    model: str | None = None,
    cache_dir: Path = CACHE_DIR,
    client=None,
) -> dict[str, Any]:
    load_env()
    model = model or env("GRADER_MODEL", "") or ""
    prompt = build_prompt(question, claim, evidence)
    key = hashlib.sha256(f"{model}\nclaim_support\n{prompt}".encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.exists():
        entry = json.loads(path.read_text(encoding="utf-8"))
        return {"label": entry["label"], "model": entry["model"], "cached": True, "reply": entry["reply"]}
    client = client or _client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=8,
    )
    reply = (response.choices[0].message.content or "").strip()
    entry = {
        "model": str(response.model or model),
        "kind": "claim_support",
        "prompt": prompt,
        "reply": reply,
        "label": parse_label(reply),
        "usage": {
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
        },
    }
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"label": entry["label"], "model": entry["model"], "cached": False, "reply": reply}


def claim_evidence(record: dict[str, Any], claim: dict[str, Any]) -> list[dict[str, Any]]:
    texts = record.get("evidence_texts") or {}
    return [{"id": c, "text": texts[c]} for c in claim.get("citations", []) if c in texts]


def grade_records(
    records: list[dict[str, Any]], model: str | None = None, workers: int = 4
) -> list[list[dict[str, Any]]]:
    """One label per kept claim per record. Abstentions and claim-less records give []."""
    client = _client()
    jobs: list[tuple[int, str, str, list[dict[str, Any]]]] = []
    for index, record in enumerate(records):
        if record.get("abstained"):
            continue
        for claim in record.get("claims") or []:
            if "support" not in claim:  # arms without S4 have nothing to check
                continue
            jobs.append((index, record["query"], claim["text"], claim_evidence(record, claim)))

    def one(job):
        index, question, claim, evidence = job
        return index, grade_claim(question, claim, evidence, model=model, client=client)

    out: list[list[dict[str, Any]]] = [[] for _ in records]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, grade in pool.map(one, jobs):
            out[index].append(grade)
    return out


def support_rate(
    per_record: list[list[str]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Pooled supported / kept, with a bootstrap interval over questions."""
    kept = sum(len(labels) for labels in per_record)
    supported = sum(labels.count("supported") for labels in per_record)
    out: dict[str, Any] = {
        "kept_claims": kept,
        "supported": supported,
        "rate": supported / kept if kept else math.nan,
        "n_questions": len(per_record),
        "n_questions_with_claims": sum(1 for labels in per_record if labels),
    }
    if not per_record or kept == 0:
        out["rate_ci"] = [math.nan, math.nan]
        return out
    rng = np.random.default_rng(seed)
    counts = np.array([[labels.count("supported"), len(labels)] for labels in per_record])
    draws = rng.integers(0, len(per_record), size=(resamples, len(per_record)))
    sampled = counts[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(sampled[:, 1] > 0, sampled[:, 0] / np.maximum(sampled[:, 1], 1), np.nan)
    finite = rates[~np.isnan(rates)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if finite.size else (math.nan, math.nan)
    out["rate_ci"] = [round(float(lo), 4), round(float(hi), 4)]
    out["bootstrap"] = {"resamples": resamples, "seed": seed}
    return out
