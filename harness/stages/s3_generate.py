"""Stage S3: build the evidence prompt, generate, and post-process (spec section 6).

The system prompt and the abstain token come from ``generator`` in
judge/questions.v1.json and are identical across arms B, C, D and E. Only the
evidence differs.

Post-processing, in the spec's words:
  * the abstain token alone means ABSTAIN, detected by string, no judge
  * a sentence carrying one or more ``[id]`` tags is a claim
  * a sentence with no tag that contains a digit, a proper noun or a date is
    an uncited claim: kept in arm B, stripped in arms C and D
  * an ``[id]`` that is not in the evidence set is a fabricated citation: the
    sentence is stripped and the event is counted
"""

from __future__ import annotations

import re
from typing import Any

from harness.config import generator_settings
from harness.generate.openai_compat import generate

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")
_LEADING_CITES_RE = re.compile(r"^((?:\s*\[[A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*\]\s*)+)")
_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
)


def citation_ids(sentence: str) -> list[str]:
    """Ids cited in a sentence. Accepts [r01c0] and [r01c0, r02c1]; a bracket
    whose content is not a comma-separated list of ids is not a citation."""
    ids: list[str] = []
    for group in _BRACKET_RE.findall(sentence):
        tokens = [t.strip() for t in group.split(",")]
        if tokens and all(_ID_RE.match(t) for t in tokens):
            ids.extend(tokens)
    return ids


def strip_citations(sentence: str) -> str:
    return _BRACKET_RE.sub(lambda m: "" if citation_ids(m.group(0)) else m.group(0), sentence)


def system_prompt() -> str:
    return str(generator_settings()["system_prompt"])


def abstain_token() -> str:
    return str(generator_settings()["abstain_token"])


def build_user_message(
    query: str,
    evidence: list[dict[str, Any]],
    conflicts: list[dict[str, Any]] | None = None,
) -> str:
    """The user message exactly as the spec's section 6 lays it out."""
    lines = ["EVIDENCE:"]
    lines += [f"[{item['id']}] {item['text']}" for item in evidence]
    if conflicts:
        lines.append("CONFLICTS:")
        lines += [f"[{item['id']}] {item['text']}" for item in conflicts]
    lines += ["", f"QUESTION: {query}"]
    return "\n".join(lines)


def is_abstain(text: str) -> bool:
    """The abstain token, alone. Trailing punctuation or quotes are tolerated."""
    token = abstain_token()
    stripped = text.strip().strip("`\"'").strip()
    if stripped == token:
        return True
    return stripped.rstrip(".!:").strip() == token


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps citation tags with the sentence they follow."""
    pieces = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    out: list[str] = []
    for piece in pieces:
        match = _LEADING_CITES_RE.match(piece)
        if match and out:
            out[-1] = f"{out[-1]} {match.group(1).strip()}"
            piece = piece[match.end():].strip()
            if not piece:
                continue
        out.append(piece)
    return out


def has_fact_shape(sentence: str) -> bool:
    """Digit, date, or a proper noun anywhere but the first word."""
    body = strip_citations(sentence).strip()
    if re.search(r"\d", body):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", body)
    if any(w.lower() in _MONTHS for w in words):
        return True
    return any(w[0].isupper() for w in words[1:] if w.lower() not in {"i"})


def postprocess(answer_text: str, evidence_ids: set[str], keep_uncited: bool) -> dict[str, Any]:
    """Apply the section 6 rules to a raw generation."""
    if is_abstain(answer_text):
        return {
            "abstained": True,
            "answer": "",
            "claims": [],
            "citations": [],
            "uncited_claims": [],
            "stripped": [],
            "fabricated_citations": 0,
            "n_sentences": 0,
        }
    kept: list[str] = []
    claims: list[dict[str, Any]] = []
    uncited: list[str] = []
    stripped: list[dict[str, Any]] = []
    fabricated = 0
    sentences = split_sentences(answer_text)
    for sentence in sentences:
        ids = citation_ids(sentence)
        if ids:
            bad = [i for i in ids if i not in evidence_ids]
            if bad:
                fabricated += len(bad)
                stripped.append({"sentence": sentence, "reason": "fabricated_citation", "ids": bad})
                continue
            claims.append({"text": sentence, "citations": list(dict.fromkeys(ids))})
            kept.append(sentence)
        elif has_fact_shape(sentence):
            uncited.append(sentence)
            if keep_uncited:
                kept.append(sentence)
            else:
                stripped.append({"sentence": sentence, "reason": "uncited_claim"})
        else:
            kept.append(sentence)
    citations = sorted({c for claim in claims for c in claim["citations"]})
    return {
        "abstained": False,
        "answer": " ".join(kept).strip(),
        "claims": claims,
        "citations": citations,
        "uncited_claims": uncited,
        "stripped": stripped,
        "fabricated_citations": fabricated,
        "n_sentences": len(sentences),
    }


def generate_answer(
    query: str,
    evidence: list[dict[str, Any]],
    conflicts: list[dict[str, Any]] | None = None,
    seed: int | None = None,
    keep_uncited: bool = False,
    thinking: bool = False,
    max_tokens: int = 400,
) -> dict[str, Any]:
    """Prompt the generator with the evidence block and post-process its answer."""
    user = build_user_message(query, evidence, conflicts)
    generation = generate(system_prompt(), user, seed=seed, thinking=thinking, max_tokens=max_tokens)
    evidence_ids = {item["id"] for item in evidence} | {item["id"] for item in (conflicts or [])}
    record = postprocess(generation.text, evidence_ids, keep_uncited=keep_uncited)
    record.update(
        {
            "answer_raw": generation.text,
            "generator_model": generation.model,
            "seed": generation.seed,
            "thinking": generation.thinking,
            "latency_s": generation.latency_s,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "finish_reason": generation.finish_reason,
            "evidence_ids": sorted(evidence_ids),
        }
    )
    return record
