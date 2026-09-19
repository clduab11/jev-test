"""The Judge interface.

Every arm sends byte-identical questions to a judge. Only the transport differs:
Jev over HTTPS, LitJev or a Laya shim over localhost, Gemma through
system-one-adapter, or classical models behind the same method signature.

The wire format is TypeSafe's ``/v1/systemone``:

    request  {"model": str, "state": JSON, "questions": {id: Question}}
    response {"model": str, "answers": {id: Answer}, "usage": {...}}

Question and Answer shapes are copied from docs.typesafe.ai/api (read 2026-09-19).
Question ids are for code; they are not sent to the model in any semantic sense,
so the full meaning must live in ``instructions`` and ``criteria``.

Rules this module enforces on behalf of docs/JUDGMENT_SPEC.md:
  * every call is cached by (model, canonical state, canonical questions)
  * the responding model id is recorded on every answer set
  * thresholds are never read here; they belong to the stage code
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Union

JSONValue = Union[str, int, float, bool, None, dict, list]


# --------------------------------------------------------------------------- #
# Questions (what we send)
# --------------------------------------------------------------------------- #

Question = Mapping[str, Any]
"""A dict with ``type`` in {"noul", "choice", "score"}, ``instructions``
(str or structured object), and ``criteria`` (shape depends on type).
Loaded verbatim from judge/questions.v1.json; never constructed by hand in
stage code."""


def load_questions(spec_path: Path, stage: str, group: str = "questions") -> dict[str, Question]:
    """Return the question map for one stage from questions.v1.json.

    ``stage`` is a key under ``stages`` (for example ``"S1_passage_gate"``).
    ``group`` selects ``questions``, ``claim_questions`` or ``answer_questions``.
    """
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return dict(spec["stages"][stage][group])


# --------------------------------------------------------------------------- #
# Answers (what we get back)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NoulAnswer:
    noul: float  # P(yes), 0..1. No confidence field by design.


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass(frozen=True)
class SystemOneResponse:
    model: str  # the versioned id that actually answered; log it
    answers: dict[str, Answer]
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    raw: dict = field(default_factory=dict, repr=False)


def parse_answers(payload: Mapping[str, Any]) -> dict[str, Answer]:
    """Convert a raw ``answers`` map into typed dataclasses."""
    out: dict[str, Answer] = {}
    for qid, a in payload.items():
        t = a.get("type")
        if t == "noul":
            out[qid] = NoulAnswer(noul=float(a["noul"]))
        elif t == "choice":
            out[qid] = ChoiceAnswer(
                choice=str(a["choice"]),
                probabilities={k: float(v) for k, v in a.get("probabilities", {}).items()},
                confidence=float(a.get("confidence", 0.0)),
            )
        elif t == "score":
            out[qid] = ScoreAnswer(
                score=float(a["score"]),
                probabilities={str(k): float(v) for k, v in a.get("probabilities", {}).items()},
                legend={str(k): str(v) for k, v in a.get("legend", {}).items()},
                confidence=float(a.get("confidence", 0.0)),
            )
        else:
            raise ValueError(f"unknown answer type {t!r} for question {qid!r}")
    return out


def to_plain(answers: Mapping[str, Answer]) -> dict[str, dict[str, Any]]:
    """Answers as JSON-ready dicts with a ``type`` field, for result files."""
    out: dict[str, dict[str, Any]] = {}
    for qid, answer in answers.items():
        if isinstance(answer, NoulAnswer):
            out[qid] = {"type": "noul", "noul": answer.noul}
        elif isinstance(answer, ChoiceAnswer):
            out[qid] = {
                "type": "choice",
                "choice": answer.choice,
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
            }
        else:
            out[qid] = {
                "type": "score",
                "score": answer.score,
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
                "legend": dict(answer.legend),
            }
    return out


def run_many(
    judge: Judge, items: Sequence[tuple[JSONValue, Mapping[str, Question]]]
) -> list[SystemOneResponse]:
    """Send several (state, questions) pairs, through the judge's pool when it has one."""
    many = getattr(judge, "system_one_many", None)
    if callable(many):
        return list(many(list(items)))
    return [judge.system_one(state, questions) for state, questions in items]


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #


def cache_key(model: str, state: JSONValue, questions: Mapping[str, Question]) -> str:
    """Stable hash of (model, state, questions) with sorted keys.

    Two calls that differ only in dict ordering must hit the same cache entry.
    """
    canonical = json.dumps(
        {"model": model, "state": state, "questions": questions},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# The protocol every judge implements
# --------------------------------------------------------------------------- #


class Judge(Protocol):
    """One method. Stage code depends on nothing else."""

    model: str

    def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
    ) -> SystemOneResponse: ...


# Implementations to write, in this order (see README "For developers"):
#   harness/judge/http.py      HttpJudge(base_url, api_key, model, path="/v1/systemone")
#                              with disk cache in json_cache/ and a 4-worker pool + backoff.
#                              Covers Jev (api.typesafe.ai), LitJev (localhost), the Laya shim,
#                              and OpenRouter Decisions (different path; pass path=...).
#   harness/judge/adapter.py   AdapterJudge wrapping system_one_adapter.SystemOneAdapterClient
#                              pointed at llama-server, llm_answer_mode="probabilities".
#   harness/judge/classical.py ClassicalJudge mapping each question id to a local model.
