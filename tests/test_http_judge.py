"""HttpJudge against a fake transport. No network, no sleeping."""

from __future__ import annotations

import json

import httpx
import pytest

from harness.judge.base import ChoiceAnswer, NoulAnswer, ScoreAnswer, cache_key
from harness.judge.http import HttpJudge, JudgeRequestError

MODEL = "jev-1.13.0"
STATE = {"query": "What is the capital of France?", "today": "2026-09-19"}
QUESTIONS = {
    "needs_search": {
        "type": "noul",
        "instructions": "Does answering `query` require specific facts?",
        "criteria": {"true": "A checkable fact.", "false": "No checkable fact."},
    },
    "category": {
        "type": "choice",
        "instructions": "Which search category best fits `query`?",
        "criteria": {"general": "Everyday facts.", "news": "Current events."},
    },
    "difficulty": {
        "type": "score",
        "instructions": "How hard is `query` to answer?",
        "criteria": ["Trivial", "Moderate", "Hard"],
    },
}
ANSWERS = {
    "needs_search": {"type": "noul", "noul": 0.97},
    "category": {
        "type": "choice",
        "choice": "general",
        "probabilities": {"general": 0.92, "news": 0.08},
        "confidence": 0.92,
    },
    "difficulty": {
        "type": "score",
        "score": 0.3,
        "probabilities": {"0": 0.75, "1": 0.2, "2": 0.05},
        "legend": {"0": "Trivial", "1": "Moderate", "2": "Hard"},
        "confidence": 0.75,
    },
}


def ok_response(model: str = MODEL, answers: dict | None = None) -> httpx.Response:
    payload = {
        "model": model,
        "answers": answers if answers is not None else ANSWERS,
        "usage": {"input_tokens": 312, "output_tokens": 48},
    }
    return httpx.Response(200, json=payload)


def make_judge(tmp_path, handler, **kwargs):
    """Return (judge, calls, sleeps) where calls collects every request seen."""
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    judge = HttpJudge(
        base_url="https://judge.test",
        api_key="secret",
        model=MODEL,
        cache_dir=tmp_path / "cache",
        transport=httpx.MockTransport(recording_handler),
        sleep=sleeps.append,
        **kwargs,
    )
    return judge, calls, sleeps


def test_cache_miss_then_hit(tmp_path):
    judge, calls, _ = make_judge(tmp_path, lambda req, n: ok_response())

    first = judge.system_one(STATE, QUESTIONS)
    assert first.cached is False
    assert first.model == MODEL
    assert first.input_tokens == 312 and first.output_tokens == 48
    assert len(calls) == 1

    cache_file = tmp_path / "cache" / f"{cache_key(MODEL, STATE, QUESTIONS)}.json"
    assert cache_file.exists()
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert stored["model"] == MODEL
    assert stored["request"] == {"model": MODEL, "state": STATE, "questions": QUESTIONS}
    assert stored["answers"] == ANSWERS

    second = judge.system_one(STATE, QUESTIONS)
    assert second.cached is True
    assert second.answers == first.answers
    assert second.model == MODEL
    assert len(calls) == 1, "a cache hit must not touch the network"


def test_cache_key_is_order_insensitive(tmp_path):
    judge, calls, _ = make_judge(tmp_path, lambda req, n: ok_response())
    judge.system_one({"b": 1, "a": 2}, QUESTIONS)
    reordered_questions = dict(reversed(list(QUESTIONS.items())))
    hit = judge.system_one({"a": 2, "b": 1}, reordered_questions)
    assert hit.cached is True
    assert len(calls) == 1


def test_request_shape_and_auth_header(tmp_path):
    judge, calls, _ = make_judge(tmp_path, lambda req, n: ok_response())
    judge.system_one(STATE, QUESTIONS)
    request = calls[0]
    assert request.method == "POST"
    assert str(request.url) == "https://judge.test/v1/systemone"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body == {"model": MODEL, "state": STATE, "questions": QUESTIONS}


def test_parses_noul_choice_and_score(tmp_path):
    judge, _, _ = make_judge(tmp_path, lambda req, n: ok_response())
    answers = judge.system_one(STATE, QUESTIONS).answers

    assert answers["needs_search"] == NoulAnswer(noul=0.97)

    category = answers["category"]
    assert isinstance(category, ChoiceAnswer)
    assert category.choice == "general"
    assert category.probabilities == {"general": 0.92, "news": 0.08}
    assert category.confidence == 0.92

    difficulty = answers["difficulty"]
    assert isinstance(difficulty, ScoreAnswer)
    assert difficulty.score == 0.3
    assert difficulty.probabilities == {"0": 0.75, "1": 0.2, "2": 0.05}
    assert difficulty.legend == {"0": "Trivial", "1": "Moderate", "2": "Hard"}
    assert difficulty.confidence == 0.75


def test_429_then_success_honours_retry_after(tmp_path):
    def handler(request, n):
        if n == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "slow down"})
        return ok_response()

    judge, calls, sleeps = make_judge(tmp_path, handler)
    result = judge.system_one(STATE, QUESTIONS)
    assert result.cached is False
    assert result.answers["needs_search"] == NoulAnswer(noul=0.97)
    assert len(calls) == 2
    assert len(sleeps) == 1 and sleeps[0] >= 2.0


def test_529_and_5xx_are_retried_with_growing_backoff(tmp_path):
    statuses = iter([529, 503])

    def handler(request, n):
        status = next(statuses, None)
        return httpx.Response(status, text="overloaded") if status else ok_response()

    judge, calls, sleeps = make_judge(tmp_path, handler)
    judge.system_one(STATE, QUESTIONS)
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert 0 < sleeps[0] <= 0.5 and 0 < sleeps[1] <= 1.0


def test_retries_run_out(tmp_path):
    judge, calls, sleeps = make_judge(
        tmp_path, lambda req, n: httpx.Response(429, text="still busy"), max_retries=2
    )
    with pytest.raises(JudgeRequestError) as excinfo:
        judge.system_one(STATE, QUESTIONS)
    assert excinfo.value.status_code == 429
    assert len(calls) == 3 and len(sleeps) == 2
    assert not list((tmp_path / "cache").glob("*.json")), "failures are never cached"


def test_other_4xx_raises_without_retry(tmp_path):
    judge, calls, sleeps = make_judge(
        tmp_path,
        lambda req, n: httpx.Response(422, json={"detail": "questions.category.criteria missing"}),
    )
    with pytest.raises(JudgeRequestError) as excinfo:
        judge.system_one(STATE, QUESTIONS)
    assert excinfo.value.status_code == 422
    assert "criteria missing" in str(excinfo.value)
    assert len(calls) == 1 and sleeps == []


def test_transport_error_is_retried(tmp_path):
    def handler(request, n):
        if n == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return ok_response()

    judge, calls, sleeps = make_judge(tmp_path, handler)
    judge.system_one(STATE, QUESTIONS)
    assert len(calls) == 2 and len(sleeps) == 1


def test_batch_preserves_order_and_uses_cache(tmp_path):
    def handler(request, n):
        query = json.loads(request.content)["state"]["query"]
        answers = {"needs_search": {"type": "noul", "noul": float(query[-1]) / 10}}
        return ok_response(answers=answers)

    judge, calls, _ = make_judge(tmp_path, handler, max_workers=4)
    items = [({"query": f"q{i}"}, {"needs_search": QUESTIONS["needs_search"]}) for i in range(8)]
    results = judge.system_one_many(items)
    assert [r.answers["needs_search"].noul for r in results] == [i / 10 for i in range(8)]
    assert all(r.cached is False for r in results)
    assert len(calls) == 8

    again = judge.system_one_many(items)
    assert all(r.cached is True for r in again)
    assert len(calls) == 8


def test_responding_model_is_recorded_not_assumed(tmp_path):
    judge, _, _ = make_judge(tmp_path, lambda req, n: ok_response(model="jev-1.13.0-rc2"))
    assert judge.system_one(STATE, QUESTIONS).model == "jev-1.13.0-rc2"


def test_refuses_latest_alias(tmp_path):
    with pytest.raises(ValueError):
        HttpJudge("https://judge.test", "k", "jev-latest", cache_dir=tmp_path)
    HttpJudge("https://judge.test", "k", "jev-latest", cache_dir=tmp_path, allow_alias=True)


def test_custom_path_and_headers_for_other_providers(tmp_path):
    judge, calls, _ = make_judge(
        tmp_path,
        lambda req, n: ok_response(),
        path="/api/alpha/decisions",
        extra_headers={"HTTP-Referer": "https://github.com/clduab11/jev-test"},
    )
    judge.system_one(STATE, QUESTIONS)
    assert str(calls[0].url) == "https://judge.test/api/alpha/decisions"
    assert calls[0].headers["HTTP-Referer"] == "https://github.com/clduab11/jev-test"
