"""The Laya shim: answer translation, and a real HTTP round trip through HttpJudge
with a fake predict function. Laya itself is not needed."""

from __future__ import annotations

import threading

from harness.judge import laya_shim
from harness.judge.adapter import fallback_answers
from harness.judge.base import ChoiceAnswer, NoulAnswer
from harness.judge.http import HttpJudge

QUESTIONS = {
    "needs_search": {"type": "noul", "instructions": "Fact?", "criteria": {"true": "yes", "false": "no"}},
    "category": {"type": "choice", "instructions": "Which?", "criteria": {"general": "g", "news": "n", "it": "i"}},
}


def fake_predict(state, questions):
    return {
        "answers": {
            "needs_search": {"noul": 0.9},
            "category": {"choice": "news", "probabilities": {"general": 0.2, "news": 0.7, "it": 0.1}},
        }
    }


def test_translate_fills_in_confidence_and_tolerates_key_names():
    answers = laya_shim.translate({"query": "q"}, QUESTIONS, fake_predict)
    assert answers["needs_search"] == {"type": "noul", "noul": 0.9}
    assert answers["category"]["choice"] == "news" and answers["category"]["confidence"] == 0.7

    def other_names(state, questions):
        return {"needs_search": {"probability": 0.2}, "category": {"probs": {"general": 0.6, "news": 0.4}}}

    answers = laya_shim.translate({}, QUESTIONS, other_names)
    assert answers["needs_search"]["noul"] == 0.2
    assert answers["category"]["choice"] == "general" and answers["category"]["confidence"] == 0.6

    empty = laya_shim.translate({}, QUESTIONS, lambda s, q: {})
    assert empty["needs_search"]["noul"] == 0.5 and empty["category"]["choice"] == "general"


def test_http_round_trip_through_httpjudge(tmp_path):
    server = laya_shim.serve(fake_predict, port=0, model_name="laya-test")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        judge = HttpJudge(f"http://127.0.0.1:{port}", api_key="", model="laya-test", cache_dir=tmp_path)
        response = judge.system_one({"query": "What is the capital of France?"}, QUESTIONS)
        assert response.model == "laya-test" and response.cached is False
        assert response.answers["needs_search"] == NoulAnswer(noul=0.9)
        category = response.answers["category"]
        assert isinstance(category, ChoiceAnswer) and category.choice == "news"
        assert response.input_tokens > 0
        assert judge.system_one({"query": "What is the capital of France?"}, QUESTIONS).cached is True
        judge.close()
    finally:
        server.shutdown()
        server.server_close()


def test_adapter_fallback_answers_are_uncertain_and_well_formed():
    out = fallback_answers(QUESTIONS)
    assert out["needs_search"] == {"type": "noul", "noul": 0.5}
    assert out["category"]["choice"] == "general" and abs(sum(out["category"]["probabilities"].values()) - 1) < 1e-9
    assert abs(out["category"]["confidence"] - 1 / 3) < 1e-9
