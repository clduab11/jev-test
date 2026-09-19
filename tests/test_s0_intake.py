"""Stage S0 policy against the pre-registered thresholds, with a fake judge."""

from __future__ import annotations

from harness.config import thresholds
from harness.stages import s0_intake
from tests.fakes import FakeJudge, choice, noul

T = thresholds()
SOURCES = ("web", "palace", "both")
RECENCY = ("any", "year", "month", "week", "day")
CATEGORIES = ("general", "news", "science", "it")


def answers(needs=0.9, source=("web", 0.9), recency=("any", 0.9), category=("general", 0.9)):
    return {
        "needs_search": noul(needs),
        "source": choice(source[0], source[1], SOURCES),
        "recency": choice(recency[0], recency[1], RECENCY),
        "category": choice(category[0], category[1], CATEGORIES),
    }


def test_questions_come_from_the_spec_file():
    qs = s0_intake.questions()
    assert set(qs) == {"needs_search", "source", "recency", "category"}
    assert qs["needs_search"]["type"] == "noul"
    assert set(qs["recency"]["criteria"]) == set(RECENCY)


def test_confident_answers_map_straight_through():
    d = s0_intake.decide(answers(recency=("month", 0.8), category=("it", 0.7)), palace_has_verified_hits=False)
    assert d["retrieve"] and d["retrieve_judged"]
    assert d["source"] == "web"
    assert d["time_range"] == "month"
    assert d["category"] == "it"


def test_low_confidence_widens_and_defaults():
    floor = T["choice_floor"]
    d = s0_intake.decide(
        answers(source=("web", floor - 0.01), recency=("day", floor - 0.01), category=("news", floor - 0.01)),
        palace_has_verified_hits=False,
    )
    assert d["source"] == "both", "uncertain source widens to both, never narrows"
    assert d["time_range"] is None
    assert d["category"] == "general"


def test_palace_alone_needs_verified_hits():
    d = s0_intake.decide(answers(source=("palace", 0.9)), palace_has_verified_hits=False)
    assert d["source"] == "both"
    d = s0_intake.decide(answers(source=("palace", 0.9)), palace_has_verified_hits=True)
    assert d["source"] == "palace"


def test_needs_search_threshold_and_forcing():
    below = T["needs_search_min"] - 0.01
    d = s0_intake.decide(answers(needs=below), palace_has_verified_hits=False)
    assert d["retrieve"] is False and d["retrieve_judged"] is False
    d = s0_intake.decide(answers(needs=below, source=("palace", 0.9)), True, force_search=True, force_web=True)
    assert d["retrieve"] is True and d["retrieve_judged"] is False
    assert d["source"] == "web" and d["source_judged"] == "palace"
    assert d["forced"] == {"search": True, "web": True}


def test_since_window_is_date_math_in_code():
    assert s0_intake.since_for(None, "2026-09-19") is None
    assert s0_intake.since_for("year", "2026-09-19") == "2025-09-19"
    assert s0_intake.since_for("week", "2026-09-19") == "2026-09-12"
    assert s0_intake.since_for("day", "2026-03-01") == "2026-02-28"


def test_run_sends_the_spec_state_and_logs_the_model():
    judge = FakeJudge(lambda state, qs: answers(recency=("year", 0.9)), model="fake-judge-9")
    out = s0_intake.run(judge, "Who won?", "2026-09-19", palace_has_verified_hits=True, force_web=True)
    assert judge.calls[0][0] == {"query": "Who won?", "today": "2026-09-19", "palace_has_verified_hits": True}
    assert set(judge.calls[0][1]) == set(s0_intake.questions())
    assert out["model"] == "fake-judge-9" and out["n_requests"] == 1
    assert out["since"] == "2025-09-19"
    assert out["answers"]["recency"]["choice"] == "year"
