"""A judge that answers from a function, so every stage can be proved without Jev."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from harness.judge.base import (
    Answer,
    ChoiceAnswer,
    JSONValue,
    NoulAnswer,
    Question,
    SystemOneResponse,
)


def noul(p: float) -> NoulAnswer:
    return NoulAnswer(noul=float(p))


def choice(name: str, confidence: float, options: Sequence[str] | None = None) -> ChoiceAnswer:
    """A choice with ``confidence`` on ``name`` and the rest spread over ``options``."""
    options = [o for o in (options or []) if o != name]
    rest = (1.0 - confidence) / len(options) if options else 0.0
    probabilities = {name: float(confidence), **{o: rest for o in options}}
    return ChoiceAnswer(choice=name, probabilities=probabilities, confidence=float(confidence))


class FakeJudge:
    """Answers come from ``answer_fn(state, questions)``; every call is recorded."""

    def __init__(
        self,
        answer_fn: Callable[[JSONValue, Mapping[str, Question]], Mapping[str, Answer]],
        model: str = "fake-judge-1.0",
        input_tokens: int = 100,
        output_tokens: int = 10,
    ) -> None:
        self.answer_fn = answer_fn
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[tuple[JSONValue, dict[str, Question]]] = []

    def system_one(self, state: JSONValue, questions: Mapping[str, Question]) -> SystemOneResponse:
        questions = dict(questions)
        self.calls.append((state, questions))
        answers = dict(self.answer_fn(state, questions))
        missing = set(questions) - set(answers)
        if missing:
            raise AssertionError(f"fake judge did not answer {sorted(missing)}")
        return SystemOneResponse(
            model=self.model,
            answers=answers,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached=False,
            raw={},
        )

    def system_one_many(self, items):
        return [self.system_one(state, questions) for state, questions in items]

    def states(self, kind: str) -> list[Any]:
        """States of one stage: s0 (has today), s1 (passage), s2 (evidence), s4 (claim), s4a (answer)."""
        keys = {"s0": "today", "s1": "passage", "s2": "evidence", "s4": "claim", "s4a": "answer"}
        return [s for s, _ in self.calls if isinstance(s, dict) and keys[kind] in s]
