"""Arm A: the generator alone. Question in, answer out, no evidence block.

This is the baseline hallucination measurement. There is no evidence to cite,
so the spec's evidence system prompt does not apply; the prompt below keeps
only the length rule so answers stay comparable in shape. The question is sent
bare: a QUESTION: prefix made the model talk about missing context.
"""

from __future__ import annotations

from typing import Any

from harness.generate.openai_compat import generate

ARM = "A"
SYSTEM_PROMPT = "Answer the question in at most five sentences."


def answer(query: str, seed: int | None = None, thinking: bool = False) -> dict[str, Any]:
    generation = generate(SYSTEM_PROMPT, query, seed=seed, thinking=thinking)
    return {
        "arm": ARM,
        "answer": generation.text.strip(),
        "answer_raw": generation.text,
        "abstained": False,
        "claims": [],
        "citations": [],
        "fabricated_citations": 0,
        "evidence_ids": [],
        "latency_s": generation.latency_s,
        "generator_model": generation.model,
        "seed": generation.seed,
        "thinking": generation.thinking,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "finish_reason": generation.finish_reason,
    }
