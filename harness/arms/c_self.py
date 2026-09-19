"""Arm C-self: the full arm D pipeline with the generator as its own judge.

Every stage, threshold and prompt is arm D's; only the judge object differs
(``harness.judge.adapter.AdapterJudge`` pointed at the same local Gemma that
writes the answers). If this arm comes close to arm D, the win comes from
decomposing decisions into typed questions; if it falls well short, the
calibrated judge is the ingredient (spec section 10).
"""

from __future__ import annotations

from typing import Any

from harness.arms import d_jev

ARM = "C-self"


def answer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    record = d_jev.answer(*args, **kwargs)
    record["arm"] = ARM
    return record
