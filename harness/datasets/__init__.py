"""Dataset loaders. Each module exposes subset(n, seed), by_id(ids) and source_info();
rows are {question_id, query, gold, metadata}. Loaders download at run time and
cache under data/, which is git-ignored: ship loaders, not copies (spec section 11).
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

KNOWN = ("simpleqa", "freshqa")


def module(dataset: str) -> ModuleType:
    if dataset not in KNOWN:
        raise ValueError(f"unknown dataset {dataset!r}; known: {KNOWN}")
    return import_module(f"harness.datasets.{dataset}")
