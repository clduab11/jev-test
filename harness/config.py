"""Paths, the pre-registered question file, and environment loading.

Thresholds and generator settings come from judge/questions.v1.json and from
nowhere else. Secrets and endpoints come from .env, which is git-ignored.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "judge" / "questions.v1.json"

_env_loaded = False


def load_env() -> None:
    """Load .env once. Values already in the environment win."""
    global _env_loaded
    if not _env_loaded:
        load_dotenv(ROOT / ".env", override=False)
        _env_loaded = True


def env(name: str, default: str | None = None) -> str | None:
    load_env()
    return os.getenv(name, default)


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def thresholds() -> dict:
    return dict(load_spec()["thresholds"])


def generator_settings() -> dict:
    return dict(load_spec()["generator"])


def spec_version() -> str:
    return str(load_spec()["spec_version"])
