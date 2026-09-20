"""Non-ASCII question text must survive the logging path on any locale.

The 500-question arm A run died twenty seconds in on 2026-09-19 because this
machine encodes redirected stdout as cp1252 and question simpleqa-0110 contains
a Croatian letter. These tests spawn real subprocesses with the locale codec
forced, because the failure only appears when stdout is redirected and the
parent process cannot see it.
"""

from __future__ import annotations

import os
import subprocess
import sys

from harness.config import ROOT

CRASHER = "Kristić — ć, ž, 中文, \U0001f600"


def _run(code: str, utf8_env: bool) -> subprocess.CompletedProcess[bytes]:
    """Run ``code`` with stdout redirected to a pipe and the locale codec forced."""
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "utf-8" if utf8_env else "cp1252"
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, env=env, cwd=ROOT, check=False
    )


def test_locale_codec_really_does_break_without_the_fix():
    """The guard rail only means something if the failure is real here."""
    result = _run(f"print({CRASHER!r})", utf8_env=False)
    assert result.returncode != 0
    assert b"UnicodeEncodeError" in result.stderr


def test_configure_stdout_lets_non_ascii_through():
    code = f"from harness.config import configure_stdout\nconfigure_stdout()\nprint({CRASHER!r})"
    result = _run(code, utf8_env=False)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "Kristi" in result.stdout.decode("utf-8", "replace")


def test_orchestrator_child_env_forces_utf8():
    from scripts.run_web_track import child_env

    env = child_env()
    assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUTF8"] == "1"


def test_the_question_that_broke_the_run_is_still_in_the_snapshot():
    """A regression only stays fixed while the case that caused it is covered."""
    import json

    manifest = ROOT / "snapshots" / "simpleqa-500-20260919" / "manifest.json"
    if not manifest.exists():  # the snapshot is git-ignored; skip on a fresh clone
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entry = next((q for q in data["questions"] if q["question_id"] == "simpleqa-0110"), None)
    assert entry is not None
    assert any(ord(c) > 127 for c in entry["query"]), "expected non-ASCII in simpleqa-0110"
