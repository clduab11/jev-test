"""SimpleQA loader. Downloads at run time, caches under data/, never committed.

Primary source: the simple-evals CSV published by OpenAI (MIT licence).
Fallback: a Hugging Face mirror. Whichever one served the file is written to
data/simpleqa/source.json next to the CSV, with its sha256.

Question ids are ``simpleqa-<row>`` where row is the 0-based line in the
original CSV, so a subset is reproducible from the seed alone.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from harness.config import ROOT

PRIMARY_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
MIRROR_URL = "https://huggingface.co/datasets/basicv8vc/SimpleQA/resolve/main/simple_qa_test_set.csv"
DATA_DIR = ROOT / "data" / "simpleqa"
CSV_PATH = DATA_DIR / "simple_qa_test_set.csv"
SOURCE_PATH = DATA_DIR / "source.json"
DEFAULT_SEED = 20260919


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(force: bool = False, timeout: float = 120.0) -> Path:
    """Fetch the CSV once. Tries the primary URL, then the mirror."""
    if CSV_PATH.exists() and not force:
        return CSV_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in (PRIMARY_URL, MIRROR_URL):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                body = response.content
            if b"problem" not in body[:200]:
                raise ValueError("response does not look like the SimpleQA CSV")
            CSV_PATH.write_bytes(body)
            SOURCE_PATH.write_text(
                json.dumps(
                    {
                        "url": url,
                        "sha256": _sha256(CSV_PATH),
                        "bytes": len(body),
                        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            return CSV_PATH
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("could not download SimpleQA: " + "; ".join(errors))


def source_info() -> dict[str, Any]:
    download()
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def load_all() -> list[dict[str, Any]]:
    """Every row as {question_id, query, gold, metadata}."""
    path = download()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            try:
                metadata = ast.literal_eval(record.get("metadata") or "{}")
            except (ValueError, SyntaxError):
                metadata = {"raw": record.get("metadata")}
            rows.append(
                {
                    "question_id": f"simpleqa-{index:04d}",
                    "query": (record.get("problem") or "").strip(),
                    "gold": (record.get("answer") or "").strip(),
                    "metadata": metadata,
                }
            )
    return rows


def subset(n: int = 20, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """A seeded random subset, returned in original CSV order."""
    rows = load_all()
    picked = random.Random(seed).sample(range(len(rows)), n)
    return [rows[i] for i in sorted(picked)]


def by_id(question_ids: list[str]) -> dict[str, dict[str, Any]]:
    wanted = set(question_ids)
    return {row["question_id"]: row for row in load_all() if row["question_id"] in wanted}
