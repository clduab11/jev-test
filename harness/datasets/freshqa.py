"""FreshQA loader. Downloads the current sheet at run time, caches under data/, never committed.

FreshQA (Vu et al., 2023; github.com/freshllms/freshqa, Apache 2.0) lives in a
Google Sheet that its authors refresh weekly, so the answers move. Every
download records the date and the file hash in data/freshqa/source.json, and
a snapshot manifest copies that record, because a FreshQA number is only
meaningful next to the date its answers were read.

The CSV export has two preamble rows before the header:
``id, split, question, effective_year, next_review, false_premise, num_hops,
fact_type, source, answer_0 .. answer_9, note``. Read on 2026-09-19 it held
600 rows: 500 TEST and 100 DEV; fact_type never-changing 225, slow-changing
220, fast-changing 155; false_premise TRUE on 149.

Question ids are ``freshqa-<id>`` from the sheet's own id column. The web
track asks for 200 fast-changing items; the fast-changing slice of the TEST
split is smaller than that, so ``subset`` returns the whole slice and the
snapshot manifest records the real count.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from harness.config import ROOT

SHEET_ID = "1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM"
EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
REPO_URL = "https://github.com/freshllms/freshqa"
DATA_DIR = ROOT / "data" / "freshqa"
CSV_PATH = DATA_DIR / "freshqa.csv"
SOURCE_PATH = DATA_DIR / "source.json"
DEFAULT_SEED = 20260919
FACT_TYPES = ("never-changing", "slow-changing", "fast-changing")
ANSWER_COLUMNS = tuple(f"answer_{i}" for i in range(10))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(force: bool = False, timeout: float = 120.0) -> Path:
    """Fetch the sheet export once and record where and when it came from."""
    if CSV_PATH.exists() and not force:
        return CSV_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(EXPORT_URL)
        response.raise_for_status()
        body = response.content
    if b"fact_type" not in body[:4000]:
        raise RuntimeError("the FreshQA export does not look like the question sheet")
    CSV_PATH.write_bytes(body)
    SOURCE_PATH.write_text(
        json.dumps(
            {
                "url": EXPORT_URL,
                "repo": REPO_URL,
                "sha256": _sha256(body),
                "bytes": len(body),
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return CSV_PATH


def source_info() -> dict[str, Any]:
    download()
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def parse_rows(text: str) -> list[dict[str, Any]]:
    """Rows from the export text: skips the preamble, keeps the ten answer columns."""
    raw = list(csv.reader(io.StringIO(text)))
    header_index = next(
        (i for i, row in enumerate(raw) if "question" in row and "fact_type" in row), None
    )
    if header_index is None:
        raise ValueError("no header row with question and fact_type columns")
    header = raw[header_index]
    rows: list[dict[str, Any]] = []
    for values in raw[header_index + 1 :]:
        if not any(v.strip() for v in values):
            continue
        record = dict(zip(header, values, strict=False))
        golds = [record.get(col, "").strip() for col in ANSWER_COLUMNS]
        golds = [g for g in golds if g]
        rows.append(
            {
                "question_id": f"freshqa-{record.get('id', '').strip()}",
                "query": (record.get("question") or "").strip(),
                "gold": golds[0] if golds else "",
                "golds": golds,
                "metadata": {
                    "split": (record.get("split") or "").strip(),
                    "effective_year": (record.get("effective_year") or "").strip(),
                    "next_review": (record.get("next_review") or "").strip(),
                    "false_premise": (record.get("false_premise") or "").strip().upper() == "TRUE",
                    "num_hops": (record.get("num_hops") or "").strip(),
                    "fact_type": (record.get("fact_type") or "").strip(),
                    "source": (record.get("source") or "").strip(),
                },
            }
        )
    return rows


def load_all() -> list[dict[str, Any]]:
    return parse_rows(download().read_text(encoding="utf-8"))


def select(
    rows: list[dict[str, Any]],
    n: int | None = 200,
    seed: int = DEFAULT_SEED,
    split: str | None = "TEST",
    fact_types: tuple[str, ...] | None = ("fast-changing",),
    include_false_premise: bool = True,
) -> list[dict[str, Any]]:
    """The slice the web track wants, in sheet order; a seeded sample when it is larger than n."""
    picked = [
        r
        for r in rows
        if (split is None or r["metadata"]["split"] == split)
        and (fact_types is None or r["metadata"]["fact_type"] in fact_types)
        and (include_false_premise or not r["metadata"]["false_premise"])
    ]
    if n is not None and len(picked) > n:
        chosen = sorted(random.Random(seed).sample(range(len(picked)), n))
        picked = [picked[i] for i in chosen]
    return picked


def subset(n: int = 200, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Fast-changing TEST questions, all of them when the slice is smaller than n."""
    return select(load_all(), n=n, seed=seed)


def by_id(question_ids: list[str]) -> dict[str, dict[str, Any]]:
    wanted = set(question_ids)
    return {row["question_id"]: row for row in load_all() if row["question_id"] in wanted}
