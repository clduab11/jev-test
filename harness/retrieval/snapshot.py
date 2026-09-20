"""Freeze a retrieval snapshot once, replay it offline for every arm.

    uv run python -m harness.retrieval.snapshot --dataset simpleqa --n 20 --seed 20260919 --id pilot-20260919
    uv run python -m harness.retrieval.snapshot --id pilot-20260919 --replay-check

``freeze`` runs one SearXNG search per question, fetches every result page,
chunks it, files every chunk into the evidence wing of the snapshot's own
palace at snapshots/<id>/palace, and writes snapshots/<id>/manifest.json.
The manifest carries the raw SearXNG response, the ranked results, every
fetch outcome, and for every chunk its citation id, drawer id, sha256 and
token count. The palace directory is git-ignored; the manifest is committed.

``replay`` rebuilds the candidate list for one question from the manifest
and the palace alone. No network.

Two choices the spec leaves open are fixed here and recorded in the manifest:
  * the search is frozen with category ``general`` and no time range, because
    a snapshot is taken before any judge decision exists and must serve arms
    that make different S0 choices; S0 answers are still logged per arm
  * every result page is fetched, not only the first ``max_pages``, so that
    arm D can pick any surviving snippet offline and arm B can take its top 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from harness.config import ROOT, configure_stdout, spec_version, thresholds
from harness.retrieval import fetch as fetchmod
from harness.retrieval import searxng
from harness.retrieval.palace import Palace, file_chunk, get_drawers, open_palace

SNAPSHOT_ROOT = ROOT / "snapshots"


def snapshot_dir(snapshot_id: str) -> Path:
    return SNAPSHOT_ROOT / snapshot_id


def manifest_path(snapshot_id: str) -> Path:
    return snapshot_dir(snapshot_id) / "manifest.json"


def palace_path(snapshot_id: str) -> Path:
    return snapshot_dir(snapshot_id) / "palace"


def load_manifest(snapshot_id: str) -> dict[str, Any]:
    path = manifest_path(snapshot_id)
    if not path.exists():
        raise FileNotFoundError(f"no manifest for snapshot {snapshot_id!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(snapshot_id: str, manifest: dict[str, Any]) -> None:
    path = manifest_path(snapshot_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows: a reader has the file open; wait, then try again
            time.sleep(0.25 * (attempt + 1))
    tmp.replace(path)


def cite_id(rank: int, chunk_index: int) -> str:
    return f"r{rank:02d}c{chunk_index}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_questions(dataset: str, n: int, seed: int, question_ids: list[str] | None) -> list[dict[str, Any]]:
    from harness import datasets

    loader = datasets.module(dataset)
    if question_ids:
        rows = loader.by_id(question_ids)
        return [rows[qid] for qid in question_ids]
    return loader.subset(n=n, seed=seed)


def _fetch_all(urls: list[str], workers: int, timeout: float) -> list[fetchmod.FetchResult]:
    with httpx.Client(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": fetchmod.USER_AGENT}
    ) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda u: fetchmod.fetch_and_chunk(u, timeout=timeout, client=client), urls))


def freeze_question(
    palace: Palace,
    snapshot_id: str,
    dataset: str,
    row: dict[str, Any],
    category: str,
    time_range: str | None,
    workers: int,
    fetch_timeout: float,
) -> dict[str, Any]:
    """Search, fetch, chunk and file one question. Returns its manifest entry."""
    t = thresholds()
    searched_at = fetchmod.now_iso()
    raw = searxng.search_raw(row["query"], category=category, time_range=time_range)
    results = searxng.normalize(raw, t["max_snippets"])
    fetched = _fetch_all([r["url"] for r in results], workers=workers, timeout=fetch_timeout)

    pages: list[dict[str, Any]] = []
    filed_new = 0
    filed_dup = 0
    for result, page in zip(results, fetched, strict=True):
        chunks_out: list[dict[str, Any]] = []
        for chunk in page.chunks:
            drawer_id, created = file_chunk(
                palace,
                text=chunk["text"],
                url=result["url"],
                chunk_index=chunk["chunk_index"],
                query_id=row["question_id"],
                dataset=dataset,
                engine_rank=result["rank"],
                engines=result["engines"],
                fetched_at=page.fetched_at,
                published_date=result.get("published_date"),
                title=result.get("title"),
            )
            filed_new += int(created)
            filed_dup += int(not created)
            chunks_out.append(
                {
                    "cite_id": cite_id(result["rank"], chunk["chunk_index"]),
                    "drawer_id": drawer_id,
                    "chunk_index": chunk["chunk_index"],
                    "n_tokens": chunk["n_tokens"],
                    "n_chars": len(chunk["text"]),
                    "sha256": _sha256(chunk["text"]),
                    "created": created,
                }
            )
        pages.append(
            {
                "rank": result["rank"],
                "url": result["url"],
                "final_url": page.final_url,
                "status": page.status,
                "content_type": page.content_type,
                "error": page.error,
                "n_bytes": page.n_bytes,
                "text_chars": len(page.text),
                "fetched_at": page.fetched_at,
                "elapsed_s": page.elapsed_s,
                "chunks": chunks_out,
            }
        )
    return {
        "question_id": row["question_id"],
        "query": row["query"],
        "category": category,
        "time_range": time_range,
        "searched_at": searched_at,
        "filed_at": fetchmod.now_iso(),
        "n_results": len(results),
        "n_fetched_ok": sum(1 for p in fetched if p.ok),
        "n_chunks": sum(len(p["chunks"]) for p in pages),
        "n_chunks_new": filed_new,
        "n_chunks_dedup": filed_dup,
        "searxng_raw": raw,
        "results": results,
        "pages": pages,
    }


def freeze(
    dataset: str,
    question_ids: list[str] | None = None,
    snapshot_id: str = "",
    n: int = 20,
    seed: int = 20260919,
    category: str = "general",
    time_range: str | None = None,
    workers: int = 8,
    fetch_timeout: float = 10.0,
    sleep_s: float = 0.0,
    redo_below: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Freeze search results and pages for a question set. Resumes if interrupted.

    ``sleep_s`` waits between searches so public engines are not suspended for
    too many requests. ``redo_below`` re-searches every frozen question that
    got fewer than that many results and replaces its manifest entry; chunks
    already in the palace are de-duplicated by (url, chunk_index).
    """
    if not snapshot_id:
        raise ValueError("snapshot_id is required")
    rows = _load_questions(dataset, n, seed, question_ids)
    path = manifest_path(snapshot_id)
    if path.exists():
        manifest = load_manifest(snapshot_id)
    else:
        from harness import datasets

        manifest = {
            "snapshot_id": snapshot_id,
            "dataset": dataset,
            "spec_version": spec_version(),
            "seed": seed,
            "n": n,
            "question_ids": [r["question_id"] for r in rows],
            "dataset_source": datasets.module(dataset).source_info(),
            "searxng_url": searxng.base_url(),
            "category": category,
            "time_range": time_range,
            "thresholds": {
                k: thresholds()[k]
                for k in ("max_snippets", "chunk_tokens", "chunk_overlap", "max_chunks_per_page", "passage_max_chars")
            },
            "tokenizer": "tiktoken cl100k_base",
            "palace_path": str(palace_path(snapshot_id).relative_to(ROOT)),
            "created_at": fetchmod.now_iso(),
            "questions": [],
        }
    existing = {q["question_id"]: q for q in manifest["questions"]}
    done = {qid for qid, q in existing.items() if redo_below is None or q["n_results"] >= redo_below}
    palace = open_palace(palace_path(snapshot_id))
    searched_any = False
    for index, row in enumerate(rows, 1):
        if row["question_id"] in done:
            continue
        if sleep_s and searched_any:
            time.sleep(sleep_s)
        searched_any = True
        started = time.perf_counter()
        entry = freeze_question(palace, snapshot_id, dataset, row, category, time_range, workers, fetch_timeout)
        if row["question_id"] in existing:
            manifest["questions"] = [
                entry if q["question_id"] == row["question_id"] else q for q in manifest["questions"]
            ]
        else:
            manifest["questions"].append(entry)
        manifest["updated_at"] = fetchmod.now_iso()
        manifest["palace_drawers"] = palace.count()
        _write_manifest(snapshot_id, manifest)
        engines = sorted({e for r in entry["results"] for e in r["engines"]})
        suspended = [u[0] for u in (entry["searxng_raw"].get("unresponsive_engines") or []) if u]
        log(
            f"[{index}/{len(rows)}] {row['question_id']}: {entry['n_results']} results, "
            f"{entry['n_fetched_ok']} pages fetched, {entry['n_chunks']} chunks "
            f"({entry['n_chunks_new']} new) in {time.perf_counter() - started:.1f}s; "
            f"engines {','.join(engines) or 'none'}; suspended {','.join(sorted(set(suspended))) or 'none'}"
        )
    return manifest


def replay(snapshot_id: str, question_id: str, palace: Palace | None = None) -> dict[str, Any]:
    """Candidates for one question from the manifest and the palace. No network."""
    manifest = load_manifest(snapshot_id)
    entry = next((q for q in manifest["questions"] if q["question_id"] == question_id), None)
    if entry is None:
        raise KeyError(f"{question_id!r} is not in snapshot {snapshot_id!r}")
    palace = palace or open_palace(palace_path(snapshot_id), read_only=True)
    wanted = [c["drawer_id"] for p in entry["pages"] for c in p["chunks"]]
    drawers = get_drawers(palace, wanted)
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for result, page in zip(entry["results"], entry["pages"], strict=True):
        chunks: list[dict[str, Any]] = []
        for chunk in page["chunks"]:
            drawer = drawers.get(chunk["drawer_id"])
            if drawer is None or _sha256(drawer["text"]) != chunk["sha256"]:
                missing.append(chunk["drawer_id"])
                continue
            chunks.append(
                {
                    "cite_id": chunk["cite_id"],
                    "drawer_id": chunk["drawer_id"],
                    "chunk_index": chunk["chunk_index"],
                    "n_tokens": chunk["n_tokens"],
                    "text": drawer["text"],
                }
            )
        results.append(
            {**result, "fetched": page["error"] is None, "fetched_at": page.get("fetched_at"), "chunks": chunks}
        )
    if missing:
        raise RuntimeError(
            f"{len(missing)} chunk(s) of {question_id} are missing from the palace or changed: {missing[:3]}"
        )
    return {
        "snapshot_id": snapshot_id,
        "question_id": question_id,
        "query": entry["query"],
        "category": entry["category"],
        "time_range": entry["time_range"],
        "results": results,
    }


class _NoNetwork:
    """Context manager that makes any socket connection raise."""

    def __enter__(self):
        self._orig = socket.socket.connect

        def refuse(sock, address):
            raise RuntimeError(f"network disabled during replay check; tried {address!r}")

        socket.socket.connect = refuse  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._orig  # type: ignore[method-assign]


def replay_check(snapshot_id: str, log=print) -> bool:
    """Replay every question with sockets disabled; report per-question counts."""
    manifest = load_manifest(snapshot_id)
    palace = open_palace(palace_path(snapshot_id), read_only=True)
    ok = True
    with _NoNetwork():
        for entry in manifest["questions"]:
            try:
                out = replay(snapshot_id, entry["question_id"], palace=palace)
                n_chunks = sum(len(r["chunks"]) for r in out["results"])
                log(f"  {entry['question_id']}: {len(out['results'])} results, {n_chunks} chunks replayed")
            except Exception as exc:  # noqa: BLE001 - report every failure, then fail once
                ok = False
                log(f"  {entry['question_id']}: FAIL {exc}")
    return ok


def engine_report(manifest: dict[str, Any]) -> str:
    """How many results each question got, and which engines were suspended, so a
    degraded freeze is visible before anyone runs an arm on it."""
    questions = manifest["questions"]
    buckets = {"0": 0, "1-9": 0, "10-29": 0, "30": 0}
    engines: dict[str, int] = {}
    suspended: dict[str, int] = {}
    for q in questions:
        n = q["n_results"]
        key = "0" if n == 0 else "1-9" if n < 10 else "10-29" if n < 30 else "30"
        buckets[key] += 1
        for r in q["results"]:
            for e in r["engines"]:
                engines[e] = engines.get(e, 0) + 1
        for u in q["searxng_raw"].get("unresponsive_engines") or []:
            if u:
                suspended[u[0]] = suspended.get(u[0], 0) + 1
    lines = [
        f"questions by result count: {', '.join(f'{k}: {v}' for k, v in buckets.items())}",
        f"results per engine: {', '.join(f'{k}: {v}' for k, v in sorted(engines.items()))}",
        f"questions with an engine suspended or failing: {', '.join(f'{k}: {v}' for k, v in sorted(suspended.items())) or 'none'}",
    ]
    return "\n".join(lines)


def summary_table(manifest: dict[str, Any]) -> str:
    lines = [f"{'question_id':<16} {'results':>7} {'pages':>5} {'chunks':>6} {'new':>5}"]
    for q in manifest["questions"]:
        lines.append(
            f"{q['question_id']:<16} {q['n_results']:>7} {q['n_fetched_ok']:>5} {q['n_chunks']:>6} {q['n_chunks_new']:>5}"
        )
    total = (
        sum(q["n_results"] for q in manifest["questions"]),
        sum(q["n_fetched_ok"] for q in manifest["questions"]),
        sum(q["n_chunks"] for q in manifest["questions"]),
        sum(q["n_chunks_new"] for q in manifest["questions"]),
    )
    lines.append(f"{'total':<16} {total[0]:>7} {total[1]:>5} {total[2]:>6} {total[3]:>5}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze or replay a retrieval snapshot")
    parser.add_argument("--dataset", default="simpleqa")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--id", required=True, help="snapshot id, for example pilot-20260919")
    parser.add_argument("--category", default="general")
    parser.add_argument("--time-range", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fetch-timeout", type=float, default=10.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between searches")
    parser.add_argument("--redo-below", type=int, default=None, help="re-search frozen questions with fewer results than this")
    parser.add_argument("--report", action="store_true", help="print the result-count and engine report and exit")
    parser.add_argument("--replay-check", action="store_true", help="replay every question offline")
    args = parser.parse_args(argv)
    configure_stdout()

    if args.report:
        manifest = load_manifest(args.id)
        print(f"snapshot {args.id}: {len(manifest['questions'])} of {manifest['n']} questions frozen")
        print(engine_report(manifest))
        return 0
    if args.replay_check:
        print(f"replaying snapshot {args.id} with the network disabled")
        ok = replay_check(args.id)
        print(summary_table(load_manifest(args.id)))
        print(engine_report(load_manifest(args.id)))
        return 0 if ok else 1

    manifest = freeze(
        args.dataset,
        snapshot_id=args.id,
        n=args.n,
        seed=args.seed,
        category=args.category,
        time_range=args.time_range,
        workers=args.workers,
        fetch_timeout=args.fetch_timeout,
        sleep_s=args.sleep,
        redo_below=args.redo_below,
    )
    print(summary_table(manifest))
    print(engine_report(manifest))
    print(f"palace drawers: {manifest.get('palace_drawers')}  manifest: {manifest_path(args.id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
