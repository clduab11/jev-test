"""MemPalace helpers: file evidence, file verified claims, quarantine, search.

Wings and rooms follow the spec's section 8 layout:

  evidence    room = source domain or dataset name; one verbatim chunk
  verified    room = query subject slug; one kept claim
  quarantine  room = source domain; a passage that tripped the injection gate

Drawer ids are deterministic. An evidence drawer id is a hash of (url,
chunk_index), which is also the de-duplication key the spec asks for: filing
the same chunk twice is a no-op, and a citation id always resolves to one
verbatim, locally stored passage.

Metadata conventions copy ``mempalace.convo_miner.file_conversation_exchange``
(the canonical write path in MemPalace 3.10) so search, date windows and
hallways treat these drawers like any other. The one deliberate difference is
that ``chunk_index`` is real rather than always 0, because that helper does
not let callers set it.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mempalace.convo_miner import (
    ID_RECIPE,
    NORMALIZE_VERSION,
    _detect_hall_cached,
    entities_metadata,
)
from mempalace.palace import get_collection
from mempalace.searcher import search_memories

from harness.config import thresholds

EVIDENCE = "evidence"
VERIFIED = "verified"
QUARANTINE = "quarantine"
AGENT = "jev-test"

_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "is", "was", "are", "were", "who",
    "what", "when", "where", "which", "how", "many", "much", "did", "does", "do", "and", "or",
    "by", "with", "from", "as", "that", "this", "it", "its", "be", "has", "have", "had",
}


@dataclass
class Palace:
    path: Path
    collection: Any

    def count(self) -> int:
        return int(self.collection.count())


def open_palace(path: str | Path, read_only: bool = False) -> Palace:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return Palace(path=path, collection=get_collection(str(path), read_only=read_only))


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "unknown").lower()
    return host.removeprefix("www.")


def subject_slug(query: str, words: int = 3) -> str:
    """First three content words of the query, lowercased; the verified-wing room."""
    tokens = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOP]
    return "_".join(tokens[:words]) or "general"


def evidence_drawer_id(url: str, chunk_index: int) -> str:
    digest = hashlib.sha1(f"{url}\n{chunk_index}".encode()).hexdigest()[:16]
    return f"ev_{digest}_c{chunk_index}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _clean_meta(extra: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        elif not isinstance(value, (str, int, float, bool)):
            value = str(value)
        out[key] = value
    return out


def file_drawer(
    palace: Palace,
    *,
    drawer_id: str,
    wing: str,
    room: str,
    text: str,
    source_file: str,
    chunk_index: int = 0,
    authored_at: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Upsert one verbatim drawer. Returns (drawer_id, created)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("refusing to file an empty drawer")
    existing = palace.collection.get(ids=[drawer_id])
    if existing and existing.get("ids"):
        return drawer_id, False
    filed_at = _now()
    metadata: dict[str, Any] = {
        "wing": wing,
        "room": room,
        "hall": _detect_hall_cached(text),
        "source_file": source_file,
        "chunk_index": int(chunk_index),
        "added_by": AGENT,
        "filed_at": filed_at,
        "entities": entities_metadata(text),
        "authored_at": authored_at or filed_at,
        "ingest_mode": "convos",
        "extract_mode": "exchange",
        "normalize_version": NORMALIZE_VERSION,
        "id_recipe": ID_RECIPE,
    }
    for key, value in _clean_meta(extra_metadata or {}).items():
        metadata.setdefault(key, value)
    palace.collection.upsert(ids=[drawer_id], documents=[text], metadatas=[metadata])
    return drawer_id, True


def file_chunk(
    palace: Palace,
    *,
    text: str,
    url: str,
    chunk_index: int,
    query_id: str,
    dataset: str,
    engine_rank: int,
    engines: list[str] | str,
    fetched_at: str,
    published_date: str | None = None,
    title: str | None = None,
) -> tuple[str, bool]:
    """File one page chunk into the evidence wing (stage M1). Dedup by (url, chunk_index)."""
    return file_drawer(
        palace,
        drawer_id=evidence_drawer_id(url, chunk_index),
        wing=EVIDENCE,
        room=domain_of(url),
        text=text,
        source_file=url,
        chunk_index=chunk_index,
        authored_at=published_date or fetched_at,
        extra_metadata={
            "url": url,
            "fetched_at": fetched_at,
            "query_id": query_id,
            "dataset": dataset,
            "engine_rank": int(engine_rank),
            "engines": engines,
            "published_date": published_date,
            "title": title,
        },
    )


def file_verified(
    palace: Palace,
    *,
    text: str,
    query: str,
    query_id: str,
    arm: str,
    support_prob: float,
    support_choice: str,
    judge_model: str,
    evidence_drawer_ids: list[str],
    verified_at: str | None = None,
) -> tuple[str, bool]:
    """File one kept claim into the verified wing (stage M2)."""
    verified_at = verified_at or _now()
    digest = hashlib.sha1(f"{query_id}\n{text}".encode()).hexdigest()[:16]
    return file_drawer(
        palace,
        drawer_id=f"vf_{digest}",
        wing=VERIFIED,
        room=subject_slug(query),
        text=text,
        source_file=f"claim:{query_id}",
        authored_at=verified_at,
        extra_metadata={
            "support_prob": float(support_prob),
            "support_choice": support_choice,
            "judge_model": judge_model,
            "evidence_drawer_ids": evidence_drawer_ids,
            "verified_at": verified_at,
            "query_id": query_id,
            "arm": arm,
        },
    )


def quarantine(
    palace: Palace,
    *,
    text: str,
    url: str,
    chunk_index: int,
    injection_prob: float,
    fetched_at: str,
) -> tuple[str, bool]:
    """Keep a passage that tripped the injection gate as evidence of poisoning."""
    digest = hashlib.sha1(f"{url}\n{chunk_index}".encode()).hexdigest()[:16]
    return file_drawer(
        palace,
        drawer_id=f"qr_{digest}_c{chunk_index}",
        wing=QUARANTINE,
        room=domain_of(url),
        text=text,
        source_file=url,
        chunk_index=chunk_index,
        authored_at=fetched_at,
        extra_metadata={"injection_prob": float(injection_prob), "url": url, "fetched_at": fetched_at},
    )


def search_wing(
    palace: Palace,
    query: str,
    wing: str,
    n: int | None = None,
    max_distance: float | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search inside one wing. Returns text, drawer_id, similarity and provenance.
    ``since`` is an ISO date floor on authored_at, applied by MemPalace (spec section 3)."""
    t = thresholds()
    n = int(n or t["palace_top_k"])
    max_distance = float(t["palace_max_distance"] if max_distance is None else max_distance)
    envelope = search_memories(
        query, str(palace.path), wing=wing, n_results=n, max_distance=max_distance, since=since
    )
    hits = envelope.get("results", []) if isinstance(envelope, dict) else []
    return [
        {
            "drawer_id": hit.get("drawer_id"),
            "text": hit.get("text", ""),
            "similarity": hit.get("similarity"),
            "distance": hit.get("distance"),
            "wing": hit.get("wing"),
            "room": hit.get("room"),
            "source_file": hit.get("source_file"),
            "authored_at": hit.get("authored_at"),
        }
        for hit in hits
    ]


def search_evidence_for_question(
    palace: Palace,
    query: str,
    question_id: str,
    n: int | None = None,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Semantic search of the evidence wing restricted to one question's drawers.

    Used by the replay-mode refinement round: a second look at the frozen
    result set with a different query. Goes to the collection directly because
    ``search_memories`` has no metadata filter beyond wing and room.
    """
    t = thresholds()
    n = int(n or t["palace_top_k"])
    max_distance = float(t["palace_max_distance"] if max_distance is None else max_distance)
    got = palace.collection.query(
        query_texts=[query],
        n_results=n,
        where={"$and": [{"wing": EVIDENCE}, {"query_id": question_id}]},
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict[str, Any]] = []
    ids = (got.get("ids") or [[]])[0]
    docs = (got.get("documents") or [[]])[0]
    metas = (got.get("metadatas") or [[]])[0]
    dists = (got.get("distances") or [[]])[0]
    for drawer_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
        meta = dict(meta or {})
        if max_distance and dist is not None and float(dist) > max_distance:
            continue
        engines = meta.get("engines")
        out.append(
            {
                "drawer_id": drawer_id,
                "text": doc,
                "similarity": None if dist is None else 1.0 - float(dist),
                "distance": dist,
                "wing": meta.get("wing"),
                "room": meta.get("room"),
                "url": meta.get("url"),
                "title": meta.get("title"),
                "chunk_index": meta.get("chunk_index"),
                "engine_rank": meta.get("engine_rank"),
                "engines": engines.split(",") if isinstance(engines, str) and engines else [],
                "fetched_at": meta.get("fetched_at"),
                "authored_at": meta.get("authored_at"),
            }
        )
    return out


def flag_drawer(palace: Palace, drawer_id: str, injection_prob: float) -> bool:
    """Mark a recalled drawer that tripped the injection gate. Returns False if absent."""
    got = palace.collection.get(ids=[drawer_id], include=["metadatas"])
    if not got or not got.get("ids"):
        return False
    metadata = dict((got.get("metadatas") or [{}])[0] or {})
    metadata["flagged_injection"] = float(injection_prob)
    metadata["flagged_at"] = _now()
    palace.collection.update(ids=[drawer_id], metadatas=[metadata])
    return True


def get_drawers(palace: Palace, drawer_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Verbatim text and metadata for the given drawer ids, keyed by id."""
    if not drawer_ids:
        return {}
    got = palace.collection.get(ids=list(drawer_ids), include=["documents", "metadatas"])
    out: dict[str, dict[str, Any]] = {}
    for drawer_id, doc, meta in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", []), strict=False):
        out[drawer_id] = {"text": doc, "metadata": dict(meta or {})}
    return out
