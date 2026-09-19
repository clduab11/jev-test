"""Stage S1: the passage gate (spec section 4).

One request per (query, passage). Runs on SearXNG snippets, on chunks of the
pages that survived, and on palace drawers. A drawer gets no shortcut (rule
11). The five questions come verbatim from judge/questions.v1.json; the
routing is the spec's first-match-wins table with thresholds from the same
file.

``source_type`` is assigned here, in code, from the hostname (spec: "from a
domain allowlist or from the drawer's wing, never by the judge"). The judge
sees the label as context and nothing more.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from harness.config import SPEC_PATH, thresholds
from harness.judge.base import Answer, Judge, NoulAnswer, Question, load_questions, run_many
from harness.retrieval import palace as palacemod
from harness.retrieval.fetch import cap_chars, clean_text

STAGE = "S1_passage_gate"
SOURCE_TYPES = (
    "official_documentation",
    "encyclopedia",
    "news",
    "forum",
    "blog",
    "vendor",
    "memory_evidence",
    "memory_verified",
    "unknown",
)

# --------------------------------------------------------------------------- #
# The domain allowlist. This is the one S1 policy the spec leaves to code.
#
# Rules, checked in this order: exact host, then hostname prefix, then
# registered-domain suffix (longest first). The list is short on purpose. A
# label is only worth sending when it is nearly always right, and ``unknown``
# is an honest answer for the long tail (databases, personal sites, .gov,
# .edu, mirrors). Edit the three tables, not the function.
# --------------------------------------------------------------------------- #

_EXACT: dict[str, str] = {
    "news.ycombinator.com": "forum",
    "learn.microsoft.com": "official_documentation",
    "developer.mozilla.org": "official_documentation",
    "docs.python.org": "official_documentation",
    "abcnews.go.com": "news",
    "news.google.com": "news",
    "news.sky.com": "news",
}

_PREFIXES: tuple[tuple[str, str], ...] = (
    ("docs.", "official_documentation"),
    ("doc.", "official_documentation"),
    ("developer.", "official_documentation"),
    ("developers.", "official_documentation"),
    ("forum.", "forum"),
    ("forums.", "forum"),
    ("community.", "forum"),
    ("discuss.", "forum"),
    ("boards.", "forum"),
    ("blog.", "blog"),
    ("blogs.", "blog"),
    ("news.", "news"),
    ("store.", "vendor"),
    ("shop.", "vendor"),
)

_SUFFIXES: dict[str, str] = {
    # encyclopedias and their mirrors
    "wikipedia.org": "encyclopedia",
    "wikiwand.com": "encyclopedia",
    "wiki2.org": "encyclopedia",
    "wikidata.org": "encyclopedia",
    "britannica.com": "encyclopedia",
    "encyclopedia.com": "encyclopedia",
    "newworldencyclopedia.org": "encyclopedia",
    "scholarpedia.org": "encyclopedia",
    "dbpedia.org": "encyclopedia",
    # documentation hosts
    "readthedocs.io": "official_documentation",
    "readthedocs.org": "official_documentation",
    "kubernetes.io": "official_documentation",
    "rust-lang.org": "official_documentation",
    "nodejs.org": "official_documentation",
    "go.dev": "official_documentation",
    # news
    "reuters.com": "news",
    "apnews.com": "news",
    "bbc.com": "news",
    "bbc.co.uk": "news",
    "nytimes.com": "news",
    "theguardian.com": "news",
    "washingtonpost.com": "news",
    "cnn.com": "news",
    "nbcnews.com": "news",
    "cbsnews.com": "news",
    "npr.org": "news",
    "bloomberg.com": "news",
    "ft.com": "news",
    "wsj.com": "news",
    "economist.com": "news",
    "aljazeera.com": "news",
    "cbc.ca": "news",
    "abc.net.au": "news",
    "dw.com": "news",
    "france24.com": "news",
    "techcrunch.com": "news",
    "theverge.com": "news",
    "arstechnica.com": "news",
    "wired.com": "news",
    "zdnet.com": "news",
    "cnet.com": "news",
    "engadget.com": "news",
    "politico.com": "news",
    "axios.com": "news",
    "latimes.com": "news",
    "usatoday.com": "news",
    "time.com": "news",
    "newsweek.com": "news",
    "forbes.com": "news",
    "businessinsider.com": "news",
    "telegraph.co.uk": "news",
    "independent.co.uk": "news",
    "thetimes.co.uk": "news",
    "thehindu.com": "news",
    "scmp.com": "news",
    "japantimes.co.jp": "news",
    "straitstimes.com": "news",
    "smh.com.au": "news",
    "globalnews.ca": "news",
    "ctvnews.ca": "news",
    "foxnews.com": "news",
    "huffpost.com": "news",
    "vox.com": "news",
    "propublica.org": "news",
    # forums and question sites
    "reddit.com": "forum",
    "stackexchange.com": "forum",
    "stackoverflow.com": "forum",
    "superuser.com": "forum",
    "serverfault.com": "forum",
    "askubuntu.com": "forum",
    "mathoverflow.net": "forum",
    "quora.com": "forum",
    # blog platforms
    "medium.com": "blog",
    "substack.com": "blog",
    "wordpress.com": "blog",
    "blogspot.com": "blog",
    "blogger.com": "blog",
    "tumblr.com": "blog",
    "dev.to": "blog",
    "hashnode.dev": "blog",
    "ghost.io": "blog",
    "livejournal.com": "blog",
    "typepad.com": "blog",
    # vendors: the seller's or maker's own site
    "amazon.com": "vendor",
    "ebay.com": "vendor",
    "walmart.com": "vendor",
    "bestbuy.com": "vendor",
    "target.com": "vendor",
    "newegg.com": "vendor",
    "etsy.com": "vendor",
    "aliexpress.com": "vendor",
    "alibaba.com": "vendor",
    "apple.com": "vendor",
    "microsoft.com": "vendor",
    "samsung.com": "vendor",
    "sony.com": "vendor",
    "dell.com": "vendor",
    "hp.com": "vendor",
    "lenovo.com": "vendor",
    "nvidia.com": "vendor",
    "intel.com": "vendor",
    "amd.com": "vendor",
    "oracle.com": "vendor",
    "adobe.com": "vendor",
    "salesforce.com": "vendor",
    "ikea.com": "vendor",
    "nike.com": "vendor",
    "tesla.com": "vendor",
}
_SUFFIX_ORDER = sorted(_SUFFIXES, key=len, reverse=True)


def source_type_for(domain: str | None, wing: str | None = None) -> str:
    """Label a passage by where it came from. Memory wings win over hostnames."""
    if wing == palacemod.VERIFIED:
        return "memory_verified"
    if wing in (palacemod.EVIDENCE, palacemod.QUARANTINE) and (domain or "palace") == "palace":
        return "memory_evidence"
    host = (domain or "").strip().lower().removeprefix("www.")
    if not host or host == "palace":
        return "memory_evidence" if wing else "unknown"
    if host in _EXACT:
        return _EXACT[host]
    for prefix, kind in _PREFIXES:
        if host.startswith(prefix):
            return kind
    for suffix in _SUFFIX_ORDER:
        if host == suffix or host.endswith("." + suffix):
            return _SUFFIXES[suffix]
    return "unknown"


# --------------------------------------------------------------------------- #
# Passages and state
# --------------------------------------------------------------------------- #


def questions() -> dict[str, Question]:
    return load_questions(SPEC_PATH, STAGE)


def prepare_text(text: str, max_chars: int | None = None) -> str:
    """HTML and code fences stripped, whitespace normalised, capped at passage_max_chars."""
    max_chars = int(max_chars or thresholds()["passage_max_chars"])
    return cap_chars(clean_text(text or ""), max_chars)


def make_passage(
    *,
    id: str,
    text: str,
    kind: str,
    lane: str,
    title: str | None = "",
    domain: str | None = "unknown",
    wing: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The passage dict every stage after S1 reads. ``kind`` is snippet, chunk or
    drawer; ``lane`` is web or palace (which rank lists it joins)."""
    return {
        "id": id,
        "text": prepare_text(text),
        "title": (title or "").strip(),
        "domain": (domain or "unknown").lower(),
        "source_type": source_type_for(domain, wing),
        "kind": kind,
        "lane": lane,
        **extra,
    }


def build_state(query: str, passage: Mapping[str, Any]) -> dict[str, Any]:
    """Exactly the section 4 state: the query and one passage, nothing else."""
    return {
        "query": query,
        "passage": {
            "id": passage["id"],
            "title": passage.get("title") or "",
            "domain": passage.get("domain") or "unknown",
            "source_type": passage.get("source_type") or "unknown",
            "text": passage["text"],
        },
    }


def _noul(answers: Mapping[str, Any], qid: str) -> float:
    value = answers[qid]
    return float(value.noul) if isinstance(value, NoulAnswer) else float(value)


def route(answers: Mapping[str, Any], t: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """The spec's first-match-wins table. Returns (route, reason)."""
    t = t or thresholds()
    if _noul(answers, "contains_prompt_injection") > float(t["injection_max"]):
        return "DROP", "injection"
    if _noul(answers, "is_promotional_or_boilerplate") > float(t["boilerplate_max"]):
        return "DROP", "boilerplate"
    if _noul(answers, "is_relevant") < float(t["relevant_min"]):
        return "DROP", "off_topic"
    if _noul(answers, "contradicts_query_premise") > float(t["contradicts_min"]):
        return "CONFLICT", "contradicts_premise"
    if _noul(answers, "contains_answer_evidence") > float(t["evidence_min"]):
        return "EVIDENCE", "evidence"
    return "DROP", "nothing_usable"


def _quarantine(palace: palacemod.Palace, passage: Mapping[str, Any], prob: float) -> dict[str, Any]:
    """Side effect for an injection: file a fetched chunk to the quarantine wing,
    or flag a recalled drawer in place. A snippet is only dropped and logged; the
    spec quarantines fetched chunks. Never raises; the outcome is recorded."""
    try:
        if passage.get("kind") == "drawer" and passage.get("drawer_id"):
            flagged = palacemod.flag_drawer(palace, passage["drawer_id"], prob)
            return {"flagged": passage["drawer_id"], "ok": flagged}
        if passage.get("kind") == "chunk" and passage.get("url"):
            drawer_id, created = palacemod.quarantine(
                palace,
                text=passage["text"],
                url=passage["url"],
                chunk_index=int(passage.get("chunk_index") or 0),
                injection_prob=prob,
                fetched_at=passage.get("fetched_at") or palacemod._now(),
            )
            return {"quarantined": drawer_id, "created": created}
        return {"skipped": "not a fetched chunk or a drawer"}
    except Exception as exc:  # noqa: BLE001 - a palace failure must not stop the pipeline
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def gate(
    judge: Judge,
    query: str,
    passages: list[dict[str, Any]],
    quarantine_palace: palacemod.Palace | None = None,
    t: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge every passage (one request each, through the judge's pool) and route it.

    Returns evidence, conflicts and dropped lists (each passage carries a
    ``gate`` block with its route, reason and the five probabilities), plus
    request and token counts.
    """
    t = t or thresholds()
    qs = questions()
    items = [(build_state(query, p), qs) for p in passages]
    responses = run_many(judge, items) if items else []
    evidence: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for passage, response in zip(passages, responses, strict=True):
        answers: Mapping[str, Answer] = response.answers
        plain = {qid: float(a.noul) for qid, a in answers.items() if isinstance(a, NoulAnswer)}
        verdict, reason = route(answers, t)
        judged = {
            **passage,
            "gate": {
                "route": verdict,
                "reason": reason,
                "answers": plain,
                "model": response.model,
                "cached": response.cached,
            },
        }
        if reason == "injection" and quarantine_palace is not None:
            judged["gate"]["injection"] = _quarantine(
                quarantine_palace, passage, plain.get("contains_prompt_injection", 1.0)
            )
        reasons[reason] += 1
        if verdict == "EVIDENCE":
            evidence.append(judged)
        elif verdict == "CONFLICT":
            conflicts.append(judged)
        else:
            dropped.append(judged)
    return {
        "evidence": evidence,
        "conflicts": conflicts,
        "dropped": dropped,
        "reasons": dict(reasons),
        "n_requests": len(items),
        "n_cached": sum(1 for r in responses if r.cached),
        "input_tokens": sum(r.input_tokens for r in responses),
        "output_tokens": sum(r.output_tokens for r in responses),
        "model": next((r.model for r in responses), None),
    }
