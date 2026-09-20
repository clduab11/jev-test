"""Mechanical check of Markdown files against Wikipedia's "Signs of AI writing" list.

Usage:  python scripts/ai_signs_check.py README.md docs/*.md
Exit code is 1 when any flag is found, so it can gate a commit.
Word list and patterns were taken from the page as read on 2026-09-19.
"""
import pathlib
import re
import sys

WORDS = r"""additionally|boasts?|bolster\w*|crucial|delve|emphasi[sz]\w*|enduring|garner\w*|intricate|intricacies|interplay|landscape|meticulous\w*|pivotal|underscor\w*|vibrant|align\w* with|enhanc\w*|foster\w*|highlighting|showcas\w*|tapestry|testament|seamless\w*|robust\w*|leverag\w*|navigat\w*|realm|comprehensive|notably|moreover|furthermore|ensur\w*|streamlin\w*|cutting-edge|innovative|game-?changer|unlock\w*|harness(?:es|ing| the)|elevat\w*|empower\w*|transformative|serves as|stands as|plays? an? \w+ role|vital role|associated with|connected to|worth noting|important to note|in today'?s|at its core|deep dive|in conclusion|in summary|^overall,|significant\w*|essential\w*|journey|embark\w*|excited|i hope this helps|it'?s important|multifaceted|nuanced|holistic|synergy|paradigm|groundbreaking|revolutioni[sz]\w*|state-of-the-art|best practices|in the realm|a wide range of|plays a (?:key|crucial)"""
PATTERNS = {
 "em dash": r"—",
 "spaced en dash": r"\s–\s",
 "curly quote": r"[\u2018\u2019\u201c\u201d]",
 "emoji": r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
 "bold-label list item": r"^\s*[-*]\s+\*\*[^*]+\*\*\s*:",
 "not only/just X but": r"\bnot (?:only|just|merely)\b",
 "'rather than'": r"\brather than\b",
 "'not X, but Y'": r"\bnot\b[^.\n]{0,40},\s*but\b",
 "horizontal rule": r"^(?:---|\*\*\*)\s*$",
 "AI vocabulary": r"\b(?:%s)\b" % WORDS,
}
TOTAL=0
for path in sys.argv[1:]:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    print(f"=== {path} ===")
    hits = 0
    for name, pat in PATTERNS.items():
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```") : pass
            for m in re.finditer(pat, line, flags=re.IGNORECASE|re.MULTILINE):
                hits += 1
                print(f"  [{name}] line {i}: ...{line[max(0,m.start()-40):m.end()+40].strip()}...")
    heads = [l for l in text.splitlines() if l.startswith("#")]
    print("  headings:"); [print("   ", h) for h in heads]
    print(f"  total flags: {hits}")
    TOTAL+=hits
sys.exit(1 if TOTAL else 0)
