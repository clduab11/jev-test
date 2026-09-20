"""Run the web track unattended: finish the freeze, re-search short questions when
the engines allow it, run arms A, B and D over the snapshot, grade, print the bars.

    uv run --with "system-one-adapter[openai]>=0.2.0" python scripts/run_web_track.py \
        --snapshot simpleqa-500-20260919 --n 500 --cself-pilot

Every step is skipped when its output already exists, so the script can be
restarted after any failure and picks up where it stopped. Progress goes to
results/web_track.log and to stdout.

Steps, in order:
  0. (optional) arm C-self on the 20-question pilot, resumable; it only needs
     the local generator, so it runs while the freeze is still fetching
  1. wait for the running freeze to reach n questions; if the manifest stops
     changing for 15 minutes, run the freeze here (it resumes)
  2. re-search the questions that got fewer than 10 results; then, if a probe
     search shows more than one engine answering, re-search everything below 30
  3. arms A, B and D over the snapshot, each resumable
  4. grade everything and print the pre-registered bars
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = ROOT / "results" / "web_track.log"


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def child_env() -> dict[str, str]:
    """UTF-8 in every child, whatever the machine's locale codec is."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run(args: list[str]) -> int:
    log("run: " + " ".join(args))
    with LOG.open("a", encoding="utf-8") as handle:
        process = subprocess.run(
            [sys.executable, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            text=True, check=False, env=child_env(),
        )
    log(f"exit {process.returncode}")
    return process.returncode


def manifest_state(snapshot: str) -> tuple[int, float]:
    path = ROOT / "snapshots" / snapshot / "manifest.json"
    if not path.exists():
        return 0, 0.0
    for _ in range(5):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return len(data.get("questions", [])), path.stat().st_mtime
        except (OSError, ValueError):
            time.sleep(1.0)
    return 0, path.stat().st_mtime


def results_complete(arm: str, snapshot: str, n: int) -> bool:
    path = ROOT / "results" / f"{arm}_simpleqa_{snapshot}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("complete", True) and len(data.get("records", [])) >= n


def graded(arm: str, snapshot: str) -> bool:
    path = ROOT / "results" / f"{arm}_simpleqa_{snapshot}.json"
    if not path.exists():
        return False
    try:
        return "metrics" in json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def engines_answering() -> set[str]:
    from harness.retrieval import searxng

    try:
        raw = searxng.search_raw("Eiffel Tower height in metres", category="general")
    except Exception as exc:  # noqa: BLE001 - a probe failure just means "assume nothing recovered"
        log(f"probe search failed: {exc}")
        return set()
    answering = {e for r in raw.get("results", []) for e in (r.get("engines") or [])}
    suspended = [u[0] for u in (raw.get("unresponsive_engines") or []) if u]
    log(f"probe: {len(raw.get('results', []))} results, engines answering {sorted(answering)}, suspended {sorted(set(suspended))}")
    return answering


def wait_for_freeze(snapshot: str, n: int, seed: int, sleep_s: float, stale_minutes: float = 15.0) -> None:
    while True:
        count, mtime = manifest_state(snapshot)
        if count >= n:
            log(f"freeze complete: {count} of {n}")
            return
        age_min = (time.time() - mtime) / 60 if mtime else 1e9
        if age_min > stale_minutes:
            log(f"freeze at {count} of {n}, manifest untouched for {age_min:.0f} min; running it here")
            run(["-m", "harness.retrieval.snapshot", "--dataset", "simpleqa", "--n", str(n), "--seed", str(seed), "--id", snapshot, "--sleep", str(sleep_s)])
            continue
        log(f"freeze at {count} of {n}; waiting")
        time.sleep(120)


def main() -> int:
    parser = argparse.ArgumentParser(description="run the web track end to end")
    parser.add_argument("--snapshot", default="simpleqa-500-20260919")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--sleep", type=float, default=20.0, help="seconds between searches when this script freezes")
    parser.add_argument("--cself-pilot", action="store_true", help="also run arm C-self on the 20-question pilot first")
    parser.add_argument("--pilot", default="pilot-20260919")
    parser.add_argument("--skip-redo", action="store_true")
    args = parser.parse_args()
    log(f"web track start: snapshot {args.snapshot}, n {args.n}")

    if args.cself_pilot and not results_complete("C-self", args.pilot, 20):
        run(["-m", "harness.run", "--arm", "C-self", "--dataset", "simpleqa", "--snapshot", args.pilot, "--resume"])

    wait_for_freeze(args.snapshot, args.n, args.seed, args.sleep)

    if not args.skip_redo:
        run(["-m", "harness.retrieval.snapshot", "--dataset", "simpleqa", "--n", str(args.n), "--seed", str(args.seed), "--id", args.snapshot, "--sleep", str(args.sleep), "--redo-below", "10"])
        answering = engines_answering()
        if len(answering) >= 2:
            run(["-m", "harness.retrieval.snapshot", "--dataset", "simpleqa", "--n", str(args.n), "--seed", str(args.seed), "--id", args.snapshot, "--sleep", str(args.sleep), "--redo-below", "30"])
        else:
            log("only one engine answers; skipping the full re-search. Run it later with --redo-below 30 once the engines recover.")
    run(["-m", "harness.retrieval.snapshot", "--id", args.snapshot, "--report"])

    for arm in ("A", "B", "D"):
        if results_complete(arm, args.snapshot, args.n):
            log(f"arm {arm}: results complete, skipping")
            continue
        code = run(["-m", "harness.run", "--arm", arm, "--dataset", "simpleqa", "--snapshot", args.snapshot, "--resume"])
        if code != 0:
            log(f"arm {arm} failed; stopping so the failure is visible")
            return code

    to_grade = [str(ROOT / "results" / f"{arm}_simpleqa_{args.snapshot}.json") for arm in ("A", "B", "D")]
    if args.cself_pilot and results_complete("C-self", args.pilot, 20):
        to_grade.append(str(ROOT / "results" / f"C-self_simpleqa_{args.pilot}.json"))
    run(["-m", "harness.run", "--grade", *to_grade])
    log("web track done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
