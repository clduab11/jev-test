"""A /v1/systemone shim over the Laya decision model (arm C-laya).

    uv run --with laya python -m harness.judge.laya_shim --port 8100

Laya (convaiinnovations/laya, Apache 2.0, about 421M parameters) answers the
same noul, choice and score questions in Python: ``agent.predict(state,
questions)``. It serves no HTTP, so this module wraps it in TypeSafe's wire
format and HttpJudge talks to it like any other judge (LOCAL_JUDGE_BASE_URL,
LOCAL_JUDGE_MODEL). Every arm sends byte-identical questions; only the base
URL changes.

Written before Laya was installed on this machine. The translation of its
answer dicts is defensive about key names and is covered by a test that uses a
fake predict function and a real HTTP round trip through HttpJudge. The first
real call should be ``scripts/smoke_jev.py`` pointed at this shim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .base import Question

Predict = Callable[[Any, Mapping[str, Question]], Mapping[str, Any]]


def _first(answer: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in answer and answer[key] is not None:
            return answer[key]
    return None


def translate(
    state: Any, questions: Mapping[str, Question], predict: Predict
) -> dict[str, dict[str, Any]]:
    """Call ``predict`` and shape its answers the way ``/v1/systemone`` returns them."""
    result = predict(state, dict(questions))
    raw = result.get("answers", result) if isinstance(result, Mapping) else {}
    answers: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        answer = raw.get(qid) or {}
        kind = question.get("type")
        if kind == "noul":
            p = _first(answer, ("noul", "probability", "p_true", "p", "value"))
            answers[qid] = {"type": "noul", "noul": float(0.5 if p is None else p)}
        elif kind == "choice":
            probs = _first(answer, ("probabilities", "probs", "distribution")) or {}
            probs = {str(k): float(v) for k, v in dict(probs).items()}
            options = [str(k) for k in (question.get("criteria") or {})]
            choice = _first(answer, ("choice", "label", "answer"))
            if choice is None:
                choice = max(probs, key=probs.get) if probs else (options[0] if options else "")
            confidence = _first(answer, ("confidence",))
            if confidence is None:
                confidence = probs.get(str(choice), 0.0)
            answers[qid] = {
                "type": "choice",
                "choice": str(choice),
                "probabilities": probs,
                "confidence": float(confidence),
            }
        elif kind == "score":
            probs = _first(answer, ("probabilities", "probs", "distribution")) or {}
            probs = {str(k): float(v) for k, v in dict(probs).items()}
            legend = _first(answer, ("legend",)) or {
                str(i): str(c) for i, c in enumerate(question.get("criteria") or [])
            }
            score = _first(answer, ("score", "expected", "value"))
            confidence = _first(answer, ("confidence",))
            answers[qid] = {
                "type": "score",
                "score": float(0.5 if score is None else score),
                "probabilities": probs,
                "legend": {str(k): str(v) for k, v in dict(legend).items()},
                "confidence": float(max(probs.values()) if confidence is None and probs else (confidence or 0.0)),
            }
    return answers


def make_handler(predict: Predict, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/health"):
                self._send(200, {"ok": True, "model": model_name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                state = body["state"]
                questions = body["questions"]
            except (ValueError, KeyError) as exc:
                self._send(422, {"error": f"bad request: {exc}"})
                return
            try:
                answers = translate(state, questions, predict)
            except Exception as exc:  # noqa: BLE001 - report the model failure to the caller
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"[:500]})
                return
            approx_tokens = len(json.dumps({"state": state, "questions": questions})) // 4
            self._send(
                200,
                {
                    "model": model_name,
                    "answers": answers,
                    "usage": {"input_tokens": approx_tokens, "output_tokens": 0},
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            return None

    return Handler


def serve(predict: Predict, host: str = "127.0.0.1", port: int = 8100, model_name: str = "laya-english") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(predict, model_name))
    return server


def load_laya(model_id: str = "convaiinnovations/laya", subfolder: str | None = None) -> Predict:
    import laya  # pip install laya; needs torch and transformers

    agent = laya.load(model_id, subfolder=subfolder) if subfolder else laya.load(model_id)
    return lambda state, questions: agent.predict(state, questions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="serve Laya on /v1/systemone")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default=None, help="for example 'multilingual'")
    parser.add_argument("--model-name", default="laya-english", help="the model id reported on the wire")
    args = parser.parse_args(argv)
    predict = load_laya(args.model, args.subfolder)
    server = serve(predict, args.host, args.port, args.model_name)
    print(f"laya shim on http://{args.host}:{args.port}/v1/systemone as {args.model_name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
