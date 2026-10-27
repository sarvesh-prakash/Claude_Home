"""Tiny local web UI (stdlib only)."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from .detector import analyze as heuristic
from . import ml

USE_ML = ml.available()


def analyze(text):
    if USE_ML:
        try:
            return ml.analyze_combined(text)
        except Exception as e:  # offline / model missing -> fall back
            print('ML scorer failed, using heuristics:', e)
    return heuristic(text)

PAGE = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_bytes()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        r = analyze(self.rfile.read(min(n, 1_000_000)).decode("utf-8", "replace"))
        self._send(200, json.dumps({"score": r.score, "label": r.label, "signals": r.signals,
                                    "sentences": r.sentences, "word_count": r.word_count,
                                    "engine": "model" if USE_ML else "browser"}).encode(),
                   "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(port=8000):
    print(f"Open http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
