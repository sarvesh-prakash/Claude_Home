"""CLI: python -m ai_detector [file|-]   |   python -m ai_detector --serve [port]"""
import sys
from .detector import analyze

def main(argv):
    if argv and argv[0] == "--serve":
        from .server import serve
        return serve(int(argv[1]) if len(argv) > 1 else 8000)
    text = sys.stdin.read() if not argv or argv[0] == "-" else open(argv[0], encoding="utf-8").read()
    r = analyze(text)
    print(f"AI likelihood: {r.score}%  ->  {r.label}  ({r.word_count} words)")
    for k, v in r.signals.items():
        print(f"  {k}: {v}")
    print("\nSentence scores:")
    for s, sc in r.sentences:
        print(f"  [{sc:5.1f}] {s[:100]}")

if __name__ == "__main__":
    main(sys.argv[1:])
