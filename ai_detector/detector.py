"""Heuristic AI-text detector (stylometric, no external models).

Scores text 0-100 for "likely AI-generated". It combines signals that tend to
differ between LLM output and human writing: low burstiness (uniform sentence
length), low lexical diversity, stock "AI" phrases and transitions, repeated
n-grams, and absence of informal markers. These are weak signals: results are
probabilistic and must not be used as proof of misconduct.
"""
import math
import re
import statistics
from dataclasses import dataclass, field

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
WORD = re.compile(r"[A-Za-z']+")

AI_PHRASES = [
    "delve", "tapestry", "testament to", "in today's", "it is important to note",
    "it's important to note", "it is worth noting", "plays a crucial role",
    "plays a vital role", "in conclusion", "in summary", "furthermore",
    "moreover", "additionally", "overall,", "landscape", "multifaceted",
    "leverage", "foster", "underscore", "pivotal", "navigate the", "realm of",
    "ever-evolving", "rapidly evolving", "holistic", "seamless", "robust",
    "comprehensive", "game-changer", "shed light", "a wide range of",
    "not only", "paramount", "intricate", "embark", "unlock the",
]
INFORMAL = [
    "i ", "i'm", "i've", "my ", "lol", "honestly", "kinda", "gonna", "yeah",
    "!", "?", "…", "—", "(", "don't", "can't", "won't",
]


@dataclass
class Result:
    score: float
    label: str
    signals: dict = field(default_factory=dict)
    sentences: list = field(default_factory=list)  # (text, score)
    word_count: int = 0


def split_sentences(text):
    parts = []
    for para in re.split(r"\n+", text.strip()):
        if para.strip():
            parts.extend(s.strip() for s in SENT_SPLIT.split(para.strip()) if s.strip())
    return parts


def words(text):
    return [w.lower() for w in WORD.findall(text)]


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _burstiness(sents):
    lens = [len(words(s)) for s in sents if words(s)]
    if len(lens) < 3:
        return 0.5
    cv = statistics.pstdev(lens) / (statistics.mean(lens) or 1)
    # humans typically cv ~0.5-0.8; LLMs ~0.2-0.4
    return _clamp(1 - (cv - 0.2) / 0.5)


def _lexical(ws):
    if len(ws) < 30:
        return 0.5
    window = 100
    chunks = [ws[i:i + window] for i in range(0, max(1, len(ws) - window + 1), window // 2)]
    ttr = statistics.mean(len(set(c)) / len(c) for c in chunks)
    # moderate TTR (~0.6-0.7) is typical for LLM text; very high or low skews human
    return _clamp(1 - abs(ttr - 0.65) / 0.2) * 0.6


def _phrases(text, nwords):
    low = text.lower()
    hits = sum(low.count(p) for p in AI_PHRASES)
    per_100 = hits / max(nwords, 1) * 100
    return _clamp(per_100 / 3.0), hits


def _repetition(ws):
    if len(ws) < 40:
        return 0.0
    tri = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    repeated = len(tri) - len(set(tri))
    return _clamp(repeated / len(tri) * 8)


def _informality(text):
    low = text.lower() + " "
    hits = sum(low.count(m) for m in INFORMAL)
    per_100 = hits / max(len(words(text)), 1) * 100
    return _clamp(per_100 / 6.0)


def _sentence_score(s):
    ws = words(s)
    if len(ws) < 4:
        return 0.3
    p, _ = _phrases(s, len(ws))
    inf = _informality(s)
    length_term = _clamp(1 - abs(len(ws) - 22) / 22)  # LLMs favour ~15-30 words
    return _clamp(0.15 + 0.45 * p + 0.3 * length_term - 0.35 * inf + 0.2)


def analyze(text):
    ws = words(text)
    sents = split_sentences(text)
    if len(ws) < 40:
        return Result(0.0, "Too short (need 40+ words)", {}, [(s, 0.0) for s in sents], len(ws))

    burst = _burstiness(sents)
    lex = _lexical(ws)
    phr, hits = _phrases(text, len(ws))
    rep = _repetition(ws)
    inf = _informality(text)

    raw = 0.30 * burst + 0.15 * lex + 0.30 * phr + 0.10 * rep + 0.15 * (1 - inf)
    # squash for a more decisive spread
    score = 100 / (1 + math.exp(-9 * (raw - 0.42)))
    label = ("Likely AI-generated" if score >= 70
             else "Possibly AI-assisted" if score >= 40
             else "Likely human-written")
    return Result(
        round(score, 1), label,
        {"uniform_sentence_length": round(burst, 2), "lexical_profile": round(lex, 2),
         "ai_phrase_density": round(phr, 2), "ai_phrase_hits": hits,
         "ngram_repetition": round(rep, 2), "informality": round(inf, 2)},
        [(s, round(_sentence_score(s) * 100, 1)) for s in sents],
        len(ws),
    )
