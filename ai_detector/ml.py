"""Language-model detector: perplexity + burstiness under a causal LM (default GPT-2).

LLM text is statistically "unsurprising" to another LM (low perplexity) and its
per-sentence perplexity varies little (low burstiness). Requires `torch` and
`transformers`; the model is downloaded from the Hugging Face hub on first use.
"""
import math
import statistics

from .detector import analyze as heuristic_analyze, split_sentences

_cache = {}


def available():
    try:
        import torch, transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load(model_name="gpt2"):
    if model_name not in _cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        _cache[model_name] = (tok, model, torch)
    return _cache[model_name]


def _token_logprobs(text, tok, model, torch, max_len=512):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
    if ids.shape[1] < 2:
        return []
    with torch.no_grad():
        logits = model(ids).logits[0, :-1]
    lp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    chosen = lp[torch.arange(len(tgt)), tgt]
    ranks = (lp > chosen.unsqueeze(1)).sum(1)
    return list(zip(chosen.tolist(), ranks.tolist()))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def model_scores(text, model_name="gpt2"):
    """Returns (score 0-100, signals dict, [(sentence, score)])."""
    tok, model, torch = load(model_name)
    sents = split_sentences(text)
    per_sent = []
    all_lp = []
    for s in sents:
        toks = _token_logprobs(s, tok, model, torch)
        if len(toks) >= 4:
            lps = [t[0] for t in toks]
            per_sent.append((s, -statistics.mean(lps)))  # mean NLL
            all_lp.extend(toks)
        else:
            per_sent.append((s, None))
    if len(all_lp) < 30:
        return 0.0, {}, [(s, 0.0) for s in sents]

    nll = -statistics.mean(t[0] for t in all_lp)
    ppl = math.exp(nll)
    top10 = sum(1 for t in all_lp if t[1] < 10) / len(all_lp)
    vals = [v for _, v in per_sent if v is not None]
    burst = statistics.pstdev(vals) if len(vals) > 2 else 1.0

    # Calibrated against typical GPT-2 behaviour: AI text ppl ~10-30, human ~40-120.
    s_ppl = _sigmoid((math.log(45) - math.log(ppl)) * 2.5)
    s_top = _sigmoid((top10 - 0.80) * 18)
    s_burst = _sigmoid((0.85 - burst) * 4)
    score = 100 * (0.5 * s_ppl + 0.3 * s_top + 0.2 * s_burst)

    sent_scores = [(s, round(100 * _sigmoid((math.log(45) - v) * 2.5), 1) if v is not None else 0.0)
                   for s, v in per_sent]
    signals = {"perplexity": round(ppl, 1), "top10_token_share": round(top10, 2),
               "sentence_ppl_spread": round(burst, 2)}
    return round(score, 1), signals, sent_scores


def analyze_combined(text, model_name="gpt2", weight_ml=0.6):
    """Blend language-model and heuristic scores into one Result."""
    base = heuristic_analyze(text)
    if base.word_count < 40:
        return base
    ml_score, ml_sig, ml_sents = model_scores(text, model_name)
    base.score = round(weight_ml * ml_score + (1 - weight_ml) * base.score, 1)
    base.label = ("Likely AI-generated" if base.score >= 70
                  else "Possibly AI-assisted" if base.score >= 40 else "Likely human-written")
    base.signals = {**ml_sig, **base.signals, "model_score": ml_score}
    if len(ml_sents) == len(base.sentences):
        base.sentences = [(s, round(weight_ml * m + (1 - weight_ml) * h, 1))
                          for (s, h), (_, m) in zip(base.sentences, ml_sents)]
    return base
