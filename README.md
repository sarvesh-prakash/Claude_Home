# Claude_Home

## AI Detector

A dependency-free, Turnitin-style AI-writing detector (stylometric heuristics, no ML model).

```
python -m ai_detector essay.txt      # CLI
python -m ai_detector --serve 8000   # dashboard at http://localhost:8000
pip install -r requirements-ml.txt   # optional: enables the GPT-2 perplexity scorer
python -m pytest tests               # tests
```

Signals: sentence-length uniformity, lexical profile, stock AI phrases, n-gram repetition, informality.
Outputs an overall 0-100 score plus per-sentence highlighting.

**Limitations:** this is not Turnitin. Commercial detectors use trained neural classifiers; this uses
hand-tuned heuristics and is far less accurate. No detector is reliable enough to prove misconduct.

### Two engines
- **Heuristics** (always on): `ai_detector/detector.py`, also ported to JS inside `web/index.html`, so the dashboard works as a static page with no backend.
- **Language model** (optional): `ai_detector/ml.py` scores perplexity, top-10 token share and per-sentence spread under GPT-2 and blends it 60/40 with the heuristics. The server uses it automatically when `torch` and `transformers` are installed. The model downloads from Hugging Face on first run. The thresholds are hand-calibrated, not trained on a labelled dataset, and the GPT-2 path has only been smoke-tested with a tiny random model, not validated on real essays.
