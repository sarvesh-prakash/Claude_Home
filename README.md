# Claude_Home

## AI Detector

A dependency-free, Turnitin-style AI-writing detector (stylometric heuristics, no ML model).

```
python -m ai_detector essay.txt      # CLI
python -m ai_detector --serve 8000   # web UI at http://localhost:8000
python -m pytest tests               # tests
```

Signals: sentence-length uniformity, lexical profile, stock AI phrases, n-gram repetition, informality.
Outputs an overall 0-100 score plus per-sentence highlighting.

**Limitations:** this is not Turnitin. Commercial detectors use trained neural classifiers; this uses
hand-tuned heuristics and is far less accurate. No detector is reliable enough to prove misconduct.
