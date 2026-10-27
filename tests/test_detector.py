from ai_detector import analyze

AI = ("In today's rapidly evolving digital landscape, it is important to note that technology plays a crucial role "
      "in shaping society. Furthermore, artificial intelligence serves as a pivotal tool to foster innovation across "
      "a wide range of industries. Moreover, organizations must leverage robust and comprehensive solutions to navigate "
      "the multifaceted challenges of the modern era. In conclusion, the integration of these technologies is a testament "
      "to human ingenuity and will continue to underscore the importance of seamless collaboration.")
HUMAN = ("I missed the bus again this morning. Honestly? Third time this week! So I walked, which took forever, and "
         "my shoes — the good ones, ugh — got soaked because of course it started pouring halfway there. When I finally "
         "showed up my boss just laughed. Can't say I blame him. Anyway, I'm gonna buy an umbrella today, lol, and maybe "
         "set two alarms instead of one because I clearly can't be trusted with just the one.")

def test_ai_scores_higher_than_human():
    assert analyze(AI).score > analyze(HUMAN).score + 30

def test_labels():
    assert analyze(AI).label == "Likely AI-generated"
    assert analyze(HUMAN).label == "Likely human-written"

def test_short_text():
    assert analyze("Too short.").score == 0.0

def test_model_scorer_runs_if_installed():
    import pytest
    from ai_detector import ml
    if not ml.available():
        pytest.skip("torch/transformers not installed")
