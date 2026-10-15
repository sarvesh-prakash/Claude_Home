"""Snapshot -> screen -> realised-return -> evaluate path on synthetic data."""
import pandas as pd

from optscreen.backtest import evaluate, realize_returns, screen_snapshots
from optscreen.data import save_snapshot
from optscreen.screen import Underlying
from test_screen import ASOF, R, SPOT, make_chain


def test_backtest_roundtrip(tmp_path):
    root = tmp_path / "snaps"
    cheap = {(60, 35.0)}
    save_snapshot(make_chain(cheap, seed=1), Underlying("TEST", SPOT, ASOF, r=R), root)
    later = ASOF + pd.Timedelta(days=10)
    save_snapshot(make_chain(set(), seed=2), Underlying("TEST", SPOT, later, r=R), root)

    picks = screen_snapshots(root)
    picks = picks[picks["asof"] == ASOF]
    assert picks["flagged"].any()
    real = realize_returns(picks, root, horizon=10)
    assert len(real) > 0 and real["ret"].notna().all()
    out = evaluate(real, n_boot=50)
    assert out["n_flagged"] >= 1
