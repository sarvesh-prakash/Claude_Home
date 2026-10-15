"""Forward-return evaluation of flagged puts versus matched controls.

Works on the daily snapshots written by `optscreen snapshot`. Entry is at the
ask; exit is at the bid of the same contract `horizon` days later (or intrinsic
value if it expired). Controls are liquid, unflagged puts from the same
snapshot in the same expiry/moneyness bucket.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_snapshot
from .screen import Filters, screen_chain


def screen_snapshots(root="data/snapshots", filt: Filters = Filters()) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(root).glob("*/*.csv")):
        chain, u = load_snapshot(p)
        res = screen_chain(chain, u, filt)
        if not res.empty:
            frames.append(res)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def realize_returns(picks: pd.DataFrame, root="data/snapshots", horizon: int = 10,
                    slack: int = 3) -> pd.DataFrame:
    """Attach `exit_value` and `ret` using later snapshots; drops rows with no exit."""
    out = picks.copy()
    out["exit_value"] = np.nan
    cache: dict = {}
    for i, row in out.iterrows():
        target = row["asof"] + pd.Timedelta(days=horizon)
        for off in range(-slack, slack + 1):
            d = target + pd.Timedelta(days=off)
            p = Path(root) / d.strftime("%Y-%m-%d") / f"{row['ticker']}.csv"
            if not p.exists():
                continue
            if p not in cache:
                cache[p] = load_snapshot(p)
            chain, u = cache[p]
            if row["expiry"] <= d:
                out.at[i, "exit_value"] = max(row["strike"] - u.spot, 0.0)
            else:
                m = chain[(pd.to_datetime(chain["expiry"]) == row["expiry"])
                          & (chain["strike"] == row["strike"]) & (chain["kind"] == "put")]
                if not m.empty:
                    out.at[i, "exit_value"] = float(m["bid"].iloc[0])
            break
    out = out.dropna(subset=["exit_value"])
    out["ret"] = out["exit_value"] / out["ask"] - 1.0
    return out


def evaluate(realized: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Flagged vs matched-control mean return with a cluster bootstrap CI."""
    df = realized.copy()
    df["bucket"] = (pd.cut(df["dte"], [0, 30, 60, 120, 400]).astype(str) + "|"
                    + pd.cut(df["logmoney"], [-1, -0.1, -0.03, 0.03, 1]).astype(str))
    df["cluster"] = df["ticker"] + df["asof"].astype(str)
    df["ctrl_mean"] = df.groupby(["asof", "bucket"])["ret"].transform(
        lambda s: s.mean() if len(s) else np.nan)
    fl = df[df["flagged"]]
    if fl.empty:
        return {"n_flagged": 0}
    excess = fl["ret"] - fl["ctrl_mean"]
    rng = np.random.default_rng(seed)
    clusters = fl["cluster"].unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, len(clusters))
        boots.append(np.mean(np.concatenate(
            [excess[fl["cluster"] == c].values for c in pick])))
    return {
        "n_flagged": int(len(fl)),
        "n_control": int((~df["flagged"]).sum()),
        "flagged_mean_ret": float(fl["ret"].mean()),
        "control_mean_ret": float(df.loc[~df["flagged"], "ret"].mean()),
        "excess_mean": float(excess.mean()),
        "excess_ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
    }
