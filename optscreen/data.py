"""Data loading: live chains from yfinance, CSV chains, and daily snapshots.

yfinance has no option history, so `snapshot` saves each day's chain to
data/snapshots/YYYY-MM-DD/TICKER.csv. Run it daily to build your own history
for backtesting. For serious research use ORATS / OptionMetrics instead and
adapt `load_chain_csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .screen import Underlying

CHAIN_COLS = ["expiry", "strike", "kind", "bid", "ask", "volume", "open_interest"]


def load_chain_csv(path, colmap: dict | None = None) -> pd.DataFrame:
    """Load a chain CSV; `colmap` renames vendor columns to CHAIN_COLS."""
    df = pd.read_csv(path)
    if colmap:
        df = df.rename(columns=colmap)
    missing = [c for c in CHAIN_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"chain is missing columns: {missing}")
    df["kind"] = df["kind"].str.lower()
    df[["volume", "open_interest"]] = df[["volume", "open_interest"]].fillna(0)
    return df


def fetch_yfinance(ticker: str, r: float = 0.04, max_expiries: int = 6):
    """Fetch (chain, Underlying) from Yahoo. Live data only, no history."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    hist = t.history(period="3mo")
    if hist.empty:
        raise ValueError(f"no price history for {ticker}")
    spot = float(hist["Close"].iloc[-1])
    rets = np.log(hist["Close"]).diff().dropna()
    rv = float(rets.tail(30).std() * np.sqrt(252)) if len(rets) >= 20 else None
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    q = float(info.get("dividendYield") or 0.0)
    q = q / 100 if q > 1 else q  # Yahoo sometimes reports percent
    frames = []
    for exp in t.options[:max_expiries]:
        oc = t.option_chain(exp)
        for kind, tbl in (("call", oc.calls), ("put", oc.puts)):
            f = tbl.rename(columns={"openInterest": "open_interest"})[
                ["strike", "bid", "ask", "volume", "open_interest"]].copy()
            f["kind"], f["expiry"] = kind, exp
            frames.append(f)
    chain = pd.concat(frames, ignore_index=True).fillna({"volume": 0, "open_interest": 0})
    earn = None
    try:
        cal = t.calendar
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if d:
            earn = pd.Timestamp(d[0])
    except Exception:
        pass
    u = Underlying(ticker, spot, pd.Timestamp.today().normalize(), r=r, q=q,
                   rv=rv, next_earnings=earn)
    return chain[CHAIN_COLS], u


def save_snapshot(chain: pd.DataFrame, u: Underlying, root="data/snapshots") -> Path:
    d = Path(root) / pd.Timestamp(u.asof).strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    out = chain.copy()
    out["spot"], out["r"], out["q"] = u.spot, u.r, u.q
    out["rv"] = u.rv if u.rv is not None else np.nan
    p = d / f"{u.ticker}.csv"
    out.to_csv(p, index=False)
    return p


def load_snapshot(path) -> tuple[pd.DataFrame, Underlying]:
    path = Path(path)
    df = pd.read_csv(path)
    asof = pd.Timestamp(path.parent.name)
    rv = df["rv"].iloc[0]
    u = Underlying(path.stem, float(df["spot"].iloc[0]), asof, r=float(df["r"].iloc[0]),
                   q=float(df["q"].iloc[0]), rv=None if pd.isna(rv) else float(rv))
    return df[CHAIN_COLS], u
