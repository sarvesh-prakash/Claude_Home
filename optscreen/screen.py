"""Screening pipeline: one underlying's option chain -> ranked cheap puts.

Chain columns: expiry, strike, kind ('call'/'put'), bid, ask, volume, open_interest.
Signals combined into `score` (all oriented so positive = put looks cheap):
  resid_z      market IV below the underlying's SVI slice
  cp_spread    call IV - put IV at the same strike (Cremers-Weinbaum parity signal)
  ivrv_z       -log(put IV / realised vol)
Candidates must also clear liquidity filters and show positive edge net of the
ask (we assume you pay the ask, not the mid).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .pricing import crr_price, implied_vol
from .surface import fit_residuals


@dataclass
class Underlying:
    ticker: str
    spot: float
    asof: pd.Timestamp
    r: float = 0.04          # risk-free rate, continuous
    q: float = 0.0           # dividend yield, continuous
    borrow: float = 0.0      # annualised borrow cost; acts like extra yield
    rv: Optional[float] = None            # realised vol (annualised), e.g. 30d
    next_earnings: Optional[pd.Timestamp] = None


@dataclass
class Filters:
    max_spread_pct: float = 0.25
    min_open_interest: int = 100
    min_volume: int = 10
    min_dte: int = 14
    max_dte: int = 200
    max_abs_logmoney: float = 0.30
    skip_earnings: bool = True
    min_edge_ratio: float = 0.05   # (fit price - ask) / ask


def _prep(chain: pd.DataFrame, u: Underlying, n_steps: int) -> pd.DataFrame:
    df = chain.copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["T"] = (df["expiry"] - pd.Timestamp(u.asof)).dt.days / 365.0
    df["dte"] = (df["expiry"] - pd.Timestamp(u.asof)).dt.days
    df = df[(df["T"] > 0) & (df["bid"] > 0) & (df["ask"] >= df["bid"])].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    qe = u.q + u.borrow
    df["F"] = u.spot * np.exp((u.r - qe) * df["T"])
    df["k"] = np.log(df["strike"] / df["F"])
    df["logmoney"] = np.log(df["strike"] / u.spot)
    df["iv"] = [
        implied_vol(m, u.spot, K, T, u.r, qe, kind, True, n_steps)
        for m, K, T, kind in zip(df["mid"], df["strike"], df["T"], df["kind"])
    ]
    return df.dropna(subset=["iv"]).reset_index(drop=True)


def screen_chain(chain: pd.DataFrame, u: Underlying, filt: Filters = Filters(),
                 n_steps: int = 100) -> pd.DataFrame:
    """Return every liquid put with signals, `edge_ratio`, `score` and `flagged`."""
    df = _prep(chain, u, n_steps)
    if df.empty:
        return df
    qe = u.q + u.borrow

    # liquidity-weighted surface fit using tight-spread quotes
    df["fit_weight"] = 1.0 / (1.0 + 10 * df["spread_pct"])
    df = fit_residuals(df)

    # parity signal: call IV - put IV at identical (expiry, strike)
    piv = df.pivot_table(index=["expiry", "strike"], columns="kind", values="iv")
    if {"call", "put"} <= set(piv.columns):
        cp = (piv["call"] - piv["put"]).rename("cp_spread").reset_index()
    else:
        cp = pd.DataFrame(columns=["expiry", "strike", "cp_spread"])

    puts = df[df["kind"] == "put"].merge(cp, on=["expiry", "strike"], how="left")
    if puts.empty:
        return puts

    puts["ivrv_z"] = -np.log(puts["iv"] / u.rv) if u.rv else np.nan
    puts["fit_price"] = [
        crr_price(u.spot, K, T, u.r, qe, f, "put", True, n_steps) if np.isfinite(f) else np.nan
        for K, T, f in zip(puts["strike"], puts["T"], puts["iv_fit"])
    ]
    puts["edge"] = puts["fit_price"] - puts["ask"]
    puts["edge_ratio"] = puts["edge"] / puts["ask"]

    # liquidity / sanity filters
    ok = (
        (puts["spread_pct"] <= filt.max_spread_pct)
        & (puts["open_interest"] >= filt.min_open_interest)
        & (puts["volume"] >= filt.min_volume)
        & puts["dte"].between(filt.min_dte, filt.max_dte)
        & (puts["logmoney"].abs() <= filt.max_abs_logmoney)
    )
    if filt.skip_earnings and u.next_earnings is not None:
        ok &= ~(puts["expiry"] >= pd.Timestamp(u.next_earnings))
    liquid = puts[ok].copy()

    # composite score: mean of available clipped signals (cp_spread scaled to vol-point z)
    comps = pd.DataFrame({
        "resid_z": liquid["resid_z"].clip(-4, 4),
        "cp": (liquid["cp_spread"] / 0.03).clip(-4, 4),
        "ivrv": (liquid["ivrv_z"] / 0.25).clip(-4, 4),
    })
    liquid["score"] = comps.mean(axis=1, skipna=True)
    liquid["flagged"] = (liquid["edge_ratio"] >= filt.min_edge_ratio) & (liquid["score"] > 0)
    liquid.insert(0, "ticker", u.ticker)
    liquid.insert(1, "asof", pd.Timestamp(u.asof))
    liquid["spot"] = u.spot
    return liquid.sort_values("score", ascending=False).reset_index(drop=True)
