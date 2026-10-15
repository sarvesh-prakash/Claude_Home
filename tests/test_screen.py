"""Synthetic end-to-end test: inject cheap puts into an SVI-generated chain and
check the screener flags them while leaving fairly priced puts alone."""
import numpy as np
import pandas as pd
import pytest

from optscreen.pricing import crr_price
from optscreen.screen import Filters, Underlying, screen_chain
from optscreen.surface import fit_svi, svi_iv, svi_total_variance

SPOT, R, ASOF = 40.0, 0.04, pd.Timestamp("2026-01-02")
SVI = dict(a=0.01, b=0.12, rho=-0.5, m=0.0, s=0.15)


def make_chain(cheap=(), seed=0):
    """cheap: iterable of (expiry_days, strike) puts to underprice by 15 vol pts."""
    rng = np.random.default_rng(seed)
    rows = []
    for days in (30, 60, 90):
        T = days / 365
        F = SPOT * np.exp(R * T)
        for K in np.arange(30, 52.5, 2.5):
            k = np.log(K / F)
            iv = float(svi_iv(k, T, [SVI['a'] * T / 0.25, SVI['b'] * T / 0.25,
                                       SVI['rho'], SVI['m'], SVI['s']]))
            for kind in ("call", "put"):
                sig = iv + rng.normal(0, 0.003)
                if kind == "put" and (days, K) in cheap:
                    sig -= 0.15
                px = crr_price(SPOT, K, T, R, 0.0, sig, kind, True, 100)
                half = max(0.02, 0.01 * px)
                rows.append(dict(expiry=ASOF + pd.Timedelta(days=days), strike=K, kind=kind,
                                 bid=max(px - half, 0.01), ask=px + half,
                                 volume=500, open_interest=1000))
    return pd.DataFrame(rows)


def test_svi_fit_recovers_surface():
    k = np.linspace(-0.4, 0.4, 25)
    T = 0.25
    iv = svi_iv(k, T, list(SVI.values()))
    p = fit_svi(k, iv, T)
    assert p is not None
    assert np.max(np.abs(svi_iv(k, T, p) - iv)) < 0.005


def test_screen_flags_injected_cheap_puts():
    cheap = {(60, 35.0), (90, 40.0)}
    chain = make_chain(cheap)
    u = Underlying("TEST", SPOT, ASOF, r=R)
    res = screen_chain(chain, u, Filters(skip_earnings=False))
    flagged = res[res["flagged"]]
    got = {((e - ASOF).days, K) for e, K in zip(flagged["expiry"], flagged["strike"])}
    assert cheap <= got, f"missed {cheap - got}"
    assert len(got - cheap) <= 2, f"too many false positives: {got - cheap}"


def test_clean_chain_has_few_flags():
    res = screen_chain(make_chain(), Underlying("TEST", SPOT, ASOF, r=R), Filters(skip_earnings=False))
    assert res["flagged"].sum() <= 3
