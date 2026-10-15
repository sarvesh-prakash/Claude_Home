"""Option pricing: Black-Scholes, CRR American binomial, implied vol.

`q` is the continuous dividend yield. Borrow cost for hard-to-borrow names
acts like extra yield, so pass q + borrow as `q`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S, K, T, r, q, sigma, kind="put"):
    """European Black-Scholes price. kind is 'put' or 'call'."""
    if T <= 0 or sigma <= 0:
        fwd = S * np.exp(-q * T) - K * np.exp(-r * T)
        return max(-fwd, 0.0) if kind == "put" else max(fwd, 0.0)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, q, sigma):
    """Black-Scholes vega (price change per 1.00 change in vol)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def crr_price(S, K, T, r, q, sigma, kind="put", american=True, n=200):
    """Cox-Ross-Rubinstein binomial price (American by default), vectorised."""
    if T <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:  # extreme inputs; fall back to European BS
        return bs_price(S, K, T, r, q, sigma, kind)
    sign = -1.0 if kind == "put" else 1.0
    j = np.arange(n + 1)
    V = np.maximum(sign * (S * u ** (n - 2 * j) - K), 0.0)
    for i in range(n - 1, -1, -1):
        V = disc * (p * V[: i + 1] + (1.0 - p) * V[1 : i + 2])
        if american:
            nodes = S * u ** (i - 2 * np.arange(i + 1))
            V = np.maximum(V, sign * (nodes - K))
    return float(V[0])


def implied_vol(price, S, K, T, r, q, kind="put", american=True, n=100,
                lo=0.01, hi=5.0):
    """Implied vol from a price; NaN if the price is outside no-arbitrage bounds."""
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    if american and price < intrinsic - 1e-9:
        return np.nan

    def f(sig):
        return crr_price(S, K, T, r, q, sig, kind, american, n) - price

    try:
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-5, maxiter=60)
    except (ValueError, RuntimeError):
        return np.nan
