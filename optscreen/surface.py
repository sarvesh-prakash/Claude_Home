"""Per-expiry SVI vol-surface fit.

Raw SVI total variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + s^2)),
with k = ln(K/F). Options that sit far off the fitted slice are candidates
for "cheap" (IV below fit) or "rich" (IV above fit).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k, a, b, rho, m, s):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def _fit_once(k, tv, w):
    scale = tv.std() + 1e-9

    def resid(x):
        a, b, rho, m, s = x
        return w * (svi_total_variance(k, a, b, rho, m, s) - tv) / scale

    x0 = [max(tv.min() * 0.5, 1e-4), 0.1 * tv.max() + 1e-3, -0.3, float(np.median(k)), 0.1]
    lb = [-1.0, 1e-6, -0.999, k.min() - 1.0, 1e-3]
    ub = [max(tv.max(), 0.01) * 2, 10.0, 0.999, k.max() + 1.0, 2.0]
    try:
        sol = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.3,
                            x_scale="jac", max_nfev=2000)
    except ValueError:
        return None
    return sol.x if np.all(np.isfinite(sol.x)) else None


def fit_svi(k, iv, T, weights=None, n_clip=3.0):
    """Fit raw SVI to one expiry slice. Returns params array or None.

    Two passes: fit, drop points more than `n_clip` robust sigmas off the curve,
    refit. This keeps mispriced quotes (the ones we want to flag) from pulling
    the surface toward themselves.
    """
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    if ok.sum() < 6 or T <= 0:
        return None
    w = np.ones(len(k)) if weights is None else np.asarray(weights, float)
    k, iv, w = k[ok], iv[ok], w[ok]
    tv = iv**2 * T
    p = _fit_once(k, tv, w)
    if p is None:
        return None
    r = svi_total_variance(k, *p) - tv
    mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
    keep = np.abs(r - np.median(r)) <= n_clip * mad
    if keep.sum() >= 6 and not keep.all():
        p2 = _fit_once(k[keep], tv[keep], w[keep])
        if p2 is not None:
            return p2
    return p


def svi_iv(k, T, params):
    tv = np.maximum(svi_total_variance(np.asarray(k, float), *params), 1e-10)
    return np.sqrt(tv / T)


def fit_residuals(df):
    """Add `iv_fit` and `resid_z` columns to a one-underlying chain.

    Expects columns expiry, T, k, iv (and optional fit_weight). Residual
    z-scores are computed per expiry with a robust (MAD) scale. A positive
    resid_z means the market IV sits BELOW the surface, i.e. the option looks cheap.
    """
    out = df.copy()
    out["iv_fit"] = np.nan
    out["resid_z"] = np.nan
    for _, g in out.groupby("expiry"):
        T = g["T"].iloc[0]
        wts = g["fit_weight"].values if "fit_weight" in g else None
        params = fit_svi(g["k"].values, g["iv"].values, T, wts)
        if params is None:
            continue
        fit = svi_iv(g["k"].values, T, params)
        diff = fit - g["iv"].values
        mad = np.nanmedian(np.abs(diff - np.nanmedian(diff))) * 1.4826
        out.loc[g.index, "iv_fit"] = fit
        out.loc[g.index, "resid_z"] = diff / max(mad, 0.005)
    return out
