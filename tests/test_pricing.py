import numpy as np
from optscreen.pricing import bs_price, crr_price, implied_vol


def test_put_call_parity():
    S, K, T, r, q, s = 50, 55, 0.5, 0.04, 0.01, 0.35
    c, p = bs_price(S, K, T, r, q, s, "call"), bs_price(S, K, T, r, q, s, "put")
    assert abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-10


def test_crr_european_converges_to_bs():
    args = (40, 42, 0.75, 0.05, 0.0, 0.3)
    assert abs(crr_price(*args, "put", american=False, n=800) - bs_price(*args, "put")) < 0.01


def test_american_put_geq_european():
    args = (40, 50, 1.0, 0.05, 0.0, 0.3)
    assert crr_price(*args, "put", True, 300) > crr_price(*args, "put", False, 300) + 0.05


def test_implied_vol_roundtrip():
    S, K, T, r, q = 30, 28, 0.4, 0.04, 0.0
    price = crr_price(S, K, T, r, q, 0.55, "put", True, 100)
    assert abs(implied_vol(price, S, K, T, r, q, "put", True, 100) - 0.55) < 1e-3


def test_implied_vol_rejects_below_intrinsic():
    assert np.isnan(implied_vol(1.0, 30, 40, 0.5, 0.04, 0.0, "put"))
