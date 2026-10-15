# Claude_Home

## optscreen: cheap-put screener for Russell 2000 names

Flags puts whose implied vol sits below the underlying's own fitted vol surface,
nets the edge against the ask, and gives you a way to test whether the flags
make money out of sample. It does not claim an edge exists; see the caveats.

### How it works
1. `pricing.py`: Black-Scholes, Cox-Ross-Rubinstein American binomial (dividend yield + borrow cost), implied-vol solver.
2. `surface.py`: per-expiry SVI fit with outlier-clipped refit, so mispriced quotes don't drag the surface toward themselves. Gives `resid_z` (positive = IV below surface).
3. `screen.py`: signals combined into `score`:
   - `resid_z`: market IV below the SVI slice
   - `cp_spread`: call IV minus put IV at the same strike (Cremers-Weinbaum parity signal)
   - `ivrv_z`: put IV versus realised vol
   A put is `flagged` only if it passes liquidity filters (spread, open interest, volume, DTE, moneyness, no earnings inside the life) **and** the fitted-surface price exceeds the ask by `min_edge_ratio` (default 5%).
4. `backtest.py`: entry at the ask, exit at the later bid (or intrinsic if expired); flagged puts are compared with unflagged liquid puts in the same date/expiry/moneyness bucket, with a cluster-bootstrap CI.

### Use
```
pip install -r requirements.txt
python -m optscreen screen IWM AAPL                 # live chains via Yahoo
python -m optscreen screen --universe r2000.txt     # your Russell 2000 ticker list
python -m optscreen snapshot --universe r2000.txt   # run daily to build history
python -m optscreen backtest --horizon 10           # after weeks of snapshots
python -m pytest
```
Chain CSVs need: `expiry, strike, kind, bid, ask, volume, open_interest` (`load_chain_csv` takes a column map for ORATS/OptionMetrics exports).

### Status and caveats
- The test suite checks pricing, SVI recovery, that the screener flags injected cheap puts in a synthetic chain, and the backtest plumbing. It does **not** show the signals make money; that needs real data.
- Yahoo gives live chains only and quotes are delayed or stale, so treat live output as a prototype. Backtest quality depends on your own snapshots or a paid source (ORATS, OptionMetrics).
- The Russell 2000 constituent list is not included; supply it (FTSE Russell or CRSP point-in-time to avoid survivorship bias).
- Expect small-cap spreads to eat most apparent edge; the `min_edge_ratio`, spread and open-interest filters exist for that reason.
