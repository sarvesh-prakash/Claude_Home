"""CLI: python -m optscreen {screen,snapshot,backtest} ..."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import evaluate, realize_returns, screen_snapshots
from .data import fetch_yfinance, load_chain_csv, save_snapshot
from .screen import Filters, Underlying, screen_chain

SHOW = ["ticker", "expiry", "strike", "bid", "ask", "iv", "iv_fit", "resid_z",
        "cp_spread", "edge_ratio", "spread_pct", "open_interest", "score"]


def tickers_from(args) -> list[str]:
    if args.universe:
        return [t.strip().upper() for t in Path(args.universe).read_text().split() if t.strip()]
    return [t.upper() for t in args.tickers]


def daily(a):
    """Snapshot every ticker, screen it, write reports/DATE.csv, backtest if possible."""
    from concurrent.futures import ThreadPoolExecutor

    tickers = [t.strip().upper() for t in Path(a.universe).read_text().split() if t.strip()]

    def work(t):
        try:
            chain, u = fetch_yfinance(t, a.r, a.max_expiries)
            save_snapshot(chain, u, a.root)
            return screen_chain(chain, u, Filters())
        except Exception as e:  # one bad ticker must not kill the run
            print(f"skip {t}: {e}")
            return None

    with ThreadPoolExecutor(a.workers) as ex:
        frames = [f for f in ex.map(work, tickers) if f is not None and not f.empty]
    if not frames:
        return print("nothing screened")
    res = pd.concat(frames, ignore_index=True)
    Path(a.reports).mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp.today().strftime("%Y-%m-%d")
    flagged = res[res["flagged"]].sort_values("score", ascending=False)
    flagged[SHOW + ["spot"]].to_csv(f"{a.reports}/{day}_flagged.csv", index=False)
    print(f"{len(tickers)} tickers, {len(res)} liquid puts, {len(flagged)} flagged")
    print(flagged[SHOW].head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(list(Path(a.root).glob("*"))) > 1:
        picks = screen_snapshots(a.root)
        real = realize_returns(picks, a.root, a.horizon)
        if not real.empty:
            summary = evaluate(real)
            print("backtest:", summary)
            pd.Series(summary).astype(str).to_csv(f"{a.reports}/{day}_backtest.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="optscreen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("screen", "snapshot"):
        p = sub.add_parser(name)
        p.add_argument("tickers", nargs="*")
        p.add_argument("--universe", help="file with whitespace-separated tickers (e.g. Russell 2000 list)")
        p.add_argument("--r", type=float, default=0.04)
        p.add_argument("--max-expiries", type=int, default=6)
    sub.choices["screen"].add_argument("--csv", help="screen a chain CSV instead of Yahoo")
    sub.choices["screen"].add_argument("--spot", type=float)
    sub.choices["screen"].add_argument("--top", type=int, default=20)
    d = sub.add_parser("daily", help="snapshot + screen + report (+ backtest if history exists)")
    d.add_argument("--universe", required=True)
    d.add_argument("--r", type=float, default=0.04)
    d.add_argument("--max-expiries", type=int, default=6)
    d.add_argument("--root", default="data/snapshots")
    d.add_argument("--reports", default="reports")
    d.add_argument("--horizon", type=int, default=10)
    d.add_argument("--workers", type=int, default=8)
    b = sub.add_parser("backtest")
    b.add_argument("--root", default="data/snapshots")
    b.add_argument("--horizon", type=int, default=10)
    a = ap.parse_args(argv)

    if a.cmd == "daily":
        return daily(a)

    if a.cmd == "backtest":
        res = screen_snapshots(a.root)
        if res.empty:
            return print("no snapshots found; run `snapshot` daily first")
        real = realize_returns(res, a.root, a.horizon)
        print(evaluate(real))
        return

    if a.cmd == "screen" and a.csv:
        chain = load_chain_csv(a.csv)
        u = Underlying(a.tickers[0] if a.tickers else "CSV", a.spot, pd.Timestamp.today().normalize(), r=a.r)
        jobs = [(chain, u)]
    else:
        jobs = []
        for t in tickers_from(a):
            try:
                jobs.append(fetch_yfinance(t, a.r, a.max_expiries))
            except Exception as e:
                print(f"skip {t}: {e}")

    out = []
    for chain, u in jobs:
        if a.cmd == "snapshot":
            print("saved", save_snapshot(chain, u))
        else:
            out.append(screen_chain(chain, u, Filters()))
    if a.cmd == "screen" and out:
        res = pd.concat(out, ignore_index=True)
        flagged = res[res["flagged"]].sort_values("score", ascending=False)
        print(f"{len(res)} liquid puts screened, {len(flagged)} flagged")
        print(flagged[SHOW].head(a.top).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
