#!/usr/bin/env python3
"""Run a backtest or walk-forward on NQ/MNQ 1-minute bars.

Examples
--------
  python scripts/run_backtest.py --synthetic --strategy orb
  python scripts/run_backtest.py --csv data/nq_1m.csv --strategy orb --param or_minutes=15 --param target_r=1.5
  python scripts/run_backtest.py --csv data/nq_1m.csv --strategy orb --walk-forward
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nqbt import (MNQ, NQ, STRATEGIES, CostModel, add_session_columns, exit_breakdown,  # noqa: E402
                  format_summary, load_csv, monte_carlo_drawdown, run_backtest, summary,
                  synthetic_minutes, validate, walk_forward)

DEFAULT_GRIDS = {
    "orb": {"or_minutes": [15, 30, 60], "target_r": [1.0, 1.5, 2.0, 3.0]},
    "ema": {"fast": [10, 20], "slow": [60, 120], "atr_stop": [2.0, 3.0]},
}


def parse_params(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="1-minute OHLCV CSV (timestamp,open,high,low,close,volume[,contract])")
    src.add_argument("--synthetic", action="store_true", help="use random-walk data (pipeline test only)")
    ap.add_argument("--source-tz", default="UTC", help="tz of naive CSV timestamps (default UTC)")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="orb")
    ap.add_argument("--param", action="append", default=[], help="strategy param, e.g. or_minutes=15")
    ap.add_argument("--micro", action="store_true", help="trade MNQ instead of NQ")
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--commission", type=float, default=None, help="$/contract/side")
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--capital", type=float, default=50_000)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--train-sessions", type=int, default=120)
    ap.add_argument("--test-sessions", type=int, default=20)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    spec = MNQ if args.micro else NQ
    commission = args.commission if args.commission is not None else (0.85 if args.micro else 2.50)
    costs = CostModel(commission_per_side=commission, slippage_ticks=args.slippage_ticks)

    if args.synthetic:
        df = synthetic_minutes(args.start, args.end)
        print("WARNING: synthetic random-walk data -- results are meaningless, this only tests the pipeline.\n")
    else:
        df = load_csv(args.csv, source_tz=args.source_tz)
        df = df[(df.index >= args.start) & (df.index < f"{args.end} 23:59:59")] if args.start else df
    for issue in validate(df, spec):
        print(f"DATA WARNING: {issue}")
    df = add_session_columns(df)
    print(f"{len(df):,} bars, {df['session_date'].nunique()} sessions, "
          f"{df.index[0]} -> {df.index[-1]}\n")

    strat = STRATEGIES[args.strategy]
    bt_kwargs = dict(spec=spec, costs=costs, qty=args.qty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.walk_forward:
        wf = walk_forward(df, strat, DEFAULT_GRIDS[args.strategy], train_sessions=args.train_sessions,
                          test_sessions=args.test_sessions, **bt_kwargs)
        result = wf.oos
        print("Walk-forward windows:")
        print(wf.windows.to_string(index=False))
        wf.windows.to_csv(out / f"{args.strategy}_wf_windows.csv", index=False)
        print("\nOUT-OF-SAMPLE (stitched):")
    else:
        params = parse_params(args.param)
        result = run_backtest(df, strat(df, **params), **bt_kwargs)
        print(f"Params: {params or 'defaults'}")

    print(format_summary(summary(result, args.capital)))
    print("\nExits:\n", exit_breakdown(result).to_string() if not result.trades.empty else "none")
    mc = monte_carlo_drawdown(result)
    if mc:
        print("\nMonte Carlo (trade bootstrap):")
        print(format_summary(mc))
    trades_path = out / f"{args.strategy}_trades.csv"
    result.trades.to_csv(trades_path, index=False)
    result.equity(args.capital).to_csv(out / f"{args.strategy}_equity.csv", header=["equity"])
    print(f"\nWrote {trades_path} and equity curve to {out}/")


if __name__ == "__main__":
    main()
