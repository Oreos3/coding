# nq-backtest

A small, deliberately conservative backtester for CME **NQ / MNQ** (Nasdaq-100) futures on 1-minute bars,
plus a full research plan: see **[BACKTEST_PLAN.md](BACKTEST_PLAN.md)**.

## Features
- Correct contract math: $20/pt NQ, $2/pt MNQ, 0.25 tick; quarterly roll calendar (`front_contract`)
- Globex session handling (18:00–17:00 ET trading day, RTH flag, DST-safe)
- Next-bar-open fills, adverse slippage, gap-through stops, stop-first when a bar is ambiguous,
  limit fills only when price trades through, per-side commissions
- Forced flatten at session end / contract roll / end of data
- Metrics: profit factor, expectancy, max DD, Sharpe/Sortino on daily P&L, Monte Carlo drawdown
- Rolling walk-forward optimisation with stitched out-of-sample results
- Example strategies: Opening Range Breakout (`orb`), EMA trend with ATR stop (`ema`)
- Synthetic random-walk data generator for pipeline tests and a "no-edge" benchmark

## Quick start
```bash
cd nq-backtest
pip install -e ".[dev]"      # or: pip install pandas numpy pytest
pytest -q

# pipeline smoke test on random-walk data (results are meaningless by design)
python scripts/run_backtest.py --synthetic --strategy orb

# real data
python scripts/run_backtest.py --csv data/nq_1m.csv --start 2019-01-01 --end 2024-12-31 \
    --strategy orb --param or_minutes=30 --param target_r=2.0 --slippage-ticks 1
python scripts/run_backtest.py --csv data/nq_1m.csv --strategy orb --walk-forward
python scripts/run_backtest.py --csv data/nq_1m.csv --strategy orb --micro --qty 2
```
Outputs (trades, equity curve, walk-forward windows) go to `output/`.

## Data format
CSV with `timestamp,open,high,low,close,volume[,contract]`; timestamps are bar **start** time.
Include a UTC offset in the timestamp, or pass `--source-tz` (default UTC) for naive timestamps.
Data files are git-ignored; see `data/README.md`.

## Writing a strategy
```python
from nqbt import Signals, add_session_columns, load_csv, run_backtest, summary

def my_strategy(df, lookback=20):
    pos = ...                      # pd.Series of -1/0/+1, decided at each bar's CLOSE
    return Signals(position=pos, stop_points=25.0, target_points=50.0)  # scalars or Series

df = add_session_columns(load_csv("data/nq_1m.csv"))
print(summary(run_backtest(df, my_strategy(df))))
```

## Layout
```
nqbt/contract.py     specs + roll calendar
nqbt/data.py         load / validate / sessionize / synthetic data
nqbt/engine.py       execution model and backtest loop
nqbt/strategies.py   example strategies
nqbt/metrics.py      stats, Monte Carlo
nqbt/walkforward.py  rolling walk-forward
scripts/run_backtest.py  CLI
tests/               engine correctness and no-lookahead tests
```

*Not financial advice. Futures trading involves substantial risk of loss.*
