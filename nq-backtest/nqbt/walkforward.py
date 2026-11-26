"""Rolling walk-forward optimisation (anchored by session, not by bar)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .contract import NQ
from .data import add_session_columns
from .engine import BacktestResult, Signals, run_backtest
from .metrics import summary


@dataclass
class WalkForwardResult:
    oos: BacktestResult        # stitched out-of-sample trades
    windows: pd.DataFrame      # one row per window: dates, chosen params, IS & OOS stats


def param_combinations(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def walk_forward(df: pd.DataFrame, strategy: Callable[..., Signals], grid: dict[str, list],
                 train_sessions: int = 120, test_sessions: int = 20, objective: str = "net_to_max_dd",
                 min_trades: int = 20, warmup_sessions: int = 2, **bt_kwargs) -> WalkForwardResult:
    if "session_date" not in df:
        df = add_session_columns(df)
    sessions = pd.DatetimeIndex(pd.unique(df["session_date"]))
    combos = param_combinations(grid)
    rows, oos_trades = [], []

    start = train_sessions
    while start < len(sessions):
        train = sessions[start - train_sessions:start]
        test = sessions[start:start + test_sessions]
        is_df = df[df["session_date"].isin(train)]

        best, best_score, best_stats = None, float("-inf"), None
        for params in combos:
            stats = summary(run_backtest(is_df, strategy(is_df, **params), **bt_kwargs))
            score = stats.get(objective, float("-inf")) if stats["trades"] >= min_trades else float("-inf")
            if score > best_score:
                best, best_score, best_stats = params, score, stats
        if best is None:
            best, best_stats = combos[0], {}

        # signals computed with a few prior sessions of warm-up so indicators are primed,
        # but only test-window bars are traded
        ctx = sessions[max(0, start - warmup_sessions):start + test_sessions]
        ctx_df = df[df["session_date"].isin(ctx)]
        sig = strategy(ctx_df, **best)
        oos_df = ctx_df[ctx_df["session_date"].isin(test)]
        res = run_backtest(oos_df, sig, **bt_kwargs)
        oos_stats = summary(res)
        oos_trades.append(res.trades)
        rows.append({
            "train_start": train[0], "train_end": train[-1], "test_start": test[0], "test_end": test[-1],
            **{f"p_{k}": v for k, v in best.items()},
            "is_score": best_score, "is_net": best_stats.get("net_pnl"), "is_trades": best_stats.get("trades"),
            "oos_net": oos_stats.get("net_pnl"), "oos_trades": oos_stats.get("trades"),
        })
        start += test_sessions

    trades = pd.concat([t for t in oos_trades if not t.empty], ignore_index=True) if any(
        not t.empty for t in oos_trades) else pd.DataFrame()
    oos_sessions = sessions[train_sessions:]
    spec = bt_kwargs.get("spec", NQ)
    oos = BacktestResult(trades=trades, sessions=oos_sessions, spec=spec, params={"grid": grid})
    return WalkForwardResult(oos=oos, windows=pd.DataFrame(rows))
