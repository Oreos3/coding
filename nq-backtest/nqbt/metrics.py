"""Performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    return float((peak[1:] - equity).max())


def _longest_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def summary(result: BacktestResult, starting_capital: float = 50_000.0) -> dict:
    t = result.trades
    daily = result.daily_pnl()
    out = {"sessions": len(daily), "trades": len(t)}
    if t.empty:
        return out | {"net_pnl": 0.0}
    pnl = t["pnl_net"].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    rets = daily / starting_capital
    downside = rets[rets < 0].std(ddof=0)
    out |= {
        "net_pnl": float(pnl.sum()),
        "gross_pnl": float(t["pnl_gross"].sum()),
        "commissions": float(t["commission"].sum()),
        "win_rate": float(len(wins) / len(pnl)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "expectancy": float(pnl.mean()),
        "expectancy_points": float(t["pnl_points"].mean()),
        "max_drawdown": _max_drawdown(np.cumsum(pnl)),
        "max_consec_losses": _longest_streak(pnl <= 0),
        "sharpe": float(rets.mean() / rets.std(ddof=0) * np.sqrt(252)) if rets.std(ddof=0) > 0 else 0.0,
        "sortino": float(rets.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0.0,
        "pct_days_traded": float((daily != 0).mean()),
        "avg_bars_held": float(t["bars_held"].mean()),
        "long_pnl": float(t.loc[t["side"] > 0, "pnl_net"].sum()),
        "short_pnl": float(t.loc[t["side"] < 0, "pnl_net"].sum()),
    }
    out["return_on_capital"] = out["net_pnl"] / starting_capital
    out["net_to_max_dd"] = out["net_pnl"] / out["max_drawdown"] if out["max_drawdown"] > 0 else float("inf")
    return out


def exit_breakdown(result: BacktestResult) -> pd.DataFrame:
    t = result.trades
    if t.empty:
        return pd.DataFrame()
    return t.groupby("exit_reason")["pnl_net"].agg(["count", "sum", "mean"])


def monte_carlo_drawdown(result: BacktestResult, n_sims: int = 2000, seed: int = 0,
                         percentiles=(50, 95, 99)) -> dict:
    """Bootstrap the trade sequence to estimate the drawdown distribution."""
    pnl = result.trades["pnl_net"].to_numpy() if not result.trades.empty else np.array([])
    if len(pnl) < 2:
        return {}
    rng = np.random.default_rng(seed)
    sims = rng.choice(pnl, size=(n_sims, len(pnl)), replace=True)
    eq = np.cumsum(sims, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((n_sims, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = (peak - eq).max(axis=1)
    out = {f"dd_p{p}": float(np.percentile(dd, p)) for p in percentiles}
    out["prob_net_loss"] = float((eq[:, -1] < 0).mean())
    return out


def format_summary(stats: dict) -> str:
    lines = []
    for k, v in stats.items():
        if isinstance(v, float):
            if k in {"win_rate", "pct_days_traded", "return_on_capital", "prob_net_loss"}:
                lines.append(f"{k:>20}: {v:8.1%}")
            else:
                lines.append(f"{k:>20}: {v:12,.2f}")
        else:
            lines.append(f"{k:>20}: {v}")
    return "\n".join(lines)
