"""nqbt — a small, conservative backtester for CME NQ / MNQ futures."""
from .contract import MNQ, NQ, FuturesSpec, front_contract, roll_date, third_friday
from .data import add_session_columns, load_csv, rth_only, synthetic_minutes, validate
from .engine import BacktestResult, CostModel, Signals, run_backtest
from .metrics import exit_breakdown, format_summary, monte_carlo_drawdown, summary
from .strategies import STRATEGIES, ema_trend, opening_range_breakout
from .walkforward import walk_forward

__all__ = [
    "NQ", "MNQ", "FuturesSpec", "front_contract", "roll_date", "third_friday",
    "add_session_columns", "load_csv", "rth_only", "synthetic_minutes", "validate",
    "BacktestResult", "CostModel", "Signals", "run_backtest",
    "exit_breakdown", "format_summary", "monte_carlo_drawdown", "summary",
    "STRATEGIES", "ema_trend", "opening_range_breakout", "walk_forward",
]
