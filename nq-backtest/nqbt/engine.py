"""Bar-by-bar futures backtester.

Execution model (deliberately conservative):

* A strategy emits a *desired position* (-1, 0, +1) known at each bar's CLOSE.
  It is filled at the NEXT bar's OPEN plus adverse slippage. No same-bar fills.
* Stops are stop-market orders: filled at the stop (or the open, if the bar
  gaps through it) plus adverse slippage.
* Targets are limit orders: filled at the limit with no slippage, but only if
  price trades *through* the limit by ``limit_through_ticks``.
* If a bar touches both stop and target, the STOP is assumed to fill first.
* After a stop/target exit, re-entry in the same direction is blocked until
  the desired position changes (prevents instant re-entry on a stale signal).
* Positions are flattened at the close of the last bar of each session
  (optional) and always before a contract change (roll) and at end of data.
  Signals from a session's last bar never carry into the next session.
* Commission is charged per contract per side.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contract import NQ, FuturesSpec
from .data import add_session_columns


@dataclass
class CostModel:
    commission_per_side: float = 2.50   # $/contract/side, all-in (broker + exchange + NFA)
    slippage_ticks: float = 1.0         # adverse ticks on market and stop orders


@dataclass
class Signals:
    position: pd.Series                        # desired position at bar close
    stop_points: pd.Series | float | None = None   # stop distance from fill, in index points
    target_points: pd.Series | float | None = None


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    sessions: pd.DatetimeIndex
    spec: FuturesSpec
    params: dict = field(default_factory=dict)

    def daily_pnl(self) -> pd.Series:
        """Net P&L per session (sessions without trades = 0)."""
        if self.trades.empty:
            return pd.Series(0.0, index=self.sessions)
        return self.trades.groupby("session_date")["pnl_net"].sum().reindex(self.sessions, fill_value=0.0)

    def equity(self, starting_capital: float = 0.0) -> pd.Series:
        return starting_capital + self.daily_pnl().cumsum()


def _as_array(x, index: pd.Index) -> np.ndarray:
    if x is None:
        return np.full(len(index), np.nan)
    if np.isscalar(x):
        return np.full(len(index), float(x))
    return pd.Series(x).reindex(index).to_numpy(float)


def run_backtest(df: pd.DataFrame, signals: Signals, spec: FuturesSpec = NQ,
                 costs: CostModel | None = None, qty: int = 1,
                 flatten_at_session_end: bool = True, limit_through_ticks: int = 1) -> BacktestResult:
    costs = costs or CostModel()
    if "session_date" not in df:
        df = add_session_columns(df)
    idx = df.index
    n = len(df)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    desired = np.sign(signals.position.reindex(idx).fillna(0).to_numpy(float)).astype(int)
    stop_pts = _as_array(signals.stop_points, idx)
    tgt_pts = _as_array(signals.target_points, idx)
    sess = df["session_date"].to_numpy()

    # break_after[i]: bar i is the last bar before a forced flatten
    sess_break = np.append(sess[1:] != sess[:-1], True)
    roll_break = np.zeros(n, dtype=bool)
    if "contract" in df:
        k = df["contract"].to_numpy()
        roll_break[:-1] = k[1:] != k[:-1]
    break_after = roll_break.copy()
    break_after[-1] = True
    if flatten_at_session_end:
        break_after |= sess_break

    tick = spec.tick_size
    slip = costs.slippage_ticks * tick
    through = limit_through_ticks * tick
    trades: list[dict] = []

    pos = 0
    entry_px = stop_px = tgt_px = np.nan
    entry_i = -1
    blocked = 0

    def close(i: int, px: float, reason: str) -> None:
        nonlocal pos
        pts = (px - entry_px) * pos
        gross = pts * spec.point_value * qty
        comm = 2 * costs.commission_per_side * qty
        trades.append({
            "entry_time": idx[entry_i], "exit_time": idx[i], "session_date": sess[i],
            "side": pos, "qty": qty, "entry_price": entry_px, "exit_price": px,
            "pnl_points": pts, "pnl_gross": gross, "commission": comm, "pnl_net": gross - comm,
            "exit_reason": reason, "bars_held": i - entry_i,
        })
        pos = 0

    for i in range(n):
        # 1) act on the signal from the previous bar's close, at this bar's open
        if i > 0:
            if break_after[i - 1]:
                blocked = 0  # new session/contract: fresh start, stale signal ignored
            else:
                want = desired[i - 1]
                if blocked and want != blocked:
                    blocked = 0
                if blocked and want == blocked:
                    want = 0 if pos == 0 else pos
                if want != pos:
                    if pos != 0:
                        close(i, o[i] - pos * slip, "signal")
                    if want != 0:
                        pos = want
                        entry_i = i
                        entry_px = o[i] + pos * slip
                        sp, tp = stop_pts[i - 1], tgt_pts[i - 1]
                        stop_px = entry_px - pos * sp if np.isfinite(sp) else np.nan
                        tgt_px = entry_px + pos * tp if np.isfinite(tp) else np.nan

        # 2) protective stop / profit target within this bar
        if pos != 0:
            if pos > 0:
                hit_stop = np.isfinite(stop_px) and l[i] <= stop_px
                hit_tgt = np.isfinite(tgt_px) and h[i] >= tgt_px + through
                stop_fill = min(o[i], stop_px) - slip
                tgt_fill = max(o[i], tgt_px)
            else:
                hit_stop = np.isfinite(stop_px) and h[i] >= stop_px
                hit_tgt = np.isfinite(tgt_px) and l[i] <= tgt_px - through
                stop_fill = max(o[i], stop_px) + slip
                tgt_fill = min(o[i], tgt_px)
            if hit_stop:
                side = pos
                close(i, stop_fill, "stop")
                blocked = side
            elif hit_tgt:
                side = pos
                close(i, tgt_fill, "target")
                blocked = side

        # 3) forced flatten at session end / roll / end of data
        if pos != 0 and break_after[i]:
            reason = "end_of_data" if i == n - 1 else "roll" if roll_break[i] else "session_end"
            close(i, c[i] - pos * slip, reason)

    trades_df = pd.DataFrame(trades)
    sessions = pd.DatetimeIndex(pd.unique(sess))
    return BacktestResult(trades=trades_df, sessions=sessions, spec=spec)
