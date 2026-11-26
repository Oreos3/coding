"""Example strategies. Each takes a sessionized DataFrame + params and returns ``Signals``.

All signals use only information available at the bar's close; the engine
fills them on the next bar's open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RTH_START, add_session_columns
from .engine import Signals


def _hhmm(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


def opening_range_breakout(df: pd.DataFrame, or_minutes: int = 30, target_r: float = 2.0,
                           last_entry: str = "11:30", flat_at: str = "15:55",
                           max_stop_points: float | None = None) -> Signals:
    """One trade per RTH session in the direction of the first close outside the opening range.

    Stop: opposite side of the opening range. Target: ``target_r`` x stop distance.
    """
    if "session_date" not in df:
        df = add_session_columns(df)
    m = df["minute_of_day"]
    sess = df["session_date"]
    or_end = RTH_START + or_minutes
    in_or = (m >= RTH_START) & (m < or_end)
    or_high = df["high"].where(in_or).groupby(sess).transform("max")
    or_low = df["low"].where(in_or).groupby(sess).transform("min")

    window = (m >= or_end) & (m < _hhmm(last_entry))
    raw = np.where(window & (df["close"] > or_high), 1, np.where(window & (df["close"] < or_low), -1, 0))
    raw = pd.Series(raw, index=df.index)
    first = (raw != 0) & ((raw != 0).groupby(sess).cumsum() == 1)
    direction = raw.where(first).groupby(sess).ffill().fillna(0)
    direction = direction.where((m >= or_end) & (m < _hhmm(flat_at)), 0)

    stop = pd.Series(np.where(direction > 0, df["close"] - or_low, or_high - df["close"]), index=df.index)
    if max_stop_points is not None:
        # skip the whole session if the stop at the breakout bar is too wide
        first_stop = stop.where(first).groupby(sess).transform("max")
        direction = direction.where(~(first_stop > max_stop_points), 0)
    return Signals(position=direction, stop_points=stop, target_points=stop * target_r)


def ema_trend(df: pd.DataFrame, fast: int = 20, slow: int = 60, atr_len: int = 14,
              atr_stop: float = 3.0, start: str = "09:45", flat_at: str = "15:55") -> Signals:
    """Intraday EMA crossover with an ATR stop, RTH only."""
    if "session_date" not in df:
        df = add_session_columns(df)
    m = df["minute_of_day"]
    close = df["close"]
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    prev_c = close.shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_len).mean()
    live = (m >= _hhmm(start)) & (m < _hhmm(flat_at))
    pos = np.sign(ema_f - ema_s).where(live, 0).fillna(0)
    return Signals(position=pos, stop_points=atr * atr_stop)


STRATEGIES = {
    "orb": opening_range_breakout,
    "ema": ema_trend,
}
