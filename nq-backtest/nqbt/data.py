"""Loading, validating and sessionizing 1-minute NQ bars.

Conventions
-----------
* Index: tz-aware ``DatetimeIndex`` in America/New_York, bar *start* time.
* Columns: open, high, low, close, volume (+ optional ``contract``).
* A Globex trading session runs 18:00 ET (prior day) -> 17:00 ET, so the
  session date is the ET date of ``timestamp + 6h``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import NQ, FuturesSpec, front_contract

TZ = "America/New_York"
REQUIRED = ["open", "high", "low", "close", "volume"]
RTH_START = 9 * 60 + 30  # minute of day, 09:30 ET
RTH_END = 16 * 60        # 16:00 ET (exclusive)


def load_csv(path: str, timestamp_col: str = "timestamp", source_tz: str = "UTC") -> pd.DataFrame:
    """Load OHLCV bars from CSV. Naive timestamps are assumed to be ``source_tz``."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    timestamp_col = timestamp_col.lower()
    missing = [c for c in [timestamp_col, *REQUIRED] if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    ts = _parse_timestamps(raw[timestamp_col], source_tz)
    df = raw.drop(columns=[timestamp_col])
    df.index = pd.DatetimeIndex(ts).tz_convert(TZ)
    df.index.name = "timestamp"
    return df.sort_index()


def _parse_timestamps(col: pd.Series, source_tz: str) -> pd.Series:
    try:
        ts = pd.to_datetime(col)
    except (ValueError, TypeError):  # mixed UTC offsets (e.g. across DST)
        return pd.to_datetime(col, utc=True)
    if not pd.api.types.is_datetime64_any_dtype(ts):
        return pd.to_datetime(col, utc=True)
    return ts if ts.dt.tz is not None else ts.dt.tz_localize(source_tz)


def validate(df: pd.DataFrame, spec: FuturesSpec = NQ) -> list[str]:
    """Return a list of data-quality problems (empty list == clean)."""
    issues = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if not df.index.is_monotonic_increasing:
        issues.append("index is not sorted")
    if (n := df.index.duplicated().sum()):
        issues.append(f"{n} duplicate timestamps")
    if (n := df[REQUIRED].isna().any(axis=1).sum()):
        issues.append(f"{n} rows with NaN")
    if (n := (df["high"] < df[["open", "close"]].max(axis=1)).sum()):
        issues.append(f"{n} bars with high < max(open, close)")
    if (n := (df["low"] > df[["open", "close"]].min(axis=1)).sum()):
        issues.append(f"{n} bars with low > min(open, close)")
    off_grid = ~np.isclose(np.mod(df[["open", "high", "low", "close"]].to_numpy() / spec.tick_size, 1), 0) \
        & ~np.isclose(np.mod(df[["open", "high", "low", "close"]].to_numpy() / spec.tick_size, 1), 1)
    if (n := off_grid.any(axis=1).sum()):
        issues.append(f"{n} bars with prices off the {spec.tick_size} tick grid (back-adjusted data?)")
    ret = df["close"].pct_change().abs()
    if (n := (ret > 0.05).sum()):
        issues.append(f"{n} bar-to-bar moves > 5% (bad ticks or unadjusted rolls?)")
    return issues


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    if idx.tz is None:
        raise ValueError("index must be tz-aware")
    if str(idx.tz) != TZ:
        idx = idx.tz_convert(TZ)
        out.index = idx
    out["session_date"] = (idx + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    out["minute_of_day"] = idx.hour * 60 + idx.minute
    out["rth"] = (out["minute_of_day"] >= RTH_START) & (out["minute_of_day"] < RTH_END)
    return out


def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    if "rth" not in df:
        df = add_session_columns(df)
    return df[df["rth"]]


def synthetic_minutes(start: str, end: str, seed: int = 0, start_price: float = 18000.0,
                      annual_vol: float = 0.22, spec: FuturesSpec = NQ) -> pd.DataFrame:
    """Random-walk 1-minute Globex bars for plumbing tests. NOT market data.

    Caveat: bars jump from open to close with fat-tailed returns and random wicks,
    so stop orders filled at their exact price on jump bars are slightly optimistic.
    Stop-heavy strategies can therefore look a little positive here before costs.
    Use real data plus a random-entry benchmark to judge an edge.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for d in pd.bdate_range(start, end):
        open_ts = pd.Timestamp(d - pd.Timedelta(days=1)).replace(hour=18).tz_localize(TZ)
        frames.append(pd.date_range(open_ts, periods=23 * 60, freq="min"))
    idx = frames[0].append(frames[1:])
    mod = np.asarray(idx.hour * 60 + idx.minute)
    # U-shaped intraday volatility: high at the RTH open/close, low overnight
    profile = np.where((mod >= RTH_START) & (mod < RTH_END), 1.0, 0.45)
    profile = profile + 1.5 * np.exp(-((mod - RTH_START) / 20.0) ** 2) \
        + 0.6 * np.exp(-((mod - RTH_END) / 15.0) ** 2)
    sigma = annual_vol / np.sqrt(252 * 23 * 60) * profile / profile.mean()
    rets = rng.standard_t(df=4, size=len(idx)) / np.sqrt(2) * sigma
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, sigma * close * 0.6))
    high = np.maximum(open_, close) + wick * rng.random(len(idx))
    low = np.minimum(open_, close) - wick * rng.random(len(idx))
    tick = spec.tick_size
    r = lambda x: np.round(x / tick) * tick  # noqa: E731
    df = pd.DataFrame({"open": r(open_), "high": r(high), "low": r(low), "close": r(close),
                       "volume": rng.poisson(200 * profile * 3)}, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df.index.name = "timestamp"
    sessions = (idx + pd.Timedelta(hours=6)).normalize()
    contracts = {s: front_contract(s.date()) for s in sessions.unique()}
    df["contract"] = sessions.map(contracts)
    return df
