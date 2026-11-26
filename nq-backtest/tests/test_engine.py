import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nqbt import (NQ, CostModel, Signals, add_session_columns, front_contract, monte_carlo_drawdown,
                  opening_range_breakout, run_backtest, summary, synthetic_minutes, third_friday,
                  validate, walk_forward)

TZ = "America/New_York"
NO_COST = CostModel(commission_per_side=0.0, slippage_ticks=0.0)


def bars(rows, start="2025-03-04 10:00", freq="min"):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(pd.Timestamp(start, tz=TZ), periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100
    return add_session_columns(df)


def sig(df, positions, stop=None, target=None):
    return Signals(pd.Series(positions, index=df.index, dtype=float), stop, target)


# ---------------------------------------------------------------- contract
def test_third_friday_and_roll():
    assert third_friday(2026, 3) == dt.date(2026, 3, 20)
    assert third_friday(2025, 12) == dt.date(2025, 12, 19)
    assert front_contract(dt.date(2026, 3, 11)) == "NQH26"
    assert front_contract(dt.date(2026, 3, 12)) == "NQM26"
    assert front_contract(dt.date(2025, 12, 20)) == "NQH26"


def test_point_value():
    assert NQ.point_value == 20.0


# ---------------------------------------------------------------- engine
def test_fill_timing_exact():
    df = bars([(100, 101, 99, 100), (102, 103, 101, 102), (104, 105, 103, 104), (106, 107, 105, 106)])
    res = run_backtest(df, sig(df, [1, 1, 0, 0]), costs=CostModel(commission_per_side=2.5, slippage_ticks=1))
    t = res.trades.iloc[0]
    assert t.entry_time == df.index[1] and t.entry_price == 102.25
    assert t.exit_time == df.index[3] and t.exit_price == 105.75
    assert t.pnl_points == pytest.approx(3.5)
    assert t.pnl_net == pytest.approx(3.5 * 20 - 5.0)
    assert t.exit_reason == "signal"


def test_stop_and_gap_through():
    # long from 100 with 2pt stop at 98; bar 2 gaps open to 97 -> filled at 97, not 98
    df = bars([(100, 100, 100, 100), (100, 101, 99, 100), (97, 98, 96, 97)])
    res = run_backtest(df, sig(df, [1, 1, 1], stop=2.0), costs=NO_COST)
    t = res.trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_price == 97


def test_stop_wins_when_both_touched():
    df = bars([(100, 100, 100, 100), (100, 105, 95, 100)])
    res = run_backtest(df, sig(df, [1, 1], stop=2.0, target=2.0), costs=NO_COST)
    assert res.trades.iloc[0].exit_reason == "stop"


def test_target_requires_trade_through():
    df = bars([(100, 100, 100, 100), (100, 102, 100, 101), (101, 102.25, 101, 102)])
    res = run_backtest(df, sig(df, [1, 1, 1], target=2.0), costs=NO_COST)
    t = res.trades.iloc[0]
    assert t.exit_reason == "target" and t.exit_time == df.index[2] and t.exit_price == 102


def test_no_reentry_after_stop_until_signal_changes():
    df = bars([(100, 100, 100, 100), (100, 100, 97, 98), (98, 99, 98, 99), (99, 99, 99, 99)])
    res = run_backtest(df, sig(df, [1, 1, 1, 1], stop=2.0), costs=NO_COST)
    assert len(res.trades) == 1


def test_short_side_pnl():
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100), (90, 90, 90, 90)])
    res = run_backtest(df, sig(df, [-1, -1, 0]), costs=NO_COST)
    t = res.trades.iloc[0]
    assert t.side == -1 and t.pnl_points == 10 and t.exit_reason == "end_of_data"


def test_session_end_flatten_and_no_carry():
    # 16:58 and 16:59 in session A, then 18:00 opens next session B
    idx = pd.DatetimeIndex([pd.Timestamp("2025-03-04 16:58", tz=TZ), pd.Timestamp("2025-03-04 16:59", tz=TZ),
                            pd.Timestamp("2025-03-04 18:00", tz=TZ), pd.Timestamp("2025-03-04 18:01", tz=TZ)])
    df = add_session_columns(pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                                           "volume": 1}, index=idx))
    assert df["session_date"].nunique() == 2
    res = run_backtest(df, sig(df, [1, 1, 0, 0]), costs=NO_COST)
    assert len(res.trades) == 1
    assert res.trades.iloc[0].exit_reason == "session_end"
    assert res.trades.iloc[0].exit_time == idx[1]


def test_roll_flatten():
    df = bars([(100, 100, 100, 100)] * 4)
    df["contract"] = ["NQH25", "NQH25", "NQM25", "NQM25"]
    res = run_backtest(df, sig(df, [1, 1, 1, 1]), costs=NO_COST)
    assert res.trades.iloc[0].exit_reason == "roll"


def test_no_lookahead_future_data_change():
    """Changing prices after bar k must not change any trade that closed before bar k."""
    df = synthetic_minutes("2024-03-04", "2024-03-15", seed=1)
    df = add_session_columns(df)
    base = run_backtest(df, opening_range_breakout(df)).trades
    cut = df.index[len(df) // 2]
    df2 = df.copy()
    df2.loc[df2.index > cut, ["open", "high", "low", "close"]] += 500
    alt = run_backtest(df2, opening_range_breakout(df2)).trades
    a = base[base.exit_time < cut].reset_index(drop=True)
    b = alt[alt.exit_time < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- strategy / pipeline
@pytest.fixture(scope="module")
def synth():
    return add_session_columns(synthetic_minutes("2024-01-01", "2024-06-30", seed=42))


def test_synthetic_data_is_clean(synth):
    assert validate(synth) == []


def test_orb_at_most_one_trade_per_session(synth):
    res = run_backtest(synth, opening_range_breakout(synth))
    assert not res.trades.empty
    assert res.trades.groupby("session_date").size().max() == 1
    assert (res.trades.entry_time.dt.hour * 60 + res.trades.entry_time.dt.minute >= 600).all()


def test_summary_and_monte_carlo(synth):
    res = run_backtest(synth, opening_range_breakout(synth))
    s = summary(res)
    assert s["trades"] == len(res.trades)
    assert s["net_pnl"] == pytest.approx(res.trades.pnl_net.sum())
    assert s["net_pnl"] == pytest.approx(res.daily_pnl().sum())
    mc = monte_carlo_drawdown(res, n_sims=200)
    assert mc["dd_p50"] <= mc["dd_p95"] <= mc["dd_p99"]


def test_walk_forward_runs(synth):
    wf = walk_forward(synth, opening_range_breakout, {"or_minutes": [15, 30], "target_r": [1.0, 2.0]},
                      train_sessions=60, test_sessions=20, min_trades=5)
    assert len(wf.windows) >= 3
    assert wf.oos.trades.session_date.min() > wf.windows.train_end.iloc[0]
    assert np.isfinite(summary(wf.oos)["net_pnl"])
