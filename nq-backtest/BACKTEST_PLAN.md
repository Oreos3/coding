# NQ Futures Backtest Plan

A step-by-step plan for testing an intraday strategy on CME E-mini Nasdaq-100 (NQ) / Micro (MNQ)
futures in a way that holds up when traded live. The code in `nqbt/` puts this plan into practice.

---

## 1. Hypothesis (write it down *before* touching data)

| Item | Example (Opening Range Breakout) |
|---|---|
| Edge | The first decisive close outside the first N minutes' range tends to keep going during the morning |
| Why it might exist | Order flow imbalance after the cash open; stops clustered around the overnight/OR extremes |
| Market / timeframe | NQ, 1-minute bars, Regular Trading Hours (09:30–16:00 ET) |
| Holding period | Intraday. Always flat by 15:55 ET, no overnight risk |
| Kill criteria | Out-of-sample profit factor < 1.1 after costs, or edge exists only at one parameter value |

If you can't say why the edge exists, assume any profit you find is curve-fitting.

## 2. Instrument facts the backtest must get right

| Spec | NQ | MNQ |
|---|---|---|
| Tick size | 0.25 index pts | 0.25 |
| Tick value | $5.00 | $0.50 |
| Point value | **$20** | **$2** |
| Contract months | H (Mar), M (Jun), U (Sep), Z (Dec) | same |
| Expiry | 3rd Friday of contract month (09:30 ET SOQ) | same |
| Roll | ~8 calendar days before expiry (Thursday of the prior week) | same |
| Globex hours | Sun–Fri 18:00–17:00 ET, daily halt 17:00–18:00 | same |
| Typical all-in commission | ~$2.00–$3.00 / side | ~$0.50–$1.00 / side |

Implemented in `nqbt/contract.py` (`NQ`, `MNQ`, `third_friday`, `roll_date`, `front_contract`).

## 3. Data

**Need:** at least 5 years of 1-minute OHLCV bars (more is better: 2020 covid, 2022 bear market,
2023–24 AI rally are very different market regimes). Trade-level or 1-second data is better still
for checking stop fills.

**Sources (paid, roughly cheapest first):** Databento (CME Globex MDP3, per-contract with
continuous symbology), FirstRate Data, Kibot, Portara/CQG, IQFeed, NinjaTrader/Tradovate export,
Interactive Brokers historical API (rate-limited).

**Format expected by `load_csv`:**
```
timestamp,open,high,low,close,volume,contract
2024-03-04T14:30:00Z,18250.25,18262.00,18248.50,18260.75,4123,NQH24
```
* Timestamps = bar **start**. Put the UTC offset in the timestamp or use `--source-tz`.
* `contract` column is optional but strongly recommended (the engine flattens on contract change).

**Continuous contract choice:**
* Intraday-only strategies → use **unadjusted front-month** data rolled on the roll date, and flatten
  at the roll (the engine does this). Prices are then real and on the tick grid.
* Multi-day strategies → need **back-adjusted** (Panama/ratio) data built from *both* contracts'
  prices at the roll time. Never stitch by simply joining the series. That creates fake gaps.

**Validation checklist** (`nqbt.validate`): sorted/unique timestamps, no NaNs, OHLC consistency,
prices on the 0.25 grid, no > 5 % bar-to-bar jumps, plus manual checks:
- [ ] Session counts per year ≈ 250; look for missing days other than exchange holidays
- [ ] Half-days (day after Thanksgiving, Christmas Eve) handled. They have early closes
- [ ] DST transitions: RTH open is still 09:30 ET in March/November
- [ ] Volume jumps to the new contract around each roll date

## 4. Execution & cost model (be pessimistic)

| Rule | Setting in `nqbt.engine` |
|---|---|
| Signals computed on bar close, filled at **next bar open** | always |
| Market & stop orders: adverse slippage | `slippage_ticks=1` (test 2 too) |
| Stop gap-through: fill at the open if it's beyond the stop | always |
| Limit (target) needs price to trade *through* | `limit_through_ticks=1` |
| Stop and target both inside one bar → **stop first** | always |
| Commission per side | `commission_per_side=2.50` (NQ) / `0.85` (MNQ) |
| Flat at session end, at the roll, and at end of data | `flatten_at_session_end=True` |
| No re-entry on a stale signal after a stop/target | always |

**Cost sensitivity is required:** report results at 0, 1, 2 ticks slippage. If the edge disappears
at 2 ticks, it is not tradable. Around news (CPI, FOMC, NFP), real NQ slippage can be 4+ ticks.

## 5. Methodology

### 5.1 Split the data *once*, up front
```
|-------- In-sample / research 60% --------|---- Walk-forward 25% ----|-- Holdout 15% --|
```
- **Holdout** (most recent ~12 months): do not touch it until the very end. Run it once.
  If you change anything after looking, it is no longer a holdout.

### 5.2 Baseline
1. Run default parameters on the in-sample data. Record the full `summary()` output.
2. Sanity checks: trades per day, average holding time, long vs short P&L, exit-reason
   breakdown (`exit_breakdown`). Look at 10 random trades on a chart.
3. **Benchmarks:** compare to (a) random entries with the same timing, stop and target
   (use several seeds), (b) the same strategy run on the synthetic random walk
   (`synthetic_minutes`). The strategy must clearly beat both.

### 5.3 Parameter robustness (not optimisation)
- Use a coarse grid (e.g. `or_minutes ∈ {15, 30, 60}`, `target_r ∈ {1, 1.5, 2, 3}`).
- Plot a heatmap of net P&L / profit factor. You want a **broad plateau**, not one isolated peak.
- Keep total combinations small (< 50). Every extra trial makes a good result more likely to be luck.

### 5.4 Walk-forward (`nqbt.walk_forward`)
- Rolling window: train 120 sessions (~6 months) → test 20 sessions (~1 month), step 20.
- Objective: `net_to_max_dd` (or profit factor), with a minimum number of trades.
- Report the **stitched out-of-sample** equity only. Also check whether the chosen parameters stay
  steady from window to window. If they jump around, the edge is probably noise.
- Walk-forward efficiency = OOS annualised P&L ÷ IS annualised P&L. Aim for > 0.5.

### 5.5 Robustness tests
- [ ] **Monte Carlo** trade bootstrap (`monte_carlo_drawdown`): size the account using the 95th–99th percentile drawdown, not the historical one
- [ ] **Regime split:** results by year, by VIX/VXN level, trend vs range days, high vs low ATR
- [ ] **Day-of-week / event days:** with and without FOMC/CPI/NFP days
- [ ] **Time shift:** move entry window ±5–15 min. The edge should degrade smoothly
- [ ] **Slippage stress:** 2 ticks + double commission
- [ ] **Cross-market:** run unchanged on ES / RTY / YM. A real edge usually carries over at least partly
- [ ] **Deflated Sharpe / multiple testing:** count every variant you tried and discount the result for it

### 5.6 Holdout
Run the final, frozen strategy once on the holdout. Accept if it is in line with the walk-forward
OOS results (not necessarily as good). If it fails, **stop**. Don't tune it until it passes.

## 6. Acceptance criteria (all after costs)

| Metric | Minimum |
|---|---|
| OOS trades | ≥ 200 |
| Profit factor | ≥ 1.25 |
| Expectancy | ≥ 3 × round-trip cost (≥ ~$45/contract on NQ) |
| Sharpe (daily, annualised) | ≥ 1.0 |
| Net P&L / Max DD | ≥ 2 per year traded |
| Profitable years | ≥ 70 % |
| Edge holds at 2-tick slippage | yes |
| Parameter plateau (neighbours ≥ 70 % of best) | yes |

## 7. Risk & sizing (after the strategy passes)
- Start with **MNQ** (1/10 the size of NQ) for live/paper trading.
- Risk per trade ≤ 0.5–1 % of the account, based on the stop distance in points × point value.
  One NQ point = $20, so a 100-point opening-range stop is **$2,000 per NQ contract** and $200 per MNQ.
- Daily loss limit (e.g. 2R) and a max-drawdown kill switch set from the Monte Carlo p99 drawdown.
- Keep an exchange-margin buffer: CME NQ initial margin is often in the $20–30k+ range per contract and changes with volatility.

## 8. Going live
1. Paper trade 1–3 months. Compare each live fill with what the backtest would have filled on the same bars.
2. Track **implementation shortfall** (live P&L vs backtest P&L on the same days). If it is above ~20 % of expectancy, fix execution before scaling up.
3. Go live on 1 MNQ, then scale step by step. Re-run the walk-forward every quarter.

## 9. Common pitfalls checklist
- [ ] Look-ahead: using the bar's high/low/close to enter on that same bar (engine prevents this; test `test_no_lookahead_future_data_change`)
- [ ] Opening range computed using bars after the entry time
- [ ] Timezone mistakes (UTC vs ET, DST) shifting the session by an hour
- [ ] Trading through the roll on unadjusted data (fake gap P&L)
- [ ] Survivorship of "good" periods (only testing 2023–24)
- [ ] Ignoring that stop-and-target-in-one-bar is ambiguous on 1-minute data. Check on 1-second data.
- [ ] Optimising on the full dataset and then calling a sub-period "out-of-sample"

## 10. Work breakdown

| # | Task | Output |
|---|---|---|
| 1 | Acquire ≥ 5y NQ 1-min data with contract column | `data/nq_1m.csv` |
| 2 | Run `validate`, fix holiday/DST/roll issues | data QA notes |
| 3 | Baseline ORB on in-sample + random-entry benchmark | `output/orb_*` |
| 4 | Parameter heatmap & cost sensitivity | plateau decision |
| 5 | Walk-forward + Monte Carlo + regime splits | OOS report |
| 6 | Holdout run (once) | go / no-go |
| 7 | Paper trade MNQ, measure shortfall | live vs backtest log |
