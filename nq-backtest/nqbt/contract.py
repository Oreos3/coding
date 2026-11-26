"""Contract specifications and roll calendar for CME Nasdaq-100 futures."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesSpec:
    symbol: str
    tick_size: float
    tick_value: float

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size


# E-mini Nasdaq-100: 0.25 index point tick = $5.00  ->  $20 per point
NQ = FuturesSpec("NQ", tick_size=0.25, tick_value=5.00)
# Micro E-mini Nasdaq-100: 0.25 tick = $0.50  ->  $2 per point
MNQ = FuturesSpec("MNQ", tick_size=0.25, tick_value=0.50)

QUARTERLY_MONTHS = {3: "H", 6: "M", 9: "U", 12: "Z"}


def third_friday(year: int, month: int) -> dt.date:
    """Expiration (final settlement) date for equity index futures."""
    d = dt.date(year, month, 15)  # the third Friday falls on the 15th-21st
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def roll_date(year: int, month: int, days_before: int = 8) -> dt.date:
    """Date on which volume typically moves to the next contract.

    CME equity index futures roll ~8 calendar days before expiry (the
    Thursday of the week before the third Friday).
    """
    return third_friday(year, month) - dt.timedelta(days=days_before)


def front_contract(date: dt.date, days_before: int = 8, symbol: str = "NQ") -> str:
    """Most liquid contract on ``date``, e.g. ``NQH26``."""
    for year in (date.year, date.year + 1):
        for month, code in QUARTERLY_MONTHS.items():
            if date < roll_date(year, month, days_before):
                return f"{symbol}{code}{year % 100:02d}"
    raise AssertionError("unreachable")
