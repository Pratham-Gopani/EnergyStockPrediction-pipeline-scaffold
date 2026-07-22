"""NSE trading-calendar logic, all in Asia/Kolkata (IST).

The only calendar primitive that matters is `previous_trading_day()`: the news
window for *any* day (Monday, the day after a holiday, or a plain Tuesday) is simply
"08:30:00 on the previous trading day -> 08:29:59 today". There is deliberately no
Monday-specific branch anywhere in this module -- Monday's window naturally reaches
back to Friday (or further, across a holiday) because `previous_trading_day` just
keeps stepping backward one calendar day at a time until it finds a day that is
neither a weekend nor an NSE holiday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from utils.config_loader import get_config

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class NewsWindow:
    start: datetime
    end: datetime
    reference_date: date


class TradingCalendar:
    """Weekend + configured-holiday aware NSE trading calendar."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self._holidays = self._load_holidays()
        self._start_hour = self.config.get("time_window.daily_start_hour", 8)
        self._start_minute = self.config.get("time_window.daily_start_minute", 30)

    def _load_holidays(self) -> set[date]:
        holidays_data = self.config.get("holidays") or {}
        result: set[date] = set()
        for key, values in holidays_data.items():
            if not key.startswith("holidays_"):
                continue
            for iso_str in values or []:
                result.add(date.fromisoformat(str(iso_str)))
        return result

    def is_weekend(self, d: date) -> bool:
        return d.weekday() >= 5  # Saturday=5, Sunday=6

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays

    def is_trading_day(self, d: date) -> bool:
        return not self.is_weekend(d) and not self.is_holiday(d)

    def previous_trading_day(self, d: date) -> date:
        """Step backward one calendar day at a time until a trading day is found."""
        cursor = d - timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor -= timedelta(days=1)
        return cursor

    def should_run(self, d: date) -> bool:
        return self.is_trading_day(d)

    def news_window(self, reference_date: date) -> NewsWindow:
        """Window: previous_trading_day 08:30:00 -> reference_date 08:29:59, IST."""
        prev_day = self.previous_trading_day(reference_date)
        start = datetime.combine(prev_day, time(self._start_hour, self._start_minute, 0), tzinfo=IST)
        end_time = time(self._start_hour, self._start_minute - 1, 59) if self._start_minute > 0 else time(
            self._start_hour - 1, 59, 59
        )
        end = datetime.combine(reference_date, end_time, tzinfo=IST)
        return NewsWindow(start=start, end=end, reference_date=reference_date)


def now_ist() -> datetime:
    return datetime.now(IST)
