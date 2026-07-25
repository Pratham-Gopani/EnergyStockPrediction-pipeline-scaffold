"""Tests for market.market_calendar: the news window must derive purely from
previous_trading_day(), with no Monday-specific branch, so Monday/post-holiday
cases fall out automatically.
"""

from __future__ import annotations

from datetime import date, time, timedelta

from market.market_calendar import IST, TradingCalendar


class FakeConfig:
    def __init__(self, data: dict):
        self._data = data

    def get(self, dotted_key: str, default=None):
        node = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _make_calendar(holiday_isos: list[str] | None = None) -> TradingCalendar:
    config = FakeConfig(
        {
            "holidays": {"holidays_test": holiday_isos or []},
            "time_window": {"daily_start_hour": 8, "daily_start_minute": 30},
        }
    )
    return TradingCalendar(config=config)


# ISO calendar week 33 of 2025: day=1 is Monday, guaranteed regardless of the
# actual date arithmetic elsewhere.
MONDAY = date.fromisocalendar(2025, 33, 1)
TUESDAY = MONDAY + timedelta(days=1)
WEDNESDAY = MONDAY + timedelta(days=2)
FRIDAY = MONDAY - timedelta(days=3)
SATURDAY = MONDAY + timedelta(days=5)
SUNDAY = MONDAY + timedelta(days=6)


def test_normal_weekday_window():
    calendar = _make_calendar()
    window = calendar.news_window(WEDNESDAY)

    assert window.start.date() == calendar.previous_trading_day(WEDNESDAY)
    assert window.start.date() == TUESDAY
    assert window.start.time() == time(8, 30, 0)
    assert window.end.date() == WEDNESDAY
    assert window.end.time() == time(8, 29, 59)


def test_monday_spans_the_weekend():
    calendar = _make_calendar()
    window = calendar.news_window(MONDAY)

    assert window.start.date() == FRIDAY
    assert window.start.time() == time(8, 30, 0)
    assert window.end.date() == MONDAY


def test_tuesday_after_monday_holiday_reaches_further_back():
    calendar = _make_calendar(holiday_isos=[MONDAY.isoformat()])
    window = calendar.news_window(TUESDAY)

    # Monday is a holiday, so Tuesday's window must skip weekend + Monday and
    # reach back to the prior Friday -- no special-casing required, just
    # previous_trading_day() stepping backward until it finds a trading day.
    assert window.start.date() == FRIDAY


def test_saturday_and_holiday_should_not_run():
    calendar = _make_calendar(holiday_isos=[TUESDAY.isoformat()])

    assert calendar.is_weekend(SATURDAY) is True
    assert calendar.should_run(SATURDAY) is False

    assert calendar.is_holiday(TUESDAY) is True
    assert calendar.should_run(TUESDAY) is False

    assert calendar.should_run(WEDNESDAY) is True


def test_window_uses_ist_timezone():
    calendar = _make_calendar()
    window = calendar.news_window(WEDNESDAY)
    assert window.start.tzinfo == IST
    assert window.end.tzinfo == IST
