"""
IST clock, NSE trading-day/holiday logic, and weekly-expiry resolution.
"""

import datetime as dt

from config import IST, NSE_HOLIDAYS_2026, MARKET_OPEN, MARKET_CLOSE, EOD_SQUAREOFF


def now_ist():
    return dt.datetime.now(IST)


def is_trading_day(date):
    return date.weekday() < 5 and date not in NSE_HOLIDAYS_2026


def previous_trading_day(date):
    d = date - dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def is_market_open_now():
    n = now_ist()
    if not is_trading_day(n.date()):
        return False
    return MARKET_OPEN <= n.time() <= MARKET_CLOSE


def is_eod_squareoff_time():
    return now_ist().time() >= EOD_SQUAREOFF


def get_current_weekly_expiry():
    today = now_ist().date()
    days_ahead = (1 - today.weekday()) % 7
    expiry = today + dt.timedelta(days=days_ahead)

    while not is_trading_day(expiry):
        expiry -= dt.timedelta(days=1)

    return expiry
