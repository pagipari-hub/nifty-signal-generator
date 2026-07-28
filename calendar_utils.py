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
    """
    Resolves the expiry date to trade against today.

    Base logic (unchanged): NIFTY's weekly expiry is Tuesday. Target the
    nearest Tuesday on/after today, then walk backward if that date is
    an NSE holiday.

    FIX (2026-07-28, Mon/Tue thin-premium root cause): on Monday, the
    nearest Tuesday is tomorrow (1 day to expiry); on Tuesday, it's
    today (0 days to expiry -- expiry day itself). Combined with
    typically low VIX on these two days, ATM+-100/ATM+-400 strikes were
    consistently landing at very thin premiums (e.g. entry ~9, SL ~13,
    target ~1) -- theta has already eaten most of the time value and
    there's not enough of the underlying's range left to reliably clear
    the distance to target before SL. On Monday and Tuesday specifically,
    roll forward to NEXT week's Tuesday instead, giving the strategy a
    full week of time value to work with on exactly the two days it was
    thinnest. Wednesday through Friday are unaffected -- the nearest
    Tuesday on those days is already next week's contract, which is the
    behavior that was already working fine.
    """
    today = now_ist().date()
    days_ahead = (1 - today.weekday()) % 7
    expiry = today + dt.timedelta(days=days_ahead)

    if today.weekday() in (0, 1):  # Monday, Tuesday -> roll to next week's expiry
        expiry += dt.timedelta(days=7)

    while not is_trading_day(expiry):
        expiry -= dt.timedelta(days=1)

    return expiry
