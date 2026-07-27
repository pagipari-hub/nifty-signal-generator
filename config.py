"""
Constants and settings shared across the strategy runner.

Nothing in this module has side effects or reads the clock -- it's pure
configuration, safe to import from anywhere without creating import
cycles.
"""

import os
import datetime as dt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ---- Files (all committed back to the repo by the GitHub Action) ----
STATE_FILE = "state.json"
LOCK_FILE = "run.lock"
PREV_DAY_CACHE_FILE = "prev_day_candles.json"

# ---- Webhook ----
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SHARED_SECRET")
WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_RETRY_DELAY_SECONDS = 2

# FIX (overlapping-run guard): the rate-limit errors traced back to a run
# hitting getCandleData and immediately getting "exceeding access rate" on
# its very first call -- before it could have exhausted any limit itself.
# Angel One enforces rate limits per API key across ALL concurrent
# sessions, so the likely cause is a previous 5-min run still mid-retry
# (each retry now backs off up to 45s, so a run can legitimately take
# longer than the 5-min cron interval) overlapping with the next
# scheduled run and doubling up requests in the same window. A simple
# file-based lock stops a new run from starting while a previous one is
# still active, without needing any change to the GitHub Actions workflow
# YAML (concurrency: settings there are a good belt-and-suspenders
# addition too, but this guard works standalone).
LOCK_STALE_SECONDS = 240  # shorter than the 5-min cron interval, so a
                          # legitimately-running process won't block the
                          # *next* scheduled trigger, but a genuinely
                          # crashed run's stale lock still gets cleared.

NSE_HOLIDAYS_2026 = {
    dt.date(2026, 1, 26),
    dt.date(2026, 3, 3),
    dt.date(2026, 3, 26),
    dt.date(2026, 3, 31),
    dt.date(2026, 4, 3),
    dt.date(2026, 4, 14),
    dt.date(2026, 5, 1),
    dt.date(2026, 5, 28),
    dt.date(2026, 6, 26),
    dt.date(2026, 9, 14),
    dt.date(2026, 10, 2),
    dt.date(2026, 10, 20),
    dt.date(2026, 11, 10),
    dt.date(2026, 11, 24),
    dt.date(2026, 12, 25),
}

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
EOD_SQUAREOFF = dt.time(15, 20)
STRIKE_LOCK_TIME = dt.time(9, 30)

# ---- Pending-signal / entry-pricing tuning ----
PENDING_SIGNAL_MAX_CANDLES = 5   # resting window: candles N+1 .. N+5
ENTRY_LIMIT_DISCOUNT = 0.95      # limit = EMA5[N] * this
LOW_PREMIUM_SL_THRESHOLD = 99    # Rs. -- below this, SL floor kicks in
LOW_PREMIUM_SL_MIN_PCT = 0.10    # SL floor = entry_limit * (1 + this)
TARGET_RISK_REWARD = 2           # target = entry - RR * (SL - entry)

# ---- Buy-side signal engine (wired into main.py/state.json via
# buy_signal_engine.py: pending_buy_signal / open_buy_position) ----
BUY_PENDING_SIGNAL_MAX_CANDLES = 5   # resting window: candles N+1 .. N+5
BUY_TARGET_RISK_REWARD = 2           # target = entry + RR * (entry - SL)
