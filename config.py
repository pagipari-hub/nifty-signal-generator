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

# NEW (2026-08-12, sell-side squeeze gate): a fresh sell crossover is
# blocked -- not entered -- if spread_atr_ratio (see
# indicators.compute_squeeze_metrics()) on the trigger candle is below
# this cutoff. Diagnostics themselves (log_signal_debug's [squeeze diag]
# line) remain UNCONDITIONAL -- this constant only affects the entry
# decision, not what gets logged.
SELL_SQUEEZE_SPREAD_ATR_MIN = 0.5

# NEW (2026-08-13, sell-side scan cutoff): mirrors the old buy-side
# late-day cutoff but for the sell side -- a fresh sell crossover or PDL
# breakdown (see PDL_MIN_PREV_DAY_VOLUME below) is not allowed to open a
# NEW pending_signal at/after this time. Set slightly later than the old
# buy-side 14:45 cutoff (14:55) because a sell entry only needs to clear
# PENDING_SIGNAL_MAX_CANDLES (5 candles, ~25 min) before EOD_SQUAREOFF
# (15:20). Does NOT apply to managing an already-resting pending_signal
# or an already-open open_position -- those keep being checked every run
# regardless of time, right up through the existing EOD_SQUAREOFF
# handling. See calendar_utils.is_before_sell_scan_cutoff().
SELL_SCAN_CUTOFF_TIME = dt.time(14, 55)

# NEW (2026-08-13, PDL fallback entry -- SELL side only, both PE and CE
# legs). Design agreed with Pragnesh: when neither leg's EMA/VWAP
# crossover fires, a sell entry can instead trigger off that leg's own
# option premium closing below its previous trading day's low (PDL) --
# a confirmed-close breakdown, not just an intrabar touch. EMA/VWAP
# always takes priority on a same-run collision (checked first in
# signal_engine.scan_for_new_signal()); PDL is the fallback, sharing the
# SAME single pending_signal/open_position slot per leg (mutually
# exclusive with an EMA/VWAP signal on that leg, not a second parallel
# signal). SL/target/low-premium-floor formulas are shared with the
# existing sell-side constants above (LOW_PREMIUM_SL_THRESHOLD,
# LOW_PREMIUM_SL_MIN_PCT, TARGET_RISK_REWARD) -- no separate constants
# needed for those; see pending.compute_pending_signal_pdl().
#
# PDL itself is locked ONCE per day, immediately after the 9:30 strike
# lock (see candle_priming.lock_daily_pdl()), reusing the SAME
# previous-day candle fetch candle_priming.py already performs for
# EMA25 warm-up (full prior session, MARKET_OPEN-MARKET_CLOSE) -- zero
# additional API calls. Because ATM (and therefore which strikes are
# "today's" sell legs) can shift day to day, a given day's locked strike
# may have been deep OTM and thinly traded on the PREVIOUS day -- its
# previous-day low in that case isn't a meaningful support level, just
# noise from a barely-traded contract. This threshold gates that: if a
# leg's total previous-day volume (summed across the full session's
# candles) falls under this, PDL fallback is disabled for that leg for
# the day (falls back to EMA/VWAP-only, unchanged) rather than trading
# off an unreliable level. Provisional starting value, NOT yet
# calibrated against real paper data -- revisit once enough days of
# actual daily_pdl volume totals have been logged and observed, same
# "log first, calibrate later" pattern as SELL_SQUEEZE_SPREAD_ATR_MIN
# above.
PDL_MIN_PREV_DAY_VOLUME = 50000

# NEW (2026-09-15, sell-side trailing Supertrend SL): Pragnesh's call --
# replace the sell side's fixed SL (max(trigger high, trigger VWAP), or
# the trigger candle's own high for a PDL-sourced signal) with a
# trailing Supertrend(10,2) SL once a position is OPEN. Applies
# uniformly regardless of which entry path produced the position --
# EMA/VWAP or PDL fallback (see pending.compute_pending_signal_pdl()) --
# since exit management is orthogonal to how the signal was triggered.
# Deliberately a SEPARATE ATR period from the existing ATR_PERIOD (14)
# in indicators.py, which feeds squeeze diagnostics only -- reusing that
# period here would silently change squeeze-diagnostic values too, which
# is not the intent.
#
# Trailing behavior: recomputed every run while a position is open,
# monotonic-tighten only (for a short, "tighter" means the SL value
# DECREASES toward price, since SL sits above price) -- new_sl =
# min(current_sl, live_supertrend_upper_band), never loosens. No floor
# at entry price -- the trailing SL is allowed to drop below entry
# (locks in guaranteed profit on stop-out); intentional, not a bug. See
# position.manage_spread_exit() for where this is applied.
#
# Seeding: at fill time, the position's initial sl_price is seeded from
# the LIVE Supertrend upper-band value computed on the fill candle (not
# from the pending signal's already-locked SL, whichever formula
# produced it) -- Pragnesh's explicit call. While a signal is still
# PENDING (unfilled), SL display/logic is UNCHANGED (still whatever
# compute_pending_signal() / compute_pending_signal_pdl() computed) --
# Supertrend trailing only starts once the position is actually open.
#
# Target: target_price is still computed and stored (see
# TARGET_RISK_REWARD note above) but is no longer used to decide exits --
# an open sell position now exits ONLY via trailing-SL touch or EOD
# square-off. KNOWN CROSS-REPO GAP: the webhook server (separate repo,
# not visible/editable from here) still receives target_price in the
# MANAGE_SPREAD payload and may still be checking it server-side for the
# bracket decision -- until that side is updated to ignore target_price
# for sell-side exits, a position could still close early on a target
# touch there even though this repo's own local pre-check no longer
# treats target as an exit condition. Flagged, not yet resolved.
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 2
