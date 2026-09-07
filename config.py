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
# this cutoff. Pragnesh's call, 2026-08-12: sell-side ONLY (buy side
# stays squeeze-free by design, gated on time-of-day instead -- see
# BUY_SCAN_WINDOWS below). Picked as a hard threshold, not shadow-mode,
# per Pragnesh's explicit choice on 2026-08-12 -- single real data point
# behind it so far (0.017 on the 2026-07-30 NIFTY04AUG2624350CE whipsaw
# case study in logging_utils.py's docstring); revisit once more
# paper-mode sell-side squeeze data accumulates. Diagnostics themselves
# (log_signal_debug's [squeeze diag] line) remain UNCONDITIONAL -- this
# constant only affects the entry decision, not what gets logged.
SELL_SQUEEZE_SPREAD_ATR_MIN = 0.5

# NEW (2026-08-12, buy-side scan time windows): Pragnesh's call, same
# session as the sell-side squeeze gate above -- buy side stays
# squeeze-free by design (buy signals are the lagged EMA5/VWAP
# confirmation of a move that's already partly played out, per
# BUY_ENGINE_INTEGRATION.md section 1 -- squeeze filtering doesn't apply
# the same way here), but is instead restricted to two intraday windows
# for NEW signal scanning only. Excludes the 11:45-13:30 midday lull and
# anything after 14:45. Does NOT apply to managing an already-resting
# pending_buy_signal or an already-open open_buy_position -- those keep
# being monitored every run regardless of time, same as the sell side's
# always-run reasoning for open positions (see
# BUY_ENGINE_INTEGRATION.md section 4). Diagnostic squeeze logging in
# buy_signal_engine.py remains unconditional, unaffected by this window.
BUY_SCAN_WINDOWS = [
    (dt.time(9, 30), dt.time(11, 45)),
    (dt.time(13, 30), dt.time(14, 45)),
]

# NEW (2026-08-14, buy-side squeeze gate -- REVERSES the earlier
# "buy stays squeeze-free by design" call above). Real paper-mode
# evidence overturned that assumption: two same-day NIFTY18AUG2624200PE
# buy trades (11:16 entry, spread_atr_ratio=0.50; 11:36 re-entry,
# spread_atr_ratio=0.35) both fired cleanly (genuine fresh crossover,
# setup correctly formed per is_fresh_crossover_signal_buy()) while
# EMA5/EMA25/VWAP were still tightly bunched, and both were stopped out
# within 5-14 candles for near-identical losses (-244.03, -243.95) --
# the same tight-SL-from-tight-bunching mechanism as the sell-side
# whipsaw case study, just reached via a different path than assumed.
# The original reasoning (EMA5's lag already filters squeeze-driven
# noise before a buy signal can fire) turned out not to hold: the lag
# changes WHICH candle a signal fires on, not whether the three lines
# are still bunched at that later candle. Pragnesh's call, 2026-08-14:
# mirror the sell-side threshold/mechanism exactly (same 0.5 cutoff,
# same hard-gate-not-shadow-mode posture, same continue-not-return
# behavior) rather than inventing a separate buy-specific value --
# revisit once more paper-mode buy-side squeeze data accumulates, same
# as the sell-side constant's own note. Applied only in
# scan_for_new_buy_signal_live() (the live-wired scan) -- see that
# function's docstring for placement. Squeeze-diagnostic logging itself
# (the [squeeze diag] line already printed unconditionally in both the
# standalone and live buy scans) is unaffected -- this constant only
# gates the entry decision.
#
# UNCHANGED by the 2026-09-04 ROC rework: this gate operates on
# EMA5/EMA25/VWAP bunching, orthogonal to whatever generates the
# crossover -- still checked in scan_for_new_buy_signal_live() exactly
# where it was before (see buy_signal_engine.py's top docstring).
BUY_SQUEEZE_SPREAD_ATR_MIN = 0.5

# NEW (2026-08-13, sell-side scan cutoff): mirrors BUY_SCAN_WINDOWS's
# late-day cutoff but for the sell side -- a fresh sell crossover or PDL
# breakdown (see PDL_MIN_PREV_DAY_VOLUME below) is not allowed to open a
# NEW pending_signal at/after this time. Set slightly later than the buy
# side's 14:45 cutoff (14:55) because a sell entry only needs to clear
# PENDING_SIGNAL_MAX_CANDLES (5 candles, ~25 min) before EOD_SQUAREOFF
# (15:20), whereas a filled buy position needs to survive as a live,
# unmonitored-except-every-5-min position for longer -- so the sell side
# can safely scan a little closer to the close. Does NOT apply to
# managing an already-resting pending_signal or an already-open
# open_position -- those keep being checked every run regardless of time,
# right up through the existing EOD_SQUAREOFF handling. See
# calendar_utils.is_before_sell_scan_cutoff().
SELL_SCAN_CUTOFF_TIME = dt.time(14, 55)

# NEW (2026-07-31): buy-side counterpart to the two constants above --
# same threshold/pct (Pragnesh's call), applied in the opposite direction
# since buy's SL sits BELOW entry, not above. See
# buy_signal_engine.compute_pending_buy_signal()'s docstring for why this
# was missing until now (all 7 paper buy trades to date hit SL, several
# on sub-Rs.99 premiums, with no floor at all on that side).
#
# UNCHANGED by the 2026-09-04 ROC rework -- still applied on top of the
# new prev-candle-low SL formula, same widen-only contract.
BUY_LOW_PREMIUM_SL_THRESHOLD = 99   # Rs. -- below this, SL floor kicks in
BUY_LOW_PREMIUM_SL_MIN_PCT = 0.10   # SL floor = entry_limit * (1 - this)

# ---- Buy-side signal engine (wired into main.py/state.json via
# buy_signal_engine.py: pending_buy_signal / open_buy_position) ----
BUY_PENDING_SIGNAL_MAX_CANDLES = 5   # resting window: candles N+1 .. N+5

# NEW (2026-09-04, ROC replacement): entry trigger is now
# ROC(BUY_ROC_PERIOD) on 5-min candle closes crossing above
# BUY_ROC_CROSS_LEVEL, with the additional filter that the trigger
# candle's own close must be above VWAP. Replaces the EMA5/EMA25/VWAP
# crossover entirely on the buy side -- see buy_signal_engine.py's
# module docstring and is_fresh_crossover_signal_buy() for the full
# writeup. Period/level match the TradingView ROC settings Pragnesh
# shared (Length=18); the "cross above 1" level is Pragnesh's explicit
# choice, not TradingView's default zero-line cross. Does NOT affect or
# replace BUY_SQUEEZE_SPREAD_ATR_MIN above -- that gate still applies on
# top of this new trigger, unchanged.
BUY_ROC_PERIOD = 18
BUY_ROC_CROSS_LEVEL = 1.0

# CHANGED (2026-09-04, ROC replacement): target risk:reward tightened
# from 1:2 to 1:1 as part of the same rework -- Pragnesh's explicit call
# alongside the new ROC trigger and the new SL formula (prior candle's
# low, see buy_signal_engine.compute_pending_buy_signal()). Was 2.
BUY_TARGET_RISK_REWARD = 1           # target = entry + RR * (entry - SL)

# NEW (2026-09-07, second buy trigger -- wide-spread EMA crossover):
# Pragnesh's call -- an ADDITIONAL, independent buy trigger alongside
# the ROC trigger above (both active; whichever fires first for a given
# leg in a given run wins, since there's only one pending_buy_signal
# slot). Fires when EMA5 has a FRESH crossover above BOTH EMA25 and
# VWAP (mirrors the pre-ROC buy condition, but without requiring EMA25
# below VWAP first) AND spread_atr_ratio (indicators.compute_squeeze_metrics(),
# the same metric the squeeze gates use in the opposite direction) is
# ABOVE this threshold -- i.e. EMA5/EMA25/VWAP are WIDE apart, not
# bunched. This is deliberately the inverse condition from
# BUY_SQUEEZE_SPREAD_ATR_MIN (which blocks the ROC trigger when spread
# is too TIGHT, <0.5) -- here a wide spread is required before the EMA
# crossover trigger is allowed to fire at all, on the theory that a
# clean EMA5 breakout away from EMA25/VWAP (not a whipsaw-prone tight
# bunch) is what makes this specific crossover trustworthy. See
# buy_signal_engine.py's is_fresh_crossover_signal_buy_ema() for the
# full condition and compute_pending_buy_signal_ema_squeeze() for how
# entry is priced (40% OHLC pullback, NOT the ROC trigger's close-based
# entry). SL/target formulas are otherwise identical to the ROC
# trigger's (prior candle's low + low-premium floor, 1:1 RR).
BUY_EMA_CROSSOVER_SPREAD_ATR_MIN = 2.0

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
