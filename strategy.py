"""
NIFTY weekly options strategy runner -- DATA + SIGNAL GENERATION ONLY.

Runs on GitHub Actions every 5 min during market hours.
1. Logs into Angel One (data APIs only, no static IP needed)
2. Resolves ATM strikes (short legs at ATM+-100, hedge legs at ATM+-400)
   via the official instrument master JSON (never guesses the symbol format)
3. Fetches 5-min candles, computes EMA5/EMA25/VWAP
4. Watches for a true EMA5-crosses-down-VWAP signal on a SHORT strike,
   waits for a limit fill (see ENTRY LOGIC below), and on fill sends BOTH
   legs of the credit spread (short SELL + hedge BUY) to the webhook in one
   payload
5. POSTs the signal to the webhook (which holds the Shoonya session and
   either simulates a paper fill or places a real order, depending on its
   own LIVE_MODE flag -- that decision lives in the webhook, not here)
6. Persists minimal state (pending signal awaiting fill, and current open
   position, for managing SL/target) to state.json, committed back to the
   repo by the GitHub Action.

This script must NEVER place a real order itself.

---
STRATEGY SPEC (current, agreed 2026-06-30 -- supersedes all earlier docstrings):

  STRIKES: short leg at ATM+-100 (premium collection), hedge leg at the
  SAME side's ATM+-400 (300 points further out, BUY, defined-risk hedge).
  This is a credit spread, NOT a directional "sell whichever side benefits"
  strategy -- EMA5/VWAP/EMA25 signal is checked independently per strike,
  with no directional bias. (An earlier docstring here incorrectly
  described a directional approach; that was never the actual logic and
  has been removed.)

  ENTRY:
    - Trigger on candle N: TRUE crossover -- EMA5 was >= VWAP on candle
      N-1, and is < VWAP on candle N -- AND EMA25 > EMA5 and EMA25 > VWAP
      on candle N. (State alone, e.g. "EMA5 < VWAP right now", is NOT a
      trigger -- it must be a fresh cross. This intentionally does not yet
      cover the "fires every candle while condition holds" re-entry bug
      for an already-open position; that's a separate, deliberately
      deferred item -- see NOT YET DONE below.)
    - Reference price for the limit: candle N+1's EMA5 estimated AT ITS
      OPEN, not at its close. Since EMA is only well-defined on closed
      candles, this is computed by rolling candle N's EMA5 forward one
      step using candle N+1's OPEN price in place of a close (see
      roll_ema5_forward()). This deliberately avoids using candle N's own
      (already-known, more lagging) EMA5, and avoids waiting for candle
      N+1 to fully close, both of which were observed in practice to set
      a limit price that a fast move would never retrace to.
    - limit_entry_price = that estimated EMA5 * 0.95. Computed ONCE
      (at candle N+1's open) and held fixed -- never recalculated.
    - The limit is monitored for fill over candles N+1 through N+5
      inclusive (5 candles, starting with the same candle whose open
      produced the price). If candle's high >= limit_entry_price during
      that window, the SELL fills at limit_entry_price. If the window
      elapses with no fill, the signal is cancelled -- no trade.
    - On fill, the hedge leg (same side, 300 points further OTM) is
      bought at MARKET (current LTP) -- no limit/precision needed for the
      hedge, confirmed acceptable by design.

  STOP-LOSS (computed once at fill, using the TRIGGER candle N -- not the
  fill candle):
    normal_sl_price = max(candle_N.high, vwap_at_candle_N)
    normal_sl_pct   = (normal_sl_price - entry_price) / entry_price
    sl_pct_final    = max(normal_sl_pct, 0.10)      # 10% floor, not a cap
    SL_price        = entry_price * (1 + sl_pct_final)

  TARGET: fixed 1:2 risk-reward (risk = SL_price - entry_price).
    target_price = entry_price - 2 * risk
    Trailing logic is intentionally deferred -- fixed target only for now.

NOT YET DONE (deliberately deferred):
  - True crossover detection above applies to NEW entries only. The
    existing re-entry/management path for an OPEN position still checks
    state, not transitions -- Pragnesh's explicit call is to keep
    collecting paper-trading case studies before touching that logic.
  - SL widening to VWAP+buffer / India VIX chop filter -- explicitly
    deferred pending more case studies.
  - Trailing target (only "1:2 fixed, revisit later" implemented).
  - A webhook HTTP 200 here still only means "the webhook accepted the
    request," NOT "Shoonya filled the order." Real fill confirmation has
    to happen inside the webhook (it holds the Shoonya session).
  - Startup-time reconciliation against actual broker positions.
  - Holiday-shift edge case for Tuesday expiry not yet handled.
"""

import json
import os
import sys
import time
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import angelone_client as ac

IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return dt.datetime.now(IST)


STATE_FILE = "state.json"
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # set as a GitHub Secret
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SHARED_SECRET")  # simple auth between GH Action and webhook

# Lot sizing -- LOT_SIZE is the exchange-defined unit size (verify
# periodically -- NSE revises this; confirmed 65 as of 2026-06-30, see
# NSE circular FAOP70616 / effective Jan 2026 revision). NUM_LOTS is how
# many lots we actually trade -- change ONLY this to scale position size
# up or down (1, 2, 3...).
LOT_SIZE = 65
NUM_LOTS = 1
QTY = LOT_SIZE * NUM_LOTS

ENTRY_WINDOW_CANDLES = 5          # candles N+1..N+5 to wait for a limit fill
ENTRY_LIMIT_DISCOUNT = 0.05       # limit price = ref EMA5 * (1 - this)
SL_FLOOR_PCT = 0.10                # minimum SL distance as % of entry price
HEDGE_OFFSET = 300                 # points between short strike and hedge strike

WEBHOOK_RETRY_ATTEMPTS = 3
# FIX (rate-limit backoff): previously a flat 3s delay between every retry.
# Now exponential -- 2s, then 5s, then 10s -- so repeated failures (e.g. a
# sustained webhook outage) back off instead of hammering at a fixed
# interval.
WEBHOOK_RETRY_DELAYS_SECONDS = [2, 5, 10]

# Static list of NSE trading holidays. Update yearly -- there is no reliable
# free API for this, so this has to be maintained by hand against NSE's
# published holiday calendar each December/January for the year ahead.
# Source: NSE circulars. VERIFY before relying on this near year-end.
# NSE trading holidays for 2026. Source: NSE official circular
# NSE/CMTR/71775, dated December 12, 2025 (verified directly against the
# circular text -- not a third-party aggregator). Weekend-falling holidays
# (e.g. Mahashivratri, Independence Day in 2026) are omitted since weekday
# filtering already excludes Sat/Sun; only listed here if they affect a
# weekday. Re-verify against NSE's site before each new year:
# https://www.nseindia.com/resources/exchange-communication-holidays
NSE_HOLIDAYS_2026 = {
    dt.date(2026, 1, 26),   # Republic Day
    dt.date(2026, 3, 3),    # Holi
    dt.date(2026, 3, 26),   # Shri Ram Navami
    dt.date(2026, 3, 31),   # Shri Mahavir Jayanti
    dt.date(2026, 4, 3),    # Good Friday
    dt.date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    dt.date(2026, 5, 1),    # Maharashtra Day
    dt.date(2026, 5, 28),   # Bakri Id
    dt.date(2026, 6, 26),   # Muharram
    dt.date(2026, 9, 14),   # Ganesh Chaturthi
    dt.date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    dt.date(2026, 10, 20),  # Dussehra
    dt.date(2026, 11, 10),  # Diwali-Balipratipada
    dt.date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    dt.date(2026, 12, 25),  # Christmas
    # Note: Nov 8, 2026 (Diwali Laxmi Pujan) falls on a Sunday and is
    # already excluded by weekday filtering -- not a weekday trading
    # holiday, so deliberately not listed here despite being notable.
}

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
EOD_SQUAREOFF = dt.time(15, 20)

# Strikes must be locked using the 9:30 AM spot price specifically, not
# just "whatever the first run of the day happens to see" -- the market
# opens at 9:15, so without this explicit gate a run at 9:15-9:29 would
# lock strikes ~15 min too early, off a less-settled opening-range price.
STRIKE_LOCK_TIME = dt.time(9, 30)


PREV_DAY_CACHE_FILE = "prev_day_candles.json"


def load_prev_day_cache(expected_date):
    """
    Loads the cached previous-trading-day candles, keyed by token. Returns
    an empty dict if the cache file doesn't exist, is unreadable, or was
    built for a different date than expected_date (e.g. it's a new day, so
    yesterday's cache is now stale and needs replacing).

    This exists so we fetch each previous-day session from Angel One only
    ONCE (the first run of the day, per token), instead of re-downloading
    a 2-day candle range on every single 5-min tick -- which is what was
    causing repeated rate-limit errors after the EMA warm-up fix increased
    the per-call payload size.
    """
    if not os.path.exists(PREV_DAY_CACHE_FILE):
        return {}
    try:
        with open(PREV_DAY_CACHE_FILE, "r") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if cache.get("date") != expected_date.isoformat():
        return {}  # stale -- different trading day than expected
    return cache.get("candles_by_token", {})


def save_prev_day_cache(date, candles_by_token):
    with open(PREV_DAY_CACHE_FILE, "w") as f:
        json.dump({"date": date.isoformat(), "candles_by_token": candles_by_token}, f)


def get_candles_with_cache(smart_api, token, prev_day, prev_day_cache, today_start):
    """
    Returns combined (previous trading day + today) candles for `token`,
    fetching only what's not already cached:
      - Previous day's candles: served from prev_day_cache if present for
        this token; otherwise fetched once (just that one day's session,
        not a 2-day range) and added to the cache dict (caller is
        responsible for persisting it back to disk at the end of the run).
      - Today's candles: always fetched fresh (small, cheap -- just since
        today's market open, not 2 days), since they change every run.
    """
    token_str = str(token)

    if token_str in prev_day_cache:
        prev_candles = prev_day_cache[token_str]
    else:
        prev_day_start = dt.datetime.combine(prev_day, MARKET_OPEN)
        prev_candles = ac.fetch_5min_candles(smart_api, token, start_time=prev_day_start)
        prev_day_cache[token_str] = prev_candles

    today_candles = ac.fetch_5min_candles(smart_api, token, start_time=today_start)

    # Avoid double-counting if Angel One's "previous day" response happens
    # to include any of today's candles (shouldn't, given the date-bounded
    # query, but de-dupe by timestamp defensively).
    seen_times = {c["time"] for c in prev_candles}
    merged = list(prev_candles) + [c for c in today_candles if c["time"] not in seen_times]
    return merged


def is_trading_day(date):
    return date.weekday() < 5 and date not in NSE_HOLIDAYS_2026


def previous_trading_day(date):
    """
    Walks backward from `date` to find the most recent prior trading day
    (skips weekends and NSE holidays). Used so EMA5/EMA25 can warm up using
    real history from the last session, instead of restarting cold every
    morning -- VWAP still resets at today's market open separately, since
    that's how VWAP is correctly defined.
    """
    d = date - dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def is_market_open_now():
    n = now_ist()
    if not is_trading_day(n.date()):
        return False
    return MARKET_OPEN <= n.time() <= MARKET_CLOSE


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "open_position": None,
        "pending_signal": None,
        "last_run": None,
        "heartbeat_date": None,
        "daily_strikes_date": None,
        "daily_atm": None,
        "daily_strikes": None,
    }


def save_state(state):
    state["last_run"] = now_ist().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_or_set_daily_strikes(state, smart_api):
    """
    Strikes are anchored to the SPOT PRICE AT 9:30 AM SPECIFICALLY -- not
    just "whenever the first run of the day happens to land." Recalculating
    ATM every run would chase a moving target; locking off a pre-9:30 run
    (market opens 9:15) would anchor to a less-settled opening-range price
    instead of the intended reference point.

    Snapshots ATM (and the 4 strikes derived from it -- ATM-400, ATM-100,
    ATM+100, ATM+400) once, on the first run of each trading day at/after
    9:30, and persists it in state.json so every later run that day reuses
    the SAME strikes regardless of how much spot drifts afterward.
    ATM+-100 are the SHORT (signal-checked, premium-selling) strikes;
    ATM+-400 are the HEDGE strikes (bought opposite the short, same side,
    on fill -- see find_hedge_strike()). Returns
    (atm, strikes_to_check, spot_price_at_lock).

    If today's snapshot already exists, returns it without calling
    fetch_spot_ltp again -- saves an API call on every run after the first.
    spot_price_at_lock is only non-None on the run that actually set it
    (useful for logging); later runs return None for that third value
    since they didn't re-fetch it.
    """
    today_str = now_ist().date().isoformat()

    if state.get("daily_strikes_date") == today_str and state.get("daily_strikes"):
        return state["daily_atm"], state["daily_strikes"], None

    if now_ist().time() < STRIKE_LOCK_TIME:
        # Too early to lock yet -- this run still does heartbeat/housekeeping
        # in main(), it just can't check for entry signals until strikes
        # are locked at/after 9:30.
        return None, None, None

    spot_price = ac.fetch_spot_ltp(smart_api)
    if spot_price is None:
        return None, None, None

    atm = get_atm_strike(spot_price)
    strikes = [atm - 400, atm - 100, atm + 100, atm + 400]

    state["daily_strikes_date"] = today_str
    state["daily_atm"] = atm
    state["daily_strikes"] = strikes

    return atm, strikes, spot_price


def find_hedge_strike(short_strike, atm):
    """
    Maps a SHORT strike (ATM-100 or ATM+100) to its HEDGE strike (300
    points further out, same side): ATM-100 -> ATM-400, ATM+100 -> ATM+400.
    Returns None if short_strike isn't one of the two recognized short
    strikes (defensive -- should never happen given how strikes_to_check
    is built).
    """
    if short_strike == atm - 100:
        return atm - 400
    if short_strike == atm + 100:
        return atm + 400
    return None


def is_short_strike(strike, atm):
    return strike in (atm - 100, atm + 100)


def send_heartbeat_if_needed(state):
    """
    Sends a one-line 'I'm alive' heartbeat through the webhook (which relays
    it to Telegram) once per trading day, on whichever run is the first one
    to successfully reach this point that day.
    """
    today_str = now_ist().date().isoformat()
    if state.get("heartbeat_date") == today_str:
        return  # already sent today

    send_to_webhook({
        "action": "HEARTBEAT",
        "message": f"Signal generator alive -- first successful run today at {now_ist().strftime('%H:%M:%S')} IST.",
        "time": now_ist().isoformat(),
    })
    state["heartbeat_date"] = today_str


def get_current_weekly_expiry():
    """
    Returns the current weekly NIFTY expiry date, adjusted for holidays.
    NSE shifted NIFTY's weekly expiry from Thursday to TUESDAY, effective
    September 1, 2025. If a computed Tuesday is a market holiday, NSE
    shifts the expiry to the previous trading day -- walk backward until a
    real trading day is found.
    """
    today = now_ist().date()
    days_ahead = (1 - today.weekday()) % 7  # 1 = Tuesday
    expiry = today + dt.timedelta(days=days_ahead)

    while not is_trading_day(expiry):
        expiry -= dt.timedelta(days=1)

    return expiry


def get_atm_strike(spot_price, step=50):
    return round(spot_price / step) * step


def compute_indicators(candles):
    """
    EMA5/EMA25 computed across the FULL fetched range (today + previous
    trading day) so they're warmed up by today's market open, instead of
    restarting cold every morning. VWAP is computed only on today's rows
    and resets at today's market open (correct VWAP definition).

    Returns the full warmed-up DataFrame (today's rows only, but with EMA
    columns that reflect the full history) plus separately exposes
    ema5_alpha so callers can roll EMA5 forward by one step without
    re-running the whole pandas computation (see roll_ema5_forward()).
    """
    df = pd.DataFrame(candles)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    today = now_ist().date()

    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema25"] = df["close"].ewm(span=25, adjust=False).mean()

    today_mask = df["time"].dt.date == today
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].where(today_mask, 0).cumsum()
    cum_tp_vol = (typical_price * df["volume"]).where(today_mask, 0).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, pd.NA)

    df = df[today_mask].reset_index(drop=True)

    if len(df) < 2:
        # Not enough of today's session yet to check a signal meaningfully,
        # even though EMA itself is already warmed up from yesterday.
        return None

    return df


def roll_ema5_forward(prev_ema5, new_price, span=5):
    """
    Rolls a closed-candle EMA5 value forward by exactly one step using
    `new_price` in place of that next candle's close -- this is how we
    estimate "EMA5 at the OPEN of the next candle" without waiting for
    that candle to actually close.

    Standard EWM step: new_ema = price * alpha + prev_ema * (1 - alpha),
    alpha = 2 / (span + 1). This is the same recursive formula pandas'
    .ewm(span=..., adjust=False) uses internally, applied manually for one
    extra step with an open price instead of a close price.

    Rationale (see STRATEGY SPEC above): using the trigger candle's own
    EMA5 (more lagging) as the entry reference was found in practice to
    set limit prices that fast/large moves never retraced to, leaving
    entries permanently unfilled while the underlying option decayed to
    near zero. Rolling forward with the next candle's open reacts to the
    move immediately instead of lagging it.
    """
    alpha = 2.0 / (span + 1)
    return new_price * alpha + prev_ema5 * (1 - alpha)


def check_crossover_trigger(df):
    """
    TRUE crossover only -- NOT just "EMA5 is currently below VWAP" (that
    was the old, buggy state-based check, which can be satisfied for many
    consecutive candles after the actual cross and re-fire continuously).
    Requires at least 2 of today's rows so a previous candle exists to
    compare against.

    Returns the trigger candle (as a dict) if candle N is a fresh
    downward cross with EMA25 confirmation, else None.
    """
    if df is None or len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    crossed_down = (prev["ema5"] >= prev["vwap"]) and (curr["ema5"] < curr["vwap"])
    ema25_confirms = (curr["ema25"] > curr["ema5"]) and (curr["ema25"] > curr["vwap"])

    if crossed_down and ema25_confirms:
        return curr.to_dict()
    return None


def compute_sl_price(entry_price, trigger_high, trigger_vwap):
    """
    SL = max(trigger candle's high, VWAP at trigger), expressed as a %
    distance from entry, with a 10% FLOOR (not a cap) -- if the normal
    distance is already >= 10%, it's used as-is; if it's tighter than 10%,
    10% is used instead. E.g. entry 89 -> SL >= 97.9 (~98); entry 67 ->
    SL >= 73.7 (~74).
    """
    normal_sl_price = max(trigger_high, trigger_vwap)
    normal_sl_pct = (normal_sl_price - entry_price) / entry_price
    sl_pct_final = max(normal_sl_pct, SL_FLOOR_PCT)
    return entry_price * (1 + sl_pct_final)


def is_eod_squareoff_time():
    return now_ist().time() >= EOD_SQUAREOFF


def send_to_webhook(payload):
    """
    Retries with EXPONENTIAL backoff (2s, 5s, 10s) instead of a flat delay
    -- a sustained webhook outage backs off instead of hammering at a fixed
    interval. Still returns None on total failure; callers MUST check for
    that.
    """
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set -- skipping webhook call. Payload was:", payload)
        return None

    headers = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        headers["X-Webhook-Secret"] = WEBHOOK_SECRET

    last_err = None
    for attempt in range(1, WEBHOOK_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=15)
            print(f"Webhook response [{resp.status_code}] (attempt {attempt}): {resp.text}")
            return resp
        except requests.RequestException as e:
            last_err = e
            print(f"Webhook call failed (attempt {attempt}/{WEBHOOK_RETRY_ATTEMPTS}): {e}",
                  file=sys.stderr)
            if attempt < WEBHOOK_RETRY_ATTEMPTS:
                delay = WEBHOOK_RETRY_DELAYS_SECONDS[min(attempt - 1, len(WEBHOOK_RETRY_DELAYS_SECONDS) - 1)]
                time.sleep(delay)

    print(f"Webhook call failed after {WEBHOOK_RETRY_ATTEMPTS} attempts: {last_err}",
          file=sys.stderr)
    return None


def webhook_confirmed_ok(resp):
    """
    A 200 here means "the webhook received and processed the request," not
    "Shoonya filled the order." True fill-confirmation has to be built on
    the webhook side (it holds the Shoonya session).
    """
    if resp is None:
        return False
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code == 200 and body.get("status") == "ok"


def manage_open_position(state, smart_api, instruments, expiry, prev_day, prev_day_cache, today_start):
    """
    Checks the open position's SHORT leg for SL/target/EOD exit. Exit
    decisions are based on the short leg only (the hedge leg's job is risk
    containment, not signal generation). On confirmed exit, sends an EXIT
    for both legs so the webhook can square off the whole spread.
    """
    pos = state["open_position"]
    short = pos["short_leg"]
    token_info = ac.resolve_option_token(instruments, expiry, short["strike"], short["option_type"])

    if not token_info:
        save_state(state)
        save_prev_day_cache(prev_day, prev_day_cache)
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)

    if df is not None:
        last = df.iloc[-1]
        sl_hit = last["close"] > pos["sl_price"]
        target_hit = last["close"] <= pos["target_price"]

        if sl_hit or target_hit or is_eod_squareoff_time():
            reason = "SL" if sl_hit else ("TARGET" if target_hit else "EOD")
            resp = send_to_webhook({
                "action": "EXIT",
                "reason": reason,
                "short_leg": {"symbol": short["symbol"], "qty": short["qty"], "price": float(last["close"])},
                "hedge_leg": {"symbol": pos["hedge_leg"]["symbol"], "qty": pos["hedge_leg"]["qty"]},
                "time": now_ist().isoformat(),
            })

            if webhook_confirmed_ok(resp):
                state["open_position"] = None
            else:
                print(
                    "EXIT webhook not confirmed -- leaving open_position "
                    "in state.json so the next run retries.",
                    file=sys.stderr,
                )

    save_state(state)
    save_prev_day_cache(prev_day, prev_day_cache)


def manage_pending_signal(state, smart_api, instruments, expiry, atm, prev_day, prev_day_cache, today_start):
    """
    Advances a pending signal (awaiting limit fill) by exactly one step
    per run:
      1. If we're still waiting on candle N+1's open to compute the limit
         price, compute it now from this run's latest candle (treated as
         candle N+1) and start the fill-monitoring window on this same
         candle.
      2. Otherwise, check this run's latest candle against the
         already-fixed limit price.
      3. On fill: compute SL/target, buy the hedge leg at market, send the
         combined ENTRY payload, and (on confirmation) set open_position.
      4. If the window (5 candles) elapses with no fill, cancel the
         pending signal -- no trade.
    Returns True if it consumed this run (so main() should not also scan
    for new signals this run), False otherwise.
    """
    pending = state["pending_signal"]
    strike = pending["strike"]
    option_type = pending["option_type"]

    token_info = ac.resolve_option_token(instruments, expiry, strike, option_type)
    if not token_info:
        return True  # can't proceed this run, but don't abandon the pending signal

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)
    if df is None or len(df) < 1:
        return True

    latest = df.iloc[-1]

    if pending["limit_entry_price"] is None:
        # This run's latest candle is candle N+1 -- compute the limit
        # price from its OPEN, rolling the trigger candle's EMA5 forward.
        rolled_ema5 = roll_ema5_forward(pending["trigger_ema5"], latest["open"])
        limit_entry_price = rolled_ema5 * (1 - ENTRY_LIMIT_DISCOUNT)
        pending["limit_entry_price"] = limit_entry_price
        pending["candles_elapsed"] = 0
        print(f"Pending signal {strike}{option_type}: limit price set to {limit_entry_price:.2f} "
              f"(rolled EMA5 {rolled_ema5:.2f} from open {latest['open']:.2f})")

    pending["candles_elapsed"] += 1
    limit_entry_price = pending["limit_entry_price"]

    filled = latest["high"] >= limit_entry_price

    if filled:
        entry_price = limit_entry_price
        sl_price = compute_sl_price(entry_price, pending["trigger_high"], pending["trigger_vwap"])
        risk = sl_price - entry_price
        target_price = entry_price - 2 * risk

        hedge_strike = find_hedge_strike(strike, atm)
        hedge_token_info = ac.resolve_option_token(instruments, expiry, hedge_strike, option_type)

        short_leg = {
            "symbol": token_info["symbol"], "token": token_info["token"],
            "strike": strike, "option_type": option_type,
            "side": "SELL", "price": entry_price, "qty": QTY,
        }
        # Hedge is bought at MARKET, not a limit -- no price needed/sent
        # here; the webhook (which holds the Shoonya session) places it as
        # a market order using symbol/token alone.
        hedge_leg = {
            "symbol": hedge_token_info["symbol"] if hedge_token_info else None,
            "token": hedge_token_info["token"] if hedge_token_info else None,
            "strike": hedge_strike, "option_type": option_type,
            "side": "BUY", "qty": QTY,
        }

        resp = send_to_webhook({
            "action": "ENTRY",
            "short_leg": short_leg,
            "hedge_leg": hedge_leg,
            "sl_price": sl_price,
            "target_price": target_price,
            "time": now_ist().isoformat(),
        })

        if webhook_confirmed_ok(resp):
            state["open_position"] = {
                "short_leg": short_leg,
                "hedge_leg": hedge_leg,
                "sl_price": sl_price,
                "target_price": target_price,
                "entry_time": now_ist().isoformat(),
            }
            state["pending_signal"] = None
            print(f"FILLED: SELL {short_leg['symbol']} @ {entry_price:.2f}, "
                  f"hedge BUY {hedge_leg['symbol']}, SL={sl_price:.2f}, target={target_price:.2f}")
        else:
            print(
                f"ENTRY webhook not confirmed for {short_leg['symbol']} -- "
                "NOT recording as open position. Pending signal cleared (do not retry a stale fill).",
                file=sys.stderr,
            )
            state["pending_signal"] = None

    elif pending["candles_elapsed"] >= ENTRY_WINDOW_CANDLES:
        print(f"Pending signal {strike}{option_type}: {ENTRY_WINDOW_CANDLES}-candle window elapsed, "
              f"limit {limit_entry_price:.2f} never reached -- cancelling, no trade.")
        state["pending_signal"] = None

    return True


def scan_for_new_signal(state, smart_api, instruments, expiry, strikes_to_check, atm,
                         prev_day, prev_day_cache, today_start):
    """
    Checks each SHORT strike (ATM+-100 only -- hedge strikes ATM+-400 are
    never signal-checked, they only get bought as a hedge on a short's
    fill) for a fresh crossover trigger. On the first trigger found,
    records a pending_signal (NOT yet a fill) and returns -- one signal is
    pursued at a time.
    """
    for strike in strikes_to_check:
        if not is_short_strike(strike, atm):
            continue  # hedge strikes are never signal sources

        for opt_type in ["CE", "PE"]:
            token_info = ac.resolve_option_token(instruments, expiry, strike, opt_type)
            if not token_info:
                continue

            candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
            df = compute_indicators(candles)
            if df is None:
                continue

            trigger_candle = check_crossover_trigger(df)
            if trigger_candle:
                state["pending_signal"] = {
                    "strike": strike,
                    "option_type": opt_type,
                    "trigger_time": str(trigger_candle["time"]),
                    "trigger_high": trigger_candle["high"],
                    "trigger_vwap": trigger_candle["vwap"],
                    "trigger_ema5": trigger_candle["ema5"],
                    "limit_entry_price": None,   # computed next run, from N+1's open
                    "candles_elapsed": 0,
                }
                print(f"SIGNAL: crossover trigger on {strike}{opt_type} at {trigger_candle['time']} "
                      f"-- awaiting next candle's open to set limit price.")
                save_state(state)
                save_prev_day_cache(prev_day, prev_day_cache)
                return


def main():
    if not is_market_open_now():
        print("Market closed (outside hours or holiday) -- skipping run.")
        return

    state = load_state()
    state.setdefault("pending_signal", None)
    send_heartbeat_if_needed(state)
    save_state(state)  # persist heartbeat_date immediately, don't wait for end of run

    smart_api = ac.login()
    instruments = ac.download_instrument_master()

    expiry = get_current_weekly_expiry()

    atm, strikes_to_check, spot_price_at_lock = get_or_set_daily_strikes(state, smart_api)
    if atm is None:
        if now_ist().time() < STRIKE_LOCK_TIME:
            print(f"Before {STRIKE_LOCK_TIME.strftime('%H:%M')} strike-lock time -- skipping signal check this run.")
        else:
            print("Could not fetch spot price to lock today's strikes, aborting this run.", file=sys.stderr)
        return
    if spot_price_at_lock is not None:
        print(f"Locked today's strikes: ATM={atm} (spot={spot_price_at_lock}) -> {strikes_to_check}")

    prev_day = previous_trading_day(now_ist().date())
    prev_day_cache = load_prev_day_cache(prev_day)
    today_start = dt.datetime.combine(now_ist().date(), MARKET_OPEN)

    save_state(state)  # persist today's locked strikes immediately

    # ---- Priority 1: manage an existing open position ----
    if state["open_position"] is not None:
        manage_open_position(state, smart_api, instruments, expiry, prev_day, prev_day_cache, today_start)
        return

    # ---- Priority 2: advance a pending signal awaiting fill ----
    if state["pending_signal"] is not None:
        manage_pending_signal(state, smart_api, instruments, expiry, atm, prev_day, prev_day_cache, today_start)
        save_state(state)
        save_prev_day_cache(prev_day, prev_day_cache)
        return

    # ---- Priority 3: no position, no pending signal -- scan for a new trigger ----
    scan_for_new_signal(state, smart_api, instruments, expiry, strikes_to_check, atm,
                         prev_day, prev_day_cache, today_start)

    save_state(state)
    save_prev_day_cache(prev_day, prev_day_cache)
    print("No new trigger this run.")


if __name__ == "__main__":
    main()
