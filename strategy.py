"""
NIFTY weekly options strategy runner -- DATA + SIGNAL GENERATION ONLY.

Runs on GitHub Actions every 5 min during market hours.
1. Logs into Angel One (data APIs only, no static IP needed)
2. Resolves ATM +-2 CE/PE strikes via the official instrument master JSON
   (never guesses the symbol format)
3. Fetches 5-min candles, computes EMA5/EMA25/VWAP
4. Checks entry signal; if found, computes entry/SL/target
5. POSTs the signal to the webhook (which holds the Shoonya session and
   either simulates a paper fill or places a real order, depending on its
   own LIVE_MODE flag -- that decision lives in the webhook, not here)
6. Persists minimal state (current open position, for managing SL/target)
   to state.json, committed back to the repo by the GitHub Action.

This script must NEVER place a real order itself.
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
LOCK_FILE = "run.lock"
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

PREV_DAY_CACHE_FILE = "prev_day_candles.json"


def acquire_run_lock():
    """
    Returns True if the lock was acquired (safe to proceed). Returns False
    if a fresh lock already exists, meaning another run is still active.
    A stale lock (older than LOCK_STALE_SECONDS -- i.e. from a run that
    crashed without cleaning up) is cleared and re-acquired.
    """
    if os.path.exists(LOCK_FILE):
        age = time.time() - os.path.getmtime(LOCK_FILE)
        if age < LOCK_STALE_SECONDS:
            return False
        print(
            f"Stale lock file found (age={age:.0f}s) -- previous run likely "
            "crashed without cleaning up. Clearing it and proceeding.",
            file=sys.stderr,
        )

    with open(LOCK_FILE, "w") as f:
        f.write(f"{os.getpid()} {now_ist().isoformat()}")
    return True


def release_run_lock():
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass


def load_prev_day_cache(expected_date):
    if not os.path.exists(PREV_DAY_CACHE_FILE):
        return {}
    try:
        with open(PREV_DAY_CACHE_FILE, "r") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if cache.get("date") != expected_date.isoformat():
        return {}
    return cache.get("candles_by_token", {})


def save_prev_day_cache(date, candles_by_token):
    with open(PREV_DAY_CACHE_FILE, "w") as f:
        json.dump({"date": date.isoformat(), "candles_by_token": candles_by_token}, f)


def get_candles_with_cache(smart_api, token, prev_day, prev_day_cache, today_start):
    token_str = str(token)

    if token_str in prev_day_cache:
        prev_candles = prev_day_cache[token_str]
    else:
        prev_day_start = dt.datetime.combine(prev_day, MARKET_OPEN)
        prev_candles = ac.fetch_5min_candles(smart_api, token, start_time=prev_day_start)
        # FIX: only cache a NON-EMPTY result. An empty list here almost
        # always means the fetch failed (e.g. exhausted all rate-limit
        # retries) -- a genuinely zero-candle full previous session for a
        # near-ATM weekly option is effectively impossible. Caching []
        # unconditionally "poisons" this token for the rest of the day:
        # every later run sees the key already present and skips
        # re-fetching entirely, even after the rate limit clears minutes
        # later -- degrading that leg's EMA warm-up for the whole session.
        if prev_candles:
            prev_day_cache[token_str] = prev_candles
        else:
            print(f"Previous-day candle fetch for token {token} returned empty -- "
                  "NOT caching, will retry on a later run.", file=sys.stderr)

    today_candles = ac.fetch_5min_candles(smart_api, token, start_time=today_start)

    seen_times = {c["time"] for c in prev_candles}
    merged = list(prev_candles) + [c for c in today_candles if c["time"] not in seen_times]
    return merged


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


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "open_position": None,
        "last_run": None,
        "heartbeat_date": None,
        "daily_strikes_date": None,
        "daily_atm": None,
        "daily_strikes": None,
        "pending_signal": None,
    }


def save_state(state):
    state["last_run"] = now_ist().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def build_leg_pairs(atm):
    """
    Credit-spread structure (NOT a naked strangle): each side sells one
    strike and buys a further-OTM hedge strike in the SAME option type.
      PE side: SELL ATM-100 PE  /  hedge BUY ATM-400 PE
      CE side: SELL ATM+100 CE  /  hedge BUY ATM+400 CE

    FIX (leg-type bug): the previous version resolved BOTH CE and PE for
    all 4 strikes (8 lookups/run), which is why "No match found for NIFTY
    23850 CE" etc. was spamming logs -- 23850 (ATM-100) is a PE-only leg
    and was never supposed to be looked up as a CE. Each strike now only
    resolves the option type it actually is.
    """
    return [
        {"sell_strike": atm - 100, "hedge_strike": atm - 400, "option_type": "PE"},
        {"sell_strike": atm + 100, "hedge_strike": atm + 400, "option_type": "CE"},
    ]


def get_or_set_daily_strikes(state, smart_api):
    """
    Locks today's leg pairs (sell + hedge strike per side) at/after 9:30 AM,
    same reasoning as before -- just restructured to return leg-pair dicts
    instead of a flat 4-strike list, since strikes now carry sell/hedge/
    option_type together rather than being interpreted generically.
    """
    today_str = now_ist().date().isoformat()

    if state.get("daily_strikes_date") == today_str and state.get("daily_atm") is not None:
        # FIX: rebuild AND rewrite state["daily_strikes"] on every run, not
        # just the first lock of the day. Previously this branch computed
        # leg_pairs in memory for the return value but never touched
        # state["daily_strikes"] on disk -- so a state.json already locked
        # under the old flat 4-strike format (e.g. mid-rollout) would keep
        # showing that stale format all day, only self-correcting at the
        # next day's fresh 9:30 lock. Rewriting it here is free (pure
        # computation from daily_atm, no API call) and makes state.json
        # self-consistent with the current code from the very next run.
        atm = state["daily_atm"]
        leg_pairs = build_leg_pairs(atm)
        state["daily_strikes"] = leg_pairs
        return atm, leg_pairs, None

    if now_ist().time() < STRIKE_LOCK_TIME:
        return None, None, None

    spot_price = ac.fetch_spot_ltp(smart_api)
    if spot_price is None:
        return None, None, None

    atm = get_atm_strike(spot_price)
    leg_pairs = build_leg_pairs(atm)

    state["daily_strikes_date"] = today_str
    state["daily_atm"] = atm
    state["daily_strikes"] = leg_pairs

    return atm, leg_pairs, spot_price


def send_heartbeat_if_needed(state):
    today_str = now_ist().date().isoformat()
    if state.get("heartbeat_date") == today_str:
        return

    send_to_webhook({
        "action": "HEARTBEAT",
        "message": f"Signal generator alive -- first successful run today at {now_ist().strftime('%H:%M:%S')} IST.",
        "time": now_ist().isoformat(),
    })
    state["heartbeat_date"] = today_str


def get_current_weekly_expiry():
    today = now_ist().date()
    days_ahead = (1 - today.weekday()) % 7
    expiry = today + dt.timedelta(days=days_ahead)

    while not is_trading_day(expiry):
        expiry -= dt.timedelta(days=1)

    return expiry


def get_atm_strike(spot_price, step=50):
    return round(spot_price / step) * step


def compute_indicators(candles):
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
        print(
            f"[DEBUG] compute_indicators: only {len(df)} today candle(s) available "
            f"(need >=2) -- returning None. now_ist={now_ist().isoformat()}",
            file=sys.stderr,
        )
        return None

    # DEBUG (temporary): confirm which candle is actually being treated as
    # "latest completed" vs. the current wall-clock time this run executed.
    print(
        f"[DEBUG] compute_indicators: latest completed candle time="
        f"{df.iloc[-1]['time']} | run time now_ist={now_ist().isoformat()}",
        file=sys.stderr,
    )

    return df


def check_entry_signal(df):
    if df is None or len(df) < 2:
        return False
    last = df.iloc[-1]
    return (
        last["ema5"] < last["vwap"]
        and last["ema25"] > last["ema5"]
        and last["ema25"] > last["vwap"]
    )


def log_signal_debug(symbol, df):
    """
    TEMPORARY debug helper (point 3 of the investigation). For each
    scanned leg, prints a human-readable block:

        Scanning <symbol>
        Last candle: <HH:MM>
        EMA5 = <value>
        EMA25 = <value>
        VWAP = <value>
        EMA5 < VWAP : <bool>
        EMA25 > EMA5 : <bool>      (only reached if the above was True)
        EMA25 > VWAP : <bool>      (only reached if the above was True)
        Signal = <bool>

    Conditions are printed in the SAME order and with the SAME
    short-circuiting as check_entry_signal()'s "and" chain -- once one
    condition is False, the remaining ones aren't evaluated/printed and
    we go straight to "Signal = False". This mirrors actual evaluation
    order so the log tells you exactly which check blocked a signal.

    Does not change any decision logic -- read-only observability,
    called from scan_for_new_signal() for every leg scanned, not just
    ones that fire.

    Wrapped defensively: vwap can be pandas.NA (not NaN) on a candle with
    zero cumulative volume -- e.g. a thinly-traded hedge leg's first
    candle of the day -- because compute_indicators() does
    `cum_vol.replace(0, pd.NA)` before dividing. VWAP is printed without
    a ':.2f' spec for this reason (pd.NA doesn't support it). This is
    pure logging; it must never be able to crash a real run that's
    managing live positions, so any failure here is caught and reported
    instead of propagated.
    """
    try:
        last = df.iloc[-1]
        candle_time = last["time"]
        try:
            candle_time_str = candle_time.strftime("%H:%M")
        except AttributeError:
            candle_time_str = str(candle_time)

        ema5 = last["ema5"]
        ema25 = last["ema25"]
        vwap = last["vwap"]

        print(f"Scanning {symbol}", file=sys.stderr)
        print(f"Last candle: {candle_time_str}", file=sys.stderr)
        print(f"EMA5 = {ema5:.2f}", file=sys.stderr)
        print(f"EMA25 = {ema25:.2f}", file=sys.stderr)
        print(f"VWAP = {vwap}", file=sys.stderr)

        cond_ema5_below_vwap = bool(ema5 < vwap)
        print(f"EMA5 < VWAP : {cond_ema5_below_vwap}", file=sys.stderr)
        if not cond_ema5_below_vwap:
            print("Signal = False", file=sys.stderr)
            return

        cond_ema25_above_ema5 = bool(ema25 > ema5)
        print(f"EMA25 > EMA5 : {cond_ema25_above_ema5}", file=sys.stderr)
        if not cond_ema25_above_ema5:
            print("Signal = False", file=sys.stderr)
            return

        cond_ema25_above_vwap = bool(ema25 > vwap)
        print(f"EMA25 > VWAP : {cond_ema25_above_vwap}", file=sys.stderr)

        print(f"Signal = {cond_ema25_above_vwap}", file=sys.stderr)

        # Extra context for investigation point 1 (state vs. crossover) --
        # doesn't disturb the block above, just appends one more line when
        # a signal actually fires, so we can tell a fresh cross apart from
        # an already-established state.
        if len(df) >= 2:
            prev = df.iloc[-2]
            prev_below = bool(prev["ema5"] < prev["vwap"])
            print(
                f"(prev candle EMA5 < VWAP : {prev_below} -> "
                f"{'fresh crossover' if not prev_below else 'state already held'})",
                file=sys.stderr,
            )
    except Exception as e:
        # Never let a logging/formatting problem take down a run that's
        # managing real positions. Report it and move on.
        print(f"Scanning {symbol}: debug logging failed ({e!r}) -- "
              "continuing without it.", file=sys.stderr)


def compute_entry_price(trigger_candle):
    """
    LEGACY -- no longer called by main(). Retained only because the
    currently-open single-leg position in state.json (opened before the
    pending-signal rework) was priced with this formula; keeping it here
    for reference/audit, not for reuse. New entries go through
    compute_pending_signal() instead.
    """
    low = trigger_candle["low"]
    high = trigger_candle["high"]
    return low + 0.40 * (high - low)


PENDING_SIGNAL_MAX_CANDLES = 5   # resting window: candles N+1 .. N+5
ENTRY_LIMIT_DISCOUNT = 0.95      # limit = EMA5[N] * this
LOW_PREMIUM_SL_THRESHOLD = 99    # Rs. -- below this, SL floor kicks in
LOW_PREMIUM_SL_MIN_PCT = 0.10    # SL floor = entry_limit * (1 + this)
TARGET_RISK_REWARD = 2           # target = entry - RR * (SL - entry)


def compute_pending_signal(trigger_candle, sell_leg_info, hedge_leg_info, qty):
    """
    Locks a resting SELL limit order off the just-closed trigger candle N.
    EMA5[N] stands in as the proxy for candle N+1's open (the real open
    isn't known yet at candle-N-close time -- only 5-min OHLC is available,
    not tick data). The limit is set at a discount to that proxy, so in
    practice it's a floor that the very next candle's high almost always
    clears -- the 5-candle window exists as a safety margin for a single
    illiquid/gappy print, not because a deep pullback is expected.

    entry_limit is FIXED for the whole resting window: computed once here
    from candle N, never recomputed on candles N+1..N+5.

    SL = max(trigger candle's high, trigger candle's VWAP). If entry_limit
    is under Rs.99, SL additionally floors at entry_limit * 1.10 -- this
    WIDENS the stop when it applies, it never tightens or replaces the
    high/VWAP comparison (cheap premiums are more prone to a tight
    absolute SL getting whipsawed).

    Target is fixed 1:2 risk:reward off entry_limit, using the resulting
    SL distance as risk.
    """
    trigger_high = trigger_candle["high"]
    trigger_vwap = trigger_candle["vwap"]
    entry_limit = trigger_candle["ema5"] * ENTRY_LIMIT_DISCOUNT

    sl_price = max(trigger_high, trigger_vwap)
    if entry_limit < LOW_PREMIUM_SL_THRESHOLD:
        sl_price = max(sl_price, entry_limit * (1 + LOW_PREMIUM_SL_MIN_PCT))

    risk = sl_price - entry_limit
    target_price = entry_limit - TARGET_RISK_REWARD * risk

    return {
        "sell_symbol": sell_leg_info["symbol"],
        "sell_token": sell_leg_info["token"],
        "sell_strike": sell_leg_info["strike"],
        "hedge_symbol": hedge_leg_info["symbol"],
        "hedge_token": hedge_leg_info["token"],
        "hedge_strike": hedge_leg_info["strike"],
        "option_type": sell_leg_info["option_type"],
        "qty": qty,
        "entry_limit": entry_limit,
        "sl_price": sl_price,
        "target_price": target_price,
        "trigger_time": now_ist().isoformat(),
        "candles_waited": 0,
    }


def is_eod_squareoff_time():
    return now_ist().time() >= EOD_SQUAREOFF


def force_eod_exit(smart_api, state, pos):
    """
    EOD SAFETY NET (fallback only): used when the normal candle-based exit
    check couldn't even reach a decision this run -- either
    resolve_option_token() failed, or the candle fetch failed -- AND it's
    already past EOD_SQUAREOFF. Without this, an Angel One outage or
    rate-limit storm landing right at 15:20-15:30 IST means EOD square-off
    silently never fires, since both manage_spread_exit() and
    manage_legacy_single_leg_exit() previously just `return`ed on missing
    data with no fallback.

    Deliberately bypasses BOTH token resolution and candle data -- the
    only things this needs (symbol(s), qty) are already stored on the
    position itself in state.json, since they were captured at entry
    time. This is intentionally a fallback path only, not a replacement:
    the normal candle-based exit (with real SL/target checks and a
    proper closing price) is always tried first; this only engages when
    that path couldn't run at all.

    Price is best-effort ONLY, for the Telegram/Sheets record -- it is
    NOT required for the order to execute correctly, since
    webhook.py's place_leg_order() always uses price_type="MKT" and
    ignores whatever price value is sent. A single non-retrying spot LTP
    attempt is made (ac.fetch_spot_ltp_once) specifically so this path
    doesn't itself risk getting stuck in the same rate-limit retries that
    likely caused the normal path to fail in the first place; on any
    failure this proceeds with price=None rather than delaying further.
    """
    price = None
    if smart_api is not None:
        try:
            price = ac.fetch_spot_ltp_once(smart_api)
        except Exception as e:
            print(f"force_eod_exit: spot LTP attempt raised, proceeding "
                  f"without a price: {e}", file=sys.stderr)
            price = None

    if pos.get("spread"):
        sell_leg = pos["sell_leg"]
        hedge_leg = pos["hedge_leg"]
        payload = {
            "action": "EXIT_SPREAD",
            "reason": "EOD_FORCE",
            "sell_symbol": sell_leg["symbol"],
            "hedge_symbol": hedge_leg["symbol"],
            "qty": pos["qty"],
            "price": price,
            "time": now_ist().isoformat(),
        }
    else:
        payload = {
            "action": "EXIT",
            "reason": "EOD_FORCE",
            "symbol": pos["symbol"],
            "qty": pos["qty"],
            "price": price,
            "time": now_ist().isoformat(),
        }

    resp = send_to_webhook(payload)

    if webhook_confirmed_ok(resp):
        state["open_position"] = None
        print(f"EOD FORCE-EXIT confirmed (normal exit path had no data this run) "
              f"-- position closed. price={price}")
    else:
        print(
            "EOD FORCE-EXIT webhook not confirmed -- leaving open_position in "
            "state.json so the next run retries. If runs keep failing past "
            "market close, THIS NEEDS MANUAL ATTENTION -- the position may be "
            "sitting open overnight with no further automated retry once "
            "is_market_open_now() goes False.",
            file=sys.stderr,
        )


def manage_legacy_single_leg_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    UNCHANGED dynamic-SL exit logic for single-leg positions opened before
    the pending-signal/spread rework landed (identified by the absence of
    a "spread" key). Do not change this function's core behaviour -- it
    exists only so a position opened before this rework gets managed
    through to its own exit correctly. New positions never take this
    path; see manage_spread_exit() instead.

    FIX (EOD safety net): previously, if token resolution or candle fetch
    failed, this function just returned with no fallback -- meaning EOD
    square-off could silently never fire during an outage. Now falls back
    to force_eod_exit() specifically when that happens AND it's already
    past EOD_SQUAREOFF -- see force_eod_exit() docstring. This fallback
    does not change the SL/target logic itself, only what happens when
    that logic couldn't run at all this run.
    """
    token_info = ac.resolve_option_token(instruments, expiry, pos["strike"], pos["option_type"])

    df = None
    if token_info:
        candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
        df = compute_indicators(candles)

    if df is None:
        if is_eod_squareoff_time():
            print("Candle data/token unavailable for legacy single-leg position "
                  "at/after EOD squareoff time -- falling back to force-exit.",
                  file=sys.stderr)
            force_eod_exit(smart_api, state, pos)
        return

    last = df.iloc[-1]
    sl_hit = last["close"] > last["vwap"]
    target_hit = last["close"] <= pos["target_price"]

    if sl_hit or target_hit or is_eod_squareoff_time():
        reason = "SL" if sl_hit else ("TARGET" if target_hit else "EOD")
        resp = send_to_webhook({
            "action": "EXIT",
            "reason": reason,
            "symbol": pos["symbol"],
            "qty": pos["qty"],
            "price": float(last["close"]),
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


def manage_spread_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    Exit logic for two-leg spread positions opened under the new
    pending-signal flow. SL/target are the FIXED levels locked at signal
    time (pos["sl_price"] / pos["target_price"]) -- not re-evaluated every
    candle the way the legacy dynamic close>VWAP check is. Widening the SL
    rule itself (VWAP+buffer, VIX filter) is a separate, deliberately
    deferred change -- this only fixes what price triggers the exit.

    Both legs are always closed together in a single EXIT_SPREAD webhook
    call.

    FIX (EOD safety net): previously, if token resolution or candle fetch
    failed, this function just returned with no fallback -- meaning EOD
    square-off could silently never fire during an outage. Now falls back
    to force_eod_exit() specifically when that happens AND it's already
    past EOD_SQUAREOFF -- see force_eod_exit() docstring. This fallback
    does not change the SL/target logic itself, only what happens when
    that logic couldn't run at all this run.
    """
    sell_leg = pos["sell_leg"]
    token_info = ac.resolve_option_token(instruments, expiry, sell_leg["strike"], pos["option_type"])

    df = None
    if token_info:
        candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
        df = compute_indicators(candles)

    if df is None:
        if is_eod_squareoff_time():
            print("Candle data/token unavailable for spread position "
                  "at/after EOD squareoff time -- falling back to force-exit.",
                  file=sys.stderr)
            force_eod_exit(smart_api, state, pos)
        return

    last = df.iloc[-1]
    close = float(last["close"])
    sl_hit = close > pos["sl_price"]
    target_hit = close <= pos["target_price"]

    if sl_hit or target_hit or is_eod_squareoff_time():
        reason = "SL" if sl_hit else ("TARGET" if target_hit else "EOD")
        resp = send_to_webhook({
            "action": "EXIT_SPREAD",
            "reason": reason,
            "sell_symbol": sell_leg["symbol"],
            "hedge_symbol": pos["hedge_leg"]["symbol"],
            "qty": pos["qty"],
            "price": close,
            "time": now_ist().isoformat(),
        })

        if webhook_confirmed_ok(resp):
            state["open_position"] = None
        else:
            print(
                "EXIT_SPREAD webhook not confirmed -- leaving open_position "
                "in state.json so the next run retries.",
                file=sys.stderr,
            )


def manage_pending_signal(state, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    Checks whether the resting SELL limit order (pending_signal) should
    fill, get cancelled (setup invalidated), or expire (5-candle window
    exhausted) on this run's latest closed candle.

    Order matters: fill is checked FIRST, before cancellation/expiry -- a
    real resting limit order doesn't care whether check_entry_signal() is
    still true. If price already reached the limit this candle, it fills
    regardless of whether the EMA/VWAP setup broke down by this candle's
    close.
    """
    pending = state["pending_signal"]

    token_info = ac.resolve_option_token(instruments, expiry, pending["sell_strike"], pending["option_type"])
    if not token_info:
        print("Could not resolve sell-leg token while pending_signal is resting -- skipping this run.",
              file=sys.stderr)
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)
    if df is None:
        return

    last = df.iloc[-1]

    # ---- 1. Fill check first ----
    if last["high"] >= pending["entry_limit"]:
        hedge_token_info = ac.resolve_option_token(instruments, expiry, pending["hedge_strike"], pending["option_type"])
        if not hedge_token_info:
            print("Sell leg limit reached but hedge leg token could not be resolved -- "
                  "NOT filling, leaving pending_signal in place to retry next run.",
                  file=sys.stderr)
            return

        resp = send_to_webhook({
            "action": "ENTRY_SPREAD",
            "sell_side": "SELL",
            "sell_symbol": pending["sell_symbol"],
            "hedge_side": "BUY",
            "hedge_symbol": hedge_token_info["symbol"],
            "qty": pending["qty"],
            "entry_price": pending["entry_limit"],
            "sl_price": pending["sl_price"],
            "target_price": pending["target_price"],
            "time": now_ist().isoformat(),
        })

        if webhook_confirmed_ok(resp):
            state["open_position"] = {
                "spread": True,
                "option_type": pending["option_type"],
                "sell_leg": {
                    "symbol": pending["sell_symbol"],
                    "token": pending["sell_token"],
                    "strike": pending["sell_strike"],
                },
                "hedge_leg": {
                    "symbol": hedge_token_info["symbol"],
                    "token": hedge_token_info["token"],
                    "strike": pending["hedge_strike"],
                },
                "qty": pending["qty"],
                "entry_price": pending["entry_limit"],
                "sl_price": pending["sl_price"],
                "target_price": pending["target_price"],
                "entry_time": now_ist().isoformat(),
            }
            state["pending_signal"] = None
            print(f"FILLED: SELL {pending['sell_symbol']} @ {pending['entry_limit']} "
                  f"+ hedge BUY {hedge_token_info['symbol']}, qty={pending['qty']}")
        else:
            print(
                "ENTRY_SPREAD webhook not confirmed -- leaving pending_signal in "
                "state.json so the next run retries the fill.",
                file=sys.stderr,
            )
        return

    # ---- 2. Cancellation check: has the original setup invalidated? ----
    if not check_entry_signal(df):
        print(f"pending_signal for {pending['sell_symbol']} cancelled -- "
              "EMA5/VWAP crossover condition no longer holds.")
        state["pending_signal"] = None
        return

    # ---- 3. Expiry check: 5-candle resting window exhausted? ----
    pending["candles_waited"] += 1
    if pending["candles_waited"] >= PENDING_SIGNAL_MAX_CANDLES:
        print(f"pending_signal for {pending['sell_symbol']} expired -- "
              f"unfilled after {PENDING_SIGNAL_MAX_CANDLES} candles.")
        state["pending_signal"] = None
        return

    print(f"pending_signal for {pending['sell_symbol']} still resting "
          f"({pending['candles_waited']}/{PENDING_SIGNAL_MAX_CANDLES} candles, "
          f"limit={pending['entry_limit']:.2f}, latest high={last['high']:.2f}).")


def scan_for_new_signal(state, leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    Scans the two sell legs (PE side, CE side) for a fresh EMA5<VWAP
    crossover. On a match, locks a resting pending_signal (see
    compute_pending_signal()) instead of entering immediately -- the fill
    itself happens later in manage_pending_signal() on a subsequent run.
    """
    for leg in leg_pairs:
        sell_token_info = ac.resolve_option_token(instruments, expiry, leg["sell_strike"], leg["option_type"])
        if not sell_token_info:
            continue

        candles = get_candles_with_cache(smart_api, sell_token_info["token"], prev_day, prev_day_cache, today_start)
        df = compute_indicators(candles)
        if df is None:
            continue

        # DEBUG (temporary, point 3): log full condition breakdown for
        # EVERY leg scanned this run, whether or not it fires, so it's
        # clear which specific condition is blocking a signal on each side.
        log_signal_debug(sell_token_info["symbol"], df)

        if check_entry_signal(df):
            hedge_token_info = ac.resolve_option_token(instruments, expiry, leg["hedge_strike"], leg["option_type"])
            if not hedge_token_info:
                print(f"Signal fired on {sell_token_info['symbol']} but hedge strike "
                      f"{leg['hedge_strike']}{leg['option_type']} could not be resolved -- skipping this signal.",
                      file=sys.stderr)
                continue

            trigger_candle = df.iloc[-1].to_dict()
            qty = sell_token_info["lot_size"]

            sell_leg_info = {**sell_token_info, "strike": leg["sell_strike"], "option_type": leg["option_type"]}
            hedge_leg_info = {**hedge_token_info, "strike": leg["hedge_strike"], "option_type": leg["option_type"]}

            state["pending_signal"] = compute_pending_signal(trigger_candle, sell_leg_info, hedge_leg_info, qty)
            p = state["pending_signal"]
            print(f"PENDING SIGNAL: SELL {p['sell_symbol']} resting limit @ {p['entry_limit']:.2f} "
                  f"(SL={p['sl_price']:.2f}, target={p['target_price']:.2f})")
            # DEBUG (temporary, point 4): confirm this early return is what
            # stops the other leg (CE/PE) from being scanned in the same
            # run once one side has fired. state["pending_signal"] /
            # state["open_position"] are single dicts, not lists, so the
            # other leg is intentionally deferred to a later run rather
            # than evaluated now -- see investigation notes.
            print(
                f"[DEBUG] scan_for_new_signal: stopping after {p['sell_symbol']} -- "
                "remaining leg(s) in this run's leg_pairs were not scanned "
                "(single pending_signal slot in state.json).",
                file=sys.stderr,
            )
            return

    print("No entry signal this run.")


def send_to_webhook(payload):
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
                time.sleep(WEBHOOK_RETRY_DELAY_SECONDS)

    print(f"Webhook call failed after {WEBHOOK_RETRY_ATTEMPTS} attempts: {last_err}",
          file=sys.stderr)
    return None


def webhook_confirmed_ok(resp):
    if resp is None:
        return False
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code == 200 and body.get("status") == "ok"


def main():
    if not is_market_open_now():
        print("Market closed (outside hours or holiday) -- skipping run.")
        return

    # FIX (overlapping-run guard): acquire the lock before doing any API
    # work. If a previous run is still active (fresh lock present), skip
    # this run entirely rather than firing a second concurrent session at
    # Angel One under the same API key -- see LOCK_FILE comment above for
    # why this was the likely root cause of the rate-limit errors.
    if not acquire_run_lock():
        print(
            "Another run appears to still be in progress (lock file is fresh) -- "
            "skipping this run to avoid overlapping Angel One sessions / rate limiting.",
            file=sys.stderr,
        )
        return

    try:
        state = load_state()
        send_heartbeat_if_needed(state)
        save_state(state)

        smart_api = ac.login()
        instruments = ac.download_instrument_master()

        expiry = get_current_weekly_expiry()

        atm, leg_pairs, spot_price_at_lock = get_or_set_daily_strikes(state, smart_api)
        if atm is None:
            if now_ist().time() < STRIKE_LOCK_TIME:
                print(f"Before {STRIKE_LOCK_TIME.strftime('%H:%M')} strike-lock time -- skipping signal check this run.")
            else:
                print("Could not fetch spot price to lock today's strikes, aborting this run.", file=sys.stderr)
            return
        if spot_price_at_lock is not None:
            print(f"Locked today's strikes: ATM={atm} (spot={spot_price_at_lock}) -> {leg_pairs}")

        prev_day = previous_trading_day(now_ist().date())
        prev_day_cache = load_prev_day_cache(prev_day)
        today_start = dt.datetime.combine(now_ist().date(), MARKET_OPEN)

        save_state(state)

        # ---- Manage existing open position first ----
        if state["open_position"] is not None:
            pos = state["open_position"]
            if pos.get("spread"):
                manage_spread_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)
            else:
                # FIX (backward compatibility): a position opened before this
                # rework has no "spread" key -- it must keep being managed by
                # the OLD dynamic-SL, single-leg exit logic untouched, not the
                # new fixed-SL spread logic. See manage_legacy_single_leg_exit().
                manage_legacy_single_leg_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)

            save_state(state)
            save_prev_day_cache(prev_day, prev_day_cache)
            return

        # ---- No open position: manage a resting pending_signal, if any ----
        if state.get("pending_signal") is not None:
            manage_pending_signal(state, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)
            save_state(state)
            save_prev_day_cache(prev_day, prev_day_cache)
            return

        # ---- No position, no pending signal: scan for a fresh entry trigger ----
        scan_for_new_signal(state, leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)
        save_state(state)
        save_prev_day_cache(prev_day, prev_day_cache)
    finally:
        release_run_lock()


if __name__ == "__main__":
    main()
