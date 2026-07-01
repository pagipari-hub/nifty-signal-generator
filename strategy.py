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

---
FIX LOG (production hardening pass):
  - All datetime.now() calls now use IST explicitly (the GH Actions job sets
    TZ=Asia/Kolkata, but we don't rely on that alone -- this script is
    correct even if run somewhere that doesn't set TZ).
  - Added an internal market-hours + holiday guard, run before any API calls.
  - get_current_weekly_expiry() now adjusts backward off NSE holidays.
  - send_to_webhook() now retries on failure (webhook is known to be choppy).
  - main() now gates state mutation on a confirmed-OK webhook response,
    instead of assuming the signal was acted on.
  - Strike-locking now gated on STRIKE_LOCK_TIME (9:30 AM) instead of
    "whichever run happens to be first" -- previously a 9:15-9:29 run
    could lock strikes off an unsettled opening-range price. See
    get_or_set_daily_strikes() for full reasoning.
  - LOT_SIZE hardcoded constant removed. Lot size is now read dynamically
    from Angel One's instrument master JSON (the 'lotsize' field), so the
    script automatically adapts to any future NSE revision without a code
    change. Fallback is 65 per NSE circular NSE/FAOP/70616 (Jan 2026).

NOT YET DONE (deliberately deferred -- depends on the Shoonya/webhook side):
  - A webhook HTTP 200 here still only means "the webhook accepted the
    request," NOT "Shoonya filled the order." Real fill confirmation has to
    happen inside the webhook (it holds the Shoonya session), and until
    that's built, the gating below is necessary-but-not-sufficient. Treat
    state.json as "probably right" rather than "guaranteed right" until
    that follow-up lands.
  - Startup-time reconciliation against actual broker positions.
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

# LOT_SIZE is no longer a hardcoded constant here. It is read dynamically
# from Angel One's instrument master JSON via resolve_option_token(), which
# returns a 'lot_size' field. This means any future NSE revision is picked
# up automatically the day new contracts appear in the master, with no code
# change needed. See angelone_client.py -> resolve_option_token() for details.

WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_RETRY_DELAY_SECONDS = 2

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
    """
    FIX (1.2): nothing previously stopped this script from running outside
    market hours or on a holiday -- the cron schedule restricts *automatic*
    triggers, but workflow_dispatch (manual runs) and holiday non-awareness
    were both unguarded. This is checked first in main(), before any
    API/network calls are made.
    """
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
    }


def save_state(state):
    state["last_run"] = now_ist().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_or_set_daily_strikes(state, smart_api):
    """
    The strategy's design intent is that today's strikes are anchored to
    the SPOT PRICE AT 9:30 AM SPECIFICALLY -- not just "whenever the first
    run of the day happens to land." The strategy sells premium on
    whichever side benefits from the day's move relative to that 9:30
    reference point -- if spot rises from there, sell puts; if it falls,
    sell calls -- so the strikes must stay fixed all day for that logic to
    mean what it's supposed to mean. Recalculating ATM every run would
    chase a moving target and break this; locking off a pre-9:30 run
    (market opens 9:15) would anchor to a less-settled opening-range price
    instead of the intended reference point.

    FIX (strike-lock timing): previously this locked on the first run of
    the day regardless of clock time, which meant a 9:15-9:29 run would
    lock strikes ~15 min too early. Now gated on STRIKE_LOCK_TIME (9:30):
    runs before 9:30 return (None, None, None) and do no locking at all;
    the first run AT OR AFTER 9:30 does the lock.

    Snapshots ATM (and the 4 strikes derived from it) once, on the first
    run of each trading day at/after 9:30, and persists it in state.json
    so every later run that day reuses the SAME strikes regardless of how
    much spot drifts afterward. Returns (atm, strikes_to_check, spot_price_at_lock).

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


def send_heartbeat_if_needed(state):
    """
    Sends a one-line 'I'm alive' heartbeat through the webhook (which relays
    it to Telegram) once per trading day, on whichever run is the first one
    to successfully reach this point that day.

    Why this matters: GitHub Actions scheduled workflows can silently fail
    to fire at all (known platform-side flakiness, not specific to this
    repo -- see e.g. github.com/orgs/community/discussions/185024). When
    that happens, there's no error and no log -- the job just never starts.
    A missing heartbeat by ~9:20 AM is the signal to check the Actions tab
    and, if needed, push a trivial commit to .github/workflows/ to force
    GitHub to resync the schedule.

    This does NOT detect every failure mode (e.g. it can't warn you if the
    very first run of the day is the one that fails to fire), but it does
    catch the much more common case of "the schedule silently stopped
    firing entirely partway through the morning."
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
    September 1, 2025 (SEBI directive to spread weekly expiry volume across
    the week). If NSE changes it again, update the weekday number below
    (1 = Tuesday).

    FIX (1.3): if the computed Tuesday is a market holiday, NSE shifts the
    expiry to the previous trading day. This now walks backward from the
    computed date until it lands on an actual trading day.
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
    FIX (2.3, extended): VWAP must reset at today's market open -- that part
    was already fixed and is unchanged below. But EMA5/EMA25 should NOT
    reset daily; restarting them cold every morning means the first ~2+
    hours of every session have an EMA25 that's still mostly seed-value,
    not a real 25-period average, so the strategy was effectively blind to
    entry signals until ~11:30 AM most days.

    Fix: EMA5/EMA25 are computed across the FULL fetched range (today +
    previous trading day, see previous_trading_day() and how main() calls
    fetch_5min_candles), so they're already warmed up by today's market
    open. VWAP is still computed only on today's rows. The returned
    DataFrame contains only today's rows, with EMA values carried forward
    correctly and VWAP correctly session-anchored.
    """
    df = pd.DataFrame(candles)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    today = now_ist().date()

    # EMA needs the full range (includes previous trading day) to be
    # properly warmed up by the time today's session starts.
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema25"] = df["close"].ewm(span=25, adjust=False).mean()

    # VWAP must only accumulate from today's market open -- zero out
    # contribution from any prior-day rows before the cumulative sum.
    today_mask = df["time"].dt.date == today
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].where(today_mask, 0).cumsum()
    cum_tp_vol = (typical_price * df["volume"]).where(today_mask, 0).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, pd.NA)

    # Only today's rows are relevant for signal-checking from here on --
    # previous-day rows were only needed to warm up EMA.
    df = df[today_mask].reset_index(drop=True)

    if len(df) < 2:
        # Not enough of today's session yet to check a signal meaningfully,
        # even though EMA itself is already warmed up from yesterday.
        return None

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


def compute_entry_price(trigger_candle):
    low = trigger_candle["low"]
    high = trigger_candle["high"]
    return low + 0.40 * (high - low)


def is_eod_squareoff_time():
    return now_ist().time() >= EOD_SQUAREOFF


def send_to_webhook(payload):
    """
    FIX (1.5): the webhook is known to be choppy (Shoonya-side flakiness
    surfaces here). Previously this was a single attempt with no retry --
    now retries with a short delay before giving up. Still returns None on
    total failure; callers MUST check for that (see FIX 1.4 in main()).
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
                time.sleep(WEBHOOK_RETRY_DELAY_SECONDS)

    print(f"Webhook call failed after {WEBHOOK_RETRY_ATTEMPTS} attempts: {last_err}",
          file=sys.stderr)
    return None


def webhook_confirmed_ok(resp):
    """
    FIX (1.4): previously the return value of send_to_webhook() was never
    checked at all -- state was mutated unconditionally after firing the
    request. This at least confirms the webhook accepted the request.

    CAVEAT (documented, not yet resolved): a 200 here means "the webhook
    received and processed the request," not "Shoonya filled the order."
    The webhook currently returns 200 as soon as place_order() returns an
    order ID, before any fill confirmation. True fill-confirmation has to
    be built on the webhook side (it holds the Shoonya session) and this
    function's meaning gets stronger once that lands -- it isn't weakened
    by adding this check now, but don't treat this as the final word yet.
    """
    if resp is None:
        return False
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code == 200 and body.get("status") == "ok"


def main():
    # FIX (1.2): guard first, before any login/network calls.
    if not is_market_open_now():
        print("Market closed (outside hours or holiday) -- skipping run.")
        return

    state = load_state()
    send_heartbeat_if_needed(state)
    save_state(state)  # persist heartbeat_date immediately, don't wait for end of run

    smart_api = ac.login()
    instruments = ac.download_instrument_master()

    expiry = get_current_weekly_expiry()

    # FIX (strike-locking, per strategy design intent): strikes are
    # anchored to spot price AT 9:30 AM SPECIFICALLY (see
    # STRIKE_LOCK_TIME), not recalculated every 5 minutes -- the
    # strategy's logic depends on the day's move being measured relative
    # to that fixed 9:30 reference point. See get_or_set_daily_strikes()
    # for the full reason.
    atm, strikes_to_check, spot_price_at_lock = get_or_set_daily_strikes(state, smart_api)
    if atm is None:
        if now_ist().time() < STRIKE_LOCK_TIME:
            # Not an error -- just too early to lock strikes yet. Heartbeat
            # already ran above; nothing else to do this run.
            print(f"Before {STRIKE_LOCK_TIME.strftime('%H:%M')} strike-lock time -- skipping signal check this run.")
        else:
            print("Could not fetch spot price to lock today's strikes, aborting this run.", file=sys.stderr)
        return
    if spot_price_at_lock is not None:
        print(f"Locked today's strikes: ATM={atm} (spot={spot_price_at_lock}) -> {strikes_to_check}")

    # FIX (rate-limit regression): previously we asked Angel One for the
    # full previous-day + today range on EVERY call (8 calls/run), which
    # made each call heavier and triggered repeated rate-limit failures
    # that exhausted all retries. Now: previous day's candles are fetched
    # ONCE per day (cached to disk, see load_prev_day_cache), and each run
    # only fetches today's (small, cheap) candles fresh, then merges with
    # the cached previous-day data in memory for EMA warm-up.
    prev_day = previous_trading_day(now_ist().date())
    prev_day_cache = load_prev_day_cache(prev_day)
    today_start = dt.datetime.combine(now_ist().date(), MARKET_OPEN)

    save_state(state)  # persist today's locked strikes immediately

    # ---- Manage existing open position first ----
    if state["open_position"] is not None:
        pos = state["open_position"]
        token_info = ac.resolve_option_token(instruments, expiry, pos["strike"], pos["option_type"])

        if token_info:
            candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
            df = compute_indicators(candles)

            if df is not None:
                last = df.iloc[-1]
                sl_hit = last["close"] > last["vwap"]
                target_hit = last["close"] <= pos["target_price"]

                if sl_hit or target_hit or is_eod_squareoff_time():
                    reason = "SL" if sl_hit else ("TARGET" if target_hit else "EOD")
                    resp = send_to_webhook({
                        "action": "EXIT",
                        "reason": reason,
                        "symbol": pos["symbol"],
                        "qty": pos["qty"],  # stored in position at entry time
                        "price": float(last["close"]),
                        "time": now_ist().isoformat(),
                    })

                    # FIX (1.4): only clear the position if the webhook
                    # confirmed it processed the exit. If it failed, leave
                    # state untouched -- next run will retry the exit check
                    # rather than silently believing we're flat.
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
        return

    # ---- No open position: look for a new entry signal ----
    for strike in strikes_to_check:
        for opt_type in ["CE", "PE"]:
            token_info = ac.resolve_option_token(instruments, expiry, strike, opt_type)
            if not token_info:
                continue

            candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
            df = compute_indicators(candles)
            if df is None:
                continue

            if check_entry_signal(df):
                trigger_candle = df.iloc[-1].to_dict()
                entry_price = compute_entry_price(trigger_candle)
                target_price = entry_price - 2 * (trigger_candle["high"] - entry_price)
                entry_time = now_ist().isoformat()
                lot_size = token_info["lot_size"]  # from instrument master, not a hardcoded constant

                position = {
                    "symbol": token_info["symbol"],
                    "token": token_info["token"],
                    "strike": strike,
                    "option_type": opt_type,
                    "entry_price": entry_price,
                    "qty": lot_size,
                    "target_price": target_price,
                    "entry_time": entry_time,
                }

                resp = send_to_webhook({
                    "action": "ENTRY",
                    "side": "SELL",
                    "symbol": position["symbol"],
                    "qty": lot_size,
                    "price": entry_price,
                    "target_price": target_price,
                    "time": entry_time,
                })

                # FIX (1.4): only record the position as open if the webhook
                # confirmed it. Otherwise the next run correctly sees
                # open_position as None and will re-scan for entries instead
                # of "managing" a position that was never actually placed.
                if webhook_confirmed_ok(resp):
                    state["open_position"] = position
                    save_state(state)
                    print(f"SIGNAL: SELL {position['symbol']} @ {entry_price} qty={lot_size}")
                else:
                    print(
                        f"ENTRY webhook not confirmed for {position['symbol']} -- "
                        "NOT recording as open position. Will re-evaluate next run.",
                        file=sys.stderr,
                    )
                    save_state(state)
                save_prev_day_cache(prev_day, prev_day_cache)
                return

    save_state(state)
    save_prev_day_cache(prev_day, prev_day_cache)
    print("No entry signal this run.")


if __name__ == "__main__":
    main()
