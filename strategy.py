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
        prev_day_cache[token_str] = prev_candles

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


def manage_legacy_single_leg_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    UNCHANGED dynamic-SL exit logic for single-leg positions opened before
    the pending-signal/spread rework landed (identified by the absence of
    a "spread" key). Do not change this function's behaviour -- it exists
    only so the position already open in state.json as of 2026-07-01
    (NIFTY07JUL2624050PE) gets managed through to its own exit correctly.
    New positions never take this path; see manage_spread_exit() instead.
    """
    token_info = ac.resolve_option_token(instruments, expiry, pos["strike"], pos["option_type"])
    if not token_info:
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)
    if df is None:
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
    """
    sell_leg = pos["sell_leg"]
    token_info = ac.resolve_option_token(instruments, expiry, sell_leg["strike"], pos["option_type"])
    if not token_info:
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)
    if df is None:
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


if __name__ == "__main__":
    main()
