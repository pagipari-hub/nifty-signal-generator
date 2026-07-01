import json
import os
import sys
import time
import datetime as dt
import pandas as pd
import requests

import angelone_client as ac

# --- Configuration Constants ---
STATE_FILE = "state.json"
PREV_DAY_CACHE_FILE = "prev_day_candles.json"
HOLIDAY_FILE = "nas_holidays.json"
HISTORY_DIR = "candle_history"

WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SHARED_SECRET")

WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_RETRY_DELAY_SECONDS = 2

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
EOD_SQUAREOFF = dt.time(15, 20)
STRIKE_LOCK_TIME = dt.time(9, 30)

# Automatically guarantee output directory exists for workflow tracking
os.makedirs(HISTORY_DIR, exist_ok=True)


def get_now():
    """
    Returns system time. Because the GitHub Actions workflow defines 
    TZ: Asia/Kolkata, this natively returns correct Indian Standard Time (IST).
    """
    return dt.datetime.now()


def load_market_holidays():
    """Loads closed market dates from the localized repository JSON list."""
    if os.path.exists(HOLIDAY_FILE):
        try:
            with open(HOLIDAY_FILE, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, OSError):
            print(f"Warning: {HOLIDAY_FILE} is corrupt or unreadable. Falling back.")
    return set()


MARKET_HOLIDAYS = load_market_holidays()


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
    state["last_run"] = get_now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_trading_day(date_obj):
    # Check weekends first
    if date_obj.weekday() >= 5:
        return False
    # Check holiday registry strings
    return date_obj.date().isoformat() not in MARKET_HOLIDAYS


def previous_trading_day(date_obj):
    d = date_obj - dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d.date()


def is_market_open_now():
    now = get_now()
    if not is_trading_day(now):
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def is_eod_squareoff_time():
    return get_now().time() >= EOD_SQUAREOFF


def get_current_weekly_expiry():
    today = get_now().date()
    days_ahead = (1 - today.weekday()) % 7  # Tuesday target per exchange parameters
    expiry = today + dt.timedelta(days=days_ahead)
    while not is_trading_day(dt.datetime.combine(expiry, MARKET_OPEN)):
        expiry -= dt.timedelta(days=1)
    return expiry


def get_atm_strike(spot_price, step=50):
    return round(spot_price / step) * step


def get_or_set_daily_strikes(state, smart_api):
    today_str = get_now().date().isoformat()
    if state.get("daily_strikes_date") == today_str and state.get("daily_strikes"):
        return state["daily_atm"], state["daily_strikes"]

    if get_now().time() < STRIKE_LOCK_TIME:
        return None, None

    spot_price = ac.fetch_spot_ltp(smart_api)
    if spot_price is None:
        return None, None

    atm = get_atm_strike(spot_price)
    strikes = [atm - 100, atm + 100]  # Target strategy anchors

    state["daily_strikes_date"] = today_str
    state["daily_atm"] = atm
    state["daily_strikes"] = strikes
    return atm, strikes


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


def save_prev_day_cache(date_obj, candles_by_token):
    with open(PREV_DAY_CACHE_FILE, "w") as f:
        json.dump({"date": date_obj.isoformat(), "candles_by_token": candles_by_token}, f)


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


def compute_indicators(candles):
    df = pd.DataFrame(candles)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema25"] = df["close"].ewm(span=25, adjust=False).mean()

    today = get_now().date()
    today_mask = df["time"].dt.date == today
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].where(today_mask, 0).cumsum()
    cum_tp_vol = (typical_price * df["volume"]).where(today_mask, 0).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, pd.NA)

    return df


def check_entry_signal(df):
    if df is None or len(df) < 2:
        return False, None

    today = get_now().date()
    today_df = df[df["time"].dt.date == today]
    if len(today_df) < 1:
        return False, None

    idx_t = today_df.index[-1]
    if idx_t - 1 < df.index[0]:
        return False, None

    candle_t = df.loc[idx_t]
    candle_prev = df.loc[idx_t - 1]

    # Crossover condition check: EMA5 crossed under VWAP cleanly on closed candle T
    was_above = candle_prev["ema5"] >= candle_prev["vwap"]
    is_below = candle_t["ema5"] < candle_t["vwap"]

    if was_above and is_below:
        return True, idx_t
    return False, None


def calculate_custom_ema5_open(df, idx_t, next_open_price):
    """
    Derives the dynamic, forward-looking EMA5 at the boundary shift.
    Seeds Candle T+1's Open print directly into the 26 historical lookback array.
    """
    multiplier = 2 / (5 + 1)
    ema5_t = df.loc[idx_t, "ema5"]
    ema5_open = (next_open_price * multiplier) + (ema5_t * (1 - multiplier))
    return float(round(ema5_open, 2))


def send_to_webhook(payload):
    if not WEBHOOK_URL:
        print("WEBHOOK_URL environment secret missing. Payload:", payload)
        return None

    headers = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        headers["X-Webhook-Secret"] = WEBHOOK_SECRET

    for attempt in range(1, WEBHOOK_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=15)
            return resp
        except requests.RequestException:
            if attempt < WEBHOOK_RETRY_ATTEMPTS:
                time.sleep(WEBHOOK_RETRY_DELAY_SECONDS)
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
        print("Market closed. Skipping run.")
        return

    state = load_state()
    smart_api = ac.login()
    instruments = ac.download_instrument_master()
    expiry = get_current_weekly_expiry()

    atm, strikes_to_check = get_or_set_daily_strikes(state, smart_api)
    if atm is None:
        return

    prev_day = previous_trading_day(get_now())
    prev_day_cache = load_prev_day_cache(prev_day)
    today_start = dt.datetime.combine(get_now().date(), MARKET_OPEN)

    # -------------------------------------------------------------------------
    # PART 1: SPREAD MONITORING & EXIT OPERATIONS
    # -------------------------------------------------------------------------
    if state["open_position"] is not None:
        pos = state["open_position"]
        token_info = ac.resolve_option_token(instruments, expiry, pos["short_strike"], pos["option_type"])
        
        if token_info:
            candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
            df = compute_indicators(candles)
            if df is not None:
                current_price = df.iloc[-1]["close"]

                sl_hit = current_price >= pos["sl_price"]
                target_hit = current_price <= pos["target_price"]

                if sl_hit or target_hit or is_eod_squareoff_time():
                    reason = "SL" if sl_hit else ("TARGET" if target_hit else "EOD")
                    
                    exit_payload = {
                        "action": "EXIT_SPREAD",
                        "reason": reason,
                        "short_leg": {"symbol": pos["short_symbol"], "token": pos["short_token"], "qty": pos["qty"]},
                        "long_leg": {"symbol": pos["long_symbol"], "token": pos["long_token"], "qty": pos["qty"]},
                        "time": get_now().isoformat()
                    }
                    
                    resp = send_to_webhook(exit_payload)
                    if webhook_confirmed_ok(resp):
                        state["open_position"] = None
                        save_state(state)
        return

    # -------------------------------------------------------------------------
    # PART 2: CORE SCANNING & ENTRY SIGNALS
    # -------------------------------------------------------------------------
    for strike in strikes_to_check:
        for opt_type in ["CE", "PE"]:
            token_info = ac.resolve_option_token(instruments, expiry, strike, opt_type)
            if not token_info:
                continue

            candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
            df = compute_indicators(candles)
            
            is_triggered, idx_t = check_entry_signal(df)
            if is_triggered:
                live_candles = ac.fetch_5min_candles(smart_api, token_info["token"], start_time=today_start)
                if not live_candles:
                    continue
                next_candle_open = live_candles[-1]["open"]

                # Run custom boundary calculations
                ema5_open = calculate_custom_ema5_open(df, idx_t, next_candle_open)
                entry_limit_price = float(round(ema5_open * 0.95, 2))

                trigger_candle = df.loc[idx_t]
                sl_price = float(round(max(trigger_candle["high"], trigger_candle["vwap"]), 2))

                risk_amount = sl_price - entry_limit_price
                target_price = float(round(entry_limit_price - (2 * risk_amount), 2))

                # Identify protection hedge leg parameters (ATM +- 400)
                long_strike = strike + 300 if opt_type == "CE" else strike - 300
                long_token_info = ac.resolve_option_token(instruments, expiry, long_strike, opt_type)
                if not long_token_info:
                    continue

                long_live_candles = ac.fetch_5min_candles(smart_api, long_token_info["token"], start_time=today_start)
                long_open = long_live_candles[-1]["open"] if long_live_candles else 0.0

                # Package credit spread bundle matrix
                spread_payload = {
                    "action": "ENTRY_SPREAD",
                    "option_type": opt_type,
                    "short_leg": {
                        "symbol": token_info["symbol"],
                        "token": token_info["token"],
                        "limit_price": entry_limit_price,
                        "qty": token_info["lot_size"]
                    },
                    "long_leg": {
                        "symbol": long_token_info["symbol"],
                        "token": long_token_info["token"],
                        "limit_price": float(round(long_open * 1.05, 2)), # Buffer added to guarantee protective fill
                        "qty": long_token_info["lot_size"]
                    },
                    "sl_price": sl_price,
                    "target_price": target_price,
                    "time": get_now().isoformat()
                }

                # Save raw frame snapshot for auditing verification inside candle_history/
                df.to_json(f"{HISTORY_DIR}/{token_info['symbol']}_{get_now().date().isoformat()}.json")

                resp = send_to_webhook(spread_payload)
                if webhook_confirmed_ok(resp):
                    state["open_position"] = {
                        "option_type": opt_type,
                        "short_strike": strike,
                        "short_symbol": token_info["symbol"],
                        "short_token": token_info["token"],
                        "long_symbol": long_symbol_info := long_token_info["symbol"],
                        "long_token": long_token_info["token"],
                        "qty": token_info["lot_size"],
                        "entry_price": entry_limit_price,
                        "sl_price": sl_price,
                        "target_price": target_price
                    }
                    save_state(state)
                    save_prev_day_cache(prev_day, prev_day_cache)
                    return

    save_state(state)
    save_prev_day_cache(prev_day, prev_day_cache)


if __name__ == "__main__":
    main()
