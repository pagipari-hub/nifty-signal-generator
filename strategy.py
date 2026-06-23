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
import datetime as dt

import pandas as pd
import requests

import angelone_client as ac

STATE_FILE = "state.json"
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # set as a GitHub Secret
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SHARED_SECRET")  # simple auth between GH Action and webhook

LOT_SIZE = 75  # NIFTY lot size -- confirm current value before going live;
                # NSE revises this periodically.


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"open_position": None, "last_run": None}


def save_state(state):
    state["last_run"] = dt.datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_current_weekly_expiry():
    """
    Returns the current weekly NIFTY expiry date.

    IMPORTANT: NSE shifted NIFTY's weekly expiry from Thursday to TUESDAY,
    effective September 1, 2025 (SEBI directive to spread weekly expiry
    volume across the week). This was previously Thursday -- if NSE changes
    it again in the future, update the weekday number below (1 = Tuesday).

    Also: if the computed Tuesday is a market holiday, NSE shifts the
    expiry to the previous trading day -- this is NOT yet handled below.
    Check NSE's holiday calendar before relying on this near holidays.
    """
    today = dt.date.today()
    days_ahead = (1 - today.weekday()) % 7  # 1 = Tuesday
    return today + dt.timedelta(days=days_ahead)


def get_atm_strike(spot_price, step=50):
    return round(spot_price / step) * step


def compute_indicators(candles):
    df = pd.DataFrame(candles)
    if df.empty or len(df) < 25:
        return None

    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema25"] = df["close"].ewm(span=25, adjust=False).mean()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_tp_vol"] = (typical_price * df["volume"]).cumsum()
    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]

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
    return dt.datetime.now().time() >= dt.time(15, 20)


def send_to_webhook(payload):
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set -- skipping webhook call. Payload was:", payload)
        return None

    headers = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        headers["X-Webhook-Secret"] = WEBHOOK_SECRET

    try:
        resp = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=15)
        print(f"Webhook response [{resp.status_code}]: {resp.text}")
        return resp
    except requests.RequestException as e:
        print(f"Webhook call failed: {e}", file=sys.stderr)
        return None


def main():
    state = load_state()
    smart_api = ac.login()
    instruments = ac.download_instrument_master()

    expiry = get_current_weekly_expiry()
    spot_price = ac.fetch_spot_ltp(smart_api)
    if spot_price is None:
        print("Could not fetch spot price, aborting this run.", file=sys.stderr)
        return

    atm = get_atm_strike(spot_price)
    strikes_to_check = [atm - 100, atm - 50, atm, atm + 50, atm + 100]

    # ---- Manage existing open position first ----
    if state["open_position"] is not None:
        pos = state["open_position"]
        token_info = ac.resolve_option_token(instruments, expiry, pos["strike"], pos["option_type"])

        if token_info:
            candles = ac.fetch_5min_candles(smart_api, token_info["token"])
            df = compute_indicators(candles)

            if df is not None:
                last = df.iloc[-1]
                sl_hit = last["close"] > last["vwap"]
                target_hit = last["close"] <= pos["target_price"]

                if sl_hit or target_hit or is_eod_squareoff_time():
                    reason = "SL" if sl_hit else ("TARGET" if target_hit else "EOD")
                    send_to_webhook({
                        "action": "EXIT",
                        "reason": reason,
                        "symbol": pos["symbol"],
                        "qty": pos["qty"],
                        "price": float(last["close"]),
                        "time": dt.datetime.now().isoformat(),
                    })
                    state["open_position"] = None

        save_state(state)
        return

    # ---- No open position: look for a new entry signal ----
    for strike in strikes_to_check:
        for opt_type in ["CE", "PE"]:
            token_info = ac.resolve_option_token(instruments, expiry, strike, opt_type)
            if not token_info:
                continue

            candles = ac.fetch_5min_candles(smart_api, token_info["token"])
            df = compute_indicators(candles)
            if df is None:
                continue

            if check_entry_signal(df):
                trigger_candle = df.iloc[-1].to_dict()
                entry_price = compute_entry_price(trigger_candle)
                target_price = entry_price - 2 * (trigger_candle["high"] - entry_price)

                position = {
                    "symbol": token_info["symbol"],
                    "token": token_info["token"],
                    "strike": strike,
                    "option_type": opt_type,
                    "entry_price": entry_price,
                    "qty": LOT_SIZE,
                    "target_price": target_price,
                    "entry_time": dt.datetime.now().isoformat(),
                }

                send_to_webhook({
                    "action": "ENTRY",
                    "side": "SELL",
                    "symbol": position["symbol"],
                    "qty": LOT_SIZE,
                    "price": entry_price,
                    "target_price": target_price,
                    "time": position["entry_time"],
                })

                state["open_position"] = position
                save_state(state)
                print(f"SIGNAL: SELL {position['symbol']} @ {entry_price}")
                return

    save_state(state)
    print("No entry signal this run.")


if __name__ == "__main__":
    main()
