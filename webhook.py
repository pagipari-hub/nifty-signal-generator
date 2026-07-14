"""
All communication with the webhook that holds the Shoonya session and
actually places (or paper-simulates) orders. This module never decides
whether to trade -- it only ships payloads and reports whether the
webhook confirmed them.
"""

import sys
import time

import requests

from config import WEBHOOK_URL, WEBHOOK_SECRET, WEBHOOK_RETRY_ATTEMPTS, WEBHOOK_RETRY_DELAY_SECONDS
from calendar_utils import now_ist


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


def send_strike_lock_alert(atm, leg_pairs, spot_price):
    """
    Fires once per day, on the run that actually performs the 9:30 strike
    lock (main.py only calls this when spot_price_at_lock is non-None --
    i.e. NOT on the same-day rebuild branch in instrument.py that just
    re-derives leg_pairs from an already-locked daily_atm on every
    subsequent run). Mirrors send_heartbeat_if_needed()'s one-shot
    pattern, but the "already sent today" guard lives in the caller
    (spot_price_at_lock is None on every run after the first), not here.
    """
    lines = [
        f"🔒 Strikes locked for {now_ist().strftime('%Y-%m-%d')}",
        f"ATM = {atm}  (spot = {spot_price})",
    ]
    for leg in leg_pairs:
        lines.append(
            f"{leg['option_type']}: SELL {leg['sell_strike']} / HEDGE {leg['hedge_strike']}"
        )
    message = "\n".join(lines)

    send_to_webhook({
        "action": "STRIKES_LOCKED",
        "message": message,
        "atm": atm,
        "spot_price": spot_price,
        "leg_pairs": leg_pairs,
        "time": now_ist().isoformat(),
    })
