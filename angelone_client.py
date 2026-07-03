"""
Angel One SmartAPI client -- DATA FETCHING ONLY.

This is used by the GitHub Action to fetch live 5-min option candles and
generate signals. It does NOT place orders -- order placement happens via
the separate webhook (Shoonya), because only order APIs require a
whitelisted static IP. Data APIs do not, so this can safely run from
GitHub Actions' rotating IPs.

Credentials read from environment variables (GitHub Secrets):
  ANGEL_CLIENT_CODE
  ANGEL_PASSWORD       (or PIN, depending on your account type)
  ANGEL_TOTP_SECRET    (base32 secret used to GENERATE the live TOTP code)
  ANGEL_API_KEY
"""

import os
import sys
import json
import time
import random
import datetime as dt

import pyotp
import requests
from SmartApi import SmartConnect
from SmartApi.smartExceptions import DataException

INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
INSTRUMENT_MASTER_CACHE = "instrument_master.json"

# FIX (rate-limit retry rework): the old retry config (3 attempts, flat 3s
# delay) was tuned as if Angel One only enforced a per-second cap. Their
# own forum confirms getCandleData also has a per-MINUTE cap (~180/min)
# on top of the per-second one, and there are widespread reports of
# "exceeding access rate" firing even under the documented per-second
# limit -- consistent with the cap being enforced per API key across ALL
# concurrent sessions, not just the current process. A flat 3s x 3
# attempts (9s total) can never clear a per-minute window that's already
# been exhausted by an overlapping run. Switched to exponential backoff
# with jitter and a much longer ceiling so a retry sequence can actually
# survive a per-minute cap being hit, not just a per-second blip.
RATE_LIMIT_RETRY_ATTEMPTS = 4
RATE_LIMIT_BASE_DELAY_SECONDS = 5
RATE_LIMIT_MAX_DELAY_SECONDS = 45
RATE_LIMIT_JITTER_SECONDS = 2
PRE_CALL_DELAY_SECONDS = 1.5


def _rate_limit_backoff_delay(attempt):
    """
    Exponential backoff: 5s, 10s, 20s, 40s... capped at
    RATE_LIMIT_MAX_DELAY_SECONDS, plus a small random jitter so multiple
    retrying calls (e.g. across the several tokens fetched per run) don't
    all retry in lockstep and re-collide on the same rate-limit window.
    """
    base = RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
    capped = min(base, RATE_LIMIT_MAX_DELAY_SECONDS)
    jitter = random.uniform(0, RATE_LIMIT_JITTER_SECONDS)
    return capped + jitter


def _call_with_retry(label, func, attempts=RATE_LIMIT_RETRY_ATTEMPTS):
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except DataException as e:
            last_err = e
            msg = str(e)
            if "exceeding access rate" in msg.lower() or "access denied" in msg.lower():
                if attempt < attempts:
                    delay = _rate_limit_backoff_delay(attempt)
                    print(
                        f"{label}: rate-limited (attempt {attempt}/{attempts}), "
                        f"backing off {delay:.1f}s: {msg}",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                print(
                    f"{label}: rate-limited (attempt {attempt}/{attempts}), "
                    f"giving up: {msg}",
                    file=sys.stderr,
                )
            raise
    raise last_err


def login():
    client_code = os.environ["ANGEL_CLIENT_CODE"]
    password = os.environ["ANGEL_PASSWORD"]
    totp_secret = os.environ["ANGEL_TOTP_SECRET"]
    api_key = os.environ["ANGEL_API_KEY"]

    totp = pyotp.TOTP(totp_secret).now()

    smart_api = SmartConnect(api_key=api_key)
    session_data = smart_api.generateSession(client_code, password, totp)

    if not session_data or not session_data.get("status"):
        print(f"LOGIN FAILED: {session_data}", file=sys.stderr)
        sys.exit(1)

    print("Angel One login OK.")
    return smart_api


def download_instrument_master(force_refresh=False):
    if not force_refresh and os.path.exists(INSTRUMENT_MASTER_CACHE):
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(INSTRUMENT_MASTER_CACHE))
        if mtime.date() == dt.date.today():
            with open(INSTRUMENT_MASTER_CACHE, "r") as f:
                return json.load(f)

    print("Downloading fresh instrument master JSON...")
    resp = requests.get(INSTRUMENT_MASTER_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    with open(INSTRUMENT_MASTER_CACHE, "w") as f:
        json.dump(data, f)

    return data


def resolve_option_token(instruments, expiry_date, strike, option_type):
    """
    Looks up the exact Angel One 'symbol', 'token', and 'lot_size' for a
    NIFTY weekly option from the instrument master list.

    FIX (reverted the day-stripping change): a prior "fix" here assumed
    Angel One's master JSON strips leading zeros from the expiry day
    field and built expiry_str as e.g. "7JUL2026" instead of "07JUL2026".
    That assumption was wrong -- it broke every strike lookup on
    single-digit-day expiries (e.g. 2026-07-07), since it silently never
    matched anything and just returned None with a "No match found" log
    line. Confirmed wrong two ways: (1) the historically successful trade
    NIFTY07JUL2624050PE was resolved correctly under the original
    zero-padded logic, before that change landed; (2) real tradingsymbols
    in candle_history (e.g. NIFTY07JUL2623550CE) use a zero-padded day.
    Standard strftime("%d%b%Y") already zero-pads correctly -- no manual
    int-cast stripping needed.
    """
    expiry_str = expiry_date.strftime("%d%b%Y").upper()  # e.g. 07JUL2026

    for inst in instruments:
        if (
            inst.get("name") == "NIFTY"
            and inst.get("exch_seg") == "NFO"
            and inst.get("instrumenttype") == "OPTIDX"
            and inst.get("expiry") == expiry_str
            and inst.get("symbol", "").endswith(option_type)
        ):
            try:
                inst_strike = float(inst.get("strike", -1)) / 100
            except (ValueError, TypeError):
                continue

            if int(inst_strike) == int(strike):
                return {
                    "symbol": inst.get("symbol"),
                    "token": inst.get("token"),
                    "lot_size": int(inst.get("lotsize", 65)),
                }

    print(f"No match found for NIFTY {strike} {option_type} expiry {expiry_str}",
          file=sys.stderr)
    return None


def fetch_5min_candles(smart_api, token, start_time=None, lookback_minutes=180):
    # DEBUG (temporary -- investigating possible timezone mismatch): `now`
    # here is dt.datetime.now(), i.e. whatever timezone the process's
    # system clock is in. GitHub Actions runners default to UTC unless the
    # workflow explicitly sets TZ=Asia/Kolkata, while Angel One's
    # getCandleData expects fromdate/todate in IST. If the runner is UTC,
    # `todate` sent below would be ~5.5 hours behind actual IST "now",
    # which could silently truncate the candle window during market hours.
    # Not changing behavior yet -- logging both clocks so this can be
    # confirmed or ruled out from a live run's output first.
    now = dt.datetime.now()
    start = start_time if start_time is not None else now - dt.timedelta(minutes=lookback_minutes)

    params = {
        "exchange": "NFO",
        "symboltoken": token,
        "interval": "FIVE_MINUTE",
        "fromdate": start.strftime("%Y-%m-%d %H:%M"),
        "todate": now.strftime("%Y-%m-%d %H:%M"),
    }

    print(
        f"[DEBUG] fetch_5min_candles token={token} "
        f"system_now={now.isoformat()} (tzinfo={now.tzinfo}) "
        f"params.fromdate={params['fromdate']} params.todate={params['todate']}",
        file=sys.stderr,
    )

    time.sleep(PRE_CALL_DELAY_SECONDS)

    try:
        response = _call_with_retry(
            "fetch_5min_candles",
            lambda: smart_api.getCandleData(params),
        )
    except DataException as e:
        print(f"Candle fetch failed for token {token} after retries: {e}", file=sys.stderr)
        return []

    if not response or not response.get("status"):
        print(f"Candle fetch failed for token {token}: {response}", file=sys.stderr)
        return []

    candles = []
    for row in response.get("data", []):
        candles.append({
            "time": row[0],
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })

    # DEBUG (temporary): what's the actual latest candle Angel One handed
    # back for this token, vs what we asked for as todate?
    if candles:
        print(
            f"[DEBUG] token={token} latest candle returned: time={candles[-1]['time']} "
            f"close={candles[-1]['close']} (requested todate={params['todate']})",
            file=sys.stderr,
        )
    else:
        print(
            f"[DEBUG] token={token} NO candles returned for range "
            f"{params['fromdate']} -> {params['todate']}",
            file=sys.stderr,
        )

    return candles


def fetch_spot_ltp(smart_api):
    time.sleep(PRE_CALL_DELAY_SECONDS)

    try:
        response = _call_with_retry(
            "fetch_spot_ltp",
            lambda: smart_api.ltpData("NSE", "Nifty 50", "99926000"),
        )
    except DataException as e:
        print(f"Spot LTP fetch failed after retries: {e}", file=sys.stderr)
        return None

    if not response or not response.get("status"):
        print(f"Spot LTP fetch failed: {response}", file=sys.stderr)
        return None
    return float(response["data"]["ltp"])


# FIX (EOD hardening, re-integrated -- this function was dropped when an
# older draft got pasted back into the project): force_eod_exit() in
# strategy.py deliberately does NOT go through fetch_5min_candles() or
# resolve_option_token() -- those are the fragile, multi-call path that
# EOD square-off (a safety net, not a signal decision) shouldn't depend
# on. All it needs is SOME price for the Telegram/Sheets log line, since
# the actual exit order is placed MKT (price_type="MKT" in both
# place_real_order() and place_real_spread_order() in webhook.py) --
# execution doesn't depend on this value at all.
#
# Kept deliberately lighter-weight than fetch_spot_ltp(): fewer retry
# attempts, because this runs right at the EOD_SQUAREOFF boundary and
# must not let a rate-limit retry sequence (which can legitimately take
# up to ~45s per attempt under the normal retry config) delay a
# time-critical flatten. If it fails, the caller proceeds with a None
# price rather than blocking -- a missing log price is cosmetic, a
# delayed EOD exit is not.
_EOD_LTP_RETRY_ATTEMPTS = 2


def fetch_spot_ltp_once(smart_api):
    time.sleep(PRE_CALL_DELAY_SECONDS)

    try:
        response = _call_with_retry(
            "fetch_spot_ltp_once (EOD)",
            lambda: smart_api.ltpData("NSE", "Nifty 50", "99926000"),
            attempts=_EOD_LTP_RETRY_ATTEMPTS,
        )
    except DataException as e:
        print(f"[EOD] Spot LTP fetch failed after {_EOD_LTP_RETRY_ATTEMPTS} attempts "
              f"-- proceeding with EOD exit anyway, price will be logged as unknown: {e}",
              file=sys.stderr)
        return None

    if not response or not response.get("status"):
        print(f"[EOD] Spot LTP fetch failed: {response} -- proceeding with EOD exit anyway.",
              file=sys.stderr)
        return None
    return float(response["data"]["ltp"])
