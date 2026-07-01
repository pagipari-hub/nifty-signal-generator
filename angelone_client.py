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
import datetime as dt

import pyotp
import requests
from SmartConnect import SmartConnect
from smartExceptions import DataException

INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
INSTRUMENT_MASTER_CACHE = "instrument_master.json"

# Angel One's rate limiter mitigation configurations
RATE_LIMIT_RETRY_ATTEMPTS = 3
RATE_LIMIT_RETRY_DELAY_SECONDS = 3
PRE_CALL_DELAY_SECONDS = 1.5


def _call_with_retry(label, func, attempts=RATE_LIMIT_RETRY_ATTEMPTS,
                      delay=RATE_LIMIT_RETRY_DELAY_SECONDS):
    """
    Calls func() and retries on Angel One rate-limit errors (DataException
    with 'exceeding access rate' in the message, or a JSON decode failure
    caused by the same root cause). Re-raises on the final attempt, or
    immediately for any other kind of error.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except DataException as e:
            last_err = e
            msg = str(e)
            if "exceeding access rate" in msg.lower() or "access denied" in msg.lower():
                print(
                    f"{label}: rate-limited (attempt {attempt}/{attempts}): {msg}",
                    file=sys.stderr,
                )
                if attempt < attempts:
                    time.sleep(delay)
                    continue
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
    """
    Downloads (or loads cached) Angel One's official instrument master JSON.
    This is the ONLY reliable way to resolve a strike+expiry+CE/PE to the
    exact 'symbol' and 'token' Angel One expects -- never guess the format.
    """
    if not force_refresh and os.path.exists(INSTRUMENT_MASTER_CACHE):
        # Cache for the trading day -- verified via workflow dates
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

    FIX (Formatting Anchor): Angel One's master JSON strings strip leading zeros
    from the day field (e.g., '02JUL2026' is stored natively as '2JUL2026').
    Using explicit int casting on '%d' cleanly eliminates padding conflicts.
    """
    day_str = str(int(expiry_date.strftime("%d")))
    month_year_str = expiry_date.strftime("%b%Y").upper()
    expiry_str = f"{day_str}{month_year_str}"  # Converts 02JUL2026 -> 2JUL2026

    for inst in instruments:
        if (
            inst.get("name") == "NIFTY"
            and inst.get("exch_seg") == "NFO"
            and inst.get("instrumenttype") == "OPTIDX"
            and inst.get("expiry") == expiry_str
            and inst.get("symbol", "").endswith(option_type)
        ):
            try:
                # Angel One stores strike metrics scaled up * 100
                inst_strike = float(inst.get("strike", -1)) / 100  
            except (ValueError, TypeError):
                continue

            if int(inst_strike) == int(strike):
                return {
                    "symbol": inst.get("symbol"),
                    "token": inst.get("token"),
                    "lot_size": int(inst.get("lotsize", 65)),  # Fallback 65 per 2026 NSE requirements
                }

    print(f"No match found for NIFTY {strike} {option_type} expiry {expiry_str}",
          file=sys.stderr)
    return None


def fetch_5min_candles(smart_api, token, start_time=None, lookback_minutes=180):
    """
    Fetches 5-min OHLC candles for the given NFO token from Angel One.
    Returns list of dicts (oldest first): time, open, high, low, close, volume.
    """
    now = dt.datetime.now()
    start = start_time if start_time is not None else now - dt.timedelta(minutes=lookback_minutes)

    params = {
        "exchange": "NFO",
        "symboltoken": token,
        "interval": "FIVE_MINUTE",
        "fromdate": start.strftime("%Y-%m-%d %H:%M"),
        "todate": now.strftime("%Y-%m-%d %H:%M"),
    }

    time.sleep(PRE_CALL_DELAY_SECONDS)  # Anti-throttling buffer spacing

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

    return candles


def fetch_spot_ltp(smart_api):
    """
    Fetches NIFTY 50 index spot LTP, used to compute ATM strike.
    Constant index identifier token used: 99926000
    """
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
