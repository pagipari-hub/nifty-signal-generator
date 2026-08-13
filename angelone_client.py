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

# FIX (2026-07-10, session-per-run root cause): this file previously had
# no concept of session persistence at all -- every run called login()
# below, which performs a full TOTP-based generateSession(). Angel One's
# own docs state a session stays active until 12 midnight unless
# explicitly logged out, and the SDK exposes a lightweight
# generateToken(refresh_token) renewal (no TOTP required) for exactly
# this case -- distinct from, and much lighter than, generateSession().
# Repeatedly hitting generateSession() every 5 minutes, all day, is a
# strong suspect for the "exceeding access rate" failures that were
# landing consistently on the FIRST data call right after a fresh login
# (2026-07-10 case study: 09:35 and 09:45 runs, CE leg fetch failed
# outright both times immediately post-login, while later calls in the
# same run succeeded).
#
# SESSION_CACHE_FILE is deliberately NOT committed to git (unlike
# state.json) -- it must only ever be restored/saved via the GitHub
# Actions cache (see signal_generator.yml), the same mechanism already
# used for INSTRUMENT_MASTER_CACHE. Putting a live refresh token into
# state.json would bake a real credential into the repo's permanent git
# history on every commit; the Actions cache has no such permanence.
SESSION_CACHE_FILE = "angel_session.json"

# FIX (2026-07-10, AG8001-on-renewal root cause -- source-confirmed, NOT
# YET VERIFIED against a live run): the SDK's own generateToken() was
# tried first and failed 4/4 real attempts with AG8001 "Invalid Token" on
# refresh tokens under 10 minutes old. Two follow-up theories (missing
# access_token in the constructor; a baked-in "Bearer " prefix on the
# jwtToken) were both tested and ruled out. Inspecting the SDK's actual
# source (requestHeaders() in smartapi-python's smartConnect.py) shows it
# NEVER includes an "Authorization" header -- structurally absent from
# that method, not conditional on what's passed to the constructor --
# while Angel One's own REST docs for this exact endpoint
# (generateTokens) explicitly require "Authorization: Bearer <token>"
# alongside the refresh token in the request body. This bypasses the
# SDK's generateToken() entirely and calls the documented REST endpoint
# directly, building every header per Angel One's own example, so
# renewal no longer depends on however the SDK does (or doesn't) attach
# auth internally.
#
# Built post-market-close (2026-07-10, 15:30+ IST) -- first real test is
# the second run of the next trading day (first run of the day forces a
# full TOTP login since there's no cache yet; the SECOND run is the
# actual test of this renewal path).
GENERATE_TOKEN_URL = "https://apiconnect.angelone.in/rest/auth/angelbroking/jwt/v1/generateTokens"


def _strip_bearer_prefix(token):
    """
    FIX (2026-07-15, AG8001 root cause CONFIRMED): the 2026-07-10 debug
    print (jwtToken starts_with_Bearer=...) was added specifically to
    test this theory, and the docstring above GENERATE_TOKEN_URL notes it
    as "tested and ruled out" at the time. The 2026-07-15 09:31 IST live
    run's actual debug output showed it firing True:
    "jwtToken starts_with_Bearer=True len=1233 | refreshToken
    starts_with_Bearer=False len=1027" -- Angel One's generateSession()
    IS returning jwtToken with a literal "Bearer " prefix baked into the
    string (refreshToken in the same response does NOT have it -- the
    API is not uniform about this).

    login_with_cache() was caching that jwtToken verbatim as
    access_token, and _renew_session_direct() then built the
    Authorization header as f"Bearer {access_token}" -- doubling the
    prefix into "Bearer Bearer eyJ...", which Angel One's generateTokens
    endpoint correctly rejects as AG8001 Invalid Token. This explains
    why every renewal attempt failed, including on refresh tokens only
    seconds old -- it was never actually about token age.

    Applied defensively at every point a token is cached or used to
    build an Authorization header, so this self-heals regardless of
    which endpoint's response does or doesn't carry the prefix.
    """
    if token and token.strip().lower().startswith("bearer "):
        return token.strip()[len("Bearer "):].strip()
    return token


def _renew_session_direct(api_key, access_token, refresh_token):
    """
    Direct REST call to Angel One's session-renewal endpoint, bypassing
    the SDK's own generateToken() -- see FIX note above GENERATE_TOKEN_URL
    for why. Returns a dict with jwtToken/refreshToken/feedToken on
    success, or None on any failure (bad response shape, network error,
    non-2xx, or an explicit success=False/status=False body). Never
    raises -- callers treat None exactly like a failed SDK call and fall
    back to full login.

    Response shape is defensively checked for BOTH "status" (used by
    generateSession's success response) and "success" (used by this
    endpoint's own documented AG8001 error response) keys, since Angel
    One's API is inconsistent about which key different endpoints use --
    confirmed inconsistency, not a guess: generateSession successes use
    "status": true, while every AG8001 error seen today used "success":
    false. The successful-response shape for THIS endpoint hasn't been
    observed yet, so both are checked rather than assuming one.

    FIX (2026-07-15): strips any "Bearer " prefix from BOTH tokens before
    building the Authorization header -- see _strip_bearer_prefix() for
    the root-cause writeup. Defensive here too (not just at the caching
    point in login_with_cache()) in case an already-cached session from
    before this fix still carries the doubled prefix.
    """
    access_token = _strip_bearer_prefix(access_token)
    refresh_token = _strip_bearer_prefix(refresh_token)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-UserType": "USER",
        "X-SourceID": "WEB",
        "X-ClientLocalIP": "127.0.0.1",
        "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "00:00:00:00:00:00",
        "X-PrivateKey": api_key,
    }
    payload = {"refreshToken": refresh_token}

    try:
        resp = requests.post(GENERATE_TOKEN_URL, headers=headers, json=payload, timeout=15)
    except requests.RequestException as e:
        print(f"Direct session renewal request failed: {e!r}", file=sys.stderr)
        return None

    try:
        body = resp.json()
    except ValueError:
        print(
            f"Direct session renewal: non-JSON response (status={resp.status_code}): "
            f"{resp.text[:200]}",
            file=sys.stderr,
        )
        return None

    ok = body.get("status") is True or body.get("success") is True
    if not ok:
        print(f"Direct session renewal failed (http_status={resp.status_code}): {body}",
              file=sys.stderr)
        return None

    data = body.get("data") or {}
    if not data.get("jwtToken") or not data.get("refreshToken"):
        print(f"Direct session renewal: response missing expected token fields: {body}",
              file=sys.stderr)
        return None

    return data


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
    """
    UNCHANGED -- full TOTP-based login, called on every invocation. Kept
    exactly as-is for backward compatibility / reference. New code should
    call login_with_cache() instead (see below), which only falls back to
    this full flow when there's no valid cached session for today.
    """
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


def _load_session_cache():
    if not os.path.exists(SESSION_CACHE_FILE):
        return None
    try:
        with open(SESSION_CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_session_cache(tokens):
    with open(SESSION_CACHE_FILE, "w") as f:
        json.dump(tokens, f)


def login_with_cache():
    """
    Session-aware login. Tries, in order:
      1. A cached session from earlier today (angel_session.json, restored
         via GH Actions cache -- see SESSION_CACHE_FILE note above) --
         renewed via the lightweight generateToken(refresh_token) call,
         which needs no TOTP and is far cheaper than generateSession().
      2. A full TOTP-based generateSession(), if there's no cache for
         today, the cache file is missing/corrupt, or the renewal call
         itself fails (e.g. refresh token expired/invalidated).

    Either path ends with a fresh angel_session.json written for the next
    run to pick up, and returns a ready-to-use smart_api object -- same
    contract as login(), so callers don't need any other changes.
    """
    api_key = os.environ["ANGEL_API_KEY"]
    today_str = dt.date.today().isoformat()  # workflow sets TZ=Asia/Kolkata

    cached = _load_session_cache()

    if cached and cached.get("session_date") == today_str and cached.get("refresh_token"):
        renewed = _renew_session_direct(
            api_key, cached.get("access_token"), cached["refresh_token"]
        )

        if renewed:
            print("Angel One session renewed via direct REST call (no TOTP).")
            # FIX (2026-07-15): strip any "Bearer " prefix before caching
            # or handing to the SDK -- see _strip_bearer_prefix() for the
            # root-cause writeup. Unconfirmed whether this renewal
            # endpoint's own response ever carries the prefix (only
            # generateSession's jwtToken has been observed doing so so
            # far), but stripping defensively here is cheap insurance
            # against the same AG8001 loop recurring via this path.
            clean_jwt = _strip_bearer_prefix(renewed["jwtToken"])
            clean_refresh = _strip_bearer_prefix(renewed["refreshToken"])
            smart_api = SmartConnect(api_key=api_key)
            smart_api.setAccessToken(clean_jwt)
            smart_api.setRefreshToken(clean_refresh)
            if renewed.get("feedToken"):
                smart_api.setFeedToken(renewed["feedToken"])
            _save_session_cache({
                "access_token": clean_jwt,
                "refresh_token": clean_refresh,
                "feed_token": renewed.get("feedToken"),
                "session_date": today_str,
            })
            return smart_api

        print("Cached session renewal (direct REST) failed -- falling back to full login.",
              file=sys.stderr)

    # Full TOTP login: first run of the day, no cache, or renewal above failed.
    client_code = os.environ["ANGEL_CLIENT_CODE"]
    password = os.environ["ANGEL_PASSWORD"]
    totp_secret = os.environ["ANGEL_TOTP_SECRET"]
    totp = pyotp.TOTP(totp_secret).now()

    smart_api = SmartConnect(api_key=api_key)
    session_data = smart_api.generateSession(client_code, password, totp)

    if not session_data or not session_data.get("status"):
        print(f"LOGIN FAILED: {session_data}", file=sys.stderr)
        sys.exit(1)

    print("Angel One login OK (full TOTP login).")

    # DEBUG (kept, 2026-07-10 -- this print is what CONFIRMED the AG8001
    # root cause on 2026-07-15's live run: jwtToken came back with
    # starts_with_Bearer=True. Left in place going forward as a canary --
    # if Angel One ever changes this API behavior, this line will show it
    # immediately rather than the bug silently reappearing.
    raw_jwt = session_data["data"]["jwtToken"]
    raw_refresh = session_data["data"]["refreshToken"]
    print(
        f"[DEBUG] jwtToken starts_with_Bearer={raw_jwt.startswith('Bearer ')} "
        f"len={len(raw_jwt)} | refreshToken starts_with_Bearer="
        f"{raw_refresh.startswith('Bearer ')} len={len(raw_refresh)}",
        file=sys.stderr,
    )

    # FIX (2026-07-15, AG8001 root cause): strip the confirmed "Bearer "
    # prefix from jwtToken (and defensively from refreshToken too) before
    # ever caching it. See _strip_bearer_prefix() docstring for the full
    # writeup -- this is the actual fix; everywhere else (renewal
    # endpoint calls, renewal-success caching) is defensive backup.
    clean_jwt = _strip_bearer_prefix(raw_jwt)
    clean_refresh = _strip_bearer_prefix(raw_refresh)

    _save_session_cache({
        "access_token": clean_jwt,
        "refresh_token": clean_refresh,
        "feed_token": session_data["data"].get("feedToken") or smart_api.getfeedToken(),
        "session_date": today_str,
    })
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


def fetch_5min_candles(smart_api, token, start_time=None, end_time=None, lookback_minutes=180):
    """
    FIX (2026-07-07, VWAP/EMA divergence root cause): previously `todate`
    always defaulted to dt.datetime.now() -- the moment THIS call runs --
    with no way to cap it. That's fine for a normal "today so far" fetch,
    but get_candles_with_cache()'s prev-day warm-up fetch calls this with
    only start_time=prev_day_start and no upper bound, so `todate` ended
    up being "right now" every time -- meaning that "prev-day" fetch
    actually spanned from previous-day market open all the way through
    THE CURRENT LIVE RUN, silently pulling today's candles into what got
    cached and persisted as prev_day_candles.json. Later runs then merged
    that already-contaminated prev-day series with a fresh today-fetch,
    and any dedup mismatch on the "time" string was enough to distort
    EMA5/EMA25 -- confirmed independently twice: (1) the original VWAP-
    divergence case study, and (2) a live paper-trading case on
    2026-07-06 where the system's implied EMA5 (from a locked entry
    limit) didn't match the EMA5 visibly plotted on the chart for the
    same candle.

    Adding an optional end_time parameter (defaults to now() for
    backward compatibility -- normal "up to this moment" fetches are
    unaffected) lets prev-day fetches be explicitly bounded to the
    previous day's own close, so they can never again bleed into the
    current run's live candles.

    FIX (2026-08-13, stale-todate-across-retries root cause -- missing
    entry candle): when end_time is None ("fetch up to now", i.e. every
    live today-candle fetch), `todate` was still only computed ONCE,
    before PRE_CALL_DELAY_SECONDS and before _call_with_retry()'s own
    backoff loop. _rate_limit_backoff_delay() is exponential (5s, 10s,
    20s, 40s+jitter) -- a couple of rate-limit retries can burn 30-75+
    seconds. The retry then SUCCEEDS (no exception, no error log), so
    nothing downstream flags a problem -- but the `todate` it asked for
    was frozen at the pre-backoff timestamp. If the entry/trigger
    candle's 5-min window only closed DURING that backoff wait, Angel
    One was never even asked for it: the fetch "succeeds" against a
    stale window, silently short of the newest candle. Confirmed
    2026-08-13: a rate-limit backoff on a live run coincided with the
    run's own entry candle being absent from the fetched series even
    though the call itself reported success.

    Fix: when end_time was not explicitly passed, `todate` is now
    recomputed fresh on EVERY attempt inside _call_with_retry (including
    the first), not calculated once upfront -- so a late-succeeding
    retry asks for data up through the actual current time, not the
    pre-backoff one. `start`/`fromdate` is deliberately NOT recomputed
    per attempt (only the upper bound caused the missing-candle bug).
    When end_time WAS explicitly passed (the prev-day bounded fetch),
    behaviour is unchanged -- that end never moves.
    """
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
    end_is_now = end_time is None
    end = end_time if end_time is not None else now
    start = start_time if start_time is not None else end - dt.timedelta(minutes=lookback_minutes)

    def _build_params():
        # See FIX (2026-08-13) above: only re-anchor `todate` to the
        # current wall-clock time when the caller wanted "up to now" in
        # the first place. An explicit end_time (prev-day fetch) stays
        # fixed no matter how many attempts this takes.
        current_end = dt.datetime.now() if end_is_now else end
        return {
            "exchange": "NFO",
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": start.strftime("%Y-%m-%d %H:%M"),
            "todate": current_end.strftime("%Y-%m-%d %H:%M"),
        }

    params = _build_params()

    print(
        f"[DEBUG] fetch_5min_candles token={token} "
        f"system_now={now.isoformat()} (tzinfo={now.tzinfo}) "
        f"params.fromdate={params['fromdate']} params.todate={params['todate']}",
        file=sys.stderr,
    )

    time.sleep(PRE_CALL_DELAY_SECONDS)

    def _do_call():
        nonlocal params
        params = _build_params()
        return smart_api.getCandleData(params)

    try:
        response = _call_with_retry(
            "fetch_5min_candles",
            _do_call,
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


def fetch_option_ltp(smart_api, symbol, token):
    """
    NEW (P&L tracking, 2026-07-06): fetches the last-traded price for a
    single NFO option leg (used for the hedge leg's entry/exit price,
    since the hedge is never itself the leg whose candles are polled
    every run). Mirrors fetch_spot_ltp()'s call pattern -- same
    pre-call delay, same retry/backoff via _call_with_retry(), same
    "return None on any failure rather than raise" contract -- but hits
    the NFO exchange with the option's own tradingsymbol/token instead
    of the hardcoded NSE "Nifty 50" spot token.

    Callers (scan_for_new_signal(), manage_spread_exit()) already treat
    a None return as "log it and continue without blocking the real
    signal/exit" -- this function preserves that contract rather than
    raising, so a single failed LTP fetch can never take down a run
    that's managing a live position.
    """
    time.sleep(PRE_CALL_DELAY_SECONDS)

    try:
        response = _call_with_retry(
            "fetch_option_ltp",
            lambda: smart_api.ltpData("NFO", symbol, str(token)),
        )
    except DataException as e:
        print(f"Option LTP fetch failed for {symbol} (token={token}) after retries: {e}",
              file=sys.stderr)
        return None

    if not response or not response.get("status"):
        print(f"Option LTP fetch failed for {symbol} (token={token}): {response}",
              file=sys.stderr)
        return None
    return float(response["data"]["ltp"])
