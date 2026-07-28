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

This is the orchestrator only -- see the sibling modules for the actual
logic: config, calendar_utils, locking, state, market_data, instrument,
indicators, signal_engine, pending, position, webhook, logging_utils,
candle_priming.
"""

import sys
import time
import datetime as dt

# ---------------------------------------------------------------------
# DEBUG (temporary): log every outbound HTTP request/response at the
# requests.Session level. This catches EVERYTHING built on `requests` --
# SmartApi's internal calls (login, getCandleData, ltpData, etc.) and
# our own webhook.py POSTs -- regardless of import order, because we're
# patching the Session class itself, not a specific instance. Placed
# here, before angelone_client/webhook are imported below, so no call
# is ever made before the patch is active.
#
# Only method, URL, status code, and elapsed time are logged -- never
# headers or body, since Angel One calls carry a live JWT in
# Authorization and our own webhook calls carry X-Webhook-Secret.
#
# FIX (2026-07-28, rate-limit-mystery investigation): the previous
# version of this patch logged whole-second timestamps and the raw URL
# (`[HTTP] {ts}  {method:6s} {url}  -> {resp.status_code}  ({elapsed:.2f}s)`).
# That was enough to confirm WHICH calls ran but not their true spacing --
# whole-second resolution can't distinguish "0.9s apart" from "0.1s
# apart", and long raw URLs push the actually-useful info off small
# screens / off the visible part of a scrolled log. Switched to
# millisecond-precision timestamps and a short friendly operation name
# (derived from the URL) so real call-to-call gaps are visible at a
# glance, matching the format used when eyeballing this against Angel
# One's rate-limit windows.
#
# Only recognizes endpoints THIS process actually calls (Angel One data
# APIs, our own webhook POST, ipify). Order placement (placeOrder) is a
# Shoonya call made by the separate bridge repo on Render, in a
# different process -- it will never show up in this log. If that needs
# to be correlated against these timestamps, this same patch (or its
# friendly-name table) needs to be added to the bridge's own code too.
#
# This is a temporary diagnostic (same spirit as the other DEBUG blocks
# in indicators.py / angelone_client.py) -- pull it out once the
# session-renewal / webhook-retry-duplication / rate-limit questions are
# answered.
# ---------------------------------------------------------------------
import requests

_orig_session_request = requests.Session.request

# Ordered (first-match-wins) substring -> friendly-name table. Checked
# case-insensitively against the full URL. Extend this if Angel One's
# SDK starts hitting an endpoint not already covered here -- unmatched
# URLs still get logged, just with a fallback label (see
# _friendly_call_name below), so nothing is ever silently dropped.
_CALL_NAME_PATTERNS = [
    ("generatetokens", "generateTokens"),
    ("generatesession", "generateSession"),
    ("getcandledata", "getCandleData"),
    ("gethistoricaldata", "getCandleData"),
    ("ltpdata", "getLTP"),
    ("getltpdata", "getLTP"),
    ("placeorder", "placeOrder"),
    ("modifyorder", "modifyOrder"),
    ("cancelorder", "cancelOrder"),
    ("orderbook", "getOrderBook"),
    ("orderhistory", "getOrderHistory"),
    ("ipify.org", "getPublicIP"),
]


def _friendly_call_name(url):
    """
    Maps a full request URL to a short, readable operation name for the
    HTTP log line. Falls back to the last non-empty path segment (or the
    bare host, if the path is empty) so an unrecognized endpoint is still
    identifiable rather than silently mislabeled.
    """
    lowered = url.lower()
    for needle, label in _CALL_NAME_PATTERNS:
        if needle in lowered:
            return label

    path = url.split("?", 1)[0].rstrip("/")
    segment = path.rsplit("/", 1)[-1]
    if segment and not segment.startswith("http"):
        return segment

    # Bare host fallback (e.g. a webhook POST straight to a domain root).
    host = url.split("//", 1)[-1].split("/", 1)[0]
    return host or url


def _logged_session_request(self, method, url, *args, **kwargs):
    t0 = time.monotonic()
    ts = dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # millisecond precision
    call_name = _friendly_call_name(url)
    try:
        resp = _orig_session_request(self, method, url, *args, **kwargs)
        elapsed = time.monotonic() - t0
        print(
            f"{ts}  {call_name:16s} -> {resp.status_code}  ({elapsed:.3f}s)",
            file=sys.stderr,
        )
        return resp
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(
            f"{ts}  {call_name:16s} -> EXCEPTION {e!r}  ({elapsed:.3f}s)",
            file=sys.stderr,
        )
        raise


requests.Session.request = _logged_session_request
# ---------------------------------------------------------------------
# End of HTTP logging patch.
# ---------------------------------------------------------------------

import angelone_client as ac
from config import MARKET_OPEN
from calendar_utils import now_ist, is_market_open_now, previous_trading_day, get_current_weekly_expiry
from locking import acquire_run_lock, release_run_lock
from state import load_state, save_state
from market_data import load_prev_day_cache, save_prev_day_cache
from instrument import get_or_set_daily_strikes
from signal_engine import scan_for_new_signal
from pending import manage_pending_signal
from position import manage_legacy_single_leg_exit, manage_spread_exit
from buy_signal_engine import (
    manage_open_buy_position_live,
    manage_pending_buy_signal_live,
    scan_for_new_buy_signal_live,
)
from webhook import send_heartbeat_if_needed, send_strike_lock_alert
from candle_priming import prime_candle_cache
from config import STRIKE_LOCK_TIME


def _pending_signal_matches_locked_legs(pending, leg_pairs):
    """
    FIX (2026-07-24, stale-pending_signal-off-a-different-ATM bug):
    manage_pending_signal() previously trusted whatever was sitting in
    state["pending_signal"] unconditionally -- it only ever reads
    pending["sell_strike"]/pending["option_type"] from state.json, and
    never cross-checks those against the CURRENT run's freshly-locked
    leg_pairs. state["daily_strikes_date"]/daily_atm get correctly
    overwritten every new trading day, but nothing analogous ever
    existed for pending_signal -- it has no created-date/ATM tag at
    all, so a pending_signal computed under one day's (or one moment's)
    ATM lock can silently persist into a LATER run under a DIFFERENT
    ATM lock, and still get filled as if it were current.

    Confirmed root cause, 2026-07-24 case study: a pending_signal for
    SELL 23800 PE / HEDGE 23500 PE -- which only exists under
    build_leg_pairs(23900) -- was still sitting in state.json on a
    9:32 run where the FRESH lock that same run performed was
    ATM=23700 (leg_pairs: PE 23600/23300, CE 23800/24100). Because
    state["open_position"] was None and state["pending_signal"] was
    truthy, main() routed straight to manage_pending_signal() without
    ever consulting the leg_pairs this run itself had just locked --
    filling a real (paper) trade on a strike/side that today's actual
    locked pair never included at all.

    Fix: before handing a resting pending_signal to manage_pending_signal(),
    confirm its (sell_strike, option_type) pair actually appears in
    TODAY's currently-locked leg_pairs. If it doesn't, the signal is
    stale (left over from an earlier ATM lock that was never cleared)
    and is discarded here -- never filled, never partially managed.
    """
    valid_pairs = {(leg["sell_strike"], leg["option_type"]) for leg in leg_pairs}
    return (pending.get("sell_strike"), pending.get("option_type")) in valid_pairs


def main():
    if not is_market_open_now():
        print("Market closed (outside hours or holiday) -- skipping run.")
        return

    # FIX (overlapping-run guard): acquire the lock before doing any API
    # work. If a previous run is still active (fresh lock present), skip
    # this run entirely rather than firing a second concurrent session at
    # Angel One under the same API key -- see locking.py for why this was
    # the likely root cause of the rate-limit errors.
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

        # FIX (2026-07-10, session-per-run root cause): was ac.login() --
        # a full TOTP-based generateSession() on every single run, every
        # 5 minutes, all day. ac.login_with_cache() reuses a cached
        # session (angel_session.json, restored via GH Actions cache --
        # see signal_generator.yml) via the lightweight
        # generateToken(refresh_token) renewal when one exists for today,
        # and only falls back to a full TOTP login otherwise. See
        # angelone_client.py's login_with_cache() docstring for the full
        # root-cause writeup (2026-07-10 case study: CE leg candle fetch
        # failing outright on the first data call right after a fresh
        # login, both at 09:35 and 09:45).
        smart_api = ac.login_with_cache()
        instruments = ac.download_instrument_master()

        expiry = get_current_weekly_expiry()

        atm, leg_pairs, spot_price_at_lock = get_or_set_daily_strikes(state, smart_api)
        if atm is None:
            if now_ist().time() < STRIKE_LOCK_TIME:
                print(f"Before {STRIKE_LOCK_TIME.strftime('%H:%M')} strike-lock time -- skipping signal check this run.")
            else:
                print("Could not fetch spot price to lock today's strikes, aborting this run.", file=sys.stderr)
            return

        # FIX (moved up, 2026-07-15): prev_day / prev_day_cache / today_start
        # now need to exist BEFORE the strike-lock block below, since
        # prime_candle_cache() (called only on the run that just performed
        # the lock) needs them to warm up the two sell legs' candle cache
        # immediately after lock -- see candle_priming.py's docstring for
        # why this is a dedicated step rather than just waiting for
        # scan_for_new_signal() to hit the same fetch later in this run.
        # Root cause this addresses: 2026-07-15 case study, where token
        # 57345's prev-day fetch was rate-limited on both the 9:31 and
        # 9:35 runs (each already loaded with a full TOTP re-login just
        # before it), leaving EMA25 computed with zero previous-day
        # warm-up (136.51 vs. an expected ~188) and blocking what should
        # have been a live entry signal on the PE leg both times.
        prev_day = previous_trading_day(now_ist().date())
        prev_day_cache = load_prev_day_cache(prev_day)
        today_start = dt.datetime.combine(now_ist().date(), MARKET_OPEN)

        if spot_price_at_lock is not None:
            print(f"Locked today's strikes: ATM={atm} (spot={spot_price_at_lock}) -> {leg_pairs}")
            send_strike_lock_alert(atm, leg_pairs, spot_price_at_lock)
            prime_candle_cache(leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)

        save_state(state)

        # ---- Buy-side: check open buy position EVERY run, regardless of ----
        # ---- what the sell side is doing this run (Pragnesh's call: a ----
        # ---- live buy position is always monitored, no exceptions -- ----
        # ---- unlike buy signal scanning/pending management below, which ----
        # ---- only runs when the sell side has nothing to do this run). ----
        if state.get("open_buy_position") is not None:
            manage_open_buy_position_live(state, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)
            save_state(state)
            save_prev_day_cache(prev_day, prev_day_cache)

        # ---- Manage existing open position first ----
        if state["open_position"] is not None:
            pos = state["open_position"]
            if pos.get("spread"):
                manage_spread_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)
            else:
                # FIX (backward compatibility): a position opened before this
                # rework has no "spread" key -- it must keep being managed by
                # the OLD dynamic-SL, single-leg exit logic untouched, not the
                # new fixed-SL spread logic. See position.manage_legacy_single_leg_exit().
                manage_legacy_single_leg_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)

            save_state(state)
            save_prev_day_cache(prev_day, prev_day_cache)
            return

        # ---- No open position: manage a resting pending_signal, if any ----
        if state.get("pending_signal") is not None:
            pending = state["pending_signal"]

            # FIX (2026-07-24): validate the resting signal actually belongs
            # to TODAY's currently-locked leg_pairs before trusting it --
            # see _pending_signal_matches_locked_legs() docstring above for
            # the full root-cause writeup. A mismatch means this
            # pending_signal is stale (left over from an earlier ATM lock
            # that was never cleared) and must be discarded, never filled.
            if not _pending_signal_matches_locked_legs(pending, leg_pairs):
                print(
                    f"[STALE PENDING_SIGNAL] {pending.get('sell_symbol', '?')} "
                    f"(sell_strike={pending.get('sell_strike')}, option_type={pending.get('option_type')}) "
                    f"does not match any leg in today's locked leg_pairs ({leg_pairs}) -- "
                    "discarding without filling or managing it, and continuing to scan fresh this run.",
                    file=sys.stderr,
                )
                state["pending_signal"] = None
                # Deliberately fall through to scan_for_new_signal() below in
                # this SAME run, rather than return -- a stale signal being
                # discarded shouldn't cost this run its chance to catch a
                # genuine fresh crossover on today's real leg_pairs.
            else:
                manage_pending_signal(state, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)
                save_state(state)
                save_prev_day_cache(prev_day, prev_day_cache)
                return

        # ---- No position, no (valid) pending signal: scan for a fresh entry trigger ----
        scan_for_new_signal(state, leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)

        # ---- Buy-side signal scanning/pending management: only reached ----
        # ---- when the sell side had nothing to do this run (Pragnesh's ----
        # ---- call: acceptable to skip on runs where sell is busy -- the ----
        # ---- entry-condition overlap this would miss is rare outside ----
        # ---- choppy markets, unlike an OPEN buy position, which is always ----
        # ---- checked above regardless of the sell branch taken this run). ----
        if state.get("pending_buy_signal") is not None:
            manage_pending_buy_signal_live(state, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)
        else:
            scan_for_new_buy_signal_live(state, leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start)

        save_state(state)
        save_prev_day_cache(prev_day, prev_day_cache)
    finally:
        release_run_lock()


if __name__ == "__main__":
    main()
