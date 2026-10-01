"""
NIFTY weekly options strategy runner -- DATA + SIGNAL GENERATION ONLY.

Runs on GitHub Actions every 5 min during market hours.
1. Logs into Angel One (data APIs only, no static IP needed)
2. Resolves ATM +-2 CE/PE strikes via the official instrument master JSON
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
import datetime as dt

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
from webhook import send_heartbeat_if_needed, send_strike_lock_alert
from candle_priming import prime_candle_cache, lock_daily_pdl
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

        # FIX (2026-07-29, run-level duplicate-fetch root cause): plain
        # dict, created fresh here at the very top of each run and
        # threaded through every function below that can fetch today's
        # candles for a token. This is NOT prev_day_cache (which is
        # loaded/saved to disk further down and only ever covers the
        # PREVIOUS day) -- this is purely in-memory, purely for THIS
        # run, and exists so that if two different functions in the same
        # run both need the same token's candles, only the first
        # actually hits the network; the second gets the identical
        # result from here at zero extra cost.
        run_candle_cache = {}

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
        # prime_candle_cache() needs them -- see candle_priming.py's docstring.
        prev_day = previous_trading_day(now_ist().date())
        prev_day_cache = load_prev_day_cache(prev_day)
        today_start = dt.datetime.combine(now_ist().date(), MARKET_OPEN)

        if spot_price_at_lock is not None:
            print(f"Locked today's strikes: ATM={atm} (spot={spot_price_at_lock}) -> {leg_pairs}")
            send_strike_lock_alert(atm, leg_pairs, spot_price_at_lock)
            prime_candle_cache(leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_candle_cache)
            # NEW (2026-08-13, PDL fallback entry -- sell side only):
            # locks state["daily_pdl"] once for the day, immediately
            # after priming -- see candle_priming.lock_daily_pdl()'s own
            # docstring for why this specific ordering (reads directly
            # from prev_day_cache, which priming's call just populated
            # moments earlier in this same run, rather than re-fetching).
            lock_daily_pdl(state, leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_candle_cache)

        save_state(state)

        # ---- Manage existing open position first ----
        if state["open_position"] is not None:
            pos = state["open_position"]
            if pos.get("spread"):
                manage_spread_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_candle_cache)
            else:
                # FIX (backward compatibility): a position opened before this
                # rework has no "spread" key -- it must keep being managed by
                # the OLD dynamic-SL, single-leg exit logic untouched, not the
                # new fixed-SL spread logic.
                manage_legacy_single_leg_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_candle_cache)

            save_state(state)
            save_prev_day_cache(prev_day, prev_day_cache)
            return

        # ---- No open position: manage a resting pending_signal, if any ----
        if state.get("pending_signal") is not None:
            pending = state["pending_signal"]

            # FIX (2026-07-24): validate the resting signal actually belongs
            # to TODAY's currently-locked leg_pairs before trusting it.
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
                manage_pending_signal(state, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_candle_cache)
                save_state(state)
                save_prev_day_cache(prev_day, prev_day_cache)
                return

        # ---- No position, no (valid) pending signal: scan for a fresh entry trigger ----
        scan_for_new_signal(state, leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_candle_cache)

        save_state(state)
        save_prev_day_cache(prev_day, prev_day_cache)
    finally:
        release_run_lock()


if __name__ == "__main__":
    main()
