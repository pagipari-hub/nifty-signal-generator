"""
BUY-side signal engine -- STANDALONE MODULE, wired into main.py.

CHANGED (2026-09-04): the entry trigger was reworked from an EMA5/EMA25/
VWAP crossover to an ROC-based trigger, per Pragnesh's explicit spec:

  - Trigger (this candle): ROC(BUY_ROC_PERIOD) crosses above
    BUY_ROC_CROSS_LEVEL (previous candle's ROC <= level, this candle's
    ROC > level) -- a genuine transition, not a persisting state, same
    "fresh crossover, not already-held" reasoning as the old EMA-based
    check and as the sell side's own crossover detection.
  - Filter: the trigger candle's own CLOSE must be above VWAP.
  - On that trigger candle N, rest a BUY limit at N's own CLOSE (not
    EMA5 anymore), to be filled on a later candle whose LOW touches it
    -- still a pullback entry, same fill mechanics as before.

SL = LOW of the candle immediately BEFORE the trigger candle (candle
N-1) -- no longer min(low, vwap) on the trigger candle itself. The
low-premium SL floor (Rs.99 / 10%, config.BUY_LOW_PREMIUM_SL_THRESHOLD /
BUY_LOW_PREMIUM_SL_MIN_PCT) is REINSTATED on top of this new SL --
widen-only, same as before, same as the sell side's own version.

Target = entry + BUY_TARGET_RISK_REWARD * (entry - SL), fixed 1:1 by
default (config.BUY_TARGET_RISK_REWARD, changed from 1:2 to 1:1 as part
of this same rework).

Same-strike guard REMOVED (2026-09-04): a buy signal used to be skipped
if the sell side already had an open position on that exact strike
(_strike_has_open_sell_position(), now deleted). Pragnesh's explicit
call: the buy side should fire independent of whatever the sell side is
doing, since a long option's downside is capped at the premium paid --
low enough cost that the same-strike overlap risk this guard was
protecting against is no longer worth the missed signals it was
blocking. Run-order gating in main.py (buy scanning only runs when sell
has nothing to do that run) and the single-slot pending_buy_signal /
open_buy_position limit are UNCHANGED -- only the same-strike check
itself was removed, not the surrounding run-order structure.

EMA5/EMA25 are no longer read by this module's own signal logic (VWAP
still is, as the trigger-candle filter). They remain available on the
shared indicator dataframe (indicators.compute_indicators()) for the
SELL side and for the squeeze diagnostics below, which are left
unconditional/unchanged -- costs nothing to keep logging them even
though they no longer gate a buy entry.
"""

import sys

import pandas as pd

from config import (
    BUY_PENDING_SIGNAL_MAX_CANDLES, BUY_TARGET_RISK_REWARD,
    BUY_ROC_PERIOD, BUY_ROC_CROSS_LEVEL,
    BUY_LOW_PREMIUM_SL_THRESHOLD, BUY_LOW_PREMIUM_SL_MIN_PCT,
    BUY_SCAN_WINDOWS,
)
from calendar_utils import now_ist
from pending import round_to_half
from indicators import compute_squeeze_metrics


def _condition_holds_buy(row):
    """
    CHANGED (2026-09-04): ROC-based state check, replacing the old
    EMA25/EMA5/VWAP condition. True when this candle's ROC is above
    BUY_ROC_CROSS_LEVEL AND its own close is above VWAP. Used by
    check_entry_signal_buy() (state check, for cancellation) --
    is_fresh_crossover_signal_buy() below adds the transition
    requirement on top of this for scanning brand-new entries.

    Guards against a NaN ROC (e.g. very first candles of the whole
    fetched history, before BUY_ROC_PERIOD prior closes exist) by
    treating NaN as "condition not met" rather than raising or silently
    comparing NaN (which pandas would otherwise evaluate as False anyway
    for `>`, but being explicit here makes the guard visible in code
    rather than relying on that NaN-comparison quirk).
    """
    roc = row["roc"]
    if roc is None or pd.isna(roc):
        return False
    return bool(roc > BUY_ROC_CROSS_LEVEL and row["close"] > row["vwap"])


def check_entry_signal_buy(df):
    """
    STATE check (not transition-based): is the ROC-based buy condition
    true on the latest candle? Used to decide whether a resting pending
    buy signal's setup has been INVALIDATED, not to scan for brand-new
    entries (see is_fresh_crossover_signal_buy() for that) -- same role
    this function has always played, just backed by the new ROC
    condition now instead of the old EMA-based one.
    """
    if df is None or len(df) < 1:
        return False
    return _condition_holds_buy(df.iloc[-1])


def is_fresh_crossover_signal_buy(df):
    """
    CHANGED (2026-09-04): TRANSITION check, now ROC-based. Fires when:

      - PREVIOUS candle: ROC <= BUY_ROC_CROSS_LEVEL (not yet crossed)
      - CURRENT candle:  ROC >  BUY_ROC_CROSS_LEVEL  AND  close > VWAP

    Replaces the old two-part check (pre-cross EMA25<EMA5<VWAP ordering
    on the previous candle, full EMA25<EMA5/EMA5>VWAP condition on the
    current one) with a single ROC-cross-above-level transition, plus
    the same "not already holding" guard against a re-fire on an
    unchanged state -- if ROC was already above the level last candle
    too, this is a persisting state, not a fresh cross, and won't fire
    again here (same reasoning as the sell side's own crossover check
    and the old EMA-based version of this function).

    NaN ROC on either candle (e.g. right at the edge of the fetched
    history, before BUY_ROC_PERIOD prior closes exist) is treated as
    "not enough data to judge a cross" -- fails closed, does not fire.
    """
    if df is None or len(df) < 2:
        return False

    current = df.iloc[-1]
    previous = df.iloc[-2]

    if pd.isna(current["roc"]) or pd.isna(previous["roc"]):
        return False

    crossed_above = previous["roc"] <= BUY_ROC_CROSS_LEVEL and current["roc"] > BUY_ROC_CROSS_LEVEL
    above_vwap = current["close"] > current["vwap"]
    return bool(crossed_above and above_vwap)


def compute_pending_buy_signal(trigger_candle, buy_leg_info, qty, prev_candle):
    """
    Locks a resting BUY limit order off the just-closed trigger candle N.

    CHANGED (2026-09-04, ROC rework): entry/SL/target formulas replaced
    wholesale, per Pragnesh's explicit spec:

      entry_limit = trigger candle N's own CLOSE (was EMA5[N]). Still a
      pullback entry -- the limit rests at this level and fills on a
      LATER candle whose LOW comes down to touch it (see
      check_fill_buy()), not at N's own close as a market order.

      sl_price = candle N-1's LOW. prev_candle is now a REQUIRED
      argument, not optional -- is_fresh_crossover_signal_buy() already
      requires df.iloc[-2] to exist for the ROC-cross check itself, so a
      valid prev_candle is guaranteed to be available by the time this
      is called from either scan path.

      target_price = entry_limit + BUY_TARGET_RISK_REWARD * risk, RR now
      1:1 by default (config.BUY_TARGET_RISK_REWARD, was 1:3 then 1:2 at
      various earlier points -- see that constant's own docstring).

    REINSTATED (2026-09-04, low-premium SL floor): Pragnesh's call --
    bring back the same widen-only floor the sell side and the old
    EMA-based buy engine both had (config.BUY_LOW_PREMIUM_SL_THRESHOLD /
    BUY_LOW_PREMIUM_SL_MIN_PCT, Rs.99 / 10%, unchanged values), now
    applied on top of the NEW prev-candle-low SL instead of the old
    min(low, vwap) SL. If entry_limit is under Rs.99, sl_price is
    additionally floored at entry_limit * (1 - 10%) -- this can only
    WIDEN the stop (push it further from entry), never tighten it or
    override the prev-candle-low value when that's already wider. Same
    "checked against the ROUNDED entry_limit" reasoning as the sell
    side's own version of this floor (a raw value near the threshold
    could legitimately round across it).

    Rounding: entry_limit rounded first, then the low-premium floor
    check runs against that rounded value, then sl_price (prev-candle
    low vs. the floor, whichever is lower/wider) is rounded, then
    risk/target_price are computed off the already-rounded values --
    same order as the sell side and the original buy engine, so the
    real risk:reward sent to the webhook matches BUY_TARGET_RISK_REWARD
    exactly rather than drifting off raw floats.

    Returns None if the computed risk (entry_limit - sl_price) is <= 0
    -- i.e. candle N-1's low (or the low-premium floor) is at or above
    N's own close, which shouldn't normally happen on a genuine upward
    ROC cross but is checked defensively rather than sending a
    nonsensical/inverted SL to the webhook. Callers must check for a
    None return and skip the signal (log + continue to the next leg)
    rather than treating it as a valid pending signal.
    """
    entry_limit = round_to_half(trigger_candle["close"])

    sl_price = prev_candle["low"]
    if entry_limit < BUY_LOW_PREMIUM_SL_THRESHOLD:
        sl_price = min(sl_price, entry_limit * (1 - BUY_LOW_PREMIUM_SL_MIN_PCT))
    sl_price = round_to_half(sl_price)

    risk = entry_limit - sl_price
    if risk <= 0:
        return None

    target_price = round_to_half(entry_limit + BUY_TARGET_RISK_REWARD * risk)

    return {
        "buy_symbol": buy_leg_info["symbol"],
        "buy_token": buy_leg_info["token"],
        "strike": buy_leg_info["strike"],
        "option_type": buy_leg_info["option_type"],
        "qty": qty,
        "entry_limit": entry_limit,
        "sl_price": sl_price,
        "target_price": target_price,
        "risk": risk,
        "trigger_time": now_ist().isoformat(),
        "candles_waited": 0,
    }


def check_fill_buy(pending, last_candle):
    """
    Fill check for a resting pending buy signal: has a later candle's LOW
    come down far enough to touch the resting BUY limit? UNCHANGED by the
    2026-09-04 ROC rework -- still `last["low"] <= pending["entry_limit"]`,
    regardless of what entry_limit is now anchored to.
    """
    return last_candle["low"] <= pending["entry_limit"]


def scan_for_new_buy_signal(candles_by_symbol, compute_indicators_fn, log=True):
    """
    Standalone scan across a dict of {leg_info: candles} for a fresh
    ROC-based buy crossover. Mirrors the live scan's shape but stays
    decoupled from state.json / smart_api, so it can run against
    historical candle_history/ CSVs (see test_buy_signal_engine.py)
    without a broker session.

    candles_by_symbol: dict of {leg_info_dict: list_of_candle_dicts},
    where leg_info_dict has at least "symbol", "token", "strike",
    "option_type", and "lot_size" (qty).

    Returns the first pending buy signal found (dict, see
    compute_pending_buy_signal()), or None if nothing fired this scan
    -- either because no leg had a fresh crossover, or because a
    crossover fired but compute_pending_buy_signal() returned None
    (risk <= 0, logged and skipped). Only ever returns at most one
    signal per call, same single-slot reasoning as the sell engine.
    """
    for leg_info, candles in candles_by_symbol.items():
        df = compute_indicators_fn(candles)
        if df is None:
            continue

        if is_fresh_crossover_signal_buy(df):
            trigger_candle = df.iloc[-1].to_dict()
            prev_candle = df.iloc[-2].to_dict()
            qty = leg_info.get("lot_size", leg_info.get("qty"))
            pending = compute_pending_buy_signal(trigger_candle, leg_info, qty, prev_candle)

            if pending is None:
                if log:
                    print(
                        f"[buy scan] ROC crossover fired on {leg_info.get('symbol', '?')} "
                        f"but computed risk <= 0 (prev candle low >= trigger close) -- "
                        "skipping this signal.",
                        file=sys.stderr,
                    )
                continue

            if log:
                _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(df.iloc[-1])
                ratio_str = f"{spread_atr_ratio:.2f}" if spread_atr_ratio is not None else "n/a"
                print(
                    f"PENDING BUY SIGNAL: BUY {pending['buy_symbol']} resting limit @ "
                    f"{pending['entry_limit']:.2f} (SL={pending['sl_price']:.2f}, "
                    f"target={pending['target_price']:.2f}) "
                    f"[roc={trigger_candle['roc']:.2f}] "
                    f"[squeeze diag: spread_pct={spread_pct:.3f}% spread/atr={ratio_str}]"
                )
            return pending
        elif log:
            last = df.iloc[-1]
            _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(last)
            ratio_str = f"{spread_atr_ratio:.2f}" if spread_atr_ratio is not None else "n/a"
            roc_str = f"{last['roc']:.2f}" if pd.notna(last["roc"]) else "n/a"
            print(
                f"[buy scan] {leg_info.get('symbol', '?')}: roc={roc_str} "
                f"close={last['close']:.2f} vwap={last['vwap']:.2f} "
                f"close>vwap={bool(last['close'] > last['vwap'])} "
                f"[squeeze diag: spread_pct={spread_pct:.3f}% spread/atr={ratio_str}] "
                "-- no fresh cross.",
                file=sys.stderr,
            )

    if log:
        print("No buy entry signal this scan.")
    return None


def manage_pending_buy_signal(pending, df, max_candles=BUY_PENDING_SIGNAL_MAX_CANDLES):
    """
    Standalone lifecycle check for a resting pending buy signal against a
    fresh candle dataframe: fill / cancel / expire / still-resting.
    UNCHANGED in shape by the 2026-09-04 ROC rework -- fill checked
    first, then cancellation (now backed by the ROC-based
    check_entry_signal_buy()), then expiry.

    Returns one of:
      {"status": "filled", "candle": <row dict>}
      {"status": "cancelled"}
      {"status": "expired"}
      {"status": "resting", "candles_waited": <int>}
    """
    if df is None or len(df) < 1:
        return {"status": "resting", "candles_waited": pending["candles_waited"]}

    last = df.iloc[-1]

    # ---- 1. Fill check first (a real resting limit doesn't care whether
    # the setup is still technically valid the moment price reaches it) ----
    if check_fill_buy(pending, last):
        candle_dict = last.to_dict() if hasattr(last, "to_dict") else dict(last)
        return {"status": "filled", "candle": candle_dict}

    # ---- 2. Cancellation check: has the original setup invalidated? ----
    if not check_entry_signal_buy(df):
        return {"status": "cancelled"}

    # ---- 3. Expiry check: resting window exhausted? ----
    candles_waited = pending["candles_waited"] + 1
    if candles_waited >= max_candles:
        return {"status": "expired"}

    return {"status": "resting", "candles_waited": candles_waited}


# ============================================================================
# LIVE-WIRED functions below -- these actually call the broker/webhook and
# read/write state.json, unlike everything above (which is pure logic, no
# side effects).
# ============================================================================

import angelone_client as ac
from calendar_utils import is_eod_squareoff_time, is_within_buy_scan_window
from market_data import get_candles_with_cache
from indicators import compute_indicators
from webhook import send_to_webhook, webhook_confirmed_ok


def scan_for_new_buy_signal_live(state, leg_pairs, instruments, expiry, smart_api,
                                  prev_day, prev_day_cache, today_start, run_cache=None):
    """
    Live-wired scan for the buy engine. Reuses the SAME locked sell
    strikes (one CE, one PE) the sell engine already tracks -- no new
    token lookups beyond what the sell side already resolves each day.

    REMOVED (2026-09-04): the same-strike guard
    (_strike_has_open_sell_position()) that used to skip a strike here if
    the sell side had an open position on it. Pragnesh's explicit call:
    the buy side should fire independent of the sell side's state --
    low-cost long option, the same-strike overlap this guard protected
    against is no longer judged worth the missed signals. Every leg in
    leg_pairs is now scanned unconditionally on every call to this
    function (still subject to run-order gating in main.py and the
    single-slot pending_buy_signal limit, both UNCHANGED).

    Mirrors signal_engine.scan_for_new_signal()'s single-slot,
    stop-after-first-fire shape.

    FIX (2026-07-29, run-level dedup): accepts run_cache, passed through
    to get_candles_with_cache() -- see market_data.py's docstring.

    CHANGED (2026-09-04, ROC rework): the entry condition itself
    (is_fresh_crossover_signal_buy) and entry/SL/target computation
    (compute_pending_buy_signal) are now ROC-based -- see this module's
    top docstring and each function's own docstring for the full
    writeup. The scope of WHICH strikes get scanned (both sell strikes,
    ATM+100 CE and ATM-100 PE, no hedge legs) is UNCHANGED from the
    2026-07-29 scope-correction fix.

    NEW (2026-08-12, buy-side scan time window): a fresh crossover that
    would otherwise create a new pending_buy_signal is still blocked
    outside config.BUY_SCAN_WINDOWS (09:30-11:45, 13:30-14:45) -- see
    calendar_utils.is_within_buy_scan_window(). Squeeze-diagnostic
    logging above this check remains unconditional. A time-blocked leg
    does NOT return -- loop continues to the next leg this run, same
    reasoning as the sell-side squeeze gate.
    """
    for leg in leg_pairs:
        strike = leg["sell_strike"]
        option_type = leg["option_type"]

        token_info = ac.resolve_option_token(instruments, expiry, strike, option_type)
        if not token_info:
            continue

        candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start, run_cache)
        df = compute_indicators(candles)
        if df is None:
            continue

        # NEW (2026-07-30, squeeze diagnostics): kept unconditional --
        # see this module's top docstring for why this still runs even
        # though the buy entry no longer gates on it.
        _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(df.iloc[-1])
        ratio_str = f"{spread_atr_ratio:.2f}" if spread_atr_ratio is not None else "n/a"
        last = df.iloc[-1]
        roc_str = f"{last['roc']:.2f}" if pd.notna(last["roc"]) else "n/a"
        print(
            f"[buy scan] {option_type} {strike}: roc={roc_str} "
            f"close={last['close']:.2f} vwap={last['vwap']:.2f} "
            f"[squeeze diag: spread_pct={spread_pct:.3f}% spread/atr={ratio_str}]",
            file=sys.stderr,
        )

        if is_fresh_crossover_signal_buy(df):
            # NEW (2026-08-12, buy-side scan time window): checked here,
            # AFTER the squeeze diagnostic already printed above, but
            # BEFORE a new pending_buy_signal gets created. Does not
            # affect an already-resting pending_buy_signal or
            # already-open open_buy_position (managed elsewhere,
            # unconditionally, every run).
            if not is_within_buy_scan_window():
                print(
                    f"[buy scan] fresh ROC crossover on {option_type} {strike} outside "
                    f"allowed scan windows ({BUY_SCAN_WINDOWS}) -- not entering. "
                    "Continuing to next leg this run.",
                    file=sys.stderr,
                )
                continue

            trigger_candle = df.iloc[-1].to_dict()
            prev_candle = df.iloc[-2].to_dict()
            qty = token_info["lot_size"]
            leg_info = {**token_info, "strike": strike, "option_type": option_type}

            pending_signal = compute_pending_buy_signal(trigger_candle, leg_info, qty, prev_candle)
            if pending_signal is None:
                print(
                    f"[buy scan] ROC crossover fired on {option_type} {strike} but computed "
                    "risk <= 0 (prev candle low >= trigger close) -- skipping this signal, "
                    "continuing to next leg this run.",
                    file=sys.stderr,
                )
                continue

            state["pending_buy_signal"] = pending_signal
            p = state["pending_buy_signal"]
            print(f"PENDING BUY SIGNAL: BUY {p['buy_symbol']} resting limit @ {p['entry_limit']:.2f} "
                  f"(SL={p['sl_price']:.2f}, target={p['target_price']:.2f}) "
                  f"[roc={trigger_candle['roc']:.2f}] "
                  f"[squeeze diag: spread_pct={spread_pct:.3f}% spread/atr={ratio_str}]")
            return

    print("No buy entry signal this run.")


def manage_pending_buy_signal_live(state, instruments, expiry, smart_api,
                                    prev_day, prev_day_cache, today_start, run_cache=None):
    """
    Live-wired lifecycle for the resting pending buy signal. UNCHANGED in
    shape by the 2026-09-04 ROC rework: EOD discard check first, then
    fill, then cancellation (now via the ROC-based check_entry_signal_buy()),
    then expiry.

    FIX (2026-07-29, run-level dedup): accepts run_cache, passed through
    to get_candles_with_cache().
    """
    pending = state["pending_buy_signal"]

    if is_eod_squareoff_time():
        print(f"pending_buy_signal for {pending['buy_symbol']} discarded -- "
              "EOD square-off time reached while still unfilled.")
        state["pending_buy_signal"] = None
        return

    token_info = ac.resolve_option_token(instruments, expiry, pending["strike"], pending["option_type"])
    if not token_info:
        print("Could not resolve buy-leg token while pending_buy_signal is resting -- skipping this run.",
              file=sys.stderr)
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start, run_cache)
    df = compute_indicators(candles)
    if df is None:
        return

    last = df.iloc[-1]

    # ---- 1. Fill check first ----
    if check_fill_buy(pending, last):
        resp = send_to_webhook({
            "action": "ENTRY_BUY",
            "side": "BUY",
            "symbol": pending["buy_symbol"],
            "qty": pending["qty"],
            "entry_price": pending["entry_limit"],
            "sl_price": pending["sl_price"],
            "target_price": pending["target_price"],
            "time": pending["trigger_time"],
        })

        if webhook_confirmed_ok(resp):
            state["open_buy_position"] = {
                "symbol": pending["buy_symbol"],
                "token": pending["buy_token"],
                "strike": pending["strike"],
                "option_type": pending["option_type"],
                "qty": pending["qty"],
                "entry_price": pending["entry_limit"],
                "sl_price": pending["sl_price"],
                "target_price": pending["target_price"],
            }
            state["pending_buy_signal"] = None
            print(f"FILLED: BUY {pending['buy_symbol']} @ {pending['entry_limit']}, qty={pending['qty']}")
        else:
            print(
                "ENTRY_BUY webhook not confirmed -- leaving pending_buy_signal in "
                "state.json so the next run retries the fill.",
                file=sys.stderr,
            )
        return

    # ---- 2. Cancellation check: has the original setup invalidated? ----
    if not check_entry_signal_buy(df):
        print(f"pending_buy_signal for {pending['buy_symbol']} cancelled -- "
              "ROC/VWAP condition no longer holds.")
        state["pending_buy_signal"] = None
        return

    # ---- 3. Expiry check: resting window exhausted? ----
    pending["candles_waited"] += 1
    if pending["candles_waited"] >= BUY_PENDING_SIGNAL_MAX_CANDLES:
        print(f"pending_buy_signal for {pending['buy_symbol']} expired -- "
              f"unfilled after {BUY_PENDING_SIGNAL_MAX_CANDLES} candles.")
        state["pending_buy_signal"] = None
        return

    print(f"pending_buy_signal for {pending['buy_symbol']} still resting "
          f"({pending['candles_waited']}/{BUY_PENDING_SIGNAL_MAX_CANDLES} candles, "
          f"limit={pending['entry_limit']:.2f}, latest low={last['low']:.2f}).")


def manage_open_buy_position_live(state, instruments, expiry, smart_api,
                                   prev_day, prev_day_cache, today_start, run_cache=None):
    """
    Live-wired exit management for a filled buy position. UNCHANGED by
    the 2026-09-04 ROC rework -- SL/target were already fixed numbers
    locked at fill time (just computed differently now, see
    compute_pending_buy_signal()); this function only ever reads
    pos["sl_price"]/pos["target_price"], never recomputes them.

    Checks high/low touch (not close-only), and forces a flatten at EOD
    square-off since the buy side stays intraday. SL is BELOW entry,
    target is ABOVE -- SL touch = low <= sl_price, target touch = high
    >= target_price. On a same-candle double-touch, SL wins.

    FIX (2026-07-29, run-level dedup): accepts run_cache, passed through
    to get_candles_with_cache().
    """
    pos = state["open_buy_position"]

    token_info = ac.resolve_option_token(instruments, expiry, pos["strike"], pos["option_type"])
    if not token_info:
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start, run_cache)
    df = compute_indicators(candles)
    if df is None:
        return

    last = df.iloc[-1]

    if is_eod_squareoff_time():
        resp = send_to_webhook({
            "action": "EXIT_BUY",
            "reason": "EOD",
            "symbol": pos["symbol"],
            "qty": pos["qty"],
            "price": float(last["close"]),
            "entry_price": pos["entry_price"],
        })
        if webhook_confirmed_ok(resp):
            state["open_buy_position"] = None
            print(f"BUY position for {pos['symbol']} squared off at EOD @ {float(last['close']):.2f}")
        else:
            print(
                "EXIT_BUY (EOD) webhook not confirmed -- leaving open_buy_position "
                "in state.json so the next run retries.",
                file=sys.stderr,
            )
        return

    candle_high = float(last["high"])
    candle_low = float(last["low"])
    sl_price = pos["sl_price"]
    target_price = pos["target_price"]

    sl_hit = candle_low <= sl_price
    target_hit = candle_high >= target_price

    if sl_hit or target_hit:
        reason = "SL" if sl_hit else "TARGET"  # SL-wins tie-break on a same-candle double-touch
        exit_price = sl_price if sl_hit else target_price
        resp = send_to_webhook({
            "action": "EXIT_BUY",
            "reason": reason,
            "symbol": pos["symbol"],
            "qty": pos["qty"],
            "price": exit_price,
            "entry_price": pos["entry_price"],
        })
        if webhook_confirmed_ok(resp):
            state["open_buy_position"] = None
            print(f"BUY position closed: reason={reason} price={exit_price}")
        else:
            print(
                f"EXIT_BUY ({reason}) webhook not confirmed -- leaving open_buy_position "
                "in state.json so the next run retries.",
                file=sys.stderr,
            )
        return

    print(f"BUY position for {pos['symbol']} still open "
          f"(high={candle_high:.2f}, low={candle_low:.2f}, SL={sl_price:.2f}, target={target_price:.2f}).")
