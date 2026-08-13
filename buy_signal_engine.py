"""
BUY-side signal engine -- NEW, STANDALONE MODULE.

This is the mirror-image counterpart to signal_engine.py/pending.py's
SELL-side crossover logic, built for the setup Pragnesh observed working
well as a follow-up when the sell side's SL hits: a bullish EMA5/VWAP
crossover with EMA25 underneath both, on the SAME option premium series
(i.e. going long the option itself when the reversal fires, not selling
it).

Deliberately NOT wired into main.py / state.json's existing
"pending_signal" / "open_position" keys yet -- this module is meant to
be built and validated on its own first (e.g. against candle_history/
CSVs) before being integrated into the live run loop. When that
integration happens, this will plug in alongside the sell side using its
own state keys (see the docstring on manage_pending_buy_signal() below
for the suggested keys), not replace or share the sell side's.

Entry conditions (as specified):
  - EMA25 is below BOTH EMA5 and VWAP        (ema25 < ema5 and ema25 < vwap)
  - EMA5 crosses VWAP from below             (prev candle: ema5 <= vwap,
                                               this candle: ema5 > vwap)
  -> On that trigger candle N, rest a BUY limit at EMA5[N], to be filled
     on a later candle whose LOW touches that level (a pullback entry,
     the buy-side mirror of the sell engine resting a SELL limit for a
     later HIGH to reach it).

SL = min(trigger candle's LOW, trigger candle's VWAP) -- the buy-side
mirror of the sell engine's SL = max(high, vwap). Taking the lower of
the two gives the stop more room below entry, same reasoning as the
sell side taking the higher of the two to give its stop more room above
entry.

Target = entry + BUY_TARGET_RISK_REWARD * (entry - SL)  (fixed 1:3 by
default, see config.BUY_TARGET_RISK_REWARD).
"""

import sys

from config import (
    BUY_PENDING_SIGNAL_MAX_CANDLES, BUY_TARGET_RISK_REWARD,
    BUY_LOW_PREMIUM_SL_THRESHOLD, BUY_LOW_PREMIUM_SL_MIN_PCT,
    BUY_SCAN_WINDOWS,
)
from calendar_utils import now_ist
from pending import round_to_half
from indicators import compute_squeeze_metrics


def _condition_holds_buy(row):
    """
    Shared EMA5/EMA25/VWAP condition check for a single candle row, buy
    side. Mirrors signal_engine.py's _condition_holds() but inverted:
    EMA25 must sit below BOTH EMA5 and VWAP, and EMA5 must currently be
    ABOVE VWAP (the "reclaimed VWAP" bullish state).
    """
    return (
        row["ema5"] > row["vwap"]
        and row["ema25"] < row["ema5"]
        and row["ema25"] < row["vwap"]
    )


def check_entry_signal_buy(df):
    """
    STATE check (not transition-based): is the buy-side condition true on
    the latest candle? Mirrors signal_engine.check_entry_signal() -- used
    to decide whether a resting pending buy signal's setup has been
    INVALIDATED, not to scan for brand-new entries (see
    is_fresh_crossover_signal_buy() for that).
    """
    if df is None or len(df) < 2:
        return False
    return _condition_holds_buy(df.iloc[-1])


def _pre_cross_ordering_holds(row):
    """
    The specific ordering required on the candle BEFORE the cross:
    EMA25 < EMA5 < VWAP -- i.e. the setup already forming (EMA25 lowest,
    EMA5 in the middle, VWAP still on top, not yet reclaimed). Fixes a
    gap in the original fresh-crossover check, which only asked "was the
    buy condition false last candle" -- true for this ordering, but also
    true for unrelated states (e.g. EMA25 above EMA5 entirely, or EMA5
    far below both). Only THIS specific pre-cross ordering counts as a
    genuine setup.
    """
    return row["ema25"] < row["ema5"] < row["vwap"]


def is_fresh_crossover_signal_buy(df):
    """
    TRANSITION check: did EMA5 just cross above VWAP THIS candle, with
    the setup (EMA25 < EMA5 < VWAP) already correctly formed on the
    PREVIOUS candle -- as opposed to merely "the buy condition wasn't
    true last candle" for some unrelated reason. Mirrors
    signal_engine.is_fresh_crossover_signal()'s reasoning exactly --
    without this, a resting signal that expired unfilled could
    immediately re-fire on the very next scan purely because the state
    was still held, with no actual new crossover.

    Requires:
      - PREVIOUS candle: EMA25 < EMA5 < VWAP (setup forming, not yet crossed)
      - CURRENT candle: full buy condition holds (EMA5 has now crossed
        above VWAP, EMA25 still below both)
    """
    if df is None or len(df) < 2:
        return False
    current = _condition_holds_buy(df.iloc[-1])
    previous_setup_forming = _pre_cross_ordering_holds(df.iloc[-2])
    return current and previous_setup_forming


def compute_pending_buy_signal(trigger_candle, buy_leg_info, qty, prev_candle=None):
    """
    Locks a resting BUY limit order off the just-closed trigger candle N.

    entry_limit = trigger candle's EMA5, taken literally (no discount
    applied) -- as specified: "take entry in next candle at ema5". This
    is a pullback entry: the limit rests at EMA5[N] and fills on a LATER
    candle whose LOW comes down to touch it (see check_fill_buy()),
    rather than chasing the trigger candle's own close/high.

    entry_limit is FIXED for the whole resting window, computed once here
    from candle N only -- never recomputed on candles N+1..N+BUY_
    PENDING_SIGNAL_MAX_CANDLES.

    SL = min(trigger candle's LOW, trigger candle's VWAP), additionally
    floored against candle N-1's LOW when available (see FIX below).

    Target = entry + BUY_TARGET_RISK_REWARD * (entry - SL), i.e. a fixed
    1:3 risk:reward off entry_limit by default.

    FIX (2026-07-29, SL-too-tight): the original two-way min(low, vwap)
    was landing too close to entry on tight-range trigger candles,
    leaving almost no room before an ordinary pullback stopped the
    position out. Pragnesh's call: additionally floor SL against the
    LOW of the candle immediately BEFORE the trigger candle (candle
    N-1) when one is available -- take the WIDEST (i.e. lowest) of
    trigger_low, trigger_vwap, and prev_candle_low, never narrower than
    the original two-way min. prev_candle is optional (None right after
    market open, when there's no N-1 candle yet) -- SL falls back to the
    original two-way min in that case.

    FIX (2026-07-29, unrounded prices): entry_limit/sl_price/target_price
    are now rounded to the nearest 0.5 tick via pending.round_to_half()
    -- mirrors the sell side's 2026-07-17 rounding fix
    (pending.compute_pending_signal()). Previously these were raw floats
    (entry_limit was a raw EMA5 value with no rounding at all).
    Rounded in the same order as the sell side: entry_limit first, then
    sl_price, then risk/target_price computed off the already-rounded
    values -- so the real risk:reward sent to the webhook matches
    BUY_TARGET_RISK_REWARD exactly rather than drifting off raw floats.

    NEW (2026-07-31, low-premium SL floor): mirrors the sell side's
    LOW_PREMIUM_SL_THRESHOLD/MIN_PCT treatment (pending.compute_pending_signal()),
    inverted for direction -- buy's SL sits BELOW entry, so the floor
    widens it DOWNWARD (min, not max) when entry_limit is under
    BUY_LOW_PREMIUM_SL_THRESHOLD (Rs.99, same as sell). Same widen-only
    contract: this can only push SL further from entry, never closer than
    what min(low, vwap, prev_low) already computed. Checked against the
    ROUNDED entry_limit, same reasoning as the sell side's NOTE on this.
    All 7 paper buy trades to date hit SL, several on sub-Rs.99 premiums
    (e.g. Rs.3.95, Rs.17.95, Rs.17.07) with no floor at all on this side
    before this fix -- real motivation for adding it now rather than
    leaving it deferred.
    """
    trigger_low = trigger_candle["low"]
    trigger_vwap = trigger_candle["vwap"]
    entry_limit = round_to_half(trigger_candle["ema5"])

    sl_price = min(trigger_low, trigger_vwap)
    if prev_candle is not None:
        sl_price = min(sl_price, prev_candle["low"])
    if entry_limit < BUY_LOW_PREMIUM_SL_THRESHOLD:
        sl_price = min(sl_price, entry_limit * (1 - BUY_LOW_PREMIUM_SL_MIN_PCT))
    sl_price = round_to_half(sl_price)

    risk = entry_limit - sl_price
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
    come down far enough to touch the resting BUY limit? Mirrors the
    sell engine's fill check (`last["high"] >= pending["entry_limit"]`),
    inverted for a buy limit resting BELOW where price was at trigger
    time: `last["low"] <= pending["entry_limit"]`.
    """
    return last_candle["low"] <= pending["entry_limit"]


def scan_for_new_buy_signal(candles_by_symbol, compute_indicators_fn, log=True):
    """
    Standalone scan across a dict of {leg_info: candles} for a fresh
    buy-side crossover. This mirrors signal_engine.scan_for_new_signal()'s
    shape but is deliberately decoupled from state.json / smart_api / the
    live run loop -- it takes already-fetched candles and an indicator
    function, so it can be called equally from:
      (a) a future integration into main.py (passing real candles +
          indicators.compute_indicators), or
      (b) a standalone backtest/validation script (see
          test_buy_signal_engine.py) against historical candle_history/
          CSVs, without needing a broker session at all.

    candles_by_symbol: dict of {leg_info_dict: list_of_candle_dicts},
    where leg_info_dict has at least "symbol", "token", "strike",
    "option_type", and "lot_size" (qty).

    Returns the first pending buy signal found (dict, see
    compute_pending_buy_signal()), or None if nothing fired this scan.
    Only ever returns at most one signal per call, same single-slot
    reasoning as the sell engine (see its docstring on why it stops after
    the first leg that fires).
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
            if log:
                # NEW (2026-07-30, squeeze diagnostics): log spread_pct /
                # spread-to-ATR on the actual trigger candle for every
                # fired signal too, not just no-fire scans -- this is the
                # data we need later to check "did signals that got
                # stopped out disproportionately have a tight spread_pct
                # or spread_atr_ratio here". No gating yet, see
                # indicators.compute_squeeze_metrics() docstring.
                _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(df.iloc[-1])
                ratio_str = f"{spread_atr_ratio:.2f}" if spread_atr_ratio is not None else "n/a"
                print(
                    f"PENDING BUY SIGNAL: BUY {pending['buy_symbol']} resting limit @ "
                    f"{pending['entry_limit']:.2f} (SL={pending['sl_price']:.2f}, "
                    f"target={pending['target_price']:.2f}) "
                    f"[squeeze diag: spread_pct={spread_pct:.3f}% spread/atr={ratio_str}]"
                )
            return pending
        elif log:
            last = df.iloc[-1]
            _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(last)
            ratio_str = f"{spread_atr_ratio:.2f}" if spread_atr_ratio is not None else "n/a"
            print(
                f"[buy scan] {leg_info.get('symbol', '?')}: ema5={last['ema5']:.2f} "
                f"ema25={last['ema25']:.2f} vwap={last['vwap']:.2f} "
                f"ema25<ema5={bool(last['ema25'] < last['ema5'])} "
                f"ema25<vwap={bool(last['ema25'] < last['vwap'])} "
                f"ema5>vwap={bool(last['ema5'] > last['vwap'])} "
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
    Mirrors pending.manage_pending_signal()'s three-step order (fill
    checked first, then cancellation, then expiry), but returns a result
    dict instead of mutating state.json directly -- this keeps the
    module usable standalone (backtest) as well as from a future live
    integration, which would be responsible for translating the result
    into its own state keys (e.g. state["pending_buy_signal"] /
    state["open_buy_position"], kept separate from the sell side's
    "pending_signal" / "open_position" so the two setups never collide).

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
# side effects). NOT yet called from main.py -- see main.py wiring
# discussion for the open run-order question before these get hooked in.
# ============================================================================

import angelone_client as ac
from calendar_utils import is_eod_squareoff_time, is_within_buy_scan_window
from market_data import get_candles_with_cache
from indicators import compute_indicators
from webhook import send_to_webhook, webhook_confirmed_ok


def _strike_has_open_sell_position(state, strike, option_type):
    """
    Pragnesh's call: both sides stay intraday, but a buy signal is
    skipped if the SELL side already has an open position on that exact
    strike. Checks state["open_position"] in both the current spread
    format (sell_leg + hedge_leg) and the legacy single-leg format (see
    position.py's manage_legacy_single_leg_exit() docstring for why that
    legacy shape still needs handling).
    """
    pos = state.get("open_position")
    if pos is None:
        return False

    if pos.get("spread"):
        for leg_key in ("sell_leg", "hedge_leg"):
            leg = pos.get(leg_key)
            if leg and leg.get("strike") == strike and pos.get("option_type") == option_type:
                return True
        return False

    # Legacy single-leg shape
    return pos.get("strike") == strike and pos.get("option_type") == option_type


def scan_for_new_buy_signal_live(state, leg_pairs, instruments, expiry, smart_api,
                                  prev_day, prev_day_cache, today_start, run_cache=None):
    """
    Live-wired scan for the buy engine. Reuses the SAME locked sell
    strikes (one CE, one PE) the sell engine already tracks -- no new
    token lookups beyond what the sell side already resolves each day --
    but skips any strike that currently has an open SELL position on it
    (Pragnesh's call: skip on same-strike overlap, both sides intraday).

    Mirrors signal_engine.scan_for_new_signal()'s single-slot,
    stop-after-first-fire shape.

    FIX (2026-07-29, run-level dedup): get_candles_with_cache() only ever
    de-duplicated the PREVIOUS day's candles, and only across separate
    runs -- nothing stopped this loop from re-fetching TODAY's candles
    for a token the sell engine had already fetched moments earlier in
    the same run. Confirmed via a 2026-07-28 live run log showing two
    separate getCandleData calls, seconds apart, for the same sell-strike
    token. Now accepts run_cache and passes it through -- when
    scan_for_new_signal() (or priming) already fetched this token this
    run, this call becomes a free in-memory hit instead of a second real
    API call. See market_data.get_candles_with_cache()'s docstring for
    the full root-cause writeup.

    FIX (2026-07-29, hedge-leg + CE-only scope correction): previously
    this looped BOTH sell_strike and hedge_strike as independent buy
    crossover candidates -- taking real buy trades on the hedge strike
    too, which was never intended (BUY_ENGINE_INTEGRATION.md section 3
    only ever describes the SELL strike as what the buy engine watches)
    -- and was hardcoded to CE only. Pragnesh's call: limit buy scanning
    to exactly the two SELL strikes (ATM+100 CE and ATM-100 PE),
    dropping both hedge strikes from buy-signal scanning entirely, and
    extending the mirror-image bullish-crossover check to the PE sell
    strike too (no longer CE-only). The crossover/entry/SL/target math
    itself (is_fresh_crossover_signal_buy, compute_pending_buy_signal)
    was always option-type-agnostic -- it operates on whatever candle
    series it's given -- so this is purely a scope change in which
    strike/option_type this loop resolves and fetches, not a change to
    any underlying signal logic.

    NEW (2026-08-12, buy-side scan time window): a fresh crossover that
    would otherwise create a new pending_buy_signal is now blocked
    outside config.BUY_SCAN_WINDOWS (09:30-11:45, 13:30-14:45) -- see
    calendar_utils.is_within_buy_scan_window(). Squeeze-diagnostic
    logging above this check remains unconditional; this gate is
    time-of-day only, buy side stays squeeze-free by design. A
    time-blocked leg does NOT return -- loop continues to the next leg
    this run, same reasoning as the sell-side squeeze gate.
    """
    for leg in leg_pairs:
        strike = leg["sell_strike"]
        option_type = leg["option_type"]

        if _strike_has_open_sell_position(state, strike, option_type):
            print(f"[buy scan] skipping {option_type} {strike} -- sell side already has an open position on this strike.")
            continue

        token_info = ac.resolve_option_token(instruments, expiry, strike, option_type)
        if not token_info:
            continue

        candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start, run_cache)
        df = compute_indicators(candles)
        if df is None:
            continue

        # NEW (2026-07-30, squeeze diagnostics): log spread_pct /
        # spread-to-ATR for this strike's latest candle on every live
        # run, fire or no-fire -- this loop runs against real paper-mode
        # data, so it's the actual source for calibrating squeeze
        # thresholds later. No gating yet, see
        # indicators.compute_squeeze_metrics() docstring.
        _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(df.iloc[-1])
        ratio_str = f"{spread_atr_ratio:.2f}" if spread_atr_ratio is not None else "n/a"
        print(
            f"[buy scan] {option_type} {strike}: "
            f"[squeeze diag: spread_pct={spread_pct:.3f}% spread/atr={ratio_str}]",
            file=sys.stderr,
        )

        if is_fresh_crossover_signal_buy(df):
            # NEW (2026-08-12, buy-side scan time window): checked here,
            # AFTER the squeeze diagnostic already printed above (so
            # diagnostics stay unconditional), but BEFORE a new
            # pending_buy_signal gets created. Pragnesh's call: buy side
            # stays squeeze-free by design -- this is a time-of-day gate,
            # not a squeeze gate. Does not affect an already-resting
            # pending_buy_signal or already-open open_buy_position (those
            # are managed elsewhere, unconditionally, every run -- see
            # calendar_utils.is_within_buy_scan_window()'s docstring).
            if not is_within_buy_scan_window():
                print(
                    f"[buy scan] fresh crossover on {option_type} {strike} outside "
                    f"allowed scan windows ({BUY_SCAN_WINDOWS}) -- not entering. "
                    "Continuing to next leg this run.",
                    file=sys.stderr,
                )
                continue

            trigger_candle = df.iloc[-1].to_dict()
            prev_candle = df.iloc[-2].to_dict()
            qty = token_info["lot_size"]
            leg_info = {**token_info, "strike": strike, "option_type": option_type}

            state["pending_buy_signal"] = compute_pending_buy_signal(trigger_candle, leg_info, qty, prev_candle)
            p = state["pending_buy_signal"]
            print(f"PENDING BUY SIGNAL: BUY {p['buy_symbol']} resting limit @ {p['entry_limit']:.2f} "
                  f"(SL={p['sl_price']:.2f}, target={p['target_price']:.2f}) "
                  f"[squeeze diag: spread_pct={spread_pct:.3f}% spread/atr={ratio_str}]")
            return

    print("No buy entry signal this run.")


def manage_pending_buy_signal_live(state, instruments, expiry, smart_api,
                                    prev_day, prev_day_cache, today_start, run_cache=None):
    """
    Live-wired lifecycle for the resting pending buy signal. Mirrors
    pending.manage_pending_signal()'s live wiring exactly: EOD discard
    check first, then fill (before cancellation/expiry -- a real resting
    limit doesn't care whether the setup is still technically valid the
    moment price reaches it), then cancellation, then expiry.

    FIX (2026-07-29, run-level dedup): now accepts run_cache and passes
    it through to get_candles_with_cache() -- see that function's
    docstring in market_data.py for the full root-cause writeup.
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
              "EMA5/VWAP crossover condition no longer holds.")
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
    Live-wired exit management for a filled buy position. Checks
    high/low touch (not close-only -- mirrors position.manage_spread_exit()'s
    intrabar reasoning: a real resting SL/target order fires the moment
    price TOUCHES it, not only if the candle closes past it), and forces
    a flatten at EOD square-off since the buy side stays intraday too
    (Pragnesh's call: not positional).

    SL is BELOW entry for a long option, target is ABOVE -- opposite
    orientation from the sell side's bracket, so SL touch = low <=
    sl_price, target touch = high >= target_price. On a same-candle
    double-touch (rare, wide-range candle), SL wins -- same
    protect-capital-first tie-break as the sell side's bracket.

    FIX (2026-07-29, run-level dedup): now accepts run_cache and passes
    it through to get_candles_with_cache() -- see that function's
    docstring in market_data.py for the full root-cause writeup. This
    function is called unconditionally at the top of every run when a
    buy position is open (before the sell side runs), so it's often the
    FIRST thing to populate run_cache for a given token this run, same
    role priming plays on the lock run.
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
