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

from config import BUY_PENDING_SIGNAL_MAX_CANDLES, BUY_TARGET_RISK_REWARD
from calendar_utils import now_ist


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


def compute_pending_buy_signal(trigger_candle, buy_leg_info, qty):
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

    SL = min(trigger candle's LOW, trigger candle's VWAP).

    Target = entry + BUY_TARGET_RISK_REWARD * (entry - SL), i.e. a fixed
    1:3 risk:reward off entry_limit by default.
    """
    trigger_low = trigger_candle["low"]
    trigger_vwap = trigger_candle["vwap"]
    entry_limit = trigger_candle["ema5"]

    sl_price = min(trigger_low, trigger_vwap)

    risk = entry_limit - sl_price
    target_price = entry_limit + BUY_TARGET_RISK_REWARD * risk

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
            qty = leg_info.get("lot_size", leg_info.get("qty"))
            pending = compute_pending_buy_signal(trigger_candle, leg_info, qty)
            if log:
                print(
                    f"PENDING BUY SIGNAL: BUY {pending['buy_symbol']} resting limit @ "
                    f"{pending['entry_limit']:.2f} (SL={pending['sl_price']:.2f}, "
                    f"target={pending['target_price']:.2f})"
                )
            return pending
        elif log:
            last = df.iloc[-1]
            print(
                f"[buy scan] {leg_info.get('symbol', '?')}: ema5={last['ema5']:.2f} "
                f"ema25={last['ema25']:.2f} vwap={last['vwap']:.2f} "
                f"ema25<ema5={bool(last['ema25'] < last['ema5'])} "
                f"ema25<vwap={bool(last['ema25'] < last['vwap'])} "
                f"ema5>vwap={bool(last['ema5'] > last['vwap'])} -- no fresh cross.",
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
from calendar_utils import is_eod_squareoff_time
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
    Live-wired scan for the CE-side buy engine. Reuses the SAME locked CE
    strikes (sell_strike, hedge_strike) the sell engine already tracks --
    Pragnesh's call: no new token lookups or candle feeds, stays inside
    existing Angel One rate limits -- but skips any CE strike that
    currently has an open SELL position on it (Pragnesh's call: skip on
    same-strike overlap, both sides intraday).

    Mirrors signal_engine.scan_for_new_signal()'s single-slot,
    stop-after-first-fire shape.

    FIX (2026-07-29, run-level dedup): the "no new token lookups or
    candle feeds" claim in the docstring above was true for token
    lookups, but NOT for candle feeds -- this loop's "sell_strike" branch
    resolves to the exact same token signal_engine.scan_for_new_signal()
    already fetches earlier in the same run, and was calling
    get_candles_with_cache() again from scratch for it, with no memory
    of the earlier fetch. Confirmed via a 2026-07-28 live run log showing
    two separate getCandleData calls, seconds apart, for the same CE
    sell-strike token. Now accepts run_cache and passes it through --
    when scan_for_new_signal() (or priming) already fetched this token
    this run, this call becomes a free in-memory hit instead of a second
    real API call. See market_data.get_candles_with_cache()'s docstring
    for the full root-cause writeup.
    """
    for leg in leg_pairs:
        if leg["option_type"] != "CE":
            continue

        for strike_key in ("sell_strike", "hedge_strike"):
            strike = leg[strike_key]

            if _strike_has_open_sell_position(state, strike, "CE"):
                print(f"[buy scan] skipping CE {strike} -- sell side already has an open position on this strike.")
                continue

            token_info = ac.resolve_option_token(instruments, expiry, strike, "CE")
            if not token_info:
                continue

            candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start, run_cache)
            df = compute_indicators(candles)
            if df is None:
                continue

            if is_fresh_crossover_signal_buy(df):
                trigger_candle = df.iloc[-1].to_dict()
                qty = token_info["lot_size"]
                leg_info = {**token_info, "strike": strike, "option_type": "CE"}

                state["pending_buy_signal"] = compute_pending_buy_signal(trigger_candle, leg_info, qty)
                p = state["pending_buy_signal"]
                print(f"PENDING BUY SIGNAL: BUY {p['buy_symbol']} resting limit @ {p['entry_limit']:.2f} "
                      f"(SL={p['sl_price']:.2f}, target={p['target_price']:.2f})")
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
