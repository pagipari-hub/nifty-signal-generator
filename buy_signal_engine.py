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
  -> On that trigger candle N, rest a BUY limit at entry_limit (see
     below), to be filled on a later candle whose LOW touches that
     level (a pullback entry, the buy-side mirror of the sell engine
     resting a SELL limit for a later HIGH to reach it).

FIX (2026-07-27, entry/SL/target rework):

Entry: entry_limit = pending.compute_entry_price(trigger_candle) -- the
SAME 40% pullback formula used on the sell side (trigger candle's own
low + 0.40 * (high - low)), replacing the earlier EMA5-anchored entry.
EMA5 lags a fast-moving price the same way on the buy side as it did on
the sell side (see pending.compute_entry_price()'s docstring for the
original root-cause writeup) -- the formula itself is direction-
agnostic (just "40% up the trigger candle's own range from its low"),
and check_fill_buy()'s `low <= entry_limit` already handles a pullback
correctly in the buy direction.

SL = LOW of the candle immediately BEFORE the trigger/setup candle
(candle N-1 -- the same candle _pre_cross_ordering_holds() already
checks for the setup-forming ordering), not the candle before the
fill. Fixed at signal-detection time along with entry_limit, never
recomputed while the signal rests -- anchoring SL to whatever candle
happens to fill the resting limit (which can be several candles after
N) would place the stop unpredictably close to entry.

Target = entry + BUY_TARGET_RISK_REWARD * (entry - SL), fixed 1:2
risk:reward (see config.BUY_TARGET_RISK_REWARD).

Strikes watched: ONLY the two locked SELL-strikes (CE sell_strike +
PE sell_strike) -- the hedge strikes are never scanned for a buy
signal. FIX (2026-07-27): scan_for_new_buy_signal_live() previously
looped over BOTH sell_strike and hedge_strike for the CE leg only,
which meant a buy trade could fire on a hedge strike (never intended --
the hedge leg is purely a risk cap for the sell side's spread, not a
tradeable buy candidate) and the PE sell-strike was never watched at
all. Now both sell-strikes (CE and PE) are watched, and neither hedge
strike is.
"""

import sys

from config import BUY_PENDING_SIGNAL_MAX_CANDLES, BUY_TARGET_RISK_REWARD
from calendar_utils import now_ist
from pending import compute_entry_price, round_to_half


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


def compute_pending_buy_signal(trigger_candle, prev_candle_low, buy_leg_info, qty):
    """
    Locks a resting BUY limit order off the just-closed trigger candle N.

    entry_limit = pending.compute_entry_price(trigger_candle) -- the
    same 40% pullback (trigger candle's own low + 0.40 * (high - low))
    used on the sell side, rounded to the nearest 0.5 tick. This is a
    pullback entry: the limit rests at entry_limit and fills on a LATER
    candle whose LOW comes down to touch it (see check_fill_buy()),
    rather than chasing the trigger candle's own close/high.

    entry_limit is FIXED for the whole resting window, computed once here
    from candle N only -- never recomputed on candles N+1..N+BUY_
    PENDING_SIGNAL_MAX_CANDLES.

    SL = LOW of candle N-1 (the candle immediately before the trigger/
    setup candle), passed in as prev_candle_low -- fixed here at signal-
    detection time, same as entry_limit, never recomputed against
    whichever candle actually fills the resting limit.

    Target = entry + BUY_TARGET_RISK_REWARD * (entry - SL), i.e. a fixed
    1:2 risk:reward off entry_limit by default.

    NOTE: on a rare gap-up setup candle, prev_candle_low could sit above
    entry_limit, giving a non-positive "risk". This function does not
    guard against that case -- it's flagged here rather than silently
    handled, since deciding what to do (skip the signal entirely? floor
    the SL some other way?) hasn't been agreed yet.
    """
    entry_limit = round_to_half(compute_entry_price(trigger_candle))
    sl_price = round_to_half(prev_candle_low)

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
            prev_candle_low = df.iloc[-2]["low"]
            qty = leg_info.get("lot_size", leg_info.get("qty"))
            pending = compute_pending_buy_signal(trigger_candle, prev_candle_low, leg_info, qty)
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
                                  prev_day, prev_day_cache, today_start):
    """
    Live-wired scan for the buy engine. Watches ONLY the two locked
    SELL-strikes -- CE sell_strike and PE sell_strike -- reusing the
    same locked strikes the sell engine already tracks (Pragnesh's
    call: no new token lookups or candle feeds, stays inside existing
    Angel One rate limits) -- but skips a strike if the SELL side
    currently has an open position on it (Pragnesh's call: skip on
    same-strike overlap, both sides intraday).

    FIX (2026-07-27, hedge-strike bug): this previously looped over
    BOTH sell_strike and hedge_strike for the CE leg only, which meant
    a buy trade could fire on a hedge strike -- never intended, since
    the hedge leg exists purely as a risk cap for the sell side's
    spread, not as a tradeable buy candidate -- while the PE
    sell-strike was never watched at all. Now only sell_strike is
    scanned, across BOTH option types (CE and PE) -- two strikes total,
    no hedge strikes.

    Mirrors signal_engine.scan_for_new_signal()'s single-slot,
    stop-after-first-fire shape.
    """
    for leg in leg_pairs:
        option_type = leg["option_type"]
        strike = leg["sell_strike"]

        if _strike_has_open_sell_position(state, strike, option_type):
            print(f"[buy scan] skipping {option_type} {strike} -- sell side already has an open position on this strike.")
            continue

        token_info = ac.resolve_option_token(instruments, expiry, strike, option_type)
        if not token_info:
            continue

        candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
        df = compute_indicators(candles)
        if df is None:
            continue

        if is_fresh_crossover_signal_buy(df):
            trigger_candle = df.iloc[-1].to_dict()
            prev_candle_low = df.iloc[-2]["low"]
            qty = token_info["lot_size"]
            leg_info = {**token_info, "strike": strike, "option_type": option_type}

            state["pending_buy_signal"] = compute_pending_buy_signal(trigger_candle, prev_candle_low, leg_info, qty)
            p = state["pending_buy_signal"]
            print(f"PENDING BUY SIGNAL: BUY {p['buy_symbol']} resting limit @ {p['entry_limit']:.2f} "
                  f"(SL={p['sl_price']:.2f}, target={p['target_price']:.2f})")
            return

    print("No buy entry signal this run.")


def manage_pending_buy_signal_live(state, instruments, expiry, smart_api,
                                    prev_day, prev_day_cache, today_start):
    """
    Live-wired lifecycle for the resting pending buy signal. Mirrors
    pending.manage_pending_signal()'s live wiring exactly: EOD discard
    check first, then fill (before cancellation/expiry -- a real resting
    limit doesn't care whether the setup is still technically valid the
    moment price reaches it), then cancellation, then expiry.
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

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
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
                                   prev_day, prev_day_cache, today_start):
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
    """
    pos = state["open_buy_position"]

    token_info = ac.resolve_option_token(instruments, expiry, pos["strike"], pos["option_type"])
    if not token_info:
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
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
