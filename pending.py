"""
Pending-signal lifecycle: locking a resting SELL limit order off a
trigger candle, then on each subsequent run deciding whether it fills,
gets cancelled (setup invalidated), or expires (resting window
exhausted).
"""

import sys

import angelone_client as ac
from config import (
    PENDING_SIGNAL_MAX_CANDLES,
    ENTRY_LIMIT_DISCOUNT,
    LOW_PREMIUM_SL_THRESHOLD,
    LOW_PREMIUM_SL_MIN_PCT,
    TARGET_RISK_REWARD,
)
from calendar_utils import now_ist, is_eod_squareoff_time
from market_data import get_candles_with_cache
from indicators import compute_indicators
from signal_engine import check_entry_signal
from webhook import send_to_webhook, webhook_confirmed_ok


def compute_entry_price(trigger_candle):
    """
    FIX (2026-07-17, EMA5-lag root cause): this was the formula used
    before the pending-signal rework, then marked LEGACY when
    compute_pending_signal() switched to EMA5[N] * ENTRY_LIMIT_DISCOUNT
    as a proxy for candle N+1's open. That EMA5-based anchor turned out
    to be the wrong choice specifically for FRESH, STRONG crossovers --
    the exact condition this strategy's entry signal is designed to
    detect -- because EMA5 lags a fast-falling price by several points
    per candle (confirmed 2026-07-17: EMA5 dropped 103.02 -> 98.67
    across a single candle purely from ordinary EMA recursion, no stale
    prior-day data involved). The resulting entry_limit sat ABOVE the
    trigger candle's own high, making the resting SELL limit
    structurally unreachable on a real momentum move -- see the
    2026-07-17 case study (NIFTY21JUL2624100PE: EMA5-based entry_limit
    97.87 vs a next-candle high of only 95.15, while price kept falling
    to 77.95 over the next hour with the "signal" never able to fill).

    Reinstated as the ACTIVE formula for compute_pending_signal(): 40%
    pullback from the trigger candle's own low toward its own high.
    Anchored entirely to where price actually traded this candle, not a
    lagging multi-candle average, so it moves with price instead of
    behind it. Verified against the 2026-07-17 candles: this formula
    would have priced the entry at 92.57 (09:30 candle), comfortably
    reached by the very next candle's high of 95.15, vs. the EMA5-based
    97.87 which was never reached.

    NOTE (deliberately NOT changed in this pass): SL and target formulas
    below are UNCHANGED -- still computed off max(high, vwap) and fixed
    1:2 risk:reward, same as before. Revisiting SL/target (e.g. a
    swing-high trailing stop, wider risk:reward) is intentionally scoped
    OUT of this fix and left for a separate change, so this pass is
    isolated to "is the entry price reachable", not bundled with a
    change to how risk/reward is managed once filled.
    """
    low = trigger_candle["low"]
    high = trigger_candle["high"]
    return low + 0.40 * (high - low)


def round_to_half(price):
    """
    Rounds a price to the nearest Rs. 0.5 tick (e.g. 87.07 -> 87.0,
    95.78 -> 96.0, 69.66 -> 69.5). Applied to entry_limit FIRST, then
    sl_price/target_price are computed from that already-rounded
    entry_limit and rounded again themselves -- so all three numbers
    that actually go out in the ENTRY_SPREAD payload are clean 0.5
    multiples, and risk/reward is computed off the same rounded numbers
    that get sent to the webhook/Telegram, not off raw floats rounded
    independently afterward.

    FIX (2026-07-17): re-added after being silently dropped -- the
    2026-07-17 EMA5-lag fix (see compute_entry_price() above) was built
    on top of an older pending.py baseline that predated this rounding
    pass entirely, so the earlier 0.5-rounding fix never carried forward
    into that rewrite. Not a revert, just two changes landing on
    divergent versions of this file. Reapplied here on top of the now-
    active compute_entry_price() pullback formula, without touching
    that formula itself.
    """
    return round(price * 2) / 2


def compute_pending_signal(trigger_candle, sell_leg_info, hedge_leg_info, qty):
    """
    Locks a resting SELL limit order off the just-closed trigger candle N.

    entry_limit is FIXED for the whole resting window: computed once here
    from candle N, never recomputed on candles N+1..N+5.

    entry_limit uses compute_entry_price() -- a 40% pullback from the
    trigger candle's own low toward its own high (see that function's
    docstring for the 2026-07-17 EMA5-lag root-cause writeup).

    SL = max(trigger candle's high, trigger candle's VWAP). If entry_limit
    is under Rs.99, SL additionally floors at entry_limit * 1.10 -- this
    WIDENS the stop when it applies, it never tightens or replaces the
    high/VWAP comparison (cheap premiums are more prone to a tight
    absolute SL getting whipsawed).

    Target is fixed 1:2 risk:reward off entry_limit, using the resulting
    SL distance as risk.

    FIX (2026-07-17, rounding re-added): entry_limit/sl_price/target_price
    are rounded to the nearest 0.5 -- entry_limit FIRST (since the
    low-premium SL-floor threshold check and the risk/target math both
    key off it), then sl_price and target_price are computed from that
    already-rounded entry_limit and rounded again themselves at the end.
    Keeps entry/SL/target internally consistent: risk = sl_price -
    entry_limit uses the same rounded entry_limit that's actually sent
    to the webhook, rather than rounding all three independently after
    computing off raw floats (which could quietly drift the real RR
    away from TARGET_RISK_REWARD).

    NOTE: the LOW_PREMIUM_SL_THRESHOLD check runs against the ROUNDED
    entry_limit, not the raw one -- a raw value near Rs.99 could
    legitimately round across that threshold, and using the rounded
    value keeps the floor check consistent with what's actually locked.
    """
    trigger_high = trigger_candle["high"]
    trigger_vwap = trigger_candle["vwap"]
    entry_limit = round_to_half(compute_entry_price(trigger_candle))

    sl_price = max(trigger_high, trigger_vwap)
    if entry_limit < LOW_PREMIUM_SL_THRESHOLD:
        sl_price = max(sl_price, entry_limit * (1 + LOW_PREMIUM_SL_MIN_PCT))
    sl_price = round_to_half(sl_price)

    risk = sl_price - entry_limit
    target_price = round_to_half(entry_limit - TARGET_RISK_REWARD * risk)

    return {
        "sell_symbol": sell_leg_info["symbol"],
        "sell_token": sell_leg_info["token"],
        "sell_strike": sell_leg_info["strike"],
        "hedge_symbol": hedge_leg_info["symbol"],
        "hedge_token": hedge_leg_info["token"],
        "hedge_strike": hedge_leg_info["strike"],
        "option_type": sell_leg_info["option_type"],
        "qty": qty,
        "entry_limit": entry_limit,
        "sl_price": sl_price,
        "target_price": target_price,
        "trigger_time": now_ist().isoformat(),
        "candles_waited": 0,
    }


def manage_pending_signal(state, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    Checks whether the resting SELL limit order (pending_signal) should
    fill, get cancelled (setup invalidated), or expire (5-candle window
    exhausted) on this run's latest closed candle.

    Order matters: fill is checked FIRST, before cancellation/expiry -- a
    real resting limit order doesn't care whether check_entry_signal() is
    still true. If price already reached the limit this candle, it fills
    regardless of whether the EMA/VWAP setup broke down by this candle's
    close.

    FIX (2026-07-25, cross-day pending_signal survival bug): this
    function previously had NO end-of-day check at all -- only fill,
    cancellation (setup invalidated), and 5-candle expiry. open_position
    is always force-flattened at EOD_SQUAREOFF (see position.py's
    manage_spread_exit()/manage_legacy_single_leg_exit()), but a resting
    pending_signal had no equivalent: if it simply hadn't filled, hadn't
    been cancelled, and hadn't yet hit PENDING_SIGNAL_MAX_CANDLES by
    market close, it stayed in state.json completely unchanged and
    carried straight into the next trading day's state.json with no
    same-day-only marker on it whatsoever.

    Confirmed root cause, 2026-07-23/24 case study: a pending_signal
    created on 2026-07-23 under that day's ATM=23900 lock (SELL 23800 PE
    / HEDGE 23500 PE) never resolved by that day's 15:20 EOD square-off
    -- because pending_signal had no EOD check to resolve it. It sat
    untouched in state.json overnight, survived into 2026-07-24 (a day
    locked under a completely different ATM=23700), and was filled the
    next morning at 09:32 the moment a fresh candle's high finally
    touched its stale entry_limit -- a real (paper) trade placed on a
    strike/side that day's actual locked leg_pairs never included at
    all. See instrument.py's leg-pair validation (added in main.py, same
    incident) for the complementary guard that stops a stale signal from
    being FILLED even if one somehow still gets through; THIS fix stops
    it from ever surviving past its own trading day in the first place,
    which is the actual root cause.

    This check runs FIRST, ahead of even the fill check: a resting order
    this close to close has no time left to be managed properly if it
    fills, so it's discarded outright rather than filled, regardless of
    whether price happened to touch entry_limit this same candle.
    """
    pending = state["pending_signal"]

    if is_eod_squareoff_time():
        print(f"pending_signal for {pending['sell_symbol']} discarded -- "
              "EOD square-off time reached while still unfilled. A resting "
              "signal must not survive past its own trading day.")
        state["pending_signal"] = None
        return

    token_info = ac.resolve_option_token(instruments, expiry, pending["sell_strike"], pending["option_type"])
    if not token_info:
        print("Could not resolve sell-leg token while pending_signal is resting -- skipping this run.",
              file=sys.stderr)
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)
    if df is None:
        return

    last = df.iloc[-1]

    # ---- 1. Fill check first ----
    if last["high"] >= pending["entry_limit"]:
        hedge_token_info = ac.resolve_option_token(instruments, expiry, pending["hedge_strike"], pending["option_type"])
        if not hedge_token_info:
            print("Sell leg limit reached but hedge leg token could not be resolved -- "
                  "NOT filling, leaving pending_signal in place to retry next run.",
                  file=sys.stderr)
            return

        resp = send_to_webhook({
            "action": "ENTRY_SPREAD",
            "sell_side": "SELL",
            "sell_symbol": pending["sell_symbol"],
            "hedge_side": "BUY",
            "hedge_symbol": hedge_token_info["symbol"],
            "qty": pending["qty"],
            "entry_price": pending["entry_limit"],
            "hedge_entry_price": pending.get("hedge_entry_price"),
            "sl_price": pending["sl_price"],
            "target_price": pending["target_price"],
            "time": now_ist().isoformat(),
        })

        if webhook_confirmed_ok(resp):
            state["open_position"] = {
                "spread": True,
                "option_type": pending["option_type"],
                "sell_leg": {
                    "symbol": pending["sell_symbol"],
                    "token": pending["sell_token"],
                    "strike": pending["sell_strike"],
                },
                "hedge_leg": {
                    "symbol": hedge_token_info["symbol"],
                    "token": hedge_token_info["token"],
                    "strike": pending["hedge_strike"],
                },
                "qty": pending["qty"],
                "entry_price": pending["entry_limit"],
                "hedge_entry_price": pending.get("hedge_entry_price"),
                "sl_price": pending["sl_price"],
                "target_price": pending["target_price"],
                "entry_time": now_ist().isoformat(),
            }
            state["pending_signal"] = None
            print(f"FILLED: SELL {pending['sell_symbol']} @ {pending['entry_limit']} "
                  f"+ hedge BUY {hedge_token_info['symbol']}, qty={pending['qty']}")
        else:
            print(
                "ENTRY_SPREAD webhook not confirmed -- leaving pending_signal in "
                "state.json so the next run retries the fill.",
                file=sys.stderr,
            )
        return

    # ---- 2. Cancellation check: has the original setup invalidated? ----
    if not check_entry_signal(df):
        print(f"pending_signal for {pending['sell_symbol']} cancelled -- "
              "EMA5/VWAP crossover condition no longer holds.")
        state["pending_signal"] = None
        return

    # ---- 3. Expiry check: 5-candle resting window exhausted? ----
    pending["candles_waited"] += 1
    if pending["candles_waited"] >= PENDING_SIGNAL_MAX_CANDLES:
        print(f"pending_signal for {pending['sell_symbol']} expired -- "
              f"unfilled after {PENDING_SIGNAL_MAX_CANDLES} candles.")
        state["pending_signal"] = None
        return

    print(f"pending_signal for {pending['sell_symbol']} still resting "
          f"({pending['candles_waited']}/{PENDING_SIGNAL_MAX_CANDLES} candles, "
          f"limit={pending['entry_limit']:.2f}, latest high={last['high']:.2f}).")
