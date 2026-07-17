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
from calendar_utils import now_ist
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


def compute_pending_signal(trigger_candle, sell_leg_info, hedge_leg_info, qty):
    """
    Locks a resting SELL limit order off the just-closed trigger candle N.

    entry_limit is FIXED for the whole resting window: computed once here
    from candle N, never recomputed on candles N+1..N+5.

    FIX (2026-07-17): entry_limit now uses compute_entry_price() -- a 40%
    pullback from the trigger candle's own low toward its own high --
    instead of the previous EMA5[N] * ENTRY_LIMIT_DISCOUNT proxy. See
    compute_entry_price()'s docstring above for the full root-cause
    writeup (EMA5 lags too far behind price on a fresh, strong crossover,
    making the old entry_limit structurally unreachable). ENTRY_LIMIT_
    DISCOUNT / trigger_candle["ema5"] are no longer used here as a
    result -- left in config.py / the trigger_candle dict for now since
    other code may still reference them, but no longer part of this
    calculation.

    SL = max(trigger candle's high, trigger candle's VWAP). If entry_limit
    is under Rs.99, SL additionally floors at entry_limit * 1.10 -- this
    WIDENS the stop when it applies, it never tightens or replaces the
    high/VWAP comparison (cheap premiums are more prone to a tight
    absolute SL getting whipsawed).

    Target is fixed 1:2 risk:reward off entry_limit, using the resulting
    SL distance as risk. UNCHANGED in this pass -- see compute_entry_price()
    docstring note above.
    """
    trigger_high = trigger_candle["high"]
    trigger_vwap = trigger_candle["vwap"]
    entry_limit = compute_entry_price(trigger_candle)

    sl_price = max(trigger_high, trigger_vwap)
    if entry_limit < LOW_PREMIUM_SL_THRESHOLD:
        sl_price = max(sl_price, entry_limit * (1 + LOW_PREMIUM_SL_MIN_PCT))

    risk = sl_price - entry_limit
    target_price = entry_limit - TARGET_RISK_REWARD * risk

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
    """
    pending = state["pending_signal"]

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
