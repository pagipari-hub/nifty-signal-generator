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


def round_to_half(price):
    """
    Rounds a price to the nearest Rs. 0.5 tick (e.g. 95.42 -> 95.5,
    104.96 -> 105.0, 76.34 -> 76.5). Applied to entry_limit FIRST, then
    sl_price/target_price are computed from that already-rounded
    entry_limit and rounded again themselves -- so all three numbers
    that actually go out in the ENTRY_SPREAD payload are clean 0.5
    multiples, and the risk/reward relationship between them is computed
    off the same rounded numbers a human (or the broker) would actually
    see and act on, not off raw floats that get rounded independently
    afterward.

    NOTE: this rounds to the nearest 0.5, which is a DIFFERENT (coarser)
    grid than webhook.py's round_up_to_tick() (nearest 0.05, rounding
    UP, applied only at the moment of real order placement in LIVE_MODE).
    That's intentional -- this function produces the "clean" display/
    strategy-level price that gets locked into pending_signal and shown
    on Telegram; round_up_to_tick() does a SECOND, broker-tick-aligned
    rounding pass on top of whatever comes out of here, right before the
    real Shoonya order is placed. The two don't conflict: 0.5 is always
    already a multiple of 0.05, so round_up_to_tick() is a no-op on an
    already-round_to_half()'d price in practice, it's just defensively
    still applied since it's the one place with the real tick-size
    contract for the broker.
    """
    return round(price * 2) / 2


def compute_entry_price(trigger_candle):
    """
    LEGACY -- no longer called by main(). Retained only because the
    currently-open single-leg position in state.json (opened before the
    pending-signal rework) was priced with this formula; keeping it here
    for reference/audit, not for reuse. New entries go through
    compute_pending_signal() instead.
    """
    low = trigger_candle["low"]
    high = trigger_candle["high"]
    return low + 0.40 * (high - low)


def compute_pending_signal(trigger_candle, sell_leg_info, hedge_leg_info, qty):
    """
    Locks a resting SELL limit order off the just-closed trigger candle N.
    EMA5[N] stands in as the proxy for candle N+1's open (the real open
    isn't known yet at candle-N-close time -- only 5-min OHLC is available,
    not tick data). The limit is set at a discount to that proxy, so in
    practice it's a floor that the very next candle's high almost always
    clears -- the 5-candle window exists as a safety margin for a single
    illiquid/gappy print, not because a deep pullback is expected.

    entry_limit is FIXED for the whole resting window: computed once here
    from candle N, never recomputed on candles N+1..N+5.

    SL = max(trigger candle's high, trigger candle's VWAP). If entry_limit
    is under Rs.99, SL additionally floors at entry_limit * 1.10 -- this
    WIDENS the stop when it applies, it never tightens or replaces the
    high/VWAP comparison (cheap premiums are more prone to a tight
    absolute SL getting whipsawed).

    Target is fixed 1:2 risk:reward off entry_limit, using the resulting
    SL distance as risk.

    FIX (2026-07-16, price rounding): entry_limit/sl_price/target_price
    were previously left as raw floats straight out of the arithmetic
    (e.g. entry=95.42, SL=104.96, target=76.34 on a live paper fill) --
    not tradeable/orderable tick sizes. round_to_half() is applied to
    entry_limit FIRST (since the low-premium SL-floor threshold check
    and the risk/target math both key off it), then sl_price and
    target_price are computed from that already-rounded entry_limit and
    rounded again themselves at the end. This keeps entry/SL/target
    internally consistent -- risk = sl_price - entry_limit uses the same
    rounded entry_limit that actually gets sent to the webhook, rather
    than computing risk off raw numbers and rounding the three outputs
    independently afterward (which could silently drift the real RR away
    from TARGET_RISK_REWARD by up to ~0.5 on each leg).

    NOTE: the LOW_PREMIUM_SL_THRESHOLD check below now runs against the
    ROUNDED entry_limit, not the raw one -- a raw value on either side of
    Rs.99 could legitimately round across that threshold, and using the
    rounded value keeps the floor check consistent with the entry_limit
    that's actually locked and sent.
    """
    trigger_high = trigger_candle["high"]
    trigger_vwap = trigger_candle["vwap"]
    entry_limit = round_to_half(trigger_candle["ema5"] * ENTRY_LIMIT_DISCOUNT)

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
