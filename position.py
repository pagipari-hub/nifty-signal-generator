"""
Open-position management: legacy single-leg exit (unchanged, for the one
position opened before the spread rework) and the current spread exit
(which hands the SL/target bracket decision to webhook.py).
"""

import sys

import angelone_client as ac
from calendar_utils import now_ist, is_eod_squareoff_time
from market_data import get_candles_with_cache
from indicators import compute_indicators
from webhook import send_to_webhook, webhook_confirmed_ok


def manage_legacy_single_leg_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    UNCHANGED dynamic-SL exit logic for single-leg positions opened before
    the pending-signal/spread rework landed (identified by the absence of
    a "spread" key). Do not change this function's behaviour -- it exists
    only so the position already open in state.json as of 2026-07-01
    (NIFTY07JUL2624050PE) gets managed through to its own exit correctly.
    New positions never take this path; see manage_spread_exit() instead.
    """
    token_info = ac.resolve_option_token(instruments, expiry, pos["strike"], pos["option_type"])
    if not token_info:
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)
    if df is None:
        return

    last = df.iloc[-1]
    sl_hit = last["close"] > last["vwap"]
    target_hit = last["close"] <= pos["target_price"]

    if sl_hit or target_hit or is_eod_squareoff_time():
        reason = "SL" if sl_hit else ("TARGET" if target_hit else "EOD")
        resp = send_to_webhook({
            "action": "EXIT",
            "reason": reason,
            "symbol": pos["symbol"],
            "qty": pos["qty"],
            "price": float(last["close"]),
            "time": now_ist().isoformat(),
        })

        if webhook_confirmed_ok(resp):
            state["open_position"] = None
        else:
            print(
                "EXIT webhook not confirmed -- leaving open_position "
                "in state.json so the next run retries.",
                file=sys.stderr,
            )


def manage_spread_exit(state, pos, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start):
    """
    FIX (OCO bracket moved to webhook.py, 2026-07-03): previously this
    function decided SL/target itself using only the latest candle's
    CLOSE (`close > pos["sl_price"]` / `close <= pos["target_price"]`).
    That let price spike through SL intrabar and close back on the safe
    side, reporting "no SL hit" even though a real resting SL order at a
    broker fires the moment price TOUCHES it. Confirmed case (2026-07-03):
    SL=67.81, price spiked above and closed back under within the same
    5-min candle -- the old close-only check never fired, leaving the
    position open with unbounded intrabar risk (Pragnesh's "what if it
    flies to 99?" scenario).

    This function no longer decides SL vs target at all. It just fetches
    the candle (unchanged) and hands high/low/close, plus the position's
    fixed SL/target levels, to webhook.py via the new MANAGE_SPREAD
    action. webhook.py owns the actual bracket decision -- see
    check_spread_bracket() there: checks against HIGH/LOW (not close),
    with an explicit SL-wins tie-break if a single candle's range touches
    both levels (Pragnesh's call: a same-candle double-touch reads as a
    pullback, protect capital first).

    EOD square-off is kept as a SEPARATE, simpler path on purpose -- it
    isn't an SL/target bracket decision, just "flatten regardless of
    price", so it still goes straight through EXIT_SPREAD rather than
    MANAGE_SPREAD.

    NOTE: this only fixes WHICH price triggers the exit and WHO decides
    it. It does not yet place a real broker-side SL/target order in
    LIVE_MODE -- see CHANGELOG for that as a separate, larger piece of
    work (confirmed real-order entry -> real bracket placement).
    """
    sell_leg = pos["sell_leg"]
    token_info = ac.resolve_option_token(instruments, expiry, sell_leg["strike"], pos["option_type"])
    if not token_info:
        return

    candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start)
    df = compute_indicators(candles)
    if df is None:
        return

    last = df.iloc[-1]

    # DEBUG (temporary): this function no longer decides SL/target itself
    # (see FIX note above -- that now lives in webhook.py), but it's
    # still useful to see locally what candle data is being sent up and
    # what the webhook decided, without needing to cross-reference two
    # services' logs. Read-only, wrapped so a formatting issue here can
    # never block a real exit.
    try:
        candle_time_str = last["time"].strftime("%H:%M")
    except Exception:
        candle_time_str = str(last.get("time"))
    try:
        close = float(last["close"])
        low = float(last["low"])
        high = float(last["high"])
        print(f"Checking exit: {sell_leg['symbol']}", file=sys.stderr)
        print(f"Last candle: {candle_time_str}", file=sys.stderr)
        print(f"Close = {close:.2f}  Low = {low:.2f}  High = {high:.2f}", file=sys.stderr)
        print(f"SL = {pos['sl_price']:.2f}  Target = {pos['target_price']:.2f}", file=sys.stderr)
        print(f"Target touched intra-candle (low <= target) : {low <= pos['target_price']}", file=sys.stderr)
        print(f"SL touched intra-candle (high >= SL) : {high >= pos['sl_price']}", file=sys.stderr)
    except Exception as e:
        print(f"Checking exit: {sell_leg['symbol']}: debug logging failed ({e!r}) -- continuing without it.",
              file=sys.stderr)

    # ---- EOD square-off: unconditional flatten, not a bracket decision ----
    if is_eod_squareoff_time():
        # NEW (P&L tracking, 2026-07-06): capture hedge leg's current
        # price at EOD exit, same single-LTP-call approach as at entry.
        # Never blocks the exit itself -- if this fetch fails, exit still
        # proceeds and hedge_exit_price is simply logged as None.
        hedge_exit_price = ac.fetch_option_ltp(
            smart_api, pos["hedge_leg"]["symbol"], pos["hedge_leg"]["token"]
        )
        if hedge_exit_price is None:
            print(f"Could not fetch hedge leg LTP for {pos['hedge_leg']['symbol']} at EOD exit -- "
                  "hedge_exit_price will be logged as null; P&L for this trade will be incomplete.",
                  file=sys.stderr)

        resp = send_to_webhook({
            "action": "EXIT_SPREAD",
            "reason": "EOD",
            "sell_symbol": sell_leg["symbol"],
            "hedge_symbol": pos["hedge_leg"]["symbol"],
            "qty": pos["qty"],
            "price": float(last["close"]),
            "hedge_exit_price": hedge_exit_price,
            "entry_price": pos["entry_price"],
            "hedge_entry_price": pos.get("hedge_entry_price"),
            "time": now_ist().isoformat(),
        })

        if webhook_confirmed_ok(resp):
            state["open_position"] = None
        else:
            print(
                "EXIT_SPREAD (EOD) webhook not confirmed -- leaving open_position "
                "in state.json so the next run retries.",
                file=sys.stderr,
            )
        return

    # ---- SL/target bracket check -- decision now lives in webhook.py ----
    # FIX (2026-07-06): the hedge-price fetch below used to run
    # UNCONDITIONALLY on every run a position is open -- including every
    # "still open, nothing happened" run, which is most of them. That
    # added an extra API call (plus PRE_CALL_DELAY_SECONDS) BEFORE the
    # webhook call on every single run, including the one run that
    # actually matters most: the run where SL/target just got hit,
    # where the webhook needs to be reached as fast as possible, not
    # slower. It also added unnecessary rate-limit exposure on every
    # idle run for zero benefit on those runs.
    #
    # Fix: a lightweight LOCAL pre-check, mirroring webhook.py's
    # check_spread_bracket() (same HIGH/LOW touch check, same SL-wins
    # tie-break), used ONLY to decide whether it's worth fetching the
    # hedge price this run. This never overrides or duplicates the
    # actual decision -- webhook.py's check_spread_bracket() remains the
    # single authoritative source of truth for whether the position
    # actually closes. Worst case if this local mirror ever drifts out
    # of sync with webhook.py's real logic: hedge_current_price is
    # occasionally None on an actual close (P&L logged as incomplete for
    # that one trade, exit itself unaffected) or fetched once
    # unnecessarily on a run that turns out not to close -- neither
    # case blocks or delays the real exit decision, which is why the
    # webhook call itself is now issued immediately, before this fetch,
    # rather than after it.
    candle_high = float(last["high"])
    candle_low = float(last["low"])
    sl_price = pos["sl_price"]
    target_price = pos["target_price"]
    bracket_likely_hit = (candle_high >= sl_price) or (candle_low <= target_price)

    hedge_current_price = None
    if bracket_likely_hit:
        hedge_current_price = ac.fetch_option_ltp(
            smart_api, pos["hedge_leg"]["symbol"], pos["hedge_leg"]["token"]
        )
        if hedge_current_price is None:
            print(f"Bracket looks hit (local pre-check) but hedge leg LTP fetch failed for "
                  f"{pos['hedge_leg']['symbol']} -- proceeding with the exit regardless; "
                  "P&L for this trade will be logged as incomplete.",
                  file=sys.stderr)

    resp = send_to_webhook({
        "action": "MANAGE_SPREAD",
        "sell_symbol": sell_leg["symbol"],
        "hedge_symbol": pos["hedge_leg"]["symbol"],
        "qty": pos["qty"],
        "candle_high": candle_high,
        "candle_low": candle_low,
        "sl_price": sl_price,
        "target_price": target_price,
        "hedge_current_price": hedge_current_price,
        "entry_price": pos["entry_price"],
        "hedge_entry_price": pos.get("hedge_entry_price"),
        "time": now_ist().isoformat(),
    })

    if resp is None:
        print(
            "MANAGE_SPREAD webhook call failed outright -- leaving open_position "
            "in state.json so the next run retries.",
            file=sys.stderr,
        )
        return

    try:
        body = resp.json()
    except ValueError:
        body = {}

    if resp.status_code == 200 and body.get("status") == "ok":
        if body.get("closed"):
            print(f"Position closed by bracket check: reason={body.get('reason')} "
                  f"price={body.get('price')}")
            state["open_position"] = None
        else:
            print("Exit = False (still open, per webhook's bracket check)", file=sys.stderr)
    else:
        print(
            f"MANAGE_SPREAD webhook not confirmed (status={resp.status_code}, "
            f"body={body}) -- leaving open_position in state.json so the next "
            "run retries.",
            file=sys.stderr,
        )
