"""
Crossover detection logic (state-based vs. transition-based) and the
per-run scan across both sides (PE/CE) for a fresh entry trigger.
"""

import sys

import angelone_client as ac
from market_data import get_candles_with_cache
from indicators import compute_indicators, compute_squeeze_metrics
from logging_utils import log_signal_debug
from config import SELL_SQUEEZE_SPREAD_ATR_MIN
from calendar_utils import is_before_sell_scan_cutoff

# NOTE: compute_pending_signal is imported lazily inside
# scan_for_new_signal() below, not at module level. pending.py's
# manage_pending_signal() needs check_entry_signal() from THIS module
# (to decide whether a resting order's setup has invalidated), so a
# top-level "from pending import compute_pending_signal" here would
# create a signal_engine <-> pending circular import. Deferring the
# import to call time breaks the cycle: by the time
# scan_for_new_signal() actually runs, both modules have finished
# loading.


def _condition_holds(row):
    """
    Shared EMA5/EMA25/VWAP condition check for a single candle row. Used
    by both check_entry_signal() (state-based, "does the setup still
    hold") and is_fresh_crossover_signal() (transition-based, "did the
    setup JUST start holding").
    """
    return (
        row["ema5"] < row["vwap"]
        and row["ema25"] > row["ema5"]
        and row["ema25"] > row["vwap"]
    )


def check_entry_signal(df):
    """
    STATE check: is the EMA5/VWAP/EMA25 condition true on the latest
    candle, regardless of whether it just started or has held for a
    while? Deliberately state-based, NOT transition-based -- this is used
    by manage_pending_signal() to decide whether a resting limit order's
    setup has been INVALIDATED (see that function's cancellation check).
    A resting order shouldn't get cancelled just because the condition
    "isn't fresh" anymore; it should only get cancelled if the condition
    actually stopped holding. Do NOT use this for scanning brand-new
    entries -- see is_fresh_crossover_signal() for that.
    """
    if df is None or len(df) < 2:
        return False
    return _condition_holds(df.iloc[-1])


def is_fresh_crossover_signal(df):
    """
    FIX (2026-07-07, re-entry-on-held-state bug): check_entry_signal()
    above only checks whether the condition is true RIGHT NOW, not
    whether it just started being true. Since a resting pending_signal
    can expire unfilled (5-candle window) while the underlying condition
    never actually broke down, the OLD scan_for_new_signal() -- which
    called check_entry_signal() -- would immediately re-fire a brand new
    pending_signal on the very next scan, even though there was no new
    crossover at all, just the same still-held state. Confirmed on live
    paper data twice: the whipsaw SL day case study, and a 2026-07-06
    trade where a 9:35 crossover's pending signal expired unfilled and a
    "new" signal re-fired at 10:00 purely because EMA5 had never gone
    back above VWAP in between -- the debug log's own line
    ("prev candle EMA5 < VWAP : True -> state already held") confirmed
    this at the time.

    This requires the PREVIOUS candle to NOT satisfy the condition and
    the CURRENT candle to satisfy it -- i.e. an actual transition, not
    just a persisting state. Used only by scan_for_new_signal() for
    brand-new entries; manage_pending_signal()'s cancellation check
    correctly keeps using the state-based check_entry_signal() above.
    """
    if df is None or len(df) < 2:
        return False
    current = _condition_holds(df.iloc[-1])
    previous = _condition_holds(df.iloc[-2])
    return current and not previous


def scan_for_new_signal(state, leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_cache=None):
    """
    Scans the two sell legs (PE side, CE side) for a fresh EMA5<VWAP
    crossover. On a match, locks a resting pending_signal (see
    compute_pending_signal()) instead of entering immediately -- the fill
    itself happens later in manage_pending_signal() on a subsequent run.

    FIX (2026-07-07): now uses is_fresh_crossover_signal() instead of
    check_entry_signal() -- see that function's docstring for the full
    root-cause writeup. In short: check_entry_signal() is state-based
    ("is the condition true now"), which let this function re-fire a
    brand new pending_signal on the very next scan after a prior one
    expired unfilled, even with no actual new crossover -- just the same
    still-held EMA5<VWAP state. is_fresh_crossover_signal() additionally
    requires the previous candle to NOT have satisfied the condition, so
    a genuinely fresh cross is required for a new signal to fire.

    FIX (2026-07-29, run-level dedup): now accepts run_cache and passes
    it straight through to get_candles_with_cache() -- see that
    function's docstring for the full root-cause writeup. This is the
    function that usually fetches a sell-strike token FIRST each run, so
    it's typically what populates run_cache for the buy engine
    (scan_for_new_buy_signal_live()) to reuse afterward on the same run.

    NEW (2026-08-12, sell-side squeeze gate): a fresh crossover that
    would otherwise fire is now blocked if the trigger candle's
    spread_atr_ratio (see indicators.compute_squeeze_metrics()) is below
    config.SELL_SQUEEZE_SPREAD_ATR_MIN -- Pragnesh's call, sell-side
    only, hard threshold (not shadow-mode). A blocked leg does NOT
    return -- the loop continues to the next leg pair this run, since a
    squeeze-vetoed leg never actually fired and shouldn't consume the
    single-slot stop-after-first-fire behavior that only applies to a
    leg that genuinely fires. Diagnostic logging (log_signal_debug's
    [squeeze diag] line) already runs unconditionally above this check,
    for every leg scanned, fired or not -- this gate only affects
    whether a fired crossover is allowed to become a pending_signal.

    NEW (2026-08-13, sell-side scan cutoff): a fresh crossover at/after
    14:55 IST (config.SELL_SCAN_CUTOFF_TIME) is also blocked from
    creating a new pending_signal -- checked first, before the squeeze
    gate, since it's a cheap time check. Same continue-not-return
    behavior. Does NOT affect an already-resting pending_signal or an
    already-open sell position -- those are managed elsewhere
    (pending.manage_pending_signal(), position.py) and keep running
    every run regardless of this cutoff, right up to the existing
    EOD_SQUAREOFF handling.
    """
    from pending import compute_pending_signal  # local import -- see NOTE at top of file

    for leg in leg_pairs:
        sell_token_info = ac.resolve_option_token(instruments, expiry, leg["sell_strike"], leg["option_type"])
        if not sell_token_info:
            continue

        candles = get_candles_with_cache(smart_api, sell_token_info["token"], prev_day, prev_day_cache, today_start, run_cache)
        df = compute_indicators(candles)
        if df is None:
            continue

        # DEBUG (temporary, point 3): log full condition breakdown for
        # EVERY leg scanned this run, whether or not it fires, so it's
        # clear which specific condition is blocking a signal on each side.
        log_signal_debug(sell_token_info["symbol"], df)

        if is_fresh_crossover_signal(df):
            # NEW (2026-08-13, sell-side scan cutoff): checked first --
            # cheapest check, no computation needed -- before the squeeze
            # gate below. A fresh crossover at/after 14:55 does not
            # create a new pending_signal; an already-resting or
            # already-open sell position is untouched by this (see
            # calendar_utils.is_before_sell_scan_cutoff()'s docstring).
            if not is_before_sell_scan_cutoff():
                print(
                    f"SELL signal on {sell_token_info['symbol']} blocked -- past sell scan "
                    "cutoff (14:55). Continuing to next leg this run.",
                    file=sys.stderr,
                )
                continue

            # NEW (2026-08-12, sell-side squeeze gate): check BEFORE
            # resolving the hedge token / fetching hedge LTP, so a
            # squeeze-blocked leg doesn't waste those calls. None means
            # ATR wasn't available yet (e.g. very start of fetched
            # history) -- treated as "not enough data to judge squeeze",
            # so it does NOT block (fail-open, consistent with squeeze
            # being a diagnostic-first feature elsewhere in the codebase).
            _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(df.iloc[-1])
            if spread_atr_ratio is not None and spread_atr_ratio < SELL_SQUEEZE_SPREAD_ATR_MIN:
                print(
                    f"SELL signal on {sell_token_info['symbol']} blocked -- squeeze detected "
                    f"(spread_pct={spread_pct:.3f}%, spread/atr={spread_atr_ratio:.3f} < "
                    f"{SELL_SQUEEZE_SPREAD_ATR_MIN}). Continuing to next leg this run.",
                    file=sys.stderr,
                )
                continue

            hedge_token_info = ac.resolve_option_token(instruments, expiry, leg["hedge_strike"], leg["option_type"])
            if not hedge_token_info:
                print(f"Signal fired on {sell_token_info['symbol']} but hedge strike "
                      f"{leg['hedge_strike']}{leg['option_type']} could not be resolved -- skipping this signal.",
                      file=sys.stderr)
                continue

            # NEW (P&L tracking, 2026-07-06): capture the hedge leg's
            # current price at the moment the signal fires, so real
            # two-leg P&L can be computed later. This is a SEPARATE,
            # single LTP call -- not a candle fetch -- so it does not add
            # a second polling target to the 5-min loop. If this fetch
            # fails for any reason, the hedge entry price is simply
            # logged as None rather than blocking the signal -- P&L
            # tracking must never be able to block a real trade signal.
            hedge_entry_price = ac.fetch_option_ltp(
                smart_api, hedge_token_info["symbol"], hedge_token_info["token"]
            )
            if hedge_entry_price is None:
                print(f"Could not fetch hedge leg LTP for {hedge_token_info['symbol']} at signal time -- "
                      "hedge_entry_price will be logged as null; P&L for this trade will be incomplete.",
                      file=sys.stderr)

            trigger_candle = df.iloc[-1].to_dict()
            qty = sell_token_info["lot_size"]

            sell_leg_info = {**sell_token_info, "strike": leg["sell_strike"], "option_type": leg["option_type"]}
            hedge_leg_info = {**hedge_token_info, "strike": leg["hedge_strike"], "option_type": leg["option_type"]}

            state["pending_signal"] = compute_pending_signal(trigger_candle, sell_leg_info, hedge_leg_info, qty)
            state["pending_signal"]["hedge_entry_price"] = hedge_entry_price
            p = state["pending_signal"]
            print(f"PENDING SIGNAL: SELL {p['sell_symbol']} resting limit @ {p['entry_limit']:.2f} "
                  f"(SL={p['sl_price']:.2f}, target={p['target_price']:.2f})")
            # DEBUG (temporary, point 4): confirm this early return is what
            # stops the other leg (CE/PE) from being scanned in the same
            # run once one side has fired. state["pending_signal"] /
            # state["open_position"] are single dicts, not lists, so the
            # other leg is intentionally deferred to a later run rather
            # than evaluated now -- see investigation notes.
            print(
                f"[DEBUG] scan_for_new_signal: stopping after {p['sell_symbol']} -- "
                "remaining leg(s) in this run's leg_pairs were not scanned "
                "(single pending_signal slot in state.json).",
                file=sys.stderr,
            )
            return

    print("No entry signal this run.")
