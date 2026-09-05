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
N-1) -- no longer min(low, vwap) on the trigger candle itself.
prev_candle is now a REQUIRED argument to compute_pending_buy_signal()
(was optional before), since is_fresh_crossover_signal_buy() already
requires df.iloc[-2] to exist for the ROC-cross check itself. The
low-premium SL floor (Rs.99 / 10%, config.BUY_LOW_PREMIUM_SL_THRESHOLD /
BUY_LOW_PREMIUM_SL_MIN_PCT) is UNCHANGED -- still applied on top, still
widen-only.

Target = entry + BUY_TARGET_RISK_REWARD * (entry - SL) -- RR changed
from 1:2 to 1:1 as part of this same rework (config.BUY_TARGET_RISK_REWARD).

IMPORTANT -- what this rework does NOT touch: the 2026-08-14 buy-side
squeeze gate (config.BUY_SQUEEZE_SPREAD_ATR_MIN, see its own docstring
in config.py for the real paper-mode case study that added it) is
UNCHANGED and still gates entries in both scan_for_new_buy_signal() and
scan_for_new_buy_signal_live(), in the same place (checked right after
a fresh crossover fires, before the scan-window check, using the same
spread_atr_ratio already computed for the unconditional diagnostic
line). This gate operates on EMA5/EMA25/VWAP bunching, which has nothing
to do with the ROC-based trigger condition itself -- it's an orthogonal
filter on the trigger candle, so it carries over unchanged regardless
of what generates the crossover.

EMA5/EMA25 are no longer read by the CONDITION functions below (VWAP
still is, as the trigger-candle filter) -- but they remain available on
the shared indicator dataframe (indicators.compute_indicators()) for
the SELL side and for the squeeze gate/diagnostics above, unaffected.
"""

import sys

import pandas as pd

from config import (
    BUY_PENDING_SIGNAL_MAX_CANDLES, BUY_TARGET_RISK_REWARD,
    BUY_ROC_PERIOD, BUY_ROC_CROSS_LEVEL,
    BUY_LOW_PREMIUM_SL_THRESHOLD, BUY_LOW_PREMIUM_SL_MIN_PCT,
    BUY_SCAN_WINDOWS, BUY_SQUEEZE_SPREAD_ATR_MIN,
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

    NaN ROC (e.g. very first candles of the whole fetched history,
    before BUY_ROC_PERIOD prior closes exist) is treated as "condition
    not met" rather than raising.
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
    this function has always played, just backed by the ROC condition
    now instead of the old EMA-based one.
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
    again here.

    NaN ROC on either candle is treated as "not enough data to judge a
    cross" -- fails closed, does not fire.
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

    CHANGED (2026-09-04, ROC rework): entry/SL formulas replaced, per
    Pragnesh's explicit spec:

      entry_limit = trigger candle N's own CLOSE (was EMA5[N]). Still a
      pullback entry -- the limit rests at this level and fills on a
      LATER candle whose LOW comes down to touch it (see
      check_fill_buy()), not at N's own close as a market order.

      sl_price = candle N-1's LOW (was min(N's low, N's vwap), further
      floored against N-1's low). prev_candle is now a REQUIRED argument
      -- is_fresh_crossover_signal_buy() already requires df.iloc[-2] to
      exist for the ROC-cross check itself, so a valid prev_candle is
      guaranteed to be available by the time this is called from either
      scan path.

      target_price = entry_limit + BUY_TARGET_RISK_REWARD * risk, RR now
      1:1 by default (config.BUY_TARGET_RISK_REWARD, was 1:2 before this
      rework).

    UNCHANGED (2026-07-31, low-premium SL floor): mirrors the sell
    side's LOW_PREMIUM_SL_THRESHOLD/MIN_PCT treatment, inverted for
    direction -- buy's SL sits BELOW entry, so the floor widens it
    DOWNWARD (min, not max) when entry_limit is under
    BUY_LOW_PREMIUM_SL_THRESHOLD (Rs.99). Widen-only: this can only push
    SL further from entry, never closer than what candle N-1's low
    already computed. Checked against the ROUNDED entry_limit, same
    reasoning as before.

    Rounding: entry_limit rounded first, then the low-premium floor
    check runs against that rounded value, then sl_price is rounded,
    then risk/target_price computed off the already-rounded values --
    same order as before, so the real risk:reward sent to the webhook
    matches BUY_TARGET_RISK_REWARD exactly rather than drifting off raw
    floats.

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

    NEW (2026-08-14, buy-side squeeze gate): mirrors the sell-side
    squeeze gate in signal_engine.scan_for_new_signal() -- a fresh
    crossover that would otherwise return a pending signal is now
    blocked if spread_atr_ratio on the trigger candle is below
    config.BUY_SQUEEZE_SPREAD_ATR_MIN. This REVERSES the earlier "buy
    stays squeeze-free by design" call -- see BUY_SQUEEZE_SPREAD_ATR_MIN's
    docstring in config.py for the real paper-mode case study that
    overturned it (2026-08-14, NIFTY18AUG2624200PE, two same-day
    whipsaw entries at spread_atr_ratio 0.50 and 0.35). Added here (the
    standalone/backtest scan) as well as in scan_for_new_buy_signal_live()
    below, so test_buy_signal_engine.py's replay against candle_history/
    CSVs reflects the same gating live trading now has -- a backtest
    that silently omitted this gate would no longer represent what the
    live system actually does. None (ATR not yet available) fails open,
    same reasoning as the sell-side gate.
    """
    for leg_info, candles in candles_by_symbol.items():
        df = compute_indicators_fn(candles)
        if df is None:
            continue

        if is_fresh_crossover_signal_buy(df):
            _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(df.iloc[-1])
            ratio_str = f"{spread_atr_ratio:.2f}" if spread_atr_ratio is not None else "n/a"

            if spread_atr_ratio is not None and spread_atr_ratio < BUY_SQUEEZE_SPREAD_ATR_MIN:
                if log:
                    print(
                        f"BUY signal on {leg_info.get('symbol', '?')} blocked -- squeeze "
                        f"detected (spread_pct={spread_pct:.3f}%, spread/atr={ratio_str} < "
                        f"{BUY_SQUEEZE_SPREAD_ATR_MIN}). Continuing to next leg this scan.",
                        file=sys.stderr,
                    )
                continue

            trigger_candle = df.iloc[-1].to_dict()
            prev_candle = df.iloc[-2].to_dict()
            qty = leg_info.get("lot_size", leg_info.get("qty"))
            pending = compute_pending_buy_signal(trigger_candle, leg_info, qty, prev_candle)

            # NEW (2026-09-04, ROC rework): compute_pending_buy_signal()
            # can now return None if risk <= 0 (prev candle's low, or the
            # low-premium floor, at or above the trigger candle's close)
            # -- log and continue to the next leg rather than treating
            # None as a valid pending signal.
            if pending is None:
                if log:
                    print(
                        f"[buy scan] ROC crossover fired on {leg_info.get('symbol', '?')} "
                        "but computed risk <= 0 (prev candle low >= trigger close) -- "
                        "skipping this signal.",
                        file=sys.stderr,
                    )
                continue

            if log:
                # NEW (2026-07-30, squeeze diagnostics): log spread_pct /
                # spread-to-ATR on the actual trigger candle for every
                # fired signal too, not just no-fire scans -- this is the
                # data we need later to check "did signals that got
                # stopped out disproportionately have a tight spread_pct
                # or spread_atr_ratio here". Diagnostic logging stays
                # unconditional; BUY_SQUEEZE_SPREAD_ATR_MIN above is what
                # now actually gates the entry.
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


def scan_for_new_buy_signal_live(state, leg_pairs, instruments, expiry, smart_api,
                                  prev_day, prev_day_cache, today_start, run_cache=None):
    """
    Live-wired scan for the buy engine. Reuses the SAME locked sell
    strikes (one CE, one PE) the sell engine already tracks -- no new
    token lookups beyond what the sell side already resolves each day.

    REMOVED (2026-09-04): the same-strike guard
    (_strike_has_open_sell_position(), now deleted) that used to skip a
    strike here if the sell side had an open position on it. Pragnesh's
    explicit call: the buy side should fire independent of the sell
    side's state -- a long option's downside is capped at the premium
    paid, low enough cost that the same-strike overlap risk this guard
    protected against is no longer worth the missed signals it was
    blocking. Every leg in leg_pairs is now scanned unconditionally on
    every call to this function (still subject to run-order gating in
    main.py and the single-slot pending_buy_signal limit, both
    UNCHANGED, and still subject to the squeeze gate and scan-window
    gate below, both also UNCHANGED).

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

    NEW (2026-08-14, buy-side squeeze gate -- REVERSES the earlier "buy
    stays squeeze-free by design" call above and in
    BUY_ENGINE_INTEGRATION.md section 1). Real paper-mode evidence
    overturned that assumption: two same-day NIFTY18AUG2624200PE buy
    trades (11:16 entry, spread_atr_ratio=0.50; 11:36 re-entry,
    spread_atr_ratio=0.35) both fired as genuine fresh crossovers --
    setup correctly formed, no bug in the detection logic -- while
    EMA5/EMA25/VWAP were still tightly bunched at the trigger candle,
    and both were stopped out within minutes for near-identical losses
    (-244.03, -243.95). The original reasoning (EMA5's lag already
    filters squeeze-driven noise before a buy signal can fire) doesn't
    hold: the lag changes WHICH candle a signal fires on, not whether
    the three lines are still bunched together at that later candle.
    Pragnesh's call, 2026-08-14: mirror the sell-side gate exactly (same
    BUY_SQUEEZE_SPREAD_ATR_MIN=0.5 cutoff value as the sell side's own
    SELL_SQUEEZE_SPREAD_ATR_MIN, same hard-gate-not-shadow-mode posture,
    same continue-not-return -- a squeeze-blocked leg didn't genuinely
    fire, so it must not consume the single-slot stop-after-first-fire
    behavior). Checked here using the spread_atr_ratio already computed
    above for the unconditional squeeze-diagnostic log line -- no second
    computation needed. Checked BEFORE the buy-side scan-window gate
    below (arbitrary order between two independent O(1) gates; squeeze
    first only to mirror the sell side's own ordering, which checks
    squeeze before doing any further per-leg work).

    CHANGED (2026-09-04, ROC rework): the entry condition itself
    (is_fresh_crossover_signal_buy) and entry/SL/target computation
    (compute_pending_buy_signal) are now ROC-based -- see this module's
    top docstring. The squeeze gate and scan-window gate below are
    UNCHANGED, both mechanically and in placement -- they operate on
    EMA5/EMA25/VWAP bunching and time-of-day respectively, neither of
    which has anything to do with what generates the crossover.
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

        # NEW (2026-07-30, squeeze diagnostics): log spread_pct /
        # spread-to-ATR for this strike's latest candle on every live
        # run, fire or no-fire -- this loop runs against real paper-mode
        # data. UNCHANGED by the ROC rework.
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
            # NEW (2026-08-14, buy-side squeeze gate): see docstring
            # above for the full case-study writeup. spread_atr_ratio is
            # already computed above for the unconditional diagnostic
            # line -- reused here, not recomputed. None (ATR not yet
            # available, e.g. very start of fetched history) fails open,
            # same as the sell-side gate's treatment.
            if spread_atr_ratio is not None and spread_atr_ratio < BUY_SQUEEZE_SPREAD_ATR_MIN:
                print(
                    f"[buy scan] fresh crossover on {option_type} {strike} blocked -- "
                    f"squeeze detected (spread_pct={spread_pct:.3f}%, "
                    f"spread/atr={spread_atr_ratio:.3f} < {BUY_SQUEEZE_SPREAD_ATR_MIN}). "
                    "Continuing to next leg this run.",
                    file=sys.stderr,
                )
                continue

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

            pending_signal = compute_pending_buy_signal(trigger_candle, leg_info, qty, prev_candle)

            # NEW (2026-09-04, ROC rework): compute_pending_buy_signal()
            # can now return None if risk <= 0 -- log and continue to the
            # next leg rather than storing None as a "valid" pending
            # signal in state.json.
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
