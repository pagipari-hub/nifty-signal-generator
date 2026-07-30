"""
Read-only debug/observability logging. Nothing in this module may ever
be allowed to affect a real decision -- see log_signal_debug()'s own
docstring for why it's defensively wrapped.
"""

import sys

from indicators import compute_squeeze_metrics


def log_signal_debug(symbol, df):
    """
    TEMPORARY debug helper (point 3 of the investigation). For each
    scanned leg, prints a human-readable block:

        Scanning <symbol>
        Last candle: <HH:MM>
        EMA5 = <value>
        EMA25 = <value>
        VWAP = <value>
        EMA5 < VWAP : <bool>
        EMA25 > EMA5 : <bool>      (only reached if the above was True)
        EMA25 > VWAP : <bool>      (only reached if the above was True)
        Signal = <bool>

    Conditions are printed in the SAME order and with the SAME
    short-circuiting as check_entry_signal()'s "and" chain -- once one
    condition is False, the remaining ones aren't evaluated/printed and
    we go straight to "Signal = False". This mirrors actual evaluation
    order so the log tells you exactly which check blocked a signal.

    Does not change any decision logic -- read-only observability,
    called from scan_for_new_signal() for every leg scanned, not just
    ones that fire.

    Wrapped defensively: vwap can be pandas.NA (not NaN) on a candle with
    zero cumulative volume -- e.g. a thinly-traded hedge leg's first
    candle of the day -- because compute_indicators() does
    `cum_vol.replace(0, pd.NA)` before dividing. VWAP is printed without
    a ':.2f' spec for this reason (pd.NA doesn't support it). This is
    pure logging; it must never be able to crash a real run that's
    managing live positions, so any failure here is caught and reported
    instead of propagated.
    """
    try:
        last = df.iloc[-1]
        candle_time = last["time"]
        try:
            candle_time_str = candle_time.strftime("%H:%M")
        except AttributeError:
            candle_time_str = str(candle_time)

        ema5 = last["ema5"]
        ema25 = last["ema25"]
        vwap = last["vwap"]

        print(f"Scanning {symbol}", file=sys.stderr)
        print(f"Last candle: {candle_time_str}", file=sys.stderr)
        print(f"EMA5 = {ema5:.2f}", file=sys.stderr)
        print(f"EMA25 = {ema25:.2f}", file=sys.stderr)
        print(f"VWAP = {vwap}", file=sys.stderr)

        cond_ema5_below_vwap = bool(ema5 < vwap)
        print(f"EMA5 < VWAP : {cond_ema5_below_vwap}", file=sys.stderr)
        if not cond_ema5_below_vwap:
            print("Signal = False", file=sys.stderr)
            return

        cond_ema25_above_ema5 = bool(ema25 > ema5)
        print(f"EMA25 > EMA5 : {cond_ema25_above_ema5}", file=sys.stderr)
        if not cond_ema25_above_ema5:
            print("Signal = False", file=sys.stderr)
            return

        cond_ema25_above_vwap = bool(ema25 > vwap)
        print(f"EMA25 > VWAP : {cond_ema25_above_vwap}", file=sys.stderr)

        print(f"Signal = {cond_ema25_above_vwap}", file=sys.stderr)

        # Extra context for investigation point 1 (state vs. crossover) --
        # doesn't disturb the block above, just appends one more line when
        # a signal actually fires, so we can tell a fresh cross apart from
        # an already-established state.
        if len(df) >= 2:
            prev = df.iloc[-2]
            prev_below = bool(prev["ema5"] < prev["vwap"])
            print(
                f"(prev candle EMA5 < VWAP : {prev_below} -> "
                f"{'fresh crossover' if not prev_below else 'state already held'})",
                file=sys.stderr,
            )

        # NEW (2026-07-30, squeeze diagnostics): sell-side counterpart to
        # the logging already added to buy_signal_engine.py's scan loops.
        # Real paper-mode evidence (2026-07-30 13:15 trigger on
        # NIFTY04AUG2624350CE: spread_pct=0.19%, spread_atr_ratio=0.017,
        # confirmed against Zerodha's own ATR(14)=8.28) showed this SELL
        # signal fired inside an extreme squeeze immediately before a
        # violent whipsaw candle filled it with very little room -- so
        # this diagnostic belongs on the sell side too, not just buy.
        # No gating yet -- see indicators.compute_squeeze_metrics()
        # docstring; this is purely for collecting real threshold data
        # from live paper runs on both sides before any filter is wired in.
        _, spread_pct, spread_atr_ratio = compute_squeeze_metrics(last)
        ratio_str = f"{spread_atr_ratio:.3f}" if spread_atr_ratio is not None else "n/a"
        print(
            f"[squeeze diag] spread_pct={spread_pct:.3f}% spread/atr={ratio_str}",
            file=sys.stderr,
        )
    except Exception as e:
        # Never let a logging/formatting problem take down a run that's
        # managing real positions. Report it and move on.
        print(f"Scanning {symbol}: debug logging failed ({e!r}) -- "
              "continuing without it.", file=sys.stderr)
