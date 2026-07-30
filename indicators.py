"""
EMA5/EMA25/VWAP computation from a raw candle list, restricted to
today's session.
"""

import sys
import datetime as dt

import pandas as pd

from calendar_utils import now_ist


CANDLE_WINDOW_SECONDS = 300  # 5-minute candles
ATR_PERIOD = 14  # Wilder's smoothing, same convention as EMA5/EMA25's span


def _drop_unclosed_last_candle(df):
    """
    FIX (2026-07-17, still-forming-candle root cause): compute_indicators()
    previously trusted df.iloc[-1] as "the latest COMPLETED candle" purely
    because it was today's last row by timestamp -- it never checked
    whether that candle's own 5-minute window had actually elapsed by
    wall-clock "now". Angel One can and does return an in-progress bar
    (e.g. a candle labelled 09:30:00, covering 09:30:00-09:34:59) as if it
    were closed, with its OHLCV still changing between calls.

    Confirmed on the 2026-07-17 09:31-09:32 IST run: token 57345's
    "09:30:00" candle was fetched three times within ~7 seconds by this
    single run (once via candle_priming's prime call, twice via
    scan_for_new_signal's own fetch) and returned a DIFFERENT close each
    time (96.55, then 97.00) -- elapsed_since_open was only ~132s at
    evaluation time, well under the 300s a closed candle requires. The
    fresh-crossover signal that fired or a resting entry_limit locked off
    that candle's EMA5 would both have been computed against a moving
    target, not a stable reference point -- compute_pending_signal()'s own
    docstring assumes entry_limit is computed "at candle-N-close time",
    which this violated.

    Fix: after restricting to today's candles, check the LAST row's own
    elapsed-since-open. If it's under CANDLE_WINDOW_SECONDS, that candle
    is still forming -- drop it. Every caller (signal checks, EMA5/EMA25,
    entry_limit/SL/target) then only ever sees genuinely closed candles.
    A dropped candle simply isn't available yet this run; it will appear
    (closed) on the next run once its window has actually elapsed, same
    as a normal 5-min cadence.

    Uses the same elapsed-time math as the existing raw-candle debug
    block (naive datetime comparison on IST wall-clock components) so
    this stays consistent with what's already being logged. Wrapped
    defensively: if anything about the last row is malformed, the
    original df is returned unchanged rather than raising -- this must
    never be able to crash a run managing a live position over a
    formatting issue.
    """
    if df.empty:
        return df

    try:
        n = now_ist()
        last_time = df.iloc[-1]["time"]
        candle_dt_naive = dt.datetime.combine(last_time.date(), last_time.time())
        now_naive = dt.datetime.combine(n.date(), n.time())
        elapsed_seconds = (now_naive - candle_dt_naive).total_seconds()

        if elapsed_seconds < CANDLE_WINDOW_SECONDS:
            print(
                f"[DEBUG] _drop_unclosed_last_candle: last candle time={last_time} "
                f"elapsed_since_open={elapsed_seconds:.0f}s < {CANDLE_WINDOW_SECONDS}s -- "
                "still forming, dropping it from this run's evaluation.",
                file=sys.stderr,
            )
            return df.iloc[:-1].reset_index(drop=True)
    except Exception as e:
        print(f"[DEBUG] _drop_unclosed_last_candle: check failed ({e!r}) -- "
              "returning df unchanged.", file=sys.stderr)

    return df


def compute_indicators(candles):
    df = pd.DataFrame(candles)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    today = now_ist().date()

    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema25"] = df["close"].ewm(span=25, adjust=False).mean()

    # NEW (2026-07-30, squeeze-detection groundwork): ATR(14), Wilder's
    # smoothing (ewm with alpha=1/period), same full-history warmup
    # pattern as EMA5/EMA25 above -- computed BEFORE the today-filter so
    # early-morning candles still get a warmed-up value from the prior
    # session's tail, exactly like EMA5/EMA25 already do. True range
    # needs the prior candle's close; the very first row in the whole
    # fetched history has no prior close, so its tr2/tr3 are NaN and
    # pandas' row-wise max() falls back to tr1 (high-low) for that one
    # row only -- not a problem in practice since that row is always
    # from well before today given the multi-day candle history fetched.
    # Purely additive: nothing below reads df["atr"] yet, this just makes
    # it available for the squeeze-metrics diagnostics being logged from
    # buy_signal_engine.py while real threshold values are gathered from
    # paper-mode data (no signal is gated on this yet).
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    today_mask = df["time"].dt.date == today
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].where(today_mask, 0).cumsum()
    cum_tp_vol = (typical_price * df["volume"]).where(today_mask, 0).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, pd.NA)

    df = df[today_mask].reset_index(drop=True)

    # FIX (2026-07-17): drop the last candle if its own window hasn't
    # closed yet -- see _drop_unclosed_last_candle() docstring above for
    # the full root-cause writeup. Must happen AFTER the today-filter
    # (so we're checking the actual latest today-candle) and BEFORE the
    # len(df) < 2 check below (an unclosed candle shouldn't count toward
    # "do we have enough data").
    df = _drop_unclosed_last_candle(df)

    if len(df) < 2:
        print(
            f"[DEBUG] compute_indicators: only {len(df)} today candle(s) available "
            f"(need >=2) -- returning None. now_ist={now_ist().isoformat()}",
            file=sys.stderr,
        )
        return None

    # DEBUG (temporary): confirm which candle is actually being treated as
    # "latest completed" vs. the current wall-clock time this run executed.
    print(
        f"[DEBUG] compute_indicators: latest completed candle time="
        f"{df.iloc[-1]['time']} | run time now_ist={now_ist().isoformat()}",
        file=sys.stderr,
    )

    # DEBUG (temporary, 2026-07-07 -- investigating possible partial/still-
    # forming candle from Angel One): print the raw time+OHLCV for the last
    # few candles, plus whether each candle's own 5-min window has actually
    # elapsed by wall-clock "now". A candle labeled e.g. 10:35:00 covers
    # 10:35:00-10:39:59; if this run's now_ist is still inside that window
    # (elapsed_seconds < 300), the row's close/high/low/volume may still be
    # changing mid-candle, and Angel One may be returning that in-progress
    # bar as if it were closed. Read-only -- does not change df (as of
    # 2026-07-17, the unclosed LAST candle is already dropped above; this
    # block now mainly documents the window-elapsed state of the remaining
    # closed candles for audit purposes), does not affect any decision.
    # Wrapped defensively so a formatting issue here can never block a
    # real run managing live positions.
    try:
        n = now_ist()
        tail = df.tail(4)
        print("[DEBUG] compute_indicators: raw last candles (time/O/H/L/C/V, window-elapsed):",
              file=sys.stderr)
        for _, row in tail.iterrows():
            candle_time = row["time"]
            # candle_time is tz-naive (from pd.to_datetime on Angel One's
            # string); compare wall-clock elapsed against the naive IST
            # clock components only, avoiding a tz-aware/naive subtraction.
            candle_dt_naive = dt.datetime.combine(candle_time.date(), candle_time.time())
            now_naive = dt.datetime.combine(n.date(), n.time())
            elapsed_seconds = (now_naive - candle_dt_naive).total_seconds()
            window_closed = elapsed_seconds >= CANDLE_WINDOW_SECONDS
            print(
                f"    time={candle_time} open={row['open']:.2f} high={row['high']:.2f} "
                f"low={row['low']:.2f} close={row['close']:.2f} volume={row['volume']:.0f} "
                f"elapsed_since_open={elapsed_seconds:.0f}s window_closed={window_closed}",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"[DEBUG] compute_indicators: raw-candle debug logging failed ({e!r}) -- "
              "continuing without it.", file=sys.stderr)

    return df


def compute_squeeze_metrics(row):
    """
    NEW (2026-07-30, diagnostic only -- does not gate any signal yet).

    How tightly EMA5/EMA25/VWAP are bunched together on a given
    indicator row -- the working theory (Pragnesh, 2026-07-30) is that
    crossover signals fired while these three are in a "squeeze" are
    disproportionately the ones that get stopped out, because (a) SL
    distance is itself derived from how close price/vwap sit to entry,
    so a squeeze mechanically produces a tight SL, and (b) a squeeze is
    also when EMA5/VWAP tend to whipsaw back and forth, producing
    repeated low-conviction "fresh crossovers" rather than one clean one.

    Returns (spread, spread_pct, spread_atr_ratio):
      - spread           = max(ema5, ema25, vwap) - min(ema5, ema25, vwap)
      - spread_pct       = spread / vwap * 100 -- scale-invariant across
                            different strikes/premium levels/days, so a
                            fixed threshold means roughly the same thing
                            regardless of whether the premium is Rs. 40
                            or Rs. 400.
      - spread_atr_ratio = spread / atr -- how tight the bunching is
                            RELATIVE to how much this specific option's
                            premium is actually moving right now. None if
                            ATR isn't available yet (e.g. very first
                            candles of the whole fetched history) or is
                            zero, since dividing by it would be
                            meaningless.

    No thresholds are applied here on purpose -- the plan is to log
    these values during paper trading first (see buy_signal_engine.py's
    scan debug output) and pick real cutoffs from what a squeeze vs. a
    clean signal actually look like in this specific data, rather than
    guessing numbers with no premium-level calibration behind them.
    """
    values = [row["ema5"], row["ema25"], row["vwap"]]
    spread = max(values) - min(values)

    vwap = row["vwap"]
    spread_pct = (spread / vwap * 100) if vwap else float("inf")

    atr = row["atr"]
    spread_atr_ratio = None
    if atr is not None and not pd.isna(atr) and atr > 0:
        spread_atr_ratio = spread / atr

    return spread, spread_pct, spread_atr_ratio
