"""
EMA5/EMA25/VWAP computation from a raw candle list, restricted to
today's session.
"""

import sys
import datetime as dt

import pandas as pd

from calendar_utils import now_ist
from config import SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER


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

    Fix: after restricting to today's candles, check the LAST row's own
    elapsed-since-open. If it's under CANDLE_WINDOW_SECONDS, that candle
    is still forming -- drop it. Every caller (signal checks, EMA5/EMA25,
    entry_limit/SL/target) then only ever sees genuinely closed candles.
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


def _compute_supertrend(df, period, multiplier):
    """
    NEW (2026-09-15, sell-side trailing SL groundwork): standard
    Supertrend(period, multiplier) computation. Adds four columns:
      - atr_<period>   -- Wilder-smoothed ATR at THIS period (deliberately
                           separate from the module-level ATR_PERIOD=14
                           used by compute_squeeze_metrics()).
      - st_upper        -- the "final upper band" series. For a SHORT
                            position this is the relevant trailing-SL
                            candidate (resistance above price) -- see
                            position.manage_spread_exit() for how it's
                            applied (monotonic-tighten only, i.e.
                            new_sl = min(old_sl, st_upper[-1])).
      - st_lower        -- the "final lower band" series (not currently
                            consumed anywhere -- sell side only uses
                            st_upper -- but computed for symmetry/future
                            use).
      - supertrend / supertrend_direction -- the standard flip-based
                            Supertrend line + direction (+1 up, -1 down).
                            Not currently used for any entry/exit decision
                            -- computed for visibility/future use only.

    Must run on the FULL multi-day history passed in -- the final-band
    recurrence needs several candles of lookback to be meaningful.
    Implemented as an explicit Python loop rather than a vectorized
    pandas operation -- the final_upper/final_lower recurrence depends on
    each row's own previous value.
    """
    if df.empty:
        df["st_upper"] = []
        df["st_lower"] = []
        df["supertrend"] = []
        df["supertrend_direction"] = []
        return df

    atr_col = f"atr_{period}"
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df[atr_col] = tr.ewm(alpha=1 / period, adjust=False).mean()

    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = (hl2 + multiplier * df[atr_col]).tolist()
    basic_lower = (hl2 - multiplier * df[atr_col]).tolist()
    closes = df["close"].tolist()

    n = len(df)
    final_upper = [0.0] * n
    final_lower = [0.0] * n
    supertrend = [0.0] * n
    supertrend_direction = [0] * n

    for i in range(n):
        if i == 0:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            supertrend[i] = final_upper[i]
            supertrend_direction[i] = -1
            continue

        if basic_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if supertrend[i - 1] == final_upper[i - 1]:
            if closes[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                supertrend_direction[i] = -1
            else:
                supertrend[i] = final_lower[i]
                supertrend_direction[i] = 1
        else:
            if closes[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
                supertrend_direction[i] = 1
            else:
                supertrend[i] = final_upper[i]
                supertrend_direction[i] = -1

    df["st_upper"] = final_upper
    df["st_lower"] = final_lower
    df["supertrend"] = supertrend
    df["supertrend_direction"] = supertrend_direction
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
    # session's tail, exactly like EMA5/EMA25 already do.
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

    # NEW (2026-09-15, sell-side trailing Supertrend SL): computed on the
    # FULL multi-day history, same warm-up reasoning as ATR(14)/EMA
    # above -- see _compute_supertrend()'s docstring for the full
    # writeup. Uses its OWN atr_10 column, deliberately separate from
    # df["atr"] (14) above, which squeeze diagnostics depend on and must
    # not change.
    df = _compute_supertrend(df, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER)

    today_mask = df["time"].dt.date == today
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].where(today_mask, 0).cumsum()
    cum_tp_vol = (typical_price * df["volume"]).where(today_mask, 0).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, pd.NA)

    df = df[today_mask].reset_index(drop=True)

    # FIX (2026-07-17): drop the last candle if its own window hasn't
    # closed yet -- see _drop_unclosed_last_candle() docstring above.
    # Must happen AFTER the today-filter (so we're checking the actual
    # latest today-candle) and BEFORE the len(df) < 2 check below.
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
    # elapsed by wall-clock "now". Read-only. Wrapped defensively so a
    # formatting issue here can never block a real run managing live
    # positions.
    try:
        n = now_ist()
        tail = df.tail(4)
        print("[DEBUG] compute_indicators: raw last candles (time/O/H/L/C/V, window-elapsed):",
              file=sys.stderr)
        for _, row in tail.iterrows():
            candle_time = row["time"]
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
    NEW (2026-07-30, diagnostic + sell-side gate support).

    How tightly EMA5/EMA25/VWAP are bunched together on a given
    indicator row -- the working theory (Pragnesh, 2026-07-30) is that
    crossover signals fired while these three are in a "squeeze" are
    disproportionately the ones that get stopped out, because (a) SL
    distance is itself derived from how close price/vwap sit to entry,
    so a squeeze mechanically produces a tight SL, and (b) a squeeze is
    also when EMA5/VWAP tend to whipsaw back and forth, producing
    repeated low-conviction "fresh crossovers" rather than one clean one.

    Used by:
      - logging_utils.log_signal_debug()'s [squeeze diag] line (always
        logged, fire or no-fire, sell side)
      - signal_engine.scan_for_new_signal()'s sell-side squeeze gate
        (SELL_SQUEEZE_SPREAD_ATR_MIN)

    NOTE (2026-09-15): deliberately still reads row["atr"] -- the
    ATR(14) column -- NOT the new row["atr_10"] added for Supertrend.
    These are two independent ATR series at different periods; squeeze
    diagnostics must keep using the same ATR(14) they've always used, so
    historical spread_atr_ratio values stay comparable.

    Returns (spread, spread_pct, spread_atr_ratio):
      - spread           = max(ema5, ema25, vwap) - min(ema5, ema25, vwap)
      - spread_pct       = spread / vwap * 100 -- scale-invariant across
                            different strikes/premium levels/days.
      - spread_atr_ratio = spread / atr -- how tight the bunching is
                            RELATIVE to how much this specific option's
                            premium is actually moving right now. None if
                            ATR isn't available yet or is zero.
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
