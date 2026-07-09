"""
EMA5/EMA25/VWAP computation from a raw candle list, restricted to
today's session.
"""

import sys
import datetime as dt

import pandas as pd

from calendar_utils import now_ist


def compute_indicators(candles):
    df = pd.DataFrame(candles)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    today = now_ist().date()

    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema25"] = df["close"].ewm(span=25, adjust=False).mean()

    today_mask = df["time"].dt.date == today
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].where(today_mask, 0).cumsum()
    cum_tp_vol = (typical_price * df["volume"]).where(today_mask, 0).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, pd.NA)

    df = df[today_mask].reset_index(drop=True)

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
    # bar as if it were closed. Read-only -- does not change df, does not
    # affect any decision. Wrapped defensively so a formatting issue here
    # can never block a real run managing live positions.
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
            window_closed = elapsed_seconds >= 300
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
