"""
Previous-day candle caching and the merged (prev-day + today) candle
fetch used by every indicator computation.
"""

import json
import os
import sys
import datetime as dt

import angelone_client as ac
from config import PREV_DAY_CACHE_FILE, MARKET_OPEN, MARKET_CLOSE


def load_prev_day_cache(expected_date):
    if not os.path.exists(PREV_DAY_CACHE_FILE):
        return {}
    try:
        with open(PREV_DAY_CACHE_FILE, "r") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if cache.get("date") != expected_date.isoformat():
        return {}
    return cache.get("candles_by_token", {})


def save_prev_day_cache(date, candles_by_token):
    with open(PREV_DAY_CACHE_FILE, "w") as f:
        json.dump({"date": date.isoformat(), "candles_by_token": candles_by_token}, f)


def get_candles_with_cache(smart_api, token, prev_day, prev_day_cache, today_start):
    """
    FIX (empty-cache poisoning, 2026-07-03): previously this cached
    whatever fetch_5min_candles() returned unconditionally -- including
    an empty list from a rate-limit failure. Once that empty result got
    written into prev_day_cache[token_str] and persisted to
    prev_day_candles.json via save_prev_day_cache(), every later run
    THAT DAY would see the token already "in" the cache and skip
    re-fetching entirely -- permanently starving that token of
    previous-day candles for the rest of the day, with no retry. This is
    exactly what produced a prev_day_candles.json labeled with the
    correct date (e.g. 2026-07-02) but silently empty candle data for
    the affected token: EMA5/EMA25 ended up computed with zero
    previous-day warm-up instead of a genuine reset-at-open or a
    correctly-seeded series.

    Fix: only write to prev_day_cache when the fetch actually returned
    data. An empty result is treated as "not yet cached" so the NEXT run
    retries the fetch instead of permanently trusting a failed attempt.

    FIX (2026-07-07, VWAP/EMA divergence root cause): this prev-day fetch
    previously passed only start_time=prev_day_start with no end_time --
    fetch_5min_candles() then defaulted todate to "right now", so this
    call actually spanned from previous-day market open through THE
    CURRENT LIVE RUN, silently pulling today's candles into what gets
    cached (and persisted to disk) as "previous day" data. Confirmed
    twice independently: the original VWAP-divergence case study, and a
    2026-07-06 live paper trade where the system's implied EMA5 (back-
    computed from a locked entry limit) didn't match the EMA5 plotted on
    the chart for the same candle. Explicitly bounding end_time to the
    previous day's own market close fixes this at the source.
    """
    token_str = str(token)

    if token_str in prev_day_cache:
        prev_candles = prev_day_cache[token_str]
    else:
        prev_day_start = dt.datetime.combine(prev_day, MARKET_OPEN)
        prev_day_end = dt.datetime.combine(prev_day, MARKET_CLOSE)
        prev_candles = ac.fetch_5min_candles(
            smart_api, token, start_time=prev_day_start, end_time=prev_day_end
        )
        if prev_candles:
            prev_day_cache[token_str] = prev_candles
        else:
            print(
                f"[WARN] get_candles_with_cache: prev-day fetch for token {token_str} "
                f"returned no candles (prev_day={prev_day.isoformat()}) -- NOT caching "
                "this empty result, will retry on next run.",
                file=sys.stderr,
            )

    today_candles = ac.fetch_5min_candles(smart_api, token, start_time=today_start)

    seen_times = {c["time"] for c in prev_candles}
    merged = list(prev_candles) + [c for c in today_candles if c["time"] not in seen_times]
    return merged
