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


def get_candles_with_cache(smart_api, token, prev_day, prev_day_cache, today_start, run_cache=None):
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

    FIX (2026-07-29, run-level duplicate-fetch root cause): prev_day_cache
    (above) only ever de-duplicates the PREVIOUS day's candles, and only
    across separate runs (it's persisted to prev_day_candles.json). It
    was never a cache for TODAY's candles, and nothing de-duplicated
    calls to THIS function within a single run -- so when both the sell
    engine (signal_engine.scan_for_new_signal() / pending.py /
    position.py) and the buy engine (buy_signal_engine.py) called this
    function for the SAME token in the SAME run (e.g. the buy engine
    re-checking the CE sell-strike the sell engine had just scanned
    moments earlier), today's candles got fetched over the wire twice --
    doubling real API calls and rate-limit exposure for zero benefit,
    since the data is identical both times. Confirmed via a 2026-07-28
    live run log showing two separate getCandleData calls, seconds
    apart, for the same token.

    Fix: an optional run_cache dict, created fresh once per main.py
    invocation and threaded through every caller. If the token's merged
    candle list is already in run_cache, return it directly with no
    network call at all. This is IN-MEMORY and per-run only -- it is
    never persisted to disk and has no relationship to prev_day_cache
    (which remains the only thing written to prev_day_candles.json).
    Passing run_cache=None (the default) preserves the exact old
    behaviour -- always fetch today's candles fresh -- so any caller not
    yet updated to pass it keeps working unchanged.
    """
    token_str = str(token)

    if run_cache is not None and token_str in run_cache:
        return run_cache[token_str]

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

    if run_cache is not None:
        run_cache[token_str] = merged

    return merged
