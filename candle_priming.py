"""
One-shot cache-priming step, run immediately after the 9:30 strike lock.

Separate from the per-run scan (signal_engine.scan_for_new_signal()) on
purpose: the strike lock is the earliest point in the day we know which
2 sell legs actually matter, so this is the earliest point we can start
warming up their prev-day + today candle cache -- before the first
signal evaluation ever runs on them. This does NOT change how many
retries a single fetch gets (that's still fetch_5min_candles()'s own
backoff, see angelone_client.py) -- it just gives the prev-day fetch one
extra, dedicated attempt right after lock, on top of whatever attempt
scan_for_new_signal() would make on this same run anyway.

Deliberately does NOT add an outer retry loop beyond what
get_candles_with_cache() / fetch_5min_candles() already do. If priming
fails here (e.g. rate-limited, same as the 2026-07-15 case), it is NOT
retried again within this run -- get_candles_with_cache() only caches a
successful fetch, so the next scheduled run (9:35) will naturally retry
the prev-day fetch on its own, same as it already does today. This is
intentional breathing room: hammering harder within a single run risks
compounding the same rate-limit problem that caused the failure in the
first place.

Only primes the 2 SELL legs (the ones scan_for_new_signal() actually
evaluates for entry signals) -- not the 2 hedge legs, which are only
looked up at fill time and don't feed into EMA/VWAP signal computation.
"""

import sys

import angelone_client as ac
from market_data import get_candles_with_cache


def prime_candle_cache(leg_pairs, instruments, expiry, smart_api, prev_day, prev_day_cache, today_start, run_cache=None):
    """
    FIX (2026-07-29, run-level dedup): now accepts run_cache and passes it
    straight through to get_candles_with_cache() -- see that function's
    docstring for the full root-cause writeup. Priming already runs
    first (immediately after the 9:30 lock), so this is a free win: once
    it warms a sell-leg token into run_cache, everything else that
    touches that same token later in this same run (sell scan, buy scan)
    gets a cache hit with zero extra network calls, on top of priming's
    existing job of warming the prev-day cache across runs.
    """
    for leg in leg_pairs:
        token_info = ac.resolve_option_token(instruments, expiry, leg["sell_strike"], leg["option_type"])
        if not token_info:
            print(
                f"[PRIME] Could not resolve token for {leg['sell_strike']}{leg['option_type']} -- "
                "skipping priming for this leg, scan_for_new_signal() will retry the lookup.",
                file=sys.stderr,
            )
            continue

        candles = get_candles_with_cache(smart_api, token_info["token"], prev_day, prev_day_cache, today_start, run_cache)

        if candles:
            print(f"[PRIME] Warmed cache for {token_info['symbol']}: {len(candles)} candles available "
                  f"(prev_day={prev_day.isoformat()}).")
        else:
            print(
                f"[PRIME] No candles available yet for {token_info['symbol']} after priming attempt "
                f"(prev_day={prev_day.isoformat()}) -- likely rate-limited this run. Not retrying further "
                "within this run; the next scheduled run will retry naturally.",
                file=sys.stderr,
            )
