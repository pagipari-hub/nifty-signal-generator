"""
ONE-OFF TEST SCRIPT -- safe to run after market close.

Pulls today's REAL candle data from Angel One (today's session is now
historical data, so this works fine even after 3:30 PM) and runs it through
the real compute_indicators()/check_entry_signal() logic from strategy.py,
purely to inspect what would have happened today.

Deliberately does NOT:
  - call send_to_webhook() (no real/paper trade signal gets sent anywhere)
  - touch state.json (no risk of corrupting real position state)
  - touch prev_day_candles.json in a way that conflicts with the real cache
    (uses a separate in-memory dict, never written to disk)

Run with: python test_pull_today_candles.py
Delete this file once you're done testing -- it's not part of the
production pipeline and doesn't need to live in the repo long-term.
"""

import sys
import datetime as dt

import strategy
import angelone_client as ac


def main():
    print("=== Standalone test: pulling today's real candle data ===")
    print("(Read-only -- no webhook calls, no state.json writes)\n")

    smart_api = ac.login()
    instruments = ac.download_instrument_master()

    expiry = strategy.get_current_weekly_expiry()
    print(f"Resolved expiry: {expiry}")

    spot_price = ac.fetch_spot_ltp(smart_api)
    if spot_price is None:
        print("Could not fetch spot price -- aborting test.", file=sys.stderr)
        return
    print(f"Spot price: {spot_price}")

    atm = strategy.get_atm_strike(spot_price)
    strikes_to_check = [atm - 400, atm - 100, atm + 100, atm + 400]
    print(f"ATM: {atm}, checking strikes: {strikes_to_check}\n")

    prev_day = strategy.previous_trading_day(strategy.now_ist().date())
    today_start = dt.datetime.combine(strategy.now_ist().date(), strategy.MARKET_OPEN)
    in_memory_cache = {}  # NOT persisted to disk -- isolated from the real cache file

    for strike in strikes_to_check:
        for opt_type in ["CE", "PE"]:
            token_info = ac.resolve_option_token(instruments, expiry, strike, opt_type)
            if not token_info:
                print(f"[{strike} {opt_type}] No instrument match -- skipping.")
                continue

            candles = strategy.get_candles_with_cache(
                smart_api, token_info["token"], prev_day, in_memory_cache, today_start
            )
            df = strategy.compute_indicators(candles)

            if df is None:
                print(f"[{strike} {opt_type}] Not enough data to compute indicators ({len(candles)} candles fetched).")
                continue

            last = df.iloc[-1]
            signal = strategy.check_entry_signal(df)
            print(
                f"[{strike} {opt_type}] rows_today={len(df)} "
                f"last_close={last['close']:.2f} ema5={last['ema5']:.2f} "
                f"ema25={last['ema25']:.2f} vwap={last['vwap']:.2f} "
                f"entry_signal={signal}"
            )

    print("\n=== Test complete -- no webhook calls made, state.json untouched ===")


if __name__ == "__main__":
    main()
