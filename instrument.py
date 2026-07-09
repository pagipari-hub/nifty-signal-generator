"""
ATM strike resolution and the daily-locked leg-pair (sell + hedge strike
per side) structure. Per-option token/symbol lookup itself lives in
angelone_client.resolve_option_token() -- this module is about deciding
*which* strikes to trade today, not about resolving a given strike to a
tradingsymbol.
"""

import angelone_client as ac
from config import STRIKE_LOCK_TIME
from calendar_utils import now_ist


def get_atm_strike(spot_price, step=50):
    return round(spot_price / step) * step


def build_leg_pairs(atm):
    """
    Credit-spread structure (NOT a naked strangle): each side sells one
    strike and buys a further-OTM hedge strike in the SAME option type.
      PE side: SELL ATM-100 PE  /  hedge BUY ATM-400 PE
      CE side: SELL ATM+100 CE  /  hedge BUY ATM+400 CE

    FIX (leg-type bug): the previous version resolved BOTH CE and PE for
    all 4 strikes (8 lookups/run), which is why "No match found for NIFTY
    23850 CE" etc. was spamming logs -- 23850 (ATM-100) is a PE-only leg
    and was never supposed to be looked up as a CE. Each strike now only
    resolves the option type it actually is.
    """
    return [
        {"sell_strike": atm - 100, "hedge_strike": atm - 400, "option_type": "PE"},
        {"sell_strike": atm + 100, "hedge_strike": atm + 400, "option_type": "CE"},
    ]


def get_or_set_daily_strikes(state, smart_api):
    """
    Locks today's leg pairs (sell + hedge strike per side) at/after 9:30 AM,
    same reasoning as before -- just restructured to return leg-pair dicts
    instead of a flat 4-strike list, since strikes now carry sell/hedge/
    option_type together rather than being interpreted generically.
    """
    today_str = now_ist().date().isoformat()

    if state.get("daily_strikes_date") == today_str and state.get("daily_atm") is not None:
        # FIX: rebuild AND rewrite state["daily_strikes"] on every run, not
        # just the first lock of the day. Previously this branch computed
        # leg_pairs in memory for the return value but never touched
        # state["daily_strikes"] on disk -- so a state.json already locked
        # under the old flat 4-strike format (e.g. mid-rollout) would keep
        # showing that stale format all day, only self-correcting at the
        # next day's fresh 9:30 lock. Rewriting it here is free (pure
        # computation from daily_atm, no API call) and makes state.json
        # self-consistent with the current code from the very next run.
        atm = state["daily_atm"]
        leg_pairs = build_leg_pairs(atm)
        state["daily_strikes"] = leg_pairs
        return atm, leg_pairs, None

    if now_ist().time() < STRIKE_LOCK_TIME:
        return None, None, None

    spot_price = ac.fetch_spot_ltp(smart_api)
    if spot_price is None:
        return None, None, None

    atm = get_atm_strike(spot_price)
    leg_pairs = build_leg_pairs(atm)

    state["daily_strikes_date"] = today_str
    state["daily_atm"] = atm
    state["daily_strikes"] = leg_pairs

    return atm, leg_pairs, spot_price
