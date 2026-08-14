"""
Loading and persisting state.json -- the single source of truth for
open_position / pending_signal / daily strike lock, committed back to the
repo by the GitHub Action after every run.
"""

import json
import os

from config import STATE_FILE
from calendar_utils import now_ist


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "open_position": None,
        "last_run": None,
        "heartbeat_date": None,
        "daily_strikes_date": None,
        "daily_atm": None,
        "daily_strikes": None,
        "pending_signal": None,
        "pending_buy_signal": None,
        "open_buy_position": None,
        # NEW (2026-08-13, PDL fallback entry): {"PE": <float or None>,
        # "CE": <float or None>}, written once per day by
        # candle_priming.lock_daily_pdl() on the same run that performs
        # the 9:30 strike lock. Deliberately has NO separate date field
        # of its own -- it is always set at the exact same lock event as
        # daily_strikes/daily_atm, so its validity is tied to
        # daily_strikes_date matching today (same as daily_strikes
        # itself). A leg's value is None when that leg's previous-day
        # data was unavailable this run (rate-limited) or failed the
        # PDL_MIN_PREV_DAY_VOLUME quality gate -- PDL fallback is simply
        # skipped for that leg for the day in either case.
        "daily_pdl": None,
    }


def save_state(state):
    state["last_run"] = now_ist().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
