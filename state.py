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
    }


def save_state(state):
    state["last_run"] = now_ist().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
