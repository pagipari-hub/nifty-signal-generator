"""
File-based lock that stops a new run from starting while a previous run
is still active (see config.py's LOCK_STALE_SECONDS comment for the
rate-limit incident this was written to fix).
"""

import os
import sys
import time

from config import LOCK_FILE, LOCK_STALE_SECONDS
from calendar_utils import now_ist


def acquire_run_lock():
    """
    Returns True if the lock was acquired (safe to proceed). Returns False
    if a fresh lock already exists, meaning another run is still active.
    A stale lock (older than LOCK_STALE_SECONDS -- i.e. from a run that
    crashed without cleaning up) is cleared and re-acquired.
    """
    if os.path.exists(LOCK_FILE):
        age = time.time() - os.path.getmtime(LOCK_FILE)
        if age < LOCK_STALE_SECONDS:
            return False
        print(
            f"Stale lock file found (age={age:.0f}s) -- previous run likely "
            "crashed without cleaning up. Clearing it and proceeding.",
            file=sys.stderr,
        )

    with open(LOCK_FILE, "w") as f:
        f.write(f"{os.getpid()} {now_ist().isoformat()}")
    return True


def release_run_lock():
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass
