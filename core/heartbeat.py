"""
Ghost Protocol — Heartbeat Registry
════════════════════════════════════

Central registry where every background task/thread/loop pings a
heartbeat on each iteration. The integrity system then checks
whether any heartbeat is stale → that task crashed silently.

Usage in any background loop:
    from core.heartbeat import pulse

    while not stop_event.is_set():
        pulse("outcome-reconciler")  # Call once per loop iteration
        ... do work ...
        stop_event.wait(300)

The integrity check reads the registry via get_all_heartbeats()
and flags any task that hasn't pulsed within its expected interval.

Created: March 12, 2026
"""

import time
import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()

# task_name -> {"last_pulse": float, "interval_s": float, "extra": dict}
_heartbeats: Dict[str, Dict[str, Any]] = {}

# Expected intervals — if a task doesn't pulse within 2x this, it's considered dead
# These are the LOOP intervals of each background task
EXPECTED_INTERVALS: Dict[str, float] = {
    # Daemon threads
    "autosave-worker":       600,    # WOLF_AUTOSAVE_S, typically 5-10 min
    "alert-worker":          60,     # Processes queue, should be alive every minute
    "open-close-scheduler":  60,     # Checks every 30s
    "outcome-reconciler":    3600 + 60,  # Every 1 hour (V2 reconciler)
    "accuracy-tracker":      600,    # Every 5 min
    "online-calibrator":     21600 + 600,  # Every 6 hours
    "price-recorder":        600,    # Every 5 min (records prediction prices)

    # Asyncio tasks
    "notification-loop":     3600 + 300,   # Runs on schedule (1h between checks)
    "doctor-cron":           3600 + 300,   # Every hour
    "news-analysis":         1800 + 300,   # Periodic
    "self-improvement":      3600 + 300,   # Periodic
    "vip-scanner":           600,          # Every few minutes
    "premarket-scanner":     600,          # Periodic
    "full-scanner":          1800,         # Periodic
    "money-game":            3600 + 300,   # Periodic
    "guardian-oracle":       600,          # Monitor loop
    "prediction-cycle":      600,          # Multi-prediction main loop
    "autopilot-check":       600,          # Accuracy autopilot check (every 5 min)
    # NOTE: retraining and reevaluation are on-demand API calls, NOT background loops
    # They were removed from heartbeat tracking to avoid false "never pulsed" alarms
}

# Grace period multiplier: task is "stale" at 2x interval, "dead" at 4x
STALE_MULTIPLIER = 2.0
DEAD_MULTIPLIER = 4.0


def pulse(task_name: str, interval_s: Optional[float] = None, **extra):
    """
    Record a heartbeat for a background task.
    Call this once per loop iteration in any background task.

    Args:
        task_name: Unique name for the task (matches thread name if possible)
        interval_s: Expected loop interval in seconds (auto-detected if registered)
        **extra: Any additional metadata to store (e.g., items_processed=5)
    """
    with _lock:
        _heartbeats[task_name] = {
            "last_pulse": time.time(),
            "interval_s": interval_s or EXPECTED_INTERVALS.get(task_name, 600),
            "extra": extra,
        }


def get_heartbeat(task_name: str) -> Optional[Dict[str, Any]]:
    """Get heartbeat info for a specific task."""
    with _lock:
        hb = _heartbeats.get(task_name)
        if hb is None:
            return None
        return {
            **hb,
            "age_s": time.time() - hb["last_pulse"],
            "status": _classify(hb),
        }


def get_all_heartbeats() -> Dict[str, Dict[str, Any]]:
    """Get all heartbeats with status classification."""
    now = time.time()
    result = {}
    with _lock:
        for name, hb in _heartbeats.items():
            age = now - hb["last_pulse"]
            interval = hb.get("interval_s", 600)
            result[name] = {
                "last_pulse": hb["last_pulse"],
                "age_s": round(age, 1),
                "interval_s": interval,
                "status": _classify(hb, now=now),
                "extra": hb.get("extra", {}),
            }
    return result


def get_registered_tasks() -> list:
    """Get list of all tasks that SHOULD be pulsing."""
    return sorted(EXPECTED_INTERVALS.keys())


def get_missing_tasks() -> list:
    """Get tasks that are registered but have NEVER pulsed."""
    with _lock:
        pulsed = set(_heartbeats.keys())
    return sorted(set(EXPECTED_INTERVALS.keys()) - pulsed)


def get_stale_tasks() -> list:
    """Get tasks that pulsed before but are now stale."""
    now = time.time()
    stale = []
    with _lock:
        for name, hb in _heartbeats.items():
            status = _classify(hb, now=now)
            if status in ("stale", "dead"):
                stale.append(name)
    return sorted(stale)


def _classify(hb: Dict, now: Optional[float] = None) -> str:
    """Classify a heartbeat as alive/stale/dead."""
    if now is None:
        now = time.time()
    age = now - hb["last_pulse"]
    interval = hb.get("interval_s", 600)

    if age <= interval * STALE_MULTIPLIER:
        return "alive"
    elif age <= interval * DEAD_MULTIPLIER:
        return "stale"
    else:
        return "dead"
