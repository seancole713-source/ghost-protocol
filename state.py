"""
Ghost Protocol — Shared Application State
==========================================
Central location for all mutable global state that route handlers
and engine modules need to access. Import from here instead of
scattering globals across wolf_app.py.

Usage:
    from state import LATEST_PREDICTIONS, LATEST_PREDICTIONS_LOCK
"""

import threading
import time
from collections import deque
from typing import Any

# ── Prediction State ──────────────────────────────────────────
LATEST_PREDICTIONS: dict[str, dict] = {}
LATEST_PREDICTIONS_LOCK = threading.Lock()

# ── Heartbeat / Health State ──────────────────────────────────
HEARTBEAT_PULSES: dict[str, float] = {}
HEARTBEAT_LOCK = threading.Lock()

# ── News Cache ────────────────────────────────────────────────
NEWS_CACHE: dict[str, Any] = {}
NEWS_CACHE_LOCK = threading.Lock()
NEWS_CACHE_TS: float = 0.0

# ── AI Memory ────────────────────────────────────────────────
AI_MEMORY_STORE: Any = None
AI_MEMORY_RING: list = []

# ── Process Metadata ─────────────────────────────────────────
START_TS: float = time.time()
STATIC_CACHE_BUST: str = str(int(time.time()))

# ── Background Task Tracking ─────────────────────────────────
BACKGROUND_TASKS: dict[str, dict] = {}
BACKGROUND_TASKS_LOCK = threading.Lock()

# ── Notification Loop Status ─────────────────────────────────
NOTIFICATION_LOOP_STATUS: dict[str, Any] = {
    "running": False,
    "last_run": None,
    "last_error": None,
    "run_count": 0,
}
