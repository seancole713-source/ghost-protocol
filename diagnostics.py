import time

from db import _ERRORS  # type: ignore


async def health_snapshot(extra=None):
    try:
        err_cnt = len(_ERRORS)
    except Exception:
        err_cnt = 0
    return {"ok": True, "timestamp": time.time(), "error_count": err_cnt, **(extra or {})}


async def quick_probe():
    return {"ok": True}


async def summary():
    # Provide a concrete diagnostic summary with invariant failures surfaced
    try:
        from state_manager import is_active  # local import to avoid cycles

        active = await is_active()
    except Exception:
        active = None
    try:
        err_cnt = len(_ERRORS)
    except Exception:
        err_cnt = 0
    # Pull invariant_fail entries from main.LOG_RING if available
    invariant_events = []
    try:
        import main  # type: ignore

        ring = getattr(main, "LOG_RING", [])
        for e in reversed(ring[-200:]):
            if e.get("kind") == "invariant_fail":
                invariant_events.append(e)
            if len(invariant_events) >= 5:
                break
    except Exception:
        pass
    # Expose current tick_id if available for CI diagnostics
    tick_id = None
    try:
        import main  # type: ignore

        last = getattr(main, "LAST_SNAPSHOT", None)
        if isinstance(last, dict):
            tick_id = last.get("tick_id")
    except Exception:
        pass
    return {
        "ok": True,
        "timestamp": time.time(),
        "active": active,
        "error_count": err_cnt,
        "invariants": invariant_events,
        "tick_id": tick_id,
    }
