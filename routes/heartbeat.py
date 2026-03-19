"""
Ghost Protocol — Heartbeat Status Route
=========================================
Background task health monitoring for the cockpit Health tab.
"""

import logging
import os

from fastapi import APIRouter

# ── Also inject wolf_helpers globals (private helper functions + shared state) ─
import wolf_helpers as _wh
globals().update({k: v for k, v in vars(_wh).items() if not k.startswith("__")})
del _wh

# ── Inject all app-config globals into this route module ─────────────────────
# Mirrors wolf_app.py's pattern: provides all module-level constants that route
# handlers reference directly, without needing per-name imports.
import engines.app_config as _ac
globals().update({k: v for k, v in vars(_ac).items() if not k.startswith("__")})
del _ac

router = APIRouter(tags=["heartbeat"])
LOGGER = logging.getLogger("ghost.routes.heartbeat")


@router.get("/api/v3/heartbeat/status")
async def api_v3_heartbeat_status():
    """Get all heartbeat task statuses for the Health tab."""
    try:
        from core.heartbeat import get_all_heartbeats, get_missing_tasks, EXPECTED_INTERVALS
        all_hb = get_all_heartbeats()
        missing = get_missing_tasks()

        # Tasks that run in web mode (before WORKER_MODE gate)
        WEB_MODE_TASKS = {
            "online-calibrator", "news-analysis", "self-improvement",
            "notification-loop", "doctor-cron", "prediction-cycle",
            "outcome-reconciler", "alert-worker", "accuracy-tracker",
            "price-recorder", "autopilot-check",
        }
        _is_worker = os.getenv("WORKER_MODE") == "1"

        tasks = {}
        # Include pulsing tasks
        for name, info in all_hb.items():
            is_web = name in WEB_MODE_TASKS
            tasks[name] = {
                "alive": info["status"] == "alive",
                "status": info["status"],
                "last_pulse": info["last_pulse"],
                "age_s": info["age_s"],
                "web_mode": is_web,
                "runs_here": _is_worker or is_web,
            }
        # Include missing tasks — show ALL, mark worker-only
        for name in missing:
            is_web_task = name in WEB_MODE_TASKS
            runs_here = _is_worker or is_web_task
            tasks[name] = {
                "alive": False,
                "status": "worker-only" if (not _is_worker and not is_web_task) else "never",
                "last_pulse": None,
                "age_s": None,
                "web_mode": is_web_task,
                "runs_here": runs_here,
            }

        alive_count = sum(1 for t in tasks.values() if t["alive"])
        applicable = sum(1 for t in tasks.values() if t["runs_here"])
        total_all = len(tasks)
        return {
            "ok": True,
            "tasks": tasks,
            "alive": alive_count,
            "total": applicable,
            "total_all": total_all,
            "health_pct": round(alive_count / applicable * 100, 1) if applicable else 0,
            "worker_mode": _is_worker,
        }
    except Exception as e:
        LOGGER.error(f"Heartbeat status failed: {e}", exc_info=True)
        return {"ok": False, "tasks": {}, "alive": 0, "total": 0, "error": str(e)}
