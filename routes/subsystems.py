"""
Ghost Protocol — V4 Subsystems Route
======================================
Full Ghost subsystem inventory — brains, memory, intelligence, tasks.
Used by the cockpit Health + Brain tabs.
"""

import logging
import os
import time as _t

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

router = APIRouter(tags=["subsystems"])
LOGGER = logging.getLogger("ghost.routes.subsystems")


@router.get("/api/v4/subsystems")
async def api_v4_subsystems():
    """
    Full Ghost subsystem inventory — every module, brain, memory, and task.
    Used by the cockpit Health + Brain tabs to show everything Ghost has.
    """
    try:
        _is_worker = os.getenv("WORKER_MODE") == "1"

        # ── 1. BRAIN MODULES ─────────────────────────────────────────
        brains = []

        # GhostBrain (main prediction brain)
        try:
            from core.ghost_brain import GhostBrain
            brains.append({"name": "Ghost Brain", "key": "ghost_brain", "active": True,
                           "desc": "Core prediction engine — directional forecasts"})
        except Exception:
            brains.append({"name": "Ghost Brain", "key": "ghost_brain", "active": False,
                           "desc": "Core prediction engine (import failed)"})

        # LearningBrain
        try:
            from core.ghost_learning_brain import apply_inversion, get_inverted_symbols
            _lb_inverted = get_inverted_symbols()
            brains.append({"name": "Learning Brain", "key": "learning_brain", "active": True,
                           "desc": f"Self-correction — {len(_lb_inverted)} symbols inverted"})
        except Exception:
            brains.append({"name": "Learning Brain", "key": "learning_brain", "active": False,
                           "desc": "Self-correction engine (import failed)"})

        # NewsBrain
        try:
            from core.intelligence_hub import get_news_brain_cache
            cache, cache_ts = get_news_brain_cache()
            has_data = bool(cache and (cache.get("major_events") or cache.get("predictions_at_risk")))
            age_min = round((_t.time() - cache_ts) / 60, 1) if cache_ts > 0 else -1
            brains.append({"name": "News Brain", "key": "news_brain", "active": has_data,
                           "desc": f"News sentiment analysis — cache {age_min}m old" if has_data else "News sentiment (no cache)"})
        except Exception:
            brains.append({"name": "News Brain", "key": "news_brain", "active": False,
                           "desc": "News sentiment analysis (not loaded)"})

        # OpusBrain (GPT-4 powered)
        try:
            from core.intelligence.opus_brain import OpusBrain
            brains.append({"name": "Opus Brain", "key": "opus_brain", "active": True,
                           "desc": "GPT-4 powered deep analysis"})
        except Exception:
            brains.append({"name": "Opus Brain", "key": "opus_brain", "active": False,
                           "desc": "GPT-4 deep analysis (not available)"})

        # ── 2. MEMORY SYSTEMS ────────────────────────────────────────
        memory = []

        # AI Memory — access globals from wolf_app at call time
        try:
            import wolf_app as _wa
            mem_active = _wa.AI_MEMORY_STORE is not None
            ring_size = len(_wa.AI_MEMORY_RING) if _wa.AI_MEMORY_RING else 0
            memory.append({"name": "AI Memory", "key": "ai_memory", "active": mem_active,
                           "desc": f"Long-term memory store — {ring_size} entries in ring" if mem_active else "Not initialized"})
        except Exception:
            memory.append({"name": "AI Memory", "key": "ai_memory", "active": False, "desc": "Not available"})

        # Market Memory (prediction store / PostgreSQL)
        try:
            from core.db_pool import get_sync_connection
            with get_sync_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM ghost_predictions")
                pred_count = cur.fetchone()[0]
            memory.append({"name": "Market Memory", "key": "market_memory", "active": True,
                           "desc": f"PostgreSQL — {pred_count} predictions stored"})
        except Exception:
            memory.append({"name": "Market Memory", "key": "market_memory", "active": False,
                           "desc": "PostgreSQL connection unavailable"})

        # Paper Tracker (ephemeral trades)
        try:
            from core.paper_tracker import get_paper_tracker
            tracker = get_paper_tracker()
            stats = tracker.get_stats() or {}
            total = stats.get("total_trades", 0)
            memory.append({"name": "Paper Tracker", "key": "paper_tracker", "active": True,
                           "desc": f"Paper trading — {total} trades tracked"})
        except Exception:
            memory.append({"name": "Paper Tracker", "key": "paper_tracker", "active": False,
                           "desc": "Paper tracker unavailable"})

        # ── 3. INTELLIGENCE HUB ──────────────────────────────────────
        intel_systems = []
        try:
            from core.intelligence_hub import get_intelligence_hub
            hub = get_intelligence_hub()
            status = hub.get_status()
            subsystem_names = [
                ("ensemble", "Multi-model consensus"),
                ("calibrator", "Confidence calibration"),
                ("trust_ladder", "Symbol trust scoring"),
                ("quality_gate", "Quality filtering"),
                ("killswitch", "Emergency kill switch"),
                ("vwap", "Volume-weighted analysis"),
                ("feed_fusion", "Multi-feed data fusion"),
                ("regime_detector", "Market regime detection"),
                ("self_improvement", "Self-improvement engine"),
            ]
            for name, desc in subsystem_names:
                loaded = status.get(f"{name}_loaded", False)
                intel_systems.append({"name": name.replace("_", " ").title(), "key": name, "active": loaded, "desc": desc})
        except Exception:
            pass

        # ── 4. BACKGROUND TASKS (full inventory) ─────────────────────
        bg_tasks = []
        WEB_TASKS = {
            "online-calibrator", "news-analysis", "self-improvement",
            "notification-loop", "doctor-cron", "prediction-cycle",
            "outcome-reconciler", "alert-worker", "accuracy-tracker",
            "price-recorder",
        }
        try:
            from core.heartbeat import get_all_heartbeats, get_missing_tasks, EXPECTED_INTERVALS
            all_hb = get_all_heartbeats()
            missing = get_missing_tasks()

            # Pulsing tasks
            for name, info in all_hb.items():
                is_web = name in WEB_TASKS
                runs_here = _is_worker or is_web
                bg_tasks.append({
                    "name": name, "status": info["status"],
                    "last_pulse": info["last_pulse"], "age_s": info["age_s"],
                    "category": "web" if is_web else "worker",
                    "runs_here": runs_here,
                })
            # Missing (never pulsed) — include ALL, mark which ones can't run here
            for name in missing:
                is_web = name in WEB_TASKS
                runs_here = _is_worker or is_web
                bg_tasks.append({
                    "name": name, "status": "never",
                    "last_pulse": None, "age_s": None,
                    "category": "web" if is_web else "worker",
                    "runs_here": runs_here,
                })
        except Exception as e:
            LOGGER.warning(f"Subsystems: heartbeat load failed: {e}")

        # ── 5. MORNING HEALTH CHECK (System Doctor) ──────────────────
        morning_health = None
        try:
            from core.system_doctor import run_system_doctor
            report = run_system_doctor()
            morning_health = {
                "overall": report.get("overall", "UNKNOWN"),
                "passed": report.get("passed", 0),
                "failed": report.get("failed", 0),
                "checks": report.get("checks", []),
                "timestamp": report.get("timestamp"),
            }
        except Exception as e:
            morning_health = {"overall": "ERROR", "error": str(e), "checks": []}

        # ── 6. SUMMARY COUNTS ────────────────────────────────────────
        brains_active = sum(1 for b in brains if b["active"])
        memory_active = sum(1 for m in memory if m["active"])
        intel_active = sum(1 for s in intel_systems if s["active"])
        tasks_alive = sum(1 for t in bg_tasks if t["status"] == "alive")
        tasks_applicable = sum(1 for t in bg_tasks if t["runs_here"])
        doctor_pass = morning_health.get("overall") == "PASS" if morning_health else False

        return {
            "ok": True,
            "worker_mode": _is_worker,
            "brains": brains,
            "brains_active": brains_active,
            "brains_total": len(brains),
            "memory": memory,
            "memory_active": memory_active,
            "memory_total": len(memory),
            "intelligence": intel_systems,
            "intel_active": intel_active,
            "intel_total": len(intel_systems),
            "tasks": bg_tasks,
            "tasks_alive": tasks_alive,
            "tasks_total": len(bg_tasks),
            "tasks_applicable": tasks_applicable,
            "morning_health": morning_health,
            "doctor_pass": doctor_pass,
            "summary": {
                "brains": f"{brains_active}/{len(brains)}",
                "memory": f"{memory_active}/{len(memory)}",
                "intelligence": f"{intel_active}/{len(intel_systems)}",
                "tasks": f"{tasks_alive}/{tasks_applicable} alive",
                "doctor": morning_health.get("overall", "?") if morning_health else "?",
            },
            "ts": _t.time(),
        }
    except Exception as e:
        LOGGER.error(f"Subsystems endpoint failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}
