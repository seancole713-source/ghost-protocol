"""Routes: admin — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse

try:
    import httpx
except ImportError:
    httpx = None

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- 7 endpoints ---

@router.get("/retrain-trigger")
async def retrain_trigger_no_auth():
    """Trigger retraining - no auth required (runs async in background)"""
    import subprocess
    import sys
    import os
    import asyncio
    from datetime import datetime
    
    # Check if already running
    if _RETRAIN_STATUS["running"]:
        return {
            "ok": False,
            "message": "Retraining already in progress",
            "started_at": _RETRAIN_STATUS["started_at"]
        }
    
    # Find script
    script_path = None
    for path in ["scripts/retrain_production_model.py", "retrain_model.py", "scripts/retrain_xgboost.py"]:
        if os.path.exists(path):
            script_path = path
            break
    
    if not script_path:
        return {"ok": False, "error": "No retrain script found", "files": os.listdir(".")[:30]}
    
    # Start background task
    async def run_retraining():
        _RETRAIN_STATUS["running"] = True
        _RETRAIN_STATUS["started_at"] = datetime.now().isoformat()
        
        try:
            result = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            _RETRAIN_STATUS["last_result"] = {
                "ok": result.returncode == 0,
                "script": script_path,
                "output": stdout.decode()[-8000:] if stdout else "",
                "errors": stderr.decode()[-2000:] if stderr else "",
                "return_code": result.returncode,
                "completed_at": datetime.now().isoformat()
            }
        except Exception as e:
            _RETRAIN_STATUS["last_result"] = {
                "ok": False,
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            }
        finally:
            _RETRAIN_STATUS["running"] = False
    
    # Start in background
    asyncio.create_task(run_retraining())
    
    return {
        "ok": True,
        "message": "Retraining started in background",
        "script": script_path,
        "started_at": _RETRAIN_STATUS["started_at"],
        "check_status_at": "/retrain-status"
    }


@router.get("/retrain-status")
async def retrain_status_check():
    """Check status of background retraining"""
    return {
        "running": _RETRAIN_STATUS["running"],
        "started_at": _RETRAIN_STATUS["started_at"],
        "last_result": _RETRAIN_STATUS["last_result"]
    }


@router.get("/api/retrain-now")
async def retrain_now():
    """Trigger model retraining via web request"""
    import subprocess
    import sys
    import os
    
    # Find the retrain script
    possible_paths = [
        "scripts/retrain_production_model.py",
        "retrain_production_model.py",
        "retrain_model.py",
        "scripts/retrain_xgboost.py"
    ]
    
    script_path = None
    for path in possible_paths:
        if os.path.exists(path):
            script_path = path
            break
    
    if not script_path:
        # List what files exist
        files = os.listdir(".")
        scripts_dir = os.listdir("scripts") if os.path.exists("scripts") else []
        return {
            "ok": False, 
            "error": "No retrain script found",
            "root_files": [f for f in files if f.endswith('.py')][:20],
            "scripts_dir": scripts_dir
        }
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.getcwd()
        )
        
        return {
            "ok": result.returncode == 0,
            "script": script_path,
            "stdout": result.stdout[-10000:] if result.stdout else "",
            "stderr": result.stderr[-3000:] if result.stderr else "",
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Timeout after 10 minutes"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/admin/config")
async def api_admin_config():
    """Get current configuration values (safe for display)."""
    try:
        config = {
            "risk": {
                "max_position_pct": float(os.getenv("RISK_MAX_POS_PCT", "5")),
                "max_daily_dd_pct": float(os.getenv("RISK_MAX_DAILY_DD_PCT", "5")),
                "stop_loss_pct": float(os.getenv("RISK_SL_PCT", "3")),
                "take_profit_pct": float(os.getenv("RISK_TP_PCT", "6")),
                "max_drawdown": float(os.getenv("MAX_RISK_DRAWDOWN", "0.05"))
            },
            "trading": {
                "sim_mode": int(os.getenv("SIM_MODE", "0")),
                "active": bool(STATE.get("active", False))
            },
            "providers": {
                "polygon_configured": bool(os.getenv("POLYGON_API_KEY")),
                "alphavantage_configured": bool(os.getenv("ALPHAVANTAGE_API_KEY")),
                "telegram_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN"))
            }
        }
        return {"ok": True, "config": config, "timestamp": time.time()}
    except Exception as e:
        LOGGER.error(f"Admin config failed: {e}")
        return {"ok": False, "error": str(e), "timestamp": time.time()}


@router.post("/api/admin/migrate/outcomes")
async def api_admin_migrate_outcomes():
    """
    Apply the ghost_prediction_outcomes migration.
    Creates the outcomes table and accuracy views for prediction tracking.
    Protected endpoint - requires admin access.
    """
    try:
        from apply_outcome_migration import apply_outcome_migration
        
        LOGGER.info("[ADMIN] Starting ghost_prediction_outcomes migration...")
        apply_outcome_migration()
        
        return {
            "ok": True,
            "message": "Migration applied successfully",
            "table": "ghost_prediction_outcomes",
            "timestamp": time.time()
        }
    except Exception as e:
        LOGGER.error(f"[ADMIN] Migration failed: {e}")
        import traceback
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.time()
        }


@router.get("/api/admin/diagnostics/predictions")
async def api_admin_diagnostics_predictions():
    """
    Diagnostic endpoint to check prediction status and reconciliation readiness.
    Returns counts of predictions by status and age.
    """
    try:
        from core.prediction_store import get_prediction_store
        import time
        
        store = get_prediction_store()
        now = time.time()
        cutoff_48h = now - (48 * 3600)
        cutoff_7d = now - (7 * 86400)
        
        # Try to query production database
        if hasattr(store, 'engine') and store.engine:
            # Using SQLAlchemy (Postgres)
            from sqlalchemy import text
            with store.engine.connect() as conn:
                # Count total predictions
                total = conn.execute(text("SELECT COUNT(*) FROM ghost_predictions")).scalar()
                
                # Count predictions ready for reconciliation (>48h old)
                ready_48h = conn.execute(text(
                    "SELECT COUNT(*) FROM ghost_predictions WHERE run_at < :cutoff"
                ), {"cutoff": cutoff_48h}).scalar()
                
                # Count predictions in last 7 days
                recent_7d = conn.execute(text(
                    "SELECT COUNT(*) FROM ghost_predictions WHERE run_at > :cutoff"
                ), {"cutoff": cutoff_7d}).scalar()
                
                # Count outcomes
                outcomes_total = conn.execute(text("SELECT COUNT(*) FROM ghost_prediction_outcomes")).scalar()
                
                # Get oldest and newest prediction
                oldest = conn.execute(text("SELECT MIN(run_at) FROM ghost_predictions")).scalar()
                newest = conn.execute(text("SELECT MAX(run_at) FROM ghost_predictions")).scalar()
                
                # Check if reconciler ran recently
                from datetime import datetime
                oldest_dt = datetime.fromtimestamp(oldest) if oldest else None
                newest_dt = datetime.fromtimestamp(newest) if newest else None
                
                return {
                    "ok": True,
                    "database": "postgres",
                    "predictions": {
                        "total": total,
                        "ready_for_reconciliation_48h": ready_48h,
                        "recent_7d": recent_7d,
                        "oldest": oldest,
                        "oldest_date": oldest_dt.isoformat() if oldest_dt else None,
                        "newest": newest,
                        "newest_date": newest_dt.isoformat() if newest_dt else None,
                        "age_days": (now - oldest) / 86400 if oldest else 0
                    },
                    "outcomes": {
                        "total": outcomes_total,
                        "reconciliation_rate": f"{outcomes_total}/{ready_48h}" if ready_48h > 0 else "0/0"
                    },
                    "reconciler_status": {
                        "expected_outcomes": ready_48h,
                        "actual_outcomes": outcomes_total,
                        "missing": ready_48h - outcomes_total if ready_48h > 0 else 0,
                        "working": outcomes_total > 0
                    },
                    "timestamp": now
                }
        else:
            return {
                "ok": False,
                "error": "Prediction store not using Postgres engine",
                "timestamp": now
            }
            
    except Exception as e:
        LOGGER.error(f"[ADMIN] Diagnostics failed: {e}")
        import traceback
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.time()
        }


@router.post("/api/admin/reconcile/outcomes")
async def api_admin_reconcile_outcomes():
    """
    Manually trigger outcome reconciliation.
    Finds predictions >48h old and reconciles their outcomes.
    Returns summary of reconciliation results.
    """
    try:
        LOGGER.info("[ADMIN] Manual reconciliation triggered")
        
        from services.outcome_reconciler_v2 import reconcile_outcomes_v2
        
        # Run reconciliation
        results = reconcile_outcomes_v2()
        
        # Get updated counts
        from core.prediction_store import get_prediction_store
        store = get_prediction_store()
        
        if hasattr(store, 'engine') and store.engine:
            from sqlalchemy import text
            with store.engine.connect() as conn:
                outcomes_total = conn.execute(text("SELECT COUNT(*) FROM ghost_prediction_outcomes")).scalar()
                
                # Get sample of reconciled outcomes
                samples = conn.execute(text("""
                    SELECT prediction_id, closed_at, hit_direction, realized_move_pct
                    FROM ghost_prediction_outcomes
                    ORDER BY closed_at DESC
                    LIMIT 10
                """)).fetchall()
                
                sample_data = [
                    {
                        "prediction_id": row[0],
                        "closed_at": row[1],
                        "hit": row[2] == 1,
                        "move_pct": float(row[3]) if row[3] else None
                    }
                    for row in samples
                ]
        else:
            outcomes_total = 0
            sample_data = []
        
        return {
            "ok": True,
            "message": "Reconciliation completed",
            "results": results,
            "outcomes_total": outcomes_total,
            "sample_outcomes": sample_data,
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"[ADMIN] Reconciliation failed: {e}")
        import traceback
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.time()
        }


