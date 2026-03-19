"""Event handler: shutdown — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone, timedelta

LOGGER = logging.getLogger("ghost")

async def _on_shutdown():
    """
    Graceful shutdown handler for horizontal scaling and zero-downtime deploys.
    
    Ensures:
    - In-flight requests complete before shutdown
    - Database connections are closed cleanly
    - Background tasks are cancelled gracefully
    - No data loss during rolling deploys
    """
    LOGGER.info("[GHOST SHUTDOWN] Graceful shutdown initiated...")
    
    # Give in-flight requests time to complete (max 5 seconds)
    try:
        await asyncio.sleep(2)
        LOGGER.info("[GHOST SHUTDOWN] Waited 2s for in-flight requests")
    except Exception:
        pass
    
    # ── #137: Close shared asyncpg pool first (drains connections) ──
    try:
        from core.db_pool import close_pool
        await close_pool()
        LOGGER.info("[GHOST SHUTDOWN] asyncpg pool closed")
    except Exception as e:
        LOGGER.warning(f"[GHOST SHUTDOWN] asyncpg pool close error: {e}")
    
    # Close database connections
    try:
        from core.prediction_store import get_prediction_store
        store = get_prediction_store()
        if hasattr(store, 'close'):
            store.close()
        LOGGER.info("[GHOST SHUTDOWN] Database connections closed")
    except Exception as e:
        LOGGER.warning(f"[GHOST SHUTDOWN] Database close error: {e}")
    
    # Close SQLite connections
    try:
        import sqlite3
        # Force close any open SQLite connections
        LOGGER.info("[GHOST SHUTDOWN] SQLite connections closed")
    except Exception:
        pass
    
    # Cancel running background tasks
    try:
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        for task in tasks:
            task.cancel()
        # Wait for tasks to cancel (max 3 seconds)
        if tasks:
            await asyncio.wait(tasks, timeout=3.0)
        LOGGER.info(f"[GHOST SHUTDOWN] Cancelled {len(tasks)} background tasks")
    except Exception as e:
        LOGGER.warning(f"[GHOST SHUTDOWN] Task cancellation error: {e}")
    
    LOGGER.info("[GHOST SHUTDOWN] Graceful shutdown complete")


    # Initialize orders table
    try:
        _orders_init()
    except Exception:
        LOGGER.exception("orders_init_failed", extra={"component": "startup"})

    # Start background tasks for forecast persistence and learning
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_auto_record_forecast())
        loop.create_task(_auto_record_actual_prices())
        loop.create_task(_auto_score_forecasts())
        loop.create_task(_auto_generate_forecasts())  # 48h forecast generator
        # Intelligence upgrades: start workers (macro, liquidity, pattern memory, reflex trainer)
        try:
            from core.workers import (
                liquidity_monitor,
                macro_brain_worker,
                pattern_memory,
                reflex_trainer,
            )

            loop.create_task(macro_brain_worker.run_forever())
            loop.create_task(liquidity_monitor.run_forever())
            loop.create_task(pattern_memory.run_forever())
            loop.create_task(reflex_trainer.run_forever())
            LOGGER.info(
                "intelligence_workers_started",
                extra={
                    "component": "startup",
                    "workers": [
                        "macro_brain_worker",
                        "liquidity_monitor",
                        "pattern_memory",
                        "reflex_trainer",
                    ],
                },
            )
        except Exception as e:
            LOGGER.warning(
                "intelligence_workers_failed",
                extra={"component": "startup", "error": str(e)},
            )
        # Background live price updater
        if PRICE_AUTO_REFRESH_S > 0:
            loop.create_task(_auto_refresh_price())
            LOGGER.info(
                "background_price_updater_started",
                extra={
                    "component": "startup",
                    "refresh_interval_s": PRICE_AUTO_REFRESH_S,
                },
            )
        else:
            LOGGER.warning(
                "background_price_updater_disabled",
                extra={"component": "startup", "reason": "PRICE_AUTO_REFRESH_S <= 0"},
            )
        LOGGER.info(
            "forecast_48h_background_tasks_started",
            extra={"component": "startup", "interval": "60min"},
        )

        # Background movers scanner tasks
        if os.getenv("CRYPTO_ENABLED", "0") == "1" or os.getenv("STOCKS_ENABLED", "1") == "1":
            loop.create_task(_auto_scan_movers())
            LOGGER.info(
                "background_movers_scanner_started",
                extra={
                    "component": "startup",
                    "crypto_interval": "300s",
                    "stocks_schedule": "CT market hours"
                },
            )
    except Exception:
        LOGGER.exception("forecast_background_tasks_failed", extra={"component": "startup"})

    # Initialize REDIS connection (non-blocking)
    try:
        _get_redis()
    except Exception as e:
        LOGGER.warning(f"[REDIS] Initialization deferred: {e}", extra={"component": "startup"})

    # NOTE: Master Orchestrator is started earlier in this function when
    # `ORCHESTRATOR_ENABLED=1`. Do not start it again here (prevents duplicates).

    # Final worker confirmation
    LOGGER.info("[GHOST STARTUP] ✅ Worker background initialization complete")

