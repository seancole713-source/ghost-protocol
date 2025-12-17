#!/usr/bin/env python3
"""
Ghost Protocol - Cascade Scheduler
===================================

Monitors active cascades and triggers updates at the right times:
- T+24h: Re-evaluation with new data
- T+42h: Final 6h high-accuracy call
- T+48h: Outcome evaluation

Runs continuously in background thread, checking every 10 minutes.
"""

import asyncio
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger("core.cascade_scheduler")

# Will be set by wolf_app.py
CASCADE_PREDICTOR = None

# Tracking state
_CASCADE_WORKER: threading.Thread | None = None
_CASCADE_STOP = threading.Event()
_LAST_CHECK = 0

# Check interval (10 minutes)
CHECK_INTERVAL = 600


def start_cascade_scheduler():
    """Start the cascade scheduler worker"""
    global _CASCADE_WORKER
    if _CASCADE_WORKER is None or not _CASCADE_WORKER.is_alive():
        _CASCADE_STOP.clear()
        _CASCADE_WORKER = threading.Thread(
            target=_cascade_loop, name="cascade-scheduler", daemon=True
        )
        _CASCADE_WORKER.start()
        LOGGER.info("📊 Cascade scheduler started (checking every 10 minutes)")
        print("[CASCADE SCHEDULER] Started - monitoring active cascades")


def stop_cascade_scheduler():
    """Stop the cascade scheduler"""
    try:
        _CASCADE_STOP.set()
        if _CASCADE_WORKER and _CASCADE_WORKER.is_alive():
            _CASCADE_WORKER.join(timeout=2.0)
            LOGGER.info("Cascade scheduler stopped")
    except Exception as e:
        LOGGER.error(f"Error stopping cascade scheduler: {e}")


def _cascade_loop():
    """
    Main scheduler loop - checks for pending cascade updates.
    
    Runs every 10 minutes and:
    1. Finds cascades needing 24h update
    2. Finds cascades needing 6h final
    3. Finds cascades needing 48h evaluation
    """
    global _LAST_CHECK
    
    LOGGER.info("Cascade scheduler loop started")
    
    while not _CASCADE_STOP.is_set():
        try:
            now = time.time()
            
            # Check every 10 minutes
            if now - _LAST_CHECK >= CHECK_INTERVAL:
                LOGGER.debug("Checking for pending cascade updates...")
                _check_pending_cascades()
                _LAST_CHECK = now
            
            # Sleep for 60 seconds between checks
            time.sleep(60)
            
        except Exception as e:
            LOGGER.error(f"Cascade scheduler loop error: {e}", exc_info=True)
            time.sleep(60)  # Sleep on error to avoid tight loop


def _check_pending_cascades():
    """Check database for cascades needing updates"""
    if not CASCADE_PREDICTOR:
        LOGGER.warning("Cascade predictor not configured, skipping checks")
        return
    
    try:
        conn = sqlite3.connect(str(CASCADE_PREDICTOR.db_path))
        conn.row_factory = sqlite3.Row
        
        now = int(time.time())
        
        # Find cascades needing 24h update
        # (created 24h ago, no h24 update sent yet)
        cursor = conn.execute("""
            SELECT cascade_id, symbol, created_at
            FROM prediction_cascades
            WHERE evaluated_at IS NULL
                AND h24_sent_at IS NULL
                AND created_at <= ?
        """, (now - (24 * 3600),))
        
        cascades_24h = cursor.fetchall()
        
        # Find cascades needing 6h final
        # (created 42h ago, no h6 update sent yet)
        cursor = conn.execute("""
            SELECT cascade_id, symbol, created_at
            FROM prediction_cascades
            WHERE evaluated_at IS NULL
                AND h6_sent_at IS NULL
                AND created_at <= ?
        """, (now - (42 * 3600),))
        
        cascades_6h = cursor.fetchall()
        
        # Find cascades needing evaluation
        # (created 48h ago, not evaluated yet)
        cursor = conn.execute("""
            SELECT cascade_id, symbol, created_at
            FROM prediction_cascades
            WHERE evaluated_at IS NULL
                AND h6_sent_at IS NOT NULL
                AND created_at <= ?
        """, (now - (48 * 3600),))
        
        cascades_eval = cursor.fetchall()
        
        conn.close()
        
        # Process 24h updates
        for cascade in cascades_24h:
            cascade_id = cascade['cascade_id']
            symbol = cascade['symbol']
            LOGGER.info(f"[CASCADE] Triggering 24h update for {symbol} ({cascade_id})")
            
            try:
                # Run update in asyncio context
                asyncio.run(CASCADE_PREDICTOR.update_cascade_24h(cascade_id))
            except Exception as e:
                LOGGER.error(f"[CASCADE] Failed 24h update for {cascade_id}: {e}", exc_info=True)
        
        # Process 6h finals
        for cascade in cascades_6h:
            cascade_id = cascade['cascade_id']
            symbol = cascade['symbol']
            LOGGER.info(f"[CASCADE] Triggering 6h final for {symbol} ({cascade_id})")
            
            try:
                asyncio.run(CASCADE_PREDICTOR.finalize_cascade_6h(cascade_id))
            except Exception as e:
                LOGGER.error(f"[CASCADE] Failed 6h final for {cascade_id}: {e}", exc_info=True)
        
        # Process evaluations
        for cascade in cascades_eval:
            cascade_id = cascade['cascade_id']
            symbol = cascade['symbol']
            LOGGER.info(f"[CASCADE] Triggering evaluation for {symbol} ({cascade_id})")
            
            try:
                asyncio.run(CASCADE_PREDICTOR.evaluate_cascade(cascade_id))
            except Exception as e:
                LOGGER.error(f"[CASCADE] Failed evaluation for {cascade_id}: {e}", exc_info=True)
        
        if cascades_24h or cascades_6h or cascades_eval:
            LOGGER.info(
                f"[CASCADE] Processed updates: "
                f"{len(cascades_24h)} x 24h, "
                f"{len(cascades_6h)} x 6h, "
                f"{len(cascades_eval)} x eval"
            )
        else:
            LOGGER.debug("No pending cascade updates")
        
    except Exception as e:
        LOGGER.error(f"Failed to check pending cascades: {e}", exc_info=True)


def force_check():
    """
    Force an immediate check for pending cascades.
    
    Useful for testing or manual triggering.
    """
    global _LAST_CHECK
    _LAST_CHECK = 0  # Reset to force check on next loop iteration
    LOGGER.info("Forced cascade check scheduled")
