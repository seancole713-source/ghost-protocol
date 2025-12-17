"""
Daily Top 10 Scheduler
======================

Runs at 6 AM every day (configurable timezone).
Scans market, finds top 10 opportunities, sends Telegram alert.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

LOGGER = logging.getLogger("top_10_scheduler")

# Scheduler state
_SCHEDULER_THREAD: threading.Thread | None = None
_SCHEDULER_STOP = threading.Event()

# Configuration
SCAN_HOUR = 6  # 6 AM
SCAN_MINUTE = 0
TIMEZONE = ZoneInfo("America/Chicago")  # Adjust to your timezone


def start_daily_scheduler():
    """Start the 6 AM daily scanner"""
    global _SCHEDULER_THREAD
    
    if _SCHEDULER_THREAD is None or not _SCHEDULER_THREAD.is_alive():
        _SCHEDULER_STOP.clear()
        _SCHEDULER_THREAD = threading.Thread(
            target=_scheduler_loop,
            name="top-10-scheduler",
            daemon=True
        )
        _SCHEDULER_THREAD.start()
        LOGGER.info(f"🎯 Daily Top 10 Scanner started (runs at {SCAN_HOUR}:00 {TIMEZONE})")


def stop_daily_scheduler():
    """Stop the daily scanner"""
    try:
        _SCHEDULER_STOP.set()
        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
            _SCHEDULER_THREAD.join(timeout=2.0)
            LOGGER.info("Daily Top 10 Scanner stopped")
    except Exception as e:
        LOGGER.error(f"Error stopping scheduler: {e}")


def _scheduler_loop():
    """
    Main scheduler loop.
    
    Checks every minute if it's 6 AM.
    When time matches, runs scan and sends alert.
    """
    LOGGER.info("Daily Top 10 Scheduler loop started")
    last_scan_date = None
    
    while not _SCHEDULER_STOP.is_set():
        try:
            now = datetime.now(TIMEZONE)
            current_date = now.date()
            
            # Check if it's 6 AM
            if (now.hour == SCAN_HOUR and 
                now.minute == SCAN_MINUTE and 
                current_date != last_scan_date):
                
                LOGGER.info(f"🎯 6 AM trigger - Running daily top 10 scan...")
                
                # Run async scan in sync context
                asyncio.run(_run_daily_scan())
                
                last_scan_date = current_date
                LOGGER.info("✅ Daily scan complete")
            
            # Sleep 60 seconds before next check
            time.sleep(60)
        
        except Exception as e:
            LOGGER.error(f"Scheduler loop error: {e}", exc_info=True)
            time.sleep(60)


async def _run_daily_scan():
    """Execute the daily scan and alert"""
    try:
        from core.daily_top_10_scanner import get_scanner
        
        scanner = get_scanner()
        
        # Scan for top 10
        opportunities = await scanner.scan_for_top_10()
        
        if not opportunities:
            LOGGER.warning("No opportunities found in daily scan")
            return
        
        # Save to database
        scanner.save_top_10(opportunities)
        
        # Send Telegram alert
        success = await scanner.send_daily_alert()
        
        if success:
            LOGGER.info(f"✅ Sent daily top 10 alert ({len(opportunities)} opportunities)")
        else:
            LOGGER.error("Failed to send daily alert")
    
    except Exception as e:
        LOGGER.error(f"Failed to run daily scan: {e}", exc_info=True)


def force_scan_now():
    """Manually trigger scan (useful for testing)"""
    LOGGER.info("⚡ Manual scan triggered")
    asyncio.run(_run_daily_scan())
