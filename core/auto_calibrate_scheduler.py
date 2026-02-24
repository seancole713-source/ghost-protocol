"""
Auto-Calibration Weekly Scheduler
Runs every Sunday at 5:00 AM CT to find optimal V3 strategies.

Ghost automatically tunes itself based on recent market data.
"""

import threading
import time
import logging
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

# Configuration
CALIBRATION_DAY = 6  # Sunday (0=Monday, 6=Sunday)
CALIBRATION_HOUR = 5  # 5:00 AM CT
CALIBRATION_MINUTE = 0
TIMEZONE = pytz.timezone('America/Chicago')  # CT

_scheduler_thread = None
_running = False


def get_next_calibration_time() -> datetime:
    """Calculate the next calibration run time (Sunday 5AM CT)"""
    now = datetime.now(TIMEZONE)
    
    # Find next Sunday
    days_until_sunday = (CALIBRATION_DAY - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour >= CALIBRATION_HOUR:
        # It's Sunday but past 5AM, wait until next Sunday
        days_until_sunday = 7
    
    next_run = now.replace(
        hour=CALIBRATION_HOUR,
        minute=CALIBRATION_MINUTE,
        second=0,
        microsecond=0
    ) + timedelta(days=days_until_sunday)
    
    return next_run


def run_weekly_calibration():
    """Execute the weekly auto-calibration"""
    try:
        from core.auto_calibrate import run_calibration
        
        logger.info("🔄 [AUTO-CALIBRATE] Starting weekly calibration...")
        
        # LEARNING FIX: Ghost now auto-applies validated strategies
        # Controlled by env vars for safety — can be disabled without redeploy
        import os
        auto_update = os.getenv("CALIBRATION_AUTO_UPDATE", "1") == "1"
        dry_run = os.getenv("CALIBRATION_DRY_RUN", "0") == "1"
        
        logger.info(f"🔄 [AUTO-CALIBRATE] Mode: auto_update={auto_update}, dry_run={dry_run}")
        
        result = run_calibration(
            test_crypto=True,
            test_stocks=True,
            auto_update=auto_update,  # Live updates (disable: CALIBRATION_AUTO_UPDATE=0)
            dry_run=dry_run            # Real writes (enable dry run: CALIBRATION_DRY_RUN=1)
        )
        
        # Log summary
        total_changes = (
            len(result['changes']['added']) +
            len(result['changes']['removed']) +
            len(result['changes']['changed'])
        )
        
        logger.info(f"🔄 [AUTO-CALIBRATE] Complete! "
                    f"Validated: {len(result['validated'])}, "
                    f"Changes: {total_changes}")
        
        if total_changes > 0:
            logger.info(f"🔄 [AUTO-CALIBRATE] Added: {list(result['changes']['added'].keys())}")
            logger.info(f"🔄 [AUTO-CALIBRATE] Removed: {list(result['changes']['removed'].keys())}")
            logger.info(f"🔄 [AUTO-CALIBRATE] Changed: {list(result['changes']['changed'].keys())}")
        
        return result
        
    except Exception as e:
        logger.error(f"🔄 [AUTO-CALIBRATE] Failed: {e}", exc_info=True)
        return None


def _scheduler_loop():
    """Background thread that runs calibration on schedule"""
    global _running
    
    logger.info("🔄 [AUTO-CALIBRATE] Scheduler thread started")
    
    while _running:
        try:
            next_run = get_next_calibration_time()
            now = datetime.now(TIMEZONE)
            
            seconds_until_run = (next_run - now).total_seconds()
            
            logger.info(f"🔄 [AUTO-CALIBRATE] Next run: {next_run.strftime('%Y-%m-%d %H:%M %Z')} "
                        f"({seconds_until_run/3600:.1f} hours)")
            
            # Sleep in 1-hour chunks to allow for graceful shutdown
            while _running and seconds_until_run > 0:
                sleep_time = min(3600, seconds_until_run)  # Max 1 hour
                time.sleep(sleep_time)
                seconds_until_run -= sleep_time
            
            if _running:
                # Time to run calibration!
                run_weekly_calibration()
            
        except Exception as e:
            logger.error(f"🔄 [AUTO-CALIBRATE] Scheduler error: {e}", exc_info=True)
            time.sleep(3600)  # Wait 1 hour on error
    
    logger.info("🔄 [AUTO-CALIBRATE] Scheduler thread stopped")


def start_weekly_calibration_scheduler():
    """Start the weekly calibration scheduler"""
    global _scheduler_thread, _running
    
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        logger.warning("🔄 [AUTO-CALIBRATE] Scheduler already running")
        return
    
    _running = True
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        daemon=True,
        name="auto-calibrate-scheduler"
    )
    _scheduler_thread.start()
    
    next_run = get_next_calibration_time()
    logger.info(f"🔄 [AUTO-CALIBRATE] Scheduler started. Next run: {next_run.strftime('%Y-%m-%d %H:%M %Z')}")


def stop_weekly_calibration_scheduler():
    """Stop the weekly calibration scheduler"""
    global _running
    _running = False
    logger.info("🔄 [AUTO-CALIBRATE] Scheduler stopping...")


def run_calibration_now():
    """Manually trigger calibration (for testing or API calls)"""
    return run_weekly_calibration()


if __name__ == "__main__":
    # Test: Run calibration immediately
    logging.basicConfig(level=logging.INFO)
    print("Testing auto-calibration...")
    result = run_calibration_now()
    if result:
        print(f"Found {len(result['validated'])} validated strategies")
