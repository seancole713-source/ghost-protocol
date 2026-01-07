"""
Simple cron-based scheduler for 6 AM morning prophecy
Uses APScheduler with async support
"""

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_SCHEDULER = None


async def send_morning_prophecy():
    """Send morning prophecy at 6 AM"""
    try:
        logger.info("🔮 6 AM TRIGGER: Sending morning prophecy...")
        
        from core.daily_top_10_scanner import DailyTop10Scanner
        from core.guardian_oracle import get_guardian_oracle
        from core.telegram_hunter import send_telegram_message
        
        # Scan for opportunities
        scanner = DailyTop10Scanner()
        top_10 = await scanner.scan_for_top_10()
        
        if not top_10:
            message = "🔮 Good morning! No high-quality opportunities today. Ghost is watching."
            send_telegram_message(message)
            logger.info("No opportunities found")
            return
        
        # Format with Guardian Oracle
        guardian = get_guardian_oracle()
        prophecy = await guardian.morning_prophecy(top_10, position_size=100.0)
        
        # Send to Telegram
        send_telegram_message(prophecy)
        
        logger.info(f"✅ Morning prophecy sent: {len(top_10)} opportunities")
        
    except Exception as e:
        logger.exception(f"Failed to send morning prophecy: {e}")


def start_cron_scheduler():
    """Start cron scheduler for 6 AM morning prophecy"""
    global _SCHEDULER
    
    if _SCHEDULER is not None:
        logger.warning("Cron scheduler already running")
        return
    
    try:
        _SCHEDULER = AsyncIOScheduler()
        
        # Schedule morning prophecy at 6:00 AM America/Chicago time
        _SCHEDULER.add_job(
            send_morning_prophecy,
            trigger=CronTrigger(hour=6, minute=0, timezone='America/Chicago'),
            id='morning_prophecy',
            name='Morning Prophecy 6 AM',
            replace_existing=True,
            misfire_grace_time=3600  # Allow up to 1 hour late execution
        )
        
        _SCHEDULER.start()
        
        logger.info("✅ Cron Scheduler started: Morning prophecy at 6:00 AM CT daily")
        
        # Log next scheduled run
        job = _SCHEDULER.get_job('morning_prophecy')
        if job and job.next_run_time:
            logger.info(f"📅 Next morning prophecy: {job.next_run_time}")
        
    except Exception as e:
        logger.exception(f"Failed to start cron scheduler: {e}")
        _SCHEDULER = None


def stop_cron_scheduler():
    """Stop cron scheduler"""
    global _SCHEDULER
    
    if _SCHEDULER:
        _SCHEDULER.shutdown()
        _SCHEDULER = None
        logger.info("Cron scheduler stopped")
