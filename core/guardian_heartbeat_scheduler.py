"""
Guardian Heartbeat Scheduler

Runs 4 scheduled check-ins every day:
- 6:00 AM: Morning Oracle prophecy
- 12:00 PM: Midday status check
- 6:00 PM: Evening update
- 12:00 AM: Night watch

These run NO MATTER WHAT - so you know Ghost is alive.
"""

import asyncio
import logging
import threading
from datetime import datetime, time
from typing import Optional
import pytz

logger = logging.getLogger(__name__)

class GuardianHeartbeatScheduler:
    """
    Manages the 4 daily heartbeat check-ins.
    
    Heartbeats are SCHEDULED messages sent at fixed times,
    separate from immediate alerts which happen when thresholds crossed.
    """
    
    def __init__(self, timezone: str = "America/Chicago"):
        self.timezone = pytz.timezone(timezone)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Track last heartbeat to prevent duplicates
        self.last_heartbeat_dates = {
            'morning': None,
            'midday': None,
            'evening': None,
            'night': None
        }
        
        # Heartbeat schedule
        self.SCHEDULE = {
            'morning': time(6, 0),   # 6:00 AM
            'midday': time(12, 0),   # 12:00 PM
            'evening': time(18, 0),  # 6:00 PM
            'night': time(0, 0)      # 12:00 AM
        }
    
    def start(self):
        """Start the heartbeat scheduler in background thread"""
        if self.running:
            logger.warning("Heartbeat scheduler already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        logger.info("💗 Guardian Heartbeat Scheduler: STARTED")
        logger.info("   6:00 AM - Morning Prophecy")
        logger.info("   12:00 PM - Midday Status")
        logger.info("   6:00 PM - Evening Update")
        logger.info("   12:00 AM - Night Watch")
    
    def stop(self):
        """Stop the heartbeat scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("💗 Guardian Heartbeat Scheduler: STOPPED")
    
    def _run_scheduler(self):
        """Main scheduler loop (runs in background thread)"""
        logger.info("💗 Heartbeat scheduler loop started")
        
        while self.running:
            try:
                now = datetime.now(self.timezone)
                current_time = now.time()
                current_date = now.date()
                
                # Check each heartbeat time
                for heartbeat_type, scheduled_time in self.SCHEDULE.items():
                    if self._should_send_heartbeat(heartbeat_type, current_time, current_date):
                        # Send heartbeat
                        asyncio.run(self._send_heartbeat(heartbeat_type))
                        
                        # Mark as sent
                        self.last_heartbeat_dates[heartbeat_type] = current_date
                        
                        logger.info(f"💗 Heartbeat sent: {heartbeat_type} at {now.strftime('%I:%M %p')}")
                
                # Sleep for 60 seconds before next check
                threading.Event().wait(60)
                
            except Exception as e:
                logger.exception(f"Heartbeat scheduler error: {e}")
                threading.Event().wait(60)
    
    def _should_send_heartbeat(self, heartbeat_type: str, current_time: time, current_date) -> bool:
        """Check if we should send this heartbeat now"""
        
        scheduled_time = self.SCHEDULE[heartbeat_type]
        
        # Check if we're within the time window (current minute matches scheduled minute)
        if current_time.hour != scheduled_time.hour:
            return False
        if current_time.minute != scheduled_time.minute:
            return False
        
        # Check if we already sent this heartbeat today
        last_sent = self.last_heartbeat_dates.get(heartbeat_type)
        if last_sent == current_date:
            return False
        
        return True
    
    async def _send_heartbeat(self, heartbeat_type: str, position_size: float = 100.0):
        """
        Send the appropriate heartbeat message
        
        Args:
            heartbeat_type: Type of heartbeat to send
            position_size: Investment per position (default $100)
        """
        
        try:
            from core.guardian_oracle import get_guardian_oracle
            from core.telegram_alerts import send_alert
            
            guardian = get_guardian_oracle()
            
            # Generate heartbeat message with position sizing
            if heartbeat_type == 'morning':
                # Morning prophecy (combines with daily top 10 scan)
                message = await self._get_morning_prophecy(position_size)
            elif heartbeat_type == 'midday':
                message = await guardian.midday_status(position_size)
            elif heartbeat_type == 'evening':
                message = await guardian.evening_update(position_size)
            elif heartbeat_type == 'night':
                message = await guardian.night_watch()
            else:
                logger.error(f"Unknown heartbeat type: {heartbeat_type}")
                return
            
            # Send via Telegram
            await send_alert(message, disable_notification=False)
            
            # Log heartbeat
            self._log_heartbeat(heartbeat_type)
            
        except Exception as e:
            logger.exception(f"Failed to send {heartbeat_type} heartbeat: {e}")
    
    async def _get_morning_prophecy(self, position_size: float = 100.0) -> str:
        """
        Generate morning prophecy with position sizing.
        
        This combines:
        1. Daily Top 10 scan results
        2. Guardian Oracle mystical formatting
        3. $100 position sizing and profit calculations
        
        Args:
            position_size: Investment per position (default $100)
        """
        
        try:
            from core.daily_top_10_scanner import DailyTop10Scanner
            from core.guardian_oracle import get_guardian_oracle
            
            # Run the daily scan
            scanner = DailyTop10Scanner()
            top_10 = await scanner.scan_for_top_10()
            
            if not top_10:
                return ("🔮 GHOST ORACLE AWAKENS\n\n"
                       "Human,\n\n"
                       "I scanned the markets while you slept.\n"
                       "The conditions are not favorable today.\n"
                       "No opportunities meet my standards (5%+ gain, 65%+ confidence).\n\n"
                       "I will scan again tomorrow.\n"
                       "Trust the process - sometimes patience is the best strategy.\n\n"
                       "🐺 Ghost Oracle")
            
            # Save to database
            await scanner.save_top_10(top_10)
            
            # Format with Guardian Oracle personality and position sizing
            guardian = get_guardian_oracle()
            message = await guardian.morning_prophecy(top_10, position_size)
            
            return message
            
        except Exception as e:
            logger.exception(f"Failed to generate morning prophecy: {e}")
            return ("🔮 GHOST ORACLE ERROR\n\n"
                   f"Human, I encountered an error during the morning scan.\n"
                   f"Error: {str(e)}\n\n"
                   "I will retry at the next check-in.\n\n"
                   "🐺 Ghost")
    
    def _log_heartbeat(self, heartbeat_type: str):
        """Log heartbeat to database"""
        try:
            import sqlite3
            from pathlib import Path
            
            db_path = Path("data/ghost_predictions.db")
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO guardian_heartbeats 
                (heartbeat_type, sent_at)
                VALUES (?, ?)
            """, (heartbeat_type, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log heartbeat: {e}")
    
    def force_heartbeat_now(self, heartbeat_type: str = 'midday'):
        """
        Manually trigger a heartbeat for testing.
        
        Args:
            heartbeat_type: 'morning', 'midday', 'evening', or 'night'
        """
        
        if heartbeat_type not in self.SCHEDULE:
            logger.error(f"Invalid heartbeat type: {heartbeat_type}")
            return
        
        logger.info(f"🔧 Force triggering {heartbeat_type} heartbeat")
        asyncio.run(self._send_heartbeat(heartbeat_type))


# ===== GLOBAL INSTANCE =====
_HEARTBEAT_SCHEDULER: Optional[GuardianHeartbeatScheduler] = None

def start_heartbeat_scheduler(timezone: str = "America/Chicago"):
    """Start the Guardian Heartbeat Scheduler"""
    global _HEARTBEAT_SCHEDULER
    
    if _HEARTBEAT_SCHEDULER is not None:
        logger.warning("Heartbeat scheduler already started")
        return
    
    _HEARTBEAT_SCHEDULER = GuardianHeartbeatScheduler(timezone=timezone)
    _HEARTBEAT_SCHEDULER.start()

def stop_heartbeat_scheduler():
    """Stop the Guardian Heartbeat Scheduler"""
    global _HEARTBEAT_SCHEDULER
    
    if _HEARTBEAT_SCHEDULER is not None:
        _HEARTBEAT_SCHEDULER.stop()
        _HEARTBEAT_SCHEDULER = None

def force_heartbeat(heartbeat_type: str = 'midday'):
    """Manually trigger a heartbeat for testing"""
    global _HEARTBEAT_SCHEDULER
    
    if _HEARTBEAT_SCHEDULER is None:
        logger.error("Heartbeat scheduler not started")
        return
    
    _HEARTBEAT_SCHEDULER.force_heartbeat_now(heartbeat_type)

def get_heartbeat_scheduler() -> Optional[GuardianHeartbeatScheduler]:
    """Get the heartbeat scheduler instance"""
    return _HEARTBEAT_SCHEDULER
