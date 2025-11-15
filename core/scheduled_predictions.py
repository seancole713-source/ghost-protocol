"""
Ghost Scheduled Market Predictions - Multi-Symbol Edition
Sends automated predictions at key market times:
- 8:00 AM ET: Pre-market multi-symbol prediction
- 12:00 PM ET: Mid-day multi-symbol update
- 4:00 PM ET: End-of-day multi-symbol summary
"""

import threading
from datetime import datetime

import pytz

# Will be set by wolf_app.py
MULTI_SYMBOL_PREDICTION_FUNC = None  # Function that returns multi-symbol predictions
TELEGRAM_SEND_MULTI_FUNC = None  # Function that sends multi-symbol Telegram alert
LOGGER = None

# Tracking state
_PREDICTION_WORKER: threading.Thread | None = None
_PREDICTION_STOP = threading.Event()
_LAST_RUN_DATES: dict[str, str] = {}  # Track last run date for each scheduled time


def _get_ny_time():
    """Get current time in New York timezone"""
    return datetime.now(pytz.timezone("America/New_York"))


def _is_market_day(dt):
    """Check if it's a weekday (Mon-Fri)"""
    return dt.weekday() <= 4


def start_prediction_scheduler():
    """Start the scheduled prediction worker"""
    global _PREDICTION_WORKER
    if _PREDICTION_WORKER is None or not _PREDICTION_WORKER.is_alive():
        _PREDICTION_STOP.clear()
        _PREDICTION_WORKER = threading.Thread(
            target=_prediction_loop, name="multi-prediction-scheduler", daemon=True
        )
        _PREDICTION_WORKER.start()
        if LOGGER:
            LOGGER.info("📅 Multi-symbol prediction scheduler started (8:00 AM, 12:00 PM, 4:00 PM ET)")
        print("[MULTI-PREDICTION SCHEDULER] Started - will send at 8:00 AM, 12:00 PM, 4:00 PM ET")


def stop_prediction_scheduler():
    """Stop the prediction scheduler"""
    try:
        _PREDICTION_STOP.set()
        if _PREDICTION_WORKER and _PREDICTION_WORKER.is_alive():
            _PREDICTION_WORKER.join(timeout=2.0)
    except Exception:
        pass


def _send_scheduled_prediction(time_label: str):
    """
    Send multi-symbol prediction at scheduled time.
    
    Args:
        time_label: Label for this scheduled run (e.g., "08:00", "12:00", "16:00")
    """
    try:
        now_str = _get_ny_time().strftime("%I:%M %p %Z")
        
        if not TELEGRAM_SEND_MULTI_FUNC:
            print(f"[MULTI-PREDICTION] ⚠️ Telegram send function not configured")
            if LOGGER:
                LOGGER.warning("Scheduled prediction skipped: TELEGRAM_SEND_MULTI_FUNC not set")
            return
        
        # Send multi-symbol Telegram alert (handles prediction generation internally)
        success = TELEGRAM_SEND_MULTI_FUNC()
        
        if success:
            print(f"[MULTI-PREDICTION] ✅ Sent scheduled multi-symbol prediction at {now_str} ({time_label})")
            if LOGGER:
                LOGGER.info(f"Scheduled multi-symbol prediction sent: {time_label}")
        else:
            print(f"[MULTI-PREDICTION] ❌ Failed to send scheduled prediction at {now_str} ({time_label})")
            if LOGGER:
                LOGGER.error(f"Scheduled multi-symbol prediction failed: {time_label}")
    
    except Exception as e:
        print(f"[MULTI-PREDICTION] ❌ Error sending scheduled prediction ({time_label}): {e}")
        if LOGGER:
            LOGGER.exception(f"Scheduled prediction error ({time_label}): {e}")


def _prediction_loop():
    """Main loop checking for scheduled prediction times"""
    global _LAST_RUN_DATES
    
    print("[MULTI-PREDICTION SCHEDULER] Loop started, checking every 30 seconds...")
    
    # Define scheduled times (ET)
    scheduled_times = [
        ("08:00", "Pre-market"),
        ("12:00", "Mid-day"),
        ("16:00", "End-of-day"),
    ]
    
    while not _PREDICTION_STOP.is_set():
        try:
            now = _get_ny_time()
            
            # Only run on market days (Mon-Fri)
            if not _is_market_day(now):
                _PREDICTION_STOP.wait(60.0)  # Check every minute on weekends
                continue
            
            current_date = now.strftime("%Y-%m-%d")
            current_time = now.time()
            
            # Check each scheduled time
            for time_str, label in scheduled_times:
                schedule_key = f"{time_str}_{current_date}"
                
                # Skip if already ran today
                if _LAST_RUN_DATES.get(time_str) == current_date:
                    continue
                
                # Parse target time
                target_time = datetime.strptime(time_str, "%H:%M").time()
                
                # Calculate time difference
                time_diff = abs(
                    (
                        datetime.combine(now.date(), current_time)
                        - datetime.combine(now.date(), target_time)
                    ).total_seconds()
                )
                
                # Run if within 2.5 minute window
                if time_diff <= 150:
                    print(f"[MULTI-PREDICTION] 🔔 Triggering {label} prediction at {now.strftime('%H:%M')}")
                    _send_scheduled_prediction(time_str)
                    _LAST_RUN_DATES[time_str] = current_date
        
        except Exception as e:
            print(f"[MULTI-PREDICTION] ❌ Loop error: {e}")
            if LOGGER:
                LOGGER.exception(f"Multi-prediction loop error: {e}")
        
        finally:
            # Check every 30 seconds
            _PREDICTION_STOP.wait(30.0)


def force_multi_prediction():
    """Manually trigger multi-symbol prediction (for testing)"""
    print("[MULTI-PREDICTION] 🧪 Forcing multi-symbol prediction...")
    _send_scheduled_prediction("manual")
