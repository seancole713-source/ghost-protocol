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
    Generate and send multi-symbol prediction at scheduled time.
    Prediction generation happens FIRST, Telegram send is best-effort.
    
    Args:
        time_label: Label for this scheduled run (e.g., "08:00", "12:00", "16:00")
    """
    now_str = _get_ny_time().strftime("%I:%M %p %Z")
    prediction_generated = False
    telegram_sent = False
    
    # PHASE 1: Generate predictions (ALWAYS attempt this first)
    try:
        if MULTI_SYMBOL_PREDICTION_FUNC:
            print(f"[MULTI-PREDICTION] 🔮 Generating predictions at {now_str} ({time_label})...")
            result = MULTI_SYMBOL_PREDICTION_FUNC()
            prediction_generated = True
            print(f"[MULTI-PREDICTION] ✅ Predictions generated successfully ({time_label})")
            if LOGGER:
                LOGGER.info(f"Scheduled predictions generated: {time_label}", extra={"result": result})
        else:
            print(f"[MULTI-PREDICTION] ⚠️ Prediction function not configured, skipping generation")
            if LOGGER:
                LOGGER.warning("Scheduled prediction generation skipped: MULTI_SYMBOL_PREDICTION_FUNC not set")
    except Exception as e:
        print(f"[MULTI-PREDICTION] ❌ Error generating predictions ({time_label}): {e}")
        if LOGGER:
            LOGGER.exception(f"Scheduled prediction generation error ({time_label}): {e}")
    
    # PHASE 2: Send Telegram alert (best effort, don't block on failure)
    try:
        if TELEGRAM_SEND_MULTI_FUNC:
            print(f"[MULTI-PREDICTION] 📱 Sending Telegram alert at {now_str} ({time_label})...")
            success = TELEGRAM_SEND_MULTI_FUNC()
            telegram_sent = success
            
            if success:
                print(f"[MULTI-PREDICTION] ✅ Telegram alert sent ({time_label})")
                if LOGGER:
                    LOGGER.info(f"Scheduled Telegram alert sent: {time_label}")
            else:
                print(f"[MULTI-PREDICTION] ⚠️ Telegram send returned False ({time_label})")
                if LOGGER:
                    LOGGER.warning(f"Scheduled Telegram alert failed: {time_label}")
        else:
            print(f"[MULTI-PREDICTION] ⚠️ Telegram function not configured, skipping alert")
            if LOGGER:
                LOGGER.warning("Telegram alert skipped: TELEGRAM_SEND_MULTI_FUNC not set")
    except Exception as e:
        print(f"[MULTI-PREDICTION] ❌ Error sending Telegram alert ({time_label}): {e}")
        if LOGGER:
            LOGGER.exception(f"Telegram send error ({time_label}): {e}")
    
    # Summary
    status = f"predictions={'✅' if prediction_generated else '❌'}, telegram={'✅' if telegram_sent else '❌'}"
    print(f"[MULTI-PREDICTION] 📊 Scheduled run complete ({time_label}): {status}")


def _prediction_loop():
    """Main loop checking for scheduled prediction times with catchup logic"""
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
                time_diff = (
                    datetime.combine(now.date(), current_time)
                    - datetime.combine(now.date(), target_time)
                ).total_seconds()
                
                # Phase 5: Extended catchup window - run if:
                # 1. Within 2.5 min window (normal trigger)
                # 2. OR past scheduled time but less than 30 min late (catchup)
                if -150 <= time_diff <= 150:
                    # Normal trigger window
                    print(f"[MULTI-PREDICTION] 🔔 Triggering {label} prediction at {now.strftime('%H:%M')}")
                    _send_scheduled_prediction(time_str)
                    _LAST_RUN_DATES[time_str] = current_date
                elif 150 < time_diff <= 1800:
                    # Catchup window (missed by 2.5-30 min)
                    print(f"[MULTI-PREDICTION] 🔄 Catchup: Running missed {label} prediction (missed by {int(time_diff/60)} min)")
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
