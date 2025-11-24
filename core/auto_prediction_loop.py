"""
Ghost Protocol Auto-Prediction Loop
Continuously generates predictions for all watchlist symbols
Runs every 5 minutes during market hours
"""

import threading
import time
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

# Will be injected by wolf_app.py
LOGGER = None
RUN_PREDICTION_FUNC = None  # Function that runs prediction for a symbol
HUNTER_STOCK_SYMBOLS = []
HUNTER_CRYPTO_SYMBOLS = []

# Loop control
_LOOP_THREAD: threading.Thread | None = None
_LOOP_STOP = threading.Event()
_LAST_RUN_TIME = 0

# Prediction interval (5 minutes)
PREDICTION_INTERVAL_SEC = 300

# Timezone
CHICAGO_TZ = ZoneInfo("America/Chicago")


def _is_market_hours():
    """Check if currently in market hours (9:30 AM - 4:00 PM CT)"""
    now = datetime.now(CHICAGO_TZ)
    
    # Skip weekends
    if now.weekday() >= 5:
        return False
    
    current_time = now.time()
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    
    return market_open <= current_time <= market_close


def _run_all_predictions():
    """Generate predictions for all watchlist symbols"""
    global _LAST_RUN_TIME
    
    if not RUN_PREDICTION_FUNC:
        LOGGER.warning("[AUTO-PREDICT] RUN_PREDICTION_FUNC not set, skipping")
        return
    
    start_time = time.time()
    stocks_success = 0
    crypto_success = 0
    errors = []
    
    # Run stock predictions
    for symbol in HUNTER_STOCK_SYMBOLS:
        try:
            result = RUN_PREDICTION_FUNC(symbol, "stock", "SHORT")
            if result and result.get("ok"):
                stocks_success += 1
                if LOGGER:
                    LOGGER.debug(f"[AUTO-PREDICT] {symbol} → {result.get('direction')} @ {result.get('confidence', 0)*100:.0f}%")
            else:
                error_msg = result.get("error", "unknown") if result else "no result"
                errors.append(f"{symbol}: {error_msg}")
                if LOGGER:
                    LOGGER.warning(f"[AUTO-PREDICT] {symbol} failed: {error_msg}")
        except Exception as e:
            errors.append(f"{symbol}: {str(e)[:100]}")
            if LOGGER:
                LOGGER.error(f"[AUTO-PREDICT] {symbol} exception: {e}")
        
        # Small delay to avoid API rate limits
        time.sleep(0.5)
    
    # Run crypto predictions
    for symbol in HUNTER_CRYPTO_SYMBOLS:
        try:
            result = RUN_PREDICTION_FUNC(symbol, "crypto", "SHORT")
            if result and result.get("ok"):
                crypto_success += 1
                if LOGGER:
                    LOGGER.debug(f"[AUTO-PREDICT] {symbol} → {result.get('direction')} @ {result.get('confidence', 0)*100:.0f}%")
            else:
                error_msg = result.get("error", "unknown") if result else "no result"
                errors.append(f"{symbol}: {error_msg}")
                if LOGGER:
                    LOGGER.warning(f"[AUTO-PREDICT] {symbol} failed: {error_msg}")
        except Exception as e:
            errors.append(f"{symbol}: {str(e)[:100]}")
            if LOGGER:
                LOGGER.error(f"[AUTO-PREDICT] {symbol} exception: {e}")
        
        # Small delay to avoid API rate limits
        time.sleep(0.5)
    
    # Update last run time
    _LAST_RUN_TIME = start_time
    
    # Log summary
    total = stocks_success + crypto_success
    duration = time.time() - start_time
    if LOGGER:
        LOGGER.info(
            f"[AUTO-PREDICT] Batch complete: {total}/{len(HUNTER_STOCK_SYMBOLS) + len(HUNTER_CRYPTO_SYMBOLS)} "
            f"({stocks_success} stocks, {crypto_success} crypto) in {duration:.1f}s"
        )
    
    if errors:
        if LOGGER:
            LOGGER.warning(f"[AUTO-PREDICT] Errors: {errors[:5]}")  # Show first 5


def _prediction_loop():
    """Main loop: Run predictions every 5 minutes"""
    print("[AUTO-PREDICT] Loop started")
    if LOGGER:
        LOGGER.info("[AUTO-PREDICT] Continuous prediction loop started (5-min interval)")
    
    while not _LOOP_STOP.is_set():
        try:
            now = time.time()
            time_since_last = now - _LAST_RUN_TIME
            
            # Only run if:
            # 1. First run OR
            # 2. 5+ minutes since last run OR
            # 3. During market hours for stocks
            should_run = (
                _LAST_RUN_TIME == 0 or  # First run
                time_since_last >= PREDICTION_INTERVAL_SEC  # Interval passed
            )
            
            if should_run:
                print(f"[AUTO-PREDICT] Running batch at {datetime.now().strftime('%H:%M:%S')}")
                _run_all_predictions()
            
            # Sleep for 60 seconds between checks
            _LOOP_STOP.wait(60.0)
            
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"[AUTO-PREDICT] Loop error: {e}", exc_info=True)
            print(f"[AUTO-PREDICT] Loop error: {e}")
            _LOOP_STOP.wait(60.0)


def start_auto_prediction_loop():
    """Start the auto-prediction background thread"""
    global _LOOP_THREAD
    
    if _LOOP_THREAD and _LOOP_THREAD.is_alive():
        print("[AUTO-PREDICT] Loop already running")
        return
    
    _LOOP_STOP.clear()
    _LOOP_THREAD = threading.Thread(
        target=_prediction_loop,
        name="auto-prediction-loop",
        daemon=True
    )
    _LOOP_THREAD.start()
    
    print("[AUTO-PREDICT] Background loop started (5-min interval)")
    if LOGGER:
        LOGGER.info("✅ Auto-Prediction Loop: STARTED (5-min interval, 25+ symbols)")


def stop_auto_prediction_loop():
    """Stop the auto-prediction loop"""
    global _LOOP_THREAD
    
    _LOOP_STOP.set()
    if _LOOP_THREAD:
        _LOOP_THREAD.join(timeout=5.0)
    
    print("[AUTO-PREDICT] Loop stopped")
    if LOGGER:
        LOGGER.info("⏹️  Auto-Prediction Loop: STOPPED")


def trigger_immediate_run():
    """Manually trigger an immediate prediction run (for testing)"""
    print("[AUTO-PREDICT] Manual trigger requested")
    _run_all_predictions()
