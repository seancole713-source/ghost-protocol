"""
Ghost Protocol Auto-Prediction Loop V2
UNLIMITED SCALE: Continuously generates predictions for thousands of symbols
Intelligent batching and parallel execution for optimal performance
Runs 24/7 with adaptive intervals
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

# Adaptive prediction intervals - faster during market hours
PREDICTION_INTERVAL_MARKET_HOURS = 600  # 10 minutes during market hours (reduced to prevent overload)
PREDICTION_INTERVAL_OFF_HOURS = 1800    # 30 minutes off-hours (reduced to prevent overload)
BATCH_SIZE = 50  # Process 50 symbols at a time in parallel
MAX_WORKERS = 10  # Parallel workers for batch processing

# PERFORMANCE FIX: Add delay between predictions to prevent resource exhaustion
PREDICTION_DELAY_S = 2.0  # 2 second delay between predictions to reduce load

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
    """Generate predictions for ALL watchlist symbols with intelligent batching"""
    global _LAST_RUN_TIME
    
    if not RUN_PREDICTION_FUNC:
        LOGGER.warning("[AUTO-PREDICT] RUN_PREDICTION_FUNC not set, skipping")
        return
    
    start_time = time.time()
    stocks_success = 0
    crypto_success = 0
    errors = []
    
    # Check market hours for stocks
    is_market_open = _is_market_hours()
    
    # Run stock predictions (ONLY during market hours)
    stock_count = len(HUNTER_STOCK_SYMBOLS)
    if is_market_open:
        if LOGGER:
            LOGGER.info(f"[AUTO-PREDICT] Market OPEN - processing {stock_count} stocks in batches of {BATCH_SIZE}")
        
        # Process stocks in batches
        for i in range(0, stock_count, BATCH_SIZE):
            batch = HUNTER_STOCK_SYMBOLS[i:i+BATCH_SIZE]
            batch_start = time.time()
            
            for symbol in batch:
                try:
                    result = RUN_PREDICTION_FUNC(symbol, "stock", "SHORT")
                    if result and result.get("ok"):
                        stocks_success += 1
                        if LOGGER and stocks_success % 10 == 0:  # Log every 10th success
                            LOGGER.debug(f"[AUTO-PREDICT] Progress: {stocks_success}/{stock_count} stocks")
                    else:
                        error_msg = result.get("error", "unknown") if result else "no result"
                        errors.append(f"{symbol}: {error_msg}")
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)[:100]}")
                
                # PERFORMANCE FIX: Add delay to prevent overwhelming server
                time.sleep(PREDICTION_DELAY_S)
            
            batch_duration = time.time() - batch_start
            if LOGGER:
                LOGGER.debug(f"[AUTO-PREDICT] Batch {i//BATCH_SIZE + 1} completed in {batch_duration:.1f}s")
    else:
        if LOGGER:
            LOGGER.info(f"[AUTO-PREDICT] Market CLOSED - skipping {stock_count} stock predictions")
    
    # Run crypto predictions (24/7 - crypto markets never close)
    # PERFORMANCE FIX: Limit to top 30 crypto to prevent resource exhaustion
    TOP_CRYPTO_LIMIT = 30
    crypto_symbols_to_process = HUNTER_CRYPTO_SYMBOLS[:TOP_CRYPTO_LIMIT]
    crypto_count = len(crypto_symbols_to_process)
    if LOGGER:
        LOGGER.info(f"[AUTO-PREDICT] Processing {crypto_count}/{len(HUNTER_CRYPTO_SYMBOLS)} crypto (24/7) in batches of {BATCH_SIZE}")
    
    # Process crypto in batches
    for i in range(0, crypto_count, BATCH_SIZE):
        batch = crypto_symbols_to_process[i:i+BATCH_SIZE]
        batch_start = time.time()
        
        for symbol in batch:
            try:
                result = RUN_PREDICTION_FUNC(symbol, "crypto", "SHORT")
                if result and result.get("ok"):
                    crypto_success += 1
                    if LOGGER and crypto_success % 10 == 0:  # Log every 10th success
                        LOGGER.debug(f"[AUTO-PREDICT] Progress: {crypto_success}/{crypto_count} crypto")
                else:
                    error_msg = result.get("error", "unknown") if result else "no result"
                    errors.append(f"{symbol}: {error_msg}")
            except Exception as e:
                errors.append(f"{symbol}: {str(e)[:100]}")
            
            # PERFORMANCE FIX: Add delay to prevent overwhelming server
            time.sleep(PREDICTION_DELAY_S)
        
        batch_duration = time.time() - batch_start
        if LOGGER:
            LOGGER.debug(f"[AUTO-PREDICT] Crypto batch {i//BATCH_SIZE + 1} completed in {batch_duration:.1f}s")
    
    # Update last run time
    _LAST_RUN_TIME = start_time
    
    # Log summary
    total = stocks_success + crypto_success
    duration = time.time() - start_time
    stock_total = stock_count if is_market_open else 0
    crypto_total = crypto_count
    market_status = "OPEN" if is_market_open else "CLOSED"
    
    if LOGGER:
        LOGGER.info(
            f"[AUTO-PREDICT] ✅ Cycle complete: {total}/{stock_total + crypto_total} predictions "
            f"({stocks_success}/{stock_total} stocks [Market {market_status}], "
            f"{crypto_success}/{crypto_total} crypto) in {duration:.1f}s "
            f"({total/duration:.1f} pred/sec)"
        )
    
    if errors and LOGGER:
        LOGGER.warning(f"[AUTO-PREDICT] {len(errors)} errors (showing first 5): {errors[:5]}")


def _prediction_loop():
    """Main loop: Adaptive intervals based on market hours"""
    print("[AUTO-PREDICT] UNLIMITED prediction loop starting...")
    if LOGGER:
        LOGGER.info("[AUTO-PREDICT] Continuous prediction loop started with adaptive intervals")
    
    while not _LOOP_STOP.is_set():
        try:
            now = time.time()
            time_since_last = now - _LAST_RUN_TIME
            
            # Adaptive interval based on market hours
            is_market_open = _is_market_hours()
            interval = PREDICTION_INTERVAL_MARKET_HOURS if is_market_open else PREDICTION_INTERVAL_OFF_HOURS
            
            # Run if first run OR interval has passed
            should_run = (
                _LAST_RUN_TIME == 0 or  # First run
                time_since_last >= interval  # Adaptive interval passed
            )
            
            if should_run:
                market_str = "Market hours" if is_market_open else "Off-hours"
                print(f"[AUTO-PREDICT] {market_str} cycle at {datetime.now().strftime('%H:%M:%S')}")
                _run_all_predictions()
            
            # Sleep for 30 seconds between checks (responsive to interval changes)
            _LOOP_STOP.wait(30.0)
            
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"[AUTO-PREDICT] Loop error: {e}", exc_info=True)
            print(f"[AUTO-PREDICT] Loop error: {e}")
            _LOOP_STOP.wait(30.0)


def start_auto_prediction_loop():
    """Start the UNLIMITED auto-prediction background thread"""
    global _LOOP_THREAD
    
    if _LOOP_THREAD and _LOOP_THREAD.is_alive():
        print("[AUTO-PREDICT] Loop already running")
        return
    
    _LOOP_STOP.clear()
    _LOOP_THREAD = threading.Thread(
        target=_prediction_loop,
        name="auto-prediction-unlimited",
        daemon=True
    )
    _LOOP_THREAD.start()
    
    stock_count = len(HUNTER_STOCK_SYMBOLS)
    crypto_count = len(HUNTER_CRYPTO_SYMBOLS)
    total_count = stock_count + crypto_count
    
    print(f"[AUTO-PREDICT] 🚀 UNLIMITED Loop started - tracking {total_count} symbols ({stock_count} stocks, {crypto_count} crypto)")
    if LOGGER:
        LOGGER.info(f"✅ Auto-Prediction Loop V2: UNLIMITED SCALE activated - {total_count} symbols (adaptive intervals: 3min market / 10min off-hours)")


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
