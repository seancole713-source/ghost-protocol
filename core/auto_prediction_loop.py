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
RUN_PREDICTION_FUNC = None  # Function that runs prediction for a symbol (sync)
RUN_PREDICTION_FUNC_ASYNC = None  # Function that runs prediction for a symbol (async)
HUNTER_STOCK_SYMBOLS = []
HUNTER_CRYPTO_SYMBOLS = []

# Get configuration from environment variables
import os

# Loop control
_LOOP_THREAD: threading.Thread | None = None
_LOOP_STOP = threading.Event()
_LAST_RUN_TIME = 0
_LOOP_RUNNING = False  # Prevent multiple loops from starting
_ACTIVE_PREDICTION_THREAD: threading.Thread | None = None  # Track active prediction worker

# Deduplication cache: Track recent predictions to prevent duplicates
_RECENT_PREDICTIONS = {}  # {symbol: timestamp}
_DEDUP_WINDOW_S = int(os.getenv("PREDICTION_DEDUP_WINDOW_S", "3300"))  # Default: 55 minutes (prevents duplicates within 60-min cycle)

# ULTRA-LIGHT intervals for Railway free tier (512MB RAM)
PREDICTION_INTERVAL_MARKET_HOURS = int(os.getenv("AUTO_PREDICT_MARKET_INTERVAL_S", "3600"))  # Default: 60 min
PREDICTION_INTERVAL_OFF_HOURS = int(os.getenv("AUTO_PREDICT_OFF_HOURS_INTERVAL_S", "7200"))  # Default: 120 min
PREDICTION_DELAY_S = float(os.getenv("AUTO_PREDICT_DELAY_S", "2.0"))  # Default: 2s between predictions
BATCH_SIZE = int(os.getenv("AUTO_PREDICT_BATCH_SIZE", "2"))  # Default: 2 concurrent predictions
MAX_WORKERS = int(os.getenv("AUTO_PREDICT_MAX_WORKERS", "10"))  # Default: 10 workers
BATCH_DELAY_S = int(os.getenv("AUTO_PREDICT_BATCH_DELAY_S", "10"))  # Default: 10s between batches

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


async def _run_all_predictions_async():
    """Generate predictions for ALL watchlist symbols with async/await (non-blocking)"""
    global _LAST_RUN_TIME, _RECENT_PREDICTIONS
    
    if not RUN_PREDICTION_FUNC_ASYNC:
        LOGGER.warning("[AUTO-PREDICT] RUN_PREDICTION_FUNC_ASYNC not set, skipping")
        return
    
    # Clean up old entries from deduplication cache
    current_time = time.time()
    _RECENT_PREDICTIONS = {
        sym: ts for sym, ts in _RECENT_PREDICTIONS.items()
        if current_time - ts < _DEDUP_WINDOW_S
    }
    
    start_time = current_time
    stocks_success = 0
    crypto_success = 0
    errors = []
    
    # Check market hours for stocks
    is_market_open = _is_market_hours()
    
    # Run stock predictions (ONLY during market hours)
    stock_count = len(HUNTER_STOCK_SYMBOLS)
    if is_market_open:
        if LOGGER:
            LOGGER.info(f"[AUTO-PREDICT] Market OPEN - processing {stock_count} stocks asynchronously")
        
        # Process stocks with async concurrency (2 at a time for stability)
        for i in range(0, stock_count, 2):  # REDUCED: 2 concurrent (was 3)
            batch = HUNTER_STOCK_SYMBOLS[i:i+2]
            
            # Filter out recently predicted symbols (deduplication)
            batch_filtered = [
                s for s in batch
                if s not in _RECENT_PREDICTIONS or (current_time - _RECENT_PREDICTIONS[s]) >= _DEDUP_WINDOW_S
            ]
            
            if not batch_filtered:
                continue  # Skip this batch if all symbols recently predicted
            
            # Create async tasks for batch
            tasks = []
            for symbol in batch_filtered:
                tasks.append(RUN_PREDICTION_FUNC_ASYNC(symbol))
            
            # Wait for all tasks in batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for symbol, result in zip(batch_filtered, results):
                try:
                    if isinstance(result, Exception):
                        errors.append(f"{symbol}: {str(result)[:100]}")
                    elif result and result.get("ok"):
                        stocks_success += 1
                        _RECENT_PREDICTIONS[symbol] = current_time  # Mark as predicted
                        if LOGGER and stocks_success % 10 == 0:
                            LOGGER.debug(f"[AUTO-PREDICT] Progress: {stocks_success}/{stock_count} stocks")
                    else:
                        error_msg = result.get("error", "unknown") if result else "no result"
                        errors.append(f"{symbol}: {error_msg}")
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)[:100]}")
            
            # ULTRA-LIGHT: 5s delay between batches to minimize Railway resource usage
            await asyncio.sleep(PREDICTION_DELAY_S)
    else:
        if LOGGER:
            LOGGER.info(f"[AUTO-PREDICT] Market CLOSED - skipping {stock_count} stock predictions")
    
    # Run crypto predictions (24/7 - crypto markets never close)
    # Process ALL crypto symbols (no artificial limits - scales to 1000+ coins)
    crypto_symbols_to_process = HUNTER_CRYPTO_SYMBOLS
    crypto_count = len(crypto_symbols_to_process)
    if LOGGER:
        LOGGER.info(f"[AUTO-PREDICT] ASYNC: Processing {crypto_count} crypto symbols with concurrency")
    
    # Process crypto with async concurrency (2 at a time for stability)
    for i in range(0, crypto_count, 2):  # REDUCED: 2 concurrent (was 3)
        batch = crypto_symbols_to_process[i:i+2]
        
        # Filter out recently predicted symbols (deduplication)
        batch_filtered = [
            s for s in batch
            if s not in _RECENT_PREDICTIONS or (current_time - _RECENT_PREDICTIONS[s]) >= _DEDUP_WINDOW_S
        ]
        
        if not batch_filtered:
            continue  # Skip this batch if all symbols recently predicted
        
        # Create async tasks for batch
        tasks = []
        for symbol in batch_filtered:
            tasks.append(RUN_PREDICTION_FUNC_ASYNC(symbol))
        
        # Wait for all tasks in batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for symbol, result in zip(batch_filtered, results):
            try:
                if isinstance(result, Exception):
                    errors.append(f"{symbol}: {str(result)[:100]}")
                elif result and result.get("ok"):
                    crypto_success += 1
                    _RECENT_PREDICTIONS[symbol] = current_time  # Mark as predicted
                    if LOGGER and crypto_success % 10 == 0:
                        LOGGER.debug(f"[AUTO-PREDICT] Progress: {crypto_success}/{crypto_count} crypto")
                else:
                    error_msg = result.get("error", "unknown") if result else "no result"
                    errors.append(f"{symbol}: {error_msg}")
            except Exception as e:
                errors.append(f"{symbol}: {str(e)[:100]}")
        
        # Delay between batches for Railway stability (configurable via BATCH_DELAY_S)
        await asyncio.sleep(BATCH_DELAY_S)
    
    # Update last run time
    _LAST_RUN_TIME = time.time()
    
    # Update global prediction counters for health score
    try:
        import wolf_app
        wolf_app._LAST_MULTI_PREDICTION_COUNTS["stocks"] = stocks_success
        wolf_app._LAST_MULTI_PREDICTION_COUNTS["crypto"] = crypto_success
    except Exception as e:
        if LOGGER:
            LOGGER.warning(f"Could not update prediction counters: {e}")
    
    # Log summary
    total = stocks_success + crypto_success
    duration = time.time() - start_time
    
    if LOGGER:
        LOGGER.info(
            f"[AUTO-PREDICT] ✅ Async cycle complete: {total}/{stock_count + crypto_count} predictions "
            f"({stocks_success}/{stock_count} stocks, {crypto_success}/{crypto_count} crypto) "
            f"in {duration:.1f}s ({total/duration:.1f} pred/sec)"
        )
    
    if errors and LOGGER:
        LOGGER.warning(f"[AUTO-PREDICT] {len(errors)} errors (showing first 5): {errors[:5]}")


def _run_all_predictions():
    """
    BACKGROUND WORKER: Run predictions in separate thread pool.
    
    CRITICAL: This runs in a background thread spawned by threading.Thread(),
    NOT in the FastAPI event loop. This prevents blocking HTTP responses.
    """
    import threading
    
    # Run in separate event loop (thread-safe)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Log that we're in background thread
        thread_id = threading.current_thread().name
        if LOGGER:
            LOGGER.info(f"[AUTO-PREDICT] Running in background thread: {thread_id}")
        
        loop.run_until_complete(_run_all_predictions_async())
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"[AUTO-PREDICT] Background worker error: {e}", exc_info=True)
    finally:
        loop.close()


# Keep old sync version for backward compatibility
def _run_all_predictions_sync():
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
                
                # ULTRA-LIGHT: 5s delay to minimize Railway resource usage
                time.sleep(PREDICTION_DELAY_S)
            
            batch_duration = time.time() - batch_start
            if LOGGER:
                LOGGER.debug(f"[AUTO-PREDICT] Batch {i//BATCH_SIZE + 1} completed in {batch_duration:.1f}s")
    else:
        if LOGGER:
            LOGGER.info(f"[AUTO-PREDICT] Market CLOSED - skipping {stock_count} stock predictions")
    
    # Run crypto predictions (24/7 - crypto markets never close)
    # ULTRA-LIGHT: Top 10 crypto only for Railway free tier
    TOP_CRYPTO_LIMIT = 10  # BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, MATIC, AVAX
    crypto_symbols_to_process = HUNTER_CRYPTO_SYMBOLS[:TOP_CRYPTO_LIMIT]
    crypto_count = len(crypto_symbols_to_process)
    if LOGGER:
        LOGGER.info(f"[AUTO-PREDICT] ULTRA-LIGHT: Processing {crypto_count}/{len(HUNTER_CRYPTO_SYMBOLS)} top crypto in batches of {BATCH_SIZE}")
    
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
            
            # ULTRA-LIGHT: 5s delay to minimize Railway resource usage
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
                # CRITICAL: Check if previous prediction cycle is still running
                global _ACTIVE_PREDICTION_THREAD
                if _ACTIVE_PREDICTION_THREAD and _ACTIVE_PREDICTION_THREAD.is_alive():
                    if LOGGER:
                        LOGGER.warning(f"[AUTO-PREDICT] Previous cycle still running ({_ACTIVE_PREDICTION_THREAD.name}), skipping new cycle to prevent duplicates")
                else:
                    market_str = "Market hours" if is_market_open else "Off-hours"
                    print(f"[AUTO-PREDICT] {market_str} cycle starting at {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Run predictions in SEPARATE background thread (fire-and-forget)
                    # This prevents blocking the prediction loop scheduler
                    _ACTIVE_PREDICTION_THREAD = threading.Thread(
                        target=_run_all_predictions,
                        name=f"prediction-cycle-{int(now)}",
                        daemon=True
                    )
                    _ACTIVE_PREDICTION_THREAD.start()
                    
                    # Don't wait for completion - let it run in background
                    print(f"[AUTO-PREDICT] Cycle dispatched to background thread: {_ACTIVE_PREDICTION_THREAD.name}")
            
            # Sleep for 30 seconds between checks (responsive to interval changes)
            _LOOP_STOP.wait(30.0)
            
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"[AUTO-PREDICT] Loop error: {e}", exc_info=True)
            print(f"[AUTO-PREDICT] Loop error: {e}")
            _LOOP_STOP.wait(30.0)


def start_auto_prediction_loop():
    """Start the ASYNC auto-prediction background thread"""
    # RE-ENABLED with ASYNC architecture - non-blocking predictions
    # Uses asyncio.run_in_executor to prevent server hangs
    # Top 10 crypto, 60-minute intervals for Railway Pro tier
    global _LOOP_THREAD, _LOOP_RUNNING
    
    # Singleton guard: prevent multiple loops
    if _LOOP_RUNNING:
        print("[AUTO-PREDICT] ⚠️ Loop already running (singleton guard)")
        if LOGGER:
            LOGGER.warning("Auto-prediction loop already running - ignoring duplicate start request")
        return
    
    if _LOOP_THREAD and _LOOP_THREAD.is_alive():
        print("[AUTO-PREDICT] Loop already running")
        return
    
    _LOOP_RUNNING = True
    print("[AUTO-PREDICT] ⚡ Starting ASYNC mode (non-blocking predictions)")
    print("[AUTO-PREDICT] ℹ️ Top 10 crypto, 60min intervals, async/await architecture")
    if LOGGER:
        LOGGER.info("⚡ Auto-predictions RE-ENABLED - ASYNC architecture")
        LOGGER.info("ℹ️ Non-blocking predictions using asyncio")
        LOGGER.info("ℹ️ Top 10 crypto, 60-minute cycles, server stays responsive")
    
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
