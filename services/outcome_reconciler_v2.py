#!/usr/bin/env python3
"""
Enhanced Ghost Prediction Outcome Reconciler
=============================================
Background task that:
1. Finds predictions where 48h window has closed
2. Fetches actual prices at t+48h using live providers
3. Computes accuracy metrics
4. Stores outcomes in Postgres

This is the CORE of Ghost's 70% accuracy measurement.
"""

import logging
import os
import time
from typing import Optional, Dict, Any
from datetime import datetime
from core.prediction_store import get_prediction_store
import psycopg2

LOGGER = logging.getLogger("ghost.outcome_reconciler_v2")

# Direction threshold (±0.25% by default)
DIRECTION_THRESHOLD_PCT = float(os.getenv("ACCURACY_DIRECTION_THRESHOLD_PCT", "0.25"))


def reconcile_outcomes_v2():
    """
    Find predictions whose 48h window has closed and reconcile their outcomes.
    Stores results in ghost_prediction_outcomes table via Postgres.
    
    CRASH PROTECTION:
    - Batch limit: Max 100 predictions per run (enforced in get_pending_outcomes)
    - Time limit: Max 5 minutes total per reconciliation run (checked per prediction)
    - Circuit breaker: Stops if >70% failure rate after 10 predictions
    - Fast-fail: Skips predictions without data immediately
    
    Returns:
        dict with counts of success/no_data/error/skipped
    """
    LOGGER.info("🔄 Starting outcome reconciliation V2...")
    
    # Track start time for timeout (signal doesn't work in background threads)
    start_time = time.time()
    max_duration = 300  # 5 minutes
    
    try:
        store = get_prediction_store()
        
        # Get pending predictions (48h window closed, no outcome yet)
        # Query is limited to 100 predictions max
        pending = store.get_pending_outcomes()
        
        if not pending:
            LOGGER.info("✅ No pending outcomes to reconcile")
            return {"success": 0, "no_data": 0, "error": 0, "skipped": 0}
        
        LOGGER.info(f"📊 Found {len(pending)} predictions ready for reconciliation")
        
        # Process each prediction
        success_count = 0
        no_data_count = 0
        error_count = 0
        skipped_count = 0
        
        for idx, pred in enumerate(pending, start=1):
            try:
                # TIME LIMIT: Check if we've exceeded max duration
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    LOGGER.warning(
                        f"⏰ Reconciliation exceeded {max_duration}s time limit "
                        f"(processed {idx-1}/{len(pending)} predictions). Stopping."
                    )
                    break
                
                result = _reconcile_single_v2(pred)
                
                if result == "success":
                    success_count += 1
                elif result == "no_data":
                    no_data_count += 1
                elif result == "error":
                    error_count += 1
                else:
                    skipped_count += 1
                
                # CIRCUIT BREAKER: Stop if >70% failures after processing at least 10
                if idx >= 10:
                    total_processed = success_count + no_data_count + error_count + skipped_count
                    failure_rate = (no_data_count + error_count) / total_processed
                    if failure_rate > 0.70:
                        LOGGER.warning(
                            f"🚨 CIRCUIT BREAKER TRIGGERED: {failure_rate*100:.1f}% failure rate "
                            f"({no_data_count + error_count}/{total_processed} failed). "
                            f"Stopping reconciliation to prevent cascade failure."
                        )
                        break
                    
            except Exception as e:
                LOGGER.error(f"❌ Unexpected error reconciling prediction {pred.get('id')}: {e}", exc_info=True)
                error_count += 1
        
        LOGGER.info(
            f"✅ Reconciliation complete: "
            f"{success_count} success, {no_data_count} no_data, "
            f"{error_count} errors, {skipped_count} skipped"
        )
        
        return {
            "success": success_count,
            "no_data": no_data_count,
            "error": error_count,
            "skipped": skipped_count,
        }
    
    except Exception as e:
        LOGGER.error(f"❌ Outcome reconciliation failed: {e}", exc_info=True)
        return {"success": 0, "no_data": 0, "error": 0, "skipped": 0}


def _reconcile_single_v2(pred: Dict[str, Any]) -> str:
    """
    Reconcile a single prediction.
    
    Returns:
        "success" - Outcome stored successfully
        "no_data" - Could not fetch actual price
        "error" - Something went wrong
    """
    pred_id = pred["id"]
    symbol = pred["symbol"]
    run_at = pred["run_at"]
    horizon_h = pred.get("horizon_h", 48)
    pred_direction = pred.get("direction", "UP")
    pred_confidence = pred.get("confidence", 0.5)
    
    # Calculate resolution time (run_at + 48h)
    t_resolve = run_at + (horizon_h * 3600)
    
    LOGGER.info(f"🔍 Reconciling prediction {pred_id} ({symbol}) - "
                f"Created: {datetime.fromtimestamp(run_at)}, "
                f"Resolve: {datetime.fromtimestamp(t_resolve)}")
    
    # Get price at prediction time (t0)
    try:
        price_t0 = _get_price_at_time(symbol, run_at)
        if price_t0 is None:
            LOGGER.warning(f"⚠️  No price at t0 for {symbol} (pred {pred_id}), marking no_data")
            _store_outcome_no_data(pred_id, symbol, run_at, t_resolve, pred_direction, pred_confidence,
                                   "Could not fetch price at prediction time")
            return "no_data"
    except Exception as e:
        LOGGER.error(f"❌ Failed to fetch t0 price for {symbol}: {e}")
        _store_outcome_no_data(pred_id, symbol, run_at, t_resolve, pred_direction, pred_confidence,
                               f"Error fetching t0 price: {str(e)[:100]}")
        return "no_data"
    
    # Get price at resolution time (t1 = t0 + 48h)
    try:
        price_t1 = _get_price_at_time(symbol, t_resolve)
        if price_t1 is None:
            LOGGER.warning(f"⚠️  No price at t1 for {symbol} (pred {pred_id}), marking no_data")
            _store_outcome_no_data(pred_id, symbol, run_at, t_resolve, pred_direction, pred_confidence,
                                   "Could not fetch price at resolution time (t+48h)")
            return "no_data"
    except Exception as e:
        LOGGER.error(f"❌ Failed to fetch t1 price for {symbol}: {e}")
        _store_outcome_no_data(pred_id, symbol, run_at, t_resolve, pred_direction, pred_confidence,
                               f"Error fetching t1 price: {str(e)[:100]}")
        return "no_data"
    
    # Compute realized movement
    realized_move_pct = ((price_t1 - price_t0) / price_t0) * 100
    
    # Determine actual direction
    if realized_move_pct > DIRECTION_THRESHOLD_PCT:
        actual_direction = "UP"
    elif realized_move_pct < -DIRECTION_THRESHOLD_PCT:
        actual_direction = "DOWN"
    else:
        actual_direction = "FLAT"
    
    # Determine if prediction was correct
    hit_direction = 1 if actual_direction == pred_direction else 0
    
    # Store outcome in Postgres
    try:
        _store_outcome_success(
            prediction_id=pred_id,
            symbol=symbol,
            closed_at=t_resolve,
            price_at_prediction=price_t0,
            price_at_resolution=price_t1,
            realized_move_pct=realized_move_pct,
            predicted_direction=pred_direction,
            actual_direction=actual_direction,
            hit_direction=hit_direction,
            predicted_confidence=pred_confidence,
        )
        
        accuracy_symbol = "✅" if hit_direction == 1 else "❌"
        LOGGER.info(
            f"{accuracy_symbol} Prediction {pred_id} ({symbol}): "
            f"Predicted {pred_direction}, Actual {actual_direction} "
            f"(${price_t0:.2f} → ${price_t1:.2f}, {realized_move_pct:+.2f}%)"
        )
        
        return "success"
        
    except Exception as e:
        LOGGER.error(f"❌ Failed to store outcome for prediction {pred_id}: {e}", exc_info=True)
        return "error"


def _get_price_at_time(symbol: str, timestamp: float) -> Optional[float]:
    """
    Fetch price for symbol at given timestamp using historical data when possible.
    
    Attempts:
    1. Check if prediction store has recorded prices near this timestamp
    2. Try Polygon historical bars (minute-level precision for recent data)
    3. Fall back to current price as approximation
    
    FAST-FAIL: Returns None immediately if price unavailable.
    Does not retry or wait - prevents hanging on missing data.
    
    Returns:
        Price as float, or None if unavailable
    """
    try:
        from datetime import datetime
        import requests
        
        # Try to get recorded price from prediction store (most accurate)
        try:
            from core.prediction_store import get_prediction_store
            store = get_prediction_store()
            # Look for predictions made within ±10 minutes of target timestamp
            # that have recorded price_at_prediction
            time_window = 600  # 10 minutes
            recent_preds = store.backend.query(
                "SELECT price_at_prediction FROM predictions "
                "WHERE symbol = ? AND run_at BETWEEN ? AND ? "
                "AND price_at_prediction IS NOT NULL "
                "ORDER BY ABS(run_at - ?) LIMIT 1",
                (symbol, timestamp - time_window, timestamp + time_window, timestamp)
            )
            if recent_preds and recent_preds[0] and recent_preds[0][0]:
                price = float(recent_preds[0][0])
                LOGGER.debug(f"✅ Found recorded price for {symbol} at {datetime.fromtimestamp(timestamp)}: ${price:.2f}")
                return price
        except Exception as e:
            LOGGER.debug(f"Could not query prediction store for historical price: {e}")
        
        # If we're within 1 hour of current time, use live price
        now = time.time()
        if abs(now - timestamp) < 3600:
            try:
                from core.crypto.crypto_providers import get_crypto_price_quorum
                import asyncio
                result = asyncio.run(get_crypto_price_quorum(symbol, use_cache=True))
                if result and result.get("price"):
                    return result["price"]
            except Exception as e:
                LOGGER.debug(f"Could not get live price: {e}")
        
        # For older timestamps (1h-30d back), try Polygon historical bars
        # Use hour-level aggregates which are available on free tier
        api_key = os.getenv("POLYGON_API_KEY")
        if api_key and abs(now - timestamp) < 2592000:  # Within 30 days
            try:
                dt = datetime.fromtimestamp(timestamp)
                # Use date-based range for better compatibility
                # Add ±1 day buffer to ensure we capture the target timestamp
                from datetime import timedelta
                target_date = dt.date()
                start_date = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
                end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
                
                # Try hourly bars first (more precise for intraday)
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/hour/{start_date}/{end_date}"
                params = {"apiKey": api_key, "sort": "asc", "limit": 1000}
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        # Find bar closest to target timestamp (results have "t" in milliseconds)
                        closest_bar = min(results, key=lambda r: abs(r["t"]/1000 - timestamp))
                        price = float(closest_bar["c"])  # Close price
                        bar_time = closest_bar["t"] / 1000
                        time_diff_hours = abs(bar_time - timestamp) / 3600
                        
                        # Only accept if within 12 hours (reasonable for 48h window)
                        if time_diff_hours < 12:
                            LOGGER.debug(f"✅ Polygon historical price for {symbol} at {dt}: ${price:.2f} (±{time_diff_hours:.1f}h)")
                            return price
                        else:
                            LOGGER.debug(f"⚠️  Closest Polygon bar for {symbol} is {time_diff_hours:.1f}h away")
                    else:
                        LOGGER.debug(f"⚠️  No Polygon bars found for {symbol} in date range {start_date} to {end_date}")
                else:
                    LOGGER.debug(f"⚠️  Polygon API returned status {response.status_code} for {symbol}")
            except Exception as e:
                LOGGER.debug(f"Polygon historical fetch failed for {symbol}: {e}")
        
        # Last resort: if within 24h, use current price (acceptable approximation)
        if abs(now - timestamp) < 86400:
            price = get_symbol_price(symbol)
            if price is not None:
                LOGGER.debug(f"⚠️  Using current price as approximation for {symbol}")
                return price
        
        # Can't get historical data - mark as no_data
        LOGGER.debug(f"⚠️  No historical price available for {symbol} at {datetime.fromtimestamp(timestamp)}")
        return None
        
    except Exception as e:
        LOGGER.debug(f"❌ Error fetching price for {symbol} (fast-failing): {e}")
        return None


def _store_outcome_success(
    prediction_id: int,
    symbol: str,
    closed_at: float,
    price_at_prediction: float,
    price_at_resolution: float,
    realized_move_pct: float,
    predicted_direction: str,
    actual_direction: str,
    hit_direction: int,
    predicted_confidence: float,
):
    """Store successful outcome in ghost_prediction_outcomes table."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise Exception("DATABASE_URL not set")
    
    conn = psycopg2.connect(database_url)
    try:
        cursor = conn.cursor()
        
        # Insert outcome
        cursor.execute("""
            INSERT INTO ghost_prediction_outcomes (
                prediction_id,
                symbol,
                closed_at, 
                price_at_prediction, 
                price_at_resolution,
                realized_move_pct, 
                predicted_direction, 
                actual_direction,
                hit_direction, 
                direction_threshold_pct,
                predicted_confidence,
                resolution_method, 
                resolution_provider,
                status
            ) VALUES (
                %s,
                %s,
                to_timestamp(%s), 
                %s, 
                %s,
                %s, 
                %s, 
                %s,
                %s, 
                %s,
                %s,
                %s, 
                %s,
                %s
            )
            ON CONFLICT (prediction_id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                closed_at = EXCLUDED.closed_at,
                price_at_resolution = EXCLUDED.price_at_resolution,
                realized_move_pct = EXCLUDED.realized_move_pct,
                actual_direction = EXCLUDED.actual_direction,
                hit_direction = EXCLUDED.hit_direction,
                status = EXCLUDED.status
        """, (
            prediction_id,
            symbol,
            closed_at,
            price_at_prediction,
            price_at_resolution,
            realized_move_pct,
            predicted_direction,
            actual_direction,
            hit_direction,
            DIRECTION_THRESHOLD_PCT,
            predicted_confidence,
            'live_provider',
            'unified_provider',
            'completed'
        ))
        
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


def _store_outcome_no_data(
    prediction_id: int,
    symbol: str,
    run_at: float,
    resolve_at: float,
    predicted_direction: str,
    predicted_confidence: float,
    notes: str,
):
    """Store outcome with status='no_data' when price cannot be fetched."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise Exception("DATABASE_URL not set")
    
    conn = psycopg2.connect(database_url)
    try:
        cursor = conn.cursor()
        
        # Insert outcome with NULL/0 for missing prices
        # price_at_prediction and price_at_resolution are NOT NULL, so use 0.0 as sentinel
        cursor.execute("""
            INSERT INTO ghost_prediction_outcomes (
                prediction_id,
                symbol,
                closed_at, 
                price_at_prediction,
                price_at_resolution,
                predicted_direction,
                predicted_confidence,
                hit_direction,
                status,
                notes,
                resolution_method
            ) VALUES (
                %s,
                %s,
                to_timestamp(%s), 
                %s,
                %s,
                %s,
                %s,
                NULL,
                %s,
                %s,
                %s
            )
            ON CONFLICT (prediction_id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                closed_at = EXCLUDED.closed_at,
                price_at_prediction = EXCLUDED.price_at_prediction,
                price_at_resolution = EXCLUDED.price_at_resolution,
                status = EXCLUDED.status,
                notes = EXCLUDED.notes
        """, (
            prediction_id,
            symbol,
            resolve_at,
            0.0,  # Sentinel value for missing price_at_prediction
            0.0,  # Sentinel value for missing price_at_resolution
            predicted_direction,
            predicted_confidence,
            'no_data',
            notes,
            'failed'
        ))
        
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


def start_reconciler_background_task():
    """
    Start outcome reconciler as background thread.
    Runs every hour to find and close expired predictions.
    """
    import threading
    
    interval_hours = int(os.getenv("OUTCOME_RECONCILE_INTERVAL_HOURS", "1"))
    enabled = int(os.getenv("OUTCOME_RECONCILE_ENABLED", "1"))
    
    if not enabled:
        LOGGER.info("⏸️  Outcome reconciler disabled (OUTCOME_RECONCILE_ENABLED=0)")
        return
    
    def reconcile_loop():
        """Background loop that runs reconciliation periodically."""
        LOGGER.info(f"🚀 Starting outcome reconciler background task (every {interval_hours}h)")
        
        # Sleep first on startup to avoid blocking server initialization
        time.sleep(60)  # Wait 60s for server to fully start before first run
        
        while True:
            try:
                reconcile_outcomes_v2()
            except Exception as e:
                LOGGER.error(f"❌ Reconciler loop error: {e}", exc_info=True)
            
            # Sleep for configured interval
            time.sleep(interval_hours * 3600)
    
    # Start background thread
    thread = threading.Thread(target=reconcile_loop, daemon=True, name="outcome_reconciler")
    thread.start()
    
    LOGGER.info(f"✅ Outcome reconciler started successfully (interval: {interval_hours}h)")


if __name__ == "__main__":
    # Can run manually for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    print("Running outcome reconciliation (one-time)...")
    reconcile_outcomes_v2()
    print("Done!")
