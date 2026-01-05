#!/usr/bin/env python3
"""
Hourly Actual Price Collector for Ghost Predictions
====================================================
Background task that collects actual prices every hour for ALL active predictions.

This ensures the reconciler has price data to compare forecasts against, solving
the "insufficient aligned points (0)" error.

How it works:
1. Runs every hour via scheduler
2. Queries all predictions with open windows (run_at + 48h > now)
3. Fetches current price for each symbol
4. Stores in prediction_store.actual_points

This is CRITICAL for Ghost's 70% accuracy measurement to work!
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional

LOGGER = logging.getLogger("ghost.actual_price_collector")


def get_current_price(symbol: str) -> Optional[float]:
    """
    Fetch current price for a symbol (crypto or stock).
    Uses multiple fallbacks for reliability.
    """
    symbol = symbol.upper()
    
    # Known crypto symbols
    CRYPTO_SYMBOLS = {
        'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK',
        'DOGE', 'SHIB', 'LTC', 'TRX', 'TON', 'XLM', 'ATOM', 'UNI', 'AAVE', 'MKR',
        'PEPE', 'BONK', 'WIF', 'FLOKI', 'FET', 'NEAR', 'INJ', 'SUI', 'SEI', 'TIA',
        'OP', 'ARB', 'APE', 'SAND', 'MANA', 'AXS', 'GRT', 'CRV', 'COMP', 'SNX',
    }
    
    try:
        # Try Coinbase for crypto
        if symbol in CRYPTO_SYMBOLS:
            try:
                from core.coinbase_provider import get_crypto_price
                price = get_crypto_price(symbol)
                if price and price > 0:
                    return price
            except Exception as e:
                LOGGER.debug(f"Coinbase failed for {symbol}: {e}")
        
        # Fall back to TurboProvider
        try:
            from core.providers.turbo_provider import get_turbo_provider
            provider = get_turbo_provider()
            
            if symbol in CRYPTO_SYMBOLS:
                result = provider.turbo_crypto_price(symbol, max_budget_s=2.0)
            else:
                result = provider.turbo_stock_price(symbol, max_budget_s=2.0)
            
            if result.get("ok") and result.get("price"):
                return float(result["price"])
        except Exception as e:
            LOGGER.debug(f"TurboProvider failed for {symbol}: {e}")
        
        return None
    except Exception as e:
        LOGGER.warning(f"All price sources failed for {symbol}: {e}")
        return None


def collect_actual_prices() -> dict:
    """
    Collect current prices for all active predictions.
    
    Returns:
        dict with counts: {"collected": N, "failed": N, "symbols": N}
    """
    LOGGER.info("🔄 Starting actual price collection...")
    
    try:
        from core.prediction_store import get_prediction_store
        store = get_prediction_store()
        
        # Get predictions with open windows (where 48h hasn't elapsed yet)
        now = int(time.time())
        
        # Query predictions where run_at + horizon_h * 3600 > now
        # This means the prediction window is still open
        try:
            if hasattr(store, 'get_active_predictions'):
                active_predictions = store.get_active_predictions()
            else:
                # Fallback: get recent predictions
                active_predictions = store.get_recent_predictions(limit=100)
        except Exception as e:
            LOGGER.warning(f"Could not get active predictions: {e}")
            active_predictions = []
        
        if not active_predictions:
            LOGGER.info("✅ No active predictions to collect prices for")
            return {"collected": 0, "failed": 0, "symbols": 0}
        
        # Get unique symbols
        symbols = list(set(p.get('symbol', '').upper() for p in active_predictions if p.get('symbol')))
        
        LOGGER.info(f"📊 Found {len(symbols)} unique symbols from {len(active_predictions)} active predictions")
        
        collected_count = 0
        failed_count = 0
        
        for symbol in symbols:
            try:
                price = get_current_price(symbol)
                
                if price and price > 0:
                    # Store the actual price point
                    timestamp = int(time.time())
                    
                    try:
                        if hasattr(store, 'append_actual_point'):
                            store.append_actual_point(symbol, timestamp, price)
                        elif hasattr(store, 'backend') and hasattr(store.backend, 'append_actual_point'):
                            store.backend.append_actual_point(symbol, timestamp, price)
                        else:
                            # Direct DB append
                            _append_actual_point_direct(symbol, timestamp, price)
                        
                        collected_count += 1
                        LOGGER.debug(f"✅ Collected {symbol}: ${price:.2f}")
                    except Exception as e:
                        LOGGER.error(f"Failed to store price for {symbol}: {e}")
                        failed_count += 1
                else:
                    LOGGER.warning(f"⚠️ No price available for {symbol}")
                    failed_count += 1
                    
            except Exception as e:
                LOGGER.error(f"❌ Error collecting price for {symbol}: {e}")
                failed_count += 1
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        LOGGER.info(
            f"✅ Price collection complete: "
            f"{collected_count} collected, {failed_count} failed, "
            f"{len(symbols)} symbols"
        )
        
        return {
            "collected": collected_count,
            "failed": failed_count,
            "symbols": len(symbols),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        LOGGER.error(f"❌ Actual price collection failed: {e}", exc_info=True)
        return {"collected": 0, "failed": 0, "symbols": 0, "error": str(e)}


def _append_actual_point_direct(symbol: str, timestamp: int, price: float):
    """
    Direct database append for actual price points.
    Fallback when prediction_store methods aren't available.
    """
    import os
    
    # Try PostgreSQL first
    db_url = os.getenv("DATABASE_URL")
    if db_url and os.getenv("PREDICTION_STORE_ENGINE") == "postgres":
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ghost_actual_prices (symbol, timestamp, price, created_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (symbol, timestamp) DO UPDATE SET price = EXCLUDED.price
                """,
                (symbol, timestamp, price)
            )
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception as e:
            LOGGER.debug(f"Postgres append failed: {e}")
    
    # Fall back to SQLite
    try:
        import sqlite3
        db_path = os.getenv("PREDICTION_DB_PATH", "/app/data/ghost_predictions.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO actual_points (symbol, ts, price)
            VALUES (?, ?, ?)
            """,
            (symbol, timestamp, price)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        LOGGER.error(f"SQLite append failed: {e}")


def start_collector_scheduler():
    """
    Start the hourly price collector as a background thread.
    Call this from your main app initialization.
    """
    import threading
    
    def collector_loop():
        LOGGER.info("🚀 Actual price collector started (hourly)")
        
        while True:
            try:
                # Collect prices
                result = collect_actual_prices()
                LOGGER.info(f"📊 Collection result: {result}")
                
            except Exception as e:
                LOGGER.error(f"❌ Collector loop error: {e}", exc_info=True)
            
            # Sleep for 1 hour
            time.sleep(3600)
    
    thread = threading.Thread(target=collector_loop, daemon=True, name="actual_price_collector")
    thread.start()
    LOGGER.info("✅ Actual price collector thread started")
    return thread


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    result = collect_actual_prices()
    print(f"Result: {result}")
