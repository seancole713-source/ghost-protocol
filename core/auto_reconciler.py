"""
Auto Reconciler - Runs hourly to resolve predictions
Connects accuracy tracking to real outcomes
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    psycopg2 = None

LOGGER = logging.getLogger(__name__)


def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for symbol"""
    try:
        # Try Coinbase first (crypto)
        from core.coinbase_provider import get_crypto_price
        price = get_crypto_price(symbol)
        if price and price > 0:
            return price
    except Exception as e:
        LOGGER.debug(f"Coinbase failed for {symbol}: {e}")
    
    try:
        # Try stock providers
        from core.providers.turbo_provider import turbo_stock_price
        price = turbo_stock_price(symbol)
        if price and price > 0:
            return price
    except Exception as e:
        LOGGER.debug(f"Stock provider failed for {symbol}: {e}")
    
    return None


def reconcile_pending_predictions() -> Dict[str, Any]:
    """
    Reconcile all pending predictions that have passed their horizon.
    Updates ghost_prediction_outcomes and accuracy_forecasts tables.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        LOGGER.warning("DATABASE_URL not set - reconciliation disabled")
        return {"reconciled": 0, "error": "No database"}
    
    if not HAS_POSTGRES:
        LOGGER.error("psycopg2 not installed")
        return {"reconciled": 0, "error": "No psycopg2"}
    
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Find predictions past their horizon (48h default)
        cur.execute("""
            SELECT p.id, p.symbol, p.predicted_direction, p.confidence, p.price_at_prediction,
                   p.prediction_time, COALESCE(p.horizon_hours, 48) as horizon_hours
            FROM ghost_predictions p
            LEFT JOIN ghost_prediction_outcomes o ON p.id = o.prediction_id
            WHERE o.id IS NULL
            AND p.prediction_time < NOW() - INTERVAL '1 hour' * COALESCE(p.horizon_hours, 48)
            AND p.prediction_time > NOW() - INTERVAL '7 days'
            ORDER BY p.prediction_time ASC
            LIMIT 100
        """)
        
        pending = cur.fetchall()
        LOGGER.info(f"Found {len(pending)} predictions to reconcile")
        
        reconciled = 0
        errors = 0
        
        for pred in pending:
            symbol = pred['symbol']
            pred_id = pred['id']
            direction = pred['predicted_direction']
            entry_price = pred['price_at_prediction']
            confidence = pred.get('confidence', 0.5)
            
            if not entry_price or entry_price <= 0:
                LOGGER.warning(f"Invalid entry price for {symbol}: {entry_price}")
                errors += 1
                continue
            
            # Get current price
            current_price = get_current_price(symbol)
            
            if not current_price:
                LOGGER.warning(f"Could not get price for {symbol}")
                errors += 1
                continue
            
            # Calculate outcome
            change_pct = ((current_price - entry_price) / entry_price) * 100
            
            if direction == "UP":
                was_correct = current_price > entry_price
            elif direction == "DOWN":
                was_correct = current_price < entry_price
            else:
                was_correct = abs(change_pct) < 1.0  # FLAT = within 1%
            
            # Insert outcome
            try:
                cur.execute("""
                    INSERT INTO ghost_prediction_outcomes
                    (prediction_id, symbol, direction, confidence, entry_price, exit_price,
                     actual_change_pct, was_correct, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (prediction_id) DO NOTHING
                """, (pred_id, symbol, direction, confidence, entry_price,
                      current_price, change_pct, was_correct))
                
                conn.commit()
                reconciled += 1
                
                LOGGER.info(f"Reconciled {symbol}: {direction} → {'✅' if was_correct else '❌'} "
                           f"({change_pct:+.2f}%)")
                
            except Exception as e:
                LOGGER.error(f"Failed to reconcile {pred_id}: {e}")
                errors += 1
                conn.rollback()
        
        # Get overall accuracy
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                ROUND(100.0 * SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as accuracy
            FROM ghost_prediction_outcomes
            WHERE created_at > NOW() - INTERVAL '7 days'
        """)
        stats = cur.fetchone()
        
        conn.close()
        
        return {
            "reconciled": reconciled,
            "errors": errors,
            "pending_checked": len(pending),
            "current_accuracy_7d": stats['accuracy'] if stats else None,
            "total_outcomes_7d": stats['total'] if stats else 0
        }
        
    except Exception as e:
        LOGGER.error(f"Reconciliation failed: {e}")
        return {"reconciled": 0, "error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = reconcile_pending_predictions()
    print(f"Reconciliation result: {result}")
