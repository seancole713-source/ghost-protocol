"""
Pattern Performance Tracker (Jan 31, 2026)

Tracks whether Ghost's pattern detections actually make money.
This validates the 85% accuracy claims in event_memory.py.

Database table: pattern_performance
Reconciler job: Checks outcomes 24-48h after detection
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class PatternDetection:
    """A detected pattern waiting for outcome verification."""
    id: Optional[int]
    pattern_type: str  # 'ELON_TWEET', 'FED_RATE_CUT', etc.
    symbol: str
    detected_at: float  # Unix timestamp
    detection_confidence: float
    entry_price: float
    direction: str  # 'UP' or 'DOWN'
    outcome_24h: Optional[float] = None  # Actual % change after 24h
    outcome_48h: Optional[float] = None  # Actual % change after 48h
    was_profitable: Optional[bool] = None
    created_at: Optional[float] = None


def _get_db_connection():
    """Get PostgreSQL connection if available."""
    try:
        import psycopg2
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            return psycopg2.connect(db_url)
    except Exception as e:
        LOGGER.debug(f"[PATTERN_TRACKER] No PostgreSQL: {e}")
    return None


def _ensure_table():
    """Create pattern_performance table if it doesn't exist."""
    conn = _get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pattern_performance (
                id SERIAL PRIMARY KEY,
                pattern_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                detected_at TIMESTAMP NOT NULL,
                detection_confidence FLOAT,
                entry_price FLOAT NOT NULL,
                direction VARCHAR(10) NOT NULL,
                outcome_24h FLOAT,
                outcome_48h FLOAT,
                was_profitable BOOLEAN,
                created_at TIMESTAMP DEFAULT NOW(),
                
                -- Index for reconciliation queries
                CONSTRAINT pattern_performance_unique UNIQUE (pattern_type, symbol, detected_at)
            );
            
            -- Index for finding patterns needing reconciliation
            CREATE INDEX IF NOT EXISTS idx_pattern_needs_reconcile 
            ON pattern_performance (detected_at) 
            WHERE outcome_24h IS NULL;
            
            -- Index for accuracy queries
            CREATE INDEX IF NOT EXISTS idx_pattern_accuracy 
            ON pattern_performance (pattern_type, was_profitable);
        """)
        conn.commit()
        LOGGER.info("[PATTERN_TRACKER] Table pattern_performance ready")
        return True
    except Exception as e:
        LOGGER.error(f"[PATTERN_TRACKER] Table creation failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


# In-memory fallback for when PostgreSQL is unavailable
_MEMORY_PATTERNS: List[PatternDetection] = []


def record_pattern_detection(
    pattern_type: str,
    symbol: str,
    direction: str,
    entry_price: float,
    confidence: float = 0.0,
) -> Optional[int]:
    """
    Record a pattern detection for later outcome verification.
    
    Called by Event Detector when it finds a pattern.
    A background job will fill in outcome_24h/48h later.
    
    Returns: pattern_id or None if failed
    """
    global _MEMORY_PATTERNS
    
    detected_at = datetime.utcnow()
    
    # Try PostgreSQL first
    conn = _get_db_connection()
    if conn:
        try:
            _ensure_table()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO pattern_performance 
                (pattern_type, symbol, detected_at, detection_confidence, entry_price, direction)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (pattern_type, symbol, detected_at) DO NOTHING
                RETURNING id
            """, (pattern_type, symbol, detected_at, confidence, entry_price, direction))
            
            result = cur.fetchone()
            pattern_id = result[0] if result else None
            conn.commit()
            
            LOGGER.info(
                f"[PATTERN_TRACKER] Recorded: {pattern_type} on {symbol} @ ${entry_price:.2f} "
                f"({direction}, {confidence:.0%} confidence)"
            )
            return pattern_id
        except Exception as e:
            LOGGER.error(f"[PATTERN_TRACKER] Failed to record: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    # Fallback to memory
    detection = PatternDetection(
        id=len(_MEMORY_PATTERNS) + 1,
        pattern_type=pattern_type,
        symbol=symbol,
        detected_at=detected_at.timestamp(),
        detection_confidence=confidence,
        entry_price=entry_price,
        direction=direction,
        created_at=time.time(),
    )
    _MEMORY_PATTERNS.append(detection)
    LOGGER.info(f"[PATTERN_TRACKER] Recorded (memory): {pattern_type} on {symbol}")
    return detection.id


async def reconcile_pattern_outcomes():
    """
    Check outcomes for patterns detected 24-48h ago.
    
    Called by a background job to update was_profitable.
    """
    conn = _get_db_connection()
    if not conn:
        LOGGER.warning("[PATTERN_TRACKER] No database for reconciliation")
        return {"reconciled": 0, "error": "no_database"}
    
    try:
        cur = conn.cursor()
        
        # Find patterns detected 24-48h ago that need reconciliation
        cur.execute("""
            SELECT id, pattern_type, symbol, detected_at, entry_price, direction
            FROM pattern_performance
            WHERE outcome_24h IS NULL
            AND detected_at < NOW() - INTERVAL '24 hours'
            AND detected_at > NOW() - INTERVAL '72 hours'
            LIMIT 50
        """)
        
        patterns = cur.fetchall()
        reconciled = 0
        
        for pattern_id, pattern_type, symbol, detected_at, entry_price, direction in patterns:
            try:
                # Get current price
                from core.price_quorum import get_price_quorum
                from core.providers.stock_providers import get_stock_providers
                from core.providers.crypto_providers import get_crypto_providers
                from core.asset_classification import is_crypto_symbol
                
                quorum = get_price_quorum()
                
                if is_crypto_symbol(symbol):
                    providers = get_crypto_providers(symbol)
                else:
                    providers = get_stock_providers(symbol)
                
                decision = quorum.get_price(symbol, providers, is_market_open=True)
                
                if not decision or not decision.price:
                    LOGGER.warning(f"[PATTERN_TRACKER] No price for {symbol}, skipping")
                    continue
                
                current_price = decision.price
                
                # Calculate actual change
                hours_elapsed = (datetime.utcnow() - detected_at).total_seconds() / 3600
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Determine if profitable based on direction
                if direction == "UP":
                    was_profitable = price_change_pct > 0
                else:
                    was_profitable = price_change_pct < 0
                
                # Update record
                if hours_elapsed >= 48:
                    cur.execute("""
                        UPDATE pattern_performance
                        SET outcome_48h = %s, was_profitable = %s
                        WHERE id = %s
                    """, (price_change_pct, was_profitable, pattern_id))
                else:
                    cur.execute("""
                        UPDATE pattern_performance
                        SET outcome_24h = %s, was_profitable = %s
                        WHERE id = %s
                    """, (price_change_pct, was_profitable, pattern_id))
                
                conn.commit()
                reconciled += 1
                
                status = "✅" if was_profitable else "❌"
                LOGGER.info(
                    f"[PATTERN_TRACKER] {status} {pattern_type} on {symbol}: "
                    f"Entry ${entry_price:.2f} → ${current_price:.2f} ({price_change_pct:+.1f}%) "
                    f"Direction: {direction}, Profitable: {was_profitable}"
                )
                
            except Exception as e:
                LOGGER.warning(f"[PATTERN_TRACKER] Failed to reconcile {symbol}: {e}")
        
        return {"reconciled": reconciled, "pending": len(patterns)}
        
    except Exception as e:
        LOGGER.error(f"[PATTERN_TRACKER] Reconciliation failed: {e}")
        return {"reconciled": 0, "error": str(e)}
    finally:
        conn.close()


def get_pattern_accuracy() -> Dict[str, Any]:
    """
    Get REAL accuracy for each pattern type based on Ghost's own detections.
    
    Returns:
        {
            "ELON_TWEET": {"detections": 12, "profitable": 7, "accuracy": 58.3},
            "FED_RATE_CUT": {"detections": 3, "profitable": 2, "accuracy": 66.7},
            ...
            "overall": {"detections": 50, "profitable": 28, "accuracy": 56.0}
        }
    """
    conn = _get_db_connection()
    if not conn:
        # Return memory-based stats
        if not _MEMORY_PATTERNS:
            return {"error": "no_data", "overall": {"detections": 0}}
        
        stats = {}
        for p in _MEMORY_PATTERNS:
            if p.pattern_type not in stats:
                stats[p.pattern_type] = {"detections": 0, "profitable": 0}
            stats[p.pattern_type]["detections"] += 1
            if p.was_profitable:
                stats[p.pattern_type]["profitable"] += 1
        
        # Calculate accuracy
        for pattern_type, data in stats.items():
            if data["detections"] > 0:
                data["accuracy"] = round(100 * data["profitable"] / data["detections"], 1)
            else:
                data["accuracy"] = 0.0
        
        total_det = sum(d["detections"] for d in stats.values())
        total_prof = sum(d["profitable"] for d in stats.values())
        stats["overall"] = {
            "detections": total_det,
            "profitable": total_prof,
            "accuracy": round(100 * total_prof / total_det, 1) if total_det > 0 else 0.0,
        }
        
        return stats
    
    try:
        cur = conn.cursor()
        
        # Get accuracy by pattern type
        cur.execute("""
            SELECT 
                pattern_type,
                COUNT(*) as detections,
                SUM(CASE WHEN was_profitable THEN 1 ELSE 0 END) as profitable
            FROM pattern_performance
            WHERE was_profitable IS NOT NULL
            GROUP BY pattern_type
            ORDER BY COUNT(*) DESC
        """)
        
        results = cur.fetchall()
        stats = {}
        
        for pattern_type, detections, profitable in results:
            profitable = profitable or 0
            accuracy = round(100 * profitable / detections, 1) if detections > 0 else 0.0
            stats[pattern_type] = {
                "detections": detections,
                "profitable": profitable,
                "accuracy": accuracy,
            }
        
        # Calculate overall
        cur.execute("""
            SELECT 
                COUNT(*) as detections,
                SUM(CASE WHEN was_profitable THEN 1 ELSE 0 END) as profitable
            FROM pattern_performance
            WHERE was_profitable IS NOT NULL
        """)
        
        total_det, total_prof = cur.fetchone()
        total_prof = total_prof or 0
        
        stats["overall"] = {
            "detections": total_det or 0,
            "profitable": total_prof,
            "accuracy": round(100 * total_prof / total_det, 1) if total_det else 0.0,
        }
        
        # Get pending (not yet reconciled)
        cur.execute("""
            SELECT COUNT(*) FROM pattern_performance WHERE was_profitable IS NULL
        """)
        pending = cur.fetchone()[0]
        stats["pending_reconciliation"] = pending
        
        return stats
        
    except Exception as e:
        LOGGER.error(f"[PATTERN_TRACKER] Failed to get accuracy: {e}")
        return {"error": str(e), "overall": {"detections": 0}}
    finally:
        conn.close()


def get_recent_detections(limit: int = 20) -> List[Dict]:
    """Get most recent pattern detections with outcomes."""
    conn = _get_db_connection()
    if not conn:
        return [
            {
                "pattern_type": p.pattern_type,
                "symbol": p.symbol,
                "detected_at": datetime.fromtimestamp(p.detected_at).isoformat(),
                "entry_price": p.entry_price,
                "direction": p.direction,
                "outcome_24h": p.outcome_24h,
                "was_profitable": p.was_profitable,
            }
            for p in _MEMORY_PATTERNS[-limit:]
        ]
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT pattern_type, symbol, detected_at, entry_price, direction,
                   outcome_24h, outcome_48h, was_profitable
            FROM pattern_performance
            ORDER BY detected_at DESC
            LIMIT %s
        """, (limit,))
        
        results = []
        for row in cur.fetchall():
            results.append({
                "pattern_type": row[0],
                "symbol": row[1],
                "detected_at": row[2].isoformat() if row[2] else None,
                "entry_price": row[3],
                "direction": row[4],
                "outcome_24h": row[5],
                "outcome_48h": row[6],
                "was_profitable": row[7],
            })
        
        return results
        
    except Exception as e:
        LOGGER.error(f"[PATTERN_TRACKER] Failed to get recent: {e}")
        return []
    finally:
        conn.close()


# Initialize table on module load
_ensure_table()
