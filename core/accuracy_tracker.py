"""
Accuracy Tracker - FIXED VERSION
Uses PostgreSQL (DATABASE_URL) instead of SQLite
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

LOGGER = logging.getLogger(__name__)

@dataclass
class ForecastRecord:
    """Single forecast record"""
    forecast_id: int
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    target_price: Optional[float]
    horizon_hours: int
    created_at: datetime
    resolved_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    was_correct: Optional[bool] = None
    pnl_pct: Optional[float] = None


class AccuracyTracker:
    """
    Tracks prediction accuracy using PostgreSQL.
    NO SQLITE - all data persists in Railway PostgreSQL.
    """
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            LOGGER.warning("DATABASE_URL not set - accuracy tracking disabled")
            self._enabled = False
            return
            
        if not HAS_POSTGRES:
            LOGGER.error("psycopg2 not installed - accuracy tracking disabled")
            self._enabled = False
            return
            
        self._enabled = True
        self._init_tables()
        LOGGER.info("AccuracyTracker initialized with PostgreSQL")
    
    def _get_conn(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(self.database_url)
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS accuracy_forecasts (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        direction VARCHAR(10) NOT NULL,
                        confidence REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        target_price REAL,
                        horizon_hours INT DEFAULT 48,
                        created_at TIMESTAMP DEFAULT NOW(),
                        resolved_at TIMESTAMP,
                        exit_price REAL,
                        was_correct BOOLEAN,
                        pnl_pct REAL,
                        prediction_id INT
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_accuracy_symbol 
                    ON accuracy_forecasts(symbol);
                    
                    CREATE INDEX IF NOT EXISTS idx_accuracy_resolved 
                    ON accuracy_forecasts(resolved_at) WHERE resolved_at IS NULL;
                    
                    CREATE TABLE IF NOT EXISTS accuracy_daily_stats (
                        id SERIAL PRIMARY KEY,
                        date DATE NOT NULL UNIQUE,
                        total_predictions INT DEFAULT 0,
                        correct_predictions INT DEFAULT 0,
                        accuracy_pct REAL,
                        avg_confidence REAL,
                        total_pnl_pct REAL DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                conn.commit()
        LOGGER.info("Accuracy tables initialized in PostgreSQL")
    
    def record_forecast(self, symbol: str, direction: str, confidence: float,
                       entry_price: float, target_price: Optional[float] = None,
                       horizon_hours: int = 48, prediction_id: Optional[int] = None) -> int:
        """Record a new forecast"""
        if not self._enabled:
            return -1
            
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO accuracy_forecasts 
                    (symbol, direction, confidence, entry_price, target_price, 
                     horizon_hours, prediction_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                """, (symbol, direction, confidence, entry_price, target_price,
                      horizon_hours, prediction_id))
                forecast_id = cur.fetchone()[0]
                conn.commit()
                
        LOGGER.info(f"Recorded forecast {forecast_id}: {symbol} {direction} @ {confidence:.0%}")
        return forecast_id
    
    def resolve_forecast(self, forecast_id: int, exit_price: float) -> Dict[str, Any]:
        """Resolve a forecast with actual outcome"""
        if not self._enabled:
            return {}
            
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get forecast
                cur.execute("""
                    SELECT * FROM accuracy_forecasts WHERE id = %s
                """, (forecast_id,))
                forecast = cur.fetchone()
                
                if not forecast:
                    return {"error": "Forecast not found"}
                
                # Calculate outcome
                entry = forecast['entry_price']
                direction = forecast['direction']
                pnl_pct = ((exit_price - entry) / entry) * 100
                
                # Determine if correct
                if direction == "UP":
                    was_correct = exit_price > entry
                elif direction == "DOWN":
                    was_correct = exit_price < entry
                else:
                    was_correct = abs(pnl_pct) < 1.0  # FLAT = within 1%
                
                # Update forecast
                cur.execute("""
                    UPDATE accuracy_forecasts SET
                        resolved_at = NOW(),
                        exit_price = %s,
                        was_correct = %s,
                        pnl_pct = %s
                    WHERE id = %s
                """, (exit_price, was_correct, pnl_pct, forecast_id))
                
                # Update daily stats
                self._update_daily_stats(cur)
                conn.commit()
                
        return {
            "forecast_id": forecast_id,
            "symbol": forecast['symbol'],
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "was_correct": was_correct
        }
    
    def _update_daily_stats(self, cur):
        """Update daily statistics"""
        today = datetime.now().date()
        cur.execute("""
            INSERT INTO accuracy_daily_stats (date, total_predictions, correct_predictions,
                                              accuracy_pct, avg_confidence, total_pnl_pct)
            SELECT 
                %s as date,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                ROUND(100.0 * SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as acc,
                ROUND(AVG(confidence) * 100, 1) as avg_conf,
                ROUND(SUM(COALESCE(pnl_pct, 0)), 2) as pnl
            FROM accuracy_forecasts
            WHERE DATE(created_at) = %s AND resolved_at IS NOT NULL
            ON CONFLICT (date) DO UPDATE SET
                total_predictions = EXCLUDED.total_predictions,
                correct_predictions = EXCLUDED.correct_predictions,
                accuracy_pct = EXCLUDED.accuracy_pct,
                avg_confidence = EXCLUDED.avg_confidence,
                total_pnl_pct = EXCLUDED.total_pnl_pct,
                updated_at = NOW()
        """, (today, today))
    
    def get_accuracy_stats(self, days: int = 30, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get accuracy statistics"""
        if not self._enabled:
            return {"enabled": False, "error": "Accuracy tracking disabled"}
            
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query
                where_clauses = ["resolved_at IS NOT NULL"]
                params = []
                
                if days:
                    where_clauses.append("created_at > NOW() - INTERVAL '%s days'")
                    params.append(days)
                    
                if symbol:
                    where_clauses.append("symbol = %s")
                    params.append(symbol)
                
                where_sql = " AND ".join(where_clauses)
                
                cur.execute(f"""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_predictions,
                        ROUND(100.0 * SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) / 
                              NULLIF(COUNT(*), 0), 1) as accuracy_pct,
                        ROUND(AVG(confidence) * 100, 1) as avg_confidence,
                        ROUND(SUM(COALESCE(pnl_pct, 0)), 2) as total_pnl_pct,
                        MIN(created_at) as first_prediction,
                        MAX(resolved_at) as last_resolved
                    FROM accuracy_forecasts
                    WHERE {where_sql}
                """, params)
                
                stats = dict(cur.fetchone())
                
                # Get by direction
                cur.execute(f"""
                    SELECT 
                        direction,
                        COUNT(*) as total,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                        ROUND(100.0 * SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) / 
                              NULLIF(COUNT(*), 0), 1) as accuracy
                    FROM accuracy_forecasts
                    WHERE {where_sql}
                    GROUP BY direction
                """, params)
                
                stats['by_direction'] = {row['direction']: dict(row) for row in cur.fetchall()}
                
                # Get top symbols
                cur.execute(f"""
                    SELECT 
                        symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                        ROUND(100.0 * SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) / 
                              NULLIF(COUNT(*), 0), 1) as accuracy
                    FROM accuracy_forecasts
                    WHERE {where_sql}
                    GROUP BY symbol
                    ORDER BY total DESC
                    LIMIT 20
                """, params)
                
                stats['by_symbol'] = [dict(row) for row in cur.fetchall()]
                
        return stats
    
    def get_pending_forecasts(self) -> List[Dict]:
        """Get forecasts that need resolution"""
        if not self._enabled:
            return []
            
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM accuracy_forecasts
                    WHERE resolved_at IS NULL
                    AND created_at < NOW() - INTERVAL '1 hour' * horizon_hours
                    ORDER BY created_at ASC
                    LIMIT 100
                """)
                return [dict(row) for row in cur.fetchall()]
    
    def get_recent_forecasts(self, limit: int = 50) -> List[Dict]:
        """Get recent forecasts"""
        if not self._enabled:
            return []
            
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM accuracy_forecasts
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                return [dict(row) for row in cur.fetchall()]
    
    def calculate_metrics(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Legacy method for compatibility - redirects to get_accuracy_stats"""
        return self.get_accuracy_stats(days=days, symbol=symbol)


# Singleton instance
_tracker: Optional[AccuracyTracker] = None

def get_accuracy_tracker() -> AccuracyTracker:
    """Get or create AccuracyTracker singleton"""
    global _tracker
    if _tracker is None:
        _tracker = AccuracyTracker()
    return _tracker


# Legacy function exports for compatibility
def get_accuracy_report(symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
    """Legacy function - get accuracy report"""
    tracker = get_accuracy_tracker()
    return tracker.get_accuracy_stats(days=days, symbol=symbol)
