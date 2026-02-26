#!/usr/bin/env python3
"""
Ghost Prediction Storage Abstraction
=====================================

Provides a unified interface for storing predictions in either SQLite or PostgreSQL.

Environment Variables:
- PREDICTION_STORE_ENGINE: "sqlite" (default) or "postgres"
- PREDICTION_DUAL_WRITE: "1" to enable dual-write mode (default: "0")
- GHOST_PREDICT_DB: SQLite database path (default: ./data/ghost_predictions.db)
- DATABASE_URL: PostgreSQL connection string

Default Behavior:
- Uses SQLite backend if PREDICTION_STORE_ENGINE is not set
- Dual-write is DISABLED by default
- All existing prediction functionality remains unchanged
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("core.prediction_store")

# =============================================================================
# PRICE VALIDATION - Prevent corrupt data from poisoning learning system
# =============================================================================
# Minimum sane prices - reject anything below these values as corrupt
MIN_VALID_PRICES = {
    'BTC': 10000,     # BTC won't be $38 ever again
    'ETH': 500,       # ETH won't be $5 again
    'SOL': 5,         # SOL minimum
    'BNB': 50,        # BNB minimum
    'XRP': 0.10,      # XRP minimum
    'ADA': 0.02,      # ADA minimum
    'AVAX': 3,        # AVAX minimum
    'DOGE': 0.005,    # DOGE minimum
    'DOT': 1,         # DOT minimum
    'LINK': 2,        # LINK minimum
    'MATIC': 0.10,    # MATIC minimum
    'SHIB': 0.000001, # SHIB minimum
    'LTC': 20,        # LTC minimum
    'ATOM': 2,        # ATOM minimum
    'UNI': 1,         # UNI minimum
}

# Maximum sane prices - reject anything above as corrupt (e.g., BTC at $10M)
MAX_VALID_PRICES = {
    'BTC': 500000,    # BTC won't be $500K soon
    'ETH': 50000,     # ETH won't be $50K soon
    'SOL': 2000,      # SOL won't be $2K soon
    'BNB': 5000,      # BNB won't be $5K soon
    'XRP': 50,        # XRP won't be $50 soon
    'ADA': 20,        # ADA won't be $20 soon
    'DOGE': 5,        # DOGE won't be $5 soon
}

# Default bounds for unknown symbols
DEFAULT_MIN_PRICE = 0.0000001  # 7 decimal places minimum
DEFAULT_MAX_PRICE = 1000000    # $1M maximum for unknown symbols


def validate_price(symbol: str, price: float, context: str = "") -> tuple[bool, str]:
    """
    Validate that a price is within sane bounds for the given symbol.
    
    Args:
        symbol: The trading symbol (BTC, ETH, AAPL, etc.)
        price: The price to validate
        context: Optional context for logging (e.g., "prediction", "reconciliation")
    
    Returns:
        (is_valid, reason): Tuple of boolean and reason string
    """
    if price is None:
        return False, f"Price is None"
    
    if not isinstance(price, (int, float)):
        return False, f"Price is not numeric: {type(price)}"
    
    if price <= 0:
        return False, f"Price is zero or negative: {price}"
    
    symbol_upper = symbol.upper()
    
    # Check minimum bound
    min_price = MIN_VALID_PRICES.get(symbol_upper, DEFAULT_MIN_PRICE)
    if price < min_price:
        return False, f"Price ${price:.8f} below minimum ${min_price} for {symbol_upper}"
    
    # Check maximum bound
    max_price = MAX_VALID_PRICES.get(symbol_upper, DEFAULT_MAX_PRICE)
    if price > max_price:
        return False, f"Price ${price:.2f} above maximum ${max_price} for {symbol_upper}"
    
    return True, "OK"


class PriceValidationError(Exception):
    """Raised when a price fails validation."""
    pass


# Configuration
PREDICTION_STORE_ENGINE = os.getenv("PREDICTION_STORE_ENGINE", "sqlite").lower()
PREDICTION_DUAL_WRITE = os.getenv("PREDICTION_DUAL_WRITE", "0") == "1"
DB_PATH = os.getenv("GHOST_PREDICT_DB", "./data/ghost_predictions.db")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Detect PostgreSQL availability
IS_POSTGRES_AVAILABLE = DATABASE_URL.startswith(("postgres://", "postgresql://"))

if PREDICTION_STORE_ENGINE == "postgres" and not IS_POSTGRES_AVAILABLE:
    LOGGER.warning("⚠️  PREDICTION_STORE_ENGINE=postgres but DATABASE_URL not set. Falling back to SQLite.")
    PREDICTION_STORE_ENGINE = "sqlite"


class PredictionRejected(Exception):
    """Raised when strict fail-closed rules reject persistence."""


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_ratio(v: object) -> tuple[int, int]:
    """Parse values like "15/15" into (15, 15)."""
    try:
        s = str(v or "").strip()
        left = s.split()[0]
        a, b = left.split("/", 1)
        return int(a), int(b)
    except Exception:
        return 0, 0


def _should_strict_fail_closed() -> bool:
    # Mirror the production intent used elsewhere: if live is enforced, fail closed.
    return (
        _truthy(os.getenv("PREDICT_FAIL_CLOSED", "0"))
        or _truthy(os.getenv("PRICE_STRICT_LIVE", "0"))
        or _truthy(os.getenv("ENFORCE_LIVE", "0"))
    )


def _fail_closed_reason_from_features(features: dict[str, Any]) -> str | None:
    """Return a rejection reason if inputs look degraded, else None.

    This is intentionally defensive and only triggers when metadata is present.
    """
    if not _should_strict_fail_closed():
        return None

    # Optional: overall feature availability
    available = features.get("available_count")
    total = features.get("feature_count")
    availability_pct = features.get("feature_availability_pct")

    try:
        min_features_for_signal = int(os.getenv("MIN_FEATURES_FOR_SIGNAL", "10"))
    except Exception:
        min_features_for_signal = 10
    try:
        min_availability_pct = float(os.getenv("MIN_FEATURE_AVAILABILITY_PCT", "50"))
    except Exception:
        min_availability_pct = 50.0

    degraded = False
    if isinstance(available, (int, float)):
        degraded = degraded or int(available) < min_features_for_signal
    if isinstance(availability_pct, (int, float)):
        degraded = degraded or float(availability_pct) < min_availability_pct
    elif isinstance(available, (int, float)) and isinstance(total, (int, float)) and float(total) > 0:
        degraded = degraded or ((float(available) / float(total)) * 100.0) < min_availability_pct

    # Optional: pillar breakdown
    pillar_breakdown = features.get("pillar_breakdown") or features.get("feature_availability")
    if isinstance(pillar_breakdown, dict):
        price_avail, _ = _parse_ratio(pillar_breakdown.get("price_engine"))
        tech_avail, _ = _parse_ratio(pillar_breakdown.get("technical_engine"))
        vol_avail, _ = _parse_ratio(pillar_breakdown.get("volume_engine"))

        # If we have pillar info, treat missing tech/vol as fatal in strict mode.
        if tech_avail <= 0 or vol_avail <= 0:
            return (
                f"fail_closed_store: pillars(price/tech/vol)={price_avail}/{tech_avail}/{vol_avail}, "
                f"available/total={available}/{total}, availability_pct={availability_pct}"
            )

    if degraded and (available is not None or availability_pct is not None or total is not None):
        return (
            f"fail_closed_store: degraded=True, available/total={available}/{total}, "
            f"availability_pct={availability_pct}"
        )

    return None


class PredictionStore:
    """
    Unified interface for prediction storage.
    
    Supports both SQLite and PostgreSQL backends with optional dual-write mode.
    """
    
    def __init__(self, backend):
        self.backend = backend
        self.dual_write_backend = None
        
        if PREDICTION_DUAL_WRITE:
            # Enable dual-write mode (write to both backends)
            if PREDICTION_STORE_ENGINE == "sqlite" and IS_POSTGRES_AVAILABLE:
                self.dual_write_backend = PostgresBackend()
                LOGGER.info("✅ Dual-write enabled: SQLite (primary) + PostgreSQL (secondary)")
            elif PREDICTION_STORE_ENGINE == "postgres":
                self.dual_write_backend = SQLiteBackend()
                LOGGER.info("✅ Dual-write enabled: PostgreSQL (primary) + SQLite (secondary)")
    
    def save_prediction(
        self,
        symbol: str,
        forecast_points: list[tuple[float, float]],
        method: str,
        confidence: float,
        direction: str,
        features: dict[str, Any],
        params: dict[str, Any],
        tag: str = "",
    ) -> int:
        """
        Save a prediction to storage with dual-write support.
        
        Returns:
            prediction_id (int) from primary backend
        """
        # =====================================================================
        # VALIDATION: Reject invalid predictions before saving
        # =====================================================================
        # BUGFIX: Reject FLAT directions - they're not actionable trades
        if direction not in ("UP", "DOWN"):
            LOGGER.warning(
                f"[PREDICTION_STORE] Rejecting {symbol} prediction with invalid direction '{direction}' - "
                f"must be UP or DOWN. Converting to UP with reduced confidence."
            )
            direction = "UP"  # Default to bullish
            confidence = max(0.50, confidence * 0.8) if confidence else 0.50
        
        # BUGFIX: Ensure confidence is valid (not 0.0 or None)
        if confidence is None or confidence <= 0.0:
            LOGGER.warning(
                f"[PREDICTION_STORE] Rejecting {symbol} prediction with invalid confidence {confidence} - "
                f"setting to minimum 0.50"
            )
            confidence = 0.50
        
        # Clamp confidence to valid range
        confidence = max(0.50, min(0.95, confidence))
        
        # Production-grade fail-closed: reject persistence on degraded inputs.
        reason = _fail_closed_reason_from_features(features or {})
        if reason:
            raise PredictionRejected(reason)

        primary_backend_name = self.backend.__class__.__name__
        
        # Write to primary backend
        start_time = time.time()
        prediction_id = self.backend.save_prediction(
            symbol, forecast_points, method, confidence, direction, features, params, tag
        )
        primary_duration_ms = int((time.time() - start_time) * 1000)
        
        LOGGER.info(
            f"[{primary_backend_name}] Saved prediction {prediction_id} for {symbol} "
            f"({len(forecast_points)} points, {primary_duration_ms}ms)"
        )
        
        # Write to secondary backend if dual-write enabled
        if self.dual_write_backend:
            secondary_backend_name = self.dual_write_backend.__class__.__name__
            try:
                start_time = time.time()
                secondary_id = self.dual_write_backend.save_prediction(
                    symbol, forecast_points, method, confidence, direction, features, params, tag
                )
                secondary_duration_ms = int((time.time() - start_time) * 1000)
                
                LOGGER.info(
                    f"[DUAL-WRITE] [{secondary_backend_name}] Saved prediction {secondary_id} "
                    f"for {symbol} ({secondary_duration_ms}ms)"
                )
            except Exception as e:
                LOGGER.error(
                    f"[DUAL-WRITE] [{secondary_backend_name}] Failed for {symbol}: {e}",
                    exc_info=True
                )
        
        return prediction_id
    
    def append_actual_points(self, prediction_id: int, actual_points: list[tuple[float, float]]):
        """Append actual price points with dual-write support."""
        primary_backend_name = self.backend.__class__.__name__
        
        self.backend.append_actual_points(prediction_id, actual_points)
        
        LOGGER.debug(
            f"[{primary_backend_name}] Appended {len(actual_points)} actual points to prediction {prediction_id}"
        )
        
        if self.dual_write_backend:
            secondary_backend_name = self.dual_write_backend.__class__.__name__
            try:
                self.dual_write_backend.append_actual_points(prediction_id, actual_points)
                LOGGER.debug(
                    f"[DUAL-WRITE] [{secondary_backend_name}] Appended {len(actual_points)} actual points "
                    f"to prediction {prediction_id}"
                )
            except Exception as e:
                LOGGER.error(
                    f"[DUAL-WRITE] [{secondary_backend_name}] Failed to append actual points "
                    f"to prediction {prediction_id}: {e}"
                )
    
    def get_prediction(self, prediction_id: int) -> Optional[dict[str, Any]]:
        """Get prediction metadata by ID."""
        return self.backend.get_prediction(prediction_id)
    
    def get_latest_prediction(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get most recent prediction for a symbol."""
        return self.backend.get_latest_prediction(symbol)
    
    def get_prediction_history(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get prediction history with outcomes for a symbol."""
        if hasattr(self.backend, 'get_prediction_history'):
            return self.backend.get_prediction_history(symbol, limit)
        return []
    
    def get_prediction_points(self, prediction_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        """Get forecast or actual points for a prediction."""
        if hasattr(self.backend, 'get_prediction_points'):
            return self.backend.get_prediction_points(prediction_id, kind)
        return []
    
    def count_predictions_since(self, since_ts: float) -> int:
        """Count predictions made since a given timestamp."""
        if hasattr(self.backend, 'count_predictions_since'):
            return self.backend.count_predictions_since(since_ts)
        return 0
    
    def get_recent_predictions(self, limit: int = 50, since_ts: float | None = None) -> list[dict[str, Any]]:
        """Get recent predictions across all symbols."""
        if hasattr(self.backend, 'get_recent_predictions'):
            return self.backend.get_recent_predictions(limit, since_ts)
        return []
    
    def create_outcome(
        self,
        prediction_id: int,
        mae: float,
        map_val: float,
        rmse: float,
        hit_direction: int,
        hit_ratio_window: float | None = None,
        notes: str = "",
    ):
        """Create outcome record for a closed prediction."""
        if hasattr(self.backend, 'create_outcome'):
            self.backend.create_outcome(
                prediction_id, mae, map_val, rmse, hit_direction, hit_ratio_window, notes
            )
    
    def get_pending_outcomes(self) -> list[dict[str, Any]]:
        """Get predictions ready for outcome reconciliation (48h window closed, no outcome yet)."""
        if hasattr(self.backend, 'get_pending_outcomes'):
            return self.backend.get_pending_outcomes()
        return []
    
    def get_active_predictions(self) -> list[dict[str, Any]]:
        """Get predictions where the 48h window is STILL OPEN (for appending actual prices)."""
        if hasattr(self.backend, 'get_active_predictions'):
            return self.backend.get_active_predictions()
        return []
    
    def get_predictions_with_outcomes(self, symbol: str) -> list[dict[str, Any]]:
        """Get predictions with their outcomes for accuracy computation."""
        if hasattr(self.backend, 'get_predictions_with_outcomes'):
            return self.backend.get_predictions_with_outcomes(symbol)
        return []
    
    def get_predictions_with_outcomes_since(self, symbol: str, since_ts: float) -> list[dict[str, Any]]:
        """Get predictions with outcomes since a timestamp for windowed accuracy."""
        if hasattr(self.backend, 'get_predictions_with_outcomes_since'):
            return self.backend.get_predictions_with_outcomes_since(symbol, since_ts)
        return []


class SQLiteBackend:
    """SQLite storage backend for predictions."""
    
    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()
        LOGGER.info(f"📁 SQLite backend initialized: {self.db_path}")
    
    def _init_db(self):
        """Initialize database schema. Idempotent."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        # predictions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                run_at REAL NOT NULL,
                horizon_h INTEGER NOT NULL DEFAULT 48,
                method TEXT NOT NULL,
                confidence REAL NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('UP','DOWN','FLAT')),
                features_json TEXT,
                params_json TEXT,
                tag TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_run ON predictions(symbol, run_at DESC)"
        )
        
        # prediction_points table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('forecast','actual')),
                price REAL NOT NULL,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_points_pred_kind ON prediction_points(prediction_id, kind)"
        )
        
        # outcomes table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                prediction_id INTEGER PRIMARY KEY,
                closed_at REAL NOT NULL,
                mae REAL NOT NULL,
                map REAL NOT NULL,
                rmse REAL NOT NULL,
                hit_direction INTEGER NOT NULL,
                hit_ratio_window REAL,
                notes TEXT,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_prediction(
        self,
        symbol: str,
        forecast_points: list[tuple[float, float]],
        method: str,
        confidence: float,
        direction: str,
        features: dict[str, Any],
        params: dict[str, Any],
        tag: str = "",
    ) -> int:
        """Save prediction to SQLite."""
        run_at = time.time()
        horizon_h = params.get("horizon_h", 48)
        
        features_json = json.dumps(features or {})
        params_json = json.dumps(params or {})
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO predictions (symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag),
            )
            prediction_id = cursor.lastrowid
            
            # Insert forecast points
            for ts, price in forecast_points:
                conn.execute(
                    "INSERT INTO prediction_points (prediction_id, ts, kind, price) VALUES (?, ?, 'forecast', ?)",
                    (prediction_id, ts, price),
                )
            
            conn.commit()
            LOGGER.info(
                f"Created prediction {prediction_id} for {symbol} with {len(forecast_points)} forecast points"
            )
            return prediction_id
        finally:
            conn.close()
    
    def append_actual_points(self, prediction_id: int, actual_points: list[tuple[float, float]]):
        """Append actual price points to SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            for ts, price in actual_points:
                # Check if point already exists
                existing = conn.execute(
                    "SELECT 1 FROM prediction_points WHERE prediction_id=? AND ts=? AND kind='actual'",
                    (prediction_id, ts),
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO prediction_points (prediction_id, ts, kind, price) VALUES (?, ?, 'actual', ?)",
                        (prediction_id, ts, price),
                    )
            conn.commit()
            LOGGER.debug(f"Appended {len(actual_points)} actual points to prediction {prediction_id}")
        finally:
            conn.close()
    
    def get_prediction(self, prediction_id: int) -> Optional[dict[str, Any]]:
        """Get prediction metadata from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT id, symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag FROM predictions WHERE id=?",
                (prediction_id,),
            ).fetchone()
            if not row:
                return None
            
            return {
                "id": row[0],
                "symbol": row[1],
                "run_at": row[2],
                "horizon_h": row[3],
                "method": row[4],
                "confidence": row[5],
                "direction": row[6],
                "features_json": row[7],
                "params_json": row[8],
                "tag": row[9],
            }
        finally:
            conn.close()
    
    def get_latest_prediction(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get most recent prediction for a symbol from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT id, symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag FROM predictions WHERE symbol=? ORDER BY run_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if not row:
                return None
            
            return {
                "id": row[0],
                "symbol": row[1],
                "run_at": row[2],
                "horizon_h": row[3],
                "method": row[4],
                "confidence": row[5],
                "direction": row[6],
                "features_json": row[7],
                "params_json": row[8],
                "tag": row[9],
            }
        finally:
            conn.close()
    
    def get_prediction_history(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get prediction history with outcomes for a symbol from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    p.id, p.symbol, p.run_at, p.horizon_h, p.method, p.confidence, p.direction, p.tag,
                    o.closed_at, o.mae, o.map, o.rmse, o.hit_direction, o.hit_ratio_window, o.notes
                FROM predictions p
                LEFT JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.symbol = ?
                ORDER BY p.run_at DESC
                LIMIT ?
                """,
                (symbol, limit)
            ).fetchall()
            
            results = []
            for row in rows:
                closed = row[8] is not None
                results.append({
                    "id": row[0],
                    "symbol": row[1],
                    "run_at": row[2],
                    "horizon_h": row[3],
                    "method": row[4],
                    "confidence": row[5],
                    "direction": row[6],
                    "tag": row[7],
                    "closed": closed,
                    "closed_at": row[8],
                    "mae": row[9],
                    "map": row[10],
                    "rmse": row[11],
                    "hit_direction": row[12],
                    "hit_ratio_window": row[13],
                    "notes": row[14],
                })
            
            return results
        finally:
            conn.close()
    
    def get_prediction_points(self, prediction_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        """Get forecast or actual points for a prediction from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            if kind:
                rows = conn.execute(
                    "SELECT id, prediction_id, ts, kind, price FROM prediction_points WHERE prediction_id=? AND kind=? ORDER BY ts",
                    (prediction_id, kind),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, prediction_id, ts, kind, price FROM prediction_points WHERE prediction_id=? ORDER BY ts",
                    (prediction_id,),
                ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "prediction_id": row[1],
                    "ts": row[2],
                    "kind": row[3],
                    "price": row[4],
                }
                for row in rows
            ]
        finally:
            conn.close()
    
    def count_predictions_since(self, since_ts: float) -> int:
        """Count predictions made since a given timestamp."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE run_at >= ?",
                (since_ts,)
            )
            return cursor.fetchone()[0] or 0
        finally:
            conn.close()
    
    def get_recent_predictions(self, limit: int = 50, since_ts: float | None = None) -> list[dict[str, Any]]:
        """Get recent predictions across all symbols with entry prices."""
        conn = sqlite3.connect(self.db_path)
        try:
            if since_ts:
                rows = conn.execute(
                    """
                    SELECT p.id, p.symbol, p.run_at, p.horizon_h, p.method, p.confidence, p.direction, p.tag,
                           (SELECT pp.price FROM prediction_points pp 
                            WHERE pp.prediction_id = p.id AND pp.kind = 'forecast' 
                            ORDER BY pp.ts ASC LIMIT 1) as entry_price
                    FROM predictions p
                    WHERE p.run_at >= ?
                    ORDER BY p.run_at DESC
                    LIMIT ?
                    """,
                    (since_ts, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT p.id, p.symbol, p.run_at, p.horizon_h, p.method, p.confidence, p.direction, p.tag,
                           (SELECT pp.price FROM prediction_points pp 
                            WHERE pp.prediction_id = p.id AND pp.kind = 'forecast' 
                            ORDER BY pp.ts ASC LIMIT 1) as entry_price
                    FROM predictions p
                    ORDER BY p.run_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "symbol": row[1],
                    "run_at": row[2],
                    "horizon_h": row[3],
                    "method": row[4],
                    "confidence": row[5],
                    "direction": row[6],
                    "tag": row[7],
                    "entry_price": row[8],
                    "price_at_prediction": row[8],  # Alias for compatibility
                }
                for row in rows
            ]
        finally:
            conn.close()
    
    def create_outcome(
        self,
        prediction_id: int,
        mae: float,
        map_val: float,
        rmse: float,
        hit_direction: int,
        hit_ratio_window: float | None = None,
        notes: str = "",
    ):
        """Create outcome record for a closed prediction in SQLite."""
        closed_at = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO outcomes (prediction_id, closed_at, mae, map, rmse, hit_direction, hit_ratio_window, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (prediction_id, closed_at, mae, map_val, rmse, hit_direction, hit_ratio_window, notes),
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_pending_outcomes(self) -> list[dict[str, Any]]:
        """Get predictions ready for outcome reconciliation."""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT p.id, p.symbol, p.run_at, p.horizon_h, p.direction
                FROM predictions p
                LEFT JOIN outcomes o ON p.id = o.prediction_id
                WHERE o.prediction_id IS NULL
                  AND (p.run_at + (p.horizon_h * 3600)) <= ?
                ORDER BY p.run_at
                """,
                (now,),
            ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "symbol": row[1],
                    "run_at": row[2],
                    "horizon_h": row[3],
                    "direction": row[4],
                }
                for row in rows
            ]
        finally:
            conn.close()
    
    def get_active_predictions(self) -> list[dict[str, Any]]:
        """Get predictions where 48h window is STILL OPEN (for appending actual prices)."""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT p.id, p.symbol, p.run_at, p.horizon_h
                FROM predictions p
                LEFT JOIN outcomes o ON p.id = o.prediction_id
                WHERE o.prediction_id IS NULL
                  AND (p.run_at + (p.horizon_h * 3600)) > ?
                ORDER BY p.run_at
                """,
                (now,),
            ).fetchall()
            
            return [
                {
                    "id": row[0],
                    "symbol": row[1],
                    "run_at": row[2],
                    "horizon_h": row[3],
                }
                for row in rows
            ]
        finally:
            conn.close()
    
    def get_predictions_with_outcomes(self, symbol: str) -> list[dict[str, Any]]:
        """Get predictions with their outcomes for accuracy computation."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT p.confidence, o.mae, o.map, o.rmse, o.hit_direction
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.symbol = ?
                """,
                (symbol,),
            ).fetchall()
            
            return [
                {
                    "confidence": row[0],
                    "mae": row[1],
                    "map": row[2],
                    "rmse": row[3],
                    "hit_direction": row[4],
                }
                for row in rows
            ]
        finally:
            conn.close()
    
    def get_predictions_with_outcomes_since(self, symbol: str, since_ts: float) -> list[dict[str, Any]]:
        """Get predictions with outcomes since a timestamp (windowed)."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT p.confidence, o.mae, o.map, o.rmse, o.hit_direction
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.symbol = ? AND o.closed_at >= ?
                """,
                (symbol, since_ts),
            ).fetchall()
            
            return [
                {
                    "confidence": row[0],
                    "mae": row[1],
                    "map": row[2],
                    "rmse": row[3],
                    "hit_direction": row[4],
                }
                for row in rows
            ]
        finally:
            conn.close()


class PostgresBackend:
    """
    PostgreSQL storage backend for predictions.
    
    Fully implements prediction storage with connection pooling,
    transactions, and proper schema mapping from SQLite.
    
    LAZY INITIALIZATION: Connection pool is created on first use to avoid
    blocking module imports when Railway Postgres is slow/unreachable.
    """
    
    def __init__(self):
        if not IS_POSTGRES_AVAILABLE:
            raise RuntimeError("PostgreSQL backend requires DATABASE_URL environment variable")
        
        # Import PostgreSQL dependencies
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
            from psycopg2.extras import RealDictCursor
            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
            self.ThreadedConnectionPool = ThreadedConnectionPool
        except ImportError as e:
            raise RuntimeError(f"PostgreSQL dependencies missing: {e}. Install: pip install psycopg2-binary")
        
        # Lazy-init: defer pool creation until first use
        self.pool = None
        self._pool_initialized = False
        self._init_lock = __import__('threading').Lock()
        
        LOGGER.info("📡 PostgreSQL backend created (lazy-init mode)")
    
    def _ensure_pool(self):
        """Ensure connection pool is initialized (lazy init with retry)."""
        if self._pool_initialized:
            return
        
        with self._init_lock:
            if self._pool_initialized:
                return
            
            # Get pool configuration from environment (with sensible defaults)
            import os
            pool_min = int(os.getenv("POSTGRES_POOL_MIN", "10"))
            pool_max = int(os.getenv("POSTGRES_POOL_MAX", "50"))
            pool_timeout = int(os.getenv("POSTGRES_POOL_TIMEOUT_S", "10"))
            
            # Retry logic for Railway Postgres connection issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    LOGGER.info(f"Initializing Postgres connection pool (attempt {attempt + 1}/{max_retries})...")
                    LOGGER.info(f"Pool config: minconn={pool_min}, maxconn={pool_max}, timeout={pool_timeout}s")
                    self.pool = self.ThreadedConnectionPool(
                        minconn=pool_min,
                        maxconn=pool_max,
                        dsn=DATABASE_URL,
                        cursor_factory=self.RealDictCursor,
                        connect_timeout=pool_timeout
                    )

                    # Initialize schema
                    try:
                        self._init_schema()
                    except Exception:
                        if self.pool:
                            self.pool.closeall()
                            self.pool = None
                        raise

                    self._pool_initialized = True
                    LOGGER.info("✅ PostgreSQL connection pool initialized (2-10 connections)")
                    return
                    
                except Exception as e:
                    LOGGER.warning(f"Connection pool init failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        # Short delay to avoid blocking healthcheck (0.5s instead of exponential backoff)
                        import time
                        time.sleep(0.5)
                    else:
                        raise RuntimeError(f"Failed to initialize PostgreSQL pool after {max_retries} attempts: {e}")
    
    def _get_connection(self):
        """Get connection from pool (ensures pool is initialized)."""
        self._ensure_pool()
        return self.pool.getconn()
    
    def _return_connection(self, conn):
        """Return connection to pool."""
        if self.pool:
            self.pool.putconn(conn)
    
    def _init_schema(self):
        """
        Initialize PostgreSQL schema for predictions.
        
        Schema matches SQLite structure but uses PostgreSQL types:
        - INTEGER → BIGSERIAL (for auto-increment IDs)
        - REAL → DOUBLE PRECISION (for timestamps/prices)
        - TEXT → TEXT/VARCHAR
        - CHECK constraints preserved
        - Foreign keys with ON DELETE CASCADE
        - Indexes for performance
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool is not initialized")

        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            
            # predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    run_at DOUBLE PRECISION NOT NULL,
                    horizon_h INTEGER NOT NULL DEFAULT 48,
                    method VARCHAR(100) NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    direction VARCHAR(10) NOT NULL CHECK(direction IN ('UP','DOWN','FLAT')),
                    features_json TEXT,
                    params_json TEXT,
                    tag VARCHAR(100)
                )
            """)
            
            # Index for symbol + run_at queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_symbol_run 
                ON predictions(symbol, run_at DESC)
            """)
            
            # prediction_points table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_points (
                    id BIGSERIAL PRIMARY KEY,
                    prediction_id BIGINT NOT NULL,
                    ts DOUBLE PRECISION NOT NULL,
                    kind VARCHAR(10) NOT NULL CHECK(kind IN ('forecast','actual')),
                    price DOUBLE PRECISION NOT NULL,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
                )
            """)
            
            # Index for prediction_id + kind queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_points_pred_kind 
                ON prediction_points(prediction_id, kind)
            """)
            
            # outcomes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    prediction_id BIGINT PRIMARY KEY,
                    closed_at DOUBLE PRECISION NOT NULL,
                    mae DOUBLE PRECISION NOT NULL,
                    map DOUBLE PRECISION NOT NULL,
                    rmse DOUBLE PRECISION NOT NULL,
                    hit_direction INTEGER NOT NULL,
                    hit_ratio_window DOUBLE PRECISION,
                    notes TEXT,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
                )
            """)
            
            # Index for outcomes lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_pred 
                ON outcomes(prediction_id)
            """)
            
            # ghost_prediction_outcomes table (used by reconciler - more detailed)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ghost_prediction_outcomes (
                    id SERIAL PRIMARY KEY,
                    prediction_id INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    closed_at TIMESTAMP NOT NULL,
                    price_at_prediction NUMERIC(20, 8) NOT NULL,
                    price_at_resolution NUMERIC(20, 8),
                    realized_move_pct NUMERIC(10, 4),
                    predicted_direction VARCHAR(10),
                    actual_direction VARCHAR(10),
                    hit_direction INTEGER,
                    direction_threshold_pct NUMERIC(5, 2) DEFAULT 0.25,
                    mae NUMERIC(20, 8),
                    mape NUMERIC(10, 4),
                    rmse NUMERIC(20, 8),
                    predicted_confidence NUMERIC(5, 4),
                    confidence_calibration NUMERIC(10, 4),
                    resolution_method VARCHAR(50),
                    resolution_provider VARCHAR(50),
                    notes TEXT,
                    status VARCHAR(20) DEFAULT 'completed',
                    reconciled_by VARCHAR(50) DEFAULT 'outcome_reconciler',
                    reconciliation_version VARCHAR(20) DEFAULT '1.0'
                )
            """)
            
            # Indexes for ghost_prediction_outcomes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gpo_prediction_id 
                ON ghost_prediction_outcomes(prediction_id)
            """)
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_gpo_unique_prediction 
                ON ghost_prediction_outcomes(prediction_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gpo_status 
                ON ghost_prediction_outcomes(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gpo_closed_at 
                ON ghost_prediction_outcomes(closed_at)
            """)
            
            conn.commit()
            LOGGER.info("✅ PostgreSQL prediction schema initialized")
            
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"Failed to initialize PostgreSQL schema: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def save_prediction(
        self,
        symbol: str,
        forecast_points: list[tuple[float, float]],
        method: str,
        confidence: float,
        direction: str,
        features: dict[str, Any],
        params: dict[str, Any],
        tag: str = "",
    ) -> int:
        """
        Save prediction to PostgreSQL with full transaction support.
        
        Maps SQLite schema to PostgreSQL:
        - Uses RETURNING clause to get inserted ID
        - Batch inserts forecast points
        - Full transaction with rollback on error
        
        DEDUP GUARD: Rejects predictions for the same symbol created within
        the last 5 minutes to prevent duplicate spam (e.g., 8 DOGE predictions
        within milliseconds of each other).
        """
        run_at = time.time()
        horizon_h = params.get("horizon_h", 48)
        
        features_json = json.dumps(features or {})
        params_json = json.dumps(params or {})
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # ── DEDUP GUARD: Reject same symbol within 5 minutes ──────────
            dedup_window_s = float(os.getenv("PREDICTION_DEDUP_WINDOW_S", "300"))
            cursor.execute(
                """
                SELECT id, run_at FROM predictions
                WHERE symbol = %s AND run_at > %s
                ORDER BY run_at DESC LIMIT 1
                """,
                (symbol, run_at - dedup_window_s),
            )
            dup = cursor.fetchone()
            if dup:
                age_s = run_at - dup["run_at"]
                LOGGER.warning(
                    f"[DEDUP] Rejecting duplicate prediction for {symbol} — "
                    f"existing prediction {dup['id']} created {age_s:.0f}s ago "
                    f"(dedup window: {dedup_window_s:.0f}s)"
                )
                raise PredictionRejected(
                    f"Duplicate: {symbol} already has prediction {dup['id']} "
                    f"from {age_s:.0f}s ago (min gap: {dedup_window_s:.0f}s)"
                )
            
            # Insert prediction and get ID using RETURNING clause
            cursor.execute(
                """
                INSERT INTO predictions (symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag),
            )
            
            result = cursor.fetchone()
            prediction_id = result['id']
            
            # Batch insert forecast points
            if forecast_points:
                point_data = [(prediction_id, ts, 'forecast', price) for ts, price in forecast_points]
                cursor.executemany(
                    "INSERT INTO prediction_points (prediction_id, ts, kind, price) VALUES (%s, %s, %s, %s)",
                    point_data
                )
            
            conn.commit()
            
            LOGGER.info(
                f"[POSTGRES] Created prediction {prediction_id} for {symbol} with {len(forecast_points)} forecast points"
            )
            
            return prediction_id
            
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"[POSTGRES] Failed to save prediction for {symbol}: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def append_actual_points(self, prediction_id: int, actual_points: list[tuple[float, float]]):
        """
        Append actual price points to PostgreSQL.
        
        Uses ON CONFLICT to avoid duplicate points (upsert pattern).
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            for ts, price in actual_points:
                # Check if point exists
                cursor.execute(
                    "SELECT 1 FROM prediction_points WHERE prediction_id=%s AND ts=%s AND kind='actual'",
                    (prediction_id, ts)
                )
                
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO prediction_points (prediction_id, ts, kind, price) VALUES (%s, %s, 'actual', %s)",
                        (prediction_id, ts, price)
                    )
            
            conn.commit()
            
            LOGGER.debug(f"[POSTGRES] Appended {len(actual_points)} actual points to prediction {prediction_id}")
            
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"[POSTGRES] Failed to append actual points to prediction {prediction_id}: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def get_prediction(self, prediction_id: int) -> dict[str, Any] | None:
        """
        Get prediction metadata from PostgreSQL.
        
        Returns dict with same structure as SQLite backend.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, symbol, run_at, horizon_h, method, confidence, direction, 
                       features_json, params_json, tag 
                FROM predictions 
                WHERE id=%s
                """,
                (prediction_id,)
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # RealDictCursor returns dict-like rows
            return {
                "id": row["id"],
                "symbol": row["symbol"],
                "run_at": row["run_at"],
                "horizon_h": row["horizon_h"],
                "method": row["method"],
                "confidence": row["confidence"],
                "direction": row["direction"],
                "features_json": row["features_json"],
                "params_json": row["params_json"],
                "tag": row["tag"],
            }
            
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get prediction {prediction_id}: {e}")
            return None
        finally:
            self._return_connection(conn)
    
    def get_latest_prediction(self, symbol: str) -> dict[str, Any] | None:
        """
        Get most recent prediction for a symbol from PostgreSQL.
        
        Uses ORDER BY run_at DESC LIMIT 1 for efficiency.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, symbol, run_at, horizon_h, method, confidence, direction, 
                       features_json, params_json, tag 
                FROM predictions 
                WHERE symbol=%s 
                ORDER BY run_at DESC 
                LIMIT 1
                """,
                (symbol,)
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "id": row["id"],
                "symbol": row["symbol"],
                "run_at": row["run_at"],
                "horizon_h": row["horizon_h"],
                "method": row["method"],
                "confidence": row["confidence"],
                "direction": row["direction"],
                "features_json": row["features_json"],
                "params_json": row["params_json"],
                "tag": row["tag"],
            }
            
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get latest prediction for {symbol}: {e}")
            return None
        finally:
            self._return_connection(conn)
    
    def get_prediction_history(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get prediction history with outcomes for a symbol from PostgreSQL.
        
        Performs LEFT JOIN with outcomes table to include accuracy data.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT
                    p.id, p.symbol, p.run_at, p.horizon_h, p.method, p.confidence, p.direction, p.tag,
                    o.closed_at, o.mae, o.map, o.rmse, o.hit_direction, o.hit_ratio_window, o.notes
                FROM predictions p
                LEFT JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.symbol = %s
                ORDER BY p.run_at DESC
                LIMIT %s
                """,
                (symbol, limit)
            )
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                closed = row["closed_at"] is not None
                results.append({
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "run_at": row["run_at"],
                    "horizon_h": row["horizon_h"],
                    "method": row["method"],
                    "confidence": row["confidence"],
                    "direction": row["direction"],
                    "tag": row["tag"],
                    "closed": closed,
                    "closed_at": row["closed_at"],
                    "mae": row["mae"],
                    "map": row["map"],
                    "rmse": row["rmse"],
                    "hit_direction": row["hit_direction"],
                    "hit_ratio_window": row["hit_ratio_window"],
                    "notes": row["notes"],
                })
            
            return results
            
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get prediction history for {symbol}: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def get_prediction_points(self, prediction_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        """
        Get forecast or actual points for a prediction from PostgreSQL.
        
        Args:
            prediction_id: Prediction ID
            kind: 'forecast', 'actual', or None (all points)
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            if kind:
                cursor.execute(
                    """
                    SELECT id, prediction_id, ts, kind, price 
                    FROM prediction_points 
                    WHERE prediction_id=%s AND kind=%s 
                    ORDER BY ts
                    """,
                    (prediction_id, kind)
                )
            else:
                cursor.execute(
                    """
                    SELECT id, prediction_id, ts, kind, price 
                    FROM prediction_points 
                    WHERE prediction_id=%s 
                    ORDER BY ts
                    """,
                    (prediction_id,)
                )
            
            rows = cursor.fetchall()
            
            return [
                {
                    "id": row["id"],
                    "prediction_id": row["prediction_id"],
                    "ts": row["ts"],
                    "kind": row["kind"],
                    "price": row["price"],
                }
                for row in rows
            ]
            
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get prediction points for {prediction_id}: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def create_outcome(
        self,
        prediction_id: int,
        mae: float,
        map_val: float,
        rmse: float,
        hit_direction: int,
        hit_ratio_window: float | None = None,
        notes: str = "",
    ):
        """
        Create outcome record for a closed prediction in PostgreSQL.
        
        Uses INSERT ... ON CONFLICT for upsert behavior.
        """
        closed_at = time.time()
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # PostgreSQL upsert (INSERT ... ON CONFLICT DO UPDATE)
            cursor.execute(
                """
                INSERT INTO outcomes (prediction_id, closed_at, mae, map, rmse, hit_direction, hit_ratio_window, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_id) 
                DO UPDATE SET 
                    closed_at = EXCLUDED.closed_at,
                    mae = EXCLUDED.mae,
                    map = EXCLUDED.map,
                    rmse = EXCLUDED.rmse,
                    hit_direction = EXCLUDED.hit_direction,
                    hit_ratio_window = EXCLUDED.hit_ratio_window,
                    notes = EXCLUDED.notes
                """,
                (prediction_id, closed_at, mae, map_val, rmse, hit_direction, hit_ratio_window, notes)
            )
            
            conn.commit()
            
            LOGGER.info(
                f"[POSTGRES] Created outcome for prediction {prediction_id}: MAE={mae:.4f}, hit={hit_direction}"
            )
            
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"[POSTGRES] Failed to create outcome for prediction {prediction_id}: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def count_predictions_since(self, since_ts: float) -> int:
        """Count predictions made since a given timestamp in PostgreSQL."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM predictions WHERE run_at >= %s",
                (since_ts,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to count predictions: {e}")
            return 0
        finally:
            self._return_connection(conn)
    
    def get_recent_predictions(self, limit: int = 50, since_ts: float | None = None) -> list[dict[str, Any]]:
        """Get recent predictions across all symbols from PostgreSQL with entry prices."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            if since_ts:
                cursor.execute(
                    """
                    SELECT p.id, p.symbol, p.run_at, p.horizon_h, p.method, p.confidence, p.direction, p.tag,
                           (SELECT pp.price FROM prediction_points pp 
                            WHERE pp.prediction_id = p.id AND pp.kind = 'forecast' 
                            ORDER BY pp.ts ASC LIMIT 1) as entry_price
                    FROM predictions p
                    WHERE p.run_at >= %s
                    ORDER BY p.run_at DESC
                    LIMIT %s
                    """,
                    (since_ts, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT p.id, p.symbol, p.run_at, p.horizon_h, p.method, p.confidence, p.direction, p.tag,
                           (SELECT pp.price FROM prediction_points pp 
                            WHERE pp.prediction_id = p.id AND pp.kind = 'forecast' 
                            ORDER BY pp.ts ASC LIMIT 1) as entry_price
                    FROM predictions p
                    ORDER BY p.run_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
            
            rows = cursor.fetchall()
            
            return [
                {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "run_at": row["run_at"],
                    "horizon_h": row["horizon_h"],
                    "method": row["method"],
                    "confidence": row["confidence"],
                    "direction": row["direction"],
                    "tag": row["tag"],
                    "entry_price": row["entry_price"],
                    "price_at_prediction": row["entry_price"],  # Alias for compatibility
                }
                for row in rows
            ]
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get recent predictions: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def get_pending_outcomes(self) -> list[dict[str, Any]]:
        """Get predictions ready for outcome reconciliation in PostgreSQL."""
        now = time.time()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # CRITICAL FIX: Join with ghost_prediction_outcomes (where reconciler writes)
            # NOT the outcomes table (which is separate/legacy)
            cursor.execute(
                """
                SELECT p.id, p.symbol, p.run_at, p.horizon_h, p.direction,
                       p.confidence, p.features_json
                FROM predictions p
                LEFT JOIN ghost_prediction_outcomes o ON p.id = o.prediction_id
                WHERE o.prediction_id IS NULL
                  AND (p.run_at + (p.horizon_h * 3600)) <= %s
                ORDER BY p.run_at
                LIMIT 100
                """,
                (now,),
            )
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                pred = {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "run_at": row["run_at"],
                    "horizon_h": row["horizon_h"],
                    "direction": row["direction"],
                    "confidence": row["confidence"] or 0.5,
                }
                
                # Extract price_at_prediction from features_json
                features_json = row.get("features_json")
                if features_json:
                    try:
                        import json
                        features = json.loads(features_json)
                        # Try both field names
                        price = features.get("current_price") or features.get("PRICE")
                        if price:
                            pred["price_at_prediction"] = float(price)
                    except (ValueError, KeyError):
                        pass
                
                results.append(pred)
            
            return results
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get pending outcomes: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def get_active_predictions(self) -> list[dict[str, Any]]:
        """Get predictions where 48h window is STILL OPEN (for appending actual prices)."""
        now = time.time()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT p.id, p.symbol, p.run_at, p.horizon_h
                FROM predictions p
                LEFT JOIN ghost_prediction_outcomes o ON p.id = o.prediction_id
                WHERE o.prediction_id IS NULL
                  AND (p.run_at + (p.horizon_h * 3600)) > %s
                ORDER BY p.run_at
                LIMIT 200
                """,
                (now,),
            )
            
            rows = cursor.fetchall()
            
            return [
                {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "run_at": row["run_at"],
                    "horizon_h": row["horizon_h"],
                }
                for row in rows
            ]
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get active predictions: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def get_predictions_with_outcomes(self, symbol: str) -> list[dict[str, Any]]:
        """Get predictions with their outcomes for accuracy computation in PostgreSQL."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # CRITICAL FIX: Use ghost_prediction_outcomes table (where reconciler writes)
            cursor.execute(
                """
                SELECT p.confidence, o.realized_move_pct as map, o.hit_direction
                FROM predictions p
                JOIN ghost_prediction_outcomes o ON p.id = o.prediction_id
                WHERE p.symbol = %s AND o.status = 'completed'
                """,
                (symbol,),
            )
            
            rows = cursor.fetchall()
            
            return [
                {
                    "confidence": row["confidence"],
                    "mae": abs(row["map"]) if row["map"] else 0,  # Approximate MAE from realized_move_pct
                    "map": row["map"] if row["map"] else 0,
                    "rmse": abs(row["map"]) if row["map"] else 0,  # Approximate RMSE
                    "hit_direction": row["hit_direction"],
                }
                for row in rows
            ]
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get predictions with outcomes: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def get_predictions_with_outcomes_since(self, symbol: str, since_ts: float) -> list[dict[str, Any]]:
        """Get predictions with outcomes since a timestamp in PostgreSQL."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # CRITICAL FIX: Use ghost_prediction_outcomes table (where reconciler writes)
            cursor.execute(
                """
                SELECT p.confidence, o.realized_move_pct as map, o.hit_direction
                FROM predictions p
                JOIN ghost_prediction_outcomes o ON p.id = o.prediction_id
                WHERE p.symbol = %s AND EXTRACT(EPOCH FROM o.closed_at) >= %s
                  AND o.status = 'completed'
                """,
                (symbol, since_ts),
            )
            
            rows = cursor.fetchall()
            
            return [
                {
                    "confidence": row["confidence"],
                    "mae": abs(row["map"]) if row["map"] else 0,
                    "map": row["map"] if row["map"] else 0,
                    "rmse": abs(row["map"]) if row["map"] else 0,
                    "hit_direction": row["hit_direction"],
                }
                for row in rows
            ]
        except Exception as e:
            LOGGER.error(f"[POSTGRES] Failed to get windowed predictions with outcomes: {e}")
            return []
        finally:
            self._return_connection(conn)


# Global prediction store instance
_PREDICTION_STORE: Optional[PredictionStore] = None


def get_prediction_store() -> PredictionStore:
    """
    Get or create the global prediction store instance.
    
    Returns:
        PredictionStore configured with appropriate backend
    """
    global _PREDICTION_STORE
    
    if _PREDICTION_STORE is None:
        if PREDICTION_STORE_ENGINE == "postgres" and IS_POSTGRES_AVAILABLE:
            backend = PostgresBackend()
            LOGGER.info("🎯 Using PostgreSQL backend for predictions")
        else:
            backend = SQLiteBackend()
            LOGGER.info("🎯 Using SQLite backend for predictions")
        
        _PREDICTION_STORE = PredictionStore(backend)
    
    return _PREDICTION_STORE
