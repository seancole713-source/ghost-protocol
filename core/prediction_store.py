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
        Save a prediction to storage.
        
        Returns:
            prediction_id (int)
        """
        # Write to primary backend
        prediction_id = self.backend.save_prediction(
            symbol, forecast_points, method, confidence, direction, features, params, tag
        )
        
        # Write to secondary backend if dual-write enabled
        if self.dual_write_backend:
            try:
                self.dual_write_backend.save_prediction(
                    symbol, forecast_points, method, confidence, direction, features, params, tag
                )
                LOGGER.debug(f"Dual-write succeeded for prediction {prediction_id}")
            except Exception as e:
                LOGGER.error(f"Dual-write failed for prediction {prediction_id}: {e}")
        
        return prediction_id
    
    def append_actual_points(self, prediction_id: int, actual_points: list[tuple[float, float]]):
        """Append actual price points for accuracy tracking."""
        self.backend.append_actual_points(prediction_id, actual_points)
        
        if self.dual_write_backend:
            try:
                self.dual_write_backend.append_actual_points(prediction_id, actual_points)
            except Exception as e:
                LOGGER.error(f"Dual-write append failed for prediction {prediction_id}: {e}")
    
    def get_prediction(self, prediction_id: int) -> Optional[dict[str, Any]]:
        """Get prediction metadata by ID."""
        return self.backend.get_prediction(prediction_id)
    
    def get_latest_prediction(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get most recent prediction for a symbol."""
        return self.backend.get_latest_prediction(symbol)


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


class PostgresBackend:
    """
    PostgreSQL storage backend for predictions.
    
    NOTE: This is a placeholder implementation with the correct interface.
    Full PostgreSQL implementation requires psycopg2 and schema migration.
    """
    
    def __init__(self):
        LOGGER.info("📡 PostgreSQL backend initialized (placeholder)")
    
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
        """Save prediction to PostgreSQL (placeholder)."""
        LOGGER.warning("PostgreSQL backend not fully implemented yet")
        return -1  # Placeholder ID
    
    def append_actual_points(self, prediction_id: int, actual_points: list[tuple[float, float]]):
        """Append actual price points to PostgreSQL (placeholder)."""
        LOGGER.warning("PostgreSQL backend not fully implemented yet")
    
    def get_prediction(self, prediction_id: int) -> Optional[dict[str, Any]]:
        """Get prediction metadata from PostgreSQL (placeholder)."""
        LOGGER.warning("PostgreSQL backend not fully implemented yet")
        return None
    
    def get_latest_prediction(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get most recent prediction for a symbol from PostgreSQL (placeholder)."""
        LOGGER.warning("PostgreSQL backend not fully implemented yet")
        return None


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
