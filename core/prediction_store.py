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
        Save a prediction to storage with dual-write support.
        
        Returns:
            prediction_id (int) from primary backend
        """
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


class PostgresBackend:
    """
    PostgreSQL storage backend for predictions.
    
    Fully implements prediction storage with connection pooling,
    transactions, and proper schema mapping from SQLite.
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
        except ImportError as e:
            raise RuntimeError(f"PostgreSQL dependencies missing: {e}. Install: pip install psycopg2-binary")
        
        # Create connection pool
        self.pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=DATABASE_URL,
            cursor_factory=RealDictCursor
        )
        
        # Initialize schema
        self._init_schema()
        
        LOGGER.info("📡 PostgreSQL backend initialized with connection pool (2-10 connections)")
    
    def _get_connection(self):
        """Get connection from pool."""
        return self.pool.getconn()
    
    def _return_connection(self, conn):
        """Return connection to pool."""
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
        conn = self._get_connection()
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
            
            conn.commit()
            LOGGER.info("✅ PostgreSQL prediction schema initialized")
            
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"Failed to initialize PostgreSQL schema: {e}")
            raise
        finally:
            self._return_connection(conn)
    
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
        """
        run_at = time.time()
        horizon_h = params.get("horizon_h", 48)
        
        features_json = json.dumps(features or {})
        params_json = json.dumps(params or {})
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
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
