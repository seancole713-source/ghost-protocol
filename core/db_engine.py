#!/usr/bin/env python3
"""
Ghost Protocol Database Engine
================================

Unified database interface supporting both SQLite (dev/legacy) and PostgreSQL (production).

Features:
- Connection pooling for PostgreSQL
- Automatic schema initialization
- Migration support
- Async + sync interfaces
- Graceful fallback to SQLite

Environment Variables:
- DATABASE_URL: PostgreSQL connection string (postgres://user:pass@host:port/db)
- WOLF_SQLITE_PATH: SQLite file path (fallback)
"""

import os
import logging
import sqlite3
from typing import Any, Optional, Dict, List, Tuple
from contextlib import contextmanager
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
WOLF_SQLITE_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")

# Detect database engine
IS_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))
IS_SQLITE = not IS_POSTGRES

LOGGER.info(f"🗄️  Database engine: {'PostgreSQL' if IS_POSTGRES else 'SQLite'}")
if IS_POSTGRES:
    LOGGER.info(f"📡 PostgreSQL host: {urlparse(DATABASE_URL).hostname}")
else:
    LOGGER.info(f"📁 SQLite path: {WOLF_SQLITE_PATH}")

# PostgreSQL imports (conditional)
if IS_POSTGRES:
    try:
        import psycopg2
        from psycopg2.pool import ThreadedConnectionPool
        from psycopg2.extras import RealDictCursor
        
        # Create connection pool (lazy initialization)
        _pg_pool: Optional[ThreadedConnectionPool] = None
        
        def get_pg_pool() -> ThreadedConnectionPool:
            """Get or create PostgreSQL connection pool"""
            global _pg_pool
            if _pg_pool is None:
                LOGGER.info("🔌 Initializing PostgreSQL connection pool...")
                _pg_pool = ThreadedConnectionPool(
                    minconn=2,
                    maxconn=20,
                    dsn=DATABASE_URL,
                    cursor_factory=RealDictCursor,
                    connect_timeout=5  # CRITICAL: 5 second timeout to prevent startup hangs
                )
                LOGGER.info("✅ PostgreSQL pool initialized (2-20 connections, 5s timeout)")
            return _pg_pool
        
    except ImportError as e:
        LOGGER.error(f"❌ PostgreSQL dependencies missing: {e}")
        LOGGER.error("Install: pip install psycopg2-binary")
        IS_POSTGRES = False
        IS_SQLITE = True


@contextmanager
def get_db_connection():
    """
    Get database connection (PostgreSQL or SQLite).
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ...")
    """
    if IS_POSTGRES:
        # PostgreSQL connection from pool
        pool = get_pg_pool()
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"Database error: {e}")
            raise
        finally:
            pool.putconn(conn)
    else:
        # SQLite connection
        os.makedirs(os.path.dirname(WOLF_SQLITE_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"Database error: {e}")
            raise
        finally:
            conn.close()


def execute_query(query: str, params: Optional[Tuple] = None, fetch: str = "none") -> Any:
    """
    Execute a database query with automatic connection management.
    
    Args:
        query: SQL query string
        params: Query parameters (tuple)
        fetch: "one", "all", or "none"
        
    Returns:
        Query results or None
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        
        if fetch == "one":
            return cursor.fetchone()
        elif fetch == "all":
            return cursor.fetchall()
        elif fetch == "none":
            return cursor.lastrowid if IS_SQLITE else cursor.rowcount
        else:
            raise ValueError(f"Invalid fetch mode: {fetch}")


def execute_many(query: str, params_list: List[Tuple]) -> int:
    """
    Execute batch insert/update with automatic connection management.
    
    Args:
        query: SQL query string
        params_list: List of parameter tuples
        
    Returns:
        Number of rows affected
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        return cursor.rowcount


def init_ghost_schema():
    """
    Initialize Ghost Protocol database schema.
    Creates all required tables for predictions, outcomes, symbols, etc.
    """
    LOGGER.info("📋 Initializing Ghost Protocol schema...")
    
    # SQLite vs PostgreSQL type mapping
    if IS_POSTGRES:
        SERIAL = "SERIAL"
        BIGSERIAL = "BIGSERIAL"
        TEXT = "TEXT"
        REAL = "REAL"
        INTEGER = "INTEGER"
        BIGINT = "BIGINT"
    else:
        SERIAL = "INTEGER PRIMARY KEY AUTOINCREMENT"
        BIGSERIAL = "INTEGER PRIMARY KEY AUTOINCREMENT"
        TEXT = "TEXT"
        REAL = "REAL"
        INTEGER = "INTEGER"
        BIGINT = "INTEGER"
    
    schema_sql = [
        # 1. Predictions table
        f"""
        CREATE TABLE IF NOT EXISTS ghost_predictions (
            id {'SERIAL PRIMARY KEY' if IS_POSTGRES else 'INTEGER PRIMARY KEY AUTOINCREMENT'},
            symbol TEXT NOT NULL,
            asset_type TEXT DEFAULT 'stock',
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            horizon_h INTEGER NOT NULL,
            run_at {BIGINT} NOT NULL,
            created_at {BIGINT} NOT NULL,
            model_version TEXT,
            provider TEXT,
            metadata TEXT
        )
        """,
        
        # 2. Prediction points (forecast path)
        f"""
        CREATE TABLE IF NOT EXISTS prediction_points (
            id {'SERIAL PRIMARY KEY' if IS_POSTGRES else 'INTEGER PRIMARY KEY AUTOINCREMENT'},
            prediction_id INTEGER NOT NULL,
            ts {BIGINT} NOT NULL,
            price REAL NOT NULL,
            kind TEXT DEFAULT 'forecast'
        )
        """,
        
        # 3. Outcomes table
        f"""
        CREATE TABLE IF NOT EXISTS outcomes (
            id {'SERIAL PRIMARY KEY' if IS_POSTGRES else 'INTEGER PRIMARY KEY AUTOINCREMENT'},
            prediction_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            asset_type TEXT DEFAULT 'stock',
            predicted_direction TEXT NOT NULL,
            actual_direction TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            actual_price_change_pct REAL NOT NULL,
            was_correct INTEGER NOT NULL,
            confidence_error REAL NOT NULL,
            evaluated_at {BIGINT} NOT NULL,
            original_price REAL,
            final_price REAL
        )
        """,
        
        # 4. Symbol universe (full market)
        f"""
        CREATE TABLE IF NOT EXISTS symbol_universe (
            id {'SERIAL PRIMARY KEY' if IS_POSTGRES else 'INTEGER PRIMARY KEY AUTOINCREMENT'},
            symbol TEXT UNIQUE NOT NULL,
            name TEXT,
            asset_type TEXT NOT NULL,
            exchange TEXT,
            sector TEXT,
            industry TEXT,
            market_cap {BIGINT},
            is_active INTEGER DEFAULT 1,
            last_price REAL,
            last_updated {BIGINT},
            metadata TEXT
        )
        """,
        
        # 5. Price cache (for volatility detection)
        f"""
        CREATE TABLE IF NOT EXISTS price_cache (
            id {'SERIAL PRIMARY KEY' if IS_POSTGRES else 'INTEGER PRIMARY KEY AUTOINCREMENT'},
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            volume {BIGINT},
            timestamp {BIGINT} NOT NULL,
            provider TEXT,
            {'CONSTRAINT price_cache_unique UNIQUE(symbol, timestamp)' if IS_POSTGRES else 'UNIQUE(symbol, timestamp)'}
        )
        """,
        
        # 6. Volatility triggers (for ultra-efficient mode)
        f"""
        CREATE TABLE IF NOT EXISTS volatility_triggers (
            id {'SERIAL PRIMARY KEY' if IS_POSTGRES else 'INTEGER PRIMARY KEY AUTOINCREMENT'},
            symbol TEXT NOT NULL,
            baseline_price REAL NOT NULL,
            current_price REAL NOT NULL,
            volatility_pct REAL NOT NULL,
            triggered_at {BIGINT} NOT NULL,
            prediction_made INTEGER DEFAULT 0,
            batch_id TEXT
        )
        """,
        
        # Indexes for performance
        "CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON ghost_predictions(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_run_at ON ghost_predictions(run_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_prediction_points_pred_id ON prediction_points(prediction_id)",
        "CREATE INDEX IF NOT EXISTS idx_outcomes_symbol ON outcomes(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_outcomes_evaluated_at ON outcomes(evaluated_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_outcomes_was_correct ON outcomes(was_correct)",
        "CREATE INDEX IF NOT EXISTS idx_symbol_universe_symbol ON symbol_universe(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_symbol_universe_active ON symbol_universe(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_price_cache_symbol_time ON price_cache(symbol, timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_volatility_triggers_symbol ON volatility_triggers(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_volatility_triggers_time ON volatility_triggers(triggered_at DESC)",
    ]
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for sql in schema_sql:
            try:
                cursor.execute(sql)
                LOGGER.debug(f"✅ Executed: {sql[:60]}...")
            except Exception as e:
                LOGGER.error(f"❌ Failed: {sql[:60]}... | Error: {e}")
                raise
        
        conn.commit()
    
    LOGGER.info("✅ Ghost Protocol schema initialized successfully")


def get_table_info(table_name: str) -> List[Dict]:
    """Get table schema information"""
    if IS_POSTGRES:
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (table_name,))
            return [dict(row) for row in cursor.fetchall()]
    else:
        query = f"PRAGMA table_info({table_name})"
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [{"name": row[1], "type": row[2], "notnull": row[3]} for row in rows]


def test_connection() -> bool:
    """Test database connection"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        LOGGER.info("✅ Database connection test passed")
        return True
    except Exception as e:
        LOGGER.error(f"❌ Database connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test and initialize
    LOGGER.info("🚀 Ghost Protocol Database Engine Test")
    if test_connection():
        init_ghost_schema()
        LOGGER.info("🎉 Database ready!")
