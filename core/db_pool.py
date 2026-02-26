#!/usr/bin/env python3
"""
🗄️ DB POOL — Shared asyncpg Connection Pool
=============================================

Upgrade #21-22 from the 200 Upgrades Blueprint.

PROBLEM:  20+ files call psycopg2.connect() directly — no pooling, no reuse,
          every DB call opens and closes a TCP connection (~200ms overhead each).

SOLUTION: ONE shared asyncpg pool created at startup, injected into all modules.

Usage:
    from core.db_pool import get_pool, get_sync_connection

    # Async (preferred):
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM ghost_symbol_accuracy")

    # Sync fallback (for legacy code migration):
    with get_sync_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")
"""

import os
import logging
import asyncio
from typing import Optional
from contextlib import contextmanager

LOGGER = logging.getLogger("db_pool")

# ═══════════════════════════════════════════════════════════════════
# POOL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN", "5"))
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX", "20"))
POOL_COMMAND_TIMEOUT = float(os.getenv("DB_COMMAND_TIMEOUT", "30"))

# ═══════════════════════════════════════════════════════════════════
# GLOBAL POOL
# ═══════════════════════════════════════════════════════════════════

_pool = None  # asyncpg.Pool, initialized at startup


def _get_db_url() -> str:
    """Get the database URL, handling Railway's postgres:// prefix."""
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


async def init_pool() -> None:
    """
    Initialize the shared asyncpg pool. Call once at startup.

    This should be called from wolf_app's startup handler:
        from core.db_pool import init_pool
        await init_pool()
    """
    global _pool
    if _pool is not None:
        LOGGER.warning("[DB_POOL] Pool already initialized — skipping")
        return

    db_url = _get_db_url()
    if not db_url:
        LOGGER.error("[DB_POOL] No DATABASE_URL set — pool NOT created")
        return

    try:
        import asyncpg
        _pool = await asyncpg.create_pool(
            db_url,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            command_timeout=POOL_COMMAND_TIMEOUT,
        )
        LOGGER.info(
            f"[DB_POOL] ✅ asyncpg pool created: "
            f"min={POOL_MIN_SIZE}, max={POOL_MAX_SIZE}, "
            f"timeout={POOL_COMMAND_TIMEOUT}s"
        )
    except Exception as e:
        LOGGER.error(f"[DB_POOL] ❌ Pool creation failed: {e}")
        _pool = None


async def close_pool() -> None:
    """Close the pool gracefully. Call at shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        LOGGER.info("[DB_POOL] Pool closed")
        _pool = None


def get_pool():
    """
    Get the shared asyncpg pool.

    Returns None if pool hasn't been initialized yet.
    Modules should check for None and fall back to direct connection.
    """
    return _pool


async def fetch(query: str, *args):
    """Execute a query and return all rows."""
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_pool() at startup.")
    async with _pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args):
    """Execute a query and return one row."""
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_pool() at startup.")
    async with _pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def execute(query: str, *args):
    """Execute a query (INSERT/UPDATE/DELETE)."""
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_pool() at startup.")
    async with _pool.acquire() as conn:
        return await conn.execute(query, *args)


async def pool_health_check() -> dict:
    """Check pool health. Returns status dict for /health endpoint."""
    if _pool is None:
        return {"status": "not_initialized", "healthy": False}

    try:
        async with _pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return {
                "status": "healthy",
                "healthy": result == 1,
                "pool_size": _pool.get_size(),
                "pool_min": _pool.get_min_size(),
                "pool_max": _pool.get_max_size(),
                "free_connections": _pool.get_idle_size(),
            }
    except Exception as e:
        return {"status": "error", "healthy": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# SYNC FALLBACK (for gradual migration of psycopg2 code)
# ═══════════════════════════════════════════════════════════════════

_sync_pool = None  # psycopg2 ThreadedConnectionPool
_sync_pool_lock = __import__("threading").Lock()

SYNC_POOL_MIN = int(os.getenv("DB_SYNC_POOL_MIN", "2"))
SYNC_POOL_MAX = int(os.getenv("DB_SYNC_POOL_MAX", "10"))


def _get_sync_pool():
    """Get or create the shared psycopg2 ThreadedConnectionPool (singleton)."""
    global _sync_pool
    if _sync_pool is not None:
        return _sync_pool

    with _sync_pool_lock:
        # Double-check inside lock
        if _sync_pool is not None:
            return _sync_pool

        db_url = _get_db_url()
        if not db_url:
            return None

        try:
            import psycopg2.pool
            _sync_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=SYNC_POOL_MIN,
                maxconn=SYNC_POOL_MAX,
                dsn=db_url,
            )
            LOGGER.info(
                f"[DB_POOL] ✅ psycopg2 sync pool created: "
                f"min={SYNC_POOL_MIN}, max={SYNC_POOL_MAX}"
            )
            return _sync_pool
        except Exception as e:
            LOGGER.error(f"[DB_POOL] ❌ Sync pool creation failed: {e}")
            return None


@contextmanager
def get_sync_connection():
    """
    Get a synchronous psycopg2 connection from the shared pool.

    Uses a ThreadedConnectionPool for connection reuse instead of
    creating a new TCP connection on every call.

    Usage:
        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
    """
    pool = _get_sync_pool()

    if pool is not None:
        conn = None
        try:
            conn = pool.getconn()
            yield conn
            conn.commit()
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                pool.putconn(conn)
    else:
        # Fallback: direct connection if pool creation failed
        import psycopg2
        db_url = _get_db_url()
        if not db_url:
            raise RuntimeError("DATABASE_URL not set")

        conn = psycopg2.connect(db_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
