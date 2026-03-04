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

POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN", "1"))
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX", "5"))
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
_sync_checkout_callers = {}  # {conn_id: "file:line:func"} — tracks who checked out each conn

SYNC_POOL_MIN = int(os.getenv("DB_SYNC_POOL_MIN", "1"))
SYNC_POOL_MAX = int(os.getenv("DB_SYNC_POOL_MAX", "5"))


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


def sync_connection():
    """
    Context manager that guarantees connection is returned to pool.

    Usage:
        with sync_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT ...")
            conn.commit()
        # conn.close() is automatic — no leak possible

    This is the PREFERRED way to use sync connections.
    """
    from contextlib import contextmanager as _cm
    @_cm
    def _inner():
        conn = get_sync_connection_raw()
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return _inner()


class _PoolConnProxy:
    """
    Proxy around a psycopg2 connection that returns it to the pool on close().

    The previous approach (monkey-patching conn.close on C extension objects)
    was unreliable — psycopg2 connections are C types and attribute assignment
    may not override the built-in close() method in all code paths.

    This proxy delegates all attribute access to the real connection but
    intercepts close() to call pool.putconn() directly.
    """
    __slots__ = ('_conn', '_pool', '_conn_id', '_closed')

    def __init__(self, conn, pool, conn_id):
        object.__setattr__(self, '_conn', conn)
        object.__setattr__(self, '_pool', pool)
        object.__setattr__(self, '_conn_id', conn_id)
        object.__setattr__(self, '_closed', False)

    def close(self):
        if object.__getattribute__(self, '_closed'):
            return
        object.__setattr__(self, '_closed', True)
        _conn = object.__getattribute__(self, '_conn')
        _pool = object.__getattribute__(self, '_pool')
        _cid = object.__getattribute__(self, '_conn_id')
        _sync_checkout_callers.pop(_cid, None)
        try:
            _conn.rollback()
        except Exception:
            pass
        try:
            _pool.putconn(_conn)
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_conn'), name)

    def __setattr__(self, name, value):
        if name in _PoolConnProxy.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_conn'), name, value)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_sync_connection_raw():
    """
    Get a psycopg2 connection whose .close() returns it to the pool.

    Unlike get_sync_connection() (a context manager), this returns
    a plain connection object.  Callers MUST call conn.close() when
    done — it will return the connection to the pool rather than
    destroying the TCP socket.

    Retries up to 3 times with backoff on pool exhaustion.

    This replaces the broken pattern:
        conn = get_sync_connection().__enter__()   # WRONG — leaks CM
    With:
        conn = get_sync_connection_raw()           # CORRECT
    """
    import time as _time
    import traceback as _tb

    pool = _get_sync_pool()

    if pool is not None:
        _max_retries = 3
        _backoff = [0.2, 0.5, 1.0]
        _last_err = None

        for _attempt in range(_max_retries):
            try:
                conn = pool.getconn()

                # Track who checked out this connection
                _caller_stack = _tb.extract_stack(limit=4)
                _caller_info = " → ".join(
                    f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
                    for frame in _caller_stack[:-1]  # skip get_sync_connection_raw itself
                )
                _conn_id = id(conn)
                _sync_checkout_callers[_conn_id] = _caller_info
                LOGGER.debug(f"[DB_POOL] checkout #{len(_sync_checkout_callers)}/{SYNC_POOL_MAX}: {_caller_info}")

                # Return a proxy that calls pool.putconn() on close()
                return _PoolConnProxy(conn, pool, _conn_id)
            except Exception as e:
                _last_err = e
                if _attempt < _max_retries - 1:
                    _wait = _backoff[_attempt]
                    LOGGER.warning(
                        f"[DB_POOL] Pool exhausted (attempt {_attempt + 1}/{_max_retries}), "
                        f"retrying in {_wait}s..."
                    )
                    _time.sleep(_wait)

        # All retries failed — DO NOT fall back to raw connect (causes 'too many clients')
        LOGGER.error(
            f"[DB_POOL] Pool exhausted after {_max_retries} retries: {_last_err}"
        )
        raise RuntimeError(f"DB pool exhausted after {_max_retries} retries") from _last_err
    else:
        # Pool not initialized — try to create it on-demand
        import psycopg2
        db_url = _get_db_url()
        if not db_url:
            raise RuntimeError("DATABASE_URL not set")
        LOGGER.warning("[DB_POOL] Sync pool not initialized, creating single connection")
        return psycopg2.connect(db_url)


def get_sync_pool_status() -> dict:
    """Return diagnostic info about the sync connection pool."""
    pool = _sync_pool
    if pool is None:
        return {"initialized": False}
    try:
        # ThreadedConnectionPool internals
        used = len(pool._used)       # connections currently checked out
        rused = len(pool._rused)     # reverse map
        pool_min = pool.minconn
        pool_max = pool.maxconn
        closed = getattr(pool, 'closed', False)
        return {
            "initialized": True,
            "closed": closed,
            "min_connections": pool_min,
            "max_connections": pool_max,
            "checked_out": used,
            "available": pool_max - used,
            "holders": list(_sync_checkout_callers.values()),
        }
    except Exception as e:
        return {"initialized": True, "error": str(e)}
