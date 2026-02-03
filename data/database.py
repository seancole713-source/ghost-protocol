"""
Database connection management.

Supports both sync (psycopg2) and async (asyncpg) connections.
Use async in FastAPI routes, sync in background jobs.
"""
import os
from typing import Optional, Any, Dict, List
from contextlib import contextmanager, asynccontextmanager
from loguru import logger

# Try to import database libraries
try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not available - sync database operations disabled")

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not available - async database operations disabled")

from config.settings import settings


class Database:
    """
    PostgreSQL database connection manager.
    
    Provides both sync and async connection methods.
    """
    
    _pool: Optional[Any] = None  # asyncpg.Pool
    _sync_conn: Optional[Any] = None  # psycopg2 connection
    
    @classmethod
    def get_url(cls) -> Optional[str]:
        """Get database URL from settings or environment."""
        return settings.DATABASE_URL or os.getenv("DATABASE_URL")
    
    # =========================================================================
    # ASYNC METHODS (for FastAPI routes)
    # =========================================================================
    
    @classmethod
    async def get_pool(cls) -> Any:
        """Get or create async connection pool."""
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg not installed")
        
        if cls._pool is None:
            url = cls.get_url()
            if not url:
                raise RuntimeError("DATABASE_URL not configured")
            
            cls._pool = await asyncpg.create_pool(
                url,
                min_size=2,
                max_size=10,
            )
            logger.info("Async database pool created")
        
        return cls._pool
    
    @classmethod
    async def close_pool(cls):
        """Close async connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("Async database pool closed")
    
    @classmethod
    @asynccontextmanager
    async def connection(cls):
        """Get an async connection from the pool."""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            yield conn
    
    @classmethod
    @asynccontextmanager
    async def transaction(cls):
        """Get an async connection with transaction."""
        async with cls.connection() as conn:
            async with conn.transaction():
                yield conn
    
    @classmethod
    async def execute(cls, query: str, *args) -> str:
        """Execute a query and return status."""
        async with cls.connection() as conn:
            return await conn.execute(query, *args)
    
    @classmethod
    async def fetch(cls, query: str, *args) -> List[Any]:
        """Execute a query and fetch all rows."""
        async with cls.connection() as conn:
            return await conn.fetch(query, *args)
    
    @classmethod
    async def fetchrow(cls, query: str, *args) -> Optional[Any]:
        """Execute a query and fetch one row."""
        async with cls.connection() as conn:
            return await conn.fetchrow(query, *args)
    
    @classmethod
    async def fetchval(cls, query: str, *args) -> Any:
        """Execute a query and fetch a single value."""
        async with cls.connection() as conn:
            return await conn.fetchval(query, *args)
    
    # =========================================================================
    # SYNC METHODS (for background jobs, scripts)
    # =========================================================================
    
    @classmethod
    def get_sync_connection(cls) -> Any:
        """Get a sync database connection."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not installed")
        
        url = cls.get_url()
        if not url:
            raise RuntimeError("DATABASE_URL not configured")
        
        return psycopg2.connect(url)
    
    @classmethod
    @contextmanager
    def sync_connection(cls):
        """Context manager for sync database connection."""
        conn = cls.get_sync_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    @classmethod
    @contextmanager
    def sync_cursor(cls, cursor_factory=None):
        """Context manager for sync database cursor."""
        with cls.sync_connection() as conn:
            cursor_factory = cursor_factory or psycopg2.extras.RealDictCursor
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur
    
    @classmethod
    def sync_execute(cls, query: str, params: tuple = None) -> int:
        """Execute a sync query and return row count."""
        with cls.sync_cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount
    
    @classmethod
    def sync_fetch(cls, query: str, params: tuple = None) -> List[Dict]:
        """Execute a sync query and fetch all rows as dicts."""
        with cls.sync_cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
    
    @classmethod
    def sync_fetchone(cls, query: str, params: tuple = None) -> Optional[Dict]:
        """Execute a sync query and fetch one row as dict."""
        with cls.sync_cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()
    
    # =========================================================================
    # HEALTH CHECK
    # =========================================================================
    
    @classmethod
    async def health_check(cls) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            result = await cls.fetchval("SELECT 1")
            return {
                "status": "connected",
                "pool_size": cls._pool.get_size() if cls._pool else 0,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    @classmethod
    def sync_health_check(cls) -> Dict[str, Any]:
        """Check database connectivity (sync version)."""
        try:
            result = cls.sync_fetchone("SELECT 1 as check")
            return {
                "status": "connected" if result else "error",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
