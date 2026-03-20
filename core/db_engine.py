"""
Core Database Engine
====================
Provides get_db_connection context manager and helpers.
Used by migration_runner, personal_watchlist, and other core modules.
"""
import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

LOGGER = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
IS_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")))


@contextmanager
def get_db_connection():
    """Get PostgreSQL connection (context manager). Auto-commits on success."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_query(sql, params=None, fetch_all=True):
    """Execute a read query and return results as list of dicts."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            if fetch_all:
                return [dict(row) for row in cur.fetchall()]
            row = cur.fetchone()
            return dict(row) if row else None


def execute_many(sql, params_list):
    """Execute a write query with many parameter sets."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params_list)
            return cur.rowcount
