#!/usr/bin/env python3
"""Price recorder for target-touch evaluation.

Target-touch evaluation needs an intrahorizon price path. This recorder periodically
stores point-in-time prices into `wolf.db` (table `price_actuals`) for all symbols
with active, not-yet-checked predictions.

The recorder is intentionally decoupled from the price-fetch implementation; callers
inject a `fetch_price(symbol)` callable.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections.abc import Callable

LOGGER = logging.getLogger("ghost.price_recorder")

DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
PRICE_RECORD_INTERVAL_S = int(os.getenv("PRICE_RECORD_INTERVAL_S", "60"))
PRICE_RECORD_LOOKBACK_H = int(os.getenv("PRICE_RECORD_LOOKBACK_H", "72"))

FetchPriceFn = Callable[[str], float | None]


def _ensure_price_tables(conn: sqlite3.Connection) -> None:
    # Minimal table required by target-touch evaluation.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_actuals (
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            PRIMARY KEY (ts, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_price_actuals_symbol_ts
        ON price_actuals(symbol, ts)
        """
    )
    conn.commit()


def _get_active_symbols(conn: sqlite3.Connection, now: int) -> list[str]:
    cutoff = now - (PRICE_RECORD_LOOKBACK_H * 3600)
    rows = conn.execute(
        """
        SELECT DISTINCT symbol
        FROM ghost_predictions
        WHERE checked = 0 AND predicted_at >= ?
        """,
        (cutoff,),
    ).fetchall()
    return [r[0] for r in rows if r and r[0]]


def record_prices_once(fetch_price: FetchPriceFn) -> dict[str, int]:
    """Record a single snapshot for all active symbols."""
    now = int(time.time())
    conn = sqlite3.connect(DB_PATH)

    try:
        _ensure_price_tables(conn)
        symbols = _get_active_symbols(conn, now)
        inserted = 0
        failed = 0

        for symbol in symbols:
            try:
                price = fetch_price(symbol)
                if price is None or price <= 0:
                    failed += 1
                    continue

                conn.execute(
                    """
                    INSERT OR REPLACE INTO price_actuals (ts, symbol, price)
                    VALUES (?, ?, ?)
                    """,
                    (now, symbol, float(price)),
                )
                inserted += 1
            except Exception:
                failed += 1

        conn.commit()
        return {"symbols": len(symbols), "inserted": inserted, "failed": failed}

    finally:
        conn.close()


async def price_recording_loop(fetch_price: FetchPriceFn) -> None:
    """Async loop: records prices every PRICE_RECORD_INTERVAL_S."""
    import asyncio

    LOGGER.info(
        "[PRICE RECORDER] started",
        extra={"interval_s": PRICE_RECORD_INTERVAL_S, "db": DB_PATH},
    )

    while True:
        try:
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(None, record_prices_once, fetch_price)
            if stats.get("inserted"):
                LOGGER.info(
                    "[PRICE RECORDER] snapshot",
                    extra=stats,
                )
        except Exception as e:
            LOGGER.error(f"[PRICE RECORDER] error: {e}", exc_info=False)

        await asyncio.sleep(PRICE_RECORD_INTERVAL_S)
