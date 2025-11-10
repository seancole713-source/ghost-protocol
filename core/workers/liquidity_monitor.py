"""
Liquidity & Flow Monitor (Heartbeat)
Tracks: DXY, Treasuries (TLT proxy), stablecoin flows (placeholder), futures funding (placeholder).
Persists a compact snapshot for /ai/decide and /forecast usage.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

from core.workers import workers_utils as U

DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
REFRESH_S = int(os.getenv("LIQUIDITY_REFRESH_S", "300"))

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS liquidity_snap (
  ts INTEGER PRIMARY KEY,
  dxy REAL,
  tlt REAL,
  vix REAL,
  flows_json TEXT
);
"""


async def ensure_table() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def _last_close(ticker: str) -> float | None:
    if yf is None:
        return None
    try:
        h = yf.Ticker(ticker).history(period="5d")
        if not h.empty and len(h["Close"]) >= 1:
            return float(h["Close"].iloc[-1])
    except Exception:
        return None
    return None


async def run_forever() -> None:
    await ensure_table()
    while True:
        try:
            now = int(time.time())
            dxy = _last_close("DXY")
            tlt = _last_close("TLT")
            vix = _last_close("^VIX")
            flows = {"stablecoins": None, "funding": None}  # TODO: wire real feeds
            row = {"ts": now, "dxy": dxy, "tlt": tlt, "vix": vix, "flows": flows}
            # persist
            conn = sqlite3.connect(DB_PATH)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO liquidity_snap (ts,dxy,tlt,vix,flows_json) VALUES (?,?,?,?,?)",
                    (now, dxy, tlt, vix, json.dumps(flows)),
                )
                conn.commit()
            finally:
                conn.close()
            await U.redis_set_json("liquidity:latest", row)
        except Exception:
            pass
        await asyncio.sleep(max(30, REFRESH_S))
