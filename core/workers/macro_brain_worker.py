"""
Macro Awareness Engine (World Brain)
Builds a live macro pressure index in [-100, 100].
- Combines: index proxies (SPY, QQQ, IWM), rates (DXY, VIX as risk proxy), and news sentiment
- Applies recency weighting and caps extreme influence.

Outputs are persisted both in Redis (if available) and SQLite for durability.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from typing import Any

try:
    import yfinance as yf
except Exception:  # optional at runtime
    yf = None  # type: ignore

from core.workers import workers_utils as U

DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
REFRESH_S = int(os.getenv("MACRO_BRAIN_REFRESH_S", "300"))  # 5 min

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS macro_pressure (
  ts INTEGER NOT NULL,
  pressure REAL NOT NULL,
  components_json TEXT NOT NULL,
  PRIMARY KEY (ts)
);
"""

TICKERS = [
    t.strip().upper()
    for t in os.getenv("MACRO_TICKERS", "SPY,QQQ,IWM,TLT,DXY,^VIX").split(",")
    if t.strip()
]


async def ensure_table() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def _score_components(components: dict[str, float], news_score: float | None) -> float:
    # Simple weighted mix with caps
    w = {
        "indices": 0.4,
        "rates_fx": 0.3,
        "vol_proxy": 0.2,
        "news": 0.1,
    }
    idx = components.get("indices", 0.0)
    rfx = components.get("rates_fx", 0.0)
    vol = components.get("vol_proxy", 0.0)
    ns = news_score if isinstance(news_score, (int, float)) else 0.0
    score = w["indices"] * idx + w["rates_fx"] * rfx + w["vol_proxy"] * vol + w["news"] * ns * 100.0
    return max(-100.0, min(100.0, score))


def compute_pressure(news_score: float | None = None) -> dict[str, Any]:
    comps: dict[str, float] = {"indices": 0.0, "rates_fx": 0.0, "vol_proxy": 0.0}
    now = int(time.time())
    if yf is None:
        return {"ok": False, "error": "yfinance-missing"}
    try:
        # Pull 5-day history for proxies
        per = "5d"
        # Indices momentum proxy
        idx_t = [t for t in TICKERS if t in ("SPY", "QQQ", "IWM")]
        idx_vals = []
        for t in idx_t:
            hist = yf.Ticker(t).history(period=per)
            if not hist.empty and len(hist["Close"]) >= 2:
                idx_vals.append(_pct(hist["Close"].iloc[-1], hist["Close"].iloc[-2]))
        comps["indices"] = sum(idx_vals) / len(idx_vals) if idx_vals else 0.0
        # Rates/FX: TLT (inverse for yields), DXY
        rfx_vals = []
        for t in ["TLT", "DXY"]:
            hist = yf.Ticker(t).history(period=per)
            if not hist.empty and len(hist["Close"]) >= 2:
                ch = _pct(hist["Close"].iloc[-1], hist["Close"].iloc[-2])
                if t == "TLT":
                    ch = -ch  # yields up => risk-off
                rfx_vals.append(ch)
        comps["rates_fx"] = sum(rfx_vals) / len(rfx_vals) if rfx_vals else 0.0
        # Vol proxy: VIX move (negative contribution)
        try:
            hist = yf.Ticker("^VIX").history(period=per)
            if not hist.empty and len(hist["Close"]) >= 2:
                vix_ch = _pct(hist["Close"].iloc[-1], hist["Close"].iloc[-2])
                comps["vol_proxy"] = -vix_ch  # higher VIX => negative pressure
        except Exception:
            pass
        pressure = _score_components(comps, news_score)
        return {"ok": True, "ts": now, "pressure": pressure, "components": comps}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def persist_pressure(row: dict[str, Any]) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO macro_pressure (ts, pressure, components_json) VALUES (?,?,?)",
            (
                int(row.get("ts", 0)),
                float(row.get("pressure", 0.0)),
                json.dumps(row.get("components", {})),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    # Mirror in Redis (if available)
    await U.redis_set_json("macro:pressure:latest", row)


async def run_forever() -> None:
    await ensure_table()
    while True:
        try:
            ns = await U.get_news_sentiment()
            row = compute_pressure(ns)
            if row.get("ok"):
                await persist_pressure(row)
        except Exception:
            pass
        await asyncio.sleep(max(30, REFRESH_S))
