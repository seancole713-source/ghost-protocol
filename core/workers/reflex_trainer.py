"""
Reflex Learning Loop (Adaptive Feedback)
Re-weights the influence of modules based on forecast accuracy.
Stores weights in SQLite and updates them gradually using exponential moving averages.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time

DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
REFRESH_S = int(os.getenv("REFLEX_REFRESH_S", "900"))  # 15 min
ALPHA = float(os.getenv("REFLEX_ALPHA", "0.1"))  # EMA smoothing

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS module_weights (
  name TEXT PRIMARY KEY,
  weight REAL NOT NULL,
  updated_ts INTEGER NOT NULL
);
"""

GET_ERRORS_SQL = """
SELECT f.symbol, f.ts_issued, f.horizon_hours, f.price_now, f.price_pred_mid,
             a.price AS actual_price, a.ts AS actual_ts
FROM forecast_48h f
LEFT JOIN price_actuals a
    ON a.symbol = f.symbol
 AND a.ts BETWEEN (f.ts_issued + f.horizon_hours*3600 - 3600)
                         AND (f.ts_issued + f.horizon_hours*3600 + 3600)
ORDER BY f.ts_issued DESC
LIMIT 500;
"""

MODULES = ["price_action", "news_sentiment", "macro_pressure", "liquidity", "analogs"]


async def ensure_table() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    sqlite3.connect(DB_PATH).execute(TABLE_SQL).close()


def _score_row(pred_dir: int, actual_dir: int) -> float:
    return 1.0 if int(pred_dir) == int(actual_dir) else -1.0


async def update_weights() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(GET_ERRORS_SQL).fetchall()
        # Start with existing weights
        cur = {name: 1.0 for name in MODULES}
        for name, w, _ in conn.execute(
            "SELECT name,weight,updated_ts FROM module_weights"
        ).fetchall():
            cur[name] = float(w)
        # Aggregate a simple score per module based on direction agreement
        s = 0.0
        n = 0
        for r in rows:
            # r: (symbol, ts_issued, horizon_hours, price_now, price_pred_mid, actual_price, actual_ts)
            try:
                price_now = float(r[3]) if r[3] is not None else None
                price_pred = float(r[4]) if r[4] is not None else None
                actual_price = float(r[5]) if r[5] is not None else None
                if price_now is None or price_pred is None or actual_price is None:
                    continue
                pred_dir = 1 if (price_pred - price_now) >= 0 else -1
                act_dir = 1 if (actual_price - price_now) >= 0 else -1
                s += _score_row(pred_dir, act_dir)
                n += 1
            except Exception:
                continue
        s_norm = s / max(1, n)
        for name in MODULES:
            prev = cur.get(name, 1.0)
            new = (1.0 - ALPHA) * prev + ALPHA * (1.0 + s_norm)
            cur[name] = max(0.2, min(2.0, new))
        # persist
        now = int(time.time())
        for name, w in cur.items():
            conn.execute(
                "INSERT OR REPLACE INTO module_weights (name,weight,updated_ts) VALUES (?,?,?)",
                (name, float(w), now),
            )
        conn.commit()
    finally:
        conn.close()


async def run_forever() -> None:
    await ensure_table()
    while True:
        try:
            await update_weights()
        except Exception:
            pass
        await asyncio.sleep(max(60, REFRESH_S))
