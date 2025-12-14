#!/usr/bin/env python3
"""Metrics for target-touch accuracy (SQLite wolf.db).

Produces per-symbol and overall accuracy using:
- `correct_1pct` (analysis tier)
- `correct_0_5pct` (execution tier)
- `direction_consistent`

Assumes predictions are stored in `ghost_predictions` and evaluated by
`core.prediction_evaluator` (touch-v1).
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")


def get_touch_accuracy_summary(*, days: int = 30, symbol: str | None = None) -> dict[str, Any]:
    now = int(time.time())
    cutoff = now - int(days) * 86400

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()

        args: list[Any] = [cutoff]
        sym_filter = ""
        if symbol:
            sym_filter = " AND symbol = ?"
            args.append(symbol.upper().strip())

        # Overall counts
        total = cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM ghost_predictions
            WHERE predicted_at >= ? {sym_filter}
            """,
            args,
        ).fetchone()["n"]

        checked = cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM ghost_predictions
            WHERE predicted_at >= ? {sym_filter} AND checked = 1
            """,
            args,
        ).fetchone()["n"]

        pending = total - checked

        # Evaluated aggregates
        agg = cur.execute(
            f"""
            SELECT
              AVG(CASE WHEN correct_1pct = 1 THEN 1.0 ELSE 0.0 END) AS acc_1,
              AVG(CASE WHEN correct_0_5pct = 1 THEN 1.0 ELSE 0.0 END) AS acc_05,
              AVG(CASE WHEN direction_consistent = 1 THEN 1.0 ELSE 0.0 END) AS dir_ok
            FROM ghost_predictions
            WHERE predicted_at >= ? {sym_filter} AND checked = 1 AND correct_1pct IS NOT NULL
            """,
            args,
        ).fetchone()

        overall = {
            "total": int(total or 0),
            "checked": int(checked or 0),
            "pending": int(pending or 0),
            "accuracy_touch_1pct": float(agg["acc_1"] or 0.0),
            "accuracy_touch_0_5pct": float(agg["acc_05"] or 0.0),
            "direction_consistency": float(agg["dir_ok"] or 0.0),
        }

        # Per-symbol breakdown (only if not filtering to a single symbol)
        by_symbol: dict[str, Any] = {}
        if not symbol:
            rows = cur.execute(
                """
                SELECT
                  symbol,
                  COUNT(*) AS total,
                  SUM(CASE WHEN checked = 1 THEN 1 ELSE 0 END) AS checked,
                  AVG(CASE WHEN checked = 1 AND correct_1pct = 1 THEN 1.0 ELSE 0.0 END) AS acc_1,
                  AVG(CASE WHEN checked = 1 AND correct_0_5pct = 1 THEN 1.0 ELSE 0.0 END) AS acc_05
                FROM ghost_predictions
                WHERE predicted_at >= ?
                GROUP BY symbol
                ORDER BY checked DESC, total DESC
                LIMIT 100
                """,
                (cutoff,),
            ).fetchall()

            for r in rows:
                sym = r["symbol"]
                total_s = int(r["total"] or 0)
                checked_s = int(r["checked"] or 0)
                by_symbol[sym] = {
                    "total": total_s,
                    "checked": checked_s,
                    "pending": total_s - checked_s,
                    "accuracy_touch_1pct": float(r["acc_1"] or 0.0),
                    "accuracy_touch_0_5pct": float(r["acc_05"] or 0.0),
                }

        return {
            "ok": True,
            "timestamp": now,
            "days": int(days),
            "symbol": symbol.upper().strip() if symbol else "ALL",
            "overall": overall,
            "by_symbol": by_symbol,
        }

    finally:
        conn.close()
