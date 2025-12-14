#!/usr/bin/env python3
"""SQLite-based calibration and gating for touch-target accuracy.

This is intentionally lightweight: it uses evaluated rows from `wolf.db` (`ghost_predictions`)
that have `correct_1pct` / `correct_0_5pct` populated by the touch evaluator.

Calibration is empirical per-symbol and per-confidence band.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")


@dataclass(frozen=True)
class TouchCalibration:
    symbol: str
    raw_confidence: float
    calibrated_1pct: float
    calibrated_0_5pct: float
    sample_size: int
    band: str

    @property
    def stage5_ok(self) -> bool:
        return self.calibrated_1pct >= 0.70

    @property
    def stage6_ok(self) -> bool:
        return self.calibrated_0_5pct >= 0.70

    @property
    def gate(self) -> str:
        if self.stage6_ok:
            return "EXECUTION"
        if self.stage5_ok:
            return "ANALYSIS"
        return "MONITOR"


def _band_for(conf: float, width: float = 0.05) -> tuple[float, float, str]:
    c = min(1.0, max(0.0, float(conf)))
    lo = (c // width) * width
    hi = min(1.0, lo + width)
    # keep nice labels
    label = f"{lo:.2f}-{hi:.2f}"
    return float(lo), float(hi), label


def calibrate_touch_confidence(
    symbol: str,
    raw_confidence: float,
    *,
    min_samples: int = 30,
    max_lookback_rows: int = 500,
) -> TouchCalibration:
    sym = (symbol or "").upper().strip()
    raw = float(raw_confidence or 0.0)

    lo, hi, label = _band_for(raw)

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # Try per-symbol + band first
        rows = cur.execute(
            """
            SELECT correct_1pct, correct_0_5pct
            FROM ghost_predictions
            WHERE checked = 1
              AND symbol = ?
              AND confidence >= ? AND confidence < ?
              AND correct_1pct IS NOT NULL
            ORDER BY checked_at DESC
            LIMIT ?
            """,
            (sym, lo, hi, max_lookback_rows),
        ).fetchall()

        def _avg(ix: int) -> float:
            vals = [float(r[ix]) for r in rows if r[ix] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        if len(rows) >= min_samples:
            return TouchCalibration(
                symbol=sym,
                raw_confidence=raw,
                calibrated_1pct=_avg(0),
                calibrated_0_5pct=_avg(1),
                sample_size=len(rows),
                band=label,
            )

        # Fallback: overall (across symbols) in band
        rows = cur.execute(
            """
            SELECT correct_1pct, correct_0_5pct
            FROM ghost_predictions
            WHERE checked = 1
              AND confidence >= ? AND confidence < ?
              AND correct_1pct IS NOT NULL
            ORDER BY checked_at DESC
            LIMIT ?
            """,
            (lo, hi, max_lookback_rows),
        ).fetchall()

        def _avg2(ix: int) -> float:
            vals = [float(r[ix]) for r in rows if r[ix] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        if len(rows) >= min_samples:
            return TouchCalibration(
                symbol=sym,
                raw_confidence=raw,
                calibrated_1pct=_avg2(0),
                calibrated_0_5pct=_avg2(1),
                sample_size=len(rows),
                band=label,
            )

        # Final fallback: no data; use raw confidence for both tiers (conservative gating happens elsewhere)
        return TouchCalibration(
            symbol=sym,
            raw_confidence=raw,
            calibrated_1pct=raw,
            calibrated_0_5pct=raw,
            sample_size=len(rows),
            band=label,
        )

    finally:
        conn.close()
