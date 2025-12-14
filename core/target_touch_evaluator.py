#!/usr/bin/env python3
"""Target-touch evaluation for Ghost predictions.

Definition (per user spec):
- A prediction is correct if price *touches the target* anytime within the horizon window.
- Direction is strict: if predicted_direction disagrees with target vs start direction, it is a fail.
- Two tiers:
  - analysis: ±1.0% tolerance around target
  - execution: ±0.5% tolerance around target

This module is SQLite-first: it evaluates using price time-series already stored in wolf.db.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class TouchEvalResult:
    ok: bool
    reason: str
    window_first: float | None = None
    window_last: float | None = None
    window_high: float | None = None
    window_low: float | None = None
    outcome_direction: str | None = None
    outcome_pct: float | None = None
    error_pct: float | None = None
    direction_consistent: int | None = None
    touch_1pct: int | None = None
    touch_0_5pct: int | None = None
    correct_1pct: int | None = None
    correct_0_5pct: int | None = None


def _fetch_window_prices(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    start_ts: int,
    end_ts: int,
) -> list[tuple[int, float]]:
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT ts, price
        FROM price_actuals
        WHERE symbol = ? AND ts >= ? AND ts <= ? AND price IS NOT NULL
        ORDER BY ts ASC
        """,
        (symbol, start_ts, end_ts),
    ).fetchall()
    if rows:
        return [(int(ts), float(price)) for ts, price in rows if price is not None]

    rows = cur.execute(
        """
        SELECT ts, price
        FROM realized_prices
        WHERE symbol = ? AND ts >= ? AND ts <= ? AND price IS NOT NULL
        ORDER BY ts ASC
        """,
        (symbol, start_ts, end_ts),
    ).fetchall()
    if rows:
        return [(int(ts), float(price)) for ts, price in rows if price is not None]

    # price_history stores provider quotes; use it as a last resort
    rows = cur.execute(
        """
        SELECT timestamp, price
        FROM price_history
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? AND price IS NOT NULL
        ORDER BY timestamp ASC
        """,
        (symbol, start_ts, end_ts),
    ).fetchall()
    return [(int(ts), float(price)) for ts, price in rows if price is not None]


def _direction_from_delta(delta: float) -> str:
    if delta > 0:
        return "UP"
    if delta < 0:
        return "DOWN"
    return "FLAT"


def evaluate_touch_target(
    *,
    predicted_direction: str | None,
    start_price: float | None,
    target_price: float | None,
    prices: list[tuple[int, float]],
) -> TouchEvalResult:
    if start_price is None or start_price <= 0:
        return TouchEvalResult(ok=False, reason="missing_start_price")
    if target_price is None or target_price <= 0:
        return TouchEvalResult(ok=False, reason="missing_target_price")
    if not prices:
        return TouchEvalResult(ok=False, reason="no_prices_in_window")

    window_first = float(prices[0][1])
    window_last = float(prices[-1][1])
    window_high = max(float(p) for _, p in prices)
    window_low = min(float(p) for _, p in prices)

    # Strict direction is based on target vs start.
    expected_direction = _direction_from_delta(target_price - start_price)
    predicted_direction_norm = (predicted_direction or "").upper().strip()

    direction_consistent = 1 if predicted_direction_norm == expected_direction else 0

    # Outcome direction/pct (for legacy fields / dashboards)
    outcome_pct = ((window_last - start_price) / start_price) * 100.0
    outcome_direction = _direction_from_delta(window_last - start_price)

    # Error is defined relative to target vs start (legacy-ish).
    error_pct = abs(window_last - target_price) / start_price * 100.0

    # Touch logic with tolerance around target price.
    tol_1 = 0.01
    tol_05 = 0.005

    if expected_direction == "UP":
        touch_1 = 1 if window_high >= target_price * (1.0 - tol_1) else 0
        touch_05 = 1 if window_high >= target_price * (1.0 - tol_05) else 0
    elif expected_direction == "DOWN":
        touch_1 = 1 if window_low <= target_price * (1.0 + tol_1) else 0
        touch_05 = 1 if window_low <= target_price * (1.0 + tol_05) else 0
    else:
        # FLAT: treat as "touched" if price stayed within tolerance of target.
        # This is conservative and keeps the definition consistent.
        touch_1 = 1 if (window_high <= target_price * (1.0 + tol_1) and window_low >= target_price * (1.0 - tol_1)) else 0
        touch_05 = 1 if (window_high <= target_price * (1.0 + tol_05) and window_low >= target_price * (1.0 - tol_05)) else 0

    correct_1 = 1 if (direction_consistent == 1 and touch_1 == 1) else 0
    correct_05 = 1 if (direction_consistent == 1 and touch_05 == 1) else 0

    return TouchEvalResult(
        ok=True,
        reason="ok",
        window_first=window_first,
        window_last=window_last,
        window_high=window_high,
        window_low=window_low,
        outcome_direction=outcome_direction,
        outcome_pct=outcome_pct,
        error_pct=error_pct,
        direction_consistent=direction_consistent,
        touch_1pct=touch_1,
        touch_0_5pct=touch_05,
        correct_1pct=correct_1,
        correct_0_5pct=correct_05,
    )


def evaluate_prediction_row(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    predicted_at: int,
    check_at: int,
    predicted_price: float | None,
    predicted_direction: str | None,
    current_price: float | None,
) -> TouchEvalResult:
    prices = _fetch_window_prices(conn, symbol=symbol, start_ts=predicted_at, end_ts=check_at)
    return evaluate_touch_target(
        predicted_direction=predicted_direction,
        start_price=current_price,
        target_price=predicted_price,
        prices=prices,
    )
