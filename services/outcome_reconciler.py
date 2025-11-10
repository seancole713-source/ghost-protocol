#!/usr/bin/env python3
"""
Ghost Prediction Outcome Reconciler

Background task that closes predictions after 48h window expires,
computes accuracy metrics by comparing forecast vs actual prices.
"""

import logging
import sqlite3
import time

from services.predictor import (
    DB_PATH,
    compute_metrics,
    create_outcome,
    get_prediction_points,
)

LOGGER = logging.getLogger("ghost.outcome_reconciler")


def reconcile_outcomes():
    """
    Find predictions where 48h window has closed but outcome is missing.
    Compute metrics and persist outcome.
    """
    now = time.time()
    conn = sqlite3.connect(DB_PATH)

    try:
        # Find predictions ready for outcome but not yet closed
        rows = conn.execute(
            """
            SELECT p.id, p.symbol, p.run_at, p.horizon_h, p.direction
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.prediction_id IS NULL
              AND (p.run_at + (p.horizon_h * 3600)) <= ?
            ORDER BY p.run_at
            """,
            (now,),
        ).fetchall()

        if not rows:
            LOGGER.debug("No predictions ready for outcome reconciliation")
            return

        LOGGER.info(f"Reconciling outcomes for {len(rows)} predictions")

        for pred_id, symbol, run_at, horizon_h, pred_direction in rows:
            try:
                _reconcile_single(pred_id, symbol, run_at, horizon_h, pred_direction)
            except Exception as e:
                LOGGER.error(f"Failed to reconcile prediction {pred_id}: {e}", exc_info=True)

    finally:
        conn.close()


def _reconcile_single(
    pred_id: int, symbol: str, run_at: float, horizon_h: int, pred_direction: str
):
    """Reconcile a single prediction."""
    # Get forecast points
    forecast_pts = get_prediction_points(pred_id, kind="forecast")
    if not forecast_pts:
        LOGGER.warning(f"Prediction {pred_id} has no forecast points, skipping")
        return

    # Get actual points
    actual_pts = get_prediction_points(pred_id, kind="actual")
    if not actual_pts:
        LOGGER.warning(f"Prediction {pred_id} has no actual points, skipping")
        return

    # Align on timestamps (find matching pairs within tolerance)
    forecast_map = {p.ts: p.price for p in forecast_pts}
    actual_map = {p.ts: p.price for p in actual_pts}

    # Match timestamps with 60s tolerance
    aligned_f = []
    aligned_a = []
    for f_ts, f_price in sorted(forecast_map.items()):
        # Find closest actual within 60s
        best_match = None
        best_diff = float("inf")
        for a_ts, a_price in actual_map.items():
            diff = abs(a_ts - f_ts)
            if diff < best_diff and diff <= 60:
                best_diff = diff
                best_match = (a_ts, a_price)

        if best_match:
            aligned_f.append(f_price)
            aligned_a.append(best_match[1])

    if len(aligned_f) < 2:
        LOGGER.warning(
            f"Prediction {pred_id} has insufficient aligned points ({len(aligned_f)}), skipping"
        )
        return

    # Compute metrics
    metrics = compute_metrics(aligned_f, aligned_a)
    mae = metrics["mae"]
    map = metrics["map"]
    rmse = metrics["rmse"]

    # Direction hit: compare price at run_at vs price at run_at+48h
    price_at_start = forecast_pts[0].price if forecast_pts else None
    price_forecast_end = forecast_pts[-1].price if forecast_pts else None

    # Find actual price closest to run_at+48h
    end_ts = run_at + (horizon_h * 3600)
    actual_end = None
    best_diff = float("inf")
    for pt in actual_pts:
        diff = abs(pt.ts - end_ts)
        if diff < best_diff:
            best_diff = diff
            actual_end = pt.price

    hit_direction = 0
    if price_at_start and price_forecast_end and actual_end:
        forecast_direction_up = price_forecast_end > price_at_start
        actual_direction_up = actual_end > price_at_start

        # Hit if directions match
        if forecast_direction_up == actual_direction_up:
            hit_direction = 1

        # Also check against declared direction
        if pred_direction == "UP" and actual_direction_up:
            hit_direction = 1
        elif pred_direction == "DOWN" and not actual_direction_up:
            hit_direction = 1
        elif pred_direction == "FLAT" and abs(actual_end - price_at_start) < (
            0.01 * price_at_start
        ):
            hit_direction = 1

    # Hit ratio window: fraction of aligned points where forecast is within 5% of actual
    hits_window = sum(
        1 for f, a in zip(aligned_f, aligned_a, strict=False) if abs(f - a) <= 0.05 * a
    )
    hit_ratio_window = hits_window / len(aligned_f) if aligned_f else 0.0

    notes = f"Aligned {len(aligned_f)} points"

    # Persist outcome
    create_outcome(
        prediction_id=pred_id,
        mae=mae,
        map=map,
        rmse=rmse,
        hit_direction=hit_direction,
        hit_ratio_window=hit_ratio_window,
        notes=notes,
    )

    LOGGER.info(
        f"Reconciled prediction {pred_id} ({symbol}): MAE={mae:.4f}, MAP={map:.2f}%, RMSE={rmse:.4f}, hit={hit_direction}"
    )
