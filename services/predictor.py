#!/usr/bin/env python3
"""
Ghost Prediction Service

Generates 48h price forecasts for stocks using existing Ghost forecasting engine,
stores predictions with full curve data, and tracks actual prices for accuracy metrics.
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Import prediction store abstraction
from core.prediction_store import get_prediction_store

LOGGER = logging.getLogger("ghost.predictor")

# Database path (legacy, now used by prediction_store)
DB_PATH = os.getenv("GHOST_PREDICT_DB", "./data/ghost_predictions.db")

# Global prediction store instance
_PREDICTION_STORE = get_prediction_store()


@dataclass
class Prediction:
    """Prediction metadata"""

    id: int | None
    symbol: str
    run_at: float  # Unix timestamp
    horizon_h: int
    method: str
    confidence: float
    direction: str  # UP/DOWN/FLAT
    features_json: str
    params_json: str
    tag: str


@dataclass
class PredictionPoint:
    """Single forecast or actual data point"""

    id: int | None
    prediction_id: int
    ts: float  # Unix timestamp
    kind: str  # forecast/actual
    price: float


@dataclass
class Outcome:
    """Prediction outcome metrics"""

    prediction_id: int
    closed_at: float
    mae: float
    map: float
    rmse: float
    hit_direction: int  # 0 or 1
    hit_ratio_window: float | None
    notes: str


def _init_db():
    """Initialize database schema. Idempotent."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # predictions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            run_at REAL NOT NULL,
            horizon_h INTEGER NOT NULL DEFAULT 48,
            method TEXT NOT NULL,
            confidence REAL NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('UP','DOWN','FLAT')),
            features_json TEXT,
            params_json TEXT,
            tag TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_run ON predictions(symbol, run_at DESC)"
    )

    # prediction_points table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ts REAL NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('forecast','actual')),
            price REAL NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_points_pred_kind ON prediction_points(prediction_id, kind, ts)"
    )

    # outcomes table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            prediction_id INTEGER NOT NULL UNIQUE,
            closed_at REAL NOT NULL,
            mae REAL NOT NULL,
            map REAL NOT NULL,
            rmse REAL NOT NULL,
            hit_direction INTEGER NOT NULL CHECK(hit_direction IN (0,1)),
            hit_ratio_window REAL,
            notes TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_pred ON outcomes(prediction_id)")

    conn.commit()
    conn.close()
    LOGGER.info(f"Prediction database initialized: {DB_PATH}")


# Initialize on module load
_init_db()


def create_prediction(
    symbol: str,
    forecast_points: list[tuple[float, float]],  # [(ts, price), ...]
    method: str = "ghost-av1",
    confidence: float = 0.6,
    direction: str = "FLAT",
    features: dict | None = None,
    params: dict | None = None,
    tag: str = "",
) -> int:
    """
    Create a new prediction with forecast points.

    Args:
        symbol: Stock ticker
        forecast_points: List of (timestamp, price) tuples for 48h forecast
        method: Forecasting method identifier
        confidence: Model confidence [0-1]
        direction: Price direction (UP/DOWN/FLAT)
        features: Feature dictionary (serialized to JSON)
        params: Model parameters (serialized to JSON)
        tag: Optional tag/label

    Returns:
        prediction_id
    """
    # Use PredictionStore abstraction (handles SQLite or PostgreSQL)
    prediction_id = _PREDICTION_STORE.save_prediction(
        symbol=symbol,
        forecast_points=forecast_points,
        method=method,
        confidence=confidence,
        direction=direction,
        features=features or {},
        params=params or {"horizon_h": 48},
        tag=tag,
    )
    
    return prediction_id


def append_actual_points(prediction_id: int, actual_points: list[tuple[float, float]]):
    """
    Append actual price points to a prediction for comparison.

    Args:
        prediction_id: Prediction ID
        actual_points: List of (timestamp, price) tuples
    """
    # Use PredictionStore abstraction
    _PREDICTION_STORE.append_actual_points(prediction_id, actual_points)


def get_prediction(prediction_id: int) -> Prediction | None:
    """Get prediction metadata by ID."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT id, symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag FROM predictions WHERE id=?",
            (prediction_id,),
        ).fetchone()
        if not row:
            return None
        return Prediction(*row)
    finally:
        conn.close()


def get_prediction_points(prediction_id: int, kind: str | None = None) -> list[PredictionPoint]:
    """Get forecast or actual points for a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if kind:
            rows = conn.execute(
                "SELECT id, prediction_id, ts, kind, price FROM prediction_points WHERE prediction_id=? AND kind=? ORDER BY ts",
                (prediction_id, kind),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, prediction_id, ts, kind, price FROM prediction_points WHERE prediction_id=? ORDER BY ts",
                (prediction_id,),
            ).fetchall()
        return [PredictionPoint(*r) for r in rows]
    finally:
        conn.close()


def get_latest_prediction(symbol: str) -> Prediction | None:
    """Get most recent prediction for a symbol."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT id, symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag FROM predictions WHERE symbol=? ORDER BY run_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if not row:
            return None
        return Prediction(*row)
    finally:
        conn.close()


def get_prediction_history(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    """Get prediction history with outcomes for a symbol."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT
                p.id, p.symbol, p.run_at, p.horizon_h, p.method, p.confidence, p.direction, p.tag,
                o.closed_at, o.mae, o.map, o.rmse, o.hit_direction, o.hit_ratio_window, o.notes
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            WHERE p.symbol = ?
            ORDER BY p.run_at DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()

        results = []
        for r in rows:
            closed = r[8] is not None
            results.append(
                {
                    "id": r[0],
                    "symbol": r[1],
                    "run_at": r[2],
                    "horizon_h": r[3],
                    "method": r[4],
                    "confidence": r[5],
                    "direction": r[6],
                    "tag": r[7],
                    "closed": closed,
                    "closed_at": r[8],
                    "mae": r[9],
                    "map": r[10],
                    "rmse": r[11],
                    "hit_direction": r[12],
                    "hit_ratio_window": r[13],
                    "notes": r[14],
                }
            )
        return results
    finally:
        conn.close()


def create_outcome(
    prediction_id: int,
    mae: float,
    map: float,
    rmse: float,
    hit_direction: int,
    hit_ratio_window: float | None = None,
    notes: str = "",
):
    """Create outcome record for a closed prediction."""
    closed_at = time.time()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO outcomes (prediction_id, closed_at, mae, map, rmse, hit_direction, hit_ratio_window, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (prediction_id, closed_at, mae, map, rmse, hit_direction, hit_ratio_window, notes),
        )
        conn.commit()
        LOGGER.info(
            f"Created outcome for prediction {prediction_id}: MAE={mae:.4f}, hit={hit_direction}"
        )
    finally:
        conn.close()


def compute_metrics(forecast: list[float], actual: list[float]) -> dict[str, float]:
    """
    Compute MAE, MAP, RMSE for aligned forecast vs actual arrays.

    Args:
        forecast: Predicted prices
        actual: Actual prices (same length)

    Returns:
        dict with mae, map, rmse
    """
    forecast = np.array(forecast, dtype=float)
    actual = np.array(actual, dtype=float)

    if len(forecast) != len(actual) or len(forecast) == 0:
        return {"mae": float("nan"), "map": float("nan"), "rmse": float("nan")}

    errors = np.abs(forecast - actual)
    mae = float(np.mean(errors))

    # MAP: avoid division by zero
    nonzero = actual != 0
    if nonzero.sum() > 0:
        map = float(np.mean(np.abs((actual[nonzero] - forecast[nonzero]) / actual[nonzero])) * 100)
    else:
        map = float("nan")

    rmse = float(np.sqrt(np.mean((forecast - actual) ** 2)))

    return {"mae": mae, "map": map, "rmse": rmse}


def get_scoreboard(symbol: str, windows: list[int] = None) -> dict[str, Any]:
    """
    Compute aggregate accuracy metrics for a symbol over time windows.

    Args:
        symbol: Stock ticker
        windows: List of day windows (e.g., [7, 30]) for windowed stats

    Returns:
        dict with overall and windowed stats
    """
    if windows is None:
        windows = [7, 30]

    conn = sqlite3.connect(DB_PATH)
    try:
        # Overall stats
        rows = conn.execute(
            """
            SELECT p.confidence, o.mae, o.map, o.rmse, o.hit_direction
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            WHERE p.symbol = ?
            """,
            (symbol,),
        ).fetchall()

        if not rows:
            return {
                "overall": {"count": 0},
                **{f"w{w}d": {"count": 0} for w in windows},
            }

        confs = [r[0] for r in rows]
        maes = [r[1] for r in rows]
        mapes = [r[2] for r in rows]
        rmses = [r[3] for r in rows]
        hits = [r[4] for r in rows]

        overall = {
            "count": len(rows),
            "hit_dir_pct": round(100 * sum(hits) / len(hits), 2) if hits else 0,
            "mae": round(float(np.mean(maes)), 4) if maes else 0,
            "map": round(float(np.mean(mapes)), 4) if mapes else 0,
            "rmse": round(float(np.mean(rmses)), 4) if rmses else 0,
            "avg_conf": round(float(np.mean(confs)), 4) if confs else 0,
        }

        # Brier-like calibration: |avg_conf - hit_rate|
        overall["calibration_gap"] = round(
            abs(overall["avg_conf"] - overall["hit_dir_pct"] / 100), 4
        )

        result = {"overall": overall}

        # Windowed stats
        now = time.time()
        for window_days in windows:
            cutoff = now - (window_days * 86400)
            windowed = conn.execute(
                """
                SELECT p.confidence, o.mae, o.map, o.rmse, o.hit_direction
                FROM predictions p
                JOIN outcomes o ON p.id = o.prediction_id
                WHERE p.symbol = ? AND o.closed_at >= ?
                """,
                (symbol, cutoff),
            ).fetchall()

            if windowed:
                w_confs = [r[0] for r in windowed]
                w_maes = [r[1] for r in windowed]
                w_mapes = [r[2] for r in windowed]
                w_rmses = [r[3] for r in windowed]
                w_hits = [r[4] for r in windowed]

                w_stats = {
                    "count": len(windowed),
                    "hit_dir_pct": round(100 * sum(w_hits) / len(w_hits), 2),
                    "mae": round(float(np.mean(w_maes)), 4),
                    "map": round(float(np.mean(w_mapes)), 4),
                    "rmse": round(float(np.mean(w_rmses)), 4),
                    "avg_conf": round(float(np.mean(w_confs)), 4),
                }
                w_stats["calibration_gap"] = round(
                    abs(w_stats["avg_conf"] - w_stats["hit_dir_pct"] / 100), 4
                )
            else:
                w_stats = {"count": 0}

            result[f"w{window_days}d"] = w_stats

        return result
    finally:
        conn.close()
