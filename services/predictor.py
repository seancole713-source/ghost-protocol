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
    """
    DEPRECATED: Legacy SQLite initialization.
    
    This function is no longer used when PREDICTION_STORE_ENGINE=postgres.
    The prediction_store abstraction handles schema initialization for both backends.
    Only called in SQLite-only mode for backward compatibility.
    """
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
    backend_name = _PREDICTION_STORE.backend.__class__.__name__
    
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
    
    LOGGER.info(
        f"[{backend_name}] Created prediction {prediction_id} for {symbol}: "
        f"{direction} @ {confidence:.2f} confidence"
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
    """
    Get prediction metadata by ID.
    
    Uses PredictionStore abstraction (supports SQLite or PostgreSQL).
    """
    pred_dict = _PREDICTION_STORE.get_prediction(prediction_id)
    if not pred_dict:
        return None
    
    return Prediction(
        id=pred_dict["id"],
        symbol=pred_dict["symbol"],
        run_at=pred_dict["run_at"],
        horizon_h=pred_dict["horizon_h"],
        method=pred_dict["method"],
        confidence=pred_dict["confidence"],
        direction=pred_dict["direction"],
        features_json=pred_dict["features_json"],
        params_json=pred_dict["params_json"],
        tag=pred_dict["tag"],
    )


def get_prediction_points(prediction_id: int, kind: str | None = None) -> list[PredictionPoint]:
    """
    Get forecast or actual points for a prediction.
    
    Uses prediction_store abstraction (supports SQLite or PostgreSQL).
    """
    points_data = _PREDICTION_STORE.get_prediction_points(prediction_id, kind)
    return [
        PredictionPoint(
            id=p["id"],
            prediction_id=p["prediction_id"],
            ts=p["ts"],
            kind=p["kind"],
            price=p["price"]
        )
        for p in points_data
    ]


def get_latest_prediction(symbol: str) -> Prediction | None:
    """
    Get most recent prediction for a symbol.
    
    Uses PredictionStore abstraction (supports SQLite or PostgreSQL).
    """
    backend_name = _PREDICTION_STORE.backend.__class__.__name__
    
    pred_dict = _PREDICTION_STORE.get_latest_prediction(symbol)
    if not pred_dict:
        LOGGER.debug(f"[{backend_name}] No prediction found for {symbol}")
        return None
    
    LOGGER.info(
        f"[{backend_name}] Retrieved prediction {pred_dict['id']} for {symbol} "
        f"(run_at={pred_dict['run_at']:.0f})"
    )
    
    return Prediction(
        id=pred_dict["id"],
        symbol=pred_dict["symbol"],
        run_at=pred_dict["run_at"],
        horizon_h=pred_dict["horizon_h"],
        method=pred_dict["method"],
        confidence=pred_dict["confidence"],
        direction=pred_dict["direction"],
        features_json=pred_dict["features_json"],
        params_json=pred_dict["params_json"],
        tag=pred_dict["tag"],
    )


def get_prediction_history(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Get prediction history with outcomes for a symbol.
    
    Uses prediction_store abstraction (supports SQLite or PostgreSQL).
    """
    history = _PREDICTION_STORE.get_prediction_history(symbol, limit)
    # Add timestamp alias for backward compatibility
    for item in history:
        item["timestamp"] = item.get("run_at")
    return history


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
    _PREDICTION_STORE.create_outcome(
        prediction_id, mae, map, rmse, hit_direction, hit_ratio_window, notes
    )
    LOGGER.info(
        f"Created outcome for prediction {prediction_id}: MAE={mae:.4f}, hit={hit_direction}"
    )


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

    # Get predictions with outcomes through prediction_store
    rows = _PREDICTION_STORE.get_predictions_with_outcomes(symbol)

    if not rows:
        return {
            "overall": {"count": 0},
            **{f"w{w}d": {"count": 0} for w in windows},
        }

    confs = [r["confidence"] for r in rows]
    maes = [r["mae"] for r in rows]
    mapes = [r["map"] for r in rows]
    rmses = [r["rmse"] for r in rows]
    hits = [r["hit_direction"] for r in rows]

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
        windowed = _PREDICTION_STORE.get_predictions_with_outcomes_since(symbol, cutoff)

        if windowed:
            w_confs = [r["confidence"] for r in windowed]
            w_maes = [r["mae"] for r in windowed]
            w_mapes = [r["map"] for r in windowed]
            w_rmses = [r["rmse"] for r in windowed]
            w_hits = [r["hit_direction"] for r in windowed]

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
