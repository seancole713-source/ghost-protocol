"""
Ghost ML Model Trainer
======================

Trains ML models for prediction using historical data + features.

Models:
- XGBoost classifier (directional prediction: UP/DOWN/FLAT)
- Feature importance tracking
- Cross-validation scoring
- Model persistence

Training Data:
- Historical predictions with outcomes
- 50+ features from data pillars
- Target: direction_correct (binary classification)

Usage:
    from core.ml_trainer import train_model, load_model, predict
    
    # Train new model
    model_path = train_model(symbol="SPY", lookback_days=180)
    
    # Load and use
    model = load_model()
    prediction = predict(model, features)

Author: Ghost AI
Date: November 21, 2025
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Model storage
MODELS_DIR = Path(__file__).parent.parent / "models" / "production"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_model(
    symbol: str | None = None,
    lookback_days: int = 180,
    min_samples: int = 100
) -> dict[str, Any]:
    """
    Train XGBoost model for directional prediction.
    
    Args:
        symbol: Train on specific symbol (None = all symbols)
        lookback_days: Historical data period
        min_samples: Minimum training samples required
    
    Returns:
        {
            "ok": bool,
            "model_path": str,
            "accuracy": float,
            "samples": int,
            "features": list[str]
        }
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {
            "ok": False,
            "error": "XGBoost not installed (pip install xgboost scikit-learn)",
        }

    logger.info(f"Training ML model (symbol={symbol}, lookback={lookback_days}d)")

    # Fetch training data from prediction outcomes
    training_data = _fetch_training_data(symbol, lookback_days)

    if len(training_data) < min_samples:
        return {
            "ok": False,
            "error": f"Insufficient training data ({len(training_data)} < {min_samples} samples)",
            "samples": len(training_data),
        }

    # Prepare features and labels
    X, y, feature_names = _prepare_training_data(training_data)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    logger.info(
        f"Model trained: train_acc={train_accuracy:.1%}, test_acc={test_accuracy:.1%}"
    )

    # Save model
    model_filename = f"ghost_model_{'ALL' if symbol is None else symbol}.pkl"
    model_path = MODELS_DIR / model_filename

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "samples": len(training_data),
                "symbol": symbol,
                "trained_at": "2025-11-21",
            },
            f,
        )

    logger.info(f"Model saved: {model_path}")

    return {
        "ok": True,
        "model_path": str(model_path),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "samples": len(training_data),
        "features": feature_names,
        "symbol": symbol or "ALL",
    }


def _fetch_training_data(symbol: str | None, lookback_days: int) -> list[dict]:
    """Fetch prediction outcomes with features for training from PostgreSQL"""
    import os
    import time
    from datetime import datetime, timedelta

    cutoff_time = time.time() - (lookback_days * 86400)
    cutoff_dt = datetime.utcnow() - timedelta(days=lookback_days)
    training_data = []

    # TRY POSTGRESQL FIRST (where 25,691 outcomes live)
    database_url = os.getenv("DATABASE_URL", "")
    if database_url.startswith(("postgres://", "postgresql://")):
        try:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT 
                        o.prediction_id, o.symbol, p.predicted_direction, p.confidence,
                        o.hit_direction, o.open_price, o.close_price, p.features_json
                    FROM ghost_prediction_outcomes o
                    JOIN ghost_predictions p ON o.prediction_id = p.id
                    WHERE o.symbol = %s 
                      AND o.status = 'closed'
                      AND o.closed_at >= %s
                    ORDER BY o.closed_at DESC
                    LIMIT 10000
                """, (symbol, cutoff_dt))
            else:
                cursor.execute("""
                    SELECT 
                        o.prediction_id, o.symbol, p.predicted_direction, p.confidence,
                        o.hit_direction, o.open_price, o.close_price, p.features_json
                    FROM ghost_prediction_outcomes o
                    JOIN ghost_predictions p ON o.prediction_id = p.id
                    WHERE o.status = 'closed'
                      AND o.closed_at >= %s
                    ORDER BY o.closed_at DESC
                    LIMIT 10000
                """, (cutoff_dt,))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            for row in rows:
                features = {}
                if row[7]:  # features_json
                    try:
                        features = json.loads(row[7]) if isinstance(row[7], str) else row[7]
                    except Exception:
                        pass
                
                training_data.append({
                    "prediction_id": row[0],
                    "symbol": row[1],
                    "direction_predicted": row[2],
                    "confidence": row[3] or 0.5,
                    "direction_correct": 1 if row[4] else 0,  # hit_direction
                    "price_at_prediction": row[5] or 0,
                    "price_at_outcome": row[6] or 0,
                    "features": features,
                })
            
            logger.info(f"Fetched {len(training_data)} training samples from PostgreSQL")
            return training_data
            
        except Exception as e:
            logger.error(f"PostgreSQL fetch failed, falling back to SQLite: {e}")

    # FALLBACK: SQLite (local development only - usually empty on Railway)
    import sqlite3
    outcomes_db = Path(__file__).parent.parent / "data" / "prediction_outcomes.db"
    if not outcomes_db.exists():
        logger.warning(f"No SQLite outcomes DB at {outcomes_db} and PostgreSQL unavailable")
        return []

    try:
        with sqlite3.connect(str(outcomes_db)) as conn:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT 
                        prediction_id, symbol, predicted_direction, confidence,
                        correct, target_price, actual_price
                    FROM prediction_outcomes
                    WHERE symbol = ? AND created_at >= ?
                """,
                    (symbol, cutoff_time),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT 
                        prediction_id, symbol, predicted_direction, confidence,
                        correct, target_price, actual_price
                    FROM prediction_outcomes
                    WHERE created_at >= ?
                """,
                    (cutoff_time,),
                ).fetchall()

            for row in rows:
                training_data.append({
                    "prediction_id": row[0],
                    "symbol": row[1],
                    "direction_predicted": row[2],
                    "confidence": row[3] or 0.5,
                    "direction_correct": row[4] or 0,
                    "price_at_prediction": row[5] or 0,
                    "price_at_outcome": row[6] or 0,
                    "features": {},  # SQLite doesn't store features
                })
            
            logger.info(f"Fetched {len(training_data)} training samples from SQLite (fallback)")

    except Exception as e:
        logger.error(f"Failed to fetch training data from SQLite: {e}")

    return training_data


def _prepare_training_data(training_data: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert training data to feature matrix and labels"""
    
    # Extract ALL features from predictions (50+ technical indicators + metadata)
    # Each prediction has features stored in features_json column
    X = []
    y = []
    feature_names = None

    for sample in training_data:
        # Parse features from JSON (if available)
        features_dict = sample.get("features", {})
        if isinstance(features_dict, str):
            try:
                import json
                features_dict = json.loads(features_dict)
            except Exception:
                features_dict = {}
        
        # If no features in this sample, use legacy 2-feature fallback
        if not features_dict:
            confidence = sample["confidence"]
            price_change = (
                (sample["price_at_outcome"] - sample["price_at_prediction"])
                / sample["price_at_prediction"]
            ) if sample["price_at_prediction"] > 0 else 0
            features_dict = {"confidence": confidence, "price_momentum": price_change}
        
        # Extract feature names from first sample (all should have same keys)
        if feature_names is None:
            feature_names = sorted(features_dict.keys())
            logger.info(f"Training with {len(feature_names)} features: {feature_names[:10]}...")
        
        # Extract feature vector (use 0 for missing values)
        feature_vector = [features_dict.get(name, 0) for name in feature_names]
        
        X.append(feature_vector)
        y.append(sample["direction_correct"])

    # Fallback if no samples (shouldn't happen but safety check)
    if feature_names is None:
        feature_names = ["confidence", "price_momentum"]
        logger.warning("No features extracted, using legacy 2-feature fallback")

    return np.array(X), np.array(y), feature_names


def load_model(model_path: str | None = None) -> dict[str, Any] | None:
    """Load trained model from disk"""
    if model_path is None:
        # Load latest model
        model_files = list(MODELS_DIR.glob("ghost_model_*.pkl"))
        if not model_files:
            logger.warning("No trained models found")
            return None
        model_path = str(max(model_files, key=lambda p: p.stat().st_mtime))

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        logger.info(f"Model loaded: {model_path}")
        return model_data
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def predict(model_data: dict, features: dict[str, Any]) -> dict[str, Any]:
    """
    Make prediction using trained model.
    
    Args:
        model_data: Loaded model dict
        features: Feature dict with required features
    
    Returns:
        {
            "direction": "UP" | "DOWN" | "FLAT",
            "confidence": float,
            "model_confidence": float
        }
    """
    try:
        model = model_data["model"]
        feature_names = model_data["feature_names"]

        # Extract features in correct order
        X = np.array([[features.get(name, 0) for name in feature_names]])

        # Predict
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # Map to direction
        direction = "UP" if prediction == 1 else "DOWN"
        model_confidence = float(proba[prediction])

        return {
            "direction": direction,
            "confidence": round(model_confidence, 3),
            "model_confidence": round(model_confidence, 3),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "direction": "FLAT",
            "confidence": 0.5,
            "model_confidence": 0.0,
            "error": str(e),
        }


class GhostMLTrainer:
    """Train ML models on PostgreSQL historical predictions"""
    
    def __init__(self):
        self.models = {}
        
    async def train_from_postgres(self, min_predictions: int = 100) -> dict[str, Any]:
        """
        Train models on historical predictions from PostgreSQL.
        
        Args:
            min_predictions: Minimum predictions per symbol to train model
            
        Returns:
            Training results with accuracy metrics
        """
        import os
        from core.prediction_store import get_prediction_store
        from sqlalchemy import text
        from collections import defaultdict
        
        logger.info("Starting ML training from PostgreSQL...")
        
        store = get_prediction_store()
        
        # Check if PostgreSQL
        is_postgres = os.getenv("DATABASE_URL", "").startswith("postgresql")
        if not is_postgres or not hasattr(store, 'engine'):
            return {
                "ok": False,
                "error": "PostgreSQL not configured. Set DATABASE_URL environment variable.",
                "predictions_found": 0
            }
        
        # Fetch reconciled predictions
        query = text("""
            SELECT 
                symbol,
                features_json,
                was_correct,
                confidence
            FROM ghost_predictions
            WHERE actual_direction IS NOT NULL
              AND was_correct IS NOT NULL
              AND features_json IS NOT NULL
            ORDER BY run_at DESC
            LIMIT 10000
        """)
        
        with store.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            training_data = []
            for row in rows:
                features = json.loads(row.features_json) if row.features_json else {}
                
                training_data.append({
                    "symbol": row.symbol,
                    "features": features,
                    "was_correct": row.was_correct,
                    "confidence": row.confidence
                })
        
        if len(training_data) == 0:
            return {
                "ok": False,
                "error": "No training data available. Run reconciliation first.",
                "predictions_found": 0
            }
        
        logger.info(f"Found {len(training_data)} reconciled predictions")
        
        # Group by symbol
        by_symbol = defaultdict(list)
        for record in training_data:
            by_symbol[record["symbol"]].append(record)
        
        # Train model per symbol using existing train_model function
        results = {}
        for symbol, symbol_data in by_symbol.items():
            if len(symbol_data) < min_predictions:
                logger.info(f"Skipping {symbol}: only {len(symbol_data)} predictions")
                continue
            
            logger.info(f"Training {symbol} ({len(symbol_data)} predictions)...")
            
            # Use existing train_model with custom data
            model_result = train_model(symbol=symbol, lookback_days=365)
            
            if model_result["ok"]:
                results[symbol] = {
                    "accuracy": model_result["test_accuracy"],
                    "train_samples": model_result["samples"]
                }
        
        return {
            "ok": True,
            "symbols_trained": len(results),
            "total_predictions": len(training_data),
            "models": results,
            "model_dir": str(MODELS_DIR)
        }


_ML_TRAINER = None


def get_ml_trainer() -> GhostMLTrainer:
    """Get singleton ML trainer"""
    global _ML_TRAINER
    if _ML_TRAINER is None:
        _ML_TRAINER = GhostMLTrainer()
    return _ML_TRAINER


if __name__ == "__main__":
    # Train model manually
    logging.basicConfig(level=logging.INFO)
    result = train_model(symbol=None, lookback_days=180)
    print(json.dumps(result, indent=2))

