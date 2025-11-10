"""
Stage 3: Multi-Model Ensemble Forecaster
Level 9→10 (90%→100%)

Combines multiple forecast models with dynamic weighting based on accuracy.
Models: Ghost-AI baseline, technical indicators, sentiment-based, moving averages.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from core.concurrency import ExecutionTimer

LOGGER = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Multi-model ensemble forecaster with dynamic weighting.

    Models:
    1. Ghost-AI: Baseline drift model (existing)
    2. Technical: RSI + MACD + Bollinger Bands
    3. Sentiment: News sentiment momentum
    4. Momentum: Simple moving average crossover

    Weighting: Inverse MAP (better models get higher weight)
    """

    def __init__(self, db_path: str = "data/ensemble_forecaster.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Default equal weights (updated dynamically)
        self.weights = {
            "ghost_ai": 0.40,  # Baseline model
            "technical": 0.25,  # TA indicators
            "sentiment": 0.20,  # News-based
            "momentum": 0.15,  # MA crossover
        }

        # Performance tracking
        self.model_mape = {model: 5.0 for model in self.weights.keys()}

        self._init_db()
        self._load_weights()
        LOGGER.info(f"Ensemble forecaster initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for ensemble predictions."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_forecasts (
                forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                forecast_time TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL,
                current_price REAL NOT NULL,

                -- Individual model predictions
                ghost_ai_pred REAL,
                technical_pred REAL,
                sentiment_pred REAL,
                momentum_pred REAL,

                -- Ensemble prediction
                ensemble_pred REAL NOT NULL,

                -- Weights used
                ghost_ai_weight REAL,
                technical_weight REAL,
                sentiment_weight REAL,
                momentum_weight REAL,

                -- Actual outcome
                actual_price REAL,
                accuracy_pct REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT PRIMARY KEY,
                map REAL DEFAULT 5.0,
                rmse REAL DEFAULT 1.0,
                bias REAL DEFAULT 0.0,
                weight REAL DEFAULT 0.25,
                forecast_count INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ensemble_symbol_time
            ON ensemble_forecasts(symbol, forecast_time DESC)
        """)

        conn.commit()
        conn.close()

    def _load_weights(self):
        """Load latest weights from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT model_name, weight, map FROM model_performance")
            rows = cursor.fetchall()

            if rows:
                for model_name, weight, map in rows:
                    if model_name in self.weights:
                        self.weights[model_name] = weight
                        self.model_mape[model_name] = map

            conn.close()
            LOGGER.info(f"Loaded weights: {self.weights}")
        except Exception as e:
            LOGGER.warning(f"Could not load weights: {e}, using defaults")

    def _save_weights(self):
        """Save current weights to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            for model_name, weight in self.weights.items():
                cursor.execute(
                    """
                    INSERT INTO model_performance (model_name, weight, map, last_updated)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name) DO UPDATE SET
                        weight = excluded.weight,
                        map = excluded.map,
                        last_updated = excluded.last_updated
                """,
                    (
                        model_name,
                        weight,
                        self.model_mape[model_name],
                        datetime.utcnow().isoformat(),
                    ),
                )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to save weights: {e}")

    def forecast(
        self,
        symbol: str,
        current_price: float,
        horizon_hours: int = 24,
        historical_prices: list[float] | None = None,
        sentiment_score: float = 0.0,
        volume: float = 0.0,
    ) -> dict:
        """
        Generate ensemble forecast combining all models.

        Args:
            symbol: Ticker symbol
            current_price: Current market price
            horizon_hours: Forecast horizon (default 24h)
            historical_prices: Last 20+ prices for technical analysis
            sentiment_score: News sentiment (-1 to +1)
            volume: Current volume (for momentum)

        Returns:
            Dict with ensemble prediction + individual model predictions
        """
        with ExecutionTimer(f"ensemble_forecast:{symbol}", logger=LOGGER):
            # Model 1: Ghost-AI baseline (drift model)
            ghost_pred = self._ghost_ai_model(current_price, sentiment_score, horizon_hours)

            # Model 2: Technical indicators
            tech_pred = self._technical_model(current_price, historical_prices)

            # Model 3: Sentiment momentum
            sent_pred = self._sentiment_model(current_price, sentiment_score, horizon_hours)

            # Model 4: Moving average momentum
            mom_pred = self._momentum_model(current_price, historical_prices)

            # Weighted ensemble
            ensemble_pred = (
                self.weights["ghost_ai"] * ghost_pred
                + self.weights["technical"] * tech_pred
                + self.weights["sentiment"] * sent_pred
                + self.weights["momentum"] * mom_pred
            )

            # Record forecast
            forecast_id = self._record_forecast(
                symbol=symbol,
                current_price=current_price,
                horizon_hours=horizon_hours,
                ghost_pred=ghost_pred,
                tech_pred=tech_pred,
                sent_pred=sent_pred,
                mom_pred=mom_pred,
                ensemble_pred=ensemble_pred,
            )

            return {
                "forecast_id": forecast_id,
                "symbol": symbol,
                "current_price": current_price,
                "horizon_hours": horizon_hours,
                "ensemble_prediction": round(ensemble_pred, 2),
                "model_predictions": {
                    "ghost_ai": round(ghost_pred, 2),
                    "technical": round(tech_pred, 2),
                    "sentiment": round(sent_pred, 2),
                    "momentum": round(mom_pred, 2),
                },
                "weights": self.weights.copy(),
                "confidence": self._calculate_confidence(
                    ghost_pred, tech_pred, sent_pred, mom_pred
                ),
            }

    def _ghost_ai_model(self, price: float, sentiment: float, horizon_hours: int) -> float:
        """
        Ghost-AI baseline: Drift model with sentiment adjustment.
        (This replicates existing forecast logic)
        """
        # Momentum: 30% of recent drift
        momentum_factor = 0.003 * (horizon_hours / 24)

        # Sentiment: 1% impact per 0.1 sentiment
        sentiment_factor = sentiment * 0.01

        # Combined drift
        drift = (momentum_factor + sentiment_factor) * price

        return price + drift

    def _technical_model(self, price: float, historical: list[float] | None) -> float:
        """
        Technical analysis model: RSI + MACD + Bollinger Bands.
        """
        if not historical or len(historical) < 14:
            # Fallback: slight mean reversion
            return price * 0.998

        prices = np.array(historical[-20:])

        # Simple RSI calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # RSI signal: >70 overbought (predict down), <30 oversold (predict up)
        if rsi > 70:
            rsi_signal = -0.005  # -0.5% correction
        elif rsi < 30:
            rsi_signal = 0.005  # +0.5% bounce
        else:
            rsi_signal = 0.0

        # Bollinger Bands: mean reversion
        ma20 = np.mean(prices)
        std20 = np.std(prices)
        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20

        if price > upper_band:
            bb_signal = -0.003  # Overbought
        elif price < lower_band:
            bb_signal = 0.003  # Oversold
        else:
            bb_signal = 0.0

        # Combined technical signal
        total_signal = rsi_signal + bb_signal

        return price * (1 + total_signal)

    def _sentiment_model(self, price: float, sentiment: float, horizon_hours: int) -> float:
        """
        Sentiment-based model: Momentum follows news sentiment.
        """
        # Strong sentiment = stronger momentum
        sentiment_momentum = sentiment * 0.015 * (horizon_hours / 24)

        return price * (1 + sentiment_momentum)

    def _momentum_model(self, price: float, historical: list[float] | None) -> float:
        """
        Moving average crossover model.
        """
        if not historical or len(historical) < 10:
            return price * 1.001  # Slight bullish bias

        prices = np.array(historical[-20:])

        # Fast MA (5-period) vs Slow MA (20-period)
        ma5 = np.mean(prices[-5:])
        ma20 = np.mean(prices)

        # Crossover signal
        if ma5 > ma20:
            # Golden cross: bullish
            momentum = 0.004
        elif ma5 < ma20:
            # Death cross: bearish
            momentum = -0.004
        else:
            momentum = 0.0

        return price * (1 + momentum)

    def _calculate_confidence(self, ghost: float, tech: float, sent: float, mom: float) -> float:
        """
        Calculate ensemble confidence based on model agreement.
        High agreement = high confidence.
        """
        preds = np.array([ghost, tech, sent, mom])
        std = np.std(preds)
        mean = np.mean(preds)

        # Coefficient of variation (inverse = confidence)
        cv = std / (mean + 1e-10)

        # Map to 0-1 range (low CV = high confidence)
        confidence = 1.0 / (1.0 + cv * 10)

        return float(min(0.99, max(0.50, confidence)))

    def _record_forecast(
        self,
        symbol: str,
        current_price: float,
        horizon_hours: int,
        ghost_pred: float,
        tech_pred: float,
        sent_pred: float,
        mom_pred: float,
        ensemble_pred: float,
    ) -> int:
        """Record ensemble forecast in database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO ensemble_forecasts (
                symbol, forecast_time, horizon_hours, current_price,
                ghost_ai_pred, technical_pred, sentiment_pred, momentum_pred,
                ensemble_pred,
                ghost_ai_weight, technical_weight, sentiment_weight, momentum_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                symbol,
                datetime.utcnow().isoformat(),
                horizon_hours,
                current_price,
                ghost_pred,
                tech_pred,
                sent_pred,
                mom_pred,
                ensemble_pred,
                self.weights["ghost_ai"],
                self.weights["technical"],
                self.weights["sentiment"],
                self.weights["momentum"],
            ),
        )

        forecast_id = cursor.lastrowid
        conn.commit()
        conn.close()

        assert forecast_id is not None, "Failed to get forecast_id"
        return forecast_id

    def update_actual(self, forecast_id: int, actual_price: float):
        """
        Update forecast with actual price and recalculate model weights.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get forecast details
        cursor.execute(
            """
            SELECT ensemble_pred, ghost_ai_pred, technical_pred, sentiment_pred, momentum_pred
            FROM ensemble_forecasts WHERE forecast_id = ?
        """,
            (forecast_id,),
        )

        row = cursor.fetchone()
        if not row:
            conn.close()
            return

        ensemble_pred, ghost_pred, tech_pred, sent_pred, mom_pred = row

        # Calculate accuracy
        accuracy_pct = abs((actual_price - ensemble_pred) / actual_price) * 100

        # Update forecast record
        cursor.execute(
            """
            UPDATE ensemble_forecasts
            SET actual_price = ?, accuracy_pct = ?
            WHERE forecast_id = ?
        """,
            (actual_price, accuracy_pct, forecast_id),
        )

        conn.commit()
        conn.close()

        # Update model performance and reweight
        self._update_model_performance(actual_price, ghost_pred, tech_pred, sent_pred, mom_pred)

    def _update_model_performance(
        self, actual: float, ghost_pred: float, tech_pred: float, sent_pred: float, mom_pred: float
    ):
        """
        Update per-model MAP and recompute weights.
        """
        # Calculate error for each model
        errors = {
            "ghost_ai": abs((actual - ghost_pred) / actual) * 100,
            "technical": abs((actual - tech_pred) / actual) * 100,
            "sentiment": abs((actual - sent_pred) / actual) * 100,
            "momentum": abs((actual - mom_pred) / actual) * 100,
        }

        # Update MAP with exponential moving average (alpha=0.3)
        for model, error in errors.items():
            self.model_mape[model] = 0.3 * error + 0.7 * self.model_mape[model]

        # Recompute weights (inverse MAP)
        inverse_mapes = {model: 1.0 / (map + 0.1) for model, map in self.model_mape.items()}
        total = sum(inverse_mapes.values())

        self.weights = {model: inv / total for model, inv in inverse_mapes.items()}

        # Save updated weights
        self._save_weights()

        LOGGER.info(f"Updated weights: {self.weights}, MAP: {self.model_mape}")

    def get_performance_report(self) -> dict:
        """Get performance report for all models."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get recent forecasts (last 30)
        cursor.execute("""
            SELECT COUNT(*), AVG(accuracy_pct)
            FROM ensemble_forecasts
            WHERE actual_price IS NOT NULL
            ORDER BY forecast_time DESC
            LIMIT 30
        """)

        total_forecasts, avg_accuracy = cursor.fetchone()

        # Get per-model stats
        model_stats = {}
        for model in self.weights.keys():
            cursor.execute(f"""
                SELECT
                    AVG(ABS((actual_price - {model}_pred) / actual_price * 100)) as map,
                    COUNT(*) as count
                FROM ensemble_forecasts
                WHERE actual_price IS NOT NULL
                ORDER BY forecast_time DESC
                LIMIT 30
            """)

            map, count = cursor.fetchone()
            model_stats[model] = {
                "map": round(map, 2) if map else 5.0,
                "weight": round(self.weights[model], 3),
                "forecast_count": count or 0,
            }

        conn.close()

        return {
            "total_forecasts": total_forecasts or 0,
            "ensemble_mape": round(avg_accuracy, 2) if avg_accuracy else 0.0,
            "model_stats": model_stats,
            "current_weights": self.weights,
        }


# Singleton instance
_ensemble_forecaster: EnsembleForecaster | None = None


def get_ensemble_forecaster() -> EnsembleForecaster:
    """Get singleton ensemble forecaster instance."""
    global _ensemble_forecaster
    if _ensemble_forecaster is None:
        _ensemble_forecaster = EnsembleForecaster()
    return _ensemble_forecaster
