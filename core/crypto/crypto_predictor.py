"""
Crypto Prediction Engine
24/7 operation with crypto-specific patterns
"""

import logging
import sqlite3
import time
import uuid
from typing import Any

import numpy as np

from .crypto_providers import CoinGeckoProvider, get_crypto_price_quorum

LOGGER = logging.getLogger(__name__)


class CryptoPredictionEngine:
    """
    Crypto prediction engine with 24/7 operation

    Key differences from stock predictions:
    - No market hours constraints
    - Higher volatility acceptance
    - Shorter forecast horizon (24h vs 48h)
    - Faster update cycles (5min vs 15min)
    """

    def __init__(self, db_path: str = "ai_memory.db"):
        self.db_path = db_path
        self.volatility_threshold = 0.05  # 5% daily moves normal in crypto
        self.horizon_hours = 24  # 24h forecasts
        self.step_minutes = 30  # 30-minute intervals
        self.coingecko = CoinGeckoProvider()

        # Initialize database tables
        self._init_tables()

    def _init_tables(self):
        """Initialize crypto prediction tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Crypto predictions
        c.execute("""
            CREATE TABLE IF NOT EXISTS crypto_predictions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                run_at REAL NOT NULL,
                horizon_h INTEGER NOT NULL,
                method TEXT,
                confidence REAL,
                direction TEXT,
                volatility REAL,
                market_cap REAL,
                volume_24h REAL,
                created_at REAL NOT NULL
            )
        """)

        # Forecast points
        c.execute("""
            CREATE TABLE IF NOT EXISTS crypto_forecast_points (
                prediction_id TEXT NOT NULL,
                ts REAL NOT NULL,
                price REAL NOT NULL,
                price_low REAL,
                price_high REAL,
                confidence REAL,
                FOREIGN KEY (prediction_id) REFERENCES crypto_predictions(id)
            )
        """)

        # Actual points (for accuracy tracking)
        c.execute("""
            CREATE TABLE IF NOT EXISTS crypto_actual_points (
                prediction_id TEXT NOT NULL,
                ts REAL NOT NULL,
                price REAL NOT NULL,
                provider TEXT,
                FOREIGN KEY (prediction_id) REFERENCES crypto_predictions(id)
            )
        """)

        # AI trading decisions
        c.execute("""
            CREATE TABLE IF NOT EXISTS crypto_decisions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                target_price REAL,
                stop_loss REAL,
                prediction_id TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES crypto_predictions(id)
            )
        """)

        conn.commit()
        conn.close()

        LOGGER.info("Crypto prediction tables initialized")

    async def generate_prediction(self, symbol: str) -> dict[str, Any]:
        """
        Generate 24h crypto prediction

        Returns:
            {
                'prediction_id': 'uuid',
                'symbol': 'BTC',
                'current_price': 43251.50,
                'direction': 'UP',
                'confidence': 0.75,
                'horizon_hours': 24,
                'volatility': 0.035,
                'timestamp': 1728741600
            }
        """
        symbol = symbol.upper()

        # 1. Fetch current price
        price_data = await get_crypto_price_quorum(symbol, use_cache=False)
        if not price_data:
            raise ValueError(f"Unable to fetch price for {symbol}")

        current_price = price_data["price"]
        run_at = time.time()

        LOGGER.info(f"Generating crypto prediction for {symbol} @ ${current_price:.2f}")

        # 2. Get historical data for pattern analysis
        history = self.coingecko.get_historical(symbol, days=7)
        if not history:
            LOGGER.warning(f"No historical data for {symbol}, using simplified forecast")
            history = [{"timestamp": run_at, "price": current_price}]

        # 3. Calculate crypto metrics
        metrics = self._calculate_metrics(history, price_data)

        # 4. Generate forecast grid
        forecast_points = self._generate_forecast_grid(
            current_price=current_price, metrics=metrics, run_at=run_at
        )

        # 5. Determine direction and confidence
        direction, confidence = self._analyze_direction(metrics, history)

        # 6. Store prediction
        prediction_id = str(uuid.uuid4())
        self._store_prediction(
            prediction_id=prediction_id,
            symbol=symbol,
            run_at=run_at,
            forecast_points=forecast_points,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
        )

        LOGGER.info(
            f"Crypto prediction generated: {symbol} {direction} "
            f"(confidence: {confidence:.0%}, volatility: {metrics['volatility']:.1%})"
        )

        return {
            "prediction_id": prediction_id,
            "symbol": symbol,
            "current_price": current_price,
            "direction": direction,
            "confidence": confidence,
            "horizon_hours": self.horizon_hours,
            "volatility": metrics["volatility"],
            "timestamp": run_at,
        }

    def _calculate_metrics(self, history: list[dict], price_data: dict) -> dict[str, Any]:
        """
        Calculate crypto-specific metrics

        Returns:
            {
                'volatility': 0.035,
                'momentum': 0.02,
                'volume_trend': 1.5,
                'rsi': 65.2
            }
        """
        if len(history) < 2:
            return {
                "volatility": 0.03,  # Default moderate volatility
                "momentum": 0.0,
                "volume_trend": 1.0,
                "rsi": 50.0,
            }

        prices = [h["price"] for h in history]

        # Calculate volatility (standard deviation of returns)
        returns = np.diff(prices) / prices[:-1]
        volatility = float(np.std(returns)) if len(returns) > 0 else 0.03

        # Calculate momentum (recent trend)
        if len(prices) >= 10:
            recent = prices[-10:]
            momentum = (recent[-1] - recent[0]) / recent[0]
        else:
            momentum = (prices[-1] - prices[0]) / prices[0]

        # Volume trend (from price_data if available)
        volume_24h = price_data.get("volume_24h", 0)
        volume_trend = 1.0  # Default neutral

        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(prices) if len(prices) >= 14 else 50.0

        return {
            "volatility": volatility,
            "momentum": momentum,
            "volume_trend": volume_trend,
            "rsi": rsi,
            "market_cap": price_data.get("market_cap", 0),
            "volume_24h": volume_24h,
        }

    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _generate_forecast_grid(
        self, current_price: float, metrics: dict, run_at: float
    ) -> list[dict[str, Any]]:
        """
        Generate forecast points with confidence bands

        48 points @ 30min intervals = 24h forecast
        """
        volatility = metrics["volatility"]
        momentum = metrics.get("momentum", 0)

        points = []
        num_steps = (self.horizon_hours * 60) // self.step_minutes

        for i in range(num_steps + 1):
            t = run_at + (i * self.step_minutes * 60)
            hours_ahead = (i * self.step_minutes) / 60

            # Base forecast with momentum drift
            # Crypto: momentum decay faster than stocks
            momentum_factor = momentum * (1 - hours_ahead / 48)  # Decay over 2x horizon
            price = current_price * (1 + momentum_factor * hours_ahead / 24)

            # Confidence bands (wider than stocks due to higher volatility)
            # Crypto bands: ±5% per day vs stocks ±2% per day
            time_factor = np.sqrt(hours_ahead / 24)
            band_width = volatility * time_factor * current_price * 2.5  # 2.5x multiplier

            confidence = max(0.5, 0.9 - (hours_ahead / 24) * 0.3)

            points.append(
                {
                    "t": t,
                    "p": round(price, 2),
                    "p_low": round(price - band_width, 2),
                    "p_high": round(price + band_width, 2),
                    "confidence": confidence,
                }
            )

        return points

    def _analyze_direction(self, metrics: dict, history: list[dict]) -> tuple:
        """
        Determine direction and confidence

        Returns: (direction, confidence)
        """
        momentum = metrics.get("momentum", 0)
        rsi = metrics.get("rsi", 50)
        volatility = metrics.get("volatility", 0.03)

        # Base confidence
        confidence = 0.7

        # Strong momentum signals
        if abs(momentum) > 0.05:  # >5% recent move
            direction = "UP" if momentum > 0 else "DOWN"
            confidence += 0.1
        elif abs(momentum) > 0.02:  # >2% move
            direction = "UP" if momentum > 0 else "DOWN"
        else:
            direction = "FLAT"
            confidence = 0.6

        # RSI adjustments
        if rsi > 70:  # Overbought
            if direction == "UP":
                confidence -= 0.1
        elif rsi < 30:  # Oversold
            if direction == "DOWN":
                confidence -= 0.1

        # Volatility penalty (high volatility = lower confidence)
        if volatility > 0.05:
            confidence -= 0.05

        # Clamp confidence
        confidence = max(0.5, min(0.95, confidence))

        return direction, confidence

    def _store_prediction(
        self,
        prediction_id: str,
        symbol: str,
        run_at: float,
        forecast_points: list[dict],
        direction: str,
        confidence: float,
        metrics: dict,
    ):
        """Store prediction and forecast points in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Store prediction metadata
        c.execute(
            """
            INSERT INTO crypto_predictions
            (id, symbol, run_at, horizon_h, method, confidence, direction,
             volatility, market_cap, volume_24h, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                prediction_id,
                symbol,
                run_at,
                self.horizon_hours,
                "ghost-crypto-v1",
                confidence,
                direction,
                metrics.get("volatility", 0),
                metrics.get("market_cap", 0),
                metrics.get("volume_24h", 0),
                time.time(),
            ),
        )

        # Store forecast points
        for point in forecast_points:
            c.execute(
                """
                INSERT INTO crypto_forecast_points
                (prediction_id, ts, price, price_low, price_high, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    prediction_id,
                    point["t"],
                    point["p"],
                    point.get("p_low"),
                    point.get("p_high"),
                    point.get("confidence", 0.8),
                ),
            )

        conn.commit()
        conn.close()

    def get_latest_prediction(self, symbol: str) -> dict[str, Any] | None:
        """Get latest prediction for symbol"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(
            """
            SELECT id, symbol, run_at, horizon_h, confidence, direction, volatility
            FROM crypto_predictions
            WHERE symbol = ?
            ORDER BY run_at DESC
            LIMIT 1
        """,
            (symbol.upper(),),
        )

        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row[0],
            "symbol": row[1],
            "timestamp": row[2],
            "horizon_hours": row[3],
            "confidence": row[4],
            "direction": row[5],
            "volatility": row[6],
        }
