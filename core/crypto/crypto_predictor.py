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
                features_json TEXT,
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
        Calculate crypto-specific metrics + FULL TECHNICAL INDICATORS (50+)

        Returns:
            dict with 50+ technical indicators + legacy metrics
        """
        if len(history) < 2:
            return {
                "volatility": 0.03,  # Default moderate volatility
                "momentum": 0.0,
                "volume_trend": 1.0,
                "rsi": 50.0,
            }

        # Convert history to DataFrame for technical indicators
        import pandas as pd
        from core.features.technical_indicators import calculate_technical_indicators

        df = pd.DataFrame(history)
        
        # Standardize column names
        if 'price' in df.columns:
            df['Close'] = df['price']
        
        # Estimate OHLC from close if not available
        if 'Close' in df.columns and 'Open' not in df.columns:
            df['Open'] = df['Close'].shift(1).fillna(df['Close'])
            df['High'] = df['Close'] * 1.005  # Estimate ±0.5% spread
            df['Low'] = df['Close'] * 0.995
        
        # Volume data (if available)
        volume_24h = price_data.get("volume_24h", 0)
        if volume_24h > 0 and 'Volume' not in df.columns:
            df['Volume'] = volume_24h  # Use same value for all rows (estimate)
        
        # Calculate ALL technical indicators (50+)
        indicators = calculate_technical_indicators(df, price_col='Close', volume_col='Volume' if 'Volume' in df.columns else None)
        
        # Legacy metrics (keep for backward compatibility)
        prices = [h["price"] for h in history]
        returns = np.diff(prices) / prices[:-1]
        volatility = float(np.std(returns)) if len(returns) > 0 else 0.03
        
        if len(prices) >= 10:
            recent = prices[-10:]
            momentum = (recent[-1] - recent[0]) / recent[0]
        else:
            momentum = (prices[-1] - prices[0]) / prices[0]
        
        rsi = self._calculate_rsi(prices) if len(prices) >= 14 else 50.0
        
        # Merge legacy metrics with new indicators
        indicators.update({
            "volatility": volatility,  # Legacy
            "momentum": momentum,  # Legacy
            "volume_trend": 1.0,  # Legacy placeholder
            "rsi": rsi,  # Legacy RSI calculation
            "market_cap": price_data.get("market_cap", 0),
            "volume_24h": volume_24h,
        })
        
        LOGGER.info(f"Calculated {len(indicators)} technical indicators")
        
        return indicators

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
        Determine direction and confidence using trend-following + sentiment.

        KEY CHANGES (Dec 2025):
        1. Uses trend_strength to detect strong trends (MAs aligned)
        2. Integrates news sentiment when available
        3. Uses momentum CONTINUATION for strong trends (not mean-reversion)
        4. RSI in trends = strength confirmation, not reversal signal

        Returns: (direction, confidence)
        """
        # Extract indicators
        momentum_7d = metrics.get("momentum_7d", metrics.get("momentum", 0))
        momentum_14d = metrics.get("momentum_14d", 0)
        rsi = metrics.get("rsi_14", metrics.get("rsi", 50))
        volatility = metrics.get("volatility", 0.03)
        trend_strength = metrics.get("trend_strength", 0)  # -1 (down), 0 (mixed), +1 (up)
        macd_histogram = metrics.get("macd_histogram", 0)
        ema_cross = metrics.get("ema_cross", 0)
        golden_cross = metrics.get("golden_cross", 0)
        death_cross = metrics.get("death_cross", 0)
        
        # Base confidence
        confidence = 0.65
        direction = "FLAT"
        
        # ========================================
        # STEP 1: Detect if we're in a TREND mode
        # ========================================
        is_trending = abs(trend_strength) > 0.5 or abs(momentum_14d) > 0.08
        
        if is_trending:
            # TREND MODE: Follow the trend, momentum continues
            LOGGER.debug(f"TREND MODE: trend_strength={trend_strength}, momentum_14d={momentum_14d}")
            
            if trend_strength > 0 or momentum_14d > 0.05:
                direction = "UP"
                confidence = 0.72
                
                # Strong trend confirmation
                if trend_strength > 0 and momentum_7d > 0 and macd_histogram > 0:
                    confidence += 0.08  # Triple confirmation
                elif trend_strength > 0 and momentum_7d > 0:
                    confidence += 0.05
                
                # RSI > 60 in uptrend = STRENGTH, not weakness
                if rsi > 60:
                    confidence += 0.03
                    
            elif trend_strength < 0 or momentum_14d < -0.05:
                direction = "DOWN"
                confidence = 0.72
                
                # Strong downtrend confirmation
                if trend_strength < 0 and momentum_7d < 0 and macd_histogram < 0:
                    confidence += 0.08
                elif trend_strength < 0 and momentum_7d < 0:
                    confidence += 0.05
                
                # RSI < 40 in downtrend = weakness, not oversold bounce
                if rsi < 40:
                    confidence += 0.03
        else:
            # RANGE MODE: Mean-reversion logic applies
            LOGGER.debug(f"RANGE MODE: trend_strength={trend_strength}, using mean-reversion")
            
            if abs(momentum_7d) > 0.03:
                direction = "UP" if momentum_7d > 0 else "DOWN"
                confidence = 0.62
            else:
                direction = "FLAT"
                confidence = 0.55
            
            # In range, RSI extremes DO signal reversals
            if rsi > 75:
                if direction == "UP":
                    confidence -= 0.08  # Likely to reverse
            elif rsi < 25:
                if direction == "DOWN":
                    confidence -= 0.08
        
        # ========================================
        # STEP 2: EMA Cross signals (strong)
        # ========================================
        if golden_cross:
            direction = "UP"
            confidence = max(confidence, 0.78)
            LOGGER.info("Golden cross detected - bullish signal")
        elif death_cross:
            direction = "DOWN"
            confidence = max(confidence, 0.78)
            LOGGER.info("Death cross detected - bearish signal")
        
        # ========================================
        # STEP 3: News Sentiment Integration
        # ========================================
        try:
            from core.news_sentiment import fetch_news_sentiment
            
            # Determine symbol from history or use BTC default
            symbol = "BTC"  # Will be passed in future refactor
            if history and isinstance(history[0], dict):
                symbol = history[0].get("symbol", "BTC")
            
            news_data = fetch_news_sentiment(symbol, limit=5)
            
            if news_data.get("ok"):
                sentiment = news_data.get("sentiment_score", 0)
                
                if abs(sentiment) > 0.3:  # Strong sentiment
                    LOGGER.info(f"News sentiment for {symbol}: {sentiment:.2f} ({news_data.get('sentiment_label')})")
                    
                    # Sentiment alignment with direction = boost confidence
                    if sentiment > 0.3 and direction == "UP":
                        confidence += 0.05
                    elif sentiment < -0.3 and direction == "DOWN":
                        confidence += 0.05
                    # Sentiment contradiction = reduce confidence
                    elif sentiment > 0.3 and direction == "DOWN":
                        confidence -= 0.05
                        LOGGER.warning(f"Direction {direction} contradicts positive news sentiment")
                    elif sentiment < -0.3 and direction == "UP":
                        confidence -= 0.05
                        LOGGER.warning(f"Direction {direction} contradicts negative news sentiment")
                    
                    # Very strong sentiment can flip direction
                    if abs(sentiment) > 0.6:
                        if sentiment > 0.6 and direction != "UP":
                            LOGGER.info(f"Very positive sentiment overriding {direction} -> UP")
                            direction = "UP"
                            confidence = 0.70
                        elif sentiment < -0.6 and direction != "DOWN":
                            LOGGER.info(f"Very negative sentiment overriding {direction} -> DOWN")
                            direction = "DOWN"
                            confidence = 0.70
        except ImportError:
            LOGGER.debug("News sentiment module not available")
        except Exception as e:
            LOGGER.debug(f"News sentiment check failed: {e}")
        
        # ========================================
        # STEP 4: Volatility adjustments
        # ========================================
        if volatility > 0.08:  # Very high volatility
            confidence -= 0.08
        elif volatility > 0.05:
            confidence -= 0.04
        
        # ========================================
        # STEP 5: Clamp and return
        # ========================================
        confidence = max(0.50, min(0.92, confidence))
        
        LOGGER.info(
            f"Direction analysis: {direction} @ {confidence:.0%} "
            f"[trend={trend_strength:.1f}, mom7={momentum_7d:.2%}, rsi={rsi:.0f}]"
        )
        
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
        import json
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Store prediction metadata with features for ML training
        features_json = json.dumps(metrics)
        
        c.execute(
            """
            INSERT INTO crypto_predictions
            (id, symbol, run_at, horizon_h, method, confidence, direction,
             volatility, market_cap, volume_24h, features_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                features_json,
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
