"""
Crypto Prediction Engine
24/7 operation with crypto-specific patterns
"""

import logging
import os
import sqlite3
import time
import uuid
from typing import Any

import numpy as np

from .crypto_providers import CoinGeckoProvider, get_crypto_price_quorum

LOGGER = logging.getLogger(__name__)

# Configuration from environment
CRYPTO_FORECAST_H = int(os.getenv("CRYPTO_FORECAST_H", "48"))
CRYPTO_LOOKBACK_H = int(os.getenv("CRYPTO_LOOKBACK_H", "96"))
CRYPTO_CACHE_TTL_S = int(os.getenv("CRYPTO_CACHE_TTL_S", "30"))


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
        self.horizon_hours = CRYPTO_FORECAST_H  # From env (default 48h)
        self.lookback_hours = CRYPTO_LOOKBACK_H  # From env (default 96h)
        self.cache_ttl = CRYPTO_CACHE_TTL_S  # From env (default 30s)
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
        
        # ============================================
        # FEEDBACK LOOP - APPLY LEARNED ADJUSTMENTS
        # ============================================
        # This is the KEY missing piece - we learn from past mistakes!
        try:
            from core.feedback_loop import get_feedback_loop
            feedback = get_feedback_loop()
            
            # Get past performance for this symbol
            symbol_perf = feedback.signal_performance.get(f"SYMBOL_{symbol}", {})
            symbol_total = symbol_perf.get("total", 0)
            symbol_correct = symbol_perf.get("correct", 0)
            
            if symbol_total >= 5:  # Need minimum sample
                symbol_accuracy = symbol_correct / symbol_total
                LOGGER.info(
                    f"📊 {symbol} historical accuracy: {symbol_accuracy:.0%} ({symbol_correct}/{symbol_total})"
                )
                
                # Store for direction analysis
                metrics["_symbol_accuracy"] = symbol_accuracy
                metrics["_symbol_total"] = symbol_total
                
                # If we've been consistently wrong on this symbol, reduce confidence
                if symbol_accuracy < 0.35:
                    LOGGER.warning(f"⚠️ {symbol} accuracy very low ({symbol_accuracy:.0%}) - will reduce confidence")
                    metrics["_accuracy_penalty"] = 0.15
                elif symbol_accuracy < 0.45:
                    metrics["_accuracy_penalty"] = 0.08
                else:
                    metrics["_accuracy_penalty"] = 0
            
            # Also check direction-specific accuracy
            for dir_name in ["UP", "DOWN"]:
                dir_perf = feedback.signal_performance.get(f"DIR_{dir_name}", {})
                dir_total = dir_perf.get("total", 0)
                dir_correct = dir_perf.get("correct", 0)
                if dir_total >= 10:
                    dir_accuracy = dir_correct / dir_total
                    metrics[f"_dir_{dir_name}_accuracy"] = dir_accuracy
                    LOGGER.debug(f"Direction {dir_name} accuracy: {dir_accuracy:.0%} ({dir_correct}/{dir_total})")
            
            # Apply feature weight adjustments from feedback loop
            adjusted_metrics = feedback.get_adjusted_features(metrics)
            if adjusted_metrics != metrics:
                LOGGER.info(f"🔄 Applied feedback loop feature adjustments for {symbol}")
                metrics = adjusted_metrics
                
        except Exception as e:
            LOGGER.debug(f"Feedback loop not available: {e}")

        # 4. Generate forecast grid
        forecast_points = self._generate_forecast_grid(
            current_price=current_price, metrics=metrics, run_at=run_at
        )

        # 5. Determine direction and confidence + collect signals used
        direction, confidence, signals_used = self._analyze_direction_with_signals(metrics, history)

        # 6. Store prediction to SQLite (local crypto db)
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

        # 7. ALSO store to PostgreSQL so outcome reconciler can find it!
        # This is the CRITICAL piece - without this, predictions can't be evaluated
        try:
            from core.prediction_store import get_prediction_store
            store = get_prediction_store()
            
            # Convert forecast points to (ts, price) tuples for predictor module
            forecast_tuples = [(p["t"], p["p"]) for p in forecast_points]
            
            # Add current_price AND signals to features so FeedbackLoop can learn!
            # THIS IS THE KEY FIX - signals_used tells us WHAT caused this prediction
            features_with_price = {
                **metrics,
                "current_price": current_price,
                "PRICE": current_price,  # Backup field name
                "signals_used": signals_used,  # 🔥 NEW: Store what signals fired!
            }
            
            pg_prediction_id = store.save_prediction(
                symbol=symbol,
                forecast_points=forecast_tuples,
                method="ghost-crypto-v1",
                confidence=confidence,
                direction=direction,
                features=features_with_price,
                params={"horizon_h": self.horizon_hours, "crypto": True},
                tag="crypto",
            )
            
            LOGGER.info(f"✅ Crypto prediction also saved to PostgreSQL (ID: {pg_prediction_id})")
        except Exception as e:
            LOGGER.warning(f"⚠️ Failed to dual-write crypto prediction to PostgreSQL: {e}")
            # Continue - SQLite write succeeded, this is a secondary write

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
        """Legacy wrapper - calls new method without signals."""
        direction, confidence, _ = self._analyze_direction_with_signals(metrics, history)
        return direction, confidence

    def _analyze_direction_with_signals(self, metrics: dict, history: list[dict]) -> tuple:
        """
        Determine direction and confidence using trend-following + sentiment.
        NOW ALSO RETURNS signals_used so FeedbackLoop can learn from mistakes!

        KEY CHANGES (Dec 2025):
        1. Uses trend_strength to detect strong trends (MAs aligned)
        2. Integrates news sentiment when available
        3. Uses momentum CONTINUATION for strong trends (not mean-reversion)
        4. RSI in trends = strength confirmation, not reversal signal

        Returns: (direction, confidence, signals_used)
        """
        # 🔥 NEW: Track which signals influenced this prediction
        signals_used = []
        
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
            signals_used.append("TREND_MODE")
            
            if trend_strength > 0 or momentum_14d > 0.05:
                direction = "UP"
                confidence = 0.72
                signals_used.append("TREND_UP")
                
                # Strong trend confirmation
                if trend_strength > 0 and momentum_7d > 0 and macd_histogram > 0:
                    confidence += 0.08  # Triple confirmation
                    signals_used.append("TRIPLE_CONFIRM_UP")
                elif trend_strength > 0 and momentum_7d > 0:
                    confidence += 0.05
                    signals_used.append("DOUBLE_CONFIRM_UP")
                
                # RSI > 60 in uptrend = STRENGTH, not weakness
                if rsi > 60:
                    confidence += 0.03
                    signals_used.append("RSI_STRENGTH_UP")
                    
            elif trend_strength < 0 or momentum_14d < -0.05:
                direction = "DOWN"
                confidence = 0.72
                signals_used.append("TREND_DOWN")
                
                # Strong downtrend confirmation
                if trend_strength < 0 and momentum_7d < 0 and macd_histogram < 0:
                    confidence += 0.08
                    signals_used.append("TRIPLE_CONFIRM_DOWN")
                elif trend_strength < 0 and momentum_7d < 0:
                    confidence += 0.05
                    signals_used.append("DOUBLE_CONFIRM_DOWN")
                
                # RSI < 40 in downtrend = weakness, not oversold bounce
                if rsi < 40:
                    confidence += 0.03
                    signals_used.append("RSI_WEAKNESS_DOWN")
        else:
            # RANGE MODE: Mean-reversion logic applies
            LOGGER.debug(f"RANGE MODE: trend_strength={trend_strength}, using mean-reversion")
            signals_used.append("RANGE_MODE")
            
            if abs(momentum_7d) > 0.03:
                direction = "UP" if momentum_7d > 0 else "DOWN"
                confidence = 0.62
                signals_used.append(f"MOMENTUM_{direction}")
            else:
                direction = "FLAT"
                confidence = 0.55
                signals_used.append("NO_MOMENTUM")
            
            # In range, RSI extremes DO signal reversals
            if rsi > 75:
                signals_used.append("RSI_OVERBOUGHT")
                if direction == "UP":
                    confidence -= 0.08  # Likely to reverse
            elif rsi < 25:
                signals_used.append("RSI_OVERSOLD")
                if direction == "DOWN":
                    confidence -= 0.08
        
        # ========================================
        # STEP 2: EMA Cross signals (strong)
        # ========================================
        if golden_cross:
            direction = "UP"
            confidence = max(confidence, 0.78)
            signals_used.append("GOLDEN_CROSS")
            LOGGER.info("Golden cross detected - bullish signal")
        elif death_cross:
            direction = "DOWN"
            confidence = max(confidence, 0.78)
            signals_used.append("DEATH_CROSS")
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
                    signals_used.append(f"NEWS_SENTIMENT_{news_data.get('sentiment_label', 'UNKNOWN').upper()}")
                    
                    # Sentiment alignment with direction = boost confidence
                    if sentiment > 0.3 and direction == "UP":
                        confidence += 0.05
                        signals_used.append("SENTIMENT_ALIGNED")
                    elif sentiment < -0.3 and direction == "DOWN":
                        confidence += 0.05
                        signals_used.append("SENTIMENT_ALIGNED")
                    # Sentiment contradiction = reduce confidence
                    elif sentiment > 0.3 and direction == "DOWN":
                        confidence -= 0.05
                        signals_used.append("SENTIMENT_CONTRADICTION")
                        LOGGER.warning(f"Direction {direction} contradicts positive news sentiment")
                    elif sentiment < -0.3 and direction == "UP":
                        confidence -= 0.05
                        signals_used.append("SENTIMENT_CONTRADICTION")
                        LOGGER.warning(f"Direction {direction} contradicts negative news sentiment")
                    
                    # Very strong sentiment can flip direction
                    if abs(sentiment) > 0.6:
                        if sentiment > 0.6 and direction != "UP":
                            LOGGER.info(f"Very positive sentiment overriding {direction} -> UP")
                            direction = "UP"
                            confidence = 0.70
                            signals_used.append("SENTIMENT_OVERRIDE_UP")
                        elif sentiment < -0.6 and direction != "DOWN":
                            LOGGER.info(f"Very negative sentiment overriding {direction} -> DOWN")
                            direction = "DOWN"
                            confidence = 0.70
                            signals_used.append("SENTIMENT_OVERRIDE_DOWN")
        except ImportError:
            LOGGER.debug("News sentiment module not available")
        except Exception as e:
            LOGGER.debug(f"News sentiment check failed: {e}")
        
        # ========================================
        # STEP 4: Volatility adjustments
        # ========================================
        if volatility > 0.08:  # Very high volatility
            confidence -= 0.08
            signals_used.append("HIGH_VOLATILITY")
        elif volatility > 0.05:
            confidence -= 0.04
            signals_used.append("MODERATE_VOLATILITY")
        
        # ========================================
        # STEP 5: FEEDBACK LOOP - Learn from mistakes!
        # ========================================
        # Apply accuracy penalty if we've been wrong on this symbol
        accuracy_penalty = metrics.get("_accuracy_penalty", 0)
        if accuracy_penalty > 0:
            confidence -= accuracy_penalty
            signals_used.append("ACCURACY_PENALTY")
            LOGGER.warning(f"⚠️ Applied accuracy penalty: -{accuracy_penalty:.0%} (low historical accuracy)")
        
        # Check direction-specific accuracy
        dir_accuracy = metrics.get(f"_dir_{direction}_accuracy")
        if dir_accuracy is not None and dir_accuracy < 0.40:
            # We've been wrong predicting this direction - reduce confidence
            dir_penalty = 0.10
            confidence -= dir_penalty
            signals_used.append("DIRECTION_ACCURACY_PENALTY")
            LOGGER.warning(
                f"⚠️ Direction {direction} has poor accuracy ({dir_accuracy:.0%}) - applying -{dir_penalty:.0%} penalty"
            )
        
        # ========================================
        # STEP 6: APPLY LEARNED SIGNAL ADJUSTMENTS
        # ========================================
        # This is the KEY learning mechanism - adjust confidence based on
        # how well each signal has performed historically
        try:
            from core.feedback_loop import get_feedback_loop
            feedback = get_feedback_loop()
            signal_adjustment = feedback.get_signals_confidence_adjustment(signals_used)
            if abs(signal_adjustment) > 0.01:
                confidence += signal_adjustment
                signals_used.append(f"LEARNED_ADJUSTMENT_{'+' if signal_adjustment > 0 else ''}{int(signal_adjustment*100)}")
                LOGGER.info(f"🧠 Applied learned signal adjustment: {signal_adjustment:+.0%}")
        except Exception as e:
            LOGGER.debug(f"Could not apply signal adjustments: {e}")
        
        # ========================================
        # STEP 7: Clamp and return WITH SIGNALS
        # ========================================
        confidence = max(0.50, min(0.92, confidence))
        
        LOGGER.info(
            f"Direction analysis: {direction} @ {confidence:.0%} "
            f"[trend={trend_strength:.1f}, mom7={momentum_7d:.2%}, rsi={rsi:.0f}] "
            f"signals={signals_used}"
        )
        
        return direction, confidence, signals_used

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
