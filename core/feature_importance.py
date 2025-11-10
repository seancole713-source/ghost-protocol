"""
APEX Feature Importance Map
Shapley value analysis for forecast explainability

Shows which features (RSI, MA, volume, sentiment, etc.)
contributed most to each prediction.

Expected Impact: +10% interpretability
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Single feature's contribution to a forecast"""

    name: str
    value: float  # Current feature value
    shapley_value: float  # Contribution to prediction (-1 to +1)
    importance: float  # Absolute importance (0-100)
    direction: str  # "bullish", "bearish", or "neutral"


@dataclass
class ImportanceAnalysis:
    """Complete feature importance breakdown"""

    symbol: str
    timestamp: int
    forecast_type: str  # "nowcast", "swing", "position"
    predicted_return: float

    # Top features by importance
    features: list[FeatureContribution]

    # Summary stats
    total_bullish_contribution: float
    total_bearish_contribution: float
    confidence_score: float  # How clear is the signal?


class FeatureImportanceAnalyzer:
    """
    Calculate Shapley values for forecast features

    Shapley values measure each feature's marginal contribution
    to the final prediction by averaging over all possible
    feature coalitions.
    """

    def __init__(self):
        self.feature_names = [
            "rsi_14",
            "ma_5_20_cross",
            "ma_20_50_cross",
            "momentum_5d",
            "momentum_20d",
            "volume_surge",
            "volatility",
            "sentiment_score",
            "price_vs_ma20",
            "trend_consistency",
        ]

    def analyze_forecast(self, symbol: str, forecast_type: str = "swing") -> ImportanceAnalysis:
        """
        Calculate feature importance for a forecast

        Args:
            symbol: Trading symbol (e.g., "WOLF")
            forecast_type: "nowcast", "swing", or "position"

        Returns:
            ImportanceAnalysis with Shapley values
        """

        LOGGER.info(f"Calculating feature importance for {symbol} ({forecast_type})")

        # 1. Extract current features
        features = self._extract_features(symbol)

        # 2. Get baseline prediction (no features)
        baseline_prediction = 0.0  # Neutral baseline

        # 3. Get full prediction (all features)
        full_prediction = self._make_prediction(features, forecast_type)

        # 4. Calculate Shapley values (simplified Monte Carlo approximation)
        shapley_values = self._calculate_shapley_values(
            features, forecast_type, baseline_prediction, full_prediction
        )

        # 5. Build FeatureContribution objects
        contributions = []
        for feature_name, shapley_val in shapley_values.items():
            feature_value = features.get(feature_name, 0.0)

            # Normalize importance to 0-100
            importance = abs(shapley_val) * 100

            # Determine direction
            if shapley_val > 0.01:
                direction = "bullish"
            elif shapley_val < -0.01:
                direction = "bearish"
            else:
                direction = "neutral"

            contributions.append(
                FeatureContribution(
                    name=feature_name,
                    value=feature_value,
                    shapley_value=shapley_val,
                    importance=importance,
                    direction=direction,
                )
            )

        # Sort by absolute importance
        contributions.sort(key=lambda x: x.importance, reverse=True)

        # 6. Calculate summary stats
        bullish_sum = sum(c.shapley_value for c in contributions if c.shapley_value > 0)
        bearish_sum = sum(c.shapley_value for c in contributions if c.shapley_value < 0)

        # Confidence: how aligned are the features?
        total_impact = abs(bullish_sum) + abs(bearish_sum)
        net_impact = bullish_sum + bearish_sum
        confidence = abs(net_impact / total_impact) * 100 if total_impact > 0 else 50.0

        return ImportanceAnalysis(
            symbol=symbol,
            timestamp=int(time.time()),
            forecast_type=forecast_type,
            predicted_return=full_prediction,
            features=contributions,
            total_bullish_contribution=bullish_sum,
            total_bearish_contribution=bearish_sum,
            confidence_score=confidence,
        )

    def _extract_features(self, symbol: str) -> dict[str, float]:
        """Extract current feature values from market data"""

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="90d")

            if hist.empty:
                LOGGER.warning(f"No data for {symbol}, using defaults")
                return {name: 0.0 for name in self.feature_names}

            features = {}

            # 1. RSI (14-period)
            if len(hist) >= 14:
                returns = pd.to_numeric(hist["Close"], errors="coerce").diff()
                gains = returns[returns > 0].tail(14).mean()  # type: ignore
                losses = -returns[returns < 0].tail(14).mean()  # type: ignore
                rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 50
                features["rsi_14"] = (rsi - 50) / 50  # Normalize to -1 to +1
            else:
                features["rsi_14"] = 0.0

            # 2. MA crossovers
            if len(hist) >= 50:
                ma5 = hist["Close"].tail(50).rolling(5).mean().iloc[-1]
                ma20 = hist["Close"].tail(50).rolling(20).mean().iloc[-1]
                ma50 = hist["Close"].tail(50).rolling(50).mean().iloc[-1]

                features["ma_5_20_cross"] = (ma5 - ma20) / ma20 if ma20 > 0 else 0.0
                features["ma_20_50_cross"] = (ma20 - ma50) / ma50 if ma50 > 0 else 0.0
            else:
                features["ma_5_20_cross"] = 0.0
                features["ma_20_50_cross"] = 0.0

            # 3. Momentum
            if len(hist) >= 20:
                current_price = hist["Close"].iloc[-1]
                price_5d_ago = hist["Close"].iloc[-6] if len(hist) >= 6 else current_price
                price_20d_ago = hist["Close"].iloc[-21] if len(hist) >= 21 else current_price

                features["momentum_5d"] = (
                    (current_price - price_5d_ago) / price_5d_ago if price_5d_ago > 0 else 0.0
                )
                features["momentum_20d"] = (
                    (current_price - price_20d_ago) / price_20d_ago if price_20d_ago > 0 else 0.0
                )
            else:
                features["momentum_5d"] = 0.0
                features["momentum_20d"] = 0.0

            # 4. Volume surge
            if len(hist) >= 20:
                recent_vol = hist["Volume"].tail(5).mean()
                avg_vol = hist["Volume"].tail(20).mean()
                features["volume_surge"] = (recent_vol / avg_vol - 1) if avg_vol > 0 else 0.0
            else:
                features["volume_surge"] = 0.0

            # 5. Volatility (20-day annualized)
            if len(hist) >= 20:
                returns_series = hist["Close"].pct_change().tail(20)
                vol = returns_series.std() * (252**0.5)
                features["volatility"] = vol  # Raw volatility
            else:
                features["volatility"] = 0.0

            # 6. Sentiment (from World Feed Fusion)
            try:
                from core.world_feed_fusion import get_feed_fusion

                fusion = get_feed_fusion()
                aggregate = fusion.get_sentiment_aggregate(symbol, "1d")
                if aggregate and aggregate.article_count > 0:
                    features["sentiment_score"] = aggregate.weighted_sentiment
                else:
                    features["sentiment_score"] = 0.0
            except Exception:
                features["sentiment_score"] = 0.0  # Fallback if fusion unavailable

            # 7. Price vs MA20
            if len(hist) >= 20:
                current_price = hist["Close"].iloc[-1]
                ma20_val = hist["Close"].tail(20).mean()
                features["price_vs_ma20"] = (
                    (current_price - ma20_val) / ma20_val if ma20_val > 0 else 0.0
                )
            else:
                features["price_vs_ma20"] = 0.0

            # 8. Trend consistency (up days in last 10)
            if len(hist) >= 10:
                price_diff = pd.to_numeric(hist["Close"], errors="coerce").tail(10).diff()
                up_days = (price_diff[price_diff > 0]).count()  # type: ignore
                features["trend_consistency"] = (up_days / 10.0 - 0.5) * 2  # -1 to +1
            else:
                features["trend_consistency"] = 0.0

            return features

        except Exception as e:
            LOGGER.error(f"Failed to extract features: {e}")
            return {name: 0.0 for name in self.feature_names}

    def _make_prediction(self, features: dict[str, float], forecast_type: str) -> float:
        """
        Make prediction using current feature values

        This is a simplified model that mimics the logic in
        multi_horizon_forecaster.py but with explicit weights
        for Shapley analysis.
        """

        # Feature weights vary by forecast type
        if forecast_type == "nowcast":
            weights = {
                "rsi_14": 0.15,
                "ma_5_20_cross": 0.20,
                "ma_20_50_cross": 0.05,
                "momentum_5d": 0.25,
                "momentum_20d": 0.05,
                "volume_surge": 0.15,
                "volatility": -0.10,  # High vol reduces confidence
                "sentiment_score": 0.10,
                "price_vs_ma20": 0.10,
                "trend_consistency": 0.05,
            }
        elif forecast_type == "position":
            weights = {
                "rsi_14": 0.10,
                "ma_5_20_cross": 0.05,
                "ma_20_50_cross": 0.25,
                "momentum_5d": 0.05,
                "momentum_20d": 0.30,
                "volume_surge": 0.05,
                "volatility": -0.15,
                "sentiment_score": 0.10,
                "price_vs_ma20": 0.15,
                "trend_consistency": 0.15,
            }
        else:  # swing
            weights = {
                "rsi_14": 0.20,
                "ma_5_20_cross": 0.15,
                "ma_20_50_cross": 0.15,
                "momentum_5d": 0.15,
                "momentum_20d": 0.10,
                "volume_surge": 0.10,
                "volatility": -0.10,
                "sentiment_score": 0.10,
                "price_vs_ma20": 0.10,
                "trend_consistency": 0.10,
            }

        # Weighted sum
        prediction = sum(features.get(name, 0.0) * weight for name, weight in weights.items())

        # Clip to reasonable range
        return max(-0.10, min(0.10, prediction))  # -10% to +10%

    def _calculate_shapley_values(
        self,
        features: dict[str, float],
        forecast_type: str,
        baseline: float,
        full_prediction: float,
    ) -> dict[str, float]:
        """
        Calculate Shapley values using Monte Carlo approximation

        True Shapley: average marginal contribution across all
        possible feature coalitions (2^n coalitions).

        Monte Carlo: sample random coalitions and approximate.
        """

        shapley_values = {name: 0.0 for name in self.feature_names}
        n_features = len(self.feature_names)
        n_samples = min(100, 2**n_features)  # Sample up to 100 coalitions

        # For each feature, calculate marginal contributions
        for feature_name in self.feature_names:
            marginal_contributions = []

            # Sample random coalitions
            for _ in range(n_samples):
                # Random subset of other features (coalition size)
                coalition_size = np.random.randint(0, n_features)
                other_features = [f for f in self.feature_names if f != feature_name]
                coalition = list(np.random.choice(other_features, coalition_size, replace=False))

                # Prediction without target feature
                features_without = {k: v for k, v in features.items() if k in coalition}
                pred_without = self._make_prediction(features_without, forecast_type)

                # Prediction with target feature
                features_with = features_without.copy()
                features_with[feature_name] = features[feature_name]
                pred_with = self._make_prediction(features_with, forecast_type)

                # Marginal contribution
                marginal = pred_with - pred_without
                marginal_contributions.append(marginal)

            # Average marginal contribution = Shapley value
            shapley_values[feature_name] = float(np.mean(marginal_contributions))

        return shapley_values

    def get_top_features(
        self, symbol: str, forecast_type: str = "swing", top_n: int = 5
    ) -> list[dict[str, Any]]:
        """
        Get top N most important features (simplified API)

        Returns:
            List of dicts with {name, value, importance, direction}
        """

        analysis = self.analyze_forecast(symbol, forecast_type)

        return [
            {
                "name": f.name,
                "value": round(f.value, 4),
                "importance": round(f.importance, 2),
                "shapley_value": round(f.shapley_value, 4),
                "direction": f.direction,
            }
            for f in analysis.features[:top_n]
        ]


# Singleton instance
_FEATURE_IMPORTANCE_ANALYZER: FeatureImportanceAnalyzer | None = None


def get_feature_importance_analyzer() -> FeatureImportanceAnalyzer:
    """Get singleton instance of feature importance analyzer"""
    global _FEATURE_IMPORTANCE_ANALYZER
    if _FEATURE_IMPORTANCE_ANALYZER is None:
        _FEATURE_IMPORTANCE_ANALYZER = FeatureImportanceAnalyzer()
    return _FEATURE_IMPORTANCE_ANALYZER
