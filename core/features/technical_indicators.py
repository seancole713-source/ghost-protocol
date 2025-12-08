"""
Ghost Technical Indicators Library
==================================

Comprehensive technical analysis indicators for prediction models.
Optimized for crypto/stock data with varying levels of completeness.

Categories:
- Volume: OBV, volume ratios, volume momentum
- Momentum: MACD, RSI, Stochastic, Williams %R, ROC
- Volatility: ATR estimates, volatility metrics
- Trend: EMAs, MAs, trend strength
- Pattern: Price patterns, momentum patterns

Dependencies:
- numpy (existing)
- pandas (existing)

Author: Ghost AI (Cycle #3)
Date: December 7, 2025
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from price/volume history"""

    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        price_col: str = "Close",
        volume_col: str | None = "Volume"
    ) -> dict[str, float]:
        """
        Calculate all indicators from price DataFrame.

        Args:
            df: DataFrame with price data (must have price_col)
            price_col: Column name for prices
            volume_col: Column name for volume (optional)

        Returns:
            Dict with indicator values
        """
        if len(df) < 2:
            return {}

        indicators = {}

        # Core indicators (always calculated)
        indicators.update(
            TechnicalIndicators._momentum_indicators(df, price_col)
        )
        indicators.update(
            TechnicalIndicators._volatility_indicators(df, price_col)
        )
        indicators.update(
            TechnicalIndicators._trend_indicators(df, price_col)
        )
        indicators.update(
            TechnicalIndicators._pattern_indicators(df, price_col)
        )

        # Volume indicators (if volume available)
        if volume_col and volume_col in df.columns:
            indicators.update(
                TechnicalIndicators._volume_indicators(df, price_col, volume_col)
            )

        return indicators

    @staticmethod
    def _momentum_indicators(df: pd.DataFrame, price_col: str) -> dict[str, float]:
        """Calculate momentum indicators"""
        prices = df[price_col].values
        indicators = {}

        # RSI (14-period)
        if len(prices) >= 15:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = -np.where(deltas < 0, deltas, 0)

            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0 if avg_gain > 0 else 50.0

            indicators["rsi_14"] = float(rsi)

            # RSI overbought/oversold signals
            indicators["rsi_overbought"] = 1.0 if rsi > 70 else 0.0
            indicators["rsi_oversold"] = 1.0 if rsi < 30 else 0.0

        # MACD (12, 26, 9)
        if len(prices) >= 26:
            ema12 = TechnicalIndicators._ema(prices, 12)
            ema26 = TechnicalIndicators._ema(prices, 26)
            macd_line = ema12 - ema26

            # Signal line (9-period EMA of MACD)
            signal_line = TechnicalIndicators._ema(macd_line, 9)
            histogram = macd_line - signal_line

            indicators["macd"] = float(macd_line[-1])
            indicators["macd_signal"] = float(signal_line[-1])
            indicators["macd_histogram"] = float(histogram[-1])

            # MACD crossover signals
            if len(histogram) >= 2:
                indicators["macd_bullish_cross"] = (
                    1.0 if histogram[-2] < 0 and histogram[-1] > 0 else 0.0
                )
                indicators["macd_bearish_cross"] = (
                    1.0 if histogram[-2] > 0 and histogram[-1] < 0 else 0.0
                )

        # Stochastic Oscillator (14, 3)
        if len(df) >= 14 and "High" in df.columns and "Low" in df.columns:
            high = df["High"].values
            low = df["Low"].values

            lowest_low = np.min(low[-14:])
            highest_high = np.max(high[-14:])

            if highest_high != lowest_low:
                stoch_k = 100 * (prices[-1] - lowest_low) / (highest_high - lowest_low)
            else:
                stoch_k = 50.0

            indicators["stochastic_k"] = float(stoch_k)
            indicators["stoch_overbought"] = 1.0 if stoch_k > 80 else 0.0
            indicators["stoch_oversold"] = 1.0 if stoch_k < 20 else 0.0

        # Williams %R (14)
        if len(df) >= 14 and "High" in df.columns and "Low" in df.columns:
            high = df["High"].values
            low = df["Low"].values

            highest_high = np.max(high[-14:])
            lowest_low = np.min(low[-14:])

            if highest_high != lowest_low:
                williams_r = -100 * (highest_high - prices[-1]) / (
                    highest_high - lowest_low
                )
            else:
                williams_r = -50.0

            indicators["williams_r"] = float(williams_r)

        # Rate of Change (10-period)
        if len(prices) >= 10:
            roc = 100 * (prices[-1] - prices[-10]) / prices[-10]
            indicators["roc_10"] = float(roc)

        # Multi-period momentum
        for period in [7, 14, 30]:
            if len(prices) >= period:
                momentum = (prices[-1] - prices[-period]) / prices[-period]
                indicators[f"momentum_{period}d"] = float(momentum)

        return indicators

    @staticmethod
    def _volatility_indicators(df: pd.DataFrame, price_col: str) -> dict[str, float]:
        """Calculate volatility indicators"""
        prices = df[price_col].values
        indicators = {}

        # Calculate returns for volatility metrics
        if len(prices) >= 2:
            returns = np.diff(prices) / prices[:-1]

            # Multi-period volatility
            for period in [7, 14, 30]:
                if len(returns) >= period:
                    vol = np.std(returns[-period:])
                    indicators[f"volatility_{period}d"] = float(vol)

            # Volatility ratio (short-term vs long-term)
            if len(returns) >= 30:
                vol_7d = np.std(returns[-7:])
                vol_30d = np.std(returns[-30:])
                if vol_30d > 0:
                    indicators["volatility_ratio"] = float(vol_7d / vol_30d)

        # ATR (Average True Range) - requires High/Low
        if (
            len(df) >= 14
            and "High" in df.columns
            and "Low" in df.columns
        ):
            high = df["High"].values
            low = df["Low"].values

            # True Range
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - prices[:-1]),
                    np.abs(low[1:] - prices[:-1]),
                ),
            )

            atr = np.mean(tr[-14:])
            indicators["atr"] = float(atr)
            indicators["atr_pct"] = float(atr / prices[-1] * 100)

        # Bollinger Bands (20-period)
        if len(prices) >= 20:
            ma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])

            bb_upper = ma20 + 2 * std20
            bb_lower = ma20 - 2 * std20

            indicators["bb_mid"] = float(ma20)
            indicators["bb_upper"] = float(bb_upper)
            indicators["bb_lower"] = float(bb_lower)
            indicators["bb_width"] = float((bb_upper - bb_lower) / ma20)

            # Bollinger Band position (0=lower, 1=upper)
            if bb_upper != bb_lower:
                bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower)
                indicators["bb_position"] = float(bb_position)

        return indicators

    @staticmethod
    def _trend_indicators(df: pd.DataFrame, price_col: str) -> dict[str, float]:
        """Calculate trend indicators"""
        prices = df[price_col].values
        indicators = {}

        # Moving Averages
        for period in [10, 20, 50, 200]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                indicators[f"ma_{period}"] = float(ma)

                # Price vs MA ratio
                price_ma_ratio = (prices[-1] - ma) / ma
                indicators[f"price_vs_ma_{period}"] = float(price_ma_ratio)

        # Exponential Moving Averages
        for period in [12, 26]:
            if len(prices) >= period:
                ema = TechnicalIndicators._ema(prices, period)
                indicators[f"ema_{period}"] = float(ema[-1])

        # EMA crossovers
        if len(prices) >= 26:
            ema12 = TechnicalIndicators._ema(prices, 12)
            ema26 = TechnicalIndicators._ema(prices, 26)

            cross = (ema12[-1] - ema26[-1]) / ema26[-1]
            indicators["ema_cross"] = float(cross)

            # Golden/Death cross signals
            if len(ema12) >= 2:
                indicators["golden_cross"] = (
                    1.0 if ema12[-2] <= ema26[-2] and ema12[-1] > ema26[-1] else 0.0
                )
                indicators["death_cross"] = (
                    1.0 if ema12[-2] >= ema26[-2] and ema12[-1] < ema26[-1] else 0.0
                )

        # Trend strength (are MAs aligned?)
        if len(prices) >= 50:
            ma10 = np.mean(prices[-10:])
            ma20 = np.mean(prices[-20:])
            ma50 = np.mean(prices[-50:])

            # Strong uptrend: ma10 > ma20 > ma50
            # Strong downtrend: ma10 < ma20 < ma50
            if ma10 > ma20 > ma50:
                trend_strength = 1.0
            elif ma10 < ma20 < ma50:
                trend_strength = -1.0
            else:
                trend_strength = 0.0

            indicators["trend_strength"] = float(trend_strength)

        return indicators

    @staticmethod
    def _pattern_indicators(df: pd.DataFrame, price_col: str) -> dict[str, float]:
        """Calculate pattern-based indicators"""
        prices = df[price_col].values
        indicators = {}

        if len(prices) < 3:
            return indicators

        # Recent price action
        indicators["price_change_1d"] = float(
            (prices[-1] - prices[-2]) / prices[-2]
        )

        if len(prices) >= 4:
            indicators["price_change_3d"] = float(
                (prices[-1] - prices[-4]) / prices[-4]
            )

        # Higher highs / Lower lows pattern
        if len(prices) >= 5:
            recent_prices = prices[-5:]

            # Count higher highs (bullish)
            higher_highs = sum(
                1 for i in range(1, len(recent_prices))
                if recent_prices[i] > max(recent_prices[:i])
            )
            indicators["higher_highs_5d"] = float(higher_highs)

            # Count lower lows (bearish)
            lower_lows = sum(
                1 for i in range(1, len(recent_prices))
                if recent_prices[i] < min(recent_prices[:i])
            )
            indicators["lower_lows_5d"] = float(lower_lows)

        # Distance from highs/lows
        if len(prices) >= 20:
            high_20 = np.max(prices[-20:])
            low_20 = np.min(prices[-20:])

            dist_from_high = (prices[-1] - high_20) / high_20
            dist_from_low = (prices[-1] - low_20) / low_20

            indicators["dist_from_high_20d"] = float(dist_from_high)
            indicators["dist_from_low_20d"] = float(dist_from_low)

        return indicators

    @staticmethod
    def _volume_indicators(
        df: pd.DataFrame, price_col: str, volume_col: str
    ) -> dict[str, float]:
        """Calculate volume indicators"""
        prices = df[price_col].values
        volumes = df[volume_col].values
        indicators = {}

        if len(volumes) < 2:
            return indicators

        # Volume moving average
        if len(volumes) >= 20:
            vol_ma = np.mean(volumes[-20:])
            if vol_ma > 0:
                indicators["volume_ratio"] = float(volumes[-1] / vol_ma)

        # Volume change
        if len(volumes) >= 2:
            vol_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0
            indicators["volume_change"] = float(vol_change)

        # On-Balance Volume (OBV)
        obv = np.zeros(len(prices))
        obv[0] = volumes[0]

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif prices[i] < prices[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]

        indicators["obv"] = float(obv[-1])

        # OBV trend (5-period slope)
        if len(obv) >= 5:
            obv_slope = (obv[-1] - obv[-5]) / 5
            indicators["obv_trend"] = float(obv_slope)

        # VWAP (Volume Weighted Average Price) - recent 20 periods
        if len(prices) >= 20 and "High" in df.columns and "Low" in df.columns:
            high = df["High"].values
            low = df["Low"].values
            typical_price = (high + low + prices) / 3

            vwap = np.sum(typical_price[-20:] * volumes[-20:]) / np.sum(volumes[-20:])
            indicators["vwap_20"] = float(vwap)

            price_vs_vwap = (prices[-1] - vwap) / vwap
            indicators["price_vs_vwap"] = float(price_vs_vwap)

        return indicators

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema


# Convenience function
def calculate_technical_indicators(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: str | None = "Volume",
) -> dict[str, float]:
    """
    Calculate all technical indicators from price DataFrame.

    Args:
        df: DataFrame with price data
        price_col: Column name for prices
        volume_col: Column name for volume (None if not available)

    Returns:
        Dict with indicator values

    Example:
        >>> df = pd.DataFrame({'Close': [100, 101, 102, 101, 103]})
        >>> indicators = calculate_technical_indicators(df)
        >>> print(indicators['rsi_14'], indicators['momentum_7d'])
    """
    try:
        return TechnicalIndicators.calculate_all(df, price_col, volume_col)
    except Exception as e:
        LOGGER.error(f"Failed to calculate technical indicators: {e}")
        return {}
