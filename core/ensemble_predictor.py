#!/usr/bin/env python3
"""
Ghost Protocol - Multi-Model Ensemble Predictor
==============================================
Combines LSTM, XGBoost, and Transformer models for superior accuracy

Target: 65-70% accuracy (up from 50% single model)
"""

import logging
import pickle
import time
import requests
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Model storage
MODELS_DIR = Path(__file__).parent.parent / "models" / "ensemble"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FEAR & GREED INDEX INTEGRATION
# ============================================================================
_FEAR_GREED_CACHE = {"value": 50, "classification": "Neutral", "updated": None}


def get_fear_greed_index() -> int:
    """
    Get current Fear & Greed Index (0-100).
    0 = Extreme Fear, 100 = Extreme Greed
    
    Uses: https://api.alternative.me/fng/
    FREE, no API key needed, 1hr cache
    """
    global _FEAR_GREED_CACHE
    
    # Return cached if less than 1 hour old
    if _FEAR_GREED_CACHE["updated"]:
        age = datetime.now() - _FEAR_GREED_CACHE["updated"]
        if age < timedelta(hours=1):
            return _FEAR_GREED_CACHE["value"]
    
    try:
        response = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=5
        )
        data = response.json()
        value = int(data["data"][0]["value"])
        classification = data["data"][0]["value_classification"]
        
        _FEAR_GREED_CACHE["value"] = value
        _FEAR_GREED_CACHE["classification"] = classification
        _FEAR_GREED_CACHE["updated"] = datetime.now()
        
        logger.info(f"[FEAR&GREED] Updated: {value} ({classification})")
        return value
        
    except Exception as e:
        logger.warning(f"[FEAR&GREED] Fetch failed: {e}, using cached: {_FEAR_GREED_CACHE['value']}")
        return _FEAR_GREED_CACHE["value"]


def get_fear_greed_signal() -> Tuple[str, float]:
    """
    Convert Fear & Greed to trading signal.
    
    Returns:
        (signal: str, confidence_modifier: float)
        
    Strategy (contrarian):
    - Extreme Fear (<20): BUY signal (+15% confidence when aligned)
    - Fear (20-40): Slight bullish (+5%)
    - Neutral (40-60): No signal
    - Greed (60-80): Slight bearish (+5% for DOWN)
    - Extreme Greed (>80): SELL signal (+15% for DOWN)
    """
    fng = get_fear_greed_index()
    
    if fng < 20:
        return "UP", 0.15  # Extreme fear = buy opportunity
    elif fng < 40:
        return "UP", 0.05  # Fear = slight bullish
    elif fng > 80:
        return "DOWN", 0.15  # Extreme greed = sell signal
    elif fng > 60:
        return "DOWN", 0.05  # Greed = slight bearish
    else:
        return "NEUTRAL", 0.0  # No signal


def get_fear_greed_info() -> Dict[str, Any]:
    """Get full Fear & Greed info for debugging."""
    fng = get_fear_greed_index()
    signal, modifier = get_fear_greed_signal()
    return {
        "value": fng,
        "classification": _FEAR_GREED_CACHE.get("classification", "Unknown"),
        "signal": signal,
        "confidence_modifier": modifier,
        "cached_at": str(_FEAR_GREED_CACHE.get("updated", "Never")),
    }


# ============================================================================
# BTC CORRELATION INTEGRATION
# ============================================================================
_BTC_TREND_CACHE = {"trend": "NEUTRAL", "change_1h": 0.0, "change_24h": 0.0, "price": 0.0, "updated": None}

# Crypto symbols that correlate with BTC
BTC_CORRELATED_SYMBOLS = {
    "BTC", "BTCUSD", "BTC-USD", "BTC/USD",
    "ETH", "ETHUSD", "ETH-USD", "ETH/USD",
    "SOL", "SOLUSD", "SOL-USD", "SOL/USD",
    "ADA", "XRP", "DOGE", "AVAX", "DOT", "MATIC", "LINK", "UNI",
    "AAVE", "ATOM", "LTC", "BCH", "XLM", "ALGO", "VET", "FIL",
    "NEAR", "APT", "ARB", "OP", "INJ", "SUI", "SEI", "TIA",
}


def get_btc_trend() -> Tuple[str, float]:
    """
    Get current BTC trend direction and strength.
    
    Returns:
        (trend: str, change_pct: float)
        trend: "UP", "DOWN", or "NEUTRAL"
        change_pct: 1-hour price change percentage
    """
    global _BTC_TREND_CACHE
    
    # Return cached if less than 5 minutes old
    if _BTC_TREND_CACHE["updated"]:
        age = datetime.now() - _BTC_TREND_CACHE["updated"]
        if age < timedelta(minutes=5):
            return _BTC_TREND_CACHE["trend"], _BTC_TREND_CACHE["change_1h"]
    
    try:
        # Use CoinGecko free API (no key needed)
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_1hr_change": "true"
            },
            timeout=5
        )
        data = response.json()
        
        price = data["bitcoin"]["usd"]
        change_1h = data["bitcoin"].get("usd_1h_change", 0) or 0
        change_24h = data["bitcoin"].get("usd_24h_change", 0) or 0
        
        # Determine trend
        if change_1h > 1.0:
            trend = "UP"
        elif change_1h < -1.0:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"
        
        _BTC_TREND_CACHE["trend"] = trend
        _BTC_TREND_CACHE["change_1h"] = change_1h
        _BTC_TREND_CACHE["change_24h"] = change_24h
        _BTC_TREND_CACHE["price"] = price
        _BTC_TREND_CACHE["updated"] = datetime.now()
        
        logger.info(f"[BTC_TREND] ${price:,.0f}, 1h: {change_1h:+.2f}%, trend: {trend}")
        return trend, change_1h
        
    except Exception as e:
        logger.warning(f"[BTC_TREND] Fetch failed: {e}, using cached")
        return _BTC_TREND_CACHE["trend"], _BTC_TREND_CACHE["change_1h"]


def get_btc_correlation_boost(symbol: str, prediction_direction: str) -> float:
    """
    Calculate confidence boost based on BTC correlation.
    
    For crypto assets, if prediction aligns with BTC trend, boost confidence.
    
    Args:
        symbol: The symbol being predicted
        prediction_direction: "UP" or "DOWN"
    
    Returns:
        Confidence multiplier (1.0 = no change, 1.1 = +10% boost)
    """
    # Normalize symbol
    symbol_upper = symbol.upper().replace("-", "").replace("/", "").replace("USD", "")
    
    # Check if this is a BTC-correlated asset
    is_crypto = any(s in symbol_upper or symbol_upper in s for s in BTC_CORRELATED_SYMBOLS)
    
    if not is_crypto:
        return 1.0  # No boost for non-crypto
    
    btc_trend, btc_change = get_btc_trend()
    
    if btc_trend == "NEUTRAL":
        return 1.0  # No signal from BTC
    
    # Calculate boost based on alignment
    if btc_trend == prediction_direction:
        # Prediction aligns with BTC - boost based on BTC move strength
        boost = min(0.15, abs(btc_change) * 0.03)  # Max +15%, 3% per 1% BTC move
        logger.info(f"[BTC_CORR] {symbol} {prediction_direction} aligns with BTC {btc_trend}, boost: +{boost:.1%}")
        return 1.0 + boost
    else:
        # Prediction conflicts with BTC - reduce confidence
        penalty = min(0.10, abs(btc_change) * 0.02)  # Max -10%
        logger.info(f"[BTC_CORR] {symbol} {prediction_direction} conflicts with BTC {btc_trend}, penalty: -{penalty:.1%}")
        return 1.0 - penalty


def get_btc_trend_info() -> Dict[str, Any]:
    """Get full BTC trend info for debugging."""
    trend, change_1h = get_btc_trend()
    return {
        "trend": trend,
        "price": _BTC_TREND_CACHE.get("price", 0),
        "change_1h": _BTC_TREND_CACHE.get("change_1h", 0),
        "change_24h": _BTC_TREND_CACHE.get("change_24h", 0),
        "cached_at": str(_BTC_TREND_CACHE.get("updated", "Never")),
        "correlated_symbols": len(BTC_CORRELATED_SYMBOLS),
    }


# ============================================================================
# VOLATILITY FILTER
# ============================================================================
# Minimum volatility thresholds - predictions on low-volatility assets are unreliable
MIN_VOLATILITY_CRYPTO = 0.5   # 0.5% minimum expected move for crypto
MIN_VOLATILITY_STOCKS = 0.3   # 0.3% minimum expected move for stocks
LOW_CONFIDENCE_THRESHOLD = 0.40  # Below this confidence = uncertain prediction


def calculate_volatility_score(price_history: List[float]) -> float:
    """
    Calculate recent volatility from price history.
    
    Returns: Average True Range as percentage of price
    """
    if not price_history or len(price_history) < 5:
        return 1.0  # Assume normal volatility if no data
    
    # Calculate percentage changes
    changes = []
    for i in range(1, len(price_history)):
        if price_history[i-1] > 0:
            pct_change = abs(price_history[i] - price_history[i-1]) / price_history[i-1] * 100
            changes.append(pct_change)
    
    if not changes:
        return 1.0
    
    # Return average volatility
    return sum(changes) / len(changes)


def should_skip_low_volatility(
    symbol: str, 
    confidence: float, 
    price_history: Optional[List[float]] = None
) -> Tuple[bool, str]:
    """
    Check if prediction should be skipped due to low volatility or confidence.
    
    Returns:
        (should_skip: bool, reason: str)
    """
    # Check confidence threshold
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return True, f"Low confidence ({confidence:.1%} < {LOW_CONFIDENCE_THRESHOLD:.0%})"
    
    # Check volatility if price history available
    if price_history and len(price_history) >= 5:
        volatility = calculate_volatility_score(price_history)
        
        # Determine threshold based on asset type
        symbol_upper = symbol.upper()
        is_crypto = any(s in symbol_upper for s in BTC_CORRELATED_SYMBOLS) or \
                   symbol_upper in ["BTCUSD", "ETHUSD", "SOLUSD"]
        
        threshold = MIN_VOLATILITY_CRYPTO if is_crypto else MIN_VOLATILITY_STOCKS
        
        if volatility < threshold:
            return True, f"Low volatility ({volatility:.2f}% < {threshold}%)"
    
    return False, "OK"


def get_volatility_info(symbol: str, price_history: Optional[List[float]] = None) -> Dict[str, Any]:
    """Get volatility analysis for debugging."""
    volatility = calculate_volatility_score(price_history) if price_history else None
    
    symbol_upper = symbol.upper()
    is_crypto = any(s in symbol_upper for s in BTC_CORRELATED_SYMBOLS)
    threshold = MIN_VOLATILITY_CRYPTO if is_crypto else MIN_VOLATILITY_STOCKS
    
    return {
        "symbol": symbol,
        "is_crypto": is_crypto,
        "volatility_pct": round(volatility, 3) if volatility else None,
        "min_threshold": threshold,
        "passes_filter": volatility >= threshold if volatility else True,
        "confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    }


@dataclass
class ModelPrediction:
    """Individual model prediction"""
    model_name: str
    direction: str  # UP/DOWN/FLAT
    confidence: float  # 0.0-1.0
    predicted_change_pct: float
    weight: float  # Model weight in ensemble


@dataclass
class EnsemblePrediction:
    """Weighted ensemble prediction"""
    direction: str
    confidence: float
    predicted_change_pct: float
    individual_predictions: list[ModelPrediction]
    model_weights: dict[str, float]
    ensemble_method: str  # weighted_vote, confidence_weighted, inverse_variance


# =============================================================================
# REMOVED: LSTMModel class (was fake - just if/else on RSI/Stochastic)
# 
# The "LSTM" was not an actual LSTM neural network. It was simple momentum
# calculations dressed up with a fancy name. Removed in Phase 2 cleanup.
# 
# If you need temporal sequence modeling, implement a real LSTM using PyTorch
# or TensorFlow with proper time-series data.
# =============================================================================


class XGBoostModel:
    """XGBoost model - TRAINED ON REAL HISTORICAL DATA (v2 with BTC correlation + Fear/Greed)"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self._loaded = False
        self.model_version = "v2"
        self._load_trained_model()
        
    def _load_trained_model(self):
        """Load the trained XGBoost model from disk (prefer v2 for now - v3 needs pipeline update)"""
        try:
            import pickle
            from pathlib import Path
            
            # NOTE: v3-hourly (75% accuracy) requires different feature extraction pipeline
            # Using v2 for now until pipeline is updated to extract hourly features
            model_path_v3 = Path(__file__).parent.parent / "models" / "trained" / "ghost_xgboost_v3_hourly.pkl"
            model_path_v2 = Path(__file__).parent.parent / "models" / "trained" / "ghost_xgboost_v2.pkl"
            model_path_v1 = Path(__file__).parent.parent / "models" / "trained" / "ghost_xgboost_v1.pkl"
            
            # Use v2 for now (pipeline produces v2-compatible features)
            # TODO: Update pipeline to extract hourly features for v3
            if model_path_v2.exists():
                model_path = model_path_v2
            elif model_path_v3.exists():
                model_path = model_path_v3  # Fallback to v3 if no v2
            else:
                model_path = model_path_v1
            
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                
                self.model = model_data["model"]
                self.feature_names = model_data["feature_names"]
                self._loaded = True
                
                # Detect version from path
                if "v3" in str(model_path):
                    self.model_version = "v3-hourly"
                elif "v2" in str(model_path):
                    self.model_version = "v2"
                else:
                    self.model_version = "v1"
                
                accuracy = model_data.get('test_accuracy', 0)
                cv_score = model_data.get('cv_score', 0)
                
                logger.info(f"✅ XGBoost {self.model_version} loaded: {accuracy:.1%} accuracy, CV={cv_score:.1%}, {len(self.feature_names)} features")
            else:
                logger.warning(f"⚠️ No trained model found, using fallback")
                
        except Exception as e:
            logger.error(f"Failed to load trained XGBoost model: {e}")
        
    def predict(self, features: Dict[str, Any]) -> ModelPrediction:
        """Generate XGBoost prediction using REAL trained model"""
        try:
            if self.model is not None and self._loaded:
                # Map feature names from prediction pipeline to training features
                # v2 features include BTC correlation, Fear/Greed, and enhanced indicators
                feature_mapping = {
                    # RSI
                    "RSI_14": "RSI_14",
                    # MACD
                    "MACD_HISTOGRAM": "MACD_HISTOGRAM",
                    "MACD_LINE": "MACD_LINE",
                    "MACD_SIGNAL": "MACD_SIGNAL",
                    # Bollinger Bands
                    "BB_POSITION": "BB_POSITION",
                    "BB_WIDTH": "BB_WIDTH",
                    "BB_UPPER": "BB_UPPER",
                    "BB_LOWER": "BB_LOWER",
                    "BB_MIDDLE": "BB_MIDDLE",
                    # Moving Averages
                    "SMA_7": "SMA_7",
                    "SMA_20": "SMA_20",
                    "SMA_50": "SMA_50",
                    "EMA_12": "EMA_12",
                    "EMA_26": "EMA_26",
                    # Stochastic
                    "STOCH_K": "STOCH_K",
                    "STOCH_D": "STOCH_D",
                    # ATR
                    "ATR_14": "ATR_14",
                    # Volume
                    "VOLUME_RATIO": "VOLUME_RATIO",
                    "VOLUME_SMA_20": "VOLUME_SMA_20",
                    "OBV": "OBV",
                    "OBV_SMA": "OBV_SMA",
                    # Momentum
                    "ROC_10": "ROC_10",
                    "MOM_10": "MOM_10",
                    # Price changes
                    "PRICE_CHANGE_1H": "PRICE_CHANGE_1H",
                    "PRICE_CHANGE_4H": "PRICE_CHANGE_4H", 
                    "PRICE_CHANGE_24H": "PRICE_CHANGE_24H",
                    # Price vs SMAs
                    "PRICE_VS_SMA_20": "PRICE_VS_SMA_20",
                    "PRICE_VS_SMA_50": "PRICE_VS_SMA_50",
                    # SMA crosses
                    "SMA_CROSS_7_20": "SMA_CROSS_7_20",
                    "SMA_CROSS_20_50": "SMA_CROSS_20_50",
                    # Volatility
                    "VOLATILITY_20D": "VOLATILITY_20",
                    "VOLATILITY_20": "VOLATILITY_20",
                    # Range
                    "DAILY_RANGE_PCT": "DAILY_RANGE_PCT",
                    
                    # === V2 ENHANCED FEATURES ===
                    # BTC Correlation (Critical for altcoin prediction)
                    "BTC_RSI": "BTC_RSI",
                    "BTC_MOMENTUM_24H": "BTC_MOMENTUM_24H",
                    "BTC_MOMENTUM_7D": "BTC_MOMENTUM_7D",
                    "BTC_MACD_BULLISH": "BTC_MACD_BULLISH",
                    "BTC_ABOVE_SMA_20": "BTC_ABOVE_SMA_20",
                    "BTC_CORRELATION": "BTC_CORRELATION",
                    
                    # Fear & Greed Index
                    "FEAR_GREED": "fear_greed_numeric",
                    "fear_greed_numeric": "fear_greed_numeric",
                    "FEAR_GREED_MA": "FEAR_GREED_MA",
                    "FEAR_GREED_EXTREME": "FEAR_GREED_EXTREME",
                    
                    # Funding Rates (leverage sentiment)
                    "FUNDING_RATE": "funding_rate_proxy",
                    "funding_rate_proxy": "funding_rate_proxy",
                    
                    # RSI Zones (v2 feature engineering)
                    "RSI_OVERSOLD": "RSI_OVERSOLD",
                    "RSI_OVERBOUGHT": "RSI_OVERBOUGHT",
                    "MACD_BULLISH": "MACD_BULLISH",
                    "ABOVE_SMA_20": "ABOVE_SMA_20",
                    "ABOVE_SMA_50": "ABOVE_SMA_50",
                    "EMA_BULLISH": "EMA_BULLISH",
                    
                    # Volume features
                    "VOLUME_SPIKE": "VOLUME_SPIKE",
                    "VOLUME_TREND": "VOLUME_TREND",
                    
                    # Market structure
                    "HIGHER_HIGH": "HIGHER_HIGH",
                    "LOWER_LOW": "LOWER_LOW",
                    "NEAR_24H_HIGH": "NEAR_24H_HIGH",
                    "NEAR_24H_LOW": "NEAR_24H_LOW",
                }
                
                # NEUTRAL DEFAULTS: Use neutral values instead of 0 to avoid DOWN bias
                # When features are missing, we assume neutral market conditions
                neutral_defaults = {
                    # Binary features default to 0.5 (uncertain) or actual neutral value
                    "RSI_OVERSOLD": 0,           # Not oversold
                    "RSI_OVERBOUGHT": 0,         # Not overbought
                    "MACD_BULLISH": 0.5,         # Uncertain
                    "ABOVE_SMA_20": 0.5,         # Uncertain 
                    "ABOVE_SMA_50": 0.5,         # Uncertain
                    "EMA_BULLISH": 0.5,          # Uncertain
                    "SMA_CROSS_20_50": 0,        # No cross
                    "NEAR_7D_HIGH": 0,           # Not near high
                    "NEAR_7D_LOW": 0,            # Not near low
                    "NEAR_30D_HIGH": 0,          # Not near high
                    "NEAR_30D_LOW": 0,           # Not near low
                    "VOLUME_SPIKE": 0,           # No spike
                    "HIGH_FUNDING": 0,           # Normal funding
                    "NEGATIVE_FUNDING": 0,       # Normal funding
                    "EXTREME_FEAR": 0,           # Not extreme fear
                    "EXTREME_GREED": 0,          # Not extreme greed
                    "BTC_MACD_BULLISH": 0.5,     # Uncertain
                    "BTC_LEADS": 0,              # No lead signal
                    
                    # Continuous features default to neutral values
                    "RSI_14": 50,                # Neutral RSI
                    "BB_POSITION": 0.5,          # Middle of bands
                    "STOCH_K": 50,               # Neutral stochastic
                    "STOCH_D": 50,               # Neutral stochastic
                    "VOLUME_RATIO": 1.0,         # Average volume
                    "fear_greed_value": 50,      # Neutral fear/greed
                    "fear_greed_numeric": 50,    # Neutral fear/greed
                    "funding_rate_proxy": 0,     # Neutral funding
                    "BTC_RSI": 50,               # Neutral BTC RSI
                    "BTC_MOMENTUM_1D": 0,        # Flat BTC
                    "BTC_MOMENTUM_7D": 0,        # Flat BTC
                    "BTC_CORRELATION": 0.5,      # Moderate correlation
                    "MOMENTUM_1D": 0,            # Flat
                    "MOMENTUM_7D": 0,            # Flat
                    "MOMENTUM_30D": 0,           # Flat
                    "ROC_10": 0,                 # No rate of change
                    "VOLATILITY_7D": 0.02,       # Normal volatility
                    "VOLATILITY_30D": 0.02,      # Normal volatility
                    "DAILY_RANGE_PCT": 2.0,      # Normal daily range
                }
                
                # Extract features in correct order
                feature_values = []
                missing_features = []
                for name in self.feature_names:
                    # Try direct match first
                    value = features.get(name, None)
                    
                    # Try mapping
                    if value is None:
                        for src, dst in feature_mapping.items():
                            if dst == name and src in features:
                                value = features.get(src)
                                break
                    
                    # Track missing features for debugging
                    if value is None:
                        missing_features.append(name)
                    
                    # Use neutral default if missing (avoid DOWN bias from 0s)
                    if value is None:
                        value = neutral_defaults.get(name, 0.0)
                    
                    feature_values.append(float(value))
                
                if missing_features and len(missing_features) < 10:
                    logger.debug(f"Missing features for XGBoost: {missing_features[:5]}...")
                
                X = np.array([feature_values])
                
                # Predict using trained model
                prediction = self.model.predict(X)[0]
                proba = self.model.predict_proba(X)[0]
                
                # Get probabilities for each class
                prob_down = float(proba[0])  # Class 0 = DOWN
                prob_up = float(proba[1])    # Class 1 = UP
                
                # RAW MODEL OUTPUT (no hacks, no bias correction, no compression)
                # Let the model speak for itself - we'll measure REAL accuracy
                logger.info(
                    f"🤖 XGBoost {self.model_version} for {features.get('symbol', '?')}: "
                    f"UP={prob_up:.1%}, DOWN={prob_down:.1%}"
                )
                
                # Simple threshold: 55% conviction required
                # (Slightly above 50% to filter pure noise)
                conviction_threshold = 0.55
                
                if prob_up >= conviction_threshold:
                    direction = "UP"
                    confidence = prob_up
                elif prob_down >= conviction_threshold:
                    direction = "DOWN"
                    confidence = prob_down
                else:
                    # Truly uncertain - FLAT (now rare: requires 48-52% band)
                    direction = "FLAT"
                    confidence = max(prob_up, prob_down)
                
                # Scale predicted change based on confidence
                if direction == "UP":
                    predicted_change = confidence * 6.0
                elif direction == "DOWN":
                    predicted_change = -confidence * 6.0
                else:
                    predicted_change = 0.0
                
                return ModelPrediction(
                    model_name=f"XGBoost-{self.model_version}",
                    direction=direction,
                    confidence=confidence,
                    predicted_change_pct=predicted_change,
                    weight=0.7 if self.model_version == "v2" else 0.6
                )
            else:
                # Fallback to feature-based prediction if model not loaded
                return self._fallback_predict(features)
                
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return self._fallback_predict(features)
    
    def _fallback_predict(self, features: Dict[str, Any]) -> ModelPrediction:
        """Fallback when trained model unavailable - NEUTRAL by default"""
        rsi = features.get("RSI_14", features.get("rsi", 50))  # Default 50 = neutral
        macd = features.get("MACD_HISTOGRAM", features.get("macd", None))  # None = no signal
        volume_ratio = features.get("VOLUME_RATIO", features.get("volume_ratio", 1.0))
        
        score = 0
        
        # RSI signals - only fire on extremes
        if rsi is not None:
            if rsi < 30:
                score += 2  # Oversold = bullish
            elif rsi > 70:
                score -= 2  # Overbought = bearish
            # 30-70 = neutral, no score change
        
        # MACD signals - only when we have data
        if macd is not None and macd != 0:  # Don't assume bearish if missing
            if macd > 0:
                score += 1
            elif macd < 0:
                score -= 1
        
        # Volume signals
        if volume_ratio is not None and volume_ratio > 1.5:
            score += 1  # High volume = conviction (in current direction)
        
        # LOWERED THRESHOLD: 1+ signal is enough for directional call
        # Old: required 2+ signals, now: 1+ signals
        if abs(score) < 1:
            direction = "FLAT"  # Truly no signal = FLAT
            confidence = 0.40
        else:
            direction = "UP" if score > 0 else "DOWN"
            confidence = min(abs(score) / 4 + 0.35, 0.70)
        
        return ModelPrediction(
            model_name="XGBoost-Fallback",
            direction=direction,
            confidence=confidence,
            predicted_change_pct=score * 2 if direction != "FLAT" else 0,
            weight=0.3  # Lower weight for fallback
        )


# =============================================================================
# REMOVED: TransformerModel class (was fake - just weighted average of RSI/MACD/BB)
# 
# The "Transformer" was not an actual Transformer neural network. It was a simple
# weighted average of technical indicators dressed up with a fancy name.
# Removed in Phase 2 cleanup.
# 
# If you need attention-based sequence modeling, implement a real Transformer
# using PyTorch or TensorFlow with proper positional encoding and self-attention.
# =============================================================================


class EnsemblePredictor:
    """
    Simplified Predictor - XGBoost Only
    
    Phase 2 Cleanup: Removed fake LSTM and Transformer models.
    Now uses only the trained XGBoost model + market regime signals.
    
    Market regime signals (Fear & Greed, BTC correlation) are kept because
    they provide legitimate external market context.
    """
    
    def __init__(self):
        self.xgboost = XGBoostModel()
        
        # Model performance tracking
        self.performance_history = {
            "XGBoost": deque(maxlen=100),
        }
        
    def predict(self, features: dict[str, Any], method: str = "confidence_weighted", symbol: str = "") -> EnsemblePrediction:
        """
        Generate prediction using XGBoost + market regime signals.
        
        Args:
            features: Feature dict with technical indicators
            method: Ignored (kept for API compatibility)
            symbol: Optional symbol for BTC correlation boost (crypto only)
        
        Returns:
            EnsemblePrediction with XGBoost result + market adjustments
        """
        start = time.time()
        
        # Get XGBoost prediction (the only real ML model)
        xgb_pred = self.xgboost.predict(features)
        predictions = [xgb_pred]
        
        # =====================================================================
        # MARKET REGIME ADJUSTMENT
        # Use Fear & Greed for SMALL adjustments in extreme conditions
        # ADDITIVE instead of multiplicative to preserve natural variation
        # =====================================================================
        try:
            fng = get_fear_greed_index()
            
            # Extreme Fear (<25): Be cautious about DOWN predictions
            # During panic, markets often bounce - don't short the bottom
            if xgb_pred.direction == "DOWN" and fng < 25:
                logger.warning(
                    f"[REGIME] XGBoost says DOWN but Fear&Greed={fng} (EXTREME FEAR). "
                    f"Small reduction - don't short during panic."
                )
                xgb_pred = ModelPrediction(
                    model_name=xgb_pred.model_name,
                    direction=xgb_pred.direction,
                    confidence=max(0.35, xgb_pred.confidence - 0.08),  # -8% additive
                    predicted_change_pct=xgb_pred.predicted_change_pct * 0.9,
                    weight=xgb_pred.weight
                )
                predictions = [xgb_pred]
            
            # Extreme Fear + UP signal: Small boost (contrarian buy)
            elif xgb_pred.direction == "UP" and fng < 30:
                logger.info(
                    f"[REGIME] XGBoost says UP in Fear&Greed={fng} (FEAR). "
                    f"Small boost - contrarian buy signal."
                )
                xgb_pred = ModelPrediction(
                    model_name=xgb_pred.model_name,
                    direction=xgb_pred.direction,
                    confidence=min(0.90, xgb_pred.confidence + 0.05),  # +5% additive
                    predicted_change_pct=xgb_pred.predicted_change_pct,
                    weight=xgb_pred.weight
                )
                predictions = [xgb_pred]
            
            # Extreme Greed (>75): Be cautious about UP predictions
            elif xgb_pred.direction == "UP" and fng > 75:
                logger.warning(
                    f"[REGIME] XGBoost says UP but Fear&Greed={fng} (GREED). "
                    f"Small reduction - market may be overheated."
                )
                xgb_pred = ModelPrediction(
                    model_name=xgb_pred.model_name,
                    direction=xgb_pred.direction,
                    confidence=max(0.35, xgb_pred.confidence - 0.05),  # -5% additive
                    predicted_change_pct=xgb_pred.predicted_change_pct * 0.9,
                    weight=xgb_pred.weight
                )
                predictions = [xgb_pred]
                
        except Exception as e:
            logger.error(f"[REGIME] Error getting Fear&Greed: {e}")
        
        # Build ensemble result (now just XGBoost)
        # PRESERVE XGBoost's raw confidence - don't cap it artificially
        # The model's actual probability distributions are valuable signal
        ensemble_result = EnsemblePrediction(
            direction=xgb_pred.direction,
            confidence=xgb_pred.confidence,  # REMOVED 75% CAP - use real ML confidence
            predicted_change_pct=xgb_pred.predicted_change_pct,
            individual_predictions=predictions,
            model_weights={"XGBoost": 1.0},
            ensemble_method="xgboost_only"
        )
        
        # =====================================================================
        # FEAR & GREED INTEGRATION
        # ADDITIVE adjustments to preserve XGBoost's natural variation
        # Small nudges (+/-5%) instead of multiplicative compression
        # =====================================================================
        try:
            fng_signal, fng_modifier = get_fear_greed_signal()
            original_confidence = ensemble_result.confidence
            
            # Skip boosting for FLAT predictions - FLAT should remain uncertain
            if ensemble_result.direction == "FLAT":
                logger.debug(f"[FEAR&GREED] Skipping boost for FLAT prediction")
            elif fng_signal != "NEUTRAL":
                if fng_signal == ensemble_result.direction:
                    # Fear & Greed AGREES with prediction - small additive boost
                    # Use smaller boost (3-8%) to preserve variation
                    additive_boost = min(fng_modifier, 0.08)  # Cap at 8%
                    boosted = min(0.90, ensemble_result.confidence + additive_boost)  # Cap final at 90%
                    logger.info(
                        f"[FEAR&GREED] {fng_signal} aligns with {ensemble_result.direction}, "
                        f"confidence {original_confidence:.1%} -> {boosted:.1%} (+{additive_boost:.1%})"
                    )
                    ensemble_result = EnsemblePrediction(
                        direction=ensemble_result.direction,
                        confidence=boosted,
                        predicted_change_pct=ensemble_result.predicted_change_pct,
                        individual_predictions=ensemble_result.individual_predictions,
                        model_weights=ensemble_result.model_weights,
                        ensemble_method=ensemble_result.ensemble_method
                    )
                elif (fng_signal == "UP" and ensemble_result.direction == "DOWN") or \
                     (fng_signal == "DOWN" and ensemble_result.direction == "UP"):
                    # Fear & Greed DISAGREES - small additive reduction
                    additive_reduction = min(fng_modifier / 2, 0.05)  # Cap at 5%
                    reduced = max(0.35, ensemble_result.confidence - additive_reduction)
                    logger.info(
                        f"[FEAR&GREED] {fng_signal} conflicts with {ensemble_result.direction}, "
                        f"confidence {original_confidence:.1%} -> {reduced:.1%} (-{additive_reduction:.1%})"
                    )
                    ensemble_result = EnsemblePrediction(
                        direction=ensemble_result.direction,
                        confidence=reduced,
                        predicted_change_pct=ensemble_result.predicted_change_pct,
                        individual_predictions=ensemble_result.individual_predictions,
                        model_weights=ensemble_result.model_weights,
                        ensemble_method=ensemble_result.ensemble_method
                    )
        except Exception as e:
            logger.warning(f"[FEAR&GREED] Integration error: {e}")
        
        # =====================================================================
        # BTC CORRELATION INTEGRATION
        # ADDITIVE adjustments for crypto based on BTC trend alignment
        # Small nudges (+/-5%) to preserve natural variation
        # =====================================================================
        if symbol and ensemble_result.direction != "FLAT":  # Skip FLAT
            try:
                btc_multiplier = get_btc_correlation_boost(symbol, ensemble_result.direction)
                if btc_multiplier != 1.0:
                    original_conf = ensemble_result.confidence
                    # Convert multiplier to additive: 1.10 -> +5%, 0.90 -> -5%
                    additive_adjust = (btc_multiplier - 1.0) * 0.5  # Halve the effect
                    additive_adjust = max(-0.05, min(0.05, additive_adjust))  # Cap at +/-5%
                    new_conf = min(0.90, max(0.35, ensemble_result.confidence + additive_adjust))
                    logger.info(
                        f"[BTC_CORR] {symbol}: confidence {original_conf:.1%} -> {new_conf:.1%} "
                        f"(adjust: {additive_adjust:+.1%})"
                    )
                    ensemble_result = EnsemblePrediction(
                        direction=ensemble_result.direction,
                        confidence=new_conf,
                        predicted_change_pct=ensemble_result.predicted_change_pct,
                        individual_predictions=ensemble_result.individual_predictions,
                        model_weights=ensemble_result.model_weights,
                        ensemble_method=ensemble_result.ensemble_method
                    )
            except Exception as e:
                logger.warning(f"[BTC_CORR] Integration error: {e}")
        
        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"Ensemble prediction: {ensemble_result.direction} "
            f"({ensemble_result.confidence:.1%}) in {duration_ms:.0f}ms"
        )
        
        return ensemble_result
    
    def _calculate_adaptive_weights(self) -> dict[str, float]:
        """Calculate model weights based on recent performance (simplified for XGBoost-only)"""
        weights = {"XGBoost": 1.0}
        
        # If we have performance history, use it
        if "XGBoost" in self.performance_history and len(self.performance_history["XGBoost"]) > 10:
            accuracy = sum(self.performance_history["XGBoost"]) / len(self.performance_history["XGBoost"])
            # Weight stays at 1.0 for single model, but we track accuracy
            logger.debug(f"XGBoost recent accuracy: {accuracy:.1%}")
        
        return weights
    
    # =========================================================================
    # REMOVED: _confidence_weighted_ensemble, _weighted_vote_ensemble, 
    # _inverse_variance_ensemble methods
    # 
    # These were only needed when combining multiple models. With XGBoost-only,
    # we use the XGBoost prediction directly (with market regime adjustments).
    # =========================================================================
    
    def update_performance(self, model_name: str, was_correct: bool) -> None:
        """Update model performance history for tracking accuracy"""
        # Normalize model name to XGBoost for backward compatibility
        if "xgboost" in model_name.lower() or "XGBoost" in model_name:
            model_name = "XGBoost"
        
        if model_name in self.performance_history:
            self.performance_history[model_name].append(1.0 if was_correct else 0.0)
            logger.debug(f"Updated {model_name} performance: {was_correct}")


# Global ensemble instance
_ensemble_predictor: EnsemblePredictor | None = None


def get_ensemble_predictor() -> EnsemblePredictor:
    """Get or create global ensemble predictor"""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsemblePredictor()
        logger.info("✅ Predictor initialized (XGBoost + Market Regime signals)")
    return _ensemble_predictor


if __name__ == "__main__":
    # Test predictor
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 Testing Predictor (XGBoost + Market Regime)")
    print("=" * 60)
    
    predictor = get_ensemble_predictor()
    
    # Test features
    test_features = {
        "RSI_14": 45,
        "MACD_HISTOGRAM": 0.5,
        "VOLUME_RATIO": 1.8,
        "STOCH_K": 50,
        "BB_POSITION": 0.5,
    }
    
    # Generate prediction
    result = predictor.predict(test_features, symbol="BTC")
    
    print(f"\n📊 Prediction Result:")
    print(f"  Direction: {result.direction}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Predicted Change: {result.predicted_change_pct:+.2f}%")
    print(f"  Method: {result.ensemble_method}")
    
    print(f"\n🎯 Model:")
    for pred in result.individual_predictions:
        print(f"  {pred.model_name}: {pred.direction} ({pred.confidence:.1%})")
    
    print("\n✅ Predictor test complete")
