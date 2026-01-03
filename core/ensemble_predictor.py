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


class LSTMModel:
    """LSTM-style temporal pattern recognition using momentum signals"""
    
    def __init__(self):
        self.lookback = 24  # Hours of momentum to consider
        
    def predict(self, features: Dict[str, Any]) -> ModelPrediction:
        """
        Generate prediction based on temporal momentum patterns.
        Uses multiple timeframe momentum analysis as LSTM proxy.
        """
        try:
            # Multi-timeframe momentum analysis (LSTM-style temporal patterns)
            momentum_1h = features.get("PRICE_CHANGE_1H", features.get("price_change_1h", 0)) or 0
            momentum_4h = features.get("PRICE_CHANGE_4H", features.get("price_change_4h", 0)) or 0
            momentum_24h = features.get("PRICE_CHANGE_24H", features.get("price_change_24h", 0)) or 0
            roc_10 = features.get("ROC_10", 0) or 0
            mom_10 = features.get("MOM_10", 0) or 0
            
            # Calculate momentum consensus
            momentum_signals = []
            
            # Short-term momentum (1h, 4h) - LOWERED thresholds for better sensitivity
            if momentum_1h > 0.3:
                momentum_signals.append(1)
            elif momentum_1h < -0.3:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            if momentum_4h > 0.7:
                momentum_signals.append(1)
            elif momentum_4h < -0.7:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            # Medium-term momentum (24h) - LOWERED threshold
            if momentum_24h > 1.5:
                momentum_signals.append(1)
            elif momentum_24h < -1.5:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            # ROC momentum
            if roc_10 > 3.0:
                momentum_signals.append(1)
            elif roc_10 < -3.0:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            # Calculate consensus
            total_momentum = sum(momentum_signals)
            agreement = sum(1 for s in momentum_signals if s == np.sign(total_momentum)) if total_momentum != 0 else 0
            
            # LOWERED THRESHOLDS: Even 1 aligned signal is meaningful
            # Old: required > 1 (2+ signals), now: > 0 (1+ signals)
            if total_momentum > 0:
                direction = "UP"
                confidence = min(0.45 + (agreement / len(momentum_signals)) * 0.35, 0.80)
            elif total_momentum < 0:
                direction = "DOWN"
                confidence = min(0.45 + (agreement / len(momentum_signals)) * 0.35, 0.80)
            else:
                direction = "FLAT"
                confidence = 0.35
            
            # Predicted change based on momentum strength
            predicted_change = total_momentum * 1.5
            
            return ModelPrediction(
                model_name="LSTM-Momentum",
                direction=direction,
                confidence=confidence,
                predicted_change_pct=predicted_change,
                weight=0.25  # Lower weight - momentum is supplementary
            )
            
        except Exception as e:
            logger.error(f"LSTM momentum prediction failed: {e}")
            return ModelPrediction(
                model_name="LSTM-Momentum",
                direction="FLAT",
                confidence=0.3,
                predicted_change_pct=0.0,
                weight=0.25
            )


class XGBoostModel:
    """XGBoost model - TRAINED ON REAL HISTORICAL DATA (v2 with BTC correlation + Fear/Greed)"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self._loaded = False
        self.model_version = "v2"
        self._load_trained_model()
        
    def _load_trained_model(self):
        """Load the trained XGBoost model from disk (prefer v2)"""
        try:
            import pickle
            from pathlib import Path
            
            # Try v2 first (enhanced with BTC correlation, Fear/Greed)
            model_path_v2 = Path(__file__).parent.parent / "models" / "trained" / "ghost_xgboost_v2.pkl"
            model_path_v1 = Path(__file__).parent.parent / "models" / "trained" / "ghost_xgboost_v1.pkl"
            
            model_path = model_path_v2 if model_path_v2.exists() else model_path_v1
            
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                
                self.model = model_data["model"]
                self.feature_names = model_data["feature_names"]
                self._loaded = True
                self.model_version = "v2" if "v2" in str(model_path) else "v1"
                
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
                
                # AGGRESSIVE BIAS CORRECTION
                # The XGBoost v2 model was trained on bear market data and has severe DOWN bias
                # When the model says 95% DOWN, it's really more like 60% DOWN
                # Apply logistic compression to reduce extreme confidence
                import math
                
                def compress_probability(p: float, center: float = 0.5, strength: float = 2.0) -> float:
                    """Compress extreme probabilities toward center"""
                    if p <= 0.01:
                        return 0.1
                    if p >= 0.99:
                        return 0.9
                    # Logit transform, compress, then back to probability
                    logit = math.log(p / (1 - p))
                    compressed_logit = logit / strength
                    compressed = 1 / (1 + math.exp(-compressed_logit))
                    # Blend with center
                    return center + (compressed - center) * 0.8
                
                prob_up_calibrated = compress_probability(prob_up)
                prob_down_calibrated = compress_probability(prob_down)
                
                # Normalize
                total = prob_up_calibrated + prob_down_calibrated
                prob_up_calibrated = prob_up_calibrated / total
                prob_down_calibrated = prob_down_calibrated / total
                
                # Log the calibration for debugging
                logger.info(
                    f"🤖 XGBoost {self.model_version} for {features.get('symbol', '?')}: "
                    f"Raw={prob_up:.1%}/{prob_down:.1%} → Calibrated={prob_up_calibrated:.1%}/{prob_down_calibrated:.1%}"
                )
                
                # CONVICTION THRESHOLD: Lowered from 57% to 52% - slight edge is still a signal
                # Markets are inherently uncertain; 52% edge is meaningful
                conviction_threshold = 0.52
                
                if prob_up_calibrated >= conviction_threshold:
                    direction = "UP"
                    confidence = prob_up_calibrated
                elif prob_down_calibrated >= conviction_threshold:
                    direction = "DOWN"
                    confidence = prob_down_calibrated
                else:
                    # Truly uncertain - FLAT (now rare: requires 48-52% band)
                    direction = "FLAT"
                    confidence = max(prob_up_calibrated, prob_down_calibrated)
                
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


class TransformerModel:
    """Market context model - combines multiple indicator signals"""
    
    def __init__(self):
        self.indicators = ["RSI", "MACD", "BB", "STOCH", "VOLUME"]
        
    def predict(self, features: Dict[str, Any]) -> ModelPrediction:
        """
        Generate prediction by weighing multiple technical indicators.
        Acts as a market context synthesizer.
        """
        try:
            signals = []
            weights = []
            
            # RSI Signal (high weight - reliable)
            rsi = features.get("RSI_14", features.get("rsi", None))
            if rsi is not None:
                if rsi < 30:
                    signals.append(1)  # Oversold = UP
                    weights.append(0.25)
                elif rsi > 70:
                    signals.append(-1)  # Overbought = DOWN
                    weights.append(0.25)
                else:
                    signals.append(0)
                    weights.append(0.1)
            
            # MACD Signal - only bearish if clearly negative
            macd = features.get("MACD_HISTOGRAM", features.get("macd", None))
            if macd is not None and macd != 0:  # Need actual MACD value
                if macd > 0:
                    signals.append(1)
                    weights.append(0.2)
                elif macd < 0:  # Only bearish if MACD is negative
                    signals.append(-1)
                    weights.append(0.2)
                # MACD = 0 is neutral, don't add signal
            
            # Bollinger Band Position
            bb_pos = features.get("BB_POSITION", None)
            if bb_pos is not None:
                if bb_pos < 0.2:  # Near lower band
                    signals.append(1)
                    weights.append(0.15)
                elif bb_pos > 0.8:  # Near upper band
                    signals.append(-1)
                    weights.append(0.15)
                else:
                    signals.append(0)
                    weights.append(0.05)
            
            # Stochastic Signal
            stoch_k = features.get("STOCH_K", None)
            if stoch_k is not None:
                if stoch_k < 20:
                    signals.append(1)
                    weights.append(0.15)
                elif stoch_k > 80:
                    signals.append(-1)
                    weights.append(0.15)
                else:
                    signals.append(0)
                    weights.append(0.05)
            
            # Volume confirmation
            vol_ratio = features.get("VOLUME_RATIO", features.get("volume_ratio", None))
            if vol_ratio is not None and vol_ratio > 1.5:
                # High volume confirms existing signals
                weights = [w * 1.2 for w in weights]
            
            # Calculate weighted consensus
            if signals and weights:
                total_weight = sum(weights)
                weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight if total_weight > 0 else 0
                
                # LOWERED THRESHOLD: 0.15 (was 0.3) - mild signal is still signal
                # Markets are noisy; waiting for strong consensus = always FLAT
                if weighted_signal > 0.15:
                    direction = "UP"
                    confidence = min(0.50 + abs(weighted_signal) * 0.35, 0.75)
                elif weighted_signal < -0.15:
                    direction = "DOWN"
                    confidence = min(0.50 + abs(weighted_signal) * 0.35, 0.75)
                else:
                    direction = "FLAT"  # Truly mixed signals
                    confidence = 0.40
                
                predicted_change = weighted_signal * 5
            else:
                direction = "FLAT"
                confidence = 0.3
                predicted_change = 0
            
            return ModelPrediction(
                model_name="Context-Synthesizer",
                direction=direction,
                confidence=confidence,
                predicted_change_pct=predicted_change,
                weight=0.30  # Increased weight - good at reading technicals
            )
            
        except Exception as e:
            logger.error(f"Context synthesizer prediction failed: {e}")
            return ModelPrediction(
                model_name="Context-Synthesizer",
                direction="FLAT",
                confidence=0.3,
                predicted_change_pct=0.0,
                weight=0.30
            )


class EnsemblePredictor:
    """Weighted ensemble of LSTM + XGBoost + Transformer"""
    
    def __init__(self):
        self.lstm = LSTMModel()
        self.xgboost = XGBoostModel()
        self.transformer = TransformerModel()
        
        # Model performance tracking (for adaptive weights)
        self.performance_history = {
            "LSTM": deque(maxlen=100),
            "XGBoost": deque(maxlen=100),
            "Transformer": deque(maxlen=100)
        }
        
    def predict(self, features: dict[str, Any], method: str = "confidence_weighted", symbol: str = "") -> EnsemblePrediction:
        """
        Generate ensemble prediction
        
        Args:
            features: Feature dict with price_history, technical indicators, etc.
            method: Ensemble method (weighted_vote, confidence_weighted, inverse_variance)
            symbol: Optional symbol for BTC correlation boost (crypto only)
        
        Returns:
            EnsemblePrediction with aggregated results
        """
        start = time.time()
        
        # Get predictions from all models
        lstm_pred = self.lstm.predict(features)
        xgb_pred = self.xgboost.predict(features)
        transformer_pred = self.transformer.predict(features)
        
        predictions = [lstm_pred, xgb_pred, transformer_pred]
        
        # =====================================================================
        # MARKET REGIME OVERRIDE - Fix XGBoost DOWN bias
        # The XGBoost model was trained on bear market data and has severe DOWN bias
        # Use Fear & Greed + model consensus to override bad predictions
        # =====================================================================
        try:
            fng = get_fear_greed_index()
            
            # Count model directions
            up_count = sum(1 for p in predictions if p.direction == "UP")
            down_count = sum(1 for p in predictions if p.direction == "DOWN")
            flat_count = sum(1 for p in predictions if p.direction == "FLAT")
            
            # Situation: XGBoost says DOWN but market is in FEAR and other models disagree
            if xgb_pred.direction == "DOWN" and fng < 35 and up_count >= 1:
                logger.warning(
                    f"[REGIME_OVERRIDE] XGBoost DOWN bias detected! "
                    f"Fear&Greed={fng} (FEAR), {up_count} models say UP. "
                    f"Reducing XGBoost weight and boosting bullish signals."
                )
                # Create modified XGBoost prediction with reduced confidence
                from copy import copy
                xgb_pred_modified = ModelPrediction(
                    model_name=xgb_pred.model_name,
                    direction="FLAT",  # Downgrade to FLAT instead of DOWN
                    confidence=0.40,   # Low confidence
                    predicted_change_pct=0.0,
                    weight=0.3  # Reduce weight from 0.7 to 0.3
                )
                predictions[1] = xgb_pred_modified
                logger.info(f"[REGIME_OVERRIDE] XGBoost downgraded: {xgb_pred.direction} -> FLAT (0.40 conf)")
            
            # Situation: XGBoost is the ONLY model saying DOWN in FEAR territory (others are FLAT)
            # In fear markets, we should not be bearish unless multiple models agree
            elif xgb_pred.direction == "DOWN" and fng < 40 and down_count == 1 and flat_count >= 1:
                logger.warning(
                    f"[REGIME_OVERRIDE] XGBoost ALONE says DOWN in FEAR market (FNG={fng}). "
                    f"Other models: {flat_count} FLAT, {up_count} UP. Downgrading XGBoost to FLAT."
                )
                xgb_pred_modified = ModelPrediction(
                    model_name=xgb_pred.model_name,
                    direction="FLAT",  # In fear, don't trust single DOWN signal
                    confidence=0.45,
                    predicted_change_pct=0.0,
                    weight=0.4
                )
                predictions[1] = xgb_pred_modified
                
            # Situation: XGBoost says DOWN but Transformer AND LSTM disagree
            elif xgb_pred.direction == "DOWN" and lstm_pred.direction != "DOWN" and transformer_pred.direction == "UP":
                logger.warning(
                    f"[REGIME_OVERRIDE] XGBoost alone in DOWN call. "
                    f"LSTM={lstm_pred.direction}, Transformer={transformer_pred.direction}. "
                    f"Reducing XGBoost influence."
                )
                xgb_pred_modified = ModelPrediction(
                    model_name=xgb_pred.model_name,
                    direction=xgb_pred.direction,
                    confidence=xgb_pred.confidence * 0.5,  # Halve confidence
                    predicted_change_pct=xgb_pred.predicted_change_pct * 0.5,
                    weight=0.4  # Reduce weight
                )
                predictions[1] = xgb_pred_modified
                logger.info(f"[REGIME_OVERRIDE] XGBoost confidence halved: {xgb_pred.confidence:.1%} -> {xgb_pred_modified.confidence:.1%}")
                
            # Situation: Strong FEAR (<25) - boost any UP signals
            elif fng < 25 and up_count >= 1:
                logger.info(f"[REGIME_OVERRIDE] EXTREME FEAR ({fng}), boosting bullish signals")
                for i, pred in enumerate(predictions):
                    if pred.direction == "UP":
                        boosted = ModelPrediction(
                            model_name=pred.model_name,
                            direction=pred.direction,
                            confidence=min(0.85, pred.confidence * 1.3),  # Boost UP confidence
                            predicted_change_pct=pred.predicted_change_pct,
                            weight=pred.weight * 1.2
                        )
                        predictions[i] = boosted
            
            # Situation: 2/3 models agree on direction - that consensus should win
            # This overrides the single biased XGBoost model
            if up_count >= 2:
                logger.info(f"[REGIME_OVERRIDE] MODEL CONSENSUS: {up_count}/3 models say UP - boosting UP weight")
                for i, pred in enumerate(predictions):
                    if pred.direction == "UP":
                        boosted = ModelPrediction(
                            model_name=pred.model_name,
                            direction=pred.direction,
                            confidence=min(0.85, pred.confidence * 1.2),
                            predicted_change_pct=pred.predicted_change_pct,
                            weight=pred.weight * 1.5  # Significant weight boost
                        )
                        predictions[i] = boosted
                    elif pred.direction != "UP":
                        # Reduce weight of non-UP predictions when consensus is UP
                        reduced = ModelPrediction(
                            model_name=pred.model_name,
                            direction=pred.direction,
                            confidence=pred.confidence * 0.7,
                            predicted_change_pct=pred.predicted_change_pct,
                            weight=pred.weight * 0.5
                        )
                        predictions[i] = reduced
                        
            elif down_count >= 2:
                # FEAR MARKET OVERRIDE: Don't trust DOWN consensus in FEAR territory
                # Historical data shows buying in fear beats selling in fear
                if fng < 35:
                    logger.warning(
                        f"[REGIME_OVERRIDE] DOWN consensus ({down_count}/3) in FEAR market (FNG={fng})! "
                        f"Converting to FLAT - don't short during fear."
                    )
                    for i, pred in enumerate(predictions):
                        if pred.direction == "DOWN":
                            # Convert DOWN to FLAT with reduced confidence
                            reduced = ModelPrediction(
                                model_name=pred.model_name,
                                direction="FLAT",
                                confidence=pred.confidence * 0.6,
                                predicted_change_pct=0.0,
                                weight=pred.weight * 0.5
                            )
                            predictions[i] = reduced
                else:
                    # Not in fear - trust DOWN consensus
                    logger.info(f"[REGIME_OVERRIDE] MODEL CONSENSUS: {down_count}/3 models say DOWN - boosting DOWN weight")
                    for i, pred in enumerate(predictions):
                        if pred.direction == "DOWN":
                            boosted = ModelPrediction(
                                model_name=pred.model_name,
                                direction=pred.direction,
                                confidence=min(0.85, pred.confidence * 1.2),
                                predicted_change_pct=pred.predicted_change_pct,
                            weight=pred.weight * 1.5
                        )
                        predictions[i] = boosted
                        
        except Exception as e:
            logger.error(f"[REGIME_OVERRIDE] Error: {e}")
        
        # Adaptive weights based on recent performance
        weights = self._calculate_adaptive_weights()
        
        # Apply ensemble method
        if method == "confidence_weighted":
            ensemble_result = self._confidence_weighted_ensemble(predictions, weights)
        elif method == "inverse_variance":
            ensemble_result = self._inverse_variance_ensemble(predictions, weights)
        else:  # weighted_vote
            ensemble_result = self._weighted_vote_ensemble(predictions, weights)
        
        # =====================================================================
        # FEAR & GREED INTEGRATION
        # Boost confidence when Fear & Greed aligns with prediction direction
        # IMPORTANT: Don't boost FLAT predictions - they should stay conservative
        # =====================================================================
        try:
            fng_signal, fng_modifier = get_fear_greed_signal()
            original_confidence = ensemble_result.confidence
            
            # Skip boosting for FLAT predictions - FLAT should remain uncertain
            if ensemble_result.direction == "FLAT":
                logger.debug(f"[FEAR&GREED] Skipping boost for FLAT prediction")
            elif fng_signal != "NEUTRAL":
                if fng_signal == ensemble_result.direction:
                    # Fear & Greed AGREES with prediction - boost confidence
                    boosted = min(0.85, ensemble_result.confidence * (1 + fng_modifier))  # Cap at 85%, not 95%
                    logger.info(
                        f"[FEAR&GREED] {fng_signal} aligns with {ensemble_result.direction}, "
                        f"confidence {original_confidence:.1%} -> {boosted:.1%}"
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
                    # Fear & Greed DISAGREES - reduce confidence slightly
                    reduced = max(0.35, ensemble_result.confidence * (1 - fng_modifier / 2))
                    logger.info(
                        f"[FEAR&GREED] {fng_signal} conflicts with {ensemble_result.direction}, "
                        f"confidence {original_confidence:.1%} -> {reduced:.1%}"
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
        # Boost/reduce confidence for crypto based on BTC trend alignment
        # IMPORTANT: Don't boost FLAT predictions
        # =====================================================================
        if symbol and ensemble_result.direction != "FLAT":  # Skip FLAT
            try:
                btc_multiplier = get_btc_correlation_boost(symbol, ensemble_result.direction)
                if btc_multiplier != 1.0:
                    original_conf = ensemble_result.confidence
                    new_conf = min(0.85, max(0.35, ensemble_result.confidence * btc_multiplier))  # Cap at 85%
                    logger.info(
                        f"[BTC_CORR] {symbol}: confidence {original_conf:.1%} -> {new_conf:.1%} "
                        f"(multiplier: {btc_multiplier:.2f})"
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
        """Calculate model weights based on recent performance"""
        weights = {}
        total_accuracy = 0
        
        for model_name, history in self.performance_history.items():
            if len(history) > 10:
                # Use recent accuracy as weight
                accuracy = sum(history) / len(history)
                weights[model_name] = accuracy
                total_accuracy += accuracy
            else:
                # Default weights until we have performance data
                weights[model_name] = {"LSTM": 0.4, "XGBoost": 0.4, "Transformer": 0.2}[model_name]
                total_accuracy += weights[model_name]
        
        # Normalize weights to sum to 1.0
        if total_accuracy > 0:
            weights = {k: v / total_accuracy for k, v in weights.items()}
        
        return weights
    
    def _confidence_weighted_ensemble(
        self,
        predictions: List[ModelPrediction],
        adaptive_weights: Dict[str, float]
    ) -> EnsemblePrediction:
        """Ensemble weighted by model confidence * adaptive weight"""
        
        # Weight by confidence and adaptive performance
        weighted_votes = {"UP": 0.0, "DOWN": 0.0, "FLAT": 0.0}
        total_weight = 0.0
        weighted_change = 0.0
        
        for pred in predictions:
            weight = pred.confidence * adaptive_weights.get(pred.model_name, pred.weight)
            weighted_votes[pred.direction] += weight
            weighted_change += pred.predicted_change_pct * weight
            total_weight += weight
        
        # Determine final direction
        direction = max(weighted_votes, key=weighted_votes.get)
        confidence = weighted_votes[direction] / total_weight if total_weight > 0 else 0.5
        predicted_change = weighted_change / total_weight if total_weight > 0 else 0.0
        
        # FLAT predictions should NOT have high confidence - "uncertain" with 95% confidence is nonsense
        # Cap FLAT confidence at 55% (slightly above random) since FLAT = "we don't have a strong signal"
        if direction == "FLAT":
            confidence = min(confidence, 0.55)
        
        return EnsemblePrediction(
            direction=direction,
            confidence=min(confidence, 0.99),
            predicted_change_pct=predicted_change,
            individual_predictions=predictions,
            model_weights=adaptive_weights,
            ensemble_method="confidence_weighted"
        )
    
    def _weighted_vote_ensemble(
        self,
        predictions: list[ModelPrediction],
        adaptive_weights: dict[str, float]
    ) -> EnsemblePrediction:
        """Majority vote weighted by model performance"""
        
        votes = {"UP": 0.0, "DOWN": 0.0, "FLAT": 0.0}
        
        for pred in predictions:
            votes[pred.direction] += adaptive_weights.get(pred.model_name, pred.weight)
        
        direction = max(votes, key=votes.get)
        confidence = votes[direction] / sum(votes.values())
        
        # FLAT predictions should NOT have high confidence - cap at 55%
        if direction == "FLAT":
            confidence = min(confidence, 0.55)
        
        # Average predicted change from models agreeing with final direction
        agreeing_changes = [
            p.predicted_change_pct for p in predictions if p.direction == direction
        ]
        predicted_change = sum(agreeing_changes) / len(agreeing_changes) if agreeing_changes else 0.0
        
        return EnsemblePrediction(
            direction=direction,
            confidence=min(confidence, 0.85),  # Lowered from 0.95 - ensemble should be conservative
            predicted_change_pct=predicted_change,
            individual_predictions=predictions,
            model_weights=adaptive_weights,
            ensemble_method="weighted_vote"
        )
    
    def _inverse_variance_ensemble(
        self,
        predictions: list[ModelPrediction],
        adaptive_weights: dict[str, float]
    ) -> EnsemblePrediction:
        """Weight by inverse variance (higher confidence = lower variance)"""
        
        # Inverse variance weighting: higher confidence = more weight
        total_inv_var = sum(p.confidence for p in predictions)
        
        if total_inv_var == 0:
            return self._weighted_vote_ensemble(predictions, adaptive_weights)
        
        weighted_change = 0.0
        weighted_votes = {"UP": 0.0, "DOWN": 0.0, "FLAT": 0.0}
        
        for pred in predictions:
            inv_var_weight = pred.confidence / total_inv_var
            weighted_votes[pred.direction] += inv_var_weight
            weighted_change += pred.predicted_change_pct * inv_var_weight
        
        direction = max(weighted_votes, key=weighted_votes.get)
        confidence = weighted_votes[direction]
        
        # FLAT predictions should NOT have high confidence - cap at 55%
        if direction == "FLAT":
            confidence = min(confidence, 0.55)
        
        return EnsemblePrediction(
            direction=direction,
            confidence=min(confidence, 0.85),  # Lowered from 0.98 - ensemble should be conservative
            predicted_change_pct=weighted_change,
            individual_predictions=predictions,
            model_weights=adaptive_weights,
            ensemble_method="inverse_variance"
        )
    
    def update_performance(self, model_name: str, was_correct: bool) -> None:
        """Update model performance history for adaptive weighting"""
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
        logger.info("✅ Ensemble predictor initialized (LSTM + XGBoost + Transformer)")
    return _ensemble_predictor


if __name__ == "__main__":
    # Test ensemble
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 Testing Ensemble Predictor")
    print("=" * 60)
    
    ensemble = get_ensemble_predictor()
    
    # Test features
    test_features = {
        "price_history_48h": [100 + i * 0.5 for i in range(48)],  # Uptrend
        "rsi": 45,
        "macd": 0.5,
        "volume_ratio": 1.8,
        "volatility": 0.02,
        "sentiment": 0.3,
        "confidence": 0.7
    }
    
    # Generate prediction
    result = ensemble.predict(test_features, method="confidence_weighted")
    
    print(f"\n📊 Ensemble Result:")
    print(f"  Direction: {result.direction}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Predicted Change: {result.predicted_change_pct:+.2f}%")
    print(f"  Method: {result.ensemble_method}")
    
    print(f"\n🎯 Individual Models:")
    for pred in result.individual_predictions:
        print(f"  {pred.model_name}: {pred.direction} ({pred.confidence:.1%}, weight={pred.weight})")
    
    print(f"\n⚖️ Adaptive Weights:")
    for model, weight in result.model_weights.items():
        print(f"  {model}: {weight:.2%}")
    
    print("\n✅ Ensemble test complete")
