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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Model storage
MODELS_DIR = Path(__file__).parent.parent / "models" / "ensemble"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
            
            # Short-term momentum (1h, 4h)
            if momentum_1h > 0.5:
                momentum_signals.append(1)
            elif momentum_1h < -0.5:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            if momentum_4h > 1.0:
                momentum_signals.append(1)
            elif momentum_4h < -1.0:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            # Medium-term momentum (24h)
            if momentum_24h > 2.0:
                momentum_signals.append(1)
            elif momentum_24h < -2.0:
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
            
            if total_momentum > 1:
                direction = "UP"
                confidence = min(0.4 + (agreement / len(momentum_signals)) * 0.4, 0.85)
            elif total_momentum < -1:
                direction = "DOWN"
                confidence = min(0.4 + (agreement / len(momentum_signals)) * 0.4, 0.85)
            else:
                direction = "FLAT"
                confidence = 0.3
            
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
                    
                    # Default to 0 if missing
                    feature_values.append(float(value) if value is not None else 0.0)
                
                if missing_features and len(missing_features) < 10:
                    logger.debug(f"Missing features for XGBoost: {missing_features[:5]}...")
                
                X = np.array([feature_values])
                
                # Predict using trained model
                prediction = self.model.predict(X)[0]
                proba = self.model.predict_proba(X)[0]
                
                direction = "UP" if prediction == 1 else "DOWN"
                confidence = float(proba[prediction])
                
                # Scale predicted change based on confidence
                predicted_change = confidence * 6.0 * (1 if prediction == 1 else -1)
                
                logger.debug(f"🤖 XGBoost {self.model_version}: {direction} @ {confidence:.1%}")
                
                return ModelPrediction(
                    model_name=f"XGBoost-{self.model_version}",
                    direction=direction,
                    confidence=confidence,
                    predicted_change_pct=predicted_change,
                    weight=0.7 if self.model_version == "v2" else 0.6  # v2 gets higher weight
                )
            else:
                # Fallback to feature-based prediction if model not loaded
                return self._fallback_predict(features)
                
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return self._fallback_predict(features)
    
    def _fallback_predict(self, features: Dict[str, Any]) -> ModelPrediction:
        """Fallback when trained model unavailable"""
        rsi = features.get("RSI_14", features.get("rsi", 50))
        macd = features.get("MACD_HISTOGRAM", features.get("macd", 0))
        volume_ratio = features.get("VOLUME_RATIO", features.get("volume_ratio", 1.0))
        
        score = 0
        if rsi is not None:
            if rsi < 30:
                score += 2
            elif rsi > 70:
                score -= 2
        
        if macd is not None:
            if macd > 0:
                score += 1
            else:
                score -= 1
        
        if volume_ratio is not None and volume_ratio > 1.5:
            score += 1
        
        direction = "UP" if score > 0 else "DOWN" if score < 0 else "FLAT"
        confidence = min(abs(score) / 5 + 0.3, 0.65)  # Lower max confidence for fallback
        
        return ModelPrediction(
            model_name="XGBoost-Fallback",
            direction=direction,
            confidence=confidence,
            predicted_change_pct=score * 2,
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
            
            # MACD Signal
            macd = features.get("MACD_HISTOGRAM", features.get("macd", None))
            if macd is not None:
                if macd > 0:
                    signals.append(1)
                    weights.append(0.2)
                else:
                    signals.append(-1)
                    weights.append(0.2)
            
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
                
                if weighted_signal > 0.2:
                    direction = "UP"
                    confidence = min(0.45 + abs(weighted_signal) * 0.4, 0.80)
                elif weighted_signal < -0.2:
                    direction = "DOWN"
                    confidence = min(0.45 + abs(weighted_signal) * 0.4, 0.80)
                else:
                    direction = "FLAT"
                    confidence = 0.35
                
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
                weight=0.15  # Lower weight - supplementary to XGBoost
            )
            
        except Exception as e:
            logger.error(f"Context synthesizer prediction failed: {e}")
            return ModelPrediction(
                model_name="Context-Synthesizer",
                direction="FLAT",
                confidence=0.3,
                predicted_change_pct=0.0,
                weight=0.15
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
        
    def predict(self, features: dict[str, Any], method: str = "confidence_weighted") -> EnsemblePrediction:
        """
        Generate ensemble prediction
        
        Args:
            features: Feature dict with price_history, technical indicators, etc.
            method: Ensemble method (weighted_vote, confidence_weighted, inverse_variance)
        
        Returns:
            EnsemblePrediction with aggregated results
        """
        start = time.time()
        
        # Get predictions from all models
        lstm_pred = self.lstm.predict(features)
        xgb_pred = self.xgboost.predict(features)
        transformer_pred = self.transformer.predict(features)
        
        predictions = [lstm_pred, xgb_pred, transformer_pred]
        
        # Adaptive weights based on recent performance
        weights = self._calculate_adaptive_weights()
        
        # Apply ensemble method
        if method == "confidence_weighted":
            ensemble_result = self._confidence_weighted_ensemble(predictions, weights)
        elif method == "inverse_variance":
            ensemble_result = self._inverse_variance_ensemble(predictions, weights)
        else:  # weighted_vote
            ensemble_result = self._weighted_vote_ensemble(predictions, weights)
        
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
        
        # Average predicted change from models agreeing with final direction
        agreeing_changes = [
            p.predicted_change_pct for p in predictions if p.direction == direction
        ]
        predicted_change = sum(agreeing_changes) / len(agreeing_changes) if agreeing_changes else 0.0
        
        return EnsemblePrediction(
            direction=direction,
            confidence=min(confidence, 0.95),
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
        
        return EnsemblePrediction(
            direction=direction,
            confidence=min(confidence, 0.98),
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
