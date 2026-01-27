#!/usr/bin/env python3
"""
Ghost Brain Intelligence Enhancement
Multi-factor reasoning engine with context awareness
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class MarketRegime(Enum):
    """Detected market regime."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class TradingAction(Enum):
    """Enhanced trading actions."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """Trading signal with reasoning."""

    name: str
    value: float  # -1 to +1, where +1 is bullish, -1 is bearish
    weight: float  # 0 to 1, importance of this signal
    reasoning: str

    @property
    def weighted_value(self) -> float:
        """Get weighted signal value."""
        return self.value * self.weight


@dataclass
class GhostDecision:
    """Enhanced Ghost trading decision."""

    action: TradingAction
    confidence: float  # 0 to 100
    reasoning: list[str]
    signals: list[TradingSignal]
    risk_score: float  # 0 to 100, higher = riskier
    market_regime: MarketRegime
    factors_analyzed: int
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "action": self.action.value,
            "confidence": round(self.confidence, 1),
            "reasoning": self.reasoning,
            "risk_score": round(self.risk_score, 1),
            "market_regime": self.market_regime.value,
            "factors_analyzed": self.factors_analyzed,
            "timestamp": self.timestamp.isoformat(),
            "signals": [
                {
                    "name": s.name,
                    "value": round(s.value, 3),
                    "weight": round(s.weight, 2),
                    "contribution": round(s.weighted_value, 3),
                    "reasoning": s.reasoning,
                }
                for s in self.signals
            ],
        }


class GhostBrain:
    """
    Enhanced Ghost Intelligence Engine
    Multi-factor analysis with context awareness
    """

    def __init__(self):
        self.decision_history: list[GhostDecision] = []

    def analyze(
        self,
        current_price: float,
        prev_close: float,
        portfolio_avg_cost: float,
        portfolio_qty: float,
        news_sentiment: float | None = None,
        forecast_confidence: float | None = None,
        forecast_direction: str | None = None,
        volatility: float | None = None,
        volume_ratio: float | None = None,
    ) -> GhostDecision:
        """
        Perform comprehensive market analysis.

        Args:
            current_price: Current market price
            prev_close: Previous closing price
            portfolio_avg_cost: Average cost basis
            portfolio_qty: Current position size
            news_sentiment: News sentiment score (-1 to +1)
            forecast_confidence: Forecast model confidence (0 to 1)
            forecast_direction: Forecast direction (up/down/neutral)
            volatility: Price volatility metric
            volume_ratio: Volume vs average

        Returns:
            GhostDecision with action, confidence, and reasoning
        """
        signals: list[TradingSignal] = []

        # 1. Momentum Analysis
        momentum_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        momentum_signal = self._analyze_momentum(momentum_pct, current_price, prev_close)
        signals.append(momentum_signal)

        # 2. Position Analysis (distance from avg cost)
        position_signal = self._analyze_position(current_price, portfolio_avg_cost, portfolio_qty)
        signals.append(position_signal)

        # 3. News Sentiment
        if news_sentiment is not None:
            sentiment_signal = self._analyze_sentiment(news_sentiment)
            signals.append(sentiment_signal)

        # 4. Forecast Analysis
        if forecast_confidence is not None and forecast_direction:
            forecast_signal = self._analyze_forecast(forecast_confidence, forecast_direction)
            signals.append(forecast_signal)

        # 5. Volatility Analysis
        if volatility is not None:
            vol_signal = self._analyze_volatility(volatility)
            signals.append(vol_signal)

        # 6. Volume Analysis
        if volume_ratio is not None:
            volume_signal = self._analyze_volume(volume_ratio)
            signals.append(volume_signal)

        # Detect market regime
        regime = self._detect_market_regime(momentum_pct, volatility, signals)

        # Aggregate signals
        total_signal = sum(s.weighted_value for s in signals)
        total_weight = sum(s.weight for s in signals)
        normalized_signal = total_signal / total_weight if total_weight > 0 else 0

        # Determine action
        action = self._signal_to_action(normalized_signal, regime)

        # Calculate confidence (based on signal strength and agreement)
        confidence = self._calculate_confidence(signals, regime)

        # Calculate risk score
        risk_score = self._calculate_risk(normalized_signal, volatility or 0, portfolio_qty, regime)

        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(signals, regime, action)

        decision = GhostDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            signals=signals,
            risk_score=risk_score,
            market_regime=regime,
            factors_analyzed=len(signals),
            timestamp=datetime.now(UTC),
        )

        self.decision_history.append(decision)
        return decision

    def _analyze_momentum(self, momentum_pct: float, current: float, prev: float) -> TradingSignal:
        """Analyze price momentum."""
        if abs(momentum_pct) < 0.5:
            signal_value = 0.0
            reasoning = f"Neutral momentum ({momentum_pct:+.2f}%)"
        elif momentum_pct > 2:
            signal_value = 0.8
            reasoning = f"Strong upward momentum ({momentum_pct:+.2f}%)"
        elif momentum_pct > 0:
            signal_value = 0.4
            reasoning = f"Positive momentum ({momentum_pct:+.2f}%)"
        elif momentum_pct < -2:
            signal_value = -0.8
            reasoning = f"Strong downward momentum ({momentum_pct:+.2f}%)"
        else:
            signal_value = -0.4
            reasoning = f"Negative momentum ({momentum_pct:+.2f}%)"

        return TradingSignal(
            name="Momentum",
            value=signal_value,
            weight=0.25,  # 25% weight
            reasoning=reasoning,
        )

    def _analyze_position(self, current: float, avg_cost: float, qty: float) -> TradingSignal:
        """Analyze current position relative to cost basis."""
        if qty == 0:
            return TradingSignal(
                name="Position", value=0.0, weight=0.1, reasoning="No position held"
            )

        pnl_pct = ((current - avg_cost) / avg_cost) * 100 if avg_cost > 0 else 0

        if pnl_pct < -10:
            # Deep underwater - consider averaging down
            signal_value = 0.3
            reasoning = f"Deep loss ({pnl_pct:.1f}%), potential avg-down opportunity"
        elif pnl_pct < -5:
            signal_value = 0.0
            reasoning = f"Moderate loss ({pnl_pct:.1f}%), hold and monitor"
        elif pnl_pct < 5:
            signal_value = 0.0
            reasoning = f"Near breakeven ({pnl_pct:.1f}%)"
        elif pnl_pct < 15:
            signal_value = -0.2
            reasoning = f"Profit zone ({pnl_pct:.1f}%), consider taking partial profits"
        else:
            signal_value = -0.5
            reasoning = f"Strong gains ({pnl_pct:.1f}%), consider profit-taking"

        return TradingSignal(
            name="Position P&L", value=signal_value, weight=0.2, reasoning=reasoning
        )

    def _analyze_sentiment(self, sentiment: float) -> TradingSignal:
        """Analyze news sentiment."""
        if sentiment > 0.3:
            reasoning = "Bullish news sentiment"
            value = 0.6
        elif sentiment > 0:
            reasoning = "Slightly positive news"
            value = 0.2
        elif sentiment < -0.3:
            reasoning = "Bearish news sentiment"
            value = -0.6
        elif sentiment < 0:
            reasoning = "Slightly negative news"
            value = -0.2
        else:
            reasoning = "Neutral news"
            value = 0.0

        return TradingSignal(name="News Sentiment", value=value, weight=0.15, reasoning=reasoning)

    def _analyze_forecast(self, confidence: float, direction: str) -> TradingSignal:
        """Analyze prediction forecast."""
        dir_lower = direction.lower()

        if "up" in dir_lower or "bull" in dir_lower:
            value = confidence * 0.8
            reasoning = f"Forecast predicts upside ({confidence * 100:.0f}% confidence)"
        elif "down" in dir_lower or "bear" in dir_lower:
            value = -confidence * 0.8
            reasoning = f"Forecast predicts downside ({confidence * 100:.0f}% confidence)"
        else:
            value = 0.0
            reasoning = f"Neutral forecast ({confidence * 100:.0f}% confidence)"

        return TradingSignal(name="Forecast", value=value, weight=0.2, reasoning=reasoning)

    def _analyze_volatility(self, volatility: float) -> TradingSignal:
        """Analyze volatility conditions."""
        if volatility > 0.3:
            reasoning = "High volatility - reduce risk"
            value = -0.3
            weight = 0.15
        elif volatility > 0.15:
            reasoning = "Elevated volatility - be cautious"
            value = -0.1
            weight = 0.1
        else:
            reasoning = "Low volatility - stable conditions"
            value = 0.1
            weight = 0.05

        return TradingSignal(name="Volatility", value=value, weight=weight, reasoning=reasoning)

    def _analyze_volume(self, volume_ratio: float) -> TradingSignal:
        """Analyze volume patterns."""
        if volume_ratio > 2.0:
            reasoning = "Exceptional volume - strong conviction"
            value = 0.3
        elif volume_ratio > 1.5:
            reasoning = "Above-average volume - increased interest"
            value = 0.2
        elif volume_ratio < 0.5:
            reasoning = "Low volume - weak conviction"
            value = -0.1
        else:
            reasoning = "Normal volume"
            value = 0.0

        return TradingSignal(name="Volume", value=value, weight=0.1, reasoning=reasoning)

    def _detect_market_regime(
        self, momentum: float, volatility: float | None, signals: list[TradingSignal]
    ) -> MarketRegime:
        """Detect current market regime."""
        if volatility and volatility > 0.3:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility and volatility < 0.1:
            return MarketRegime.LOW_VOLATILITY

        # Look at overall signal direction
        total = sum(s.weighted_value for s in signals)

        if total > 0.3 and momentum > 1:
            return MarketRegime.BULL_TREND
        elif total < -0.3 and momentum < -1:
            return MarketRegime.BEAR_TREND
        elif abs(momentum) < 0.5:
            return MarketRegime.SIDEWAYS
        else:
            return MarketRegime.UNKNOWN

    def _signal_to_action(self, signal: float, regime: MarketRegime) -> TradingAction:
        """Convert aggregate signal to trading action."""
        # Adjust thresholds based on market regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            # More conservative in volatile markets
            buy_threshold = 0.4
            strong_buy_threshold = 0.7
            sell_threshold = -0.4
            strong_sell_threshold = -0.7
        else:
            buy_threshold = 0.3
            strong_buy_threshold = 0.6
            sell_threshold = -0.3
            strong_sell_threshold = -0.6

        if signal >= strong_buy_threshold:
            return TradingAction.STRONG_BUY
        elif signal >= buy_threshold:
            return TradingAction.BUY
        elif signal <= strong_sell_threshold:
            return TradingAction.STRONG_SELL
        elif signal <= sell_threshold:
            return TradingAction.SELL
        else:
            return TradingAction.HOLD

    def _calculate_confidence(self, signals: list[TradingSignal], regime: MarketRegime) -> float:
        """Calculate decision confidence."""
        if not signals:
            return 0.0

        # Check signal agreement (are they pointing same direction?)
        signal_values = [s.weighted_value for s in signals]
        positive_count = sum(1 for v in signal_values if v > 0.1)
        negative_count = sum(1 for v in signal_values if v < -0.1)
        total_count = len(signal_values)

        agreement_ratio = max(positive_count, negative_count) / total_count

        # Base confidence on signal strength and agreement
        avg_strength = sum(abs(v) for v in signal_values) / len(signal_values)

        confidence = (agreement_ratio * 50) + (avg_strength * 50)

        # Reduce confidence in uncertain regimes
        if regime == MarketRegime.HIGH_VOLATILITY:
            confidence *= 0.8
        elif regime == MarketRegime.UNKNOWN:
            confidence *= 0.7

        return min(100, max(0, confidence))

    def _calculate_risk(
        self, signal: float, volatility: float, position_size: float, regime: MarketRegime
    ) -> float:
        """Calculate risk score for the decision."""
        risk = 0.0

        # Base risk on volatility
        risk += volatility * 40

        # Add risk for extreme signals (high conviction = higher risk)
        risk += abs(signal) * 20

        # Add risk for large positions
        if position_size > 1000:
            risk += 15
        elif position_size > 100:
            risk += 10

        # Regime-specific risk
        if regime == MarketRegime.HIGH_VOLATILITY:
            risk += 20
        elif regime == MarketRegime.BEAR_TREND:
            risk += 15

        return min(100, max(0, risk))

    def _generate_reasoning(
        self, signals: list[TradingSignal], regime: MarketRegime, action: TradingAction
    ) -> list[str]:
        """Generate human-readable reasoning."""
        reasoning = []

        # Add market regime context
        regime_msgs = {
            MarketRegime.BULL_TREND: "Market in bullish trend",
            MarketRegime.BEAR_TREND: "Market in bearish trend",
            MarketRegime.SIDEWAYS: "Market moving sideways",
            MarketRegime.HIGH_VOLATILITY: "High volatility environment",
            MarketRegime.LOW_VOLATILITY: "Low volatility environment",
        }
        if regime in regime_msgs:
            reasoning.append(regime_msgs[regime])

        # Add top 3 strongest signals
        sorted_signals = sorted(signals, key=lambda s: abs(s.weighted_value), reverse=True)
        for sig in sorted_signals[:3]:
            if abs(sig.weighted_value) > 0.05:  # Only include meaningful signals
                reasoning.append(f"{sig.name}: {sig.reasoning}")

        # Add action justification
        action_msgs = {
            TradingAction.STRONG_BUY: "Strong bullish signals align - consider increasing position",
            TradingAction.BUY: "Moderate bullish signals - consider buying",
            TradingAction.HOLD: "Mixed signals or weak conviction - maintain position",
            TradingAction.SELL: "Moderate bearish signals - consider reducing position",
            TradingAction.STRONG_SELL: "Strong bearish signals align - consider exiting position",
        }
        reasoning.append(action_msgs.get(action, "Action based on signal aggregation"))

        return reasoning


# Global instance
_ghost_brain = GhostBrain()


def get_ghost_brain() -> GhostBrain:
    """Get the global Ghost Brain instance."""
    return _ghost_brain
