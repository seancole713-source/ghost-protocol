"""
APEX-Style Trade Cards - Explainability First
Every trade includes a one-screen rationale with:
- Top 5 features
- Comparable pasts (historical analogs)
- Expected path (forecast)
- Fail conditions (stop-loss triggers)
"""

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class TradeCard:
    """
    Self-contained trade explanation card.
    Designed to fit on one screen for quick review.
    """

    # Core decision
    action: str  # BUY/SELL/HOLD
    symbol: str
    confidence: float  # 0-100
    timestamp: int

    # Top 5 features (most influential)
    top_features: list[dict[str, Any]]  # [{name, value, weight, impact}]

    # Historical analogs (similar past situations)
    analogs: list[dict[str, Any]]  # [{date, price, outcome, similarity}]

    # Expected path
    expected_return_1d: float
    expected_return_7d: float
    expected_return_30d: float
    price_target: float
    confidence_band: tuple  # (low, high)

    # Fail conditions (when to exit)
    stop_loss_price: float
    stop_loss_reason: str
    invalidation_signals: list[str]

    # Risk metrics
    var_95: float  # Value at Risk
    max_loss_estimate: float
    win_probability: float

    # Rationale summary
    rationale: str
    risks: list[str]
    catalysts: list[str]


class TradeCardGenerator:
    """Generate APEX-style trade cards for explainability."""

    def __init__(self):
        self.feature_weights = {
            "momentum": 0.25,
            "sentiment": 0.20,
            "technical": 0.20,
            "volume": 0.15,
            "volatility": 0.10,
            "macro": 0.10,
        }

    def generate_card(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price_data: pd.DataFrame,
        news_sentiment: float | None,
        forecast_data: dict,
    ) -> TradeCard:
        """
        Generate comprehensive trade card.

        Args:
            symbol: Trading symbol
            action: BUY/SELL/HOLD
            confidence: 0-100
            price_data: Historical price DataFrame
            news_sentiment: Current sentiment score
            forecast_data: Forecast output

        Returns:
            TradeCard with full explainability
        """

        current_price = float(price_data["close"].iloc[-1])

        # 1. Calculate top 5 features
        top_features = self._calculate_top_features(price_data, news_sentiment, forecast_data)

        # 2. Find historical analogs
        analogs = self._find_analogs(price_data, news_sentiment)

        # 3. Build expected path
        expected_return_1d = forecast_data.get("return_1d", 0.0)
        expected_return_7d = forecast_data.get("return_7d", 0.0)
        expected_return_30d = forecast_data.get("return_30d", 0.0)
        price_target = current_price * (1 + expected_return_7d)

        # Confidence band (±2 std)
        volatility = float(price_data["close"].pct_change().std())
        conf_low = price_target * (1 - volatility * 2)
        conf_high = price_target * (1 + volatility * 2)

        # 4. Calculate fail conditions
        stop_loss_price, stop_loss_reason = self._calculate_stop_loss(
            current_price, action, price_data
        )
        invalidation_signals = self._get_invalidation_signals(action, top_features)

        # 5. Risk metrics
        returns = price_data["close"].pct_change().dropna()
        var_95 = float(returns.quantile(0.05)) * current_price * 100  # Assume 100 shares
        max_loss_estimate = abs(current_price - stop_loss_price) * 100
        win_probability = self._estimate_win_probability(confidence, top_features)

        # 6. Rationale and risks
        rationale = self._build_rationale(action, top_features, analogs)
        risks = self._identify_risks(action, top_features, price_data)
        catalysts = self._identify_catalysts(action, news_sentiment, forecast_data)

        return TradeCard(
            action=action,
            symbol=symbol,
            confidence=confidence,
            timestamp=int(time.time()),
            top_features=top_features,
            analogs=analogs,
            expected_return_1d=expected_return_1d,
            expected_return_7d=expected_return_7d,
            expected_return_30d=expected_return_30d,
            price_target=round(price_target, 2),
            confidence_band=(round(conf_low, 2), round(conf_high, 2)),
            stop_loss_price=round(stop_loss_price, 2),
            stop_loss_reason=stop_loss_reason,
            invalidation_signals=invalidation_signals,
            var_95=round(var_95, 2),
            max_loss_estimate=round(max_loss_estimate, 2),
            win_probability=round(win_probability, 2),
            rationale=rationale,
            risks=risks,
            catalysts=catalysts,
        )

    def _calculate_top_features(
        self, price_data: pd.DataFrame, news_sentiment: float | None, forecast_data: dict
    ) -> list[dict]:
        """Calculate and rank top 5 features by impact."""

        features = []

        # 1. Momentum
        if len(price_data) >= 20:
            mom_20 = (price_data["close"].iloc[-1] / price_data["close"].iloc[-20]) - 1
            features.append(
                {
                    "name": "Momentum (20d)",
                    "value": f"{mom_20:+.2%}",  # string formatted value for display
                    "numeric_value": float(mom_20),  # raw numeric for UI bars / gauges
                    "weight": self.feature_weights["momentum"],
                    "impact": abs(mom_20) * self.feature_weights["momentum"],
                    "direction": "bullish" if mom_20 > 0 else "bearish",
                }
            )

        # 2. Sentiment
        if news_sentiment is not None:
            features.append(
                {
                    "name": "News Sentiment",
                    "value": f"{news_sentiment:+.2f}",
                    "numeric_value": float(news_sentiment),
                    "weight": self.feature_weights["sentiment"],
                    "impact": abs(news_sentiment) * self.feature_weights["sentiment"],
                    "direction": "bullish" if news_sentiment > 0 else "bearish",
                }
            )

        # 3. RSI (Technical)
        if len(price_data) >= 14:
            returns = pd.to_numeric(price_data["close"], errors="coerce").diff()
            gains = returns[returns > 0].tail(14).mean()  # type: ignore
            losses = -returns[returns < 0].tail(14).mean()  # type: ignore
            rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 50

            # RSI interpretation: <30 oversold (bullish), >70 overbought (bearish)
            rsi_signal = (rsi - 50) / 50  # Normalize to -1 to +1

            features.append(
                {
                    "name": "RSI (14)",
                    "value": f"{rsi:.1f}",
                    "numeric_value": float(rsi),
                    "weight": self.feature_weights["technical"],
                    "impact": abs(rsi_signal) * self.feature_weights["technical"],
                    "direction": "bullish" if rsi < 40 else ("bearish" if rsi > 60 else "neutral"),
                }
            )

        # 4. Volume
        if "volume" in price_data.columns and len(price_data) >= 20:
            recent_vol = price_data["volume"].iloc[-5:].mean()
            avg_vol = price_data["volume"].iloc[-20:].mean()
            vol_ratio = (recent_vol / avg_vol) if avg_vol > 0 else 1.0

            features.append(
                {
                    "name": "Volume Surge",
                    "value": f"{vol_ratio:.2f}x",
                    "numeric_value": float(vol_ratio),
                    "weight": self.feature_weights["volume"],
                    "impact": abs(vol_ratio - 1) * self.feature_weights["volume"],
                    "direction": "bullish" if vol_ratio > 1.2 else "neutral",
                }
            )

        # 5. Volatility
        if len(price_data) >= 20:
            returns = price_data["close"].pct_change()
            volatility = returns.tail(20).std() * (252**0.5)  # Annualized

            features.append(
                {
                    "name": "Volatility (20d)",
                    "value": f"{volatility:.1%}",
                    "numeric_value": float(volatility),
                    "weight": self.feature_weights["volatility"],
                    "impact": volatility * self.feature_weights["volatility"],
                    "direction": "high"
                    if volatility > 0.3
                    else ("low" if volatility < 0.15 else "normal"),
                }
            )

        # Sort by impact and return top 5
        features.sort(key=lambda x: x["impact"], reverse=True)
        return features[:5]

    def _find_analogs(self, price_data: pd.DataFrame, news_sentiment: float | None) -> list[dict]:
        """Find historical situations similar to current state."""

        analogs = []

        if len(price_data) < 60:
            return analogs

        # Current state
        current_price = float(price_data["close"].iloc[-1])
        current_returns = price_data["close"].pct_change().tail(20)
        current_vol = float(current_returns.std())
        current_mom = (current_price / price_data["close"].iloc[-20]) - 1

        # Scan history for similar patterns (simplified)
        for i in range(60, len(price_data) - 20, 5):  # Sample every 5 days
            hist_price = float(price_data["close"].iloc[i])
            hist_returns = price_data["close"].pct_change().iloc[i - 20 : i]
            hist_vol = float(hist_returns.std())
            hist_mom = (hist_price / price_data["close"].iloc[i - 20]) - 1

            # Calculate similarity (inverse of distance)
            vol_diff = abs(current_vol - hist_vol)
            mom_diff = abs(current_mom - hist_mom)
            similarity = 1.0 / (1.0 + vol_diff * 10 + mom_diff * 5)

            # Only keep if similarity > 0.5
            if similarity > 0.5:
                # Check outcome 7 days later
                if i + 7 < len(price_data):
                    future_price = float(price_data["close"].iloc[i + 7])
                    outcome = (future_price / hist_price) - 1

                    analogs.append(
                        {
                            "date": str(price_data.index[i])[:10],
                            "price": round(hist_price, 2),
                            "outcome_7d": f"{outcome:+.2%}",
                            "similarity": round(similarity, 2),
                        }
                    )

        # Sort by similarity and return top 3
        analogs.sort(key=lambda x: x["similarity"], reverse=True)
        return analogs[:3]

    def _calculate_stop_loss(
        self, current_price: float, action: str, price_data: pd.DataFrame
    ) -> tuple:
        """Calculate stop-loss price and reason."""

        # ATR-based stop loss (2x ATR)
        if "high" in price_data.columns and "low" in price_data.columns and len(price_data) >= 14:
            tr = pd.DataFrame(
                {
                    "hl": price_data["high"] - price_data["low"],
                    "hc": abs(price_data["high"] - price_data["close"].shift()),
                    "lc": abs(price_data["low"] - price_data["close"].shift()),
                }
            )
            atr = float(tr.max(axis=1).tail(14).mean())

            if action == "BUY":
                stop_loss = current_price - (atr * 2)
                reason = f"2× ATR (${atr:.2f}) below entry"
            else:  # SELL
                stop_loss = current_price + (atr * 2)
                reason = f"2× ATR (${atr:.2f}) above entry"
        else:
            # Fallback: 5% stop
            if action == "BUY":
                stop_loss = current_price * 0.95
                reason = "5% stop-loss below entry"
            else:
                stop_loss = current_price * 1.05
                reason = "5% stop-loss above entry"

        return stop_loss, reason

    def _get_invalidation_signals(self, action: str, top_features: list[dict]) -> list[str]:
        """Identify signals that would invalidate the trade thesis."""

        signals = []

        for feature in top_features[:3]:  # Top 3 features
            name = feature["name"]
            direction = feature["direction"]

            if action == "BUY":
                if "Momentum" in name and direction == "bullish":
                    signals.append("Momentum reversal: 20d return turns negative")
                if "Sentiment" in name and direction == "bullish":
                    signals.append("Sentiment reversal: News score < -0.5")
                if "RSI" in name and direction == "bullish":
                    signals.append("RSI exits oversold: RSI > 50")
            elif action == "SELL":
                if "Momentum" in name and direction == "bearish":
                    signals.append("Momentum reversal: 20d return turns positive")
                if "Sentiment" in name and direction == "bearish":
                    signals.append("Sentiment reversal: News score > +0.5")

        # Always add volume condition
        signals.append("Volume collapse: 5d avg < 0.5× 20d avg")

        return signals

    def _estimate_win_probability(self, confidence: float, top_features: list[dict]) -> float:
        """Estimate probability of profitable trade."""

        # Base probability from confidence
        base_prob = confidence / 100

        # Adjust based on feature agreement
        if len(top_features) >= 3:
            directions = [f["direction"] for f in top_features[:3]]
            bullish_count = sum(1 for d in directions if d == "bullish")
            bearish_count = sum(1 for d in directions if d == "bearish")

            # High agreement boosts probability
            if bullish_count >= 2 or bearish_count >= 2:
                base_prob = min(0.85, base_prob * 1.1)
            else:
                base_prob = max(0.45, base_prob * 0.9)

        return base_prob

    def _build_rationale(self, action: str, top_features: list[dict], analogs: list[dict]) -> str:
        """Build human-readable rationale."""

        # Primary drivers
        drivers = [f for f in top_features if f["impact"] > 0.1]

        if not drivers:
            return f"{action} signal weak - low conviction"

        rationale_parts = [f"{action} based on:"]

        for driver in drivers[:3]:
            rationale_parts.append(f"• {driver['name']}: {driver['value']} ({driver['direction']})")

        # Add analog context
        if analogs:
            outcomes = [a["outcome_7d"] for a in analogs]
            rationale_parts.append(f"• Similar past situations: {', '.join(outcomes)}")

        return " ".join(rationale_parts)

    def _identify_risks(
        self, action: str, top_features: list[dict], price_data: pd.DataFrame
    ) -> list[str]:
        """Identify key risks to the trade."""

        risks = []

        # Check for high volatility
        if len(price_data) >= 20:
            returns = price_data["close"].pct_change()
            vol = float(returns.tail(20).std() * (252**0.5))
            if vol > 0.4:
                risks.append(f"High volatility ({vol:.1%} annualized) - wide price swings likely")

        # Check for low liquidity
        if "volume" in price_data.columns:
            avg_vol = float(price_data["volume"].tail(20).mean())
            if avg_vol < 1000000:  # < 1M shares/day
                risks.append(f"Low liquidity (avg {avg_vol / 1e6:.1f}M/day) - slippage risk")

        # Check for conflicting signals
        directions = [f["direction"] for f in top_features]
        if "bullish" in directions and "bearish" in directions:
            risks.append("Mixed signals - trade thesis less certain")

        # Macro risk (simplified)
        risks.append("Macro: Market-wide correction could override technicals")

        return risks[:4]  # Top 4 risks

    def _identify_catalysts(
        self, action: str, news_sentiment: float | None, forecast_data: dict
    ) -> list[str]:
        """Identify potential catalysts that could accelerate returns."""

        catalysts = []

        if news_sentiment is not None and abs(news_sentiment) > 0.3:
            if news_sentiment > 0:
                catalysts.append("Positive news flow - sentiment momentum")
            else:
                catalysts.append("Negative news - reversion opportunity")

        # Technical catalysts
        catalysts.append("Breakout above 20d SMA would confirm trend")

        # Fundamental catalysts
        catalysts.append("Earnings report (check calendar) - volatility catalyst")

        return catalysts[:3]
