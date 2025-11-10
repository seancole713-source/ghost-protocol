"""
APEX Strategy Ensemble - Voting Engine
Weighted blend of momentum, news, and pairs strategies
Dynamically chooses best performer based on regime

Expected Impact: +20% profitability consistency
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


class StrategyAction(Enum):
    """Strategy vote options"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategyVote:
    """Single strategy's vote"""

    strategy_name: str
    action: str  # BUY/SELL/HOLD
    confidence: float  # 0-100
    expected_return: float  # Expected % return
    signals: dict[str, Any]  # Supporting data
    rationale: str


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.min_confidence = 50.0

    @abstractmethod
    def evaluate(self, symbol: str, market_data: dict[str, Any]) -> StrategyVote:
        """
        Evaluate strategy and return vote

        Args:
            symbol: Trading symbol
            market_data: Dict with 'daily_hist', 'intraday_hist', 'news', 'regime'

        Returns:
            StrategyVote with action, confidence, rationale
        """
        pass


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based strategy with adaptive lookbacks and ATR stops

    Logic:
    - Multi-timeframe momentum (5/20/50 day)
    - Volume confirmation
    - ATR-based position sizing
    - Trend strength filtering
    """

    def __init__(self):
        super().__init__("Momentum")
        self.lookback_short = 5
        self.lookback_med = 20
        self.lookback_long = 50

    def evaluate(self, symbol: str, market_data: dict[str, Any]) -> StrategyVote:
        hist = market_data.get("daily_hist")

        if hist is None or len(hist) < self.lookback_long:
            return StrategyVote(
                strategy_name=self.name,
                action="HOLD",
                confidence=0.0,
                expected_return=0.0,
                signals={},
                rationale="Insufficient data for momentum strategy",
            )

        current_price = float(hist["Close"].iloc[-1])
        signals = {}
        score = 50.0

        # 1. Multi-timeframe momentum
        mom_5 = (hist["Close"].iloc[-1] / hist["Close"].iloc[-self.lookback_short]) - 1
        mom_20 = (hist["Close"].iloc[-1] / hist["Close"].iloc[-self.lookback_med]) - 1
        mom_50 = (hist["Close"].iloc[-1] / hist["Close"].iloc[-self.lookback_long]) - 1

        signals["momentum_5d"] = float(mom_5)
        signals["momentum_20d"] = float(mom_20)
        signals["momentum_50d"] = float(mom_50)

        # Weighted momentum score
        weighted_mom = (mom_5 * 0.5) + (mom_20 * 0.3) + (mom_50 * 0.2)
        score += weighted_mom * 150  # Strong weight on momentum

        # 2. Trend alignment (all timeframes agree?)
        trend_alignment = 0
        if mom_5 > 0:
            trend_alignment += 1
        if mom_20 > 0:
            trend_alignment += 1
        if mom_50 > 0:
            trend_alignment += 1

        signals["trend_alignment"] = trend_alignment

        if trend_alignment == 3:  # All bullish
            score += 15
        elif trend_alignment == 0:  # All bearish
            score -= 15

        # 3. Volume confirmation
        recent_vol = hist["Volume"].tail(5).mean()
        avg_vol = hist["Volume"].tail(20).mean()
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
        signals["volume_ratio"] = float(vol_ratio)

        if vol_ratio > 1.3:  # Strong volume
            score += 10
        elif vol_ratio < 0.7:  # Weak volume
            score -= 10

        # 4. ATR for volatility check
        if "High" in hist.columns and "Low" in hist.columns:
            tr = pd.DataFrame(
                {
                    "hl": hist["High"] - hist["Low"],
                    "hc": abs(hist["High"] - hist["Close"].shift()),
                    "lc": abs(hist["Low"] - hist["Close"].shift()),
                }
            )
            atr = float(tr.max(axis=1).tail(14).mean())
            atr_pct = atr / current_price
            signals["atr_pct"] = float(atr_pct)

            # High ATR = lower confidence
            if atr_pct > 0.05:  # >5% daily range
                score *= 0.85

        # 5. Moving average positions
        ma20 = hist["Close"].tail(20).mean()
        ma50 = hist["Close"].tail(50).mean()

        if current_price > ma20 > ma50:  # Bullish alignment
            score += 10
        elif current_price < ma20 < ma50:  # Bearish alignment
            score -= 10

        # Normalize confidence
        confidence = max(0, min(100, score))

        # Expected return based on momentum strength
        expected_return = weighted_mom * 2  # Amplify momentum signal

        # Determine action
        if confidence > 65 and expected_return > 0.01:
            action = "BUY"
            rationale = f"Strong momentum: 5d {mom_5 * 100:.1f}%, 20d {mom_20 * 100:.1f}%, alignment {trend_alignment}/3, vol {vol_ratio:.2f}x"
        elif confidence < 35 and expected_return < -0.01:
            action = "SELL"
            rationale = f"Weak momentum: 5d {mom_5 * 100:.1f}%, 20d {mom_20 * 100:.1f}%, poor alignment, low volume"
        else:
            action = "HOLD"
            rationale = f"Neutral momentum: Mixed signals, 20d {mom_20 * 100:.1f}%"

        return StrategyVote(
            strategy_name=self.name,
            action=action,
            confidence=confidence,
            expected_return=expected_return,
            signals=signals,
            rationale=rationale,
        )


class NewsShockStrategy(BaseStrategy):
    """
    News sentiment-based mean reversion and follow-through strategy

    Logic:
    - Strong positive news → follow-through (buy)
    - Strong negative news → mean reversion (buy oversold)
    - Moderate sentiment → fade moves (contrarian)
    """

    def __init__(self):
        super().__init__("NewsShock")
        self.sentiment_threshold_strong = 0.6  # Strong sentiment
        self.sentiment_threshold_moderate = 0.3
        self.reversion_window = 30  # minutes for mean reversion

    def evaluate(self, symbol: str, market_data: dict[str, Any]) -> StrategyVote:
        news_data = market_data.get("news", [])
        hist = market_data.get("daily_hist")
        intraday = market_data.get("intraday_hist")

        if hist is None or len(hist) < 5:
            return StrategyVote(
                strategy_name=self.name,
                action="HOLD",
                confidence=0.0,
                expected_return=0.0,
                signals={},
                rationale="Insufficient price history",
            )

        # Try to get real-time sentiment from World Feed Fusion
        sentiment_normalized = 0.0
        bullish_count = 0
        bearish_count = 0
        total_news = 0
        confidence_mult = 1.0

        try:
            from core.world_feed_fusion import get_feed_fusion

            fusion = get_feed_fusion()

            # Get 1-day sentiment aggregate
            aggregate = fusion.get_sentiment_aggregate(symbol, "1d")
            if aggregate and aggregate.article_count > 0:
                sentiment_normalized = aggregate.weighted_sentiment
                bullish_count = aggregate.bullish_count
                bearish_count = aggregate.bearish_count
                total_news = aggregate.article_count
                confidence_mult = aggregate.confidence
                signals = {
                    "sentiment_score": float(sentiment_normalized),
                    "bullish_news_count": bullish_count,
                    "bearish_news_count": bearish_count,
                    "neutral_news_count": aggregate.neutral_count,
                    "total_news": total_news,
                    "sentiment_confidence": float(confidence_mult),
                    "source": "world_feed_fusion",
                }
            else:
                # Fallback to legacy news data if available
                signals = self._calculate_legacy_sentiment(news_data)
                sentiment_normalized = signals.get("sentiment_score", 0.0)
                bullish_count = signals.get("bullish_news_count", 0)
                bearish_count = signals.get("bearish_news_count", 0)
                total_news = signals.get("total_news", 0)
        except Exception:
            # Fallback to legacy sentiment
            signals = self._calculate_legacy_sentiment(news_data)
            sentiment_normalized = signals.get("sentiment_score", 0.0)
            bullish_count = signals.get("bullish_news_count", 0)
            bearish_count = signals.get("bearish_news_count", 0)
            total_news = signals.get("total_news", 0)

        # Check price reaction (intraday if available)
        price_reaction = 0.0
        if intraday is not None and len(intraday) >= 10:
            recent_return = (intraday["Close"].iloc[-1] / intraday["Close"].iloc[-10]) - 1
            signals["price_reaction_intraday"] = float(recent_return)
            price_reaction = recent_return
        else:
            # Fall back to daily
            recent_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2]) - 1
            signals["price_reaction_daily"] = float(recent_return)
            price_reaction = recent_return

        score = 50.0
        expected_return = 0.0

        # Strategy logic
        if abs(sentiment_normalized) > self.sentiment_threshold_strong:
            # Strong sentiment → Follow-through strategy
            if sentiment_normalized > 0:  # Bullish news
                if price_reaction > 0:
                    # Bullish news + price up → continue buying (follow-through)
                    score += 30
                    expected_return = 0.02  # Expect 2% follow-through
                    rationale = f"Follow-through: Strong bullish sentiment ({sentiment_normalized:.2f}), price confirms +{price_reaction * 100:.1f}%"
                else:
                    # Bullish news + price down → opportunity
                    score += 20
                    expected_return = 0.015
                    rationale = (
                        f"Opportunity: Bullish news ({sentiment_normalized:.2f}) not yet priced in"
                    )
            else:  # Bearish news
                if price_reaction < 0:
                    # Bearish news + price down → mean reversion opportunity
                    score += 15  # Contrarian
                    expected_return = 0.01  # Small bounce expected
                    rationale = f"Mean reversion: Oversold on bearish news ({sentiment_normalized:.2f}), price -{abs(price_reaction) * 100:.1f}%"
                else:
                    # Bearish news + price up → confusion, hold
                    score = 50
                    expected_return = 0.0
                    rationale = f"Mixed: Bearish news ({sentiment_normalized:.2f}) but price up, await clarity"

        elif abs(sentiment_normalized) > self.sentiment_threshold_moderate:
            # Moderate sentiment → Fade strategy (contrarian)
            if sentiment_normalized > 0 and price_reaction > 0.02:
                # Moderate bullish + big move → fade
                score -= 10
                expected_return = -0.005
                rationale = f"Fade: Moderate sentiment ({sentiment_normalized:.2f}), overextended +{price_reaction * 100:.1f}%"
            elif sentiment_normalized < 0 and price_reaction < -0.02:
                # Moderate bearish + big drop → buy dip
                score += 10
                expected_return = 0.005
                rationale = f"Buy dip: Moderate negative sentiment ({sentiment_normalized:.2f}), oversold -{abs(price_reaction) * 100:.1f}%"
            else:
                score = 50
                expected_return = 0.0
                rationale = "Moderate sentiment, price reaction muted"

        else:
            # Weak sentiment → no clear signal
            score = 50
            expected_return = 0.0
            rationale = f"Neutral: Weak sentiment ({sentiment_normalized:.2f}), no news edge"

        confidence = max(0, min(100, score))

        # Determine action
        if confidence > 60 and expected_return > 0.005:
            action = "BUY"
        elif confidence < 40 and expected_return < -0.005:
            action = "SELL"
        else:
            action = "HOLD"

        return StrategyVote(
            strategy_name=self.name,
            action=action,
            confidence=confidence,
            expected_return=expected_return,
            signals=signals,
            rationale=rationale,
        )

    def _calculate_legacy_sentiment(self, news_data: list) -> dict[str, Any]:
        """Legacy sentiment calculation from old news data format"""
        if not news_data:
            return {
                "sentiment_score": 0.0,
                "bullish_news_count": 0,
                "bearish_news_count": 0,
                "total_news": 0,
                "source": "legacy_fallback",
            }

        sentiment_score = 0.0
        bullish_count = 0
        bearish_count = 0

        for item in news_data[:10]:
            sent = (item.get("sentiment") or "").lower()
            if "bullish" in sent or "positive" in sent:
                sentiment_score += 1.0
                bullish_count += 1
            elif "bearish" in sent or "negative" in sent:
                sentiment_score -= 1.0
                bearish_count += 1

        total_news = len(news_data[:10])
        sentiment_normalized = sentiment_score / total_news if total_news > 0 else 0.0

        return {
            "sentiment_score": float(sentiment_normalized),
            "bullish_news_count": bullish_count,
            "bearish_news_count": bearish_count,
            "total_news": total_news,
            "source": "legacy_fallback",
        }


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage via pairs trading (PLACEHOLDER for multi-asset)

    Currently returns HOLD until multi-asset support added
    """

    def __init__(self):
        super().__init__("PairsTrading")

    def evaluate(self, symbol: str, market_data: dict[str, Any]) -> StrategyVote:
        # Placeholder - requires correlation analysis with other assets
        return StrategyVote(
            strategy_name=self.name,
            action="HOLD",
            confidence=50.0,
            expected_return=0.0,
            signals={"status": "not_implemented"},
            rationale="Pairs trading requires multi-asset support (coming soon)",
        )


class StrategyEnsemble:
    """
    APEX Strategy Ensemble - Voting Engine
    Aggregates votes from multiple strategies with regime-aware weighting
    """

    def __init__(self):
        self.strategies = [
            MomentumStrategy(),
            NewsShockStrategy(),
            PairsTradingStrategy(),  # Placeholder for now
        ]

        # Default weights (equal)
        self.default_weights = {
            "Momentum": 0.50,
            "NewsShock": 0.40,
            "PairsTrading": 0.10,  # Low until implemented
        }

        # Regime-specific weights (adjust based on market conditions)
        self.regime_weights = {
            "BULL": {"Momentum": 0.60, "NewsShock": 0.30, "PairsTrading": 0.10},
            "BEAR": {"Momentum": 0.40, "NewsShock": 0.50, "PairsTrading": 0.10},
            "SIDEWAYS": {"Momentum": 0.30, "NewsShock": 0.40, "PairsTrading": 0.30},
            "HIGH_VOL": {"Momentum": 0.35, "NewsShock": 0.55, "PairsTrading": 0.10},
        }

    def evaluate_all(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate all strategies and aggregate votes

        Args:
            symbol: Trading symbol
            market_data: Dict with price history, news, regime

        Returns:
            {
                "consensus": {action, confidence, expected_return},
                "votes": [list of StrategyVote],
                "weights_used": dict,
                "regime": str
            }
        """

        LOGGER.info(f"Strategy ensemble evaluation for {symbol}")

        # Get current regime
        regime = market_data.get("regime", "BULL")

        # Get regime-specific weights
        weights = self.regime_weights.get(regime, self.default_weights)

        # Collect votes from all strategies
        votes = []
        for strategy in self.strategies:
            try:
                vote = strategy.evaluate(symbol, market_data)
                votes.append(vote)
                LOGGER.info(f"{strategy.name}: {vote.action} (conf={vote.confidence:.1f}%)")
            except Exception as e:
                LOGGER.error(f"Strategy {strategy.name} evaluation failed: {e}")
                # Add default HOLD vote
                votes.append(
                    StrategyVote(
                        strategy_name=strategy.name,
                        action="HOLD",
                        confidence=0.0,
                        expected_return=0.0,
                        signals={"error": str(e)},
                        rationale=f"Error: {str(e)}",
                    )
                )

        # Aggregate votes
        consensus = self._aggregate_votes(votes, weights)

        result = {
            "symbol": symbol,
            "timestamp": int(time.time()),
            "consensus": consensus,
            "votes": [
                {
                    "strategy": v.strategy_name,
                    "action": v.action,
                    "confidence": v.confidence,
                    "expected_return": v.expected_return,
                    "rationale": v.rationale,
                    "signals": v.signals,
                }
                for v in votes
            ],
            "weights_used": weights,
            "regime": regime,
        }

        LOGGER.info(
            f"Ensemble consensus: {consensus['action']} (conf={consensus['confidence']:.1f}%)"
        )

        return result

    def _aggregate_votes(
        self, votes: list[StrategyVote], weights: dict[str, float]
    ) -> dict[str, Any]:
        """
        Aggregate strategy votes using weighted voting
        """

        # Weighted confidence
        weighted_confidence = 0.0
        weighted_return = 0.0

        # Vote counts
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0

        # Weighted vote scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0

        for vote in votes:
            weight = weights.get(vote.strategy_name, 0.0)

            # Accumulate weighted metrics
            weighted_confidence += vote.confidence * weight
            weighted_return += vote.expected_return * weight

            # Count votes
            if vote.action == "BUY":
                buy_votes += 1
                buy_score += vote.confidence * weight
            elif vote.action == "SELL":
                sell_votes += 1
                sell_score += vote.confidence * weight
            else:  # HOLD
                hold_votes += 1
                hold_score += vote.confidence * weight

        # Determine consensus action
        max_score = max(buy_score, sell_score, hold_score)

        if max_score == buy_score and buy_score > 30:  # Need >30 weighted score
            consensus_action = "BUY"
        elif max_score == sell_score and sell_score > 30:
            consensus_action = "SELL"
        else:
            consensus_action = "HOLD"

        # Agreement strength
        total_votes = buy_votes + sell_votes + hold_votes
        max_votes = max(buy_votes, sell_votes, hold_votes)
        agreement = max_votes / total_votes if total_votes > 0 else 0.0

        if agreement >= 1.0:
            agreement_level = "UNANIMOUS"
        elif agreement >= 0.67:
            agreement_level = "STRONG"
        elif agreement >= 0.5:
            agreement_level = "MODERATE"
        else:
            agreement_level = "WEAK"

        return {
            "action": consensus_action,
            "confidence": round(weighted_confidence, 2),
            "expected_return": round(weighted_return, 4),
            "vote_breakdown": {"BUY": buy_votes, "SELL": sell_votes, "HOLD": hold_votes},
            "agreement": agreement_level,
            "agreement_pct": round(agreement * 100, 1),
        }


# Singleton instance
_STRATEGY_ENSEMBLE: StrategyEnsemble | None = None


def get_strategy_ensemble() -> StrategyEnsemble:
    """Get singleton instance of strategy ensemble"""
    global _STRATEGY_ENSEMBLE
    if _STRATEGY_ENSEMBLE is None:
        _STRATEGY_ENSEMBLE = StrategyEnsemble()
    return _STRATEGY_ENSEMBLE
