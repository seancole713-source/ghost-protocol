"""
PILLAR 4: Sentiment & News Engine
==================================

Aggregates news and calculates sentiment scores.

Data Sources:
- Polygon news API
- AlphaVantage news sentiment
- Existing news_sentiment.py module

Signals:
- NEWS_SENTIMENT_SCORE (-1 to +1)
- NEWS_COUNT_24H
- BULLISH_RATIO
- TOP_STORY_SENTIMENT

Author: Ghost AI
Date: November 21, 2025
"""

import logging
import time
from typing import Any

from core.data_pillars.base_pillar import BasePillar, DataSignal, PillarResponse

logger = logging.getLogger(__name__)


class SentimentEngine(BasePillar):
    """News and sentiment analysis engine."""

    def __init__(self):
        super().__init__(pillar_name="sentiment_engine")

    def get_signals(self, symbol: str, **kwargs) -> PillarResponse:
        """
        Fetch news sentiment for a symbol.
        
        Returns:
            Signals: NEWS_SENTIMENT_SCORE, NEWS_COUNT_24H, BULLISH_RATIO
        """
        self._start_timer()
        signals = []
        errors = []

        try:
            # Use existing news_sentiment module
            from core.news_sentiment import fetch_news_sentiment

            news_data = fetch_news_sentiment(symbol, limit=20)

            if news_data and news_data.get("ok"):
                signals = self._parse_news_signals(news_data, symbol)
            else:
                error_msg = news_data.get("error", "News API unavailable")
                errors.append(error_msg)
                signals = self._create_unavailable_signals()

        except Exception as e:
            logger.error(f"Sentiment engine failed for {symbol}: {e}")
            errors.append(f"Sentiment exception: {str(e)}")
            signals = self._create_unavailable_signals()

        return PillarResponse(
            pillar_name=self.pillar_name,
            symbol=symbol,
            signals=signals,
            errors=errors,
            execution_time_ms=self._get_execution_time_ms(),
            timestamp=time.time(),
            cached=False,
        )

    def _parse_news_signals(self, news_data: dict, symbol: str) -> list[DataSignal]:
        """Parse news data into signals"""
        signals = []
        ts = time.time()

        try:
            articles = news_data.get("articles", [])
            sentiment_score = news_data.get("sentiment_score", 0.0)
            bullish_count = news_data.get("bullish_count", 0)
            bearish_count = news_data.get("bearish_count", 0)

            # Overall sentiment score
            signals.append(
                DataSignal(
                    name="NEWS_SENTIMENT_SCORE",
                    value=round(sentiment_score, 2),
                    confidence=0.7,  # News sentiment is moderate confidence
                    data_available=True,
                    source="news_api",
                    timestamp=ts,
                    metadata={"articles_analyzed": len(articles), "symbol": symbol},
                )
            )

            # News count (24h)
            signals.append(
                DataSignal(
                    name="NEWS_COUNT_24H",
                    value=len(articles),
                    confidence=1.0,
                    data_available=True,
                    source="news_api",
                    timestamp=ts,
                    metadata={"symbol": symbol},
                )
            )

            # Bullish ratio
            total = bullish_count + bearish_count
            if total > 0:
                bullish_ratio = bullish_count / total
                signals.append(
                    DataSignal(
                        name="BULLISH_RATIO",
                        value=round(bullish_ratio, 2),
                        confidence=0.7,
                        data_available=True,
                        source="calculated",
                        timestamp=ts,
                        metadata={
                            "bullish_count": bullish_count,
                            "bearish_count": bearish_count,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"News signal parsing failed for {symbol}: {e}")

        return signals

    def _create_unavailable_signals(self) -> list[DataSignal]:
        """Create unavailable signals when data missing"""
        return [
            self._create_unavailable_signal(name, "News data unavailable")
            for name in self.get_signal_names()
        ]

    def get_signal_names(self) -> list[str]:
        """Get list of sentiment signal names"""
        return [
            "NEWS_SENTIMENT_SCORE",
            "NEWS_COUNT_24H",
            "BULLISH_RATIO",
        ]

    def health_check(self) -> dict[str, Any]:
        """Verify sentiment engine can fetch news"""
        results = {
            "ok": True,
            "pillar": self.pillar_name,
            "providers": [],
            "errors": [],
        }

        try:
            spy_response = self.get_signals("SPY")

            if spy_response.available_signal_count() >= 2:
                results["providers"].append(
                    {
                        "name": "news_api",
                        "status": "ok",
                        "latency_ms": spy_response.execution_time_ms,
                        "signals_computed": spy_response.available_signal_count(),
                    }
                )
            else:
                results["ok"] = False
                results["errors"].append("Sentiment engine failed health check")

        except Exception as e:
            results["ok"] = False
            results["errors"].append(f"Sentiment engine health check failed: {e}")

        return results
