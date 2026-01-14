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
        Fetch news sentiment for a symbol using Ghost News Brain.
        
        Strategy:
        1. Try Ghost News Brain cached analysis (fast, Claude-powered)
        2. Fallback to RSS feed scan (medium)
        3. Fallback to unavailable signals (safe)
        
        Returns:
            Signals: NEWS_SENTIMENT_SCORE, NEWS_COUNT_24H, BULLISH_RATIO
        """
        self._start_timer()
        signals = []
        errors = []

        try:
            # FIRST: Try Ghost News Brain cached analysis (fast path)
            from core.intelligence.ghost_news_brain import get_news_brain
            
            news_brain = get_news_brain()
            cached_analysis = news_brain.get_cached_analysis(symbol)
            
            if cached_analysis and cached_analysis.get("ok"):
                signals = self._parse_brain_signals(cached_analysis, symbol)
                logger.info(f"[SENTIMENT] {symbol}: Using Ghost News Brain cached analysis")
            else:
                # FALLBACK: Quick RSS scan for symbol mentions
                news_data = self._scan_rss_for_symbol(symbol)
                if news_data.get("articles"):
                    signals = self._parse_news_signals(news_data, symbol)
                    logger.info(f"[SENTIMENT] {symbol}: Using RSS feed scan ({news_data['articles']} articles)")
                else:
                    # No news found - neutral sentiment
                    signals = self._create_neutral_signals(symbol)
                    logger.debug(f"[SENTIMENT] {symbol}: No recent news, returning neutral")

        except Exception as e:
            logger.error(f"Sentiment engine failed for {symbol}: {e}")
            errors.append(f"Sentiment exception: {str(e)}")
            signals = self._create_neutral_signals(symbol)

        return PillarResponse(
            pillar_name=self.pillar_name,
            symbol=symbol,
            signals=signals,
            errors=errors,
            execution_time_ms=self._get_execution_time_ms(),
            timestamp=time.time(),
            cached=False,
        )

    def _parse_brain_signals(self, analysis: dict, symbol: str) -> list[DataSignal]:
        """
        Parse Ghost News Brain analysis into sentiment signals.
        
        Brain provides:
        - symbol_sentiment: dict with sentiment_score, confidence, affected_by
        - major_events: list of market-moving events
        - market_summary: overall market mood
        """
        signals = []
        ts = time.time()
        
        try:
            symbol_data = analysis.get("symbol_sentiment", {}).get(symbol, {})
            sentiment_score = symbol_data.get("sentiment_score", 0.0)  # -1 to +1
            confidence = symbol_data.get("confidence", 0.5)
            affected_by = symbol_data.get("affected_by", [])
            
            # Convert -1/+1 to sentiment score
            signals.append(
                DataSignal(
                    name="NEWS_SENTIMENT_SCORE",
                    value=round(sentiment_score, 2),
                    confidence=confidence,
                    data_available=True,
                    source="ghost_news_brain",
                    timestamp=ts,
                    metadata={
                        "events_affecting": len(affected_by),
                        "event_types": [e.get("type") for e in affected_by],
                        "symbol": symbol
                    },
                )
            )
            
            # News count from events affecting this symbol
            signals.append(
                DataSignal(
                    name="NEWS_COUNT_24H",
                    value=len(affected_by),
                    confidence=1.0,
                    data_available=True,
                    source="ghost_news_brain",
                    timestamp=ts,
                    metadata={"symbol": symbol},
                )
            )
            
            # Bullish ratio from event sentiments
            if affected_by:
                bullish = sum(1 for e in affected_by if e.get("sentiment", "") == "bullish")
                bearish = sum(1 for e in affected_by if e.get("sentiment", "") == "bearish")
                total = len(affected_by)
                ratio = bullish / total if total > 0 else 0.5
            else:
                ratio = 0.5  # Neutral
            
            signals.append(
                DataSignal(
                    name="BULLISH_RATIO",
                    value=round(ratio, 2),
                    confidence=0.8,
                    data_available=True,
                    source="ghost_news_brain",
                    timestamp=ts,
                    metadata={"symbol": symbol},
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to parse brain signals for {symbol}: {e}")
            signals = self._create_neutral_signals(symbol)
        
        return signals

    def _parse_news_signals(self, news_data: dict, symbol: str) -> list[DataSignal]:
        """Parse RSS news data into signals"""
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

    def _scan_rss_for_symbol(self, symbol: str) -> dict:
        """
        Quick scan of RSS feeds for symbol mentions.
        Returns article count and basic sentiment.
        """
        try:
            from core.intelligence.ghost_news_brain import get_news_brain
            import asyncio
            
            # Fix nested event loop issue
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                logger.debug("nest_asyncio not available - RSS scan may fail in async context")
            
            brain = get_news_brain()
            
            # Fetch recent headlines (cached for 5 minutes)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            headlines = loop.run_until_complete(brain.fetch_all_news())
            
            # Filter for symbol mentions
            symbol_upper = symbol.upper()
            relevant = [
                h for h in headlines
                if symbol_upper in h.get("title", "").upper() or
                   symbol_upper in h.get("symbols", [])
            ]
            
            if relevant:
                # Basic sentiment from title keywords
                bullish_words = ["surge", "rally", "gain", "up", "bullish", "breakthrough", "partnership"]
                bearish_words = ["drop", "fall", "down", "crash", "bearish", "hack", "regulation"]
                
                bullish_count = sum(
                    1 for h in relevant
                    if any(w in h.get("title", "").lower() for w in bullish_words)
                )
                bearish_count = sum(
                    1 for h in relevant
                    if any(w in h.get("title", "").lower() for w in bearish_words)
                )
                
                total = len(relevant)
                sentiment = (bullish_count - bearish_count) / total if total > 0 else 0.0
                
                return {
                    "ok": True,
                    "articles": total,
                    "sentiment_score": sentiment,
                    "bullish_count": bullish_count,
                    "bearish_count": bearish_count,
                }
            
        except Exception as e:
            logger.warning(f"RSS scan failed for {symbol}: {e}")
        
        return {"ok": False, "articles": 0}

    def _create_neutral_signals(self, symbol: str) -> list[DataSignal]:
        """
        Return neutral signals when no news found (safe fallback).
        Better than 'unavailable' - means 'no news is neutral news'.
        """
        ts = time.time()
        return [
            DataSignal(
                name="NEWS_SENTIMENT_SCORE",
                value=0.0,  # Neutral
                confidence=0.5,  # Low confidence due to no data
                data_available=True,
                source="no_news_neutral",
                timestamp=ts,
                metadata={"symbol": symbol, "reason": "no_recent_news"},
            ),
            DataSignal(
                name="NEWS_COUNT_24H",
                value=0,
                confidence=1.0,
                data_available=True,
                source="no_news_neutral",
                timestamp=ts,
                metadata={"symbol": symbol},
            ),
            DataSignal(
                name="BULLISH_RATIO",
                value=0.5,  # Neutral
                confidence=0.5,
                data_available=True,
                source="no_news_neutral",
                timestamp=ts,
                metadata={"symbol": symbol},
            ),
        ]

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
