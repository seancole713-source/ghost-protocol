"""
Ghost Protocol - Deep Researcher
Combines ALL research sources into one comprehensive analysis before prediction
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional

from .earnings_calendar import check_earnings_risk
from .news_analyzer import analyze_news
from .seasonal_patterns import analyze_seasonal
from .historical_analyzer import analyze_historical

logger = logging.getLogger(__name__)


class DeepResearcher:
    """
    Comprehensive research module that analyzes:
    - Earnings calendar (avoid earnings surprises)
    - Recent news sentiment
    - Seasonal patterns (historical same-time performance)
    - 52-week range position
    - YTD performance
    - Same period last year
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes
    
    async def deep_research(self, symbol: str) -> Dict:
        """
        Perform comprehensive research on a symbol.
        Returns a research report with confidence adjustments.
        """
        symbol = symbol.upper()
        
        # Check cache
        cache_key = f"research_{symbol}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"[RESEARCH] Cache hit for {symbol}")
                return cached_data
        
        logger.info(f"[RESEARCH] Starting deep research for {symbol}")
        start_time = datetime.now()
        
        # Run all research in parallel with timeout
        try:
            earnings, news, seasonal, historical = await asyncio.wait_for(
                asyncio.gather(
                    check_earnings_risk(symbol),
                    analyze_news(symbol),
                    analyze_seasonal(symbol),
                    analyze_historical(symbol),
                    return_exceptions=True
                ),
                timeout=30  # 30 second total timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"[RESEARCH] Timeout for {symbol} after 30s")
            return self._empty_report(symbol, "Research timeout")
        except Exception as e:
            logger.error(f"[RESEARCH] Failed for {symbol}: {e}")
            return self._empty_report(symbol, str(e))
        
        # Handle any exceptions from individual tasks
        if isinstance(earnings, Exception):
            logger.warning(f"Earnings research failed for {symbol}: {earnings}")
            earnings = {"error": str(earnings)}
        if isinstance(news, Exception):
            logger.warning(f"News research failed for {symbol}: {news}")
            news = {"error": str(news)}
        if isinstance(seasonal, Exception):
            logger.warning(f"Seasonal research failed for {symbol}: {seasonal}")
            seasonal = {"error": str(seasonal)}
        if isinstance(historical, Exception):
            logger.warning(f"Historical research failed for {symbol}: {historical}")
            historical = {"error": str(historical)}
        
        # Calculate total confidence adjustment
        total_adjustment = 0
        adjustments = []
        warnings = []
        
        # Earnings adjustment
        if earnings.get("risky"):
            adj = -earnings.get("confidence_penalty", 0)
            total_adjustment += adj
            adjustments.append(f"Earnings risk: {adj}%")
            warnings.append(earnings.get("reason"))
        
        # News adjustment
        news_adj = news.get("confidence_adjustment", 0)
        if news_adj != 0:
            total_adjustment += news_adj
            adjustments.append(f"News sentiment: {'+' if news_adj > 0 else ''}{news_adj}%")
        
        # Seasonal adjustment
        seasonal_adj = seasonal.get("confidence_adjustment", 0)
        if seasonal_adj != 0:
            total_adjustment += seasonal_adj
            adjustments.append(f"Seasonal pattern: {'+' if seasonal_adj > 0 else ''}{seasonal_adj}%")
        
        # 52-week range insight
        range_52 = historical.get("52_week_range", {})
        range_position = range_52.get("range_position", 50)
        if range_position > 90:
            warnings.append("Near 52-week high - watch for resistance")
            total_adjustment -= 5
            adjustments.append("52-week range: -5%")
        elif range_position < 10:
            warnings.append("Near 52-week low - high risk")
            total_adjustment -= 5
            adjustments.append("52-week range: -5%")
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            earnings, news, seasonal, historical, total_adjustment
        )
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"[RESEARCH] Completed {symbol} in {duration_ms:.0f}ms (adjustment: {total_adjustment}%)")
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration_ms),
            
            # Raw research data
            "earnings": earnings,
            "news": news,
            "seasonal": seasonal,
            "historical": historical,
            
            # Aggregated insights
            "total_confidence_adjustment": total_adjustment,
            "adjustments": adjustments,
            "warnings": warnings,
            "recommendation": recommendation,
            
            # Quick summary
            "summary": {
                "earnings_risk": earnings.get("risky", False),
                "news_sentiment": news.get("sentiment", "unknown"),
                "seasonal_outlook": seasonal.get("recommendation", "unknown"),
                "ytd_trend": historical.get("ytd_performance", {}).get("trend", "unknown"),
                "range_position": range_position
            }
        }
        
        # Cache result
        self.cache[cache_key] = (result, datetime.now())
        
        return result
    
    def _generate_recommendation(self, earnings: Dict, news: Dict, 
                                  seasonal: Dict, historical: Dict,
                                  total_adjustment: int) -> str:
        """Generate a human-readable recommendation"""
        parts = []
        
        # Earnings warning
        if earnings.get("risky"):
            parts.append(f"⚠️ EARNINGS WARNING: {earnings.get('reason')}")
        
        # News summary
        news_sentiment = news.get("sentiment", "unknown")
        if news_sentiment == "bullish":
            parts.append("📰 News sentiment: BULLISH")
        elif news_sentiment == "bearish":
            parts.append("📰 News sentiment: BEARISH")
        
        # Seasonal insight
        seasonal_rec = seasonal.get("recommendation", "")
        if seasonal_rec and "BULLISH" in seasonal_rec:
            parts.append(f"📅 {seasonal_rec}")
        elif seasonal_rec and "BEARISH" in seasonal_rec:
            parts.append(f"📅 {seasonal_rec}")
        
        # Historical insight
        range_insight = historical.get("52_week_range", {}).get("insight", "")
        if range_insight:
            parts.append(f"📊 {range_insight}")
        
        # Final recommendation
        if total_adjustment > 10:
            parts.append("✅ RESEARCH SUPPORTS: Increase confidence")
        elif total_adjustment < -10:
            parts.append("❌ RESEARCH WARNS: Decrease confidence")
        else:
            parts.append("➡️ RESEARCH NEUTRAL: Rely on technicals")
        
        return "\n".join(parts) if parts else "No significant research findings"
    
    def _empty_report(self, symbol: str, error: str) -> Dict:
        """Return empty report on error"""
        return {
            "symbol": symbol,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "total_confidence_adjustment": 0,
            "adjustments": [],
            "warnings": ["Research failed - proceed with caution"],
            "recommendation": "Research unavailable - rely on technicals only",
            "summary": {
                "earnings_risk": False,
                "news_sentiment": "unknown",
                "seasonal_outlook": "unknown",
                "ytd_trend": "unknown",
                "range_position": 50
            }
        }


# Singleton
_researcher = None


def get_researcher() -> DeepResearcher:
    global _researcher
    if _researcher is None:
        _researcher = DeepResearcher()
    return _researcher


async def deep_research(symbol: str) -> Dict:
    """Quick access to deep research"""
    return await get_researcher().deep_research(symbol)


async def batch_research(symbols: list) -> list:
    """Research multiple symbols in parallel"""
    researcher = get_researcher()
    results = await asyncio.gather(
        *[researcher.deep_research(s) for s in symbols],
        return_exceptions=True
    )
    
    # Handle exceptions
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append({"symbol": symbols[i], "error": str(result)})
        else:
            processed.append(result)
    
    return processed
