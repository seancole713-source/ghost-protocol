"""
GHOST INTEL - API ROUTES
=========================
FastAPI endpoints for the Ghost Intel module.

Endpoints:
- GET /api/intel/now         → Top live events
- GET /api/intel/timeline    → Events over time window
- GET /api/intel/impact      → Impact score for symbol
- GET /api/intel/health      → Feed availability status
- GET /api/intel/rates       → Live rates and yields
- GET /api/intel/positioning → Market positioning analysis
- GET /api/intel/social      → Social sentiment

Author: Ghost AI
Date: 2026-01-26
"""

import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ghost_intel.sources import get_intel_sources, IntelSources
from ghost_intel.normalize import (
    IntelEvent, normalize_event, EventLayer, 
    get_deduplicator, EventDeduplicator
)
from ghost_intel.impact_model import (
    get_impact_scorer, ImpactScorer, 
    is_signal_not_noise, ImpactScore
)
from ghost_intel.positioning import get_positioning_analyzer, PositioningAnalyzer
from ghost_intel.taxonomy import EventTaxonomy, EventCategory, classify_event

logger = logging.getLogger("ghost.intel")

# Create router
router = APIRouter(prefix="/api/intel", tags=["intel"])

# Store for recent events
_event_store: List[IntelEvent] = []
_max_events = 500


@router.get("/now")
async def get_intel_now(
    limit: int = Query(default=20, ge=1, le=100),
    min_score: float = Query(default=30, ge=0, le=100),
    layer: Optional[str] = Query(default=None),
):
    """
    Get top live intelligence events.
    
    Args:
        limit: Maximum events to return
        min_score: Minimum impact score threshold
        layer: Filter by layer (macro, rates, corporate, etc.)
    
    Returns:
        List of scored events, highest impact first
    """
    logger.info(f"[INTEL] /now request: limit={limit}, min_score={min_score}, layer={layer}")
    
    sources = get_intel_sources()
    scorer = get_impact_scorer()
    deduper = get_deduplicator()
    
    # Fetch fresh data
    raw_data = await sources.fetch_all_layers()
    
    # Process and score events
    events = []
    
    # Process news
    news_data = raw_data.get("layers", {}).get("news", {})
    if news_data.get("available"):
        for article in news_data.get("articles", [])[:20]:
            event = normalize_event(
                source="polygon_news",
                data=article,
                layer=EventLayer.CORPORATE,  # Default, will be reclassified
                category="news"
            )
            
            # Reclassify based on content
            classification = classify_event(
                f"{article.get('title', '')} {article.get('description', '')}",
                article.get("tickers", [])
            )
            event.category = classification["category"].value
            event.layer = EventLayer(classification["layer"]) if classification["layer"] in [e.value for e in EventLayer] else EventLayer.CORPORATE
            event.tickers = article.get("tickers", [])
            
            # Deduplicate
            processed = deduper.process(event)
            if processed:
                events.append(processed)
    
    # Process macro data changes
    macro_data = raw_data.get("layers", {}).get("macro", {})
    if macro_data.get("available"):
        for indicator, data in macro_data.get("data", {}).items():
            if isinstance(data, dict) and data.get("value") is not None:
                event = normalize_event(
                    source="fred",
                    data={**data, "indicator": indicator},
                    layer=EventLayer.MACRO,
                    category=indicator
                )
                events.append(event)
    
    # Process rates data
    rates_data = raw_data.get("layers", {}).get("rates", {})
    if rates_data.get("available"):
        for name, data in rates_data.get("data", {}).items():
            if isinstance(data, dict) and data.get("price") is not None:
                # Only create events for significant moves
                change_pct = abs(data.get("change_pct", 0))
                if change_pct > 1.0 or name in ["vix", "us_10y", "dxy"]:
                    event = normalize_event(
                        source="yahoo",
                        data={**data, "name": name},
                        layer=EventLayer.RATES,
                        category=name
                    )
                    events.append(event)
    
    # Score all events
    scored_events = []
    for event in events:
        try:
            # Filter by layer if specified
            if layer and event.layer.value != layer:
                continue
            
            # Score
            score = scorer.score(event)
            
            # Filter by minimum score
            if score.score < min_score:
                continue
            
            scored_events.append({
                "event": event.to_dict(),
                "impact": score.to_dict(),
            })
        except Exception as e:
            logger.error(f"[INTEL] Scoring error: {e}")
    
    # Sort by score
    scored_events.sort(key=lambda x: x["impact"]["score"], reverse=True)
    
    # Update context for future scoring
    if rates_data.get("available"):
        context_update = {}
        vix_data = rates_data.get("data", {}).get("vix", {})
        if vix_data.get("price"):
            context_update["vix"] = vix_data["price"]
        
        positioning = raw_data.get("layers", {}).get("positioning", {})
        if positioning.get("put_call_ratio"):
            context_update["put_call_ratio"] = positioning["put_call_ratio"]
        
        if context_update:
            scorer.update_context(context_update)
    
    return {
        "timestamp": time.time(),
        "count": len(scored_events[:limit]),
        "total_processed": len(events),
        "events": scored_events[:limit],
        "context": {
            "vix": scorer.market_context.get("vix"),
            "put_call_ratio": scorer.market_context.get("put_call_ratio"),
        }
    }


@router.get("/timeline")
async def get_intel_timeline(
    hours: int = Query(default=24, ge=1, le=168),
    symbol: Optional[str] = Query(default=None),
):
    """
    Get events over a time window.
    
    Args:
        hours: Lookback period in hours
        symbol: Filter by symbol
    
    Returns:
        Timeline of events
    """
    logger.info(f"[INTEL] /timeline request: hours={hours}, symbol={symbol}")
    
    cutoff = time.time() - (hours * 3600)
    
    # Filter stored events
    events = [
        e.to_dict() for e in _event_store
        if e.timestamp > cutoff and (not symbol or symbol.upper() in e.tickers)
    ]
    
    # Sort by timestamp
    events.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "timestamp": time.time(),
        "hours": hours,
        "symbol": symbol,
        "count": len(events),
        "events": events,
    }


@router.get("/impact/{symbol}")
async def get_intel_impact(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168),
):
    """
    Get aggregated impact score for a symbol.
    
    Args:
        symbol: Stock ticker
        hours: Lookback period
    
    Returns:
        Aggregated impact analysis
    """
    logger.info(f"[INTEL] /impact request: symbol={symbol}, hours={hours}")
    
    sources = get_intel_sources()
    scorer = get_impact_scorer()
    
    symbol = symbol.upper()
    
    # Fetch symbol-specific data
    tasks = [
        sources.get_polygon_news(symbol, limit=10),
        sources.get_stocktwits_sentiment(symbol),
        sources.get_rates_and_liquidity(),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    news_data = results[0] if not isinstance(results[0], Exception) else {}
    social_data = results[1] if not isinstance(results[1], Exception) else {}
    rates_data = results[2] if not isinstance(results[2], Exception) else {}
    
    # Update scorer context
    if rates_data.get("available"):
        vix_data = rates_data.get("data", {}).get("vix", {})
        if vix_data.get("price"):
            scorer.update_context({"vix": vix_data["price"]})
    
    # Score news events
    event_scores = []
    if news_data.get("available"):
        for article in news_data.get("articles", []):
            event = normalize_event(
                source="polygon_news",
                data=article,
                layer=EventLayer.CORPORATE,
                category="news"
            )
            event.tickers = [symbol]
            
            score = scorer.score(event)
            event_scores.append({
                "headline": event.headline,
                "score": score.score,
                "direction": score.direction.value,
            })
    
    # Aggregate scores
    if event_scores:
        avg_score = sum(e["score"] for e in event_scores) / len(event_scores)
        max_score = max(e["score"] for e in event_scores)
        
        # Direction consensus
        bullish = sum(1 for e in event_scores if e["direction"] == "bullish")
        bearish = sum(1 for e in event_scores if e["direction"] == "bearish")
        
        if bullish > bearish + 1:
            direction = "BULLISH"
        elif bearish > bullish + 1:
            direction = "BEARISH"
        else:
            direction = "MIXED"
    else:
        avg_score = 0
        max_score = 0
        direction = "NEUTRAL"
    
    # Social sentiment
    social_sentiment = 0
    if social_data.get("available"):
        social_sentiment = social_data.get("sentiment_score", 0)
    
    return {
        "timestamp": time.time(),
        "symbol": symbol,
        "aggregate_score": round(avg_score, 1),
        "max_event_score": round(max_score, 1),
        "event_count": len(event_scores),
        "direction": direction,
        "social_sentiment": round(social_sentiment, 2),
        "events": event_scores[:5],  # Top 5 events
        "signal": "WATCH" if avg_score > 50 else "MONITOR" if avg_score > 30 else "QUIET",
    }


@router.get("/rates")
async def get_rates():
    """
    Get live rates and yields.
    
    Returns:
        Treasury yields, DXY, VIX, spreads
    """
    logger.info("[INTEL] /rates request")
    
    sources = get_intel_sources()
    data = await sources.get_rates_and_liquidity()
    
    if not data.get("available"):
        raise HTTPException(status_code=503, detail="Rates data unavailable")
    
    return {
        "timestamp": time.time(),
        "rates": data.get("data", {}),
    }


@router.get("/positioning")
async def get_positioning():
    """
    Get market positioning analysis.
    
    Returns:
        Put/call ratio, VIX analysis, fragility assessment
    """
    logger.info("[INTEL] /positioning request")
    
    sources = get_intel_sources()
    analyzer = get_positioning_analyzer()
    
    # Fetch positioning data
    rates = await sources.get_rates_and_liquidity()
    pcr = await sources.get_put_call_ratio()
    
    # Combine data for analysis
    data = {
        "vix": rates.get("data", {}).get("vix", {}).get("price", 15),
        "vix_change": rates.get("data", {}).get("vix", {}).get("change_pct", 0),
        "put_call_ratio": pcr.get("put_call_ratio", 0.9),
    }
    
    if rates.get("data", {}).get("vix_term_structure"):
        data["vix_term_structure"] = rates["data"]["vix_term_structure"]
    
    # Analyze
    analysis = analyzer.analyze(data)
    signal = analyzer.get_positioning_signal(data)
    
    return {
        "timestamp": time.time(),
        "analysis": analysis.to_dict(),
        "signal": signal,
    }


@router.get("/social/{symbol}")
async def get_social_sentiment(symbol: str):
    """
    Get social sentiment for a symbol.
    
    Returns:
        StockTwits and Reddit sentiment
    """
    logger.info(f"[INTEL] /social request: symbol={symbol}")
    
    sources = get_intel_sources()
    
    symbol = symbol.upper()
    
    # Fetch social data
    tasks = [
        sources.get_stocktwits_sentiment(symbol),
        sources.get_reddit_wsb_sentiment(symbol),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    stocktwits = results[0] if not isinstance(results[0], Exception) else {"available": False}
    reddit = results[1] if not isinstance(results[1], Exception) else {"available": False}
    
    # Aggregate sentiment
    sentiments = []
    if stocktwits.get("available"):
        sentiments.append(stocktwits.get("sentiment_score", 0))
    if reddit.get("available") and reddit.get("symbol_data"):
        # Reddit doesn't have direct sentiment, use mention count as proxy
        pass
    
    aggregate = sum(sentiments) / len(sentiments) if sentiments else 0
    
    return {
        "timestamp": time.time(),
        "symbol": symbol,
        "aggregate_sentiment": round(aggregate, 3),
        "stocktwits": stocktwits,
        "reddit_wsb": reddit,
    }


@router.get("/health")
async def get_intel_health():
    """
    Get health status of all data feeds.
    
    Returns:
        Status of each data source
    """
    logger.info("[INTEL] /health request")
    
    sources = get_intel_sources()
    health = sources.get_health()
    
    # Overall status
    available_count = sum(1 for s in health["sources"].values() if s["available"])
    total_count = len(health["sources"])
    
    if available_count == total_count:
        overall = "HEALTHY"
    elif available_count >= total_count * 0.5:
        overall = "DEGRADED"
    else:
        overall = "UNHEALTHY"
    
    return {
        "timestamp": time.time(),
        "overall_status": overall,
        "available_sources": available_count,
        "total_sources": total_count,
        "sources": health["sources"],
        "api_keys_configured": health["api_keys_configured"],
    }


@router.get("/macro")
async def get_macro_data():
    """
    Get latest macro economic data.
    
    Returns:
        CPI, NFP, GDP, PCE, etc.
    """
    logger.info("[INTEL] /macro request")
    
    sources = get_intel_sources()
    data = await sources.get_macro_data()
    
    if not data.get("available"):
        return {
            "timestamp": time.time(),
            "available": False,
            "error": data.get("error", "Macro data unavailable"),
            "hint": "Set FRED_API_KEY environment variable"
        }
    
    return {
        "timestamp": time.time(),
        "available": True,
        "data": data.get("data", {}),
    }


def register_intel_routes(app):
    """Register intel routes with FastAPI app"""
    app.include_router(router)
    logger.info("[INTEL] Routes registered: /api/intel/*")


# For testing
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Ghost Intel API")
    register_intel_routes(app)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
