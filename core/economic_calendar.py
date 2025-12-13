"""
GHOST Economic Calendar Monitor
================================
24/7 live monitoring of economic events and market-moving announcements.

Tracks:
- Federal Reserve meetings and interest rate decisions
- CPI, PPI, GDP, Unemployment reports
- Earnings announcements (company-specific)
- Corporate actions (splits, dividends, buybacks)
- Geopolitical events

Features:
- Real-time event tracking
- Impact scoring (high/medium/low)
- Pre-event warnings (alert 1 hour before)
- Post-event analysis

Author: Ghost AI
Date: 2025-11-17
"""

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Cache settings
CALENDAR_CACHE: dict[str, dict[str, Any]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


def fetch_economic_calendar(
    days_ahead: int = 7,
    importance: str = "high"
) -> dict[str, Any]:
    """
    Fetch upcoming economic events from Trading Economics or similar API.
    
    Args:
        days_ahead: Number of days to look ahead
        importance: Filter by importance (high/medium/low/all)
        
    Returns:
        Dict with upcoming events list
    """
    cache_key = f"economic_{days_ahead}_{importance}"
    cached = CALENDAR_CACHE.get(cache_key)
    
    if cached and (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]
    
    try:
        # FIXED: Implement Trading Economics API integration
        import requests
        
        api_key = os.getenv("TRADING_ECONOMICS_API_KEY")
        if not api_key:
            # Fallback to Fred API
            fred_key = os.getenv("FRED_API_KEY")
            if not fred_key:
                logger.warning("Economic calendar API not configured (need TRADING_ECONOMICS_API_KEY or FRED_API_KEY)")
                return {"ok": False, "error": "Economic calendar API not configured", "events": []}
            
            # Use Fred API for economic indicators
            url = "https://api.stlouisfed.org/fred/releases/dates"
            params = {"api_key": fred_key, "file_type": "json", "limit": 100}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code != 200:
                return {"ok": False, "error": f"Fred API error: {response.status_code}", "events": []}
            
            data = response.json()
            events = []
            for release in data.get("release_dates", [])[:20]:
                events.append({
                    "name": release.get("release_name", "Unknown"),
                    "date": release.get("date", ""),
                    "importance": "high",
                    "country": "US"
                })
        else:
            # Use Trading Economics API
            url = f"https://api.tradingeconomics.com/calendar"
            params = {
                "c": api_key,
                "f": "json",
                "importance": importance if importance != "all" else ""
            }
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code != 200:
                return {"ok": False, "error": f"Trading Economics error: {response.status_code}", "events": []}
            
            data = response.json()
            events = []
            for item in data[:50]:
                events.append({
                    "name": item.get("Event", ""),
                    "date": item.get("Date", ""),
                    "importance": item.get("Importance", "medium").lower(),
                    "country": item.get("Country", "")
                })
        
        result = {
            "ok": True,
            "events": events,
            "timestamp": datetime.now(UTC).isoformat(),
            "source": "fred" if not api_key else "trading_economics"
        }
        
        CALENDAR_CACHE[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch economic calendar: {e}")
        return {"ok": False, "error": str(e), "events": []}


def fetch_earnings_calendar(symbol: str | None = None, days_ahead: int = 14) -> dict[str, Any]:
    """
    Fetch upcoming earnings announcements.
    
    Args:
        symbol: Optional stock ticker to filter (e.g., 'AAPL')
        days_ahead: Number of days to look ahead
        
    Returns:
        Dict with earnings events
    """
    cache_key = f"earnings_{symbol}_{days_ahead}"
    cached = CALENDAR_CACHE.get(cache_key)
    
    if cached and (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]
    
    try:
        # FIXED: Implement earnings calendar API (Polygon.io or AlphaVantage)
        import requests
        
        # Try Polygon.io first
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key:
            url = f"https://api.polygon.io/v2/reference/earnings"
            params = {"apiKey": polygon_key, "limit": 100}
            if symbol:
                params["ticker"] = symbol
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                earnings = []
                for item in data.get("results", [])[:50]:
                    earnings.append({
                        "symbol": item.get("ticker", ""),
                        "date": item.get("fiscalDate", ""),
                        "eps_estimate": item.get("epsEstimate"),
                        "eps_actual": item.get("epsActual"),
                        "source": "polygon"
                    })
                
                result = {
                    "ok": True,
                    "earnings": earnings,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "source": "polygon"
                }
                
                CALENDAR_CACHE[cache_key] = {"data": result, "timestamp": time.time()}
                return result
        
        # Fallback to AlphaVantage
        alpha_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if alpha_key and symbol:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "EARNINGS",
                "symbol": symbol,
                "apikey": alpha_key
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                earnings = []
                for item in data.get("quarterlyEarnings", [])[:10]:
                    earnings.append({
                        "symbol": symbol,
                        "date": item.get("reportedDate", ""),
                        "eps_estimate": item.get("estimatedEPS"),
                        "eps_actual": item.get("reportedEPS"),
                        "source": "alphavantage"
                    })
                
                result = {
                    "ok": True,
                    "earnings": earnings,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "source": "alphavantage"
                }
                
                CALENDAR_CACHE[cache_key] = {"data": result, "timestamp": time.time()}
                return result
        
        logger.warning("Earnings calendar API not configured (need POLYGON_API_KEY or ALPHAVANTAGE_API_KEY)")
        return {"ok": False, "error": "Earnings API not configured", "earnings": []}
        
    except Exception as e:
        logger.error(f"Failed to fetch earnings calendar: {e}")
        return {"ok": False, "error": str(e), "earnings": []}


def get_upcoming_events(hours_ahead: int = 24) -> list[dict[str, Any]]:
    """
    Get all market-moving events in next N hours.
    
    Combines economic calendar + earnings calendar.
    
    Args:
        hours_ahead: Time window to check
        
    Returns:
        List of events sorted by time
    """
    events = []
    
    # Get economic events
    econ = fetch_economic_calendar(days_ahead=int(hours_ahead / 24) + 1)
    if econ.get("ok"):
        events.extend(econ.get("events", []))
    
    # Get earnings events
    earnings = fetch_earnings_calendar(days_ahead=int(hours_ahead / 24) + 1)
    if earnings.get("ok"):
        events.extend(earnings.get("upcoming_earnings", []))
    
    # Sort by time
    events.sort(key=lambda e: e.get("datetime", ""))
    
    return events


def check_pre_event_warning(symbol: str, hours_before: int = 1) -> dict[str, Any]:
    """
    Check if symbol has major event coming up soon.
    
    Used to avoid entering positions right before earnings/Fed announcements.
    
    Args:
        symbol: Stock ticker
        hours_before: Warning window (hours)
        
    Returns:
        Dict with warning flag and event details
    """
    upcoming = get_upcoming_events(hours_ahead=hours_before)
    
    # Filter for this symbol
    symbol_events = [e for e in upcoming if e.get("symbol") == symbol]
    
    if symbol_events:
        return {
            "warning": True,
            "reason": f"{len(symbol_events)} event(s) in next {hours_before}h",
            "events": symbol_events
        }
    
    return {
        "warning": False,
        "reason": "No major events detected",
        "events": []
    }


def get_market_impact_score(event: dict[str, Any]) -> float:
    """
    Score market impact of an event (0.0 to 1.0).
    
    High impact events (0.8+):
    - FOMC meetings
    - CPI reports
    - Jobs reports
    - Major earnings (AAPL, TSLA, NVDA)
    
    Medium impact (0.5-0.8):
    - GDP releases
    - Retail sales
    - Mid-cap earnings
    
    Low impact (0.0-0.5):
    - Minor indicators
    - Small-cap earnings
    
    Args:
        event: Event dict from calendar
        
    Returns:
        Impact score 0.0 to 1.0
    """
    event_type = event.get("type", "").lower()
    importance = event.get("importance", "").lower()
    
    # High impact events
    if "fomc" in event_type or "federal reserve" in event_type:
        return 0.95
    
    if "cpi" in event_type or "inflation" in event_type:
        return 0.90
    
    if "jobs" in event_type or "unemployment" in event_type:
        return 0.90
    
    if importance == "high":
        return 0.85
    
    if event_type == "earnings":
        # Big tech earnings are high impact
        symbol = event.get("symbol", "")
        if symbol in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]:
            return 0.90
        else:
            return 0.60
    
    if importance == "medium":
        return 0.50
    
    return 0.20


def should_pause_trading(hours_ahead: int = 2) -> tuple[bool, str]:
    """
    Determine if Ghost should pause trading due to upcoming major events.
    
    Args:
        hours_ahead: Warning window
        
    Returns:
        Tuple of (should_pause, reason)
    """
    upcoming = get_upcoming_events(hours_ahead=hours_ahead)
    
    high_impact_events = [e for e in upcoming if get_market_impact_score(e) > 0.8]
    
    if high_impact_events:
        event = high_impact_events[0]
        return (
            True,
            f"High-impact event in {hours_ahead}h: {event.get('type')} - {event.get('description')}"
        )
    
    return False, "No major events detected"


# Sentiment adjustment based on economic calendar
def adjust_confidence_with_calendar(
    symbol: str,
    base_confidence: float
) -> tuple[float, str]:
    """
    Adjust Ghost's confidence based on upcoming events.
    
    Reduces confidence before major events (uncertainty).
    
    Args:
        symbol: Stock ticker
        base_confidence: Original confidence
        
    Returns:
        Tuple of (adjusted_confidence, reason)
    """
    warning = check_pre_event_warning(symbol, hours_before=24)
    
    if warning.get("warning"):
        events = warning.get("events", [])
        
        if events:
            event = events[0]
            impact = get_market_impact_score(event)
            
            # High impact event coming = reduce confidence significantly
            if impact > 0.8:
                adjusted = max(0.0, base_confidence - 0.15)
                return adjusted, f"Major event in 24h: {event.get('type')} (high volatility risk)"
            
            # Medium impact = reduce slightly
            elif impact > 0.5:
                adjusted = max(0.0, base_confidence - 0.08)
                return adjusted, f"Event in 24h: {event.get('type')} (moderate risk)"
    
    return base_confidence, "No calendar risk"
