"""
GHOST INTEL - EVENT NORMALIZATION
==================================
Canonical event schema for all intelligence data.

All events from any source get normalized to this schema
before impact scoring and decision making.

Author: Ghost AI
Date: 2026-01-26
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ghost.intel")


class EventLayer(Enum):
    """The 8 intelligence layers"""
    MACRO = "macro"
    RATES = "rates"
    CORPORATE = "corporate"
    POLITICS = "politics"
    GEOPOLITICS = "geopolitics"
    INDIVIDUALS = "individuals"
    SOCIAL = "social"
    POSITIONING = "positioning"


class EventDirection(Enum):
    """Market impact direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EventScope(Enum):
    """Scope of impact"""
    TICKER = "ticker"      # Single stock
    SECTOR = "sector"      # Industry/sector
    MARKET = "market"      # Entire market
    GLOBAL = "global"      # Multiple markets


class EventHorizon(Enum):
    """Time horizon of impact"""
    IMMEDIATE = "immediate"    # Minutes
    SAME_DAY = "same_day"      # Hours
    MULTI_DAY = "multi_day"    # Days
    WEEKS = "weeks"            # Weeks
    STRUCTURAL = "structural"  # Months+


class SourceTier(Enum):
    """Source credibility tier"""
    TIER1 = 1   # Official (Fed, SEC, company filings)
    TIER2 = 2   # Major news (Reuters, Bloomberg, WSJ)
    TIER3 = 3   # Secondary news (business sites)
    TIER4 = 4   # Social media verified
    TIER5 = 5   # Social media unverified


@dataclass
class IntelEvent:
    """
    Canonical event schema for Ghost Intel.
    
    All events from any source normalize to this structure.
    """
    # Identity
    event_id: str                          # Unique hash
    
    # Source info
    source: str                            # Where it came from
    source_tier: SourceTier                # Credibility level
    
    # Classification
    layer: EventLayer                      # Which intelligence layer
    category: str                          # Sub-category (e.g., "cpi", "fomc")
    
    # Content
    headline: str                          # Main headline/summary
    
    # Optional fields with defaults
    source_url: Optional[str] = None       # Original URL
    description: Optional[str] = None      # Full description
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Original data
    
    # Timing
    timestamp: float = field(default_factory=time.time)     # When we received it
    event_time: Optional[float] = None                      # When it happened
    
    # Impact assessment (filled by impact_model)
    direction: EventDirection = EventDirection.NEUTRAL
    scope: EventScope = EventScope.MARKET
    horizon: EventHorizon = EventHorizon.SAME_DAY
    
    # Targeting
    tickers: List[str] = field(default_factory=list)        # Affected tickers
    sectors: List[str] = field(default_factory=list)        # Affected sectors
    
    # Verification
    corroborated: bool = False             # Seen from multiple sources
    source_count: int = 1                  # How many sources reported it
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    # Flags
    is_breaking: bool = False              # Breaking news
    is_scheduled: bool = False             # Scheduled event (earnings, FOMC)
    price_led: bool = False                # Price moved before this news
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate event_id if not provided"""
        if not self.event_id:
            self.event_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique event ID from content"""
        content = f"{self.source}:{self.headline}:{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "event_id": self.event_id,
            "source": self.source,
            "source_tier": self.source_tier.value,
            "source_url": self.source_url,
            "layer": self.layer.value,
            "category": self.category,
            "headline": self.headline,
            "description": self.description,
            "timestamp": self.timestamp,
            "event_time": self.event_time,
            "direction": self.direction.value,
            "scope": self.scope.value,
            "horizon": self.horizon.value,
            "tickers": self.tickers,
            "sectors": self.sectors,
            "corroborated": self.corroborated,
            "source_count": self.source_count,
            "tags": self.tags,
            "is_breaking": self.is_breaking,
            "is_scheduled": self.is_scheduled,
            "price_led": self.price_led,
        }


def normalize_event(
    source: str,
    data: Dict[str, Any],
    layer: EventLayer,
    category: str,
) -> IntelEvent:
    """
    Normalize raw data from any source into canonical IntelEvent.
    
    Args:
        source: Source identifier (e.g., "fred", "polygon_news", "stocktwits")
        data: Raw data from the source
        layer: Which intelligence layer this belongs to
        category: Sub-category (e.g., "cpi", "fomc", "earnings")
    
    Returns:
        Normalized IntelEvent
    """
    # Determine source tier
    source_tier = _get_source_tier(source)
    
    # Extract headline and description based on source type
    headline, description = _extract_content(source, data)
    
    # Extract timing
    timestamp = time.time()
    event_time = _extract_event_time(source, data)
    
    # Extract affected tickers/sectors
    tickers = _extract_tickers(source, data)
    sectors = _extract_sectors(source, data, tickers)
    
    # Determine scope
    scope = _determine_scope(tickers, sectors, layer)
    
    # Create event
    event = IntelEvent(
        event_id="",  # Will be generated
        source=source,
        source_tier=source_tier,
        source_url=data.get("url") or data.get("article_url"),
        layer=layer,
        category=category,
        headline=headline,
        description=description,
        raw_data=data,
        timestamp=timestamp,
        event_time=event_time,
        tickers=tickers,
        sectors=sectors,
        scope=scope,
    )
    
    return event


def _get_source_tier(source: str) -> SourceTier:
    """Determine source credibility tier"""
    tier1_sources = ["fred", "sec_edgar", "fed", "treasury"]
    tier2_sources = ["polygon_news", "reuters", "bloomberg", "wsj"]
    tier3_sources = ["yahoo_news", "finnhub", "alphavantage"]
    tier4_sources = ["stocktwits", "twitter_verified"]
    
    source_lower = source.lower()
    
    if any(s in source_lower for s in tier1_sources):
        return SourceTier.TIER1
    elif any(s in source_lower for s in tier2_sources):
        return SourceTier.TIER2
    elif any(s in source_lower for s in tier3_sources):
        return SourceTier.TIER3
    elif any(s in source_lower for s in tier4_sources):
        return SourceTier.TIER4
    else:
        return SourceTier.TIER5


def _extract_content(source: str, data: Dict[str, Any]) -> tuple:
    """Extract headline and description from raw data"""
    headline = ""
    description = ""
    
    # News article format
    if "title" in data:
        headline = data["title"]
        description = data.get("description", "")
    
    # FRED macro data format
    elif "value" in data:
        indicator = data.get("indicator", source)
        value = data["value"]
        prev_value = data.get("prev_value")
        
        if prev_value:
            change = ((value - prev_value) / prev_value) * 100
            direction = "up" if change > 0 else "down"
            headline = f"{indicator}: {value} ({direction} {abs(change):.1f}%)"
        else:
            headline = f"{indicator}: {value}"
        
        description = f"Latest {indicator} reading"
    
    # Social media format
    elif "sentiment_score" in data:
        symbol = data.get("symbol", "Unknown")
        sentiment = data.get("sentiment_label", "NEUTRAL")
        score = data.get("sentiment_score", 0)
        headline = f"{symbol} social sentiment: {sentiment} ({score:.2f})"
        description = f"Based on {data.get('message_count', 0)} messages"
    
    # Rates format
    elif "price" in data:
        name = data.get("name", source)
        price = data["price"]
        change = data.get("change_pct", 0)
        headline = f"{name}: {price} ({change:+.2f}%)"
    
    # Fallback
    else:
        headline = str(data)[:200]
    
    return headline, description


def _extract_event_time(source: str, data: Dict[str, Any]) -> Optional[float]:
    """Extract when the event actually happened"""
    time_fields = ["published", "published_utc", "date", "timestamp", "time"]
    
    for field in time_fields:
        if field in data:
            value = data[field]
            
            # Already a timestamp
            if isinstance(value, (int, float)):
                return float(value)
            
            # Parse ISO format
            if isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return dt.timestamp()
                except:
                    pass
    
    return None


def _extract_tickers(source: str, data: Dict[str, Any]) -> List[str]:
    """Extract affected stock tickers"""
    tickers = []
    
    # Direct tickers field
    if "tickers" in data:
        tickers = data["tickers"]
    elif "ticker" in data:
        tickers = [data["ticker"]]
    elif "symbol" in data:
        tickers = [data["symbol"]]
    
    # Clean and validate
    valid_tickers = []
    for t in tickers:
        if isinstance(t, str) and len(t) <= 5 and t.isalpha():
            valid_tickers.append(t.upper())
    
    return valid_tickers


def _extract_sectors(source: str, data: Dict[str, Any], tickers: List[str]) -> List[str]:
    """Extract affected sectors"""
    sectors = []
    
    # Direct sectors field
    if "sectors" in data:
        sectors = data["sectors"]
    elif "sector" in data:
        sectors = [data["sector"]]
    
    # Infer from tickers
    if not sectors and tickers:
        from ghost_intel.taxonomy import get_ticker_sector
        for ticker in tickers:
            sector = get_ticker_sector(ticker)
            if sector and sector not in sectors:
                sectors.append(sector)
    
    return sectors


def _determine_scope(
    tickers: List[str],
    sectors: List[str],
    layer: EventLayer
) -> EventScope:
    """Determine the scope of impact"""
    # Macro and rates affect whole market
    if layer in [EventLayer.MACRO, EventLayer.RATES, EventLayer.GEOPOLITICS]:
        return EventScope.MARKET
    
    # Politics can be global or sector
    if layer == EventLayer.POLITICS:
        return EventScope.SECTOR if sectors else EventScope.MARKET
    
    # Specific tickers
    if tickers and len(tickers) == 1:
        return EventScope.TICKER
    elif tickers:
        return EventScope.SECTOR
    elif sectors:
        return EventScope.SECTOR
    else:
        return EventScope.MARKET


# =============================================================================
# DEDUPLICATION
# =============================================================================

class EventDeduplicator:
    """
    Deduplicate events using semantic + hash matching.
    Same story from multiple sources should be corroborated, not duplicated.
    """
    
    def __init__(self, window_seconds: int = 3600):
        self._seen_events: Dict[str, IntelEvent] = {}
        self._window = window_seconds
    
    def process(self, event: IntelEvent) -> Optional[IntelEvent]:
        """
        Process an event for deduplication.
        
        Returns:
            None if duplicate (original event is updated)
            Event if new
        """
        # Clean old events
        self._clean_old()
        
        # Check for semantic duplicate
        similar_key = self._find_similar(event)
        
        if similar_key:
            # Update existing event with corroboration
            existing = self._seen_events[similar_key]
            existing.corroborated = True
            existing.source_count += 1
            
            # Upgrade source tier if new source is more credible
            if event.source_tier.value < existing.source_tier.value:
                existing.source_tier = event.source_tier
            
            logger.debug(f"[INTEL] Event corroborated: {event.headline[:50]}...")
            return None
        
        # New event
        self._seen_events[event.event_id] = event
        return event
    
    def _find_similar(self, event: IntelEvent) -> Optional[str]:
        """Find semantically similar event"""
        for key, existing in self._seen_events.items():
            # Same layer and category
            if existing.layer != event.layer or existing.category != event.category:
                continue
            
            # Same tickers
            if event.tickers and existing.tickers:
                if not set(event.tickers) & set(existing.tickers):
                    continue
            
            # Similar headline (simple check)
            if self._headlines_similar(event.headline, existing.headline):
                return key
        
        return None
    
    def _headlines_similar(self, h1: str, h2: str) -> bool:
        """Check if two headlines are semantically similar"""
        # Simple word overlap check
        words1 = set(h1.lower().split())
        words2 = set(h2.lower().split())
        
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for"}
        words1 -= stopwords
        words2 -= stopwords
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return (overlap / total) > 0.5
    
    def _clean_old(self):
        """Remove events outside the dedup window"""
        cutoff = time.time() - self._window
        self._seen_events = {
            k: v for k, v in self._seen_events.items()
            if v.timestamp > cutoff
        }


# Singleton deduplicator
_deduplicator: Optional[EventDeduplicator] = None


def get_deduplicator() -> EventDeduplicator:
    """Get singleton deduplicator instance"""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = EventDeduplicator()
    return _deduplicator
