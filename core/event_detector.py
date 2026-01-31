"""
EVENT DETECTOR - Ghost's Eyes and Ears

Ghost needs to FIND events to learn from them.
This module detects market-moving events from multiple sources:

DATA SOURCES:
1. Financial APIs - Earnings, economic data, corporate actions
2. News APIs - Breaking news, announcements  
3. On-chain Data - Whale movements, exchange flows
4. Social Media - Twitter/X, Reddit sentiment
5. Exchange APIs - Listings, delistings
6. Economic Calendars - Fed meetings, jobs reports
7. Price Data - Sudden moves that indicate events

Ghost learns: "What just happened?" → "How did the market react?"
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from core.event_memory import EventMemory, EventType

LOGGER = logging.getLogger(__name__)


# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    source_type: str  # api, rss, websocket, scrape
    url: str
    api_key_env: Optional[str] = None
    refresh_interval_seconds: int = 60
    enabled: bool = True
    priority: int = 1  # Lower = higher priority


# Known free/freemium data sources Ghost can use
DATA_SOURCES = {
    # =========================================================================
    # FINANCIAL DATA - Earnings, Corporate Actions
    # =========================================================================
    "alpha_vantage": DataSource(
        name="Alpha Vantage",
        source_type="api",
        url="https://www.alphavantage.co/query",
        api_key_env="ALPHA_VANTAGE_API_KEY",
        refresh_interval_seconds=300,  # 5 min (rate limited)
        priority=1
    ),
    
    "polygon": DataSource(
        name="Polygon.io",
        source_type="api",
        url="https://api.polygon.io",
        api_key_env="POLYGON_API_KEY",
        refresh_interval_seconds=60,
        priority=1
    ),
    
    "finnhub": DataSource(
        name="Finnhub",
        source_type="api",
        url="https://finnhub.io/api/v1",
        api_key_env="FINNHUB_API_KEY",
        refresh_interval_seconds=60,
        priority=2
    ),
    
    # =========================================================================
    # NEWS - Breaking news, announcements
    # =========================================================================
    "newsapi": DataSource(
        name="NewsAPI",
        source_type="api",
        url="https://newsapi.org/v2",
        api_key_env="NEWS_API_KEY",
        refresh_interval_seconds=60,
        priority=1
    ),
    
    "cryptopanic": DataSource(
        name="CryptoPanic",
        source_type="api",
        url="https://cryptopanic.com/api/v1",
        api_key_env="CRYPTOPANIC_API_KEY",
        refresh_interval_seconds=30,
        priority=1
    ),
    
    # =========================================================================
    # ON-CHAIN DATA - Whale movements, exchange flows
    # =========================================================================
    "whale_alert": DataSource(
        name="Whale Alert",
        source_type="api",
        url="https://api.whale-alert.io/v1",
        api_key_env="WHALE_ALERT_API_KEY",
        refresh_interval_seconds=30,
        priority=1
    ),
    
    "glassnode": DataSource(
        name="Glassnode",
        source_type="api",
        url="https://api.glassnode.com/v1",
        api_key_env="GLASSNODE_API_KEY",
        refresh_interval_seconds=300,
        priority=2
    ),
    
    # =========================================================================
    # SOCIAL MEDIA - Twitter/X, Reddit
    # =========================================================================
    "twitter": DataSource(
        name="Twitter/X API",
        source_type="api",
        url="https://api.twitter.com/2",
        api_key_env="TWITTER_BEARER_TOKEN",
        refresh_interval_seconds=60,
        priority=1
    ),
    
    "reddit": DataSource(
        name="Reddit API",
        source_type="api",
        url="https://oauth.reddit.com",
        api_key_env="REDDIT_ACCESS_TOKEN",
        refresh_interval_seconds=60,
        priority=2
    ),
    
    "lunarcrush": DataSource(
        name="LunarCrush",
        source_type="api",
        url="https://api.lunarcrush.com/v2",
        api_key_env="LUNARCRUSH_API_KEY",
        refresh_interval_seconds=60,
        priority=2
    ),
    
    # =========================================================================
    # ECONOMIC DATA - Fed, jobs, GDP
    # =========================================================================
    "fred": DataSource(
        name="FRED (Federal Reserve)",
        source_type="api",
        url="https://api.stlouisfed.org/fred",
        api_key_env="FRED_API_KEY",
        refresh_interval_seconds=3600,  # Hourly (data changes slowly)
        priority=1
    ),
    
    "investing_calendar": DataSource(
        name="Economic Calendar",
        source_type="scrape",
        url="https://www.investing.com/economic-calendar",
        api_key_env=None,
        refresh_interval_seconds=300,
        priority=2
    ),
    
    # =========================================================================
    # EXCHANGE DATA - Listings, announcements
    # =========================================================================
    "binance_announcements": DataSource(
        name="Binance Announcements",
        source_type="rss",
        url="https://www.binance.com/en/support/announcement",
        api_key_env=None,
        refresh_interval_seconds=60,
        priority=1
    ),
    
    "coinbase_blog": DataSource(
        name="Coinbase Blog",
        source_type="rss",
        url="https://blog.coinbase.com/feed",
        api_key_env=None,
        refresh_interval_seconds=60,
        priority=1
    ),
}


# ============================================================================
# EVENT DETECTOR
# ============================================================================

class EventDetector:
    """
    Ghost's eyes and ears - detects market-moving events from multiple sources.
    
    Flow:
    1. Poll data sources on schedule
    2. Parse raw data into structured events
    3. Match events to EventType enum
    4. Record events for Ghost to learn from
    """
    
    def __init__(self, event_memory: Optional[EventMemory] = None):
        self.event_memory = event_memory or EventMemory()
        self.sources = DATA_SOURCES
        self.last_check: Dict[str, datetime] = {}
        self.detected_events: List[Dict] = []
        self._running = False
        
        # Track which sources are configured (have API keys)
        self.configured_sources = self._check_configured_sources()
        
        LOGGER.info(f"[EVENT_DETECTOR] Initialized with {len(self.configured_sources)} configured sources")
    
    def _check_configured_sources(self) -> List[str]:
        """Check which data sources have API keys configured"""
        configured = []
        for name, source in self.sources.items():
            if source.api_key_env is None:
                # No API key needed (RSS, scrape)
                configured.append(name)
            elif os.environ.get(source.api_key_env):
                configured.append(name)
        return configured
    
    def get_status(self) -> Dict:
        """Get detector status - which sources are active"""
        return {
            "running": self._running,
            "total_sources": len(self.sources),
            "configured_sources": len(self.configured_sources),
            "sources": {
                name: {
                    "configured": name in self.configured_sources,
                    "api_key_env": source.api_key_env,
                    "last_check": self.last_check.get(name, "never").isoformat() if isinstance(self.last_check.get(name), datetime) else "never",
                    "refresh_interval": source.refresh_interval_seconds
                }
                for name, source in self.sources.items()
            },
            "events_detected_today": len([e for e in self.detected_events if e.get("timestamp", "") > datetime.now().replace(hour=0, minute=0).isoformat()])
        }
    
    # =========================================================================
    # NEWS DETECTION
    # =========================================================================
    
    async def check_news(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Check news sources for market-moving events"""
        events = []
        
        # Check CryptoPanic (free crypto news aggregator)
        api_key = os.environ.get("CRYPTOPANIC_API_KEY")
        if api_key:
            try:
                url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&filter=rising"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for post in data.get("results", [])[:10]:
                            event = self._parse_news_to_event(post)
                            if event:
                                events.append(event)
                self.last_check["cryptopanic"] = datetime.now()
            except Exception as e:
                LOGGER.warning(f"[EVENT_DETECTOR] CryptoPanic error: {e}")
        
        # Check NewsAPI
        api_key = os.environ.get("NEWS_API_KEY")
        if api_key:
            try:
                # Search for market-moving keywords
                keywords = "fed rate|earnings|hack|whale|elon musk|bitcoin|crypto"
                url = f"https://newsapi.org/v2/everything?q={keywords}&sortBy=publishedAt&apiKey={api_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for article in data.get("articles", [])[:10]:
                            event = self._parse_news_to_event(article)
                            if event:
                                events.append(event)
                self.last_check["newsapi"] = datetime.now()
            except Exception as e:
                LOGGER.warning(f"[EVENT_DETECTOR] NewsAPI error: {e}")
        
        return events
    
    def _parse_news_to_event(self, article: Dict) -> Optional[Dict]:
        """Parse a news article into an event if it's market-moving"""
        title = article.get("title", "") or article.get("headline", "") or ""
        content = article.get("description", "") or article.get("body", "") or ""
        text = f"{title} {content}".lower()
        
        # Match against our patterns
        for pattern in self.event_memory.patterns.values():
            for keyword in pattern.keywords:
                if keyword.lower() in text:
                    return {
                        "event_type": pattern.event_type,
                        "source": "news",
                        "title": title,
                        "timestamp": datetime.now().isoformat(),
                        "matched_keyword": keyword,
                        "confidence": 0.7,
                        "raw_data": article
                    }
        
        return None
    
    # =========================================================================
    # WHALE MOVEMENT DETECTION
    # =========================================================================
    
    async def check_whale_movements(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Check for large crypto movements (whale alerts)"""
        events = []
        
        api_key = os.environ.get("WHALE_ALERT_API_KEY")
        if api_key:
            try:
                # Get transactions over $1M in last hour
                min_value = 1000000
                url = f"https://api.whale-alert.io/v1/transactions?api_key={api_key}&min_value={min_value}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for tx in data.get("transactions", []):
                            event = self._parse_whale_tx(tx)
                            if event:
                                events.append(event)
                self.last_check["whale_alert"] = datetime.now()
            except Exception as e:
                LOGGER.warning(f"[EVENT_DETECTOR] Whale Alert error: {e}")
        
        return events
    
    def _parse_whale_tx(self, tx: Dict) -> Optional[Dict]:
        """Parse a whale transaction into an event"""
        amount_usd = tx.get("amount_usd", 0)
        from_owner = tx.get("from", {}).get("owner_type", "")
        to_owner = tx.get("to", {}).get("owner_type", "")
        symbol = tx.get("symbol", "").upper()
        
        # Detect movement to exchange (potential sell)
        if to_owner == "exchange" and amount_usd > 10_000_000:
            return {
                "event_type": EventType.WHALE_SELL.value,
                "source": "whale_alert",
                "title": f"${amount_usd:,.0f} {symbol} moved to exchange",
                "timestamp": datetime.now().isoformat(),
                "affected_symbols": [symbol],
                "confidence": 0.8,
                "raw_data": tx
            }
        
        # Detect movement from exchange (potential buy/accumulation)
        if from_owner == "exchange" and amount_usd > 10_000_000:
            return {
                "event_type": EventType.WHALE_BUY.value,
                "source": "whale_alert",
                "title": f"${amount_usd:,.0f} {symbol} withdrawn from exchange",
                "timestamp": datetime.now().isoformat(),
                "affected_symbols": [symbol],
                "confidence": 0.8,
                "raw_data": tx
            }
        
        return None
    
    # =========================================================================
    # SOCIAL MEDIA DETECTION (Elon tweets, etc.)
    # =========================================================================
    
    async def check_social_media(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Check social media for market-moving posts"""
        events = []
        
        # Check Twitter/X for Elon Musk tweets
        bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
        if bearer_token:
            try:
                # Elon's Twitter ID
                elon_id = "44196397"
                url = f"https://api.twitter.com/2/users/{elon_id}/tweets"
                headers = {"Authorization": f"Bearer {bearer_token}"}
                params = {"max_results": 5, "tweet.fields": "created_at"}
                
                async with session.get(url, headers=headers, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for tweet in data.get("data", []):
                            event = self._parse_elon_tweet(tweet)
                            if event:
                                events.append(event)
                self.last_check["twitter"] = datetime.now()
            except Exception as e:
                LOGGER.warning(f"[EVENT_DETECTOR] Twitter error: {e}")
        
        # Check LunarCrush for social sentiment spikes
        api_key = os.environ.get("LUNARCRUSH_API_KEY")
        if api_key:
            try:
                url = f"https://api.lunarcrush.com/v2?data=market&type=fast&key={api_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Check for unusual social volume
                        for coin in data.get("data", [])[:20]:
                            if coin.get("social_volume_change_24h", 0) > 200:
                                events.append({
                                    "event_type": EventType.VIRAL_SOCIAL.value,
                                    "source": "lunarcrush",
                                    "title": f"{coin.get('symbol')} social volume spike +{coin.get('social_volume_change_24h')}%",
                                    "timestamp": datetime.now().isoformat(),
                                    "affected_symbols": [coin.get("symbol")],
                                    "confidence": 0.6,
                                    "raw_data": coin
                                })
                self.last_check["lunarcrush"] = datetime.now()
            except Exception as e:
                LOGGER.warning(f"[EVENT_DETECTOR] LunarCrush error: {e}")
        
        return events
    
    def _parse_elon_tweet(self, tweet: Dict) -> Optional[Dict]:
        """Parse an Elon tweet for crypto relevance"""
        text = tweet.get("text", "").lower()
        
        crypto_keywords = ["bitcoin", "btc", "doge", "dogecoin", "crypto", "ethereum", "eth"]
        
        for keyword in crypto_keywords:
            if keyword in text:
                return {
                    "event_type": EventType.ELON_TWEET.value,
                    "source": "twitter",
                    "title": f"Elon tweeted about {keyword.upper()}",
                    "timestamp": tweet.get("created_at", datetime.now().isoformat()),
                    "affected_symbols": ["DOGE", "BTC"] if "doge" in text else ["BTC"],
                    "confidence": 0.9,
                    "raw_data": tweet
                }
        
        return None
    
    # =========================================================================
    # EARNINGS & CORPORATE ACTIONS
    # =========================================================================
    
    async def check_earnings(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Check for earnings releases"""
        events = []
        
        # Check Finnhub for earnings calendar
        api_key = os.environ.get("FINNHUB_API_KEY")
        if api_key:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                url = f"https://finnhub.io/api/v1/calendar/earnings?from={today}&to={today}&token={api_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for earning in data.get("earningsCalendar", []):
                            # If actual > estimate = beat
                            actual = earning.get("epsActual")
                            estimate = earning.get("epsEstimate")
                            if actual is not None and estimate is not None:
                                if actual > estimate:
                                    event_type = EventType.EARNINGS_BEAT.value
                                    title = f"{earning.get('symbol')} beat earnings (${actual} vs ${estimate})"
                                else:
                                    event_type = EventType.EARNINGS_MISS.value
                                    title = f"{earning.get('symbol')} missed earnings (${actual} vs ${estimate})"
                                
                                events.append({
                                    "event_type": event_type,
                                    "source": "finnhub",
                                    "title": title,
                                    "timestamp": datetime.now().isoformat(),
                                    "affected_symbols": [earning.get("symbol")],
                                    "confidence": 0.95,
                                    "raw_data": earning
                                })
                self.last_check["finnhub"] = datetime.now()
            except Exception as e:
                LOGGER.warning(f"[EVENT_DETECTOR] Finnhub error: {e}")
        
        return events
    
    # =========================================================================
    # ECONOMIC DATA (Fed, Inflation, etc.)
    # =========================================================================
    
    async def check_economic_data(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Check for economic data releases"""
        events = []
        
        # Check FRED for recent releases
        api_key = os.environ.get("FRED_API_KEY")
        if api_key:
            try:
                # Check Fed Funds Rate
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={api_key}&file_type=json&sort_order=desc&limit=2"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        observations = data.get("observations", [])
                        if len(observations) >= 2:
                            current = float(observations[0].get("value", 0))
                            previous = float(observations[1].get("value", 0))
                            
                            if current > previous:
                                events.append({
                                    "event_type": EventType.FED_RATE_HIKE.value,
                                    "source": "fred",
                                    "title": f"Fed raised rates to {current}%",
                                    "timestamp": observations[0].get("date"),
                                    "affected_symbols": ["BTC", "ETH", "SPY", "QQQ"],
                                    "confidence": 1.0,
                                    "raw_data": observations[0]
                                })
                            elif current < previous:
                                events.append({
                                    "event_type": EventType.FED_RATE_CUT.value,
                                    "source": "fred",
                                    "title": f"Fed cut rates to {current}%",
                                    "timestamp": observations[0].get("date"),
                                    "affected_symbols": ["BTC", "ETH", "SPY", "QQQ"],
                                    "confidence": 1.0,
                                    "raw_data": observations[0]
                                })
                self.last_check["fred"] = datetime.now()
            except Exception as e:
                LOGGER.warning(f"[EVENT_DETECTOR] FRED error: {e}")
        
        return events
    
    # =========================================================================
    # EXCHANGE LISTINGS
    # =========================================================================
    
    async def check_exchange_listings(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Check for new exchange listings"""
        events = []
        
        # This would scrape exchange announcement pages
        # For now, we'll use keyword detection in news
        
        return events
    
    # =========================================================================
    # PRICE ANOMALY DETECTION
    # =========================================================================
    
    async def check_price_anomalies(self, session: aiohttp.ClientSession) -> List[Dict]:
        """
        Detect unusual price movements that indicate something happened.
        If price moved 10%+ in an hour, SOMETHING caused it.
        """
        events = []
        
        # Check CoinGecko for rapid price changes (free API)
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&sparkline=false&price_change_percentage=1h"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for coin in data:
                        change_1h = coin.get("price_change_percentage_1h_in_currency", 0) or 0
                        symbol = coin.get("symbol", "").upper()
                        
                        if change_1h > 15:
                            events.append({
                                "event_type": EventType.BTC_PRICE_SURGE.value if symbol == "BTC" else "unknown_pump",
                                "source": "coingecko",
                                "title": f"{symbol} pumped {change_1h:.1f}% in 1 hour",
                                "timestamp": datetime.now().isoformat(),
                                "affected_symbols": [symbol],
                                "confidence": 0.5,  # Low confidence - we don't know WHY
                                "needs_investigation": True,
                                "raw_data": coin
                            })
                        elif change_1h < -15:
                            events.append({
                                "event_type": EventType.BTC_PRICE_CRASH.value if symbol == "BTC" else "unknown_dump",
                                "source": "coingecko",
                                "title": f"{symbol} dumped {abs(change_1h):.1f}% in 1 hour",
                                "timestamp": datetime.now().isoformat(),
                                "affected_symbols": [symbol],
                                "confidence": 0.5,
                                "needs_investigation": True,
                                "raw_data": coin
                            })
        except Exception as e:
            LOGGER.warning(f"[EVENT_DETECTOR] CoinGecko error: {e}")
        
        return events
    
    # =========================================================================
    # MAIN DETECTION LOOP
    # =========================================================================
    
    async def detect_all(self) -> List[Dict]:
        """Run all detection methods and return combined events"""
        all_events = []
        
        async with aiohttp.ClientSession() as session:
            # Run all checks in parallel
            results = await asyncio.gather(
                self.check_news(session),
                self.check_whale_movements(session),
                self.check_social_media(session),
                self.check_earnings(session),
                self.check_economic_data(session),
                self.check_price_anomalies(session),
                return_exceptions=True
            )
            
            for result in results:
                if isinstance(result, list):
                    all_events.extend(result)
                elif isinstance(result, Exception):
                    LOGGER.error(f"[EVENT_DETECTOR] Detection error: {result}")
        
        # Deduplicate events
        seen = set()
        unique_events = []
        for event in all_events:
            key = f"{event.get('event_type')}_{event.get('title', '')[:50]}"
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        # Store and record events
        self.detected_events.extend(unique_events)
        
        # Record in event memory for learning
        for event in unique_events:
            self.event_memory.record_event(
                event_type=event.get("event_type"),
                affected_symbols=event.get("affected_symbols", []),
                source=event.get("source"),
                details=event.get("title")
            )
        
        LOGGER.info(f"[EVENT_DETECTOR] Detected {len(unique_events)} new events")
        return unique_events
    
    async def run_continuous(self, interval_seconds: int = 60):
        """Run continuous event detection"""
        self._running = True
        LOGGER.info(f"[EVENT_DETECTOR] Starting continuous detection (interval: {interval_seconds}s)")
        
        while self._running:
            try:
                events = await self.detect_all()
                if events:
                    LOGGER.info(f"[EVENT_DETECTOR] Found {len(events)} events this cycle")
            except Exception as e:
                LOGGER.error(f"[EVENT_DETECTOR] Error in detection cycle: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop(self):
        """Stop continuous detection"""
        self._running = False
        LOGGER.info("[EVENT_DETECTOR] Stopping detection")


# ============================================================================
# WHAT GHOST NEEDS TO FIND ANSWERS
# ============================================================================

"""
HOW GHOST FINDS ANSWERS:

1. REAL-TIME DATA SOURCES (APIs needed):
   ✓ Whale Alert API - Whale movements ($50/mo for good tier)
   ✓ CryptoPanic API - Free crypto news aggregator  
   ✓ Twitter/X API - Elon tweets, crypto influencers ($100/mo)
   ✓ Finnhub API - Earnings, corporate actions (free tier available)
   ✓ FRED API - Fed rates, economic data (free)
   ✓ CoinGecko - Price anomaly detection (free)
   
2. FREE DETECTION METHODS:
   ✓ Price anomaly detection - If price moved 15%+ in 1 hour, SOMETHING happened
   ✓ News keyword matching - Scan headlines for our 184 event types
   ✓ RSS feeds - Exchange announcements (Binance, Coinbase)
   
3. GHOST LEARNING FLOW:
   
   [Data Sources] → [Event Detector] → [Event Memory] → [Ghost Predictions]
         ↑                 ↓                  ↓                   ↓
    News APIs         Detects         Stores pattern        Adjusts
    Whale Alert       events          event → reaction      confidence
    Twitter/X                         learns over time
    Price feeds
    
4. TO ENABLE DETECTION, SET THESE ENV VARS:
   
   # Free tier available:
   export CRYPTOPANIC_API_KEY="your_key"    # Free - crypto news
   export FINNHUB_API_KEY="your_key"        # Free tier - earnings
   export FRED_API_KEY="your_key"           # Free - Fed data
   
   # Paid but valuable:
   export WHALE_ALERT_API_KEY="your_key"    # $50/mo - whale moves
   export TWITTER_BEARER_TOKEN="your_key"   # $100/mo - Elon tweets
   export LUNARCRUSH_API_KEY="your_key"     # Paid - social sentiment
   
   # CoinGecko (price anomaly) is always free
   
5. GHOST ANSWERS COME FROM:
   - Memory: "Last time Elon tweeted, DOGE pumped 30% then dumped"
   - Detection: "Elon just tweeted about DOGE"
   - Prediction: "Reduce confidence on DOGE bullish predictions"
   
   Ghost REMEMBERS + Ghost SEES = Ghost KNOWS
"""


# Quick test
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    detector = EventDetector()
    print("\n=== EVENT DETECTOR STATUS ===")
    status = detector.get_status()
    print(f"Configured sources: {status['configured_sources']}/{status['total_sources']}")
    print("\nSources:")
    for name, info in status['sources'].items():
        configured = "✓" if info['configured'] else "✗"
        key_needed = f"(needs {info['api_key_env']})" if info['api_key_env'] and not info['configured'] else ""
        print(f"  {configured} {name} {key_needed}")
    
    print("\n=== RUNNING DETECTION ===")
    events = asyncio.run(detector.detect_all())
    print(f"\nDetected {len(events)} events:")
    for event in events[:5]:
        print(f"  - [{event['event_type']}] {event['title']}")
