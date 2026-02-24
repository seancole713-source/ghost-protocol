#!/usr/bin/env python3
"""
🧠 GHOST NEWS BRAIN - Claude-powered market news analysis

UPDATED: Now fetches REAL news from:
1. CryptoPanic API (crypto news) - FREE, already configured
2. Reuters RSS feeds (geopolitical/market news) - FREE, already configured

Then sends headlines to Claude for analysis against pending predictions.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Handle optional imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    feedparser = None
    FEEDPARSER_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None
    POSTGRES_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class GhostNewsBrain:
    """
    Uses Claude to analyze breaking news and its impact on predictions.
    
    Workflow:
    1. Fetch headlines from CryptoPanic + Reuters RSS
    2. Send headlines to Claude for analysis
    3. Cross-reference with pending predictions
    4. Alert via Telegram if predictions may be wrong
    """
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        self.db_url = os.getenv("DATABASE_URL")
        
        # News source configs
        self.cryptopanic_key = os.getenv("CRYPTOPANIC_API_KEY")
        self.rss_feeds = [
            # Financial
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",  # CNBC Top
            "https://feeds.marketwatch.com/marketwatch/topstories/",  # MarketWatch
            "https://seekingalpha.com/market_currents.xml",           # Seeking Alpha
            "https://www.investing.com/rss/news.rss",                 # Investing.com
            "https://www.nasdaq.com/feed/nasdaq-original/rss.xml",    # Nasdaq
            # Crypto
            "https://www.coindesk.com/arc/outboundfeeds/rss/",        # CoinDesk
            "https://cointelegraph.com/rss",                          # Cointelegraph
            "https://decrypt.co/feed",                                # Decrypt
            # Geopolitical
            "https://feeds.bbci.co.uk/news/world/rss.xml",            # BBC World
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", # NYT World
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",  # NYT Business
            "https://www.aljazeera.com/xml/rss/all.xml",              # Al Jazeera
            # Energy
            "https://oilprice.com/rss/main",                          # OilPrice
            # Fed
            "https://www.federalreserve.gov/feeds/press_all.xml",     # Federal Reserve
        ]
        
        # Add any custom feeds from env
        custom_feeds = os.getenv("NEWS_MANUAL_FEEDS", "")
        if custom_feeds:
            self.rss_feeds.extend([f.strip() for f in custom_feeds.split(",") if f.strip()])
        
        self._ensure_table()
    
    def _ensure_table(self):
        """Create required tables if not exists"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            # News analysis table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS news_analysis (
                    analysis_id SERIAL PRIMARY KEY,
                    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    headlines_fetched INTEGER DEFAULT 0,
                    raw_response TEXT,
                    events_found INTEGER DEFAULT 0,
                    predictions_affected INTEGER DEFAULT 0,
                    alert_sent BOOLEAN DEFAULT FALSE,
                    summary TEXT
                );
            """)
            
            # Guardian alerts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS guardian_alerts (
                    alert_id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    alert_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) DEFAULT 'INFO',
                    message TEXT,
                    price_at_alert DECIMAL(18,8),
                    confidence DECIMAL(5,4),
                    prediction_id INTEGER,
                    news_event_id INTEGER,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for guardian_alerts
            cur.execute("CREATE INDEX IF NOT EXISTS idx_guardian_symbol ON guardian_alerts(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_guardian_severity ON guardian_alerts(severity);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_guardian_created ON guardian_alerts(created_at);")
            
            conn.commit()
            conn.close()
            LOGGER.info("[NEWS BRAIN] Database tables ensured (news_analysis, guardian_alerts)")
        except Exception as e:
            LOGGER.error(f"Failed to create tables: {e}")
    
    async def fetch_cryptopanic_news(self) -> List[Dict]:
        """Fetch latest crypto news from CryptoPanic API"""
        if not self.cryptopanic_key or not HTTPX_AVAILABLE:
            return []
        
        headlines = []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://cryptopanic.com/api/v1/posts/",
                    params={
                        "auth_token": self.cryptopanic_key,
                        "filter": "important",
                        "kind": "news",
                        "public": "true",
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("results", [])[:20]:
                        headlines.append({
                            "title": item.get("title", ""),
                            "source": item.get("source", {}).get("title", "CryptoPanic"),
                            "url": item.get("url", ""),
                            "published": item.get("published_at", ""),
                            "symbols": [c.get("code", "") for c in item.get("currencies", [])],
                            "type": "crypto"
                        })
        except Exception as e:
            LOGGER.error(f"CryptoPanic fetch error: {e}")
        
        return headlines
    
    async def fetch_rss_news(self) -> List[Dict]:
        """Fetch latest news from RSS feeds with async + timeout"""
        if not FEEDPARSER_AVAILABLE or not HTTPX_AVAILABLE:
            return []
        
        headlines = []
        
        async def fetch_single_feed(feed_url: str) -> List[Dict]:
            """Fetch a single RSS feed with timeout"""
            feed_headlines = []
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(feed_url)
                    if resp.status_code == 200:
                        feed = feedparser.parse(resp.text)
                        for entry in feed.entries[:8]:  # Top 8 per feed
                            published = ""
                            if hasattr(entry, 'published'):
                                published = entry.published
                            elif hasattr(entry, 'updated'):
                                published = entry.updated
                            
                            feed_headlines.append({
                                "title": entry.get("title", ""),
                                "source": feed.feed.get("title", feed_url.split("/")[2]),
                                "url": entry.get("link", ""),
                                "published": published,
                                "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
                                "type": "general"
                            })
            except Exception as e:
                LOGGER.debug(f"RSS fetch error for {feed_url}: {e}")
            return feed_headlines
        
        # Fetch all feeds in parallel with 10 second total timeout
        try:
            tasks = [fetch_single_feed(url) for url in self.rss_feeds]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=15.0
            )
            for result in results:
                if isinstance(result, list):
                    headlines.extend(result)
        except asyncio.TimeoutError:
            LOGGER.warning("RSS fetch timed out after 15 seconds")
        except Exception as e:
            LOGGER.error(f"RSS fetch error: {e}")
        
        return headlines
        
        return headlines
    
    async def fetch_all_news(self) -> List[Dict]:
        """Fetch news from all sources"""
        all_headlines = []
        
        crypto_task = self.fetch_cryptopanic_news()
        rss_task = self.fetch_rss_news()
        
        crypto_news, rss_news = await asyncio.gather(crypto_task, rss_task)
        
        all_headlines.extend(crypto_news)
        all_headlines.extend(rss_news)
        
        return all_headlines[:50]
    
    def get_pending_predictions(self) -> List[Dict]:
        """Get all pending paper trades to check against news"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return []
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT symbol, signal_direction, signal_confidence, entry_price, entry_time
                FROM paper_trades 
                WHERE outcome = 'PENDING'
                ORDER BY entry_time DESC
                LIMIT 100
            """)
            trades = cur.fetchall()
            conn.close()
            return [dict(t) for t in trades]
        except Exception as e:
            LOGGER.error(f"Failed to fetch predictions: {e}")
            return []
    
    async def analyze_news(self) -> Dict:
        """Main function: Fetch news, send to Claude, check predictions."""
        if not self.client:
            return {
                "ok": False,
                "error": "Anthropic client not available",
                "anthropic_available": ANTHROPIC_AVAILABLE,
                "api_key_present": bool(self.api_key)
            }
        
        LOGGER.info("📰 Fetching news from CryptoPanic and RSS feeds...")
        headlines = await self.fetch_all_news()
        
        if not headlines:
            return {
                "ok": True,
                "message": "No headlines fetched - check API keys and feeds",
                "headlines_fetched": 0,
                "major_events": [],
                "predictions_at_risk": [],
                "action_required": False
            }
        
        pending = self.get_pending_predictions()
        
        predictions_summary = []
        symbols_set = set()
        for p in pending:
            symbols_set.add(p['symbol'])
            predictions_summary.append(
                f"- {p['symbol']}: {p['signal_direction']} @ {float(p['signal_confidence']):.1%}"
            )
        
        predictions_text = "\n".join(predictions_summary[:50])
        symbols_text = ", ".join(sorted(symbols_set))
        
        headlines_text = ""
        for i, h in enumerate(headlines[:30], 1):
            headlines_text += f"{i}. [{h['type'].upper()}] {h['title']}\n"
            if h.get('summary'):
                headlines_text += f"   Summary: {h['summary'][:100]}...\n"
        
        prompt = f"""You are Ghost Protocol's News Brain. Analyze these REAL news headlines and identify market-moving events that could affect our predictions.

CURRENT DATE/TIME: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

═══════════════════════════════════════════════════════════════
LATEST NEWS HEADLINES (fetched just now):
═══════════════════════════════════════════════════════════════
{headlines_text}

═══════════════════════════════════════════════════════════════
OUR PENDING PREDICTIONS:
═══════════════════════════════════════════════════════════════
{predictions_text}

SYMBOLS WE'RE TRACKING: {symbols_text}

═══════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════

1. Identify MAJOR market-moving events from these headlines
2. For each major event, determine affected sectors and symbols
3. Cross-reference with our predictions - flag any that may be WRONG

Respond in this exact JSON format:
{{
    "analysis_time": "{datetime.utcnow().isoformat()}",
    "headlines_analyzed": {len(headlines)},
    "major_events": [
        {{
            "headline": "The actual headline",
            "severity": "CRITICAL|HIGH|MEDIUM|LOW",
            "event_type": "GEOPOLITICAL|ECONOMIC|CORPORATE|REGULATORY|CRYPTO",
            "summary": "1-2 sentence explanation of market impact",
            "affected_sectors": ["oil", "defense", "crypto", "tech", "financials"],
            "bullish_symbols": ["SYM1", "SYM2"],
            "bearish_symbols": ["SYM1", "SYM2"]
        }}
    ],
    "predictions_at_risk": [
        {{
            "symbol": "SYM",
            "our_prediction": "UP|DOWN",
            "likely_actual": "UP|DOWN",
            "reason": "Why this prediction may be wrong based on news",
            "risk_level": "HIGH|MEDIUM|LOW"
        }}
    ],
    "market_summary": "2-3 sentence overall market outlook based on today's news",
    "action_required": true|false,
    "recommendation": "Specific advice for Ghost trader"
}}

SEVERITY GUIDE:
- CRITICAL: War, invasion, major attack, market crash, pandemic
- HIGH: Fed surprise, major bankruptcy, significant geopolitical event
- MEDIUM: Important earnings, policy changes, sector-specific news
- LOW: Routine news, minor updates

Only set action_required=true if there are HIGH or CRITICAL events affecting our predictions."""

        full_response = ""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            for block in response.content:
                if hasattr(block, 'text'):
                    full_response += block.text
            
            json_start = full_response.find('{')
            json_end = full_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = full_response[json_start:json_end]
                analysis = json.loads(json_str)
                analysis["ok"] = True
                analysis["headlines_fetched"] = len(headlines)
            else:
                analysis = {
                    "ok": False,
                    "error": "Could not parse JSON from Claude response",
                    "raw_response": full_response[:500],
                    "headlines_fetched": len(headlines),
                    "major_events": [],
                    "predictions_at_risk": [],
                    "action_required": False
                }
                
        except json.JSONDecodeError as e:
            analysis = {
                "ok": False,
                "error": f"JSON parse error: {str(e)}",
                "headlines_fetched": len(headlines),
                "major_events": [],
                "predictions_at_risk": [],
                "action_required": False
            }
        except Exception as e:
            analysis = {
                "ok": False,
                "error": f"Claude API error: {str(e)}",
                "headlines_fetched": len(headlines),
                "major_events": [],
                "predictions_at_risk": [],
                "action_required": False
            }
        
        self._log_analysis(analysis, full_response)
        
        return analysis
    
    def _log_analysis(self, analysis: Dict, raw_response: str):
        """Log analysis to database"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO news_analysis 
                (headlines_fetched, raw_response, events_found, predictions_affected, summary)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                analysis.get("headlines_fetched", 0),
                raw_response[:10000] if raw_response else "",
                len(analysis.get("major_events", [])),
                len(analysis.get("predictions_at_risk", [])),
                analysis.get("market_summary", "")
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to log analysis: {e}")
    
    async def send_alert(self, analysis: Dict) -> bool:
        """Send Telegram alert if action required"""
        if not analysis.get("action_required"):
            return False
        
        if not HTTPX_AVAILABLE:
            LOGGER.warning("httpx not available for Telegram")
            return False
        
        events = analysis.get("major_events", [])
        at_risk = analysis.get("predictions_at_risk", [])
        
        severity_emoji = {
            "CRITICAL": "🚨🚨🚨",
            "HIGH": "⚠️⚠️",
            "MEDIUM": "📰",
            "LOW": "📋"
        }
        
        max_severity = "LOW"
        for event in events:
            sev = event.get("severity", "LOW")
            if sev == "CRITICAL":
                max_severity = "CRITICAL"
                break
            elif sev == "HIGH" and max_severity not in ["CRITICAL"]:
                max_severity = "HIGH"
            elif sev == "MEDIUM" and max_severity not in ["CRITICAL", "HIGH"]:
                max_severity = "MEDIUM"
        
        emoji = severity_emoji.get(max_severity, "📋")
        
        message = f"""{emoji} GHOST NEWS BRAIN ALERT {emoji}

📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
📰 Headlines Analyzed: {analysis.get('headlines_fetched', 0)}

"""
        
        if events:
            message += "🌍 MAJOR EVENTS:\n\n"
            for event in events[:3]:
                message += f"▸ {event.get('headline', 'Unknown')}\n"
                message += f"  Severity: {event.get('severity', '?')} | Type: {event.get('event_type', '?')}\n"
                if event.get('summary'):
                    message += f"  {event['summary'][:100]}\n"
                message += "\n"
        
        if at_risk:
            message += f"⚠️ {len(at_risk)} PREDICTIONS AT RISK:\n\n"
            for pred in at_risk[:5]:
                message += f"• {pred['symbol']}: Predicted {pred['our_prediction']}, "
                message += f"likely {pred['likely_actual']}\n"
                message += f"  → {pred['reason'][:80]}\n"
        
        rec = analysis.get('recommendation', 'Review affected predictions')
        message += f"\n💡 RECOMMENDATION:\n{rec}"
        
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
            if bot_token and chat_id:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": message,
                            "parse_mode": "HTML"
                        }
                    )
                return True
        except Exception as e:
            LOGGER.error(f"Failed to send Telegram alert: {e}")
        
        return False
    
    def get_status(self) -> Dict:
        """Get current status of the News Brain"""
        return {
            "enabled": bool(self.client),
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "api_key_present": bool(self.api_key),
            "api_key_preview": f"{self.api_key[:8]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 12 else None,
            "cryptopanic_configured": bool(self.cryptopanic_key),
            "rss_feeds_count": len(self.rss_feeds),
            "rss_feeds": self.rss_feeds,
            "postgres_available": POSTGRES_AVAILABLE,
            "db_configured": bool(self.db_url),
            "httpx_available": HTTPX_AVAILABLE,
            "feedparser_available": FEEDPARSER_AVAILABLE,
        }
    
    def get_cached_analysis(self, symbol: str = None) -> Dict:
        """
        Get most recent news analysis from database (cached for predictions).
        
        This allows predictions to use news context WITHOUT triggering
        expensive Claude API calls on every prediction.
        
        Args:
            symbol: If provided, filter for symbol-specific sentiment
        
        Returns:
            {
                "ok": True/False,
                "analysis_time": timestamp,
                "symbol_sentiment": {
                    "RNDR": {"sentiment_score": 0.6, "confidence": 0.8, "affected_by": [...]},
                    ...
                },
                "major_events": [...],
                "market_summary": "...",
                "cache_age_minutes": 15
            }
        """
        if not self.db_url or not POSTGRES_AVAILABLE:
            return {"ok": False, "error": "Database not available"}
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get most recent analysis (should be < 30 minutes old)
            cur.execute("""
                SELECT 
                    analysis_id,
                    analysis_time,
                    headlines_fetched,
                    raw_response,
                    events_found,
                    summary,
                    EXTRACT(EPOCH FROM (NOW() - analysis_time))/60 as age_minutes
                FROM news_analysis 
                ORDER BY analysis_time DESC 
                LIMIT 1
            """)
            
            row = cur.fetchone()
            conn.close()
            
            if not row:
                return {"ok": False, "error": "No cached analysis found"}
            
            # Check if cache is too old (> 60 minutes = stale)
            age_minutes = float(row['age_minutes'])
            if age_minutes > 60:
                return {
                    "ok": False,
                    "error": f"Cache too old ({age_minutes:.0f} minutes)",
                    "stale": True
                }
            
            # Parse raw response (stored as JSON string)
            try:
                import json
                raw_analysis = json.loads(row['raw_response']) if row['raw_response'] else {}
            except:
                raw_analysis = {}
            
            # Build symbol-specific sentiment map
            symbol_sentiment = {}
            major_events = raw_analysis.get("major_events", [])
            
            for event in major_events:
                # Bullish symbols
                for sym in event.get("bullish_symbols", []):
                    if sym not in symbol_sentiment:
                        symbol_sentiment[sym] = {
                            "sentiment_score": 0.0,
                            "confidence": 0.0,
                            "affected_by": []
                        }
                    symbol_sentiment[sym]["sentiment_score"] += 0.3  # Bullish boost
                    symbol_sentiment[sym]["affected_by"].append({
                        "headline": event.get("headline"),
                        "sentiment": "bullish",
                        "type": event.get("event_type"),
                        "severity": event.get("severity")
                    })
                
                # Bearish symbols
                for sym in event.get("bearish_symbols", []):
                    if sym not in symbol_sentiment:
                        symbol_sentiment[sym] = {
                            "sentiment_score": 0.0,
                            "confidence": 0.0,
                            "affected_by": []
                        }
                    symbol_sentiment[sym]["sentiment_score"] -= 0.3  # Bearish penalty
                    symbol_sentiment[sym]["affected_by"].append({
                        "headline": event.get("headline"),
                        "sentiment": "bearish",
                        "type": event.get("event_type"),
                        "severity": event.get("severity")
                    })
            
            # Normalize sentiment scores (-1 to +1) and calculate confidence
            for sym in symbol_sentiment:
                score = symbol_sentiment[sym]["sentiment_score"]
                affected_count = len(symbol_sentiment[sym]["affected_by"])
                
                # Clamp to -1/+1 range
                score = max(-1.0, min(1.0, score))
                symbol_sentiment[sym]["sentiment_score"] = round(score, 2)
                
                # Confidence based on event count and severity
                confidence = min(0.9, 0.5 + (affected_count * 0.15))
                symbol_sentiment[sym]["confidence"] = round(confidence, 2)
            
            result = {
                "ok": True,
                "analysis_time": row['analysis_time'].isoformat() if row['analysis_time'] else None,
                "symbol_sentiment": symbol_sentiment,
                "major_events": major_events,
                "market_summary": raw_analysis.get("market_summary", ""),
                "cache_age_minutes": round(age_minutes, 1),
                "headlines_analyzed": row.get('headlines_fetched', 0),
            }
            
            # If specific symbol requested, return just that
            if symbol:
                sym_data = symbol_sentiment.get(symbol.upper())
                if sym_data:
                    result["symbol"] = symbol.upper()
                    result["sentiment"] = sym_data
                else:
                    # Symbol not mentioned in recent news
                    result["symbol"] = symbol.upper()
                    result["sentiment"] = None
                    result["note"] = "Symbol not mentioned in recent news"
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Failed to get cached analysis: {e}")
            return {"ok": False, "error": str(e)}
    
    def get_last_analysis(self) -> Dict:
        """Get the most recent analysis from database"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return {"status": "no_database", "message": "Database not configured"}
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT * FROM news_analysis 
                ORDER BY analysis_time DESC 
                LIMIT 1
            """)
            row = cur.fetchone()
            conn.close()
            if row:
                return dict(row)
            return {"status": "no_analysis", "message": "No analyses yet"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get analysis history from database"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return []
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT analysis_id, analysis_time, headlines_fetched, events_found, 
                       predictions_affected, alert_sent, summary
                FROM news_analysis 
                ORDER BY analysis_time DESC 
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            LOGGER.error(f"Failed to get history: {e}")
            return []

    # ========================================================================
    # BREAKING NEWS AUTO-PAUSE
    # ========================================================================
    
    def _set_trading_paused(self, paused: bool, reason: str, duration_hours: int = 4):
        """
        Set trading pause flag in database.
        When paused, no new trades should be opened.
        """
        if not self.db_url or not POSTGRES_AVAILABLE:
            return False
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            # Create system_state table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key VARCHAR(50) PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            if paused:
                pause_until = datetime.utcnow() + timedelta(hours=duration_hours)
                state = json.dumps({
                    "paused": True,
                    "reason": reason,
                    "paused_at": datetime.utcnow().isoformat(),
                    "pause_until": pause_until.isoformat(),
                    "duration_hours": duration_hours
                })
            else:
                state = json.dumps({"paused": False})
            
            cur.execute("""
                INSERT INTO system_state (key, value, updated_at) 
                VALUES ('trading_pause', %s, CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = CURRENT_TIMESTAMP
            """, (state, state))
            
            conn.commit()
            conn.close()
            
            LOGGER.warning(f"🛑 TRADING {'PAUSED' if paused else 'RESUMED'}: {reason}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to set trading pause: {e}")
            return False
    
    def get_trading_pause_status(self) -> Dict:
        """Get current trading pause status"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return {"paused": False, "reason": "Database not available"}
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute("SELECT value FROM system_state WHERE key = 'trading_pause'")
            row = cur.fetchone()
            conn.close()
            
            if not row:
                return {"paused": False, "reason": "No pause state set"}
            
            state = json.loads(row[0])
            
            # Check if pause has expired
            if state.get("paused") and state.get("pause_until"):
                pause_until = datetime.fromisoformat(state["pause_until"])
                if datetime.utcnow() > pause_until:
                    self._set_trading_paused(False, "Auto-pause expired")
                    return {"paused": False, "reason": "Auto-pause expired"}
            
            return state
        except Exception as e:
            return {"paused": False, "error": str(e)}
    
    def _create_guardian_alert(self, symbol: str, alert_type: str, severity: str, 
                                message: str, news_event: Dict = None):
        """Create a guardian alert for a critical news event"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO guardian_alerts 
                (symbol, alert_type, severity, message, created_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING alert_id
            """, (symbol, alert_type, severity, message))
            
            alert_id = cur.fetchone()[0]
            conn.commit()
            conn.close()
            
            LOGGER.info(f"[GUARDIAN ALERT] Created alert {alert_id}: {severity} - {message[:50]}")
            return alert_id
        except Exception as e:
            LOGGER.error(f"Failed to create guardian alert: {e}")
            return None
    
    async def handle_critical_event(self, event: Dict, auto_pause: bool = True) -> Dict:
        """
        Handle a CRITICAL severity news event.
        - Creates guardian alerts for affected symbols
        - Optionally pauses trading
        - Sends urgent Telegram notification
        
        Args:
            event: The major event dict from analysis
            auto_pause: Whether to automatically pause trading (default True)
        
        Returns:
            Dict with actions taken
        """
        actions_taken = {
            "event": event.get("headline", "Unknown event"),
            "severity": event.get("severity", "UNKNOWN"),
            "alerts_created": [],
            "trading_paused": False,
            "telegram_sent": False
        }
        
        # Create alerts for all affected symbols
        affected = []
        affected.extend(event.get("bullish_symbols", []))
        affected.extend(event.get("bearish_symbols", []))
        
        for symbol in set(affected):
            alert_id = self._create_guardian_alert(
                symbol=symbol,
                alert_type="NEWS_CRITICAL",
                severity="CRITICAL",
                message=f"{event.get('headline', 'Major event')}: {event.get('summary', '')}"
            )
            if alert_id:
                actions_taken["alerts_created"].append({"symbol": symbol, "alert_id": alert_id})
        
        # Auto-pause trading for CRITICAL events
        if auto_pause:
            pause_reason = f"CRITICAL NEWS: {event.get('headline', 'Unknown event')[:100]}"
            if self._set_trading_paused(True, pause_reason, duration_hours=4):
                actions_taken["trading_paused"] = True
        
        # Send urgent Telegram
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
            if bot_token and chat_id and HTTPX_AVAILABLE:
                message = f"""🚨🚨🚨 CRITICAL NEWS EVENT 🚨🚨🚨

⚡ {event.get('headline', 'Unknown')}

📊 Type: {event.get('event_type', 'Unknown')}
📝 {event.get('summary', 'No summary')}

🎯 Affected: {', '.join(affected[:10])}

{'🛑 TRADING AUTO-PAUSED FOR 4 HOURS' if actions_taken['trading_paused'] else '⚠️ MANUAL ACTION REQUIRED'}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"""
                
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        json={"chat_id": chat_id, "text": message}
                    )
                actions_taken["telegram_sent"] = True
        except Exception as e:
            LOGGER.error(f"Failed to send critical event telegram: {e}")
        
        return actions_taken
    
    async def analyze_news_with_auto_pause(self) -> Dict:
        """
        Enhanced analyze_news that automatically handles CRITICAL events.
        
        This is the main entry point for production use - it:
        1. Analyzes all news
        2. Auto-pauses trading on CRITICAL events
        3. Creates guardian alerts
        4. Sends Telegram notifications
        
        Returns:
            Dict with analysis results and any auto-pause actions taken
        """
        # Run standard analysis
        analysis = await self.analyze_news()
        
        analysis["auto_pause_actions"] = []
        analysis["trading_paused"] = False
        
        # Check for CRITICAL events
        for event in analysis.get("major_events", []):
            if event.get("severity") == "CRITICAL":
                LOGGER.warning(f"🚨 CRITICAL EVENT DETECTED: {event.get('headline', 'Unknown')}")
                
                actions = await self.handle_critical_event(event, auto_pause=True)
                analysis["auto_pause_actions"].append(actions)
                
                if actions.get("trading_paused"):
                    analysis["trading_paused"] = True
        
        # Also check HIGH severity for predictions at risk
        for event in analysis.get("major_events", []):
            if event.get("severity") == "HIGH":
                # Create guardian alerts for HIGH events but don't pause
                affected = set(event.get("bullish_symbols", []) + event.get("bearish_symbols", []))
                for symbol in affected:
                    self._create_guardian_alert(
                        symbol=symbol,
                        alert_type="NEWS_HIGH_IMPACT",
                        severity="HIGH",
                        message=f"{event.get('headline', 'Major event')}: {event.get('summary', '')}"
                    )
        
        # Send standard alert if action required
        await self.send_alert(analysis)
        
        return analysis
    
    def resume_trading(self, reason: str = "Manual resume") -> bool:
        """Manually resume trading after a pause"""
        return self._set_trading_paused(False, reason)


# Singleton
_news_brain: Optional[GhostNewsBrain] = None


def get_news_brain() -> GhostNewsBrain:
    """Get or create the news brain singleton"""
    global _news_brain
    if _news_brain is None:
        _news_brain = GhostNewsBrain()
    return _news_brain


def reset_news_brain():
    """Reset singleton (useful if env vars change)"""
    global _news_brain
    _news_brain = None


def analyze_breaking_news(pending_predictions: List[Dict] = None) -> Dict:
    """Convenience function - runs async analyze_news"""
    brain = get_news_brain()
    return asyncio.run(brain.analyze_news())


async def test_news_brain():
    """Test the news brain"""
    print("🧠 Ghost News Brain - Testing...")
    print("=" * 60)
    
    brain = GhostNewsBrain()
    
    print("\n📊 STATUS:")
    status = brain.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    print("\n📰 FETCHING NEWS...")
    headlines = await brain.fetch_all_news()
    print(f"  Fetched {len(headlines)} headlines")
    
    for h in headlines[:5]:
        print(f"  - [{h['type']}] {h['title'][:60]}...")
    
    if brain.client:
        print("\n🔍 RUNNING FULL ANALYSIS...")
        analysis = await brain.analyze_news()
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print("\n⚠️ Skipping analysis - Anthropic client not configured")


if __name__ == "__main__":
    asyncio.run(test_news_brain())
