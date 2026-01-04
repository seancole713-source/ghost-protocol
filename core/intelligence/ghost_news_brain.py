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
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.reuters.com/reuters/technologyNews",
        ]
        
        # Add any custom feeds from env
        custom_feeds = os.getenv("NEWS_MANUAL_FEEDS", "")
        if custom_feeds:
            self.rss_feeds.extend([f.strip() for f in custom_feeds.split(",") if f.strip()])
        
        self._ensure_table()
    
    def _ensure_table(self):
        """Create news_analysis table if not exists"""
        if not self.db_url or not POSTGRES_AVAILABLE:
            return
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
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
            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to create news_analysis table: {e}")
    
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
        """Fetch latest news from RSS feeds (Reuters, etc.)"""
        if not FEEDPARSER_AVAILABLE:
            return []
        
        headlines = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:
                    published = ""
                    if hasattr(entry, 'published'):
                        published = entry.published
                    elif hasattr(entry, 'updated'):
                        published = entry.updated
                    
                    headlines.append({
                        "title": entry.get("title", ""),
                        "source": feed.feed.get("title", feed_url),
                        "url": entry.get("link", ""),
                        "published": published,
                        "summary": entry.get("summary", "")[:200],
                        "type": "general"
                    })
            except Exception as e:
                LOGGER.error(f"RSS fetch error for {feed_url}: {e}")
        
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
