"""
GHOST RESEARCHER - AI-Powered Deep Research Agent

Ghost uses Claude/GPT to research symbols like a human would:
- "What is this company/project?"
- "What's the latest news?"
- "What are the catalysts?"
- "What's the sentiment?"
- "What moves this stock/crypto?"

Think of it like giving Ghost a research analyst who can
do a deep Google search and summarize everything relevant.
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class ResearchReport:
    """Complete research report on a symbol"""
    symbol: str
    asset_type: str  # stock or crypto
    timestamp: str
    
    # Background
    company_name: str
    description: str
    sector: str
    market_cap: str
    
    # Recent news
    recent_news: List[Dict]
    news_sentiment: str  # bullish, bearish, neutral
    
    # Key catalysts
    upcoming_catalysts: List[str]
    recent_catalysts: List[str]
    
    # What moves this asset
    price_drivers: List[str]
    correlated_assets: List[str]
    key_risks: List[str]
    
    # AI analysis
    ai_summary: str
    ai_prediction_context: str
    confidence_modifier: float  # 0.8 = reduce 20%, 1.2 = boost 20%
    
    # Sources used
    sources: List[str]


class GhostResearcher:
    """
    Ghost's AI-powered research agent.
    
    Uses Claude or GPT to:
    1. Understand what a symbol is
    2. Find recent news and catalysts
    3. Analyze what moves the price
    4. Provide context for better predictions
    """
    
    def __init__(self):
        # Use same key pattern as wolf_app.py
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_key = (os.environ.get("OPENAI_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY", "")).strip()
        self.polygon_key = os.environ.get("POLYGON_API_KEY")
        self.alphavantage_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        
        # Get AI model from env (same as wolf_app)
        self.ai_model = os.environ.get("AGENT_MODEL", os.environ.get("AI_MODEL", "gpt-4o-mini")).strip()
        self.claude_model = os.environ.get("CLAUDE_MODEL", "claude-3-haiku-20240307").strip()
        
        # Choose AI provider - prefer Claude (faster, smarter)
        if self.anthropic_key:
            self.ai_provider = "anthropic"
            LOGGER.info(f"[RESEARCHER] Using Claude ({self.claude_model}) for research")
        elif self.openai_key:
            self.ai_provider = "openai"
            LOGGER.info(f"[RESEARCHER] Using OpenAI ({self.ai_model}) as fallback")
        else:
            self.ai_provider = None
            LOGGER.warning("[RESEARCHER] No AI API key - research limited")
        
        # Cache research to avoid repeated API calls
        self.research_cache: Dict[str, ResearchReport] = {}
        self.cache_ttl_hours = 24
    
    async def research_symbol(self, symbol: str, force_refresh: bool = False) -> ResearchReport:
        """
        Do complete research on a symbol - like a Google deep dive.
        
        Returns everything Ghost needs to understand what moves this asset.
        """
        symbol = symbol.upper()
        
        # Check cache
        if not force_refresh and symbol in self.research_cache:
            cached = self.research_cache[symbol]
            cache_time = datetime.fromisoformat(cached.timestamp)
            if datetime.now() - cache_time < timedelta(hours=self.cache_ttl_hours):
                LOGGER.info(f"[RESEARCHER] Using cached research for {symbol}")
                return cached
        
        LOGGER.info(f"[RESEARCHER] 🔍 Researching {symbol}...")
        
        # Determine if stock or crypto
        asset_type = self._determine_asset_type(symbol)
        
        # Gather data from multiple sources
        async with aiohttp.ClientSession() as session:
            # Run all data gathering in parallel
            tasks = [
                self._get_basic_info(session, symbol, asset_type),
                self._get_recent_news(session, symbol, asset_type),
                self._get_price_history(session, symbol, asset_type),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            basic_info = results[0] if not isinstance(results[0], Exception) else {}
            news = results[1] if not isinstance(results[1], Exception) else []
            price_history = results[2] if not isinstance(results[2], Exception) else {}
        
        # Use AI to analyze and synthesize
        ai_analysis = await self._ai_deep_analysis(symbol, asset_type, basic_info, news, price_history)
        
        # If AI failed, use basic analysis
        if not ai_analysis or not ai_analysis.get("summary"):
            LOGGER.info(f"[RESEARCHER] Using basic analysis for {symbol} (AI unavailable)")
            ai_analysis = self._basic_analysis(symbol, asset_type, news, price_history)
        
        # Build the research report
        report = ResearchReport(
            symbol=symbol,
            asset_type=asset_type,
            timestamp=datetime.now().isoformat(),
            company_name=basic_info.get("name", symbol),
            description=basic_info.get("description", ""),
            sector=basic_info.get("sector", "Unknown"),
            market_cap=basic_info.get("market_cap", "Unknown"),
            recent_news=news[:10],
            news_sentiment=ai_analysis.get("news_sentiment", "neutral"),
            upcoming_catalysts=ai_analysis.get("upcoming_catalysts", []),
            recent_catalysts=ai_analysis.get("recent_catalysts", []),
            price_drivers=ai_analysis.get("price_drivers", []),
            correlated_assets=ai_analysis.get("correlated_assets", []),
            key_risks=ai_analysis.get("key_risks", []),
            ai_summary=ai_analysis.get("summary", ""),
            ai_prediction_context=ai_analysis.get("prediction_context", ""),
            confidence_modifier=ai_analysis.get("confidence_modifier", 1.0),
            sources=ai_analysis.get("sources", [])
        )
        
        # Cache the report
        self.research_cache[symbol] = report
        
        LOGGER.info(f"[RESEARCHER] ✅ Research complete for {symbol}")
        return report
    
    def _determine_asset_type(self, symbol: str) -> str:
        """Determine if symbol is stock or crypto"""
        crypto_symbols = {
            "BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOT", "MATIC", "LINK", "UNI",
            "DOGE", "SHIB", "PEPE", "ATOM", "LTC", "BCH", "XLM", "ALGO", "VET", "FIL",
            "NEAR", "APT", "ARB", "OP", "INJ", "SUI", "SEI", "TIA", "BONK", "WIF"
        }
        
        if symbol in crypto_symbols or symbol.endswith("USD") or symbol.endswith("USDT"):
            return "crypto"
        return "stock"
    
    async def _get_basic_info(self, session: aiohttp.ClientSession, symbol: str, asset_type: str) -> Dict:
        """Get basic information about the symbol"""
        info = {"symbol": symbol}
        
        if asset_type == "stock" and self.polygon_key:
            try:
                url = f"https://api.polygon.io/v3/reference/tickers/{symbol}?apiKey={self.polygon_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", {})
                        info = {
                            "name": results.get("name", symbol),
                            "description": results.get("description", ""),
                            "sector": results.get("sic_description", "Unknown"),
                            "market_cap": f"${results.get('market_cap', 0):,.0f}" if results.get('market_cap') else "Unknown",
                            "homepage": results.get("homepage_url", ""),
                            "employees": results.get("total_employees", 0),
                        }
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] Polygon info error: {e}")
        
        elif asset_type == "crypto":
            try:
                # Use CoinGecko for crypto info
                symbol_map = {
                    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
                    "XRP": "ripple", "ADA": "cardano", "AVAX": "avalanche-2",
                    "DOT": "polkadot", "MATIC": "matic-network", "LINK": "chainlink",
                    "DOGE": "dogecoin", "SHIB": "shiba-inu", "PEPE": "pepe"
                }
                coin_id = symbol_map.get(symbol, symbol.lower())
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        info = {
                            "name": data.get("name", symbol),
                            "description": data.get("description", {}).get("en", "")[:500],
                            "sector": "Cryptocurrency",
                            "market_cap": f"${data.get('market_data', {}).get('market_cap', {}).get('usd', 0):,.0f}",
                            "homepage": data.get("links", {}).get("homepage", [""])[0],
                            "categories": data.get("categories", []),
                        }
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] CoinGecko info error: {e}")
        
        return info
    
    async def _get_recent_news(self, session: aiohttp.ClientSession, symbol: str, asset_type: str) -> List[Dict]:
        """Get recent news about the symbol from multiple sources"""
        news = []
        
        # Try Polygon news for stocks (has sentiment built in)
        if asset_type == "stock" and self.polygon_key:
            try:
                url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=10&apiKey={self.polygon_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for article in data.get("results", []):
                            news.append({
                                "title": article.get("title", ""),
                                "source": article.get("publisher", {}).get("name", ""),
                                "date": article.get("published_utc", ""),
                                "url": article.get("article_url", ""),
                                "sentiment": article.get("insights", [{}])[0].get("sentiment", "neutral") if article.get("insights") else "neutral"
                            })
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] Polygon news error: {e}")
        
        # Also check Alpha Vantage news (you have this key!)
        if asset_type == "stock" and self.alphavantage_key and len(news) < 5:
            try:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alphavantage_key}&limit=10"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for article in data.get("feed", [])[:10]:
                            # Alpha Vantage provides sentiment scores!
                            sentiment_score = float(article.get("overall_sentiment_score", 0))
                            sentiment = "bullish" if sentiment_score > 0.15 else "bearish" if sentiment_score < -0.15 else "neutral"
                            news.append({
                                "title": article.get("title", ""),
                                "source": article.get("source", ""),
                                "date": article.get("time_published", ""),
                                "url": article.get("url", ""),
                                "sentiment": sentiment,
                                "sentiment_score": sentiment_score
                            })
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] Alpha Vantage news error: {e}")
        
        # Try CryptoPanic for crypto
        cryptopanic_key = os.environ.get("CRYPTOPANIC_API_KEY")
        if asset_type == "crypto" and cryptopanic_key:
            try:
                url = f"https://cryptopanic.com/api/v1/posts/?auth_token={cryptopanic_key}&currencies={symbol}&limit=10&public=true"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for post in data.get("results", []):
                            news.append({
                                "title": post.get("title", ""),
                                "source": post.get("source", {}).get("title", ""),
                                "date": post.get("published_at", ""),
                                "url": post.get("url", ""),
                                "sentiment": "bullish" if post.get("votes", {}).get("positive", 0) > post.get("votes", {}).get("negative", 0) else "bearish"
                            })
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] CryptoPanic news error: {e}")
        
        # Add Santiment social data for crypto (you have this key!)
        santiment_key = os.environ.get("SANTIMENT_API_KEY")
        if asset_type == "crypto" and santiment_key:
            try:
                # Santiment GraphQL query for social volume
                query = '''
                {
                    getMetric(metric: "social_volume_total") {
                        timeseriesData(
                            slug: "%s"
                            from: "utc_now-7d"
                            to: "utc_now"
                            interval: "1d"
                        ) {
                            datetime
                            value
                        }
                    }
                }
                ''' % symbol.lower()
                
                headers = {"Authorization": f"Apikey {santiment_key}"}
                async with session.post(
                    "https://api.santiment.net/graphql",
                    json={"query": query},
                    headers=headers,
                    timeout=10
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        social_data = data.get("data", {}).get("getMetric", {}).get("timeseriesData", [])
                        if social_data:
                            recent_volume = social_data[-1].get("value", 0) if social_data else 0
                            avg_volume = sum(d.get("value", 0) for d in social_data) / len(social_data) if social_data else 0
                            if recent_volume > avg_volume * 1.5:
                                news.append({
                                    "title": f"🔥 {symbol} social volume spike: {recent_volume:.0f} (avg: {avg_volume:.0f})",
                                    "source": "Santiment",
                                    "date": datetime.now().isoformat(),
                                    "url": "",
                                    "sentiment": "bullish"  # High social = attention = potential pump
                                })
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] Santiment error: {e}")
        
        return news
    
    async def _get_price_history(self, session: aiohttp.ClientSession, symbol: str, asset_type: str) -> Dict:
        """Get recent price history for context"""
        history = {"symbol": symbol, "prices": []}
        
        if asset_type == "stock" and self.polygon_key:
            try:
                # Get last 30 days
                end = datetime.now()
                start = end - timedelta(days=30)
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}?apiKey={self.polygon_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        if results:
                            history["current_price"] = results[-1].get("c", 0)
                            history["price_30d_ago"] = results[0].get("c", 0) if results else 0
                            history["change_30d"] = ((history["current_price"] / history["price_30d_ago"]) - 1) * 100 if history["price_30d_ago"] else 0
                            history["high_30d"] = max(r.get("h", 0) for r in results)
                            history["low_30d"] = min(r.get("l", 0) for r in results)
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] Polygon price error: {e}")
        
        elif asset_type == "crypto":
            try:
                symbol_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin"}
                coin_id = symbol_map.get(symbol, symbol.lower())
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        prices = data.get("prices", [])
                        if prices:
                            history["current_price"] = prices[-1][1]
                            history["price_30d_ago"] = prices[0][1]
                            history["change_30d"] = ((history["current_price"] / history["price_30d_ago"]) - 1) * 100 if history["price_30d_ago"] else 0
            except Exception as e:
                LOGGER.warning(f"[RESEARCHER] CoinGecko price error: {e}")
        
        return history
    
    async def _ai_deep_analysis(self, symbol: str, asset_type: str, 
                                basic_info: Dict, news: List[Dict], 
                                price_history: Dict) -> Dict:
        """
        Use Claude or GPT to do deep analysis - like a research analyst.
        
        This is where Ghost gets SMART - AI synthesizes all the data
        and provides insights a human analyst would give.
        """
        
        # Build context for AI
        news_text = "\n".join([f"- {n.get('title', '')} ({n.get('sentiment', 'neutral')})" for n in news[:10]])
        
        prompt = f"""You are a professional investment research analyst. Analyze this {asset_type} and provide insights:

SYMBOL: {symbol}
NAME: {basic_info.get('name', symbol)}
SECTOR: {basic_info.get('sector', 'Unknown')}
DESCRIPTION: {basic_info.get('description', 'No description available')[:500]}

PRICE DATA (30 days):
- Current: ${price_history.get('current_price', 'N/A')}
- 30d change: {price_history.get('change_30d', 0):.1f}%
- 30d high: ${price_history.get('high_30d', 'N/A')}
- 30d low: ${price_history.get('low_30d', 'N/A')}

RECENT NEWS:
{news_text if news_text else 'No recent news available'}

Please analyze and provide:

1. NEWS_SENTIMENT: Is the recent news bullish, bearish, or neutral?

2. UPCOMING_CATALYSTS: What events could move this {asset_type} in the next 30 days? (earnings, product launches, regulatory decisions, etc.)

3. RECENT_CATALYSTS: What recent events have affected the price?

4. PRICE_DRIVERS: What are the main factors that move this {asset_type}'s price? (macro factors, sector trends, company-specific, etc.)

5. CORRELATED_ASSETS: What other assets move with this one? (sector ETFs, related stocks/cryptos, indices)

6. KEY_RISKS: What are the main risks to watch?

7. SUMMARY: A 2-3 sentence summary of the current situation.

8. PREDICTION_CONTEXT: What should a prediction model know about this {asset_type} right now?

9. CONFIDENCE_MODIFIER: Based on the news and catalysts, should predictions be:
   - More confident (1.1-1.3): Clear trend, positive catalysts
   - Normal (1.0): Mixed signals
   - Less confident (0.7-0.9): High uncertainty, negative catalysts

Respond in JSON format:
{{
    "news_sentiment": "bullish|bearish|neutral",
    "upcoming_catalysts": ["catalyst1", "catalyst2"],
    "recent_catalysts": ["catalyst1", "catalyst2"],
    "price_drivers": ["driver1", "driver2"],
    "correlated_assets": ["ASSET1", "ASSET2"],
    "key_risks": ["risk1", "risk2"],
    "summary": "...",
    "prediction_context": "...",
    "confidence_modifier": 1.0,
    "sources": ["polygon", "coingecko", "news"]
}}"""

        # Call AI
        if self.ai_provider == "anthropic":
            return await self._call_claude(prompt)
        elif self.ai_provider == "openai":
            return await self._call_openai(prompt)
        else:
            # Fallback - basic analysis without AI
            return self._basic_analysis(symbol, asset_type, news, price_history)
    
    async def _call_claude(self, prompt: str) -> Dict:
        """Call Claude API for analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "x-api-key": self.anthropic_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": self.claude_model,  # Use claude-3-haiku by default
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=60
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("content", [{}])[0].get("text", "{}")
                        # Extract JSON from response
                        try:
                            # Find JSON in response
                            start = content.find("{")
                            end = content.rfind("}") + 1
                            if start >= 0 and end > start:
                                return json.loads(content[start:end])
                        except json.JSONDecodeError:
                            LOGGER.warning("[RESEARCHER] Could not parse Claude response as JSON")
                    else:
                        error = await resp.text()
                        LOGGER.error(f"[RESEARCHER] Claude API error: {resp.status} - {error}")
        except Exception as e:
            LOGGER.error(f"[RESEARCHER] Claude call failed: {e}")
        
        return {}
    
    async def _call_openai(self, prompt: str) -> Dict:
        """Call OpenAI API for analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.ai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}
                }
                
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                        return json.loads(content)
                    else:
                        error = await resp.text()
                        LOGGER.error(f"[RESEARCHER] OpenAI API error: {resp.status} - {error}")
        except Exception as e:
            LOGGER.error(f"[RESEARCHER] OpenAI call failed: {e}")
        
        return {}
    
    def _basic_analysis(self, symbol: str, asset_type: str, news: List[Dict], price_history: Dict) -> Dict:
        """Fallback analysis without AI - still provides useful insights"""
        # Count sentiment from news
        bullish = sum(1 for n in news if n.get("sentiment") == "bullish")
        bearish = sum(1 for n in news if n.get("sentiment") == "bearish")
        
        sentiment = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
        
        # Basic price momentum
        change = price_history.get("change_30d", 0)
        momentum = "uptrend" if change > 5 else "downtrend" if change < -5 else "sideways"
        
        # Generate catalysts from news
        recent_catalysts = [n.get("title", "")[:80] for n in news[:3] if n.get("title")]
        
        # Standard catalysts by asset type
        if asset_type == "stock":
            upcoming = ["Earnings report (check IR calendar)", "Fed rate decision", "Sector rotation"]
            drivers = [f"30-day momentum: {momentum} ({change:.1f}%)", f"News sentiment: {sentiment}", "Sector performance", "Market conditions (SPY)"]
            correlated = ["SPY", "QQQ", "XLK"] if "tech" in str(price_history.get("sector", "")).lower() else ["SPY", "QQQ"]
            risks = ["Earnings miss risk", "Market correction", "Sector rotation"]
        else:
            upcoming = ["BTC price action", "Fed rate decision", "Major exchange listings"]
            drivers = [f"30-day momentum: {momentum} ({change:.1f}%)", f"News sentiment: {sentiment}", "BTC correlation", "Market sentiment"]
            correlated = ["BTC", "ETH"]
            risks = ["BTC dump risk", "Regulatory news", "Exchange issues"]
        
        # Calculate confidence modifier based on data
        confidence = 1.0
        if change > 10 and sentiment == "bullish":
            confidence = 1.1  # Strong uptrend with bullish news
        elif change < -10 and sentiment == "bearish":
            confidence = 0.9  # Downtrend with bearish news - reduce confidence
        elif abs(change) < 3:
            confidence = 0.95  # Sideways = uncertainty
        
        return {
            "news_sentiment": sentiment,
            "upcoming_catalysts": upcoming,
            "recent_catalysts": recent_catalysts if recent_catalysts else ["No major catalysts detected"],
            "price_drivers": drivers,
            "correlated_assets": correlated,
            "key_risks": risks,
            "summary": f"{symbol} is in a {momentum} with {sentiment} news sentiment. 30-day change: {change:.1f}%.",
            "prediction_context": f"Momentum: {momentum}. News: {sentiment}. Recent headlines suggest {'positive' if sentiment == 'bullish' else 'negative' if sentiment == 'bearish' else 'mixed'} outlook.",
            "confidence_modifier": confidence,
            "sources": ["polygon", "coingecko", "news_analysis"]
        }
    
    async def quick_research(self, symbol: str) -> str:
        """
        Quick research summary - returns a human-readable string.
        Good for quick context before making a prediction.
        """
        report = await self.research_symbol(symbol)
        
        summary = f"""
📊 RESEARCH: {report.symbol} ({report.asset_type.upper()})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏢 {report.company_name}
   Sector: {report.sector}
   Market Cap: {report.market_cap}

📰 NEWS SENTIMENT: {report.news_sentiment.upper()}
   Recent headlines:
"""
        for news in report.recent_news[:3]:
            summary += f"   • {news.get('title', '')[:60]}...\n"
        
        summary += f"""
🎯 UPCOMING CATALYSTS:
"""
        for catalyst in report.upcoming_catalysts[:3]:
            summary += f"   • {catalyst}\n"
        
        summary += f"""
📈 PRICE DRIVERS:
"""
        for driver in report.price_drivers[:3]:
            summary += f"   • {driver}\n"
        
        summary += f"""
⚠️ KEY RISKS:
"""
        for risk in report.key_risks[:3]:
            summary += f"   • {risk}\n"
        
        summary += f"""
🤖 AI ANALYSIS:
   {report.ai_summary}

📋 PREDICTION CONTEXT:
   {report.ai_prediction_context}

🎚️ CONFIDENCE MODIFIER: {report.confidence_modifier:.1f}x
   {"↑ BOOST predictions" if report.confidence_modifier > 1 else "↓ REDUCE confidence" if report.confidence_modifier < 1 else "→ Normal confidence"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return summary


# =============================================================================
# HOW GHOST USES THIS
# =============================================================================

"""
GHOST + CLAUDE = SMART PREDICTIONS

Before making a prediction, Ghost can now:

1. RESEARCH THE SYMBOL:
   researcher = GhostResearcher()
   report = await researcher.research_symbol("NVDA")
   
2. GET CONTEXT:
   - What is this company?
   - What's the recent news?
   - What are the catalysts?
   - What moves the price?
   
3. ADJUST PREDICTIONS:
   if report.confidence_modifier < 1.0:
       # High uncertainty - reduce confidence
       prediction.confidence *= report.confidence_modifier
   
4. PROVIDE CONTEXT TO USER:
   "NVDA prediction is BULLISH but confidence reduced due to:
    - Upcoming earnings (catalyst uncertainty)
    - Recent AI chip export restrictions (regulatory risk)"

EXAMPLE FLOW:
━━━━━━━━━━━━━
User asks: "Should I buy NVDA?"

Ghost:
1. Checks Event Memory: Any recent events affecting NVDA?
2. Runs Research: What's the full context?
3. Gets AI Analysis: What does Claude think?
4. Makes Prediction: Combines technical + fundamental + AI
5. Returns: "BULLISH 75% confidence - but watch for earnings on Feb 15"

THIS IS HOW GHOST BECOMES TRULY INTELLIGENT.
"""


# Quick test
if __name__ == "__main__":
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_research():
        researcher = GhostResearcher()
        
        # Test with a stock
        print("\n" + "="*60)
        print("TESTING GHOST RESEARCHER")
        print("="*60)
        
        summary = await researcher.quick_research("NVDA")
        print(summary)
    
    asyncio.run(test_research())
