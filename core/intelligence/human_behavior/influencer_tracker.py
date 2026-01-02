"""
Ghost Protocol - Influencer Tracker
Track what key market influencers are saying.

INSIGHT: When Elon tweets, markets move.
         When Saylor buys, BTC rallies.
         When Buffett speaks, everyone listens.

Track the people who move markets.
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Key influencers to track
INFLUENCERS = {
    # Crypto influencers
    "crypto": [
        {"name": "Elon Musk", "handle": "elonmusk", "impact": "extreme", "assets": ["BTC", "DOGE", "TSLA"]},
        {"name": "Michael Saylor", "handle": "saylor", "impact": "high", "assets": ["BTC"]},
        {"name": "CZ Binance", "handle": "caborz", "impact": "high", "assets": ["BNB", "crypto"]},
        {"name": "Vitalik Buterin", "handle": "VitalikButerin", "impact": "high", "assets": ["ETH"]},
        {"name": "Cathie Wood", "handle": "CathieDWood", "impact": "moderate", "assets": ["BTC", "ARKK"]},
        {"name": "Brian Armstrong", "handle": "brian_armstrong", "impact": "moderate", "assets": ["COIN", "crypto"]},
        {"name": "Donald Trump", "handle": "realDonaldTrump", "impact": "extreme", "assets": ["BTC", "crypto", "stocks"]},
    ],
    
    # Stock market influencers  
    "stocks": [
        {"name": "Warren Buffett", "handle": None, "impact": "extreme", "assets": ["BRK", "stocks"]},
        {"name": "Elon Musk", "handle": "elonmusk", "impact": "extreme", "assets": ["TSLA"]},
        {"name": "Jim Cramer", "handle": "jimcramer", "impact": "moderate", "inverse": True, "assets": ["stocks"]},  # Famous for being wrong
        {"name": "Nancy Pelosi", "handle": None, "impact": "high", "assets": ["stocks"]},  # Insider trading memes
        {"name": "Cathie Wood", "handle": "CathieDWood", "impact": "moderate", "assets": ["growth stocks"]},
    ],
    
    # Macro/Fed influencers
    "macro": [
        {"name": "Jerome Powell", "handle": None, "impact": "extreme", "assets": ["all"]},
        {"name": "Janet Yellen", "handle": None, "impact": "high", "assets": ["all"]},
    ]
}


class InfluencerTracker:
    """
    Track influencer statements and their market impact.
    
    KEY PATTERNS:
    - Elon + crypto mention = instant volatility
    - Saylor buying = BTC bullish
    - Fed chair + hawkish = risk-off
    - Trump + crypto = market pump
    """
    
    def __init__(self):
        self.cache = {}
        self.recent_events = []
    
    async def check_recent_influencer_activity(self, symbol: str) -> Dict:
        """
        Check if any major influencers have mentioned this asset recently.
        
        Returns analysis of influencer impact.
        """
        relevant_influencers = self._get_relevant_influencers(symbol)
        
        # Check news for influencer mentions
        mentions = await self._search_influencer_mentions(symbol, relevant_influencers)
        
        if not mentions:
            return {
                "symbol": symbol,
                "has_activity": False,
                "signal": "NEUTRAL",
                "confidence_adjustment": 0,
                "message": "No recent influencer activity detected",
                "warnings": [],
                "positives": []
            }
        
        # Analyze impact
        signal = "NEUTRAL"
        confidence_adjustment = 0
        warnings = []
        positives = []
        
        for mention in mentions:
            influencer = mention.get("influencer", {})
            impact = influencer.get("impact", "low")
            sentiment = mention.get("sentiment", "neutral")
            
            if impact == "extreme":
                if sentiment == "bullish":
                    confidence_adjustment += 20
                    positives.append(f"🚀 {influencer['name']} BULLISH on {symbol}")
                elif sentiment == "bearish":
                    confidence_adjustment -= 20
                    warnings.append(f"⚠️ {influencer['name']} BEARISH on {symbol}")
            elif impact == "high":
                if sentiment == "bullish":
                    confidence_adjustment += 10
                    positives.append(f"✅ {influencer['name']} positive on {symbol}")
                elif sentiment == "bearish":
                    confidence_adjustment -= 10
                    warnings.append(f"⚠️ {influencer['name']} negative on {symbol}")
            
            # Handle Jim Cramer inverse
            if influencer.get("inverse"):
                confidence_adjustment = -confidence_adjustment
                if confidence_adjustment > 0:
                    positives.append(f"📉 Cramer SOLD = Actually bullish (inverse Cramer)")
                elif confidence_adjustment < 0:
                    warnings.append(f"📈 Cramer BOUGHT = Actually bearish (inverse Cramer)")
        
        if confidence_adjustment > 5:
            signal = "BULLISH"
        elif confidence_adjustment < -5:
            signal = "BEARISH"
        
        return {
            "symbol": symbol,
            "has_activity": True,
            "signal": signal,
            "confidence_adjustment": max(-30, min(30, confidence_adjustment)),
            "mentions": mentions,
            "warnings": warnings,
            "positives": positives,
            "interpretation": self._generate_interpretation(mentions, signal)
        }
    
    def _get_relevant_influencers(self, symbol: str) -> List[Dict]:
        """Get influencers relevant to this symbol"""
        relevant = []
        
        # Check if crypto or stock
        crypto_symbols = {"BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "DOT", "LINK"}
        is_crypto = symbol.upper() in crypto_symbols
        
        if is_crypto:
            relevant.extend(INFLUENCERS.get("crypto", []))
        else:
            relevant.extend(INFLUENCERS.get("stocks", []))
        
        # Always add macro influencers
        relevant.extend(INFLUENCERS.get("macro", []))
        
        return relevant
    
    async def _search_influencer_mentions(self, symbol: str, influencers: List[Dict]) -> List[Dict]:
        """Search for recent influencer mentions of this asset"""
        mentions = []
        
        # Use news API to find influencer mentions
        polygon_key = os.getenv("POLYGON_API_KEY", "")
        if polygon_key:
            for influencer in influencers:
                # Search for influencer name + symbol in news
                url = "https://api.polygon.io/v2/reference/news"
                params = {
                    "ticker": symbol,
                    "limit": 10,
                    "apiKey": polygon_key
                }
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params, timeout=10) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                for article in data.get("results", []):
                                    title = article.get("title", "").lower()
                                    desc = article.get("description", "").lower()
                                    
                                    # Check if influencer mentioned
                                    name_lower = influencer["name"].lower()
                                    if name_lower in title or name_lower in desc:
                                        sentiment = self._analyze_mention_sentiment(title + " " + desc)
                                        mentions.append({
                                            "influencer": influencer,
                                            "headline": article.get("title"),
                                            "date": article.get("published_utc"),
                                            "sentiment": sentiment
                                        })
                except Exception as e:
                    logger.debug(f"Error searching for {influencer['name']}: {e}")
        
        return mentions
    
    def _analyze_mention_sentiment(self, text: str) -> str:
        """Analyze sentiment of influencer mention"""
        bullish_words = ["buy", "bullish", "moon", "pump", "love", "great", "amazing", "positive", "support"]
        bearish_words = ["sell", "bearish", "dump", "crash", "bad", "negative", "warn", "concern", "risk"]
        
        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        return "neutral"
    
    def _generate_interpretation(self, mentions: List[Dict], signal: str) -> str:
        """Generate interpretation"""
        parts = []
        
        if mentions:
            parts.append("👁️ INFLUENCER ACTIVITY DETECTED")
            parts.append("")
            for m in mentions[:3]:
                inf = m.get("influencer", {})
                parts.append(f"📢 {inf.get('name')}: {m.get('sentiment', 'neutral').upper()}")
                parts.append(f"   \"{m.get('headline', '')[:100]}...\"")
                parts.append("")
        
        if signal == "BULLISH":
            parts.append("✅ Influencer sentiment is BULLISH")
        elif signal == "BEARISH":
            parts.append("⚠️ Influencer sentiment is BEARISH")
        else:
            parts.append("➡️ Influencer sentiment is MIXED/NEUTRAL")
        
        return "\n".join(parts)


# Singleton
_tracker = None

def get_influencer_tracker() -> InfluencerTracker:
    global _tracker
    if _tracker is None:
        _tracker = InfluencerTracker()
    return _tracker


async def check_influencers(symbol: str) -> Dict:
    """Quick access to influencer check"""
    return await get_influencer_tracker().check_recent_influencer_activity(symbol)
