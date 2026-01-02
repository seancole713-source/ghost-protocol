"""
Ghost Protocol - Social Velocity Tracker
Detect when social media activity is accelerating BEFORE mainstream news.

INSIGHT: Crowd knows before CNBC reports.
         Reddit/Twitter velocity often precedes price moves.
         Influencer posts create immediate volatility.

Data Sources:
- LunarCrush (crypto social)
- Santiment social data
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY", "")
LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")


class SocialVelocityTracker:
    """
    Track social media velocity and sentiment acceleration.
    
    KEY PATTERNS:
    - Mention velocity spike = Something happening
    - Sentiment shift = Crowd opinion changing  
    - Influencer post = Immediate volatility
    - Reddit surge = Retail interest
    """
    
    def __init__(self):
        self.cache = {}
        self.baseline = {}  # Store baseline metrics per symbol
    
    async def get_social_metrics(self, symbol: str) -> Dict:
        """Get current social metrics from multiple sources"""
        metrics = {
            "santiment": {},
            "lunarcrush": {}
        }
        
        # Santiment social data
        if SANTIMENT_API_KEY:
            metrics["santiment"] = await self._santiment_social(symbol)
        
        # LunarCrush for crypto
        if LUNARCRUSH_API_KEY:
            metrics["lunarcrush"] = await self._lunarcrush_social(symbol)
        
        return metrics
    
    async def _santiment_social(self, symbol: str) -> Dict:
        """Fetch social data from Santiment"""
        url = "https://api.santiment.net/graphql"
        
        slug = self._symbol_to_slug(symbol)
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        day_ago = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query = """
        {
            socialVolume: getMetric(metric: "social_volume_total") {
                current: timeseriesData(slug: "%s", from: "%s", to: "%s", interval: "1h") {
                    datetime
                    value
                }
                baseline: timeseriesData(slug: "%s", from: "%s", to: "%s", interval: "1d") {
                    datetime
                    value
                }
            }
            sentiment: getMetric(metric: "sentiment_balance") {
                timeseriesData(slug: "%s", from: "%s", to: "%s", interval: "1h") {
                    datetime
                    value
                }
            }
        }
        """ % (slug, day_ago, now, slug, week_ago, now, slug, day_ago, now)
        
        headers = {"Authorization": f"Apikey {SANTIMENT_API_KEY}"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"query": query}, headers=headers, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        social_vol = data.get("data", {}).get("socialVolume", {})
                        sentiment = data.get("data", {}).get("sentiment", {})
                        
                        # Calculate current vs baseline
                        current_data = social_vol.get("current", {}).get("timeseriesData", [])
                        baseline_data = social_vol.get("baseline", {}).get("timeseriesData", [])
                        
                        current_vol = current_data[-1].get("value", 0) if current_data else 0
                        baseline_vol = sum(d.get("value", 0) for d in baseline_data) / len(baseline_data) if baseline_data else 1
                        
                        velocity_ratio = current_vol / baseline_vol if baseline_vol > 0 else 1
                        
                        # Get sentiment
                        sentiment_data = sentiment.get("timeseriesData", [])
                        current_sentiment = sentiment_data[-1].get("value", 0) if sentiment_data else 0
                        
                        return {
                            "current_volume": current_vol,
                            "baseline_volume": baseline_vol,
                            "velocity_ratio": round(velocity_ratio, 2),
                            "current_sentiment": current_sentiment,
                            "sentiment_label": "POSITIVE" if current_sentiment > 0.2 else "NEGATIVE" if current_sentiment < -0.2 else "NEUTRAL"
                        }
        except Exception as e:
            logger.error(f"Santiment social error: {e}")
        
        return {}
    
    async def _lunarcrush_social(self, symbol: str) -> Dict:
        """Fetch social data from LunarCrush (crypto)"""
        url = f"https://api.lunarcrush.com/v2"
        params = {
            "data": "assets",
            "symbol": symbol,
            "key": LUNARCRUSH_API_KEY
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        asset = data.get("data", [{}])[0]
                        
                        return {
                            "galaxy_score": asset.get("galaxy_score", 0),
                            "social_score": asset.get("social_score", 0),
                            "social_volume_24h": asset.get("social_volume_24h", 0),
                            "social_volume_change": asset.get("social_volume_24h_percent_change", 0),
                            "tweet_sentiment": asset.get("tweet_sentiment", 0),
                            "news_articles_24h": asset.get("news_24h", 0),
                            "influencer_posts_24h": asset.get("influencer_posts_24h", 0)
                        }
        except Exception as e:
            logger.error(f"LunarCrush error: {e}")
        
        return {}
    
    def _symbol_to_slug(self, symbol: str) -> str:
        """Convert symbol to Santiment slug"""
        slugs = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple",
            "ADA": "cardano",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "LINK": "chainlink",
            "MATIC": "matic-network",
            "AVAX": "avalanche"
        }
        return slugs.get(symbol.upper(), symbol.lower())
    
    async def analyze_social_velocity(self, symbol: str) -> Dict:
        """
        Analyze social velocity for early signals.
        
        KEY SIGNALS:
        - Velocity ratio > 3x = Something brewing
        - Velocity ratio > 5x = Major event happening
        - Sentiment flip = Crowd changing mind
        - Influencer post = Immediate volatility expected
        """
        metrics = await self.get_social_metrics(symbol)
        
        santiment = metrics.get("santiment", {})
        lunarcrush = metrics.get("lunarcrush", {})
        
        if not santiment and not lunarcrush:
            return {
                "symbol": symbol,
                "has_data": False,
                "signal": "NEUTRAL",
                "signal_strength": 0,
                "confidence_adjustment": 0,
                "message": "No social data available (may need API keys)"
            }
        
        signal = "NEUTRAL"
        signal_strength = 0
        confidence_adjustment = 0
        warnings = []
        positives = []
        
        # Analyze Santiment data
        if santiment:
            velocity_ratio = santiment.get("velocity_ratio", 1)
            sentiment = santiment.get("current_sentiment", 0)
            sentiment_label = santiment.get("sentiment_label", "NEUTRAL")
            
            # VELOCITY SIGNALS
            if velocity_ratio >= 5:
                signal_strength += 40
                warnings.append(f"🚨 SOCIAL EXPLOSION: {velocity_ratio}x normal volume!")
            elif velocity_ratio >= 3:
                signal_strength += 25
                warnings.append(f"⚠️ HIGH SOCIAL VELOCITY: {velocity_ratio}x normal volume")
            elif velocity_ratio >= 2:
                signal_strength += 10
                warnings.append(f"📈 ELEVATED SOCIAL ACTIVITY: {velocity_ratio}x normal volume")
            
            # SENTIMENT SIGNALS
            if sentiment > 0.5:
                signal = "BULLISH"
                confidence_adjustment += 10
                positives.append(f"✅ STRONG POSITIVE SENTIMENT: {sentiment:.2f}")
            elif sentiment < -0.5:
                signal = "BEARISH"
                confidence_adjustment -= 10
                warnings.append(f"⚠️ STRONG NEGATIVE SENTIMENT: {sentiment:.2f}")
        
        # Analyze LunarCrush data
        if lunarcrush:
            social_change = lunarcrush.get("social_volume_change", 0)
            influencer_posts = lunarcrush.get("influencer_posts_24h", 0)
            galaxy_score = lunarcrush.get("galaxy_score", 50)
            
            if social_change > 100:
                signal_strength += 20
                warnings.append(f"⚠️ SOCIAL VOLUME SURGE: +{social_change}% in 24h")
            
            if influencer_posts >= 3:
                signal_strength += 15
                warnings.append(f"👁️ INFLUENCER ATTENTION: {influencer_posts} posts in 24h")
            
            if galaxy_score >= 80:
                positives.append(f"✅ HIGH GALAXY SCORE: {galaxy_score}/100 (bullish signal)")
                confidence_adjustment += 5
            elif galaxy_score <= 30:
                warnings.append(f"⚠️ LOW GALAXY SCORE: {galaxy_score}/100 (bearish signal)")
                confidence_adjustment -= 5
        
        # High velocity without clear direction = UNCERTAINTY
        if signal_strength > 30 and abs(confidence_adjustment) < 5:
            warnings.append("⚠️ HIGH VOLATILITY EXPECTED: Strong social activity but unclear direction")
        
        return {
            "symbol": symbol,
            "has_data": True,
            "signal": signal,
            "signal_strength": min(signal_strength, 100),
            "confidence_adjustment": max(-20, min(20, confidence_adjustment)),
            
            # Raw metrics
            "metrics": {
                "santiment": santiment,
                "lunarcrush": lunarcrush
            },
            
            # Signals
            "warnings": warnings,
            "positives": positives,
            
            # Interpretation
            "interpretation": self._generate_interpretation(signal, signal_strength, warnings, positives)
        }
    
    def _generate_interpretation(self, signal: str, strength: int,
                                  warnings: List, positives: List) -> str:
        """Generate human-readable interpretation"""
        parts = []
        
        if strength >= 40:
            parts.append("🚨 SOCIAL ALERT: UNUSUAL ACTIVITY DETECTED")
            parts.append("Social media is buzzing about this asset.")
            parts.append("This often precedes significant price movement.")
        elif strength >= 20:
            parts.append("📊 SOCIAL ACTIVITY: ELEVATED")
            parts.append("Above-normal social interest detected.")
        else:
            parts.append("📊 SOCIAL ACTIVITY: NORMAL")
        
        parts.append("")
        if warnings:
            parts.extend(warnings)
        if positives:
            parts.extend(positives)
        
        return "\n".join(parts)


# Singleton
_tracker = None

def get_social_tracker() -> SocialVelocityTracker:
    global _tracker
    if _tracker is None:
        _tracker = SocialVelocityTracker()
    return _tracker


async def analyze_social_velocity(symbol: str) -> Dict:
    """Quick access to social velocity analysis"""
    return await get_social_tracker().analyze_social_velocity(symbol)
