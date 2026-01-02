"""
Ghost Protocol - Narrative Detector
Identify the STORY that's driving the market right now.

INSIGHT: Markets move on narratives, not just numbers.
         "AI Revolution" drove NVDA 10x
         "Inflation hedge" drove BTC adoption
         "Meme stock rebellion" drove GME

When you identify the narrative, you can ride it.
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Pre-defined market narratives
NARRATIVES = {
    # Bullish narratives
    "AI_REVOLUTION": {
        "keywords": ["ai", "artificial intelligence", "chatgpt", "llm", "machine learning", "nvidia", "gpu"],
        "sectors": ["tech", "semiconductors"],
        "sentiment": "BULLISH",
        "strength": "strong"
    },
    "BITCOIN_HALVING": {
        "keywords": ["halving", "btc halving", "bitcoin halving", "supply shock", "scarcity"],
        "assets": ["BTC", "bitcoin"],
        "sentiment": "BULLISH",
        "strength": "strong",
        "cyclical": True
    },
    "INSTITUTIONAL_ADOPTION": {
        "keywords": ["institutional", "etf", "blackrock", "fidelity", "pension", "endowment"],
        "sentiment": "BULLISH",
        "strength": "strong"
    },
    "FED_PIVOT": {
        "keywords": ["fed pivot", "rate cut", "dovish", "soft landing", "pause"],
        "sentiment": "BULLISH",
        "strength": "strong"
    },
    "MONEY_PRINTER": {
        "keywords": ["quantitative easing", "qe", "stimulus", "money printer", "liquidity"],
        "sentiment": "BULLISH",
        "strength": "strong"
    },
    "TRUMP_CRYPTO": {
        "keywords": ["trump", "crypto reserve", "bitcoin reserve", "strategic reserve"],
        "sentiment": "BULLISH",
        "strength": "strong"
    },
    
    # Bearish narratives
    "RECESSION_FEARS": {
        "keywords": ["recession", "economic slowdown", "layoffs", "unemployment", "job cuts"],
        "sentiment": "BEARISH",
        "strength": "strong"
    },
    "RATE_HIKE_FEARS": {
        "keywords": ["rate hike", "hawkish", "inflation", "cpi", "fed tightening"],
        "sentiment": "BEARISH",
        "strength": "strong"
    },
    "CRYPTO_WINTER": {
        "keywords": ["crypto winter", "bear market", "ftx", "collapse", "contagion"],
        "sentiment": "BEARISH",
        "strength": "strong"
    },
    "REGULATION_FEARS": {
        "keywords": ["sec", "regulation", "crackdown", "lawsuit", "enforcement", "ban"],
        "sentiment": "BEARISH",
        "strength": "moderate"
    },
    "BANK_CRISIS": {
        "keywords": ["bank failure", "bank run", "svb", "silicon valley bank", "credit suisse"],
        "sentiment": "BEARISH",
        "strength": "strong"
    },
    
    # Neutral/Rotational narratives
    "SECTOR_ROTATION": {
        "keywords": ["rotation", "value", "growth", "cyclicals", "defensives"],
        "sentiment": "NEUTRAL",
        "strength": "moderate"
    },
    "ELECTION_CYCLE": {
        "keywords": ["election", "presidential", "midterm", "political", "policy"],
        "sentiment": "NEUTRAL",
        "strength": "moderate"
    }
}


class NarrativeDetector:
    """
    Detect which narrative is driving the market.
    
    THE INSIGHT: Once you identify the narrative, you can:
    1. Trade WITH the narrative (ride the wave)
    2. Position for narrative CHANGE (catch reversals)
    3. Avoid fighting the narrative (don't short in AI boom)
    """
    
    def __init__(self):
        self.cache = {}
    
    async def detect_current_narratives(self, symbol: str = None) -> Dict:
        """
        Detect what narratives are active in the market.
        
        Method:
        1. Fetch recent news headlines
        2. Analyze keyword frequency
        3. Match to known narratives
        4. Score narrative strength
        """
        # Get recent news/headlines
        headlines = await self._fetch_recent_headlines(symbol)
        
        if not headlines:
            return {
                "symbol": symbol,
                "has_data": False,
                "active_narratives": [],
                "dominant_narrative": None,
                "confidence_adjustment": 0,
                "message": "No news data available for narrative detection"
            }
        
        # Detect narratives from headlines
        detected = self._match_narratives(headlines)
        
        # Sort by strength
        sorted_narratives = sorted(
            detected.items(),
            key=lambda x: x[1]["match_count"],
            reverse=True
        )
        
        # Get dominant narrative
        dominant = sorted_narratives[0] if sorted_narratives else None
        
        # Calculate market mood
        bullish_narratives = [n for n, d in sorted_narratives if d["sentiment"] == "BULLISH"]
        bearish_narratives = [n for n, d in sorted_narratives if d["sentiment"] == "BEARISH"]
        
        # Confidence adjustment based on narrative alignment
        confidence_adjustment = 0
        if dominant:
            narrative_name, narrative_data = dominant
            if narrative_data["sentiment"] == "BULLISH":
                confidence_adjustment += 10
            elif narrative_data["sentiment"] == "BEARISH":
                confidence_adjustment -= 10
        
        return {
            "symbol": symbol,
            "has_data": True,
            "headline_count": len(headlines),
            
            # Active narratives
            "active_narratives": [
                {
                    "name": name,
                    "sentiment": data["sentiment"],
                    "match_count": data["match_count"],
                    "sample_keywords": data["matched_keywords"][:5]
                }
                for name, data in sorted_narratives[:5]
            ],
            
            # Dominant narrative
            "dominant_narrative": {
                "name": dominant[0] if dominant else None,
                "sentiment": dominant[1]["sentiment"] if dominant else None,
                "description": self._get_narrative_description(dominant[0]) if dominant else None
            },
            
            # Market mood
            "market_mood": {
                "bullish_narrative_count": len(bullish_narratives),
                "bearish_narrative_count": len(bearish_narratives),
                "overall_mood": "BULLISH" if len(bullish_narratives) > len(bearish_narratives) else "BEARISH" if len(bearish_narratives) > len(bullish_narratives) else "MIXED"
            },
            
            # Confidence adjustment
            "confidence_adjustment": confidence_adjustment,
            
            # Interpretation
            "interpretation": self._generate_interpretation(dominant, sorted_narratives)
        }
    
    async def _fetch_recent_headlines(self, symbol: str = None) -> List[str]:
        """Fetch recent news headlines"""
        headlines = []
        
        # Try Polygon for symbol-specific news
        polygon_key = os.getenv("POLYGON_API_KEY", "")
        if polygon_key:
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                "limit": 50,
                "order": "desc",
                "apiKey": polygon_key
            }
            if symbol:
                params["ticker"] = symbol
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for article in data.get("results", []):
                                headlines.append(article.get("title", "").lower())
                                desc = article.get("description", "")
                                if desc:
                                    headlines.append(desc[:200].lower())
            except Exception as e:
                logger.error(f"Polygon news error: {e}")
        
        return headlines
    
    def _match_narratives(self, headlines: List[str]) -> Dict:
        """Match headlines to known narratives"""
        detected = {}
        
        # Combine all headlines into one text
        all_text = " ".join(headlines)
        
        for narrative_name, narrative_data in NARRATIVES.items():
            matched_keywords = []
            
            for keyword in narrative_data.get("keywords", []):
                if keyword.lower() in all_text:
                    count = all_text.count(keyword.lower())
                    matched_keywords.extend([keyword] * count)
            
            if matched_keywords:
                detected[narrative_name] = {
                    "match_count": len(matched_keywords),
                    "matched_keywords": list(set(matched_keywords)),
                    "sentiment": narrative_data.get("sentiment", "NEUTRAL"),
                    "strength": narrative_data.get("strength", "moderate")
                }
        
        return detected
    
    def _get_narrative_description(self, narrative_name: str) -> str:
        """Get human description of narrative"""
        descriptions = {
            "AI_REVOLUTION": "The market believes AI will transform everything. Tech and semiconductors leading.",
            "BITCOIN_HALVING": "Bitcoin supply halving event approaching. Historically very bullish.",
            "INSTITUTIONAL_ADOPTION": "Big institutions are entering crypto/markets. Legitimacy play.",
            "FED_PIVOT": "Market expects Fed to cut rates. Risk-on mode.",
            "MONEY_PRINTER": "Central banks pumping liquidity. All assets rising.",
            "TRUMP_CRYPTO": "Trump administration bullish on crypto. Strategic reserve narrative.",
            "RECESSION_FEARS": "Worry about economic slowdown. Defensive positioning.",
            "RATE_HIKE_FEARS": "Fed tightening fears. Growth stocks under pressure.",
            "CRYPTO_WINTER": "Prolonged crypto bear market. Caution advised.",
            "REGULATION_FEARS": "Government crackdown concerns. Uncertainty high.",
            "BANK_CRISIS": "Banking system stress. Flight to safety.",
            "SECTOR_ROTATION": "Money moving between sectors. Watch for trends.",
            "ELECTION_CYCLE": "Political uncertainty affecting markets."
        }
        return descriptions.get(narrative_name, "Unknown narrative")
    
    def _generate_interpretation(self, dominant, all_narratives) -> str:
        """Generate interpretation of narrative landscape"""
        parts = []
        
        if dominant:
            name, data = dominant
            parts.append(f"📖 DOMINANT NARRATIVE: {name}")
            parts.append(f"Sentiment: {data['sentiment']}")
            parts.append(f"Description: {self._get_narrative_description(name)}")
            parts.append("")
        
        if len(all_narratives) > 1:
            parts.append("Other active narratives:")
            for name, data in all_narratives[1:4]:
                parts.append(f"  - {name} ({data['sentiment']})")
        
        parts.append("")
        parts.append("TRADING IMPLICATION:")
        if dominant and dominant[1]["sentiment"] == "BULLISH":
            parts.append("✅ Bullish narrative dominant. Favor long positions.")
        elif dominant and dominant[1]["sentiment"] == "BEARISH":
            parts.append("⚠️ Bearish narrative dominant. Favor defensive positions.")
        else:
            parts.append("➡️ Mixed narratives. Proceed with caution.")
        
        return "\n".join(parts)


# Singleton
_detector = None

def get_narrative_detector() -> NarrativeDetector:
    global _detector
    if _detector is None:
        _detector = NarrativeDetector()
    return _detector


async def detect_narratives(symbol: str = None) -> Dict:
    """Quick access to narrative detection"""
    return await get_narrative_detector().detect_current_narratives(symbol)
