"""
Santiment Data Integration for Ghost Protocol
Provides social sentiment and on-chain metrics
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import requests

logger = logging.getLogger(__name__)


class SantimentProvider:
    """Fetch and process Santiment data"""
    
    def __init__(self):
        self.api_key = os.getenv("SANTIMENT_API_KEY", "")
        self.base_url = "https://api.santiment.net/graphql"
        self.enabled = bool(self.api_key)
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        
        if not self.enabled:
            logger.info("Santiment API key not set - social signals disabled")
    
    def _query(self, query: str) -> Optional[Dict]:
        """Execute GraphQL query"""
        if not self.enabled:
            return None
        
        try:
            headers = {"Authorization": f"Apikey {self.api_key}"}
            response = requests.post(
                self.base_url,
                json={"query": query},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Santiment query failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Santiment query error: {e}")
            return None
    
    def get_social_volume(self, symbol: str) -> Optional[Dict]:
        """
        Get social media mention volume for a symbol
        
        Returns:
            Dict with volume, change_24h, sentiment
        """
        cache_key = f"social_{symbol}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["ts"] < self.cache_ttl:
                return cached["data"]
        
        # Map symbol to Santiment slug
        slug_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple",
            "ADA": "cardano",
            "DOGE": "dogecoin",
            "LINK": "chainlink",
            "AVAX": "avalanche-2",
            "DOT": "polkadot-new",
            "MATIC": "matic-network",
            "PEPE": "pepe",
            "SHIB": "shiba-inu",
        }
        
        slug = slug_map.get(symbol.upper())
        if not slug:
            return None
        
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=7)
        
        query = f"""
        {{
            getMetric(metric: "social_volume_total") {{
                timeseriesData(
                    slug: "{slug}"
                    from: "{from_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    to: "{to_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    interval: "1d"
                ) {{
                    datetime
                    value
                }}
            }}
        }}
        """
        
        result = self._query(query)
        if not result:
            return None
        
        try:
            timeseries = result.get("data", {}).get("getMetric", {}).get("timeseriesData", [])
            if len(timeseries) >= 2:
                current = timeseries[-1]["value"]
                previous = timeseries[-2]["value"]
                change = ((current - previous) / previous * 100) if previous > 0 else 0
                
                # Calculate 7-day average
                avg = sum(t["value"] for t in timeseries) / len(timeseries)
                vs_avg = ((current - avg) / avg * 100) if avg > 0 else 0
                
                data = {
                    "symbol": symbol,
                    "volume": current,
                    "change_24h": round(change, 1),
                    "vs_7d_avg": round(vs_avg, 1),
                    "signal": "BULLISH" if change > 20 else "BEARISH" if change < -20 else "NEUTRAL"
                }
                
                self.cache[cache_key] = {"ts": time.time(), "data": data}
                return data
        except Exception as e:
            logger.error(f"Failed to parse social volume: {e}")
        
        return None
    
    def get_whale_activity(self, symbol: str) -> Optional[Dict]:
        """
        Get whale transaction activity
        
        Returns:
            Dict with large_txn_count, whale_accumulating
        """
        slug_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple"
        }
        
        slug = slug_map.get(symbol.upper())
        if not slug:
            return None
        
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=2)
        
        query = f"""
        {{
            getMetric(metric: "whale_transaction_count_100k_usd_to_inf") {{
                timeseriesData(
                    slug: "{slug}"
                    from: "{from_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    to: "{to_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    interval: "1d"
                ) {{
                    datetime
                    value
                }}
            }}
        }}
        """
        
        result = self._query(query)
        if not result:
            return None
        
        try:
            timeseries = result.get("data", {}).get("getMetric", {}).get("timeseriesData", [])
            if timeseries:
                current = timeseries[-1]["value"]
                previous = timeseries[0]["value"] if len(timeseries) > 1 else current
                change = ((current - previous) / previous * 100) if previous > 0 else 0
                
                return {
                    "symbol": symbol,
                    "whale_txns_24h": int(current),
                    "change_pct": round(change, 1),
                    "signal": "ACCUMULATING" if change > 30 else "DISTRIBUTING" if change < -30 else "NEUTRAL"
                }
        except Exception as e:
            logger.error(f"Failed to parse whale activity: {e}")
        
        return None
    
    def get_sentiment_score(self, symbol: str) -> Optional[Dict]:
        """
        Get overall sentiment score combining social + on-chain
        
        Returns:
            Dict with score (0-100), interpretation, signals
        """
        social = self.get_social_volume(symbol)
        whales = self.get_whale_activity(symbol)
        
        if not social and not whales:
            # Return a neutral fallback if no data
            return {
                "symbol": symbol,
                "score": 50,
                "interpretation": "NEUTRAL",
                "signals": ["No Santiment data available (API key required)"],
                "components": {
                    "social": None,
                    "whales": None
                }
            }
        
        # Calculate composite score
        score = 50  # Neutral baseline
        signals = []
        
        if social:
            if social["change_24h"] > 50:
                score += 15
                signals.append("High social buzz (+15)")
            elif social["change_24h"] > 20:
                score += 8
                signals.append("Rising social interest (+8)")
            elif social["change_24h"] < -30:
                score -= 10
                signals.append("Declining interest (-10)")
        
        if whales:
            if whales["signal"] == "ACCUMULATING":
                score += 20
                signals.append("Whale accumulation (+20)")
            elif whales["signal"] == "DISTRIBUTING":
                score -= 15
                signals.append("Whale distribution (-15)")
        
        # Clamp score
        score = max(0, min(100, score))
        
        return {
            "symbol": symbol,
            "score": score,
            "interpretation": "BULLISH" if score > 65 else "BEARISH" if score < 35 else "NEUTRAL",
            "signals": signals,
            "components": {
                "social": social,
                "whales": whales
            }
        }


# Singleton instance
_provider: Optional[SantimentProvider] = None


def get_santiment() -> SantimentProvider:
    """Get or create SantimentProvider singleton"""
    global _provider
    if _provider is None:
        _provider = SantimentProvider()
    return _provider


def get_sentiment_signal(symbol: str) -> Optional[Dict]:
    """Get sentiment signal for use in predictions"""
    return get_santiment().get_sentiment_score(symbol)


def is_enabled() -> bool:
    """Check if Santiment is enabled"""
    return get_santiment().enabled
