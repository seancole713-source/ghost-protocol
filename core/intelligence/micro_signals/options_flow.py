"""
Ghost Protocol - Options Flow Analyzer
Detect unusual options activity that precedes major moves.

INSIGHT: Smart money often uses options to position before events.
         Unusual put buying = Someone betting on downside
         Unusual call buying = Someone betting on upside
         Large single trades = Institutional positioning

Data Sources:
- Unusual Whales API
- Finnhub
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
UNUSUAL_WHALES_KEY = os.getenv("UNUSUAL_WHALES_API_KEY", "")


@dataclass
class OptionFlow:
    """Single options transaction"""
    symbol: str
    option_type: str  # "CALL" or "PUT"
    strike: float
    expiry: str
    premium: float
    volume: int
    open_interest: int
    sentiment: str  # "BULLISH", "BEARISH", "NEUTRAL"
    unusual_score: float  # How unusual is this trade


class OptionsFlowAnalyzer:
    """
    Analyze options flow for directional signals.
    
    KEY PATTERNS:
    - Unusual put volume = Hedge funds positioning for downside
    - Unusual call volume = Expecting upside
    - Large premium trades = Institutional conviction
    - Put/Call ratio extremes = Sentiment indicator
    """
    
    def __init__(self):
        self.cache = {}
    
    async def get_options_flow(self, symbol: str, days: int = 1) -> List[OptionFlow]:
        """Get recent options flow data"""
        flows = []
        
        if UNUSUAL_WHALES_KEY:
            flows.extend(await self._unusual_whales_flow(symbol, days))
        elif FINNHUB_API_KEY:
            flows.extend(await self._finnhub_options(symbol))
        
        return flows
    
    async def _unusual_whales_flow(self, symbol: str, days: int) -> List[OptionFlow]:
        """Fetch from Unusual Whales API"""
        url = f"https://api.unusualwhales.com/api/stock/{symbol}/options-flow"
        headers = {"Authorization": f"Bearer {UNUSUAL_WHALES_KEY}"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        flows = []
                        
                        for f in data.get("data", []):
                            flows.append(OptionFlow(
                                symbol=symbol,
                                option_type=f.get("put_call", "UNKNOWN").upper(),
                                strike=float(f.get("strike", 0)),
                                expiry=f.get("expiry", ""),
                                premium=float(f.get("premium", 0)),
                                volume=int(f.get("volume", 0)),
                                open_interest=int(f.get("open_interest", 0)),
                                sentiment=self._determine_sentiment(f),
                                unusual_score=float(f.get("unusual_score", 0))
                            ))
                        
                        return flows
        except Exception as e:
            logger.error(f"Unusual Whales error: {e}")
        
        return []
    
    async def _finnhub_options(self, symbol: str) -> List[OptionFlow]:
        """Fetch options data from Finnhub"""
        # Finnhub provides aggregated options data
        url = "https://finnhub.io/api/v1/stock/option-chain"
        params = {
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Process options chain data
                        # This gives us strikes and OI, not flow
                        pass
        except Exception as e:
            logger.error(f"Finnhub options error: {e}")
        
        return []
    
    def _determine_sentiment(self, flow_data: Dict) -> str:
        """Determine sentiment of options trade"""
        put_call = flow_data.get("put_call", "").upper()
        side = flow_data.get("side", "").upper()  # BUY or SELL
        
        if put_call == "CALL":
            return "BULLISH" if side == "BUY" else "BEARISH"
        elif put_call == "PUT":
            return "BEARISH" if side == "BUY" else "BULLISH"
        return "NEUTRAL"
    
    async def analyze_options_activity(self, symbol: str) -> Dict:
        """
        Analyze options activity for trading signals.
        
        BEARISH SIGNALS:
        - High put volume vs calls
        - Large premium put trades
        - Unusual activity on downside strikes
        
        BULLISH SIGNALS:
        - High call volume vs puts
        - Large premium call trades
        - Unusual activity on upside strikes
        """
        flows = await self.get_options_flow(symbol)
        
        if not flows:
            return {
                "symbol": symbol,
                "has_data": False,
                "signal": "NEUTRAL",
                "signal_strength": 0,
                "confidence_adjustment": 0,
                "message": "No options flow data available (may need UNUSUAL_WHALES_API_KEY)"
            }
        
        # Separate calls and puts
        calls = [f for f in flows if f.option_type == "CALL"]
        puts = [f for f in flows if f.option_type == "PUT"]
        
        # Calculate totals
        call_premium = sum(f.premium for f in calls)
        put_premium = sum(f.premium for f in puts)
        call_volume = sum(f.volume for f in calls)
        put_volume = sum(f.volume for f in puts)
        
        # Put/Call ratio
        pc_ratio = put_volume / call_volume if call_volume > 0 else 1.0
        
        # Find unusual trades
        unusual_calls = [f for f in calls if f.unusual_score > 50]
        unusual_puts = [f for f in puts if f.unusual_score > 50]
        
        # Bullish vs Bearish sentiment
        bullish_flow = sum(f.premium for f in flows if f.sentiment == "BULLISH")
        bearish_flow = sum(f.premium for f in flows if f.sentiment == "BEARISH")
        
        # Determine signal
        signal = "NEUTRAL"
        signal_strength = 0
        confidence_adjustment = 0
        warnings = []
        positives = []
        
        # BEARISH SIGNALS
        if pc_ratio > 1.5:
            signal = "BEARISH"
            signal_strength += 25
            confidence_adjustment -= 12
            warnings.append(f"⚠️ HIGH PUT/CALL RATIO: {pc_ratio:.2f} (elevated put buying)")
        
        if len(unusual_puts) >= 3:
            signal_strength += 20
            confidence_adjustment -= 10
            warnings.append(f"⚠️ UNUSUAL PUT ACTIVITY: {len(unusual_puts)} unusual put trades detected")
        
        if put_premium > call_premium * 2:
            signal_strength += 15
            confidence_adjustment -= 8
            warnings.append(f"⚠️ PUT PREMIUM DOMINANCE: ${put_premium:,.0f} vs ${call_premium:,.0f} calls")
        
        # BULLISH SIGNALS
        if pc_ratio < 0.6:
            signal = "BULLISH"
            signal_strength += 25
            confidence_adjustment += 12
            positives.append(f"✅ LOW PUT/CALL RATIO: {pc_ratio:.2f} (elevated call buying)")
        
        if len(unusual_calls) >= 3:
            signal_strength += 20
            confidence_adjustment += 10
            positives.append(f"✅ UNUSUAL CALL ACTIVITY: {len(unusual_calls)} unusual call trades detected")
        
        if call_premium > put_premium * 2:
            signal_strength += 15
            confidence_adjustment += 8
            positives.append(f"✅ CALL PREMIUM DOMINANCE: ${call_premium:,.0f} vs ${put_premium:,.0f} puts")
        
        # Final signal
        if confidence_adjustment > 5:
            signal = "BULLISH"
        elif confidence_adjustment < -5:
            signal = "BEARISH"
        
        return {
            "symbol": symbol,
            "has_data": True,
            "signal": signal,
            "signal_strength": min(signal_strength, 100),
            "confidence_adjustment": max(-25, min(25, confidence_adjustment)),
            
            # Options summary
            "summary": {
                "total_flow_count": len(flows),
                "call_count": len(calls),
                "put_count": len(puts),
                "call_premium": call_premium,
                "put_premium": put_premium,
                "call_volume": call_volume,
                "put_volume": put_volume,
                "put_call_ratio": round(pc_ratio, 2),
                "unusual_calls": len(unusual_calls),
                "unusual_puts": len(unusual_puts),
                "bullish_premium": bullish_flow,
                "bearish_premium": bearish_flow
            },
            
            # Signals
            "warnings": warnings,
            "positives": positives,
            
            # Interpretation
            "interpretation": self._generate_interpretation(signal, pc_ratio, warnings, positives)
        }
    
    def _generate_interpretation(self, signal: str, pc_ratio: float,
                                  warnings: List, positives: List) -> str:
        """Generate human-readable interpretation"""
        parts = []
        
        if signal == "BEARISH":
            parts.append("🔴 OPTIONS FLOW: BEARISH")
            parts.append(f"Put/Call ratio: {pc_ratio:.2f} (>1 = more puts)")
            parts.append("Smart money appears to be betting on downside.")
            parts.extend(warnings)
        elif signal == "BULLISH":
            parts.append("🟢 OPTIONS FLOW: BULLISH")
            parts.append(f"Put/Call ratio: {pc_ratio:.2f} (<1 = more calls)")
            parts.append("Smart money appears to be betting on upside.")
            parts.extend(positives)
        else:
            parts.append("⚪ OPTIONS FLOW: NEUTRAL")
            parts.append(f"Put/Call ratio: {pc_ratio:.2f} (balanced)")
        
        return "\n".join(parts)


# Singleton
_analyzer = None

def get_options_analyzer() -> OptionsFlowAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = OptionsFlowAnalyzer()
    return _analyzer


async def analyze_options(symbol: str) -> Dict:
    """Quick access to options analysis"""
    return await get_options_analyzer().analyze_options_activity(symbol)
