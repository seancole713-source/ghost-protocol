"""
Ghost Protocol - Whale Movement Detector
Track large wallet movements that precede major price moves.

INSIGHT: Whales accumulate in silence, then the price moves.
         Whales move to exchanges before dumping.
         Whales move from exchanges before pumping.

Data Sources: 
- Whale Alert API
- On-chain analysis
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WHALE_ALERT_API_KEY = os.getenv("WHALE_ALERT_API_KEY", "")

# Minimum USD value to consider "whale" activity
WHALE_THRESHOLD = {
    "BTC": 10_000_000,   # $10M
    "ETH": 5_000_000,    # $5M
    "DEFAULT": 1_000_000  # $1M
}


@dataclass
class WhaleTransaction:
    """Single whale transaction"""
    symbol: str
    from_type: str  # "exchange", "wallet", "unknown"
    to_type: str    # "exchange", "wallet", "unknown"
    amount: float
    amount_usd: float
    timestamp: datetime
    tx_hash: str
    significance: str  # "high", "medium", "low"


class WhaleDetector:
    """
    Detect whale movements that signal upcoming price action.
    
    KEY PATTERNS:
    - Whale → Exchange = Preparing to SELL (bearish)
    - Exchange → Whale = Accumulating (bullish)
    - Whale → Whale = Possible OTC, neutral
    - Multiple whales same direction = STRONG signal
    """
    
    def __init__(self):
        self.cache = {}
        self.exchange_addresses = self._load_exchange_addresses()
    
    def _load_exchange_addresses(self) -> set:
        """Load known exchange names"""
        return {
            "binance", "coinbase", "kraken", "ftx", "bitfinex",
            "huobi", "okex", "kucoin", "bybit", "bitget"
        }
    
    async def get_whale_transactions(self, symbol: str, hours: int = 24) -> List[WhaleTransaction]:
        """Get recent whale transactions for a symbol"""
        transactions = []
        
        # Try Whale Alert API
        if WHALE_ALERT_API_KEY:
            transactions.extend(await self._whale_alert_txns(symbol, hours))
        
        return transactions
    
    async def _whale_alert_txns(self, symbol: str, hours: int) -> List[WhaleTransaction]:
        """Fetch from Whale Alert API"""
        url = "https://api.whale-alert.io/v1/transactions"
        
        min_value = WHALE_THRESHOLD.get(symbol.upper(), WHALE_THRESHOLD["DEFAULT"])
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        params = {
            "api_key": WHALE_ALERT_API_KEY,
            "currency": symbol.lower(),
            "min_value": min_value,
            "start": start_time,
            "limit": 100
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        transactions = []
                        
                        for t in data.get("transactions", []):
                            from_type = self._classify_address(t.get("from", {}).get("owner_type", ""))
                            to_type = self._classify_address(t.get("to", {}).get("owner_type", ""))
                            
                            transactions.append(WhaleTransaction(
                                symbol=symbol.upper(),
                                from_type=from_type,
                                to_type=to_type,
                                amount=t.get("amount", 0),
                                amount_usd=t.get("amount_usd", 0),
                                timestamp=datetime.fromtimestamp(t.get("timestamp", 0)),
                                tx_hash=t.get("hash", ""),
                                significance=self._calculate_significance(t.get("amount_usd", 0), symbol)
                            ))
                        
                        return transactions
        except Exception as e:
            logger.error(f"Whale Alert error: {e}")
        
        return []
    
    def _classify_address(self, owner_type: str) -> str:
        """Classify address type"""
        if owner_type in ["exchange", "Exchange"]:
            return "exchange"
        elif owner_type in ["wallet", "Wallet", "unknown", ""]:
            return "wallet"
        return "unknown"
    
    def _calculate_significance(self, usd_value: float, symbol: str) -> str:
        """Calculate significance of transaction"""
        threshold = WHALE_THRESHOLD.get(symbol.upper(), WHALE_THRESHOLD["DEFAULT"])
        
        if usd_value >= threshold * 10:
            return "critical"
        elif usd_value >= threshold * 5:
            return "high"
        elif usd_value >= threshold * 2:
            return "medium"
        else:
            return "low"
    
    async def analyze_whale_activity(self, symbol: str, hours: int = 24) -> Dict:
        """
        Analyze whale activity for trading signals.
        
        BEARISH SIGNALS:
        - Multiple whale → exchange moves
        - Large single dump to exchange
        - Increasing exchange whale deposits
        
        BULLISH SIGNALS:
        - Multiple exchange → whale moves (accumulation)
        - Whales moving off exchanges
        - Decreasing exchange balances
        """
        transactions = await self.get_whale_transactions(symbol, hours)
        
        if not transactions:
            return {
                "symbol": symbol,
                "has_data": False,
                "signal": "NEUTRAL",
                "signal_strength": 0,
                "confidence_adjustment": 0,
                "message": "No whale transaction data available (may need WHALE_ALERT_API_KEY)"
            }
        
        # Classify transactions
        to_exchange = [t for t in transactions if t.to_type == "exchange"]
        from_exchange = [t for t in transactions if t.from_type == "exchange" and t.to_type == "wallet"]
        
        # Calculate flows
        inflow_to_exchange = sum(t.amount_usd for t in to_exchange)
        outflow_from_exchange = sum(t.amount_usd for t in from_exchange)
        net_flow = outflow_from_exchange - inflow_to_exchange  # Positive = bullish
        
        # Count significant transactions
        critical_to_exchange = len([t for t in to_exchange if t.significance in ["critical", "high"]])
        critical_from_exchange = len([t for t in from_exchange if t.significance in ["critical", "high"]])
        
        # Determine signal
        signal = "NEUTRAL"
        signal_strength = 0
        confidence_adjustment = 0
        warnings = []
        positives = []
        
        # BEARISH: Whales moving TO exchanges (preparing to sell)
        if inflow_to_exchange > 50_000_000:  # $50M+
            signal = "BEARISH"
            signal_strength += 40
            confidence_adjustment -= 20
            warnings.append(f"🚨 MASSIVE EXCHANGE INFLOW: ${inflow_to_exchange:,.0f} moved to exchanges")
        elif inflow_to_exchange > 10_000_000:  # $10M+
            signal_strength += 20
            confidence_adjustment -= 10
            warnings.append(f"⚠️ LARGE EXCHANGE INFLOW: ${inflow_to_exchange:,.0f} moved to exchanges")
        
        if critical_to_exchange >= 3:
            signal_strength += 15
            confidence_adjustment -= 8
            warnings.append(f"⚠️ MULTIPLE WHALE DEPOSITS: {critical_to_exchange} large txns to exchanges")
        
        # BULLISH: Whales moving FROM exchanges (accumulating)
        if outflow_from_exchange > 50_000_000:  # $50M+
            if signal != "BEARISH":
                signal = "BULLISH"
            signal_strength += 35
            confidence_adjustment += 18
            positives.append(f"🐋 MASSIVE ACCUMULATION: ${outflow_from_exchange:,.0f} withdrawn from exchanges")
        elif outflow_from_exchange > 10_000_000:  # $10M+
            signal_strength += 18
            confidence_adjustment += 9
            positives.append(f"✅ WHALE ACCUMULATION: ${outflow_from_exchange:,.0f} withdrawn from exchanges")
        
        if critical_from_exchange >= 3:
            signal_strength += 12
            confidence_adjustment += 6
            positives.append(f"✅ MULTIPLE WITHDRAWALS: {critical_from_exchange} large outflows from exchanges")
        
        # Net flow analysis
        if net_flow > 20_000_000:
            positives.append(f"✅ NET OUTFLOW: ${net_flow:,.0f} net leaving exchanges (bullish)")
        elif net_flow < -20_000_000:
            warnings.append(f"⚠️ NET INFLOW: ${abs(net_flow):,.0f} net entering exchanges (bearish)")
        
        # Final signal determination
        if confidence_adjustment > 5:
            signal = "BULLISH"
        elif confidence_adjustment < -5:
            signal = "BEARISH"
        
        return {
            "symbol": symbol,
            "has_data": True,
            "period_hours": hours,
            "signal": signal,
            "signal_strength": min(signal_strength, 100),
            "confidence_adjustment": max(-30, min(30, confidence_adjustment)),
            
            # Flow summary
            "flows": {
                "total_transactions": len(transactions),
                "to_exchange_count": len(to_exchange),
                "from_exchange_count": len(from_exchange),
                "inflow_to_exchange_usd": inflow_to_exchange,
                "outflow_from_exchange_usd": outflow_from_exchange,
                "net_flow_usd": net_flow,
                "net_flow_direction": "OUTFLOW (Bullish)" if net_flow > 0 else "INFLOW (Bearish)"
            },
            
            # Signals
            "warnings": warnings,
            "positives": positives,
            
            # Interpretation
            "interpretation": self._generate_interpretation(
                signal, net_flow, warnings, positives
            )
        }
    
    def _generate_interpretation(self, signal: str, net_flow: float,
                                  warnings: List, positives: List) -> str:
        """Generate human-readable interpretation"""
        parts = []
        
        if signal == "BEARISH":
            parts.append("🔴 WHALE ACTIVITY: BEARISH")
            parts.append("Large holders are moving crypto TO exchanges.")
            parts.append("This often precedes selling pressure.")
            parts.extend(warnings)
        elif signal == "BULLISH":
            parts.append("🟢 WHALE ACTIVITY: BULLISH")
            parts.append("Large holders are moving crypto FROM exchanges.")
            parts.append("This suggests accumulation and reduced sell pressure.")
            parts.extend(positives)
        else:
            parts.append("⚪ WHALE ACTIVITY: NEUTRAL")
            parts.append("No significant whale movement pattern detected.")
        
        return "\n".join(parts)


# Singleton
_detector = None

def get_whale_detector() -> WhaleDetector:
    global _detector
    if _detector is None:
        _detector = WhaleDetector()
    return _detector


async def analyze_whales(symbol: str, hours: int = 24) -> Dict:
    """Quick access to whale analysis"""
    return await get_whale_detector().analyze_whale_activity(symbol, hours)
