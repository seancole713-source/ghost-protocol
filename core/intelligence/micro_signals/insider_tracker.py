"""
Ghost Protocol - Insider Transaction Tracker
Detect when insiders (CEO, CFO, Directors) are buying or selling BEFORE news breaks.

INSIGHT: Insiders know more than us. When they sell, pay attention.
         When they buy, something good might be coming.

Data Source: SEC EDGAR Form 4 filings via Finnhub
"""

import os
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")


@dataclass
class InsiderTransaction:
    """Single insider transaction"""
    symbol: str
    insider_name: str
    insider_title: str  # CEO, CFO, Director, 10% Owner
    transaction_type: str  # BUY, SELL, OPTION_EXERCISE
    shares: int
    price: float
    value: float
    date: str
    days_ago: int


class InsiderTracker:
    """
    Track insider buying and selling patterns.
    
    KEY INSIGHT: 
    - Cluster selling by multiple insiders = RED FLAG
    - Large buying by CEO/CFO = BULLISH SIGNAL
    - Selling right before earnings = VERY BAD SIGN
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def get_insider_transactions(self, symbol: str, days: int = 90) -> List[InsiderTransaction]:
        """Get recent insider transactions for a symbol"""
        transactions = []
        
        # Try Finnhub first (easier API)
        if FINNHUB_API_KEY:
            transactions = await self._finnhub_insider(symbol, days)
        
        return transactions
    
    async def _finnhub_insider(self, symbol: str, days: int) -> List[InsiderTransaction]:
        """Fetch insider transactions from Finnhub"""
        url = "https://finnhub.io/api/v1/stock/insider-transactions"
        params = {
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        transactions = []
                        
                        for t in data.get("data", []):
                            # Parse transaction date
                            trans_date = t.get("transactionDate", "")
                            if trans_date:
                                try:
                                    date_obj = datetime.strptime(trans_date, "%Y-%m-%d")
                                    days_ago = (datetime.now() - date_obj).days
                                    
                                    if days_ago <= days:
                                        # Determine transaction type
                                        shares = t.get("change", 0)
                                        if shares > 0:
                                            trans_type = "BUY"
                                        elif shares < 0:
                                            trans_type = "SELL"
                                            shares = abs(shares)
                                        else:
                                            trans_type = "OTHER"
                                        
                                        transactions.append(InsiderTransaction(
                                            symbol=symbol,
                                            insider_name=t.get("name", "Unknown"),
                                            insider_title=t.get("filingType", "Unknown"),
                                            transaction_type=trans_type,
                                            shares=int(shares),
                                            price=float(t.get("transactionPrice", 0) or 0),
                                            value=float(abs(shares) * (t.get("transactionPrice", 0) or 0)),
                                            date=trans_date,
                                            days_ago=days_ago
                                        ))
                                except Exception as e:
                                    logger.debug(f"Error parsing transaction: {e}")
                        
                        return transactions
        except Exception as e:
            logger.error(f"Finnhub insider error: {e}")
        
        return []
    
    async def analyze_insider_activity(self, symbol: str, days: int = 30) -> Dict:
        """
        Analyze insider activity for warning signs.
        
        RED FLAGS:
        - Multiple insiders selling
        - Large value sales
        - CEO/CFO selling
        - Selling before earnings
        
        GREEN FLAGS:
        - CEO/CFO buying
        - Multiple insiders buying
        - Buying at all-time highs (confidence)
        """
        transactions = await self.get_insider_transactions(symbol, days)
        
        if not transactions:
            return {
                "symbol": symbol,
                "has_data": False,
                "signal": "NEUTRAL",
                "signal_strength": 0,
                "confidence_adjustment": 0,
                "message": "No insider transaction data available"
            }
        
        # Separate buys and sells
        buys = [t for t in transactions if t.transaction_type == "BUY"]
        sells = [t for t in transactions if t.transaction_type == "SELL"]
        
        # Calculate totals
        total_buy_value = sum(t.value for t in buys)
        total_sell_value = sum(t.value for t in sells)
        buy_count = len(buys)
        sell_count = len(sells)
        
        # Check for C-suite activity (most important)
        c_suite_sells = [t for t in sells if any(
            title in t.insider_title.upper() 
            for title in ["CEO", "CFO", "COO", "CHIEF", "PRESIDENT"]
        )]
        c_suite_buys = [t for t in buys if any(
            title in t.insider_title.upper() 
            for title in ["CEO", "CFO", "COO", "CHIEF", "PRESIDENT"]
        )]
        
        # Check for cluster activity (multiple insiders same direction)
        unique_sellers = len(set(t.insider_name for t in sells))
        unique_buyers = len(set(t.insider_name for t in buys))
        
        # Recent activity (last 7 days more important)
        recent_sells = [t for t in sells if t.days_ago <= 7]
        recent_buys = [t for t in buys if t.days_ago <= 7]
        
        # Determine signal
        signal = "NEUTRAL"
        signal_strength = 0
        confidence_adjustment = 0
        warnings = []
        positives = []
        
        # BEARISH SIGNALS
        if len(c_suite_sells) > 0:
            signal = "BEARISH"
            signal_strength += 30
            confidence_adjustment -= 15
            warnings.append(f"⚠️ C-SUITE SELLING: {len(c_suite_sells)} executives sold shares")
        
        if unique_sellers >= 3:
            signal = "BEARISH"
            signal_strength += 20
            confidence_adjustment -= 10
            warnings.append(f"⚠️ CLUSTER SELLING: {unique_sellers} different insiders sold")
        
        if total_sell_value > 10_000_000:  # $10M+
            signal_strength += 15
            confidence_adjustment -= 8
            warnings.append(f"⚠️ LARGE SALES: ${total_sell_value:,.0f} total insider selling")
        
        if len(recent_sells) >= 2:
            signal_strength += 10
            confidence_adjustment -= 5
            warnings.append(f"⚠️ RECENT SELLING: {len(recent_sells)} sales in last 7 days")
        
        # BULLISH SIGNALS
        if len(c_suite_buys) > 0:
            if signal != "BEARISH":
                signal = "BULLISH"
            signal_strength += 25
            confidence_adjustment += 12
            positives.append(f"✅ C-SUITE BUYING: {len(c_suite_buys)} executives bought shares")
        
        if unique_buyers >= 2:
            signal_strength += 15
            confidence_adjustment += 8
            positives.append(f"✅ CLUSTER BUYING: {unique_buyers} different insiders bought")
        
        if total_buy_value > 1_000_000:  # $1M+
            signal_strength += 10
            confidence_adjustment += 5
            positives.append(f"✅ SIGNIFICANT BUYING: ${total_buy_value:,.0f} total insider buying")
        
        # Determine final signal
        if confidence_adjustment > 5:
            signal = "BULLISH"
        elif confidence_adjustment < -5:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "symbol": symbol,
            "has_data": True,
            "period_days": days,
            "signal": signal,
            "signal_strength": min(signal_strength, 100),
            "confidence_adjustment": max(-30, min(30, confidence_adjustment)),
            
            # Transaction summary
            "summary": {
                "total_transactions": len(transactions),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "total_buy_value": total_buy_value,
                "total_sell_value": total_sell_value,
                "net_value": total_buy_value - total_sell_value,
                "unique_buyers": unique_buyers,
                "unique_sellers": unique_sellers,
                "c_suite_buys": len(c_suite_buys),
                "c_suite_sells": len(c_suite_sells)
            },
            
            # Key signals
            "warnings": warnings,
            "positives": positives,
            
            # Recent notable transactions
            "notable_transactions": [
                {
                    "name": t.insider_name,
                    "title": t.insider_title,
                    "type": t.transaction_type,
                    "value": t.value,
                    "date": t.date,
                    "days_ago": t.days_ago
                }
                for t in sorted(transactions, key=lambda x: x.value, reverse=True)[:5]
            ],
            
            # Interpretation
            "interpretation": self._generate_interpretation(
                signal, warnings, positives, total_buy_value, total_sell_value
            )
        }
    
    def _generate_interpretation(self, signal: str, warnings: List, 
                                  positives: List, buy_value: float, 
                                  sell_value: float) -> str:
        """Generate human-readable interpretation"""
        parts = []
        
        if signal == "BEARISH":
            parts.append("🔴 INSIDER ACTIVITY: BEARISH")
            parts.append("Insiders with knowledge of company are SELLING.")
            parts.extend(warnings)
            parts.append("")
            parts.append("RECOMMENDATION: Be cautious with long positions.")
        elif signal == "BULLISH":
            parts.append("🟢 INSIDER ACTIVITY: BULLISH")
            parts.append("Insiders with knowledge of company are BUYING.")
            parts.extend(positives)
            parts.append("")
            parts.append("RECOMMENDATION: Supports bullish thesis.")
        else:
            parts.append("⚪ INSIDER ACTIVITY: NEUTRAL")
            parts.append("No significant insider activity pattern detected.")
        
        return "\n".join(parts)


# Singleton
_tracker = None

def get_insider_tracker() -> InsiderTracker:
    global _tracker
    if _tracker is None:
        _tracker = InsiderTracker()
    return _tracker


async def analyze_insiders(symbol: str, days: int = 30) -> Dict:
    """Quick access to insider analysis"""
    return await get_insider_tracker().analyze_insider_activity(symbol, days)
