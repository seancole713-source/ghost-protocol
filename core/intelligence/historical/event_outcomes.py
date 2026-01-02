"""
Ghost Protocol - Historical Event Outcomes
What happened LAST TIME this situation occurred?

INSIGHT: History doesn't repeat, but it rhymes.
         - What did BTC do after last halving?
         - What did stocks do after last rate cut?
         - What did this stock do before last earnings?

The past is the best predictor of the future.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Historical event outcomes database
HISTORICAL_EVENTS = {
    # Bitcoin Halving Events
    "BTC_HALVING": {
        "2012-11-28": {
            "price_at_event": 12,
            "price_1_month_later": 14,
            "price_6_months_later": 130,
            "price_1_year_later": 1000,
            "return_1_year": "+8233%"
        },
        "2016-07-09": {
            "price_at_event": 650,
            "price_1_month_later": 600,
            "price_6_months_later": 1000,
            "price_1_year_later": 2500,
            "return_1_year": "+285%"
        },
        "2020-05-11": {
            "price_at_event": 8600,
            "price_1_month_later": 9500,
            "price_6_months_later": 18000,
            "price_1_year_later": 58000,
            "return_1_year": "+574%"
        },
        "2024-04-20": {
            "price_at_event": 64000,
            "expected_pattern": "Historically +300-600% within 18 months",
            "current_phase": "Post-halving accumulation"
        },
        "insight": "BTC has NEVER failed to hit new ATH within 18 months of halving"
    },
    
    # Fed Rate Decisions
    "FED_RATE_CUT": {
        "pattern": "First rate cut after hiking cycle",
        "historical_outcomes": [
            {"date": "2019-07-31", "sp500_1_month": "+2%", "sp500_6_months": "+10%"},
            {"date": "2007-09-18", "sp500_1_month": "+4%", "sp500_6_months": "-10%", "note": "Before 2008 crash"},
            {"date": "2001-01-03", "sp500_1_month": "-3%", "sp500_6_months": "-15%", "note": "Dot-com bust"}
        ],
        "insight": "Rate cuts can be bullish OR signal problems ahead. Context matters."
    },
    
    # Earnings Surprises
    "EARNINGS_BEAT": {
        "average_1_day_move": "+3.2%",
        "average_1_week_move": "+2.1%",
        "pattern": "Beat usually leads to short-term pop, then consolidation"
    },
    "EARNINGS_MISS": {
        "average_1_day_move": "-5.8%",
        "average_1_week_move": "-4.2%",
        "pattern": "Miss usually leads to sharp drop, sometimes oversold bounce"
    },
    
    # Seasonal patterns
    "SANTA_RALLY": {
        "period": "Last 5 trading days of year + first 2 of new year",
        "historical_win_rate": "79%",
        "average_return": "+1.3%",
        "insight": "Historically bullish period for stocks"
    },
    "SELL_IN_MAY": {
        "period": "May through October",
        "historical_performance": "Weaker than November-April",
        "average_may_oct_return": "+2%",
        "average_nov_apr_return": "+7%"
    },
    "JANUARY_EFFECT": {
        "period": "First month of year",
        "historical_win_rate": "65%",
        "pattern": "Small caps tend to outperform in January",
        "insight": "Tax-loss selling ends, new money enters"
    },
    
    # Black Swan Events
    "COVID_CRASH": {
        "date": "2020-03-12",
        "sp500_drop": "-34% in 23 days",
        "recovery_time": "5 months to new highs",
        "btc_drop": "-50% in 1 day",
        "btc_recovery": "9 months to new highs",
        "insight": "Sharp crashes often followed by V-shaped recovery when Fed intervenes"
    },
    "FTX_COLLAPSE": {
        "date": "2022-11-08",
        "btc_drop": "-25% in 1 week",
        "total_crypto_impact": "-$200B market cap",
        "recovery_time": "14 months to previous levels"
    }
}


class EventOutcomesDatabase:
    """
    Query historical event outcomes to inform predictions.
    
    KEY QUESTION: "What happened last time?"
    """
    
    def __init__(self):
        self.events = HISTORICAL_EVENTS
    
    async def get_similar_event_outcomes(self, event_type: str, symbol: str = None) -> Dict:
        """
        Get historical outcomes for similar events.
        
        Args:
            event_type: Type of event (EARNINGS, HALVING, FED_RATE, etc.)
            symbol: Optional symbol for symbol-specific history
        
        Returns:
            Historical outcomes and patterns
        """
        # Look up event in database
        event_data = self.events.get(event_type)
        
        if not event_data:
            return {
                "event_type": event_type,
                "has_data": False,
                "message": f"No historical data for event type: {event_type}"
            }
        
        return {
            "event_type": event_type,
            "has_data": True,
            "historical_data": event_data,
            "insight": event_data.get("insight", event_data.get("pattern", "")),
            "confidence_adjustment": self._calculate_adjustment(event_type, event_data)
        }
    
    async def what_happened_last_time(self, symbol: str, event_description: str) -> Dict:
        """
        Natural language query: "What happened last time X?"
        
        Examples:
        - "What happened to BTC after last halving?"
        - "What happened to AAPL before last earnings?"
        - "What happened during last Fed rate cut?"
        """
        event_description = event_description.lower()
        
        # Match to known patterns
        if "halving" in event_description and symbol.upper() == "BTC":
            return await self.get_similar_event_outcomes("BTC_HALVING", symbol)
        elif "rate cut" in event_description or "fed" in event_description:
            return await self.get_similar_event_outcomes("FED_RATE_CUT", symbol)
        elif "earnings" in event_description:
            return await self.get_similar_event_outcomes("EARNINGS_BEAT", symbol)
        elif "christmas" in event_description or "santa" in event_description:
            return await self.get_similar_event_outcomes("SANTA_RALLY", symbol)
        elif "covid" in event_description or "crash" in event_description:
            return await self.get_similar_event_outcomes("COVID_CRASH", symbol)
        elif "january" in event_description:
            return await self.get_similar_event_outcomes("JANUARY_EFFECT", symbol)
        
        return {
            "symbol": symbol,
            "query": event_description,
            "has_data": False,
            "message": "Could not match query to known historical pattern"
        }
    
    def _calculate_adjustment(self, event_type: str, event_data: Dict) -> int:
        """Calculate confidence adjustment based on historical pattern strength"""
        if event_type == "BTC_HALVING":
            return 15  # Strong bullish historical pattern
        elif event_type == "SANTA_RALLY":
            return 5   # Moderate bullish pattern
        elif event_type == "EARNINGS_MISS":
            return -10  # Bearish pattern
        elif event_type == "JANUARY_EFFECT":
            return 5   # Moderate bullish
        return 0
    
    async def get_seasonal_pattern(self, symbol: str, month: int = None) -> Dict:
        """
        Get seasonal pattern for current period.
        """
        if month is None:
            month = datetime.now().month
        
        # Define seasonal tendencies
        seasonal_patterns = {
            1: {"name": "January Effect", "tendency": "BULLISH", "note": "Small caps often outperform"},
            2: {"name": "February", "tendency": "NEUTRAL", "note": "Mixed historically"},
            3: {"name": "March", "tendency": "NEUTRAL", "note": "End of Q1 positioning"},
            4: {"name": "April", "tendency": "BULLISH", "note": "Tax refund season, positive"},
            5: {"name": "Sell in May", "tendency": "BEARISH", "note": "Start of weaker period"},
            6: {"name": "June", "tendency": "NEUTRAL", "note": "Mid-year repositioning"},
            7: {"name": "July", "tendency": "BULLISH", "note": "Often strong month"},
            8: {"name": "August", "tendency": "BEARISH", "note": "Vacation season, low volume"},
            9: {"name": "September", "tendency": "BEARISH", "note": "Historically worst month"},
            10: {"name": "October", "tendency": "NEUTRAL", "note": "Crash month reputation, but often positive"},
            11: {"name": "November", "tendency": "BULLISH", "note": "Strong historically"},
            12: {"name": "December", "tendency": "BULLISH", "note": "Santa rally, tax-loss harvesting ends"}
        }
        
        pattern = seasonal_patterns.get(month, {})
        
        # Check for special periods
        day = datetime.now().day
        special_period = None
        
        # Santa Rally: Last week of December + first 2 days of January
        if (month == 12 and day >= 26) or (month == 1 and day <= 2):
            special_period = {
                "name": "Santa Rally",
                "tendency": "BULLISH",
                "historical_win_rate": "79%",
                "average_return": "+1.3%"
            }
        
        # Week 1 of January (historically weak for some stocks)
        elif month == 1 and day <= 7:
            special_period = {
                "name": "First Week of January",
                "tendency": "MIXED",
                "note": "Profit-taking after year-end rally"
            }
        
        return {
            "symbol": symbol,
            "month": month,
            "day": day,
            "pattern": pattern,
            "special_period": special_period,
            "confidence_adjustment": 5 if pattern.get("tendency") == "BULLISH" else -5 if pattern.get("tendency") == "BEARISH" else 0,
            "interpretation": self._generate_seasonal_interpretation(pattern, special_period)
        }
    
    def _generate_seasonal_interpretation(self, pattern: Dict, special_period: Dict = None) -> str:
        """Generate seasonal interpretation"""
        parts = []
        
        if special_period:
            parts.append(f"📅 SPECIAL PERIOD: {special_period['name']}")
            parts.append(f"Tendency: {special_period['tendency']}")
            if special_period.get('historical_win_rate'):
                parts.append(f"Historical Win Rate: {special_period['historical_win_rate']}")
            if special_period.get('note'):
                parts.append(f"Note: {special_period['note']}")
            parts.append("")
        
        if pattern:
            parts.append(f"📊 Monthly Pattern: {pattern.get('name', 'Unknown')}")
            parts.append(f"Tendency: {pattern.get('tendency', 'NEUTRAL')}")
            parts.append(f"Note: {pattern.get('note', '')}")
        
        return "\n".join(parts)


# Singleton
_database = None

def get_event_database() -> EventOutcomesDatabase:
    global _database
    if _database is None:
        _database = EventOutcomesDatabase()
    return _database


async def get_historical_outcomes(event_type: str, symbol: str = None) -> Dict:
    """Quick access to historical outcomes"""
    return await get_event_database().get_similar_event_outcomes(event_type, symbol)


async def what_happened_last_time(symbol: str, event: str) -> Dict:
    """Natural language historical query"""
    return await get_event_database().what_happened_last_time(symbol, event)
