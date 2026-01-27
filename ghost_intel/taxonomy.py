"""
GHOST INTEL - EVENT TAXONOMY
=============================
Classification system for all intelligence events.

Categories map to the 8 intelligence layers and allow
for consistent filtering and impact assessment.

Author: Ghost AI
Date: 2026-01-26
"""

import logging
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("ghost.intel")


class EventCategory(Enum):
    """
    All event categories across the 8 layers.
    """
    # Layer 1: Macro Data
    CPI = "cpi"
    CORE_CPI = "core_cpi"
    PPI = "ppi"
    PCE = "pce"
    CORE_PCE = "core_pce"
    NFP = "nfp"
    UNEMPLOYMENT = "unemployment"
    JOBLESS_CLAIMS = "jobless_claims"
    GDP = "gdp"
    GDP_GROWTH = "gdp_growth"
    RETAIL_SALES = "retail_sales"
    ISM_MANUFACTURING = "ism_manufacturing"
    ISM_SERVICES = "ism_services"
    HOUSING = "housing"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    
    # Layer 2: Rates & Liquidity
    FOMC = "fomc"
    RATE_DECISION = "rate_decision"
    FED_SPEECH = "fed_speech"
    YIELD_MOVE = "yield_move"
    YIELD_CURVE = "yield_curve"
    DXY = "dxy"
    VIX_SPIKE = "vix_spike"
    LIQUIDITY_STRESS = "liquidity_stress"
    
    # Layer 3: Corporate
    EARNINGS = "earnings"
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    GUIDANCE_RAISE = "guidance_raise"
    GUIDANCE_LOWER = "guidance_lower"
    REVENUE = "revenue"
    BUYBACK = "buyback"
    DIVIDEND = "dividend"
    M_AND_A = "m_and_a"
    IPO = "ipo"
    INSIDER_TRADE = "insider_trade"
    SEC_FILING = "sec_filing"
    
    # Layer 4: Politics
    TARIFF = "tariff"
    SANCTION = "sanction"
    REGULATION = "regulation"
    TAX_POLICY = "tax_policy"
    ANTITRUST = "antitrust"
    INFRASTRUCTURE = "infrastructure"
    ELECTION = "election"
    LEGISLATION = "legislation"
    
    # Layer 5: Geopolitics
    CONFLICT = "conflict"
    WAR = "war"
    TERRORISM = "terrorism"
    ENERGY_CRISIS = "energy_crisis"
    OIL_SHOCK = "oil_shock"
    SHIPPING_DISRUPTION = "shipping_disruption"
    TRADE_WAR = "trade_war"
    DIPLOMATIC = "diplomatic"
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC = "pandemic"
    
    # Layer 6: Key Individuals
    CEO_STATEMENT = "ceo_statement"
    FED_CHAIR = "fed_chair"
    ELON_MUSK = "elon_musk"
    INFLUENCER = "influencer"
    INSIDER = "insider"
    
    # Layer 7: Social
    TRENDING = "trending"
    VIRAL = "viral"
    WSB_MENTION = "wsb_mention"
    STOCKTWITS = "stocktwits"
    TWITTER = "twitter"
    
    # Layer 8: Positioning
    PUT_CALL_EXTREME = "put_call_extreme"
    GAMMA_SQUEEZE = "gamma_squeeze"
    SHORT_SQUEEZE = "short_squeeze"
    OPTIONS_FLOW = "options_flow"
    VOLUME_ANOMALY = "volume_anomaly"
    LIQUIDATION = "liquidation"
    
    # Generic
    NEWS = "news"
    ALERT = "alert"
    OTHER = "other"


class EventTaxonomy:
    """
    Event classification and categorization system.
    """
    
    # Category to Layer mapping
    LAYER_MAP = {
        # Macro
        EventCategory.CPI: "macro",
        EventCategory.CORE_CPI: "macro",
        EventCategory.PPI: "macro",
        EventCategory.PCE: "macro",
        EventCategory.CORE_PCE: "macro",
        EventCategory.NFP: "macro",
        EventCategory.UNEMPLOYMENT: "macro",
        EventCategory.JOBLESS_CLAIMS: "macro",
        EventCategory.GDP: "macro",
        EventCategory.GDP_GROWTH: "macro",
        EventCategory.RETAIL_SALES: "macro",
        EventCategory.ISM_MANUFACTURING: "macro",
        EventCategory.ISM_SERVICES: "macro",
        EventCategory.HOUSING: "macro",
        EventCategory.CONSUMER_CONFIDENCE: "macro",
        
        # Rates
        EventCategory.FOMC: "rates",
        EventCategory.RATE_DECISION: "rates",
        EventCategory.FED_SPEECH: "rates",
        EventCategory.YIELD_MOVE: "rates",
        EventCategory.YIELD_CURVE: "rates",
        EventCategory.DXY: "rates",
        EventCategory.VIX_SPIKE: "rates",
        EventCategory.LIQUIDITY_STRESS: "rates",
        
        # Corporate
        EventCategory.EARNINGS: "corporate",
        EventCategory.EARNINGS_BEAT: "corporate",
        EventCategory.EARNINGS_MISS: "corporate",
        EventCategory.GUIDANCE_RAISE: "corporate",
        EventCategory.GUIDANCE_LOWER: "corporate",
        EventCategory.REVENUE: "corporate",
        EventCategory.BUYBACK: "corporate",
        EventCategory.DIVIDEND: "corporate",
        EventCategory.M_AND_A: "corporate",
        EventCategory.IPO: "corporate",
        EventCategory.INSIDER_TRADE: "corporate",
        EventCategory.SEC_FILING: "corporate",
        
        # Politics
        EventCategory.TARIFF: "politics",
        EventCategory.SANCTION: "politics",
        EventCategory.REGULATION: "politics",
        EventCategory.TAX_POLICY: "politics",
        EventCategory.ANTITRUST: "politics",
        EventCategory.INFRASTRUCTURE: "politics",
        EventCategory.ELECTION: "politics",
        EventCategory.LEGISLATION: "politics",
        
        # Geopolitics
        EventCategory.CONFLICT: "geopolitics",
        EventCategory.WAR: "geopolitics",
        EventCategory.TERRORISM: "geopolitics",
        EventCategory.ENERGY_CRISIS: "geopolitics",
        EventCategory.OIL_SHOCK: "geopolitics",
        EventCategory.SHIPPING_DISRUPTION: "geopolitics",
        EventCategory.TRADE_WAR: "geopolitics",
        EventCategory.DIPLOMATIC: "geopolitics",
        EventCategory.NATURAL_DISASTER: "geopolitics",
        EventCategory.PANDEMIC: "geopolitics",
        
        # Individuals
        EventCategory.CEO_STATEMENT: "individuals",
        EventCategory.FED_CHAIR: "individuals",
        EventCategory.ELON_MUSK: "individuals",
        EventCategory.INFLUENCER: "individuals",
        EventCategory.INSIDER: "individuals",
        
        # Social
        EventCategory.TRENDING: "social",
        EventCategory.VIRAL: "social",
        EventCategory.WSB_MENTION: "social",
        EventCategory.STOCKTWITS: "social",
        EventCategory.TWITTER: "social",
        
        # Positioning
        EventCategory.PUT_CALL_EXTREME: "positioning",
        EventCategory.GAMMA_SQUEEZE: "positioning",
        EventCategory.SHORT_SQUEEZE: "positioning",
        EventCategory.OPTIONS_FLOW: "positioning",
        EventCategory.VOLUME_ANOMALY: "positioning",
        EventCategory.LIQUIDATION: "positioning",
    }
    
    # Keywords for auto-classification
    KEYWORDS = {
        EventCategory.CPI: ["cpi", "consumer price", "inflation"],
        EventCategory.PCE: ["pce", "personal consumption"],
        EventCategory.NFP: ["nfp", "non-farm", "payrolls", "jobs report"],
        EventCategory.UNEMPLOYMENT: ["unemployment", "jobless rate"],
        EventCategory.GDP: ["gdp", "gross domestic"],
        EventCategory.FOMC: ["fomc", "federal reserve", "fed meeting"],
        EventCategory.RATE_DECISION: ["rate hike", "rate cut", "interest rate"],
        EventCategory.EARNINGS: ["earnings", "quarterly results", "q1", "q2", "q3", "q4"],
        EventCategory.GUIDANCE_RAISE: ["guidance", "outlook", "forecast", "raised guidance"],
        EventCategory.GUIDANCE_LOWER: ["lowered guidance", "cut guidance"],
        EventCategory.TARIFF: ["tariff", "trade barrier", "import duty"],
        EventCategory.SANCTION: ["sanction", "embargo", "ofac"],
        EventCategory.WAR: ["war", "invasion", "military"],
        EventCategory.CONFLICT: ["conflict", "tension", "escalation"],
        EventCategory.ELON_MUSK: ["elon", "musk", "@elonmusk"],
        EventCategory.WSB_MENTION: ["wsb", "wallstreetbets", "apes"],
        EventCategory.SHORT_SQUEEZE: ["short squeeze", "squeeze", "shorts"],
    }
    
    # Sector mapping for tickers
    SECTOR_MAP = {
        # Technology
        "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
        "GOOG": "technology", "META": "technology", "NVDA": "technology",
        "AMD": "technology", "INTC": "technology", "TSLA": "technology",
        "CRM": "technology", "ORCL": "technology", "ADBE": "technology",
        
        # Financials
        "JPM": "financials", "BAC": "financials", "WFC": "financials",
        "GS": "financials", "MS": "financials", "C": "financials",
        "BRK.B": "financials", "V": "financials", "MA": "financials",
        
        # Healthcare
        "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
        "ABBV": "healthcare", "MRK": "healthcare", "LLY": "healthcare",
        
        # Energy
        "XOM": "energy", "CVX": "energy", "COP": "energy",
        "SLB": "energy", "EOG": "energy", "OXY": "energy",
        
        # Consumer
        "AMZN": "consumer", "WMT": "consumer", "HD": "consumer",
        "MCD": "consumer", "NKE": "consumer", "SBUX": "consumer",
        "KO": "consumer", "PEP": "consumer", "PG": "consumer",
        
        # Industrial
        "BA": "industrials", "CAT": "industrials", "GE": "industrials",
        "HON": "industrials", "UPS": "industrials", "RTX": "industrials",
        
        # Crypto-related
        "COIN": "crypto", "MARA": "crypto", "RIOT": "crypto",
        "MSTR": "crypto", "WOLF": "crypto",  # Focus stock
        
        # ETFs for sector reference
        "SPY": "market", "QQQ": "technology", "XLF": "financials",
        "XLE": "energy", "XLV": "healthcare", "XLI": "industrials",
        "XLK": "technology", "XLP": "consumer_staples", "XLY": "consumer_discretionary",
    }
    
    @classmethod
    def classify(cls, text: str, tickers: List[str] = None) -> EventCategory:
        """
        Classify an event based on text content.
        
        Args:
            text: Event headline and/or description
            tickers: Optional list of affected tickers
        
        Returns:
            EventCategory
        """
        text_lower = text.lower()
        
        # Check keywords
        for category, keywords in cls.KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        # Default to NEWS
        return EventCategory.NEWS
    
    @classmethod
    def get_layer(cls, category: EventCategory) -> str:
        """Get the intelligence layer for a category"""
        return cls.LAYER_MAP.get(category, "other")
    
    @classmethod
    def get_all_categories_for_layer(cls, layer: str) -> List[EventCategory]:
        """Get all categories belonging to a layer"""
        return [cat for cat, lay in cls.LAYER_MAP.items() if lay == layer]
    
    @classmethod
    def get_impact_weight(cls, category: EventCategory) -> float:
        """
        Get base impact weight for a category.
        Higher weight = more market-moving potential.
        """
        weights = {
            # High impact macro (1.0)
            EventCategory.FOMC: 1.0,
            EventCategory.CPI: 1.0,
            EventCategory.NFP: 1.0,
            EventCategory.RATE_DECISION: 1.0,
            
            # High impact events (0.9)
            EventCategory.WAR: 0.9,
            EventCategory.GDP: 0.9,
            EventCategory.CONFLICT: 0.9,
            EventCategory.FED_CHAIR: 0.9,
            
            # Medium-high (0.8)
            EventCategory.PCE: 0.8,
            EventCategory.EARNINGS: 0.8,
            EventCategory.TARIFF: 0.8,
            EventCategory.SANCTION: 0.8,
            
            # Medium (0.7)
            EventCategory.UNEMPLOYMENT: 0.7,
            EventCategory.GUIDANCE_LOWER: 0.7,
            EventCategory.GUIDANCE_RAISE: 0.7,
            EventCategory.OIL_SHOCK: 0.7,
            
            # Medium-low (0.6)
            EventCategory.YIELD_MOVE: 0.6,
            EventCategory.VIX_SPIKE: 0.6,
            EventCategory.SHORT_SQUEEZE: 0.6,
            EventCategory.GAMMA_SQUEEZE: 0.6,
            
            # Lower (0.5)
            EventCategory.BUYBACK: 0.5,
            EventCategory.DIVIDEND: 0.5,
            EventCategory.ELON_MUSK: 0.5,
            
            # Social (0.4)
            EventCategory.WSB_MENTION: 0.4,
            EventCategory.TRENDING: 0.4,
            EventCategory.STOCKTWITS: 0.4,
            
            # Generic (0.3)
            EventCategory.NEWS: 0.3,
            EventCategory.OTHER: 0.2,
        }
        
        return weights.get(category, 0.3)


def get_ticker_sector(ticker: str) -> Optional[str]:
    """Get sector for a ticker"""
    return EventTaxonomy.SECTOR_MAP.get(ticker.upper())


def get_sector_tickers(sector: str) -> List[str]:
    """Get all tickers in a sector"""
    return [t for t, s in EventTaxonomy.SECTOR_MAP.items() if s == sector]


def classify_event(text: str, tickers: List[str] = None) -> Dict:
    """
    Classify an event and return full classification info.
    
    Returns:
        {
            "category": EventCategory,
            "layer": str,
            "impact_weight": float,
            "sectors": List[str]
        }
    """
    category = EventTaxonomy.classify(text, tickers)
    layer = EventTaxonomy.get_layer(category)
    weight = EventTaxonomy.get_impact_weight(category)
    
    sectors = []
    if tickers:
        for t in tickers:
            sector = get_ticker_sector(t)
            if sector and sector not in sectors:
                sectors.append(sector)
    
    return {
        "category": category,
        "layer": layer,
        "impact_weight": weight,
        "sectors": sectors,
    }
