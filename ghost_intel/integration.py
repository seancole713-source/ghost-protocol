"""
Ghost Intel Integration - Wires institutional intelligence into predictions.

This module provides the bridge between Ghost Intel feeds and the prediction engine.
It transforms raw intel data into actionable signals that affect:
1. Direction bias (bullish/bearish tilt based on macro + positioning)
2. Confidence adjustments (+/- based on event impact and market fragility)
3. Entry timing gates (block trades during high-impact events)
4. Trump Tariff Playbook (bond market triggers, timing patterns)
5. 2025 Winners Playbook (sector leadership, momentum stocks)
6. Trading Discipline Rules (risk management, technical signals)

Integration points in wolf_app.py:
- After feature extraction (~line 8020)
- Before ensemble prediction (~line 8170)
- As part of market gates (~line 8310)

Target impact: 10-20% accuracy improvement from institutional timing

TARIFF PLAYBOOK (Kobeissi Letter pattern):
- 10Y > 4.50%: Trump warning zone, expect pause
- 10Y > 4.60%: Pause imminent, BUY window approaching
- Mon-Tue after tariff weekend: Block panic selling
- Wed-Thu: Dip buying window opens

2025 WINNERS PLAYBOOK (Historical data):
- Sector leaders: Tech (+24%), Comms (+33.6%)
- Precious metals: Gold +64%, Silver +146% (inflation hedge)
- Storage/Memory boom: SNDK +559%, WDC +261%, MU +178%
- Semis strength: LRCX +138%, AMD +77%, NVDA +39%, AVGO +50%
- Gold miners: NEM +138% (follows gold)
- Tariff pattern: -19% H1 → recovery H2 (confirms Kobeissi playbook)

TRADING DISCIPLINE RULES (Professional trader principles):
- Risk Management: 1-2% max risk per trade, enforce stop-losses
- Reward/Risk: Require 2:1 ratio minimum (aim for $200 profit on $100 risk)
- Volatility: Need price movement to profit (block flat/dead stocks)
- Liquidity: High volume = clean entries/exits (penalize low volume)
- RSI Signals: Overbought (>70) = fade risk, Oversold (<30) = bounce potential
- VWAP: Price above VWAP = bullish, below = bearish
- Patience: A+ setups only - better to miss than force bad trades
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass

LOGGER = logging.getLogger("ghost.intel.integration")

# =============================================================================
# 2025 WINNERS DATA (Learned from historical performance)
# =============================================================================

# Sector performance 2025 - used for sector bias
SECTOR_PERFORMANCE_2025 = {
    "technology": 24.0,
    "communication_services": 33.6,
    "consumer_discretionary": 12.0,  # estimated
    "financials": 8.0,  # estimated
    "healthcare": 5.0,  # estimated
    "industrials": 10.0,  # estimated
    "materials": 15.0,  # estimated (gold miners helped)
    "energy": -5.0,  # estimated (oil weakness)
    "utilities": 3.0,  # estimated
    "real_estate": 2.0,  # estimated
    "consumer_staples": 4.0,  # estimated
}

# Top 2025 winners - momentum continuation bias
WINNERS_2025: Set[str] = {
    # Storage/Memory boom
    "SNDK", "WDC", "STX", "MU",
    # Semiconductors
    "LRCX", "AMD", "NVDA", "AVGO", "INTC",
    # Fintech/Tech
    "HOOD", "PLTR", "APP", "APH",
    # Gold miners (follows precious metals)
    "NEM", "GDX", "GOLD", "AEM", "KGC", "BTG", "NGD", "B",
    # Big tech (still performing)
    "GOOGL", "GOOG", "GE", "RTX",
    # Media recovery
    "WBD",
}

# =============================================================================
# YAHOO FINANCE MOST ACTIVE - HOT THEMES (Jan 27, 2026 Update)
# =============================================================================

# STORAGE/MEMORY MONSTER RUN - The hottest sector of 2025-2026
STORAGE_MEMORY_STOCKS: Set[str] = {
    "SNDK",  # Sandisk +1,216% (52wk!) - Cramer says "too much for me"
    "WDC",   # Western Digital +397%
    "MU",    # Micron +353% - $24B Singapore plant announced
    "STX",   # Seagate +245%
    "LITE",  # Lumentum +336% (optical)
    "LRCX",  # Lam Research +193%
}

# CLOUD INFRASTRUCTURE SURGE - Today's big gainers
CLOUD_INFRA_STOCKS: Set[str] = {
    "ZM",    # Zoom +11% today (huge volume spike)
    "DOCN",  # DigitalOcean +10%
    "NET",   # Cloudflare +9%
    "GDS",   # GDS Holdings +9% (China data centers)
    "KC",    # Kingsoft Cloud +8%
    "ANET",  # Arista Networks +5%
    "CLS",   # Celestica +221%
}

# SPACE STOCKS - The final frontier
SPACE_STOCKS: Set[str] = {
    "ASTS",  # AST SpaceMobile +464%
    "PL",    # Planet Labs +393%
    "RKLB",  # Rocket Lab +207%
    "LUNR",  # Intuitive Machines
    "FLY",   # Firefly Aerospace (volatile)
    "KRMN",  # Karman Holdings +260%
    "VSAT",  # Viasat +323%
}

# GOLD MINERS - Riding the precious metals wave
GOLD_MINERS: Set[str] = {
    "NEM",   # Newmont +201%
    "B",     # Barrick +220%
    "GFI",   # Gold Fields +225%
    "NG",    # NovaGold +224%
    "KGC",   # Kinross +242%
    "AU",    # AngloGold +277%
    "IAG",   # IAMGOLD +249%
    "EGO",   # Eldorado +205%
    "BVN",   # Buenaventura +199%
    "CGAU",  # Centerra +204%
}

# FALLEN ANGELS - Beaten down names with bounce potential
FALLEN_ANGELS: Set[str] = {
    "TTD",   # Trade Desk -70% (was $126 now $34)
    "LULU",  # Lululemon -53%
    "DUOL",  # Duolingo -56%
    "HUBS",  # HubSpot -59%
    "FISV",  # Fiserv -68%
    "SRPT",  # Sarepta -82% (biotech crash)
    "TEAM",  # Atlassian -52%
    "NOW",   # ServiceNow -43%
    "DECK",  # Deckers -54%
    "MSTR",  # Strategy -51% (Bitcoin exposure)
    "LCID",  # Lucid -60%
}

# GOVERNMENT CONTRACT RISK - Watch for contract cancellation news
GOVT_CONTRACT_RISK: Set[str] = {
    "BAH",   # Booz Allen -8% (Treasury cancelled $21M contracts)
    "LMT",   # Lockheed (DOGE exposure)
    "RTX",   # RTX (defense)
    "NOC",   # Northrop
    "GD",    # General Dynamics
}

# Silver miners - outperforming even gold (Silver +146% vs Gold +64% in 2025)
SILVER_MINERS: Set[str] = {
    "AG",    # First Majestic +376%
    "HL",    # Hecla Mining +490%
    "CDE",   # Coeur Mining +316%
    "EXK",   # Endeavour Silver +291%
    "PSLV",  # Sprott Physical Silver +226%
    "USAS",  # Americas Gold & Silver +629%
    "VZLA",  # Vizsla Silver +239%
    "SVM",   # Silvercorp +328%
    "HYMC",  # Hycroft Mining +2,256% (!)
    "PAAS",  # Pan American Silver +194%
}

# Rare Earth & Critical Minerals - GREENLAND PLAY (ties to tariff playbook!)
RARE_EARTH_MINERALS: Set[str] = {
    "USAR",  # USA Rare Earth +97%
    "MP",    # MP Materials +234%
    "TMC",   # TMC Metals Company +509%
    "CRML",  # Critical Metals +155%
    "LAC",   # Lithium Americas
    "ALB",   # Albemarle (lithium)
}

# Uranium plays - nuclear renaissance
URANIUM_STOCKS: Set[str] = {
    "DNN",   # Denison Mines +119%
    "UUUU",  # Energy Fuels +383%
    "CCJ",   # Cameco
    "UEC",   # Uranium Energy Corp
    "NXE",   # NexGen Energy
    "LEU",   # Centrus Energy
}

# Bitcoin miners - crypto proxy plays
BITCOIN_MINERS: Set[str] = {
    "MARA",  # MARA Holdings
    "RIOT",  # Riot Platforms +57%
    "CLSK",  # CleanSpark +36%
    "WULF",  # TeraWulf +204%
    "CIFR",  # Cipher Mining +291%
    "BMNR",  # Bitmine +271%
    "IREN",  # IREN +462%
    "APLD",  # Applied Digital +478%
}

# AI Infrastructure & Quantum Computing
AI_QUANTUM_STOCKS: Set[str] = {
    "CRWV",  # CoreWeave +132%
    "IONQ",  # IonQ +21%
    "QBTS",  # D-Wave Quantum +346%
    "RGTI",  # Rigetti +79%
    "SOUN",  # SoundHound AI
    "BBAI",  # BigBear.ai +57%
    "PATH",  # UiPath
    "SMR",   # NuScale Power (nuclear AI power)
}

# =============================================================================
# CRYPTO INTELLIGENCE (Jan 27, 2026)
# =============================================================================

# Privacy coins - Crushing it (+689% ZEC, +106% XMR)
PRIVACY_COINS: Set[str] = {
    "ZEC-USD",   # Zcash +689% 52wk (!)
    "XMR-USD",   # Monero +106% 52wk
    "DASH-USD",  # Dash +90% 52wk
    "BDX-USD",   # Beldex
}

# Gaming/Metaverse tokens - AXS +34% today!
GAMING_CRYPTO: Set[str] = {
    "AXS-USD",   # Axie Infinity +34% TODAY
    "SAND-USD",  # The Sandbox
    "MANA-USD",  # Decentraland
    "GALA-USD",  # Gala +307% 52wk
    "IMX10603-USD",  # Immutable
}

# DeFi momentum tokens
DEFI_MOMENTUM: Set[str] = {
    "HYPE32196-USD",  # Hyperliquid +22% today
    "AAVE-USD",       # Aave
    "UNI7083-USD",    # Uniswap
    "CRV-USD",        # Curve
    "MORPHO34104-USD", # Morpho
    "SYRUP-USD",      # Maple Finance +126%
}

# Meme coins - HIGH RISK
MEME_COINS: Set[str] = {
    "DOGE-USD",   # Dogecoin -62% from highs
    "SHIB-USD",   # Shiba Inu -56%
    "PEPE24478-USD",  # Pepe
    "BONK-USD",   # Bonk -65%
    "WIF-USD",    # dogwifhat -70%
    "TRUMP35336-USD",  # TRUMP -83% from highs (!)
    "FARTCOIN-USD",    # Fartcoin
    "TURBO-USD",       # Turbo - MEME COIN! (Added Jan 27, 2026)
}

# Layer 1 majors
LAYER1_MAJORS: Set[str] = {
    "BTC-USD",   # Bitcoin $88K (-14.5% from $126K high)
    "ETH-USD",   # Ethereum $2,934 (-8.5%)
    "SOL-USD",   # Solana $124 (-48%)
    "AVAX-USD",  # Avalanche $11.75 (-64%)
    "ADA-USD",   # Cardano $0.35 (-63%)
    "DOT-USD",   # Polkadot $1.87 (-68%)
    "NEAR-USD",  # Near $1.47 (-67%)
    "SUI20947-USD",  # Sui $1.44 (-62%)
}

# AI/GPU Infrastructure Crypto - HIGH GROWTH POTENTIAL (Added Jan 27, 2026)
# These tokens power AI infrastructure, GPU rendering, and compute markets
AI_GPU_CRYPTO: Set[str] = {
    "RNDR-USD",    # Render Network - GPU rendering for AI/3D ($7.50, strong performer)
    "FET-USD",     # Fetch.ai - AI agents and autonomous economy
    "AGIX-USD",    # SingularityNET - Decentralized AI marketplace
    "TAO22974-USD", # Bittensor - Decentralized machine learning
    "AKT-USD",     # Akash Network - Decentralized GPU compute
    "OCEAN-USD",   # Ocean Protocol - Data marketplace for AI
    "GLM-USD",     # Golem - Distributed computing power
}

# Crypto stocks (equities that move with crypto)
CRYPTO_STOCKS: Set[str] = {
    "COIN",  # Coinbase
    "MSTR",  # MicroStrategy/Strategy
    "HOOD",  # Robinhood
    "SQ",    # Block (Square)
    "PYPL",  # PayPal
}

# =============================================================================
# CURRENCY/FX INTELLIGENCE (Jan 27, 2026)
# =============================================================================

# Current FX levels (for reference)
# USD/JPY: 154.47 (yen strengthening, intervention talk)
# EUR/USD: 1.1875 (dollar weak)
# USD/MXN: 17.32 (peso strong)
# USD/INR: 91.79 (rupee at record low)
# DXY: 4-month low (debasement trade!)

FX_THRESHOLDS = {
    # Dollar Index (DXY proxy via EUR/USD)
    "eurusd_dollar_weak": 1.15,     # Above = weak dollar
    "eurusd_dollar_very_weak": 1.20, # Above = very weak dollar (gold bullish!)
    "eurusd_dollar_strong": 1.05,   # Below = strong dollar
    
    # Yen (USD/JPY) - intervention risk
    "usdjpy_intervention_risk": 160,  # Above = BOJ may intervene
    "usdjpy_yen_strong": 145,         # Below = yen strengthening (risk-off)
    "usdjpy_yen_very_strong": 140,    # Below = major risk-off
    
    # Mexican Peso (USD/MXN) - EM strength indicator
    "usdmxn_peso_strong": 18,         # Below = peso strength
    "usdmxn_peso_weak": 20,           # Above = peso weakness/risk-off
    
    # Indian Rupee (USD/INR)
    "usdinr_rupee_stress": 90,        # Above = rupee stress (EM risk)
    "usdinr_rupee_crisis": 95,        # Above = crisis territory
}

# Stocks affected by FX moves
DOLLAR_SENSITIVE_STOCKS: Set[str] = {
    # Weak dollar beneficiaries (multinational earnings boost)
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",  # Big tech (overseas revenue)
    "KO", "PEP", "PG", "JNJ", "MCD",          # Consumer multinationals
    "CAT", "DE", "BA",                         # Industrials
}

YEN_SENSITIVE_STOCKS: Set[str] = {
    # Japanese ADRs and yen-sensitive
    "TM",    # Toyota
    "HMC",   # Honda
    "SONY",  # Sony
    "NTT",   # NTT
    "MUFG",  # Mitsubishi UFJ
    "SMFG",  # Sumitomo Mitsui
}

EM_SENSITIVE_STOCKS: Set[str] = {
    # Emerging market exposure
    "EEM",   # EM ETF
    "VWO",   # Vanguard EM
    "IEMG",  # iShares EM
    "IBN",   # ICICI Bank (India)
    "HDB",   # HDFC Bank (India)
    "BABA",  # Alibaba
    "JD",    # JD.com
    "PDD",   # PDD Holdings
}

# =============================================================================
# COMMODITIES FUTURES THRESHOLDS (Jan 27, 2026)
# =============================================================================

COMMODITIES_THRESHOLDS = {
    # Gold - near all-time highs at $5,057
    "gold_bullish": 5000,       # Above = strong gold, boost miners
    "gold_extreme_bullish": 5200,  # New ATH territory
    "gold_bearish": 4500,       # Below = gold weakness
    
    # Silver - volatile, $108 (down 6.24% today!)
    "silver_bullish": 100,      # Above = silver momentum
    "silver_extreme": 120,      # Parabolic zone
    "silver_bearish": 80,       # Below = weakness
    
    # Oil - $60.29 (relatively low)
    "oil_bullish": 70,          # Above = energy inflation
    "oil_bearish": 55,          # Below = deflationary
    "oil_crisis": 100,          # Above = energy crisis mode
    
    # Natural Gas - $3.69 (down 5.21%)
    "natgas_bullish": 4.0,      # Above = utility inflation
    "natgas_spike": 6.0,        # Crisis territory
    
    # Copper - $5.88 (economic indicator)
    "copper_bullish": 5.5,      # Above = economic strength
    "copper_bearish": 4.5,      # Below = slowdown signal
}

# Sector mapping for stocks
STOCK_SECTORS = {
    # Technology / Storage / Memory
    "SNDK": "technology", "WDC": "technology", "STX": "technology", "MU": "technology",
    "LRCX": "technology", "AMD": "technology", "NVDA": "technology", "AVGO": "technology",
    "INTC": "technology", "PLTR": "technology", "APP": "technology", "APH": "technology",
    "AAPL": "technology", "MSFT": "technology", "LITE": "technology", "TSEM": "technology",
    # Cloud Infrastructure
    "ZM": "technology", "DOCN": "technology", "NET": "technology", "GDS": "technology",
    "KC": "technology", "ANET": "technology", "CLS": "technology",
    # Communication Services
    "GOOGL": "communication_services", "GOOG": "communication_services",
    "META": "communication_services", "WBD": "communication_services",
    "NFLX": "communication_services", "DIS": "communication_services",
    # Financials
    "HOOD": "financials", "JPM": "financials", "BAC": "financials", "GS": "financials",
    # Materials - Gold miners
    "NEM": "materials", "GDX": "materials", "GOLD": "materials", "AEM": "materials", "KGC": "materials",
    "BTG": "materials", "NGD": "materials", "B": "materials", "GFI": "materials", "NG": "materials",
    "AU": "materials", "IAG": "materials", "EGO": "materials", "BVN": "materials", "CGAU": "materials",
    # Materials - Silver miners
    "AG": "materials", "HL": "materials", "CDE": "materials", "EXK": "materials",
    "PSLV": "materials", "USAS": "materials", "VZLA": "materials", "SVM": "materials",
    "HYMC": "materials", "PAAS": "materials",
    # Materials - Rare Earth
    "USAR": "materials", "MP": "materials", "TMC": "materials", "CRML": "materials",
    # Energy - Uranium
    "DNN": "energy", "UUUU": "energy", "CCJ": "energy", "UEC": "energy", "LEU": "energy",
    # Industrials / Defense
    "GE": "industrials", "RTX": "industrials", "BA": "industrials", "CAT": "industrials",
    "BAH": "industrials", "LMT": "industrials", "NOC": "industrials", "GD": "industrials",
    # Space
    "ASTS": "technology", "PL": "technology", "RKLB": "technology", "LUNR": "technology",
    "FLY": "industrials", "KRMN": "technology", "VSAT": "technology",
    # Bitcoin miners / AI infra
    "MARA": "technology", "RIOT": "technology", "CLSK": "technology", "WULF": "technology",
    "CIFR": "technology", "BMNR": "technology", "IREN": "technology", "APLD": "technology",
    "CRWV": "technology", "IONQ": "technology", "QBTS": "technology", "RGTI": "technology",
    # Fallen Angels
    "TTD": "technology", "LULU": "consumer_discretionary", "DUOL": "technology",
    "HUBS": "technology", "FISV": "technology", "SRPT": "healthcare", "TEAM": "technology",
    "NOW": "technology", "DECK": "consumer_discretionary", "MSTR": "technology", "LCID": "consumer_discretionary",
}

# Precious metals tickers (for correlation)
PRECIOUS_METALS = {"GLD", "SLV", "IAU", "GOLD", "NEM", "GDX", "XAUUSD", "XAGUSD"}

# =============================================================================
# TRADING DISCIPLINE RULES (Professional trader principles)
# =============================================================================

# Risk Management Thresholds
TRADING_DISCIPLINE = {
    # Reward/Risk ratio requirements
    "min_reward_risk_ratio": 1.5,      # Minimum acceptable (prefer 2:1)
    "ideal_reward_risk_ratio": 2.0,    # Target ratio for bonus confidence
    "excellent_reward_risk_ratio": 3.0, # Excellent setup bonus
    
    # RSI (Relative Strength Index) thresholds
    "rsi_overbought": 70,              # Above = potential reversal down
    "rsi_oversold": 30,                # Below = potential bounce up
    "rsi_extreme_overbought": 80,      # Very extended, high fade risk
    "rsi_extreme_oversold": 20,        # Capitulation zone, bounce likely
    
    # Volume/Liquidity requirements
    "min_relative_volume": 0.5,        # Below = dead stock, avoid
    "high_relative_volume": 2.0,       # Above = institutional interest
    "very_high_volume": 5.0,           # Unusual activity, pay attention
    
    # Volatility requirements (ATR-based)
    "min_volatility_pct": 1.0,         # Need at least 1% daily range
    "ideal_volatility_pct": 3.0,       # Sweet spot for day trading
    "max_volatility_pct": 15.0,        # Too wild, reduce size
    
    # VWAP positioning
    "vwap_buffer_pct": 0.5,            # Within 0.5% of VWAP = neutral
}

# Cache for intel data (avoid hammering APIs)
_INTEL_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL = 60  # 1 minute cache for live feeds


@dataclass
class IntelSignal:
    """Intel-derived signal for prediction adjustment."""
    direction_bias: str  # "bullish", "bearish", "neutral"
    confidence_adjustment: float  # -0.15 to +0.15
    should_trade: bool  # False = block this prediction
    block_reason: Optional[str]  # Why blocked (if any)
    signal_sources: list  # Which intel sources contributed
    market_context: Dict[str, Any]  # VIX, positioning, etc.
    event_count: int  # Number of relevant events
    max_event_score: float  # Highest impact event score
    tariff_context: Optional[Dict[str, Any]] = None  # Tariff playbook data
    winners_context: Optional[Dict[str, Any]] = None  # 2025 winners data
    discipline_context: Optional[Dict[str, Any]] = None  # Trading discipline data


def _get_cached(key: str, ttl: float = _CACHE_TTL) -> Optional[Any]:
    """Get cached intel data if still fresh."""
    if key in _INTEL_CACHE:
        timestamp, data = _INTEL_CACHE[key]
        if time.time() - timestamp < ttl:
            return data
    return None


def _set_cached(key: str, data: Any) -> None:
    """Cache intel data."""
    _INTEL_CACHE[key] = (time.time(), data)


async def fetch_intel_context(symbol: str = None) -> Dict[str, Any]:
    """
    Fetch current intel context from live feeds.
    
    Returns:
        {
            "vix": float,
            "vix_regime": str,  # "calm", "elevated", "fear", "panic"
            "put_call_ratio": float,
            "positioning": str,  # "bullish", "bearish", "neutral"
            "fragility_score": float,  # 0-100
            "active_events": list,
            "macro_regime": str,  # "expansion", "contraction", "neutral"
            "symbol_impact": dict  # if symbol provided
        }
    """
    cache_key = f"intel_context_{symbol or 'market'}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    context = {
        "vix": 20.0,
        "vix_regime": "neutral",
        "put_call_ratio": 1.0,
        "positioning": "neutral",
        "fragility_score": 50.0,
        "active_events": [],
        "macro_regime": "neutral",
        "symbol_impact": {},
        "timestamp": time.time(),
    }
    
    try:
        # Fetch rates (VIX, yields)
        from ghost_intel.sources import fetch_live_rates
        rates = await fetch_live_rates()
        
        if rates:
            vix = rates.get("vix", {}).get("price", 20.0)
            context["vix"] = vix
            
            # Classify VIX regime
            if vix < 15:
                context["vix_regime"] = "calm"
            elif vix < 20:
                context["vix_regime"] = "neutral"
            elif vix < 25:
                context["vix_regime"] = "elevated"
            elif vix < 30:
                context["vix_regime"] = "fear"
            else:
                context["vix_regime"] = "panic"
            
            # 2s10s spread for recession signal
            spread = rates.get("spread_2s10s", {})
            if spread.get("inverted"):
                context["macro_regime"] = "recession_warning"
            
            # VIX term structure
            vix_term = rates.get("vix_term_structure", {})
            if vix_term.get("backwardation"):
                context["vix_regime"] = "panic"  # Override - backwardation = fear
                
    except Exception as e:
        LOGGER.warning(f"Failed to fetch rates: {e}")
    
    try:
        # Fetch positioning
        from ghost_intel.positioning import MarketPositioningAnalyzer
        
        analyzer = MarketPositioningAnalyzer()
        positioning = await analyzer.get_positioning_snapshot()
        
        if positioning:
            context["put_call_ratio"] = positioning.get("put_call", {}).get("ratio", 1.0)
            context["fragility_score"] = positioning.get("fragility", {}).get("score", 50.0)
            
            # Determine positioning bias
            pcr = context["put_call_ratio"]
            if pcr < 0.7:
                context["positioning"] = "bullish"  # Low put/call = complacent bulls
            elif pcr > 1.3:
                context["positioning"] = "bearish"  # High put/call = hedging
            else:
                context["positioning"] = "neutral"
                
    except Exception as e:
        LOGGER.warning(f"Failed to fetch positioning: {e}")
    
    try:
        # Fetch active events
        from ghost_intel.routes import _fetch_and_process_events
        
        events = await _fetch_and_process_events(limit=10, min_score=20)
        context["active_events"] = events.get("events", [])
        
    except Exception as e:
        LOGGER.warning(f"Failed to fetch events: {e}")
    
    # If symbol provided, get symbol-specific impact
    if symbol:
        try:
            from ghost_intel.routes import _get_symbol_impact
            
            impact = await _get_symbol_impact(symbol)
            context["symbol_impact"] = impact
            
        except Exception as e:
            LOGGER.debug(f"Failed to fetch symbol impact for {symbol}: {e}")
    
    # =========================================================================
    # TARIFF PLAYBOOK DATA (Kobeissi Letter Pattern)
    # =========================================================================
    try:
        # Get 10Y Treasury yield (Trump's warning signal)
        treasury_10y = rates.get("us_10y", {}).get("price") if rates else None
        if treasury_10y is None:
            treasury_10y = rates.get("us_10y_fred", {}).get("price") if rates else None
        context["treasury_10y"] = treasury_10y or 4.25  # Default neutral
        
        # Classify Treasury regime for tariff playbook
        t10y = context["treasury_10y"]
        if t10y > 4.60:
            context["treasury_regime"] = "trump_pause_imminent"  # 10Y > 4.60% = Trump backs off
        elif t10y > 4.50:
            context["treasury_regime"] = "trump_warning"  # 10Y > 4.50% = warning zone
        elif t10y > 4.30:
            context["treasury_regime"] = "elevated"
        else:
            context["treasury_regime"] = "normal"
        
        # Day-of-week timing for tariff playbook
        now = datetime.now(timezone.utc)
        context["day_of_week"] = now.weekday()  # 0=Mon, 6=Sun
        context["tariff_timing_window"] = _get_tariff_timing_window(now.weekday())
        
        # Check for tariff-related events in news
        tariff_events = [
            e for e in context.get("active_events", [])
            if _is_tariff_event(e)
        ]
        context["active_tariff_events"] = len(tariff_events)
        context["tariff_active"] = len(tariff_events) > 0
        
    except Exception as e:
        LOGGER.debug(f"Tariff playbook data fetch failed: {e}")
        context["treasury_10y"] = 4.25
        context["treasury_regime"] = "normal"
        context["tariff_timing_window"] = "neutral"
        context["tariff_active"] = False
    
    # =========================================================================
    # RULE 14: FULL STOCK HISTORY (IPO to now)
    # Provides: fundamentals, 52-week context, technicals, earnings
    # =========================================================================
    if symbol:
        try:
            from ghost_intel.stock_history import get_stock_context
            
            history_context = get_stock_context(symbol)
            if history_context:
                # Merge full history into context
                context["stock_history"] = history_context
                
                # Surface key metrics at top level for easy access
                context["pe_ratio"] = history_context.get("pe_ratio")
                context["market_cap"] = history_context.get("market_cap")
                context["sector"] = history_context.get("sector")
                context["industry"] = history_context.get("industry")
                context["pct_from_52w_high"] = history_context.get("pct_from_52w_high")
                context["pct_from_52w_low"] = history_context.get("pct_from_52w_low")
                context["all_time_high"] = history_context.get("all_time_high")
                context["years_trading"] = history_context.get("years_trading")
                context["days_to_earnings"] = history_context.get("days_to_earnings")
                
                # Flags for quick decisions
                context["is_near_52w_high"] = history_context.get("is_near_52w_high", False)
                context["is_near_52w_low"] = history_context.get("is_near_52w_low", False)
                context["is_oversold"] = history_context.get("is_oversold", False)
                context["is_overbought"] = history_context.get("is_overbought", False)
                context["is_high_volume"] = history_context.get("is_high_volume", False)
                
                # Technical context
                context["trend"] = history_context.get("trend")
                context["support"] = history_context.get("support")
                context["resistance"] = history_context.get("resistance")
                context["above_sma_200"] = history_context.get("above_sma_200")
                
                LOGGER.info(f"📊 Stock history loaded for {symbol}: "
                           f"52W Range: {context['pct_from_52w_low']:.1f}% to {context['pct_from_52w_high']:.1f}%")
                
        except Exception as e:
            LOGGER.debug(f"Stock history fetch failed for {symbol}: {e}")
    
    _set_cached(cache_key, context)
    return context


def _get_tariff_timing_window(weekday: int) -> str:
    """
    Determine trading window based on Kobeissi Letter tariff playbook timing.
    
    The pattern after tariff weekend announcements:
    - Mon-Tue: Panic selling (AVOID selling into panic)
    - Wed: Dip buyers emerge (START accumulating)
    - Thu-Fri: Relief rally builds (CONTINUE accumulating)
    - Weekend: Watch for new announcements
    """
    if weekday in [0, 1]:  # Monday, Tuesday
        return "panic_selling"  # Don't sell into panic
    elif weekday == 2:  # Wednesday
        return "dip_buying"  # Smart money starts buying
    elif weekday in [3, 4]:  # Thursday, Friday
        return "accumulation"  # Continue building positions
    else:  # Saturday, Sunday
        return "watch"  # Monitor for announcements


def _is_tariff_event(event: Dict[str, Any]) -> bool:
    """Check if an event is tariff-related."""
    tariff_keywords = [
        "tariff", "tariffs", "trade war", "import tax", "trade deal",
        "greenland", "denmark", "eu tariff", "china tariff", "trump tariff",
        "trade negotiation", "trade agreement", "customs duty"
    ]
    
    headline = event.get("event", {}).get("headline", "").lower()
    summary = event.get("event", {}).get("summary", "").lower()
    
    text = f"{headline} {summary}"
    return any(kw in text for kw in tariff_keywords)


def calculate_intel_signal(
    symbol: str,
    base_direction: str,
    base_confidence: float,
    intel_context: Dict[str, Any]
) -> IntelSignal:
    """
    Calculate Intel-derived signal for prediction adjustment.
    
    This is the CORE LOGIC that translates intel into trading signals.
    
    Rules:
    1. VIX Regime Gates:
       - VIX > 30 (panic): Block all BUY signals
       - VIX > 25 (fear): -10% confidence on BUY
       - VIX < 15 (calm): +5% confidence (low vol = trends persist)
       
    2. Positioning Signals:
       - Put/Call < 0.7 (complacent): Contrarian bearish bias
       - Put/Call > 1.3 (hedging): Contrarian bullish bias
       
    3. Event Impact:
       - High impact events (score > 70): Block trading
       - Medium impact (30-70): Adjust confidence by event direction
       
    4. Macro Regime:
       - Recession warning (inverted yield curve): -10% confidence, bearish bias
       
    5. Trump Tariff Playbook (Kobeissi Letter Pattern):
       - 10Y > 4.50%: Warning zone, expect pause (bullish signal)
       - 10Y > 4.60%: Pause imminent, BUY window (+10% confidence)
       - Mon-Tue during tariff event: Block SELL signals (panic trap)
       - Wed-Thu during tariff event: +5% confidence (dip buying window)
    """
    # NULL SAFETY: Ensure base_direction is never None
    if base_direction is None:
        base_direction = "FLAT"
    base_direction = str(base_direction).upper()  # Normalize to uppercase
    
    signals_used = []
    confidence_adj = 0.0
    direction_bias = "neutral"
    should_trade = True
    block_reason = None
    
    vix = intel_context.get("vix", 20.0)
    vix_regime = intel_context.get("vix_regime", "neutral")
    positioning = intel_context.get("positioning", "neutral")
    pcr = intel_context.get("put_call_ratio", 1.0)
    fragility = intel_context.get("fragility_score", 50.0)
    macro_regime = intel_context.get("macro_regime", "neutral")
    active_events = intel_context.get("active_events", [])
    symbol_impact = intel_context.get("symbol_impact", {})
    
    # =========================================================================
    # RULE 1: VIX REGIME GATES
    # =========================================================================
    if vix_regime == "panic" and base_direction == "UP":
        # Block all BUY signals during panic
        should_trade = False
        block_reason = f"VIX panic ({vix:.1f}) - no BUY signals"
        signals_used.append("VIX_PANIC_BLOCK")
        
    elif vix_regime == "fear" and base_direction == "UP":
        # Reduce confidence on BUY during fear
        confidence_adj -= 0.10
        signals_used.append(f"VIX_FEAR_{vix:.0f}")
        
    elif vix_regime == "elevated":
        # Slight caution
        confidence_adj -= 0.03
        signals_used.append(f"VIX_ELEVATED_{vix:.0f}")
        
    elif vix_regime == "calm":
        # Low VIX = trends persist
        confidence_adj += 0.05
        signals_used.append(f"VIX_CALM_{vix:.0f}")
    
    # =========================================================================
    # RULE 2: POSITIONING SIGNALS (Contrarian)
    # =========================================================================
    if positioning == "bullish" and base_direction == "UP":
        # Everyone already long - contrarian bearish
        direction_bias = "bearish"
        confidence_adj -= 0.05
        signals_used.append(f"PCR_COMPLACENT_{pcr:.2f}")
        
    elif positioning == "bearish" and base_direction == "DOWN":
        # Everyone already hedged - contrarian bullish
        direction_bias = "bullish"
        confidence_adj -= 0.05
        signals_used.append(f"PCR_HEDGED_{pcr:.2f}")
        
    elif positioning == "bearish" and base_direction == "UP":
        # Betting against the hedgers - risky but often right
        confidence_adj += 0.03
        signals_used.append(f"PCR_CONTRARIAN_{pcr:.2f}")
    
    # =========================================================================
    # RULE 3: FRAGILITY CHECK
    # =========================================================================
    if fragility > 80:
        # Market very fragile - reduce all confidence
        confidence_adj -= 0.10
        signals_used.append(f"FRAGILE_{fragility:.0f}")
        
    elif fragility > 60:
        confidence_adj -= 0.05
        signals_used.append(f"ELEVATED_FRAGILITY_{fragility:.0f}")
    
    # =========================================================================
    # RULE 4: MACRO REGIME
    # =========================================================================
    if macro_regime == "recession_warning":
        # Inverted yield curve - bearish bias
        direction_bias = "bearish"
        confidence_adj -= 0.10
        signals_used.append("YIELD_CURVE_INVERTED")
    
    # =========================================================================
    # RULE 5: ACTIVE EVENT IMPACT
    # =========================================================================
    max_event_score = 0.0
    event_count = len(active_events)
    
    for event in active_events[:5]:  # Top 5 events
        impact = event.get("impact", {})
        score = impact.get("score", 0)
        max_event_score = max(max_event_score, score)
        
        if score >= 70:
            # High impact event - block trading
            should_trade = False
            headline = event.get("event", {}).get("headline", "Unknown event")[:50]
            block_reason = f"High-impact event: {headline} (score={score:.0f})"
            signals_used.append(f"HIGH_IMPACT_EVENT_{score:.0f}")
            break
            
        elif score >= 50:
            # Medium impact - adjust confidence
            event_direction = impact.get("direction", "neutral")
            if event_direction == base_direction.lower():
                # Event aligns with prediction - boost
                confidence_adj += 0.05
                signals_used.append(f"EVENT_ALIGNED_{score:.0f}")
            elif event_direction in ["bullish", "bearish"]:
                # Event conflicts - reduce
                confidence_adj -= 0.05
                signals_used.append(f"EVENT_CONFLICT_{score:.0f}")
    
    # =========================================================================
    # RULE 6: SYMBOL-SPECIFIC IMPACT
    # =========================================================================
    if symbol_impact:
        symbol_score = symbol_impact.get("aggregate_score", 0)
        symbol_direction = symbol_impact.get("direction", "NEUTRAL")
        
        if symbol_score >= 50:
            if symbol_direction == base_direction:
                # Strong alignment
                confidence_adj += 0.08
                signals_used.append(f"SYMBOL_INTEL_{symbol_score:.0f}")
            elif symbol_direction != "NEUTRAL":
                # Conflict
                confidence_adj -= 0.08
                direction_bias = "bearish" if symbol_direction == "BEARISH" else "bullish"
                signals_used.append(f"SYMBOL_CONFLICT_{symbol_score:.0f}")
    
    # =========================================================================
    # RULE 7: TRUMP TARIFF PLAYBOOK (Kobeissi Letter Pattern)
    # =========================================================================
    # The pattern: Weekend announcement → Mon-Tue panic → Wed dip buying → Deal
    # Key trigger: 10Y Treasury yield > 4.50% = Trump warning zone
    #              10Y Treasury yield > 4.60% = Trump will pause (BUY signal)
    
    treasury_10y = intel_context.get("treasury_10y", 4.25)
    treasury_regime = intel_context.get("treasury_regime", "normal")
    tariff_active = intel_context.get("tariff_active", False)
    tariff_timing = intel_context.get("tariff_timing_window", "neutral")
    
    # 10Y Treasury yield signal (bond market forces Trump's hand)
    if treasury_regime == "trump_pause_imminent":
        # 10Y > 4.60% - Trump historically backs off here
        # This is a BULLISH signal - expect tariff pause
        if base_direction == "UP":
            confidence_adj += 0.10
            direction_bias = "bullish"
            signals_used.append(f"TARIFF_PAUSE_IMMINENT_10Y_{treasury_10y:.2f}")
            LOGGER.info(f"[{symbol}] 🎯 TARIFF PLAYBOOK: 10Y at {treasury_10y:.2f}% - pause imminent, bullish bias")
        elif base_direction == "DOWN":
            # DOWN prediction during pause signal - reduce confidence
            confidence_adj -= 0.08
            signals_used.append(f"TARIFF_PAUSE_CONFLICT_10Y_{treasury_10y:.2f}")
            
    elif treasury_regime == "trump_warning":
        # 10Y > 4.50% - warning zone, expect volatility
        confidence_adj -= 0.03
        signals_used.append(f"TARIFF_WARNING_10Y_{treasury_10y:.2f}")
    
    # Day-of-week timing during active tariff events
    if tariff_active:
        if tariff_timing == "panic_selling":
            # Mon-Tue during tariff event - DON'T sell into panic
            if base_direction == "DOWN":
                # Block SELL signals on Mon-Tue (panic trap)
                confidence_adj -= 0.10
                signals_used.append("TARIFF_PANIC_TRAP_MON_TUE")
                LOGGER.info(f"[{symbol}] 🚫 TARIFF PLAYBOOK: Mon-Tue panic trap - reducing DOWN confidence")
            elif base_direction == "UP":
                # BUY signals on Mon-Tue during tariff are risky but often right
                signals_used.append("TARIFF_EARLY_BUYER")
                
        elif tariff_timing == "dip_buying":
            # Wednesday - dip buyers emerge (smart money)
            if base_direction == "UP":
                confidence_adj += 0.05
                signals_used.append("TARIFF_DIP_BUYING_WED")
                LOGGER.info(f"[{symbol}] 🟢 TARIFF PLAYBOOK: Wednesday dip buying window")
                
        elif tariff_timing == "accumulation":
            # Thu-Fri - relief rally builds
            if base_direction == "UP":
                confidence_adj += 0.03
                signals_used.append("TARIFF_ACCUMULATION")
    
    # =========================================================================
    # RULE 8: 2025 WINNERS PLAYBOOK
    # =========================================================================
    # Historical data shows clear sector leadership and momentum persistence
    # - Storage/Memory: SNDK +559%, WDC +261%, MU +178%
    # - Semis: LRCX +138%, AMD +77%, NVDA +39%
    # - Gold miners: NEM +138% (follows precious metals)
    # - Sector leaders: Tech +24%, Comms +33.6%
    
    is_2025_winner = symbol.upper() in WINNERS_2025
    stock_sector = STOCK_SECTORS.get(symbol.upper())
    sector_performance = SECTOR_PERFORMANCE_2025.get(stock_sector, 0) if stock_sector else 0
    is_precious_metal = symbol.upper() in PRECIOUS_METALS
    
    winners_adjustment = 0.0
    
    # 2025 winner momentum bias
    if is_2025_winner:
        if base_direction == "UP":
            # Momentum continuation - winners tend to keep winning
            winners_adjustment += 0.05
            signals_used.append(f"2025_WINNER_{symbol.upper()}")
            LOGGER.info(f"[{symbol}] 🏆 2025 WINNER: Momentum continuation bias (+5%)")
        elif base_direction == "DOWN":
            # Betting against winners is risky
            winners_adjustment -= 0.03
            signals_used.append(f"2025_WINNER_FADE_RISK")
    
    # Leading sector bias
    if sector_performance >= 20:
        # Top sector (Tech, Comms)
        if base_direction == "UP":
            winners_adjustment += 0.04
            signals_used.append(f"SECTOR_LEADER_{stock_sector.upper()}")
        elif base_direction == "DOWN":
            winners_adjustment -= 0.02
            signals_used.append(f"SECTOR_LEADER_FADE_RISK")
    elif sector_performance <= 0:
        # Lagging sector (Energy)
        if base_direction == "DOWN":
            winners_adjustment += 0.03
            signals_used.append(f"SECTOR_LAGGARD_{stock_sector.upper()}")
        elif base_direction == "UP":
            winners_adjustment -= 0.02
            signals_used.append(f"SECTOR_LAGGARD_LONG_RISK")
    
    # Precious metals correlation (Gold +64%, Silver +146% in 2025)
    # If gold/silver, they tend to move together and trend strongly
    if is_precious_metal:
        # Precious metals showed extreme momentum in 2025
        if base_direction == "UP":
            winners_adjustment += 0.06
            signals_used.append("PRECIOUS_METALS_MOMENTUM")
            LOGGER.info(f"[{symbol}] 🥇 PRECIOUS METALS: Strong 2025 momentum (+6%)")
        # Don't penalize DOWN - they can correct too
    
    # =========================================================================
    # RULE 10: HOT THEMES (Yahoo Finance Most Active - Jan 2026)
    # =========================================================================
    # These themes are showing extreme volume and 52-week gains
    # High volume = institutional interest = momentum likely to continue
    
    symbol_upper = symbol.upper()
    hot_theme_adjustment = 0.0
    
    # Silver miners - OUTPERFORMING gold miners (Silver +146% vs Gold +64%)
    if symbol_upper in SILVER_MINERS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.07  # Silver miners are on fire
            signals_used.append("SILVER_MINER_HOT")
            LOGGER.info(f"[{symbol}] 🥈 SILVER MINER: Outperforming gold (+7%)")
        elif base_direction == "DOWN":
            hot_theme_adjustment -= 0.04
            signals_used.append("SILVER_MINER_FADE_RISK")
    
    # Rare Earth / Critical Minerals - GREENLAND PLAY
    # This ties directly to Trump's Greenland tariff strategy!
    if symbol_upper in RARE_EARTH_MINERALS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.06
            signals_used.append("RARE_EARTH_GREENLAND_PLAY")
            LOGGER.info(f"[{symbol}] 🌍 RARE EARTH: Greenland/critical minerals theme (+6%)")
        # Tariff resolution could hurt these - but also could explode higher
        if tariff_active:
            hot_theme_adjustment += 0.03  # Extra boost during tariff news
            signals_used.append("RARE_EARTH_TARIFF_CATALYST")
    
    # Uranium stocks - Nuclear renaissance theme
    if symbol_upper in URANIUM_STOCKS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.05
            signals_used.append("URANIUM_NUCLEAR_THEME")
            LOGGER.info(f"[{symbol}] ☢️ URANIUM: Nuclear renaissance theme (+5%)")
    
    # Bitcoin miners - crypto proxy (moves with BTC)
    if symbol_upper in BITCOIN_MINERS:
        # These are extremely volatile - adjust confidence based on direction
        if base_direction == "UP":
            hot_theme_adjustment += 0.04
            signals_used.append("BITCOIN_MINER_MOMENTUM")
        elif base_direction == "DOWN":
            # BTC miners can dump HARD
            hot_theme_adjustment += 0.02  # DOWN predictions on miners can be right
            signals_used.append("BITCOIN_MINER_VOLATILE")
    
    # AI/Quantum Computing - institutional money flooding in
    if symbol_upper in AI_QUANTUM_STOCKS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.05
            signals_used.append("AI_QUANTUM_THEME")
            LOGGER.info(f"[{symbol}] 🤖 AI/QUANTUM: Institutional interest (+5%)")
    
    # Storage/Memory monster run - the hottest sector
    if symbol_upper in STORAGE_MEMORY_STOCKS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.08  # Memory is ON FIRE
            signals_used.append("STORAGE_MEMORY_MONSTER")
            LOGGER.info(f"[{symbol}] 💾 STORAGE/MEMORY: Monster run sector (+8%)")
    
    # Cloud Infrastructure surge - institutional rotation
    if symbol_upper in CLOUD_INFRA_STOCKS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.06
            signals_used.append("CLOUD_INFRA_SURGE")
            LOGGER.info(f"[{symbol}] ☁️ CLOUD INFRA: Today's gainers (+6%)")
    
    # Space stocks - high beta momentum plays
    if symbol_upper in SPACE_STOCKS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.05
            signals_used.append("SPACE_MOMENTUM")
            LOGGER.info(f"[{symbol}] 🚀 SPACE: Momentum theme (+5%)")
        elif base_direction == "DOWN":
            # Space stocks are VOLATILE - down calls can be right
            hot_theme_adjustment += 0.03
            signals_used.append("SPACE_VOLATILE")
    
    # Gold miners - precious metals wave
    if symbol_upper in GOLD_MINERS:
        if base_direction == "UP":
            hot_theme_adjustment += 0.06
            signals_used.append("GOLD_MINER_WAVE")
            LOGGER.info(f"[{symbol}] 🥇 GOLD MINER: Precious metals wave (+6%)")
    
    winners_adjustment += hot_theme_adjustment
    confidence_adj += winners_adjustment
    
    # =========================================================================
    # RULE 11: FALLEN ANGELS & GOVERNMENT RISK (Jan 27, 2026)
    # =========================================================================
    # Two key patterns:
    # 1. Fallen Angels: Stocks down 50-80% from highs - potential bounce plays
    # 2. Government Contract Risk: Treasury cancellations = sudden drops
    
    fallen_angels_adjustment = 0.0
    
    # FALLEN ANGELS - Extreme beaten-down names
    # These have bounce potential but require careful entry timing
    if symbol_upper in FALLEN_ANGELS:
        if base_direction == "DOWN":
            # Don't chase shorts on already-crushed names
            fallen_angels_adjustment -= 0.06
            signals_used.append("FALLEN_ANGEL_DONT_SHORT")
            LOGGER.warning(f"[{symbol}] 📉 FALLEN ANGEL: Already down 50%+, don't pile on shorts")
        elif base_direction == "UP":
            # Bounce potential but risky - slight boost
            fallen_angels_adjustment += 0.03
            signals_used.append("FALLEN_ANGEL_BOUNCE_POTENTIAL")
            LOGGER.info(f"[{symbol}] 📈 FALLEN ANGEL: Bounce potential on beaten name (+3%)")
    
    # GOVERNMENT CONTRACT RISK - Defense contractors exposed to DOGE/cancellations
    if symbol_upper in GOVT_CONTRACT_RISK:
        # News-driven volatility - reduce confidence on both sides
        fallen_angels_adjustment -= 0.03
        signals_used.append("GOVT_CONTRACT_RISK")
        LOGGER.warning(f"[{symbol}] ⚠️ GOVT CONTRACT: Treasury/DOGE cancellation risk")
    
    confidence_adj += fallen_angels_adjustment
    
    # =========================================================================
    # RULE 12: CRYPTO & COMMODITIES INTELLIGENCE (Jan 27, 2026)
    # =========================================================================
    # Crypto themes:
    # - Privacy coins crushing (ZEC +689%, XMR +106%)
    # - Gaming tokens hot (AXS +34% today)
    # - Meme coins crashed (TRUMP -83%, DOGE -62%)
    # Commodities:
    # - Gold near ATH at $5,057
    # - Silver volatile (down 6.24% today)
    
    crypto_commodities_adj = 0.0
    
    # Normalize crypto symbols for matching (ZEC -> ZEC-USD, BTC -> BTC-USD)
    # This fixes the format mismatch between predictor (ZEC) and sets (ZEC-USD)
    crypto_symbol_for_match = symbol_upper
    if not symbol_upper.endswith("-USD"):
        crypto_symbol_for_match = f"{symbol_upper}-USD"
    
    # Check if this is a crypto symbol (ends in -USD or matches crypto sets)
    is_crypto = (symbol_upper.endswith("-USD") or 
                 symbol_upper in CRYPTO_STOCKS or
                 crypto_symbol_for_match in PRIVACY_COINS or
                 crypto_symbol_for_match in GAMING_CRYPTO or
                 crypto_symbol_for_match in MEME_COINS or
                 crypto_symbol_for_match in LAYER1_MAJORS or
                 crypto_symbol_for_match in DEFI_MOMENTUM or
                 crypto_symbol_for_match in AI_GPU_CRYPTO)
    
    if is_crypto:
        # Privacy coins - CRUSHING IT (ZEC +689%, XMR +106%)
        if crypto_symbol_for_match in PRIVACY_COINS:
            if base_direction == "UP":
                crypto_commodities_adj += 0.08
                signals_used.append("PRIVACY_COIN_HOT")
                LOGGER.info(f"[{symbol}] 🔐 PRIVACY COIN: +689% ZEC theme (+8%)")
        
        # AI/GPU Infrastructure - RNDR, FET, TAO etc (HIGH GROWTH)
        if crypto_symbol_for_match in AI_GPU_CRYPTO:
            if base_direction == "UP":
                crypto_commodities_adj += 0.07
                signals_used.append("AI_GPU_CRYPTO_HOT")
                LOGGER.info(f"[{symbol}] 🤖 AI/GPU CRYPTO: Infrastructure play (+7%)")
            elif base_direction == "DOWN":
                # AI is the future - don't short these hard
                crypto_commodities_adj -= 0.02
                signals_used.append("AI_GPU_DONT_SHORT")
        
        # Gaming/Metaverse - AXS +34% today
        if crypto_symbol_for_match in GAMING_CRYPTO:
            if base_direction == "UP":
                crypto_commodities_adj += 0.06
                signals_used.append("GAMING_CRYPTO_HOT")
                LOGGER.info(f"[{symbol}] 🎮 GAMING CRYPTO: AXS momentum (+6%)")
        
        # DeFi momentum
        if crypto_symbol_for_match in DEFI_MOMENTUM:
            if base_direction == "UP":
                crypto_commodities_adj += 0.05
                signals_used.append("DEFI_MOMENTUM")
        
        # Meme coins - CRASHED, don't chase
        if crypto_symbol_for_match in MEME_COINS:
            if base_direction == "UP":
                # Don't chase meme pumps
                crypto_commodities_adj -= 0.05
                signals_used.append("MEME_COIN_RISKY")
                LOGGER.warning(f"[{symbol}] 🎰 MEME COIN: High risk, most down 60-80% (-5%)")
            elif base_direction == "DOWN":
                # Shorting memes is risky too - can pump randomly
                crypto_commodities_adj -= 0.03
                signals_used.append("MEME_COIN_VOLATILE")
        
        # Layer 1 majors - more stable
        if crypto_symbol_for_match in LAYER1_MAJORS:
            # These are the "blue chips" of crypto
            crypto_commodities_adj += 0.02
            signals_used.append("LAYER1_MAJOR")
    
    # Crypto-correlated stocks
    if symbol_upper in CRYPTO_STOCKS:
        # Get BTC context if available
        btc_price = intel_context.get("btc_price", 0)
        if btc_price > 100000:
            crypto_commodities_adj += 0.05
            signals_used.append("BTC_ABOVE_100K")
        elif btc_price < 70000:
            crypto_commodities_adj -= 0.05
            signals_used.append("BTC_WEAKNESS")
    
    # Gold/Silver correlation for miners
    gold_price = intel_context.get("gold_price", 0)
    silver_price = intel_context.get("silver_price", 0)
    
    if symbol_upper in GOLD_MINERS or symbol_upper in SILVER_MINERS:
        if gold_price >= COMMODITIES_THRESHOLDS["gold_bullish"]:
            crypto_commodities_adj += 0.04
            signals_used.append(f"GOLD_ABOVE_{COMMODITIES_THRESHOLDS['gold_bullish']}")
            LOGGER.info(f"[{symbol}] 🥇 Gold ${gold_price:.0f} - bullish for miners (+4%)")
        
        if gold_price >= COMMODITIES_THRESHOLDS["gold_extreme_bullish"]:
            crypto_commodities_adj += 0.03  # Extra boost
            signals_used.append("GOLD_ATH_TERRITORY")
    
    if symbol_upper in SILVER_MINERS:
        if silver_price >= COMMODITIES_THRESHOLDS["silver_bullish"]:
            crypto_commodities_adj += 0.03
            signals_used.append(f"SILVER_ABOVE_{COMMODITIES_THRESHOLDS['silver_bullish']}")
    
    confidence_adj += crypto_commodities_adj
    
    # =========================================================================
    # RULE 13: CURRENCY/FX INTELLIGENCE (Jan 27, 2026)
    # =========================================================================
    # Key FX signals:
    # - Weak dollar (DXY 4-month low) = Gold bullish, multinationals benefit
    # - Yen intervention talk = Risk-off signal
    # - EM currency stress (INR at record low) = Risk indicator
    # - "Debasement trade" = Investors fleeing bonds/currencies to gold
    
    fx_adjustment = 0.0
    
    # Get FX data from context
    eurusd = intel_context.get("eurusd", 0)
    usdjpy = intel_context.get("usdjpy", 0)
    usdmxn = intel_context.get("usdmxn", 0)
    usdinr = intel_context.get("usdinr", 0)
    dxy_trend = intel_context.get("dxy_trend", "neutral")  # "weak", "strong", "neutral"
    
    # WEAK DOLLAR REGIME (Current state as of Jan 27)
    if dxy_trend == "weak" or eurusd >= FX_THRESHOLDS["eurusd_dollar_weak"]:
        # Weak dollar benefits multinationals and gold
        if symbol_upper in DOLLAR_SENSITIVE_STOCKS:
            if base_direction == "UP":
                fx_adjustment += 0.04
                signals_used.append("WEAK_DOLLAR_MULTINATIONAL_BOOST")
                LOGGER.info(f"[{symbol}] 💵 WEAK DOLLAR: Multinational earnings boost (+4%)")
        
        # Weak dollar = very bullish for gold/miners
        if symbol_upper in GOLD_MINERS or symbol_upper in SILVER_MINERS:
            fx_adjustment += 0.03
            signals_used.append("WEAK_DOLLAR_GOLD_BULLISH")
            LOGGER.info(f"[{symbol}] 💵 WEAK DOLLAR: Debasement trade boosting precious metals (+3%)")
    
    # STRONG DOLLAR (opposite effect)
    if dxy_trend == "strong" or (eurusd > 0 and eurusd <= FX_THRESHOLDS["eurusd_dollar_strong"]):
        if symbol_upper in DOLLAR_SENSITIVE_STOCKS:
            if base_direction == "UP":
                fx_adjustment -= 0.03
                signals_used.append("STRONG_DOLLAR_HEADWIND")
        
        if symbol_upper in GOLD_MINERS or symbol_upper in SILVER_MINERS:
            fx_adjustment -= 0.03
            signals_used.append("STRONG_DOLLAR_GOLD_BEARISH")
    
    # YEN SIGNALS (Risk-off indicator)
    if usdjpy > 0:
        if usdjpy <= FX_THRESHOLDS["usdjpy_yen_strong"]:
            # Strong yen = risk-off, defensive positioning
            if base_direction == "DOWN":
                fx_adjustment += 0.03
                signals_used.append("YEN_STRONG_RISK_OFF")
                LOGGER.warning(f"[{symbol}] 🇯🇵 YEN STRONG: Risk-off signal, SELL confidence +3%")
        
        if usdjpy >= FX_THRESHOLDS["usdjpy_intervention_risk"]:
            # BOJ intervention risk
            if symbol_upper in YEN_SENSITIVE_STOCKS:
                fx_adjustment -= 0.03
                signals_used.append("YEN_INTERVENTION_RISK")
                LOGGER.warning(f"[{symbol}] 🇯🇵 YEN: BOJ intervention risk")
    
    # EMERGING MARKET STRESS (INR at record low)
    if usdinr >= FX_THRESHOLDS["usdinr_rupee_stress"]:
        if symbol_upper in EM_SENSITIVE_STOCKS:
            # EM stress = reduce confidence
            fx_adjustment -= 0.04
            signals_used.append("EM_CURRENCY_STRESS")
            LOGGER.warning(f"[{symbol}] 🌍 EM STRESS: INR at record low, reduce confidence")
    
    if usdinr >= FX_THRESHOLDS["usdinr_rupee_crisis"]:
        if symbol_upper in EM_SENSITIVE_STOCKS:
            fx_adjustment -= 0.03  # Additional penalty
            signals_used.append("EM_CURRENCY_CRISIS")
    
    confidence_adj += fx_adjustment
    
    # =========================================================================
    # RULE 9: TRADING DISCIPLINE (Professional Trader Principles)
    # =========================================================================
    # "Successful traders treat the market as a game of probabilities"
    # Key principles:
    # - Defense over offense (protect capital first)
    # - 2:1 reward/risk minimum
    # - RSI extremes signal reversals
    # - Volume confirms moves
    # - Patience: wait for A+ setups
    
    discipline_adjustment = 0.0
    discipline_signals = []
    
    # Get trading metrics from intel_context (if available from prediction data)
    reward_risk = intel_context.get("reward_risk_ratio", 0)
    rsi = intel_context.get("rsi", 50)
    relative_volume = intel_context.get("relative_volume", 1.0)
    volatility_pct = intel_context.get("volatility_pct", 3.0)
    vwap_position = intel_context.get("vwap_position", "neutral")  # "above", "below", "neutral"
    
    # Reward/Risk ratio evaluation
    if reward_risk > 0:
        if reward_risk >= TRADING_DISCIPLINE["excellent_reward_risk_ratio"]:
            # 3:1 or better - excellent setup
            discipline_adjustment += 0.08
            discipline_signals.append(f"EXCELLENT_RR_{reward_risk:.1f}")
            LOGGER.info(f"[{symbol}] 🎯 DISCIPLINE: Excellent R/R {reward_risk:.1f}:1 (+8%)")
        elif reward_risk >= TRADING_DISCIPLINE["ideal_reward_risk_ratio"]:
            # 2:1 - good setup
            discipline_adjustment += 0.04
            discipline_signals.append(f"GOOD_RR_{reward_risk:.1f}")
        elif reward_risk < TRADING_DISCIPLINE["min_reward_risk_ratio"]:
            # Below 1.5:1 - not worth the risk
            discipline_adjustment -= 0.08
            discipline_signals.append(f"POOR_RR_{reward_risk:.1f}")
            LOGGER.warning(f"[{symbol}] ⚠️ DISCIPLINE: Poor R/R {reward_risk:.1f}:1 - reduce confidence")
    
    # RSI (Relative Strength Index) signals - contrarian at extremes
    if rsi > 0:
        if rsi >= TRADING_DISCIPLINE["rsi_extreme_overbought"]:
            # RSI > 80 - extremely overbought
            if base_direction == "UP":
                discipline_adjustment -= 0.10
                discipline_signals.append(f"RSI_EXTREME_OVERBOUGHT_{rsi:.0f}")
                LOGGER.warning(f"[{symbol}] 🔴 RSI {rsi:.0f} - extreme overbought, fade risk on BUY")
            elif base_direction == "DOWN":
                discipline_adjustment += 0.05
                discipline_signals.append(f"RSI_REVERSAL_SETUP_{rsi:.0f}")
                
        elif rsi >= TRADING_DISCIPLINE["rsi_overbought"]:
            # RSI 70-80 - overbought
            if base_direction == "UP":
                discipline_adjustment -= 0.05
                discipline_signals.append(f"RSI_OVERBOUGHT_{rsi:.0f}")
                
        elif rsi <= TRADING_DISCIPLINE["rsi_extreme_oversold"]:
            # RSI < 20 - capitulation zone
            if base_direction == "DOWN":
                discipline_adjustment -= 0.10
                discipline_signals.append(f"RSI_EXTREME_OVERSOLD_{rsi:.0f}")
                LOGGER.warning(f"[{symbol}] 🟢 RSI {rsi:.0f} - capitulation zone, bounce likely")
            elif base_direction == "UP":
                discipline_adjustment += 0.05
                discipline_signals.append(f"RSI_BOUNCE_SETUP_{rsi:.0f}")
                
        elif rsi <= TRADING_DISCIPLINE["rsi_oversold"]:
            # RSI 20-30 - oversold
            if base_direction == "UP":
                discipline_adjustment += 0.03
                discipline_signals.append(f"RSI_OVERSOLD_{rsi:.0f}")
    
    # Volume/Liquidity check - "High trading volume ensures clean entries/exits"
    if relative_volume > 0:
        if relative_volume < TRADING_DISCIPLINE["min_relative_volume"]:
            # Dead stock - no interest
            discipline_adjustment -= 0.08
            discipline_signals.append(f"LOW_VOLUME_{relative_volume:.1f}x")
            LOGGER.warning(f"[{symbol}] 💀 Low volume {relative_volume:.1f}x - dead stock")
            
        elif relative_volume >= TRADING_DISCIPLINE["very_high_volume"]:
            # 5x+ volume - unusual activity, pay attention
            discipline_adjustment += 0.05
            discipline_signals.append(f"UNUSUAL_VOLUME_{relative_volume:.1f}x")
            LOGGER.info(f"[{symbol}] 📊 Unusual volume {relative_volume:.1f}x - institutional interest")
            
        elif relative_volume >= TRADING_DISCIPLINE["high_relative_volume"]:
            # 2x+ volume - good interest
            discipline_adjustment += 0.03
            discipline_signals.append(f"HIGH_VOLUME_{relative_volume:.1f}x")
    
    # Volatility check - "Need price movement to profit"
    if volatility_pct > 0:
        if volatility_pct < TRADING_DISCIPLINE["min_volatility_pct"]:
            # Too flat - no opportunity
            discipline_adjustment -= 0.05
            discipline_signals.append(f"LOW_VOLATILITY_{volatility_pct:.1f}pct")
            
        elif volatility_pct > TRADING_DISCIPLINE["max_volatility_pct"]:
            # Too wild - reduce confidence (higher risk)
            discipline_adjustment -= 0.05
            discipline_signals.append(f"HIGH_VOLATILITY_{volatility_pct:.1f}pct")
    
    # VWAP positioning - "Where big institutions are buying"
    if vwap_position == "above" and base_direction == "UP":
        # Price above VWAP + bullish = momentum confirmed
        discipline_adjustment += 0.03
        discipline_signals.append("ABOVE_VWAP_BULLISH")
    elif vwap_position == "below" and base_direction == "DOWN":
        # Price below VWAP + bearish = momentum confirmed
        discipline_adjustment += 0.03
        discipline_signals.append("BELOW_VWAP_BEARISH")
    elif vwap_position == "above" and base_direction == "DOWN":
        # Counter-trend trade - riskier
        discipline_adjustment -= 0.02
        discipline_signals.append("ABOVE_VWAP_COUNTER_TREND")
    elif vwap_position == "below" and base_direction == "UP":
        # Trying to catch falling knife
        discipline_adjustment -= 0.02
        discipline_signals.append("BELOW_VWAP_COUNTER_TREND")
    
    confidence_adj += discipline_adjustment
    signals_used.extend(discipline_signals)
    
    # =========================================================================
    # RULE 14: FULL STOCK HISTORY INTELLIGENCE (IPO to now)
    # =========================================================================
    # Use comprehensive historical data for better decisions:
    # - 52-week positioning (near high = momentum, near low = bounce potential)
    # - Fundamentals (P/E, growth) for valuation context
    # - Technical trend confirmation
    # - Earnings proximity risk
    # - Historical volatility context
    
    history_adjustment = 0.0
    history_signals = []
    stock_history = intel_context.get("stock_history", {})
    
    if stock_history:
        # 52-WEEK CONTEXT
        is_near_52w_high = intel_context.get("is_near_52w_high", False)
        is_near_52w_low = intel_context.get("is_near_52w_low", False)
        pct_from_high = intel_context.get("pct_from_52w_high", 0)
        pct_from_low = intel_context.get("pct_from_52w_low", 0)
        
        # Near 52-week high - momentum stocks tend to keep running
        if is_near_52w_high and base_direction == "UP":
            history_adjustment += 0.04
            history_signals.append(f"NEAR_52W_HIGH_{pct_from_high:.1f}pct")
            LOGGER.info(f"[{symbol}] 📈 Near 52-week high ({pct_from_high:.1f}%) - momentum bias (+4%)")
        
        # Near 52-week low - potential bounce but risky
        if is_near_52w_low:
            if base_direction == "UP":
                # Bounce play - contrarian but needs catalyst
                history_adjustment += 0.02
                history_signals.append(f"NEAR_52W_LOW_BOUNCE_{pct_from_low:.1f}pct")
            elif base_direction == "DOWN":
                # Don't short at lows (oversold)
                history_adjustment -= 0.05
                history_signals.append("DONT_SHORT_AT_LOWS")
                LOGGER.info(f"[{symbol}] ⚠️ Near 52-week low - don't short oversold (-5%)")
        
        # OVERBOUGHT/OVERSOLD FROM HISTORY
        is_oversold = intel_context.get("is_oversold", False)
        is_overbought = intel_context.get("is_overbought", False)
        
        if is_oversold and base_direction == "UP":
            history_adjustment += 0.05
            history_signals.append("HISTORICAL_OVERSOLD_BOUNCE")
            LOGGER.info(f"[{symbol}] 🟢 Historically oversold - bounce setup (+5%)")
        
        if is_overbought and base_direction == "DOWN":
            history_adjustment += 0.04
            history_signals.append("HISTORICAL_OVERBOUGHT_FADE")
            LOGGER.info(f"[{symbol}] 🔴 Historically overbought - fade setup (+4%)")
        
        # TREND CONFIRMATION
        trend = intel_context.get("trend", "sideways")
        above_200 = intel_context.get("above_sma_200", False)
        
        if trend == "uptrend" and base_direction == "UP" and above_200:
            history_adjustment += 0.03
            history_signals.append("TREND_CONFIRMED_UP")
        elif trend == "downtrend" and base_direction == "DOWN" and not above_200:
            history_adjustment += 0.03
            history_signals.append("TREND_CONFIRMED_DOWN")
        elif trend == "uptrend" and base_direction == "DOWN":
            history_adjustment -= 0.04
            history_signals.append("COUNTER_TREND_RISK")
            LOGGER.info(f"[{symbol}] ⚠️ Shorting uptrend - counter-trend risk (-4%)")
        
        # EARNINGS PROXIMITY RISK
        days_to_earnings = intel_context.get("days_to_earnings")
        if days_to_earnings is not None:
            if 0 < days_to_earnings <= 3:
                # 3 days or less to earnings - very risky
                history_adjustment -= 0.08
                history_signals.append(f"EARNINGS_IMMINENT_{days_to_earnings}d")
                LOGGER.warning(f"[{symbol}] 📅 Earnings in {days_to_earnings} days - HIGH RISK (-8%)")
            elif days_to_earnings <= 7:
                # 1 week to earnings - elevated risk
                history_adjustment -= 0.04
                history_signals.append(f"EARNINGS_WEEK_{days_to_earnings}d")
        
        # FUNDAMENTAL VALUATION (if available)
        pe_ratio = intel_context.get("pe_ratio")
        sector = intel_context.get("sector", "Unknown")
        
        if pe_ratio:
            # Tech sectors can sustain higher P/E
            tech_sectors = ["Technology", "Communication Services"]
            if sector in tech_sectors:
                if pe_ratio > 100:
                    history_signals.append(f"HIGH_PE_{pe_ratio:.0f}")
                    if base_direction == "UP":
                        history_adjustment -= 0.02  # Slight headwind for very expensive stocks
            else:
                # Non-tech - more P/E sensitive
                if pe_ratio > 50:
                    history_signals.append(f"EXPENSIVE_PE_{pe_ratio:.0f}")
                    if base_direction == "UP":
                        history_adjustment -= 0.03
                elif pe_ratio < 15 and pe_ratio > 0:
                    history_signals.append(f"VALUE_PE_{pe_ratio:.0f}")
                    if base_direction == "UP":
                        history_adjustment += 0.02  # Value stocks can run
        
        # HIGH VOLUME BREAKOUT
        is_high_volume = intel_context.get("is_high_volume", False)
        if is_high_volume:
            if base_direction == "UP" and is_near_52w_high:
                history_adjustment += 0.05
                history_signals.append("HIGH_VOLUME_BREAKOUT")
                LOGGER.info(f"[{symbol}] 🔥 High volume + 52w high = breakout potential (+5%)")
            elif base_direction == "DOWN" and is_near_52w_low:
                history_adjustment += 0.03
                history_signals.append("HIGH_VOLUME_BREAKDOWN")
    
    confidence_adj += history_adjustment
    signals_used.extend(history_signals)
    
    # =========================================================================
    # FINALIZE SIGNAL
    # =========================================================================
    # Cap confidence adjustment (increased to 0.30 for all playbooks combined)
    confidence_adj = max(-0.30, min(0.30, confidence_adj))
    
    return IntelSignal(
        direction_bias=direction_bias,
        confidence_adjustment=confidence_adj,
        should_trade=should_trade,
        block_reason=block_reason,
        signal_sources=signals_used,
        market_context={
            "vix": vix,
            "vix_regime": vix_regime,
            "put_call_ratio": pcr,
            "positioning": positioning,
            "fragility": fragility,
            "macro_regime": macro_regime,
        },
        event_count=event_count,
        max_event_score=max_event_score,
        tariff_context={
            "treasury_10y": treasury_10y,
            "treasury_regime": treasury_regime,
            "tariff_active": tariff_active,
            "tariff_timing": tariff_timing,
            "day_of_week": intel_context.get("day_of_week", -1),
        },
        winners_context={
            "is_2025_winner": is_2025_winner,
            "sector": stock_sector,
            "sector_performance": sector_performance,
            "is_precious_metal": is_precious_metal,
            "winners_adjustment": winners_adjustment,
        },
        discipline_context={
            "reward_risk": reward_risk,
            "rsi": rsi,
            "relative_volume": relative_volume,
            "volatility_pct": volatility_pct,
            "vwap_position": vwap_position,
            "discipline_adjustment": discipline_adjustment,
            "discipline_signals": discipline_signals,
        },
    )


async def get_intel_signal_for_prediction(
    symbol: str,
    direction: str,
    confidence: float,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Main entry point for prediction engine integration.
    
    Args:
        symbol: Trading symbol
        direction: Current prediction direction ("UP", "DOWN", "FLAT")
        confidence: Current confidence (0-1)
    
    Returns:
        (adjusted_direction, adjusted_confidence, intel_metadata)
    """
    # NULL SAFETY: Normalize inputs
    if direction is None:
        direction = "FLAT"
    direction = str(direction).upper()
    
    # Check if Intel is enabled
    if os.getenv("GHOST_INTEL_ENABLED", "1") != "1":
        return direction, confidence, {"intel_enabled": False}
    
    try:
        # Fetch intel context
        context = await fetch_intel_context(symbol)
        
        # Calculate signal
        signal = calculate_intel_signal(symbol, direction, confidence, context)
        
        # Apply adjustments
        adjusted_confidence = confidence + signal.confidence_adjustment
        adjusted_confidence = max(0.0, min(0.95, adjusted_confidence))
        
        adjusted_direction = direction
        
        # Block check
        if not signal.should_trade:
            adjusted_confidence = 0.0
            adjusted_direction = "HOLD"
            LOGGER.warning(
                f"[{symbol}] 🚫 INTEL BLOCK: {signal.block_reason}"
            )
        elif signal.confidence_adjustment != 0:
            LOGGER.info(
                f"[{symbol}] 🔮 Intel adjustment: {confidence:.1%} → {adjusted_confidence:.1%} "
                f"({signal.confidence_adjustment:+.1%}) | Sources: {', '.join(signal.signal_sources[:3])}"
            )
        
        # Build metadata for logging/debugging
        metadata = {
            "intel_enabled": True,
            "intel_applied": True,
            "original_confidence": confidence,
            "adjusted_confidence": adjusted_confidence,
            "confidence_adjustment": signal.confidence_adjustment,
            "direction_bias": signal.direction_bias,
            "should_trade": signal.should_trade,
            "block_reason": signal.block_reason,
            "signal_sources": signal.signal_sources,
            "market_context": signal.market_context,
            "event_count": signal.event_count,
            "max_event_score": signal.max_event_score,
        }
        
        return adjusted_direction, adjusted_confidence, metadata
        
    except Exception as e:
        LOGGER.warning(f"[{symbol}] Intel integration failed (continuing without): {e}")
        return direction, confidence, {"intel_enabled": True, "intel_error": str(e)}


# Synchronous wrapper for wolf_app.py integration
def apply_intel_to_prediction(
    symbol: str,
    direction: str,
    confidence: float,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Synchronous wrapper for wolf_app.py integration.
    
    Call this from run_single_prediction() after feature extraction.
    """
    import asyncio
    
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in async context - need to use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    get_intel_signal_for_prediction(symbol, direction, confidence)
                )
                return future.result(timeout=5)
        except RuntimeError:
            # No running loop - we can use asyncio.run directly
            return asyncio.run(
                get_intel_signal_for_prediction(symbol, direction, confidence)
            )
    except Exception as e:
        LOGGER.warning(f"[{symbol}] Intel sync wrapper failed: {e}")
        return direction, confidence, {"intel_error": str(e)}
