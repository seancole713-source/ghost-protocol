"""
Ghost Intel Integration - Wires institutional intelligence into predictions.

This module provides the bridge between Ghost Intel feeds and the prediction engine.
It transforms raw intel data into actionable signals that affect:
1. Direction bias (bullish/bearish tilt based on macro + positioning)
2. Confidence adjustments (+/- based on event impact and market fragility)
3. Entry timing gates (block trades during high-impact events)
4. Trump Tariff Playbook (bond market triggers, timing patterns)
5. 2025 Winners Playbook (sector leadership, momentum stocks)

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
    "NEM", "GDX", "GOLD", "AEM", "KGC",
    # Big tech (still performing)
    "GOOGL", "GOOG", "GE", "RTX",
    # Media recovery
    "WBD",
}

# Sector mapping for stocks
STOCK_SECTORS = {
    # Technology
    "SNDK": "technology", "WDC": "technology", "STX": "technology", "MU": "technology",
    "LRCX": "technology", "AMD": "technology", "NVDA": "technology", "AVGO": "technology",
    "INTC": "technology", "PLTR": "technology", "APP": "technology", "APH": "technology",
    "AAPL": "technology", "MSFT": "technology",
    # Communication Services
    "GOOGL": "communication_services", "GOOG": "communication_services",
    "META": "communication_services", "WBD": "communication_services",
    "NFLX": "communication_services", "DIS": "communication_services",
    # Financials
    "HOOD": "financials", "JPM": "financials", "BAC": "financials", "GS": "financials",
    # Materials (Gold miners)
    "NEM": "materials", "GDX": "materials", "GOLD": "materials", "AEM": "materials", "KGC": "materials",
    # Industrials
    "GE": "industrials", "RTX": "industrials", "BA": "industrials", "CAT": "industrials",
}

# Precious metals tickers (for correlation)
PRECIOUS_METALS = {"GLD", "SLV", "IAU", "GOLD", "NEM", "GDX", "XAUUSD", "XAGUSD"}

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
    
    confidence_adj += winners_adjustment
    
    # =========================================================================
    # FINALIZE SIGNAL
    # =========================================================================
    # Cap confidence adjustment (increased to 0.25 for combined playbooks)
    confidence_adj = max(-0.25, min(0.25, confidence_adj))
    
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
