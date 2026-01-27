"""
Ghost Intel Integration - Wires institutional intelligence into predictions.

This module provides the bridge between Ghost Intel feeds and the prediction engine.
It transforms raw intel data into actionable signals that affect:
1. Direction bias (bullish/bearish tilt based on macro + positioning)
2. Confidence adjustments (+/- based on event impact and market fragility)
3. Entry timing gates (block trades during high-impact events)

Integration points in wolf_app.py:
- After feature extraction (~line 8020)
- Before ensemble prediction (~line 8170)
- As part of market gates (~line 8310)

Target impact: 10-20% accuracy improvement from institutional timing
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

LOGGER = logging.getLogger("ghost.intel.integration")

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
    
    _set_cached(cache_key, context)
    return context


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
    # FINALIZE SIGNAL
    # =========================================================================
    # Cap confidence adjustment
    confidence_adj = max(-0.15, min(0.15, confidence_adj))
    
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
