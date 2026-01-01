"""
GPT-4 Analyst Module
====================
AI-powered analysis of predictions using OpenAI's GPT-4.

Features:
- Analyzes prediction rationale
- Provides market context commentary
- Generates risk assessments
- Suggests entry/exit strategies
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

LOGGER = logging.getLogger("ghost.gpt4_analyst")

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
ENABLE_GPT4_ANALYST = os.getenv("ENABLE_GPT4_ANALYST", "0") == "1"

# Cache for analysis results (avoid repeated API calls)
_analysis_cache: Dict[str, Dict[str, Any]] = {}
_cache_ttl = 3600  # 1 hour


def is_enabled() -> bool:
    """Check if GPT-4 analyst is enabled and configured."""
    return ENABLE_GPT4_ANALYST and bool(OPENAI_API_KEY)


async def analyze_prediction(
    symbol: str,
    direction: str,
    confidence: float,
    current_price: float,
    target_price: float,
    features: Dict[str, Any] = None,
    market_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Analyze a prediction using GPT-4.
    
    Args:
        symbol: Asset symbol (e.g., "BTC", "AAPL")
        direction: Predicted direction ("UP" or "DOWN")
        confidence: Prediction confidence (0-1)
        current_price: Current asset price
        target_price: Predicted target price
        features: Optional technical features dict
        market_context: Optional market context dict
        
    Returns:
        Analysis dict with reasoning, risk assessment, and recommendations
    """
    if not is_enabled():
        return {
            "ok": False,
            "error": "GPT-4 Analyst not enabled",
            "enabled": False,
        }
    
    # Check cache
    cache_key = f"{symbol}_{direction}_{confidence:.2f}"
    cached = _analysis_cache.get(cache_key)
    if cached and time.time() - cached.get("timestamp", 0) < _cache_ttl:
        return cached
    
    try:
        # Build the analysis prompt
        prompt = _build_analysis_prompt(
            symbol, direction, confidence, current_price, target_price,
            features, market_context
        )
        
        # Call OpenAI API
        analysis = await _call_openai(prompt)
        
        result = {
            "ok": True,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "analysis": analysis,
            "timestamp": time.time(),
            "model": OPENAI_MODEL,
        }
        
        # Cache the result
        _analysis_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        LOGGER.error(f"GPT-4 analysis failed for {symbol}: {e}")
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol,
        }


def _build_analysis_prompt(
    symbol: str,
    direction: str,
    confidence: float,
    current_price: float,
    target_price: float,
    features: Dict[str, Any] = None,
    market_context: Dict[str, Any] = None,
) -> str:
    """Build the analysis prompt for GPT-4."""
    
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    move_pct = ((target_price - current_price) / current_price) * 100 if current_price else 0
    
    prompt = f"""You are a professional trading analyst. Analyze this prediction:

PREDICTION:
- Symbol: {symbol}
- Direction: {direction}
- Confidence: {conf_pct:.1f}%
- Current Price: ${current_price:,.2f}
- Target Price: ${target_price:,.2f}
- Expected Move: {move_pct:+.2f}%
"""
    
    if features:
        prompt += f"""
TECHNICAL FEATURES:
- RSI: {features.get('RSI_14', 'N/A')}
- MACD Signal: {features.get('MACD_SIGNAL', 'N/A')}
- Volume Ratio: {features.get('VOLUME_RATIO', 'N/A')}
- ATR%: {features.get('ATR_PERCENT', 'N/A')}
"""
    
    if market_context:
        prompt += f"""
MARKET CONTEXT:
- Market Regime: {market_context.get('market_regime', 'N/A')}
- VIX: {market_context.get('vix', 'N/A')}
- Sentiment: {market_context.get('sentiment', 'N/A')}
"""
    
    prompt += """
Provide a brief analysis (3-4 sentences) covering:
1. Whether this prediction makes sense given the data
2. Key risk factors to watch
3. Recommended position size (conservative/moderate/aggressive)

Be concise and actionable. Focus on practical trading advice."""
    
    return prompt


async def _call_openai(prompt: str) -> str:
    """Call OpenAI API with the prompt."""
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise trading analyst. Provide brief, actionable analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def get_market_commentary(symbols: list[str] = None) -> Dict[str, Any]:
    """
    Get general market commentary from GPT-4.
    
    Args:
        symbols: Optional list of symbols to focus on
        
    Returns:
        Market commentary dict
    """
    if not is_enabled():
        return {
            "ok": False,
            "error": "GPT-4 Analyst not enabled",
        }
    
    try:
        symbols_str = ", ".join(symbols[:5]) if symbols else "BTC, ETH, AAPL, TSLA"
        
        prompt = f"""Provide a brief market outlook (2-3 sentences) for:
{symbols_str}

Focus on:
- Current market sentiment
- Key levels to watch
- Risk/reward outlook

Be concise and trading-focused."""
        
        commentary = await _call_openai(prompt)
        
        return {
            "ok": True,
            "commentary": commentary,
            "symbols": symbols,
            "timestamp": time.time(),
            "model": OPENAI_MODEL,
        }
        
    except Exception as e:
        LOGGER.error(f"Market commentary failed: {e}")
        return {
            "ok": False,
            "error": str(e),
        }


def get_status() -> Dict[str, Any]:
    """Get GPT-4 Analyst status."""
    return {
        "enabled": ENABLE_GPT4_ANALYST,
        "configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "cache_size": len(_analysis_cache),
        "cache_ttl": _cache_ttl,
    }


# Sync wrapper for non-async contexts
def analyze_prediction_sync(
    symbol: str,
    direction: str,
    confidence: float,
    current_price: float,
    target_price: float,
    features: Dict[str, Any] = None,
    market_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper for analyze_prediction."""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    analyze_prediction(
                        symbol, direction, confidence, current_price, target_price,
                        features, market_context
                    )
                )
                return future.result(timeout=35)
        else:
            return loop.run_until_complete(
                analyze_prediction(
                    symbol, direction, confidence, current_price, target_price,
                    features, market_context
                )
            )
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol,
        }
