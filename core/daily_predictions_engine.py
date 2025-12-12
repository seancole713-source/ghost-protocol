"""
🎯 DAILY PREDICTIONS ENGINE
Generates 5 high-confidence daily picks every morning at 6:00 AM CT
Multi-factor scoring: Technical + Sentiment + Momentum + Macro + Risk
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

LOGGER = logging.getLogger(__name__)

CHICAGO_TZ = ZoneInfo("America/Chicago")

# ============================================================================
# CONFIGURATION
# ============================================================================

DAILY_PICKS_COUNT = int(os.getenv("DAILY_PICKS_COUNT", "5"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "60"))  # 60%
MIN_EXPECTED_GAIN = float(os.getenv("MIN_EXPECTED_GAIN", "8"))  # 8%
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "1000000"))  # $1M volume
STOCK_CRYPTO_MIX = os.getenv("STOCK_CRYPTO_MIX", "3:2")  # 3 stocks, 2 crypto


# ============================================================================
# MULTI-FACTOR SCORING ENGINE
# ============================================================================

async def calculate_technical_score(symbol: str, asset_type: str) -> float:
    """
    Technical analysis score (0-100)
    RSI, MACD, Moving Averages, Volume, Chart Patterns
    """
    try:
        from core.providers.turbo_provider import get_turbo_provider
        turbo = get_turbo_provider()
        
        data = await turbo.get_price_async(symbol)
        if not data:
            return 0.0
        
        score = 50.0  # Neutral baseline
        
        # RSI (30-70 range is tradeable)
        rsi = data.get("rsi", 50)
        if 30 <= rsi <= 45:  # Oversold but not extreme
            score += 15
        elif 55 <= rsi <= 70:  # Overbought but momentum strong
            score += 10
        elif rsi > 80 or rsi < 20:  # Extreme - reduce score
            score -= 20
        
        # MACD crossover
        macd = data.get("macd", {})
        if macd.get("signal") == "bullish":
            score += 15
        elif macd.get("signal") == "bearish":
            score -= 10
        
        # Price vs moving averages
        price = data.get("price", 0)
        sma20 = data.get("sma20", price)
        sma50 = data.get("sma50", price)
        
        if price > sma20 > sma50:  # Bullish alignment
            score += 10
        elif price < sma20 < sma50:  # Bearish alignment
            score -= 10
        
        # Volume trend
        volume = data.get("volume", 0)
        avg_volume = data.get("avg_volume", 1)
        if volume > avg_volume * 2:  # Strong volume
            score += 10
        
        return min(100, max(0, score))
        
    except Exception as e:
        LOGGER.error(f"Technical score failed for {symbol}: {e}")
        return 50.0


async def calculate_sentiment_score(symbol: str) -> float:
    """
    Sentiment analysis score (0-100)
    News + Social + Options Flow
    """
    try:
        from core.sentiment_fusion import get_aggregated_sentiment
        
        sentiment_data = await get_aggregated_sentiment(symbol)
        
        # News sentiment
        news_score = sentiment_data.get("news_sentiment", 0.0) * 30  # 0-30 points
        
        # Social sentiment (Reddit, Twitter)
        social_score = sentiment_data.get("social_sentiment", 0.0) * 25  # 0-25 points
        
        # Options flow (bullish/bearish)
        options_score = sentiment_data.get("options_sentiment", 0.0) * 25  # 0-25 points
        
        # Insider trading
        insider_score = sentiment_data.get("insider_sentiment", 0.0) * 20  # 0-20 points
        
        total = 50 + news_score + social_score + options_score + insider_score
        return min(100, max(0, total))
        
    except Exception as e:
        LOGGER.error(f"Sentiment score failed for {symbol}: {e}")
        return 50.0


async def calculate_momentum_score(symbol: str) -> float:
    """
    Momentum score (0-100)
    Price velocity, breakout strength, trend acceleration
    """
    try:
        from core.providers.turbo_provider import get_turbo_provider
        turbo = get_turbo_provider()
        
        data = await turbo.get_price_async(symbol)
        if not data:
            return 50.0
        
        score = 50.0
        
        # 24hr change
        change_24h = data.get("change_pct_24h", 0)
        if change_24h > 10:
            score += 20
        elif change_24h > 5:
            score += 10
        elif change_24h < -10:
            score -= 20
        
        # 7-day trend
        change_7d = data.get("change_pct_7d", 0)
        if change_7d > 20:
            score += 15
        elif change_7d > 10:
            score += 10
        
        # Volume acceleration
        volume = data.get("volume", 0)
        avg_volume = data.get("avg_volume", 1)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 5:
            score += 15
        elif volume_ratio > 2:
            score += 10
        
        return min(100, max(0, score))
        
    except Exception as e:
        LOGGER.error(f"Momentum score failed for {symbol}: {e}")
        return 50.0


async def calculate_market_regime_score(symbol: str, asset_type: str) -> float:
    """
    Market regime alignment score (0-100)
    Checks if asset aligns with broader market trend
    """
    try:
        from core.market_regime import get_current_regime
        
        regime = await get_current_regime()
        
        # If market is bullish and we're going long, boost score
        if regime["trend"] == "bullish" and asset_type == "stock":
            return 80.0
        elif regime["trend"] == "bearish":
            return 30.0  # Reduce score in bear market
        elif regime["trend"] == "crash":
            return 10.0  # Very low score during crash
        else:
            return 50.0  # Neutral
            
    except Exception as e:
        LOGGER.error(f"Market regime score failed: {e}")
        return 50.0


async def calculate_timing_score(symbol: str) -> float:
    """
    Entry timing score (0-100)
    Checks if this is a good time to enter (avoid chop zones, earnings, etc.)
    """
    try:
        from core.earnings_calendar import check_earnings_proximity
        
        # Check if earnings within 24hrs
        has_earnings_soon = await check_earnings_proximity(symbol, hours=24)
        if has_earnings_soon:
            return 20.0  # Low score, risky timing
        
        # Check if market hours vs after-hours
        now = datetime.now(CHICAGO_TZ)
        if 8 <= now.hour < 15:  # Market hours
            return 80.0
        else:
            return 60.0  # After hours, slightly lower
            
    except Exception as e:
        LOGGER.error(f"Timing score failed for {symbol}: {e}")
        return 50.0


async def calculate_confidence_score(symbol: str, asset_type: str) -> dict[str, Any]:
    """
    Multi-factor confidence calculation
    Returns confidence score + component scores
    """
    # Calculate all factors in parallel
    results = await asyncio.gather(
        calculate_technical_score(symbol, asset_type),
        calculate_sentiment_score(symbol),
        calculate_momentum_score(symbol),
        calculate_market_regime_score(symbol, asset_type),
        calculate_timing_score(symbol),
        return_exceptions=True
    )
    
    technical = results[0] if not isinstance(results[0], Exception) else 50.0
    sentiment = results[1] if not isinstance(results[1], Exception) else 50.0
    momentum = results[2] if not isinstance(results[2], Exception) else 50.0
    regime = results[3] if not isinstance(results[3], Exception) else 50.0
    timing = results[4] if not isinstance(results[4], Exception) else 50.0
    
    # Weighted average
    confidence = (
        technical * 0.25 +
        sentiment * 0.20 +
        momentum * 0.20 +
        regime * 0.15 +
        timing * 0.10 +
        50 * 0.10  # Volatility score (placeholder)
    )
    
    return {
        "confidence": round(confidence, 1),
        "technical": round(technical, 1),
        "sentiment": round(sentiment, 1),
        "momentum": round(momentum, 1),
        "regime": round(regime, 1),
        "timing": round(timing, 1)
    }


# ============================================================================
# EXPECTED GAIN CALCULATOR
# ============================================================================

async def calculate_expected_gain(symbol: str, confidence: float) -> dict[str, Any]:
    """
    Calculate expected gain, entry, target, peak, stop prices
    """
    try:
        from core.providers.turbo_provider import get_turbo_provider
        turbo = get_turbo_provider()
        
        data = await turbo.get_price_async(symbol)
        if not data:
            return None
        
        current_price = data.get("price", 0)
        if current_price == 0:
            return None
        
        # Historical volatility
        volatility = data.get("volatility", 0.02)  # 2% default
        
        # Base expected move (based on momentum)
        momentum_pct = data.get("change_pct_24h", 0)
        base_gain = abs(momentum_pct) * 1.5  # Expect 1.5x continuation
        
        # Adjust by confidence
        expected_gain_pct = base_gain * (confidence / 100)
        expected_gain_pct = max(MIN_EXPECTED_GAIN, expected_gain_pct)  # Min 8%
        
        # Calculate price targets
        entry_low = current_price * 0.995  # 0.5% buffer
        entry_high = current_price * 1.005
        
        target_price = current_price * (1 + expected_gain_pct * 0.75 / 100)  # Conservative
        peak_price = current_price * (1 + expected_gain_pct * 1.3 / 100)  # Optimistic
        stop_price = current_price * (1 - min(0.08, volatility * 2))  # 8% max stop
        
        return {
            "expected_gain_pct": round(expected_gain_pct, 1),
            "current_price": round(current_price, 2),
            "entry_low": round(entry_low, 2),
            "entry_high": round(entry_high, 2),
            "target": round(target_price, 2),
            "peak": round(peak_price, 2),
            "stop": round(stop_price, 2)
        }
        
    except Exception as e:
        LOGGER.error(f"Expected gain calculation failed for {symbol}: {e}")
        return None


# ============================================================================
# CANDIDATE FILTERING & RANKING
# ============================================================================

async def scan_and_score_candidates(symbols: list[str]) -> list[dict[str, Any]]:
    """
    Scan all symbols, calculate scores, filter by requirements
    """
    candidates = []
    
    LOGGER.info(f"🔍 Scanning {len(symbols)} candidates for daily picks...")
    
    for symbol in symbols:
        try:
            # Determine asset type
            asset_type = "crypto" if len(symbol) <= 5 and symbol.isupper() else "stock"
            
            # Calculate confidence
            score_data = await calculate_confidence_score(symbol, asset_type)
            confidence = score_data["confidence"]
            
            # Skip if below minimum confidence
            if confidence < MIN_CONFIDENCE:
                continue
            
            # Calculate expected gain and prices
            gain_data = await calculate_expected_gain(symbol, confidence)
            if not gain_data:
                continue
            
            expected_gain = gain_data["expected_gain_pct"]
            
            # Skip if below minimum expected gain
            if expected_gain < MIN_EXPECTED_GAIN:
                continue
            
            # Check liquidity (volume * price > MIN_LIQUIDITY)
            from core.providers.turbo_provider import get_turbo_provider
            turbo = get_turbo_provider()
            data = await turbo.get_price_async(symbol)
            
            if data:
                volume = data.get("volume", 0)
                price = data.get("price", 0)
                liquidity = volume * price
                
                if liquidity < MIN_LIQUIDITY:
                    continue
            
            # Calculate risk-adjusted rank
            risk_factor = 1 + (100 - confidence) / 100  # Higher confidence = lower risk
            rank_score = (confidence * expected_gain) / risk_factor
            
            candidates.append({
                "symbol": symbol,
                "asset_type": asset_type,
                "confidence": confidence,
                "expected_gain": expected_gain,
                "rank_score": rank_score,
                "prices": gain_data,
                "score_breakdown": score_data
            })
            
        except Exception as e:
            LOGGER.error(f"Failed to score {symbol}: {e}")
            continue
    
    # Sort by rank_score (highest first)
    candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    
    LOGGER.info(f"✅ Found {len(candidates)} qualified candidates")
    
    return candidates


async def select_daily_picks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Select top 5 picks with proper stock/crypto mix
    """
    # Parse mix ratio (e.g., "3:2" = 3 stocks, 2 crypto)
    stock_count, crypto_count = map(int, STOCK_CRYPTO_MIX.split(":"))
    
    stocks = [c for c in candidates if c["asset_type"] == "stock"]
    cryptos = [c for c in candidates if c["asset_type"] == "crypto"]
    
    # Select top picks
    selected_stocks = stocks[:stock_count]
    selected_cryptos = cryptos[:crypto_count]
    
    picks = selected_stocks + selected_cryptos
    
    # Re-rank combined list
    picks.sort(key=lambda x: x["rank_score"], reverse=True)
    
    return picks[:DAILY_PICKS_COUNT]


# ============================================================================
# DAILY BRIEFING GENERATOR
# ============================================================================

async def generate_daily_briefing() -> dict[str, Any]:
    """
    Main function: Generate daily briefing with 5 picks
    """
    start_time = time.time()
    
    LOGGER.info("🌅 Generating daily briefing...")
    
    try:
        # Get all symbols to scan
        from core.beast_scheduler import STOCK_SYMBOLS, CRYPTO_SYMBOLS
        from core.spike_detector import get_all_tracked_symbols
        
        all_symbols = await get_all_tracked_symbols(STOCK_SYMBOLS + CRYPTO_SYMBOLS)
        
        # Scan and score
        candidates = await scan_and_score_candidates(all_symbols)
        
        # Select top 5
        picks = await select_daily_picks(candidates)
        
        # Get market context
        from core.market_regime import get_current_regime
        regime = await get_current_regime()
        
        elapsed = time.time() - start_time
        
        briefing = {
            "timestamp": int(time.time()),
            "date": datetime.now(CHICAGO_TZ).strftime("%B %d, %Y"),
            "picks": picks,
            "market_context": regime,
            "candidates_scanned": len(all_symbols),
            "candidates_qualified": len(candidates),
            "generation_time": round(elapsed, 1)
        }
        
        LOGGER.info(f"✅ Daily briefing generated in {elapsed:.1f}s - {len(picks)} picks selected")
        
        return briefing
        
    except Exception as e:
        LOGGER.error(f"❌ Daily briefing generation failed: {e}", exc_info=True)
        return None


async def send_daily_briefing_to_telegram(briefing: dict[str, Any]):
    """
    Format and send daily briefing to Telegram
    """
    try:
        from core.alert_manager import send_daily_briefing_alert
        await send_daily_briefing_alert(briefing)
        LOGGER.info("📤 Daily briefing sent to Telegram")
    except Exception as e:
        LOGGER.error(f"Failed to send daily briefing: {e}", exc_info=True)


# ============================================================================
# SCHEDULED DAILY EXECUTION
# ============================================================================

async def daily_briefing_scheduler():
    """
    Runs every day at 6:00 AM CT
    """
    LOGGER.info("🕒 Daily briefing scheduler started")
    
    while True:
        try:
            now = datetime.now(CHICAGO_TZ)
            
            # Check if it's 6:00 AM CT
            if now.hour == 6 and now.minute == 0:
                briefing = await generate_daily_briefing()
                
                if briefing:
                    await send_daily_briefing_to_telegram(briefing)
                
                # Sleep for 60 seconds to avoid running multiple times in same minute
                await asyncio.sleep(60)
            
            # Check every 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            LOGGER.error(f"Daily briefing scheduler error: {e}", exc_info=True)
            await asyncio.sleep(300)  # 5 min cooldown on error
