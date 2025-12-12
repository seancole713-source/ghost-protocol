"""
🌅 DAILY PREDICTIONS ENGINE V2 - GHOST INFRASTRUCTURE
Generates autonomous daily briefing at 6:00 AM CT with 5 top picks
Uses actual Ghost prediction system + data pillars
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOGGER = logging.getLogger(__name__)
CHICAGO_TZ = ZoneInfo("America/Chicago")

# Configuration
DAILY_PICKS_COUNT = int(os.getenv("DAILY_PICKS_COUNT", "5"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE_PCT", "60"))
STOCK_CRYPTO_MIX = os.getenv("STOCK_CRYPTO_MIX", "3:2")  # 3 stocks, 2 crypto

# Will be injected by orchestrator
RUN_PREDICTION_FUNC_ASYNC = None
HUNTER_STOCK_SYMBOLS = []
HUNTER_CRYPTO_SYMBOLS = []


async def generate_daily_picks() -> dict[str, Any]:
    """
    Generate 5 daily picks using Ghost's actual prediction system
    Returns: {"picks": [...], "timestamp": "...", "stats": {...}}
    """
    try:
        LOGGER.info("🌅 Generating daily picks at 6:00 AM CT...")
        
        if not RUN_PREDICTION_FUNC_ASYNC:
            LOGGER.error("❌ RUN_PREDICTION_FUNC_ASYNC not injected")
            return {"picks": [], "error": "Prediction function not available"}
        
        # Parse stock/crypto mix
        stock_count, crypto_count = map(int, STOCK_CRYPTO_MIX.split(":"))
        
        # Run predictions on all watchlist symbols
        stock_predictions = await _batch_predictions(HUNTER_STOCK_SYMBOLS[:50], "stock")
        crypto_predictions = await _batch_predictions(HUNTER_CRYPTO_SYMBOLS[:20], "crypto")
        
        # Filter and rank
        stock_picks = _filter_and_rank(stock_predictions, stock_count)
        crypto_picks = _filter_and_rank(crypto_predictions, crypto_count)
        
        all_picks = stock_picks + crypto_picks
        
        LOGGER.info(f"✅ Generated {len(all_picks)} daily picks")
        
        return {
            "picks": all_picks,
            "timestamp": datetime.now(CHICAGO_TZ).isoformat(),
            "stats": {
                "total_evaluated": len(stock_predictions) + len(crypto_predictions),
                "stocks_selected": len(stock_picks),
                "crypto_selected": len(crypto_picks),
                "avg_confidence": sum(p["confidence"] for p in all_picks) / len(all_picks) if all_picks else 0
            }
        }
    
    except Exception as e:
        LOGGER.error(f"❌ Daily picks generation failed: {e}", exc_info=True)
        return {"picks": [], "error": str(e)}


async def _batch_predictions(symbols: list[str], asset_type: str) -> list[dict[str, Any]]:
    """
    Run predictions on batch of symbols using Ghost's prediction system
    
    Ghost's run_single_prediction_async returns:
    {
        "ok": bool,
        "prediction_id": int,
        "symbol": str,
        "direction": str,  # "UP" or "DOWN"
        "confidence": float,  # 0-100
        "current_price": float,
        "feature_count": int,
        "available_count": int,
        "duration_ms": int,
        "error": str or None
    }
    """
    predictions = []
    
    # Run predictions with concurrency limit (match Ghost's auto_prediction_loop: 2 concurrent)
    semaphore = asyncio.Semaphore(2)
    
    async def _predict(symbol: str):
        async with semaphore:
            try:
                result = await RUN_PREDICTION_FUNC_ASYNC(symbol)
                
                if result.get("ok") and result.get("confidence", 0) >= MIN_CONFIDENCE:
                    # Extract core fields from Ghost's prediction result
                    predictions.append({
                        "symbol": symbol,
                        "asset_type": asset_type,
                        "confidence": result.get("confidence", 0),
                        "signal": result.get("direction", "UNKNOWN"),  # "UP" or "DOWN"
                        "current_price": result.get("current_price"),
                        "prediction_id": result.get("prediction_id"),
                        "feature_count": result.get("feature_count", 0),
                        "duration_ms": result.get("duration_ms", 0),
                        # Calculate target/stop based on direction and confidence
                        "expected_gain": _calculate_expected_gain(result),
                        "target_price": _calculate_target(result),
                        "stop_loss": _calculate_stop(result),
                        "timestamp": datetime.now(CHICAGO_TZ).isoformat()
                    })
                    LOGGER.info(f"✅ [{symbol}] Confidence: {result.get('confidence', 0):.1f}% Signal: {result.get('direction')}")
            except Exception as e:
                LOGGER.warning(f"[{symbol}] Prediction failed: {e}")
    
    await asyncio.gather(*[_predict(s) for s in symbols], return_exceptions=True)
    
    LOGGER.info(f"Completed {len(predictions)}/{len(symbols)} predictions for {asset_type}")
    return predictions


def _calculate_expected_gain(result: dict[str, Any]) -> float:
    """
    Calculate expected gain % based on confidence and direction
    Ghost's confidence is 0-100, higher = stronger signal
    """
    confidence = result.get("confidence", 0)
    direction = result.get("direction", "")
    
    # Expected gain scales with confidence
    # 60% confidence = 5% gain, 80% = 10% gain, 95% = 15% gain
    base_gain = (confidence - 50) / 5  # 60->2%, 80->6%, 95->9%
    
    if direction == "DOWN":
        return -base_gain  # Negative for short positions
    return base_gain


def _calculate_target(result: dict[str, Any]) -> Optional[float]:
    """Calculate target price based on expected gain"""
    price = result.get("current_price")
    if not price:
        return None
    
    gain_pct = _calculate_expected_gain(result)
    return price * (1 + gain_pct / 100)


def _calculate_stop(result: dict[str, Any]) -> Optional[float]:
    """Calculate stop loss (2:1 risk/reward ratio)"""
    price = result.get("current_price")
    if not price:
        return None
    
    gain_pct = _calculate_expected_gain(result)
    stop_pct = gain_pct / 2  # Half the gain = 2:1 ratio
    return price * (1 - abs(stop_pct) / 100)


def _filter_and_rank(predictions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """
    Filter and rank predictions by confidence and expected gain
    """
    # Filter by confidence threshold
    filtered = [p for p in predictions if p["confidence"] >= MIN_CONFIDENCE]
    
    # Rank by combined score (confidence * 0.6 + expected_gain * 0.4)
    for p in filtered:
        p["score"] = p["confidence"] * 0.6 + p["expected_gain"] * 0.4
    
    # Sort by score descending
    ranked = sorted(filtered, key=lambda x: x["score"], reverse=True)
    
    return ranked[:count]


async def format_daily_briefing(picks: dict[str, Any]) -> str:
    """
    Format daily briefing for Telegram
    Clean hierarchy with ├─ └─ tree structure
    """
    if not picks.get("picks"):
        return "❌ No high-confidence picks today. Market conditions uncertain."
    
    stats = picks.get("stats", {})
    timestamp = datetime.now(CHICAGO_TZ).strftime("%Y-%m-%d %I:%M %p CT")
    
    msg = "🌅 **DAILY MARKET BRIEFING**\n"
    msg += f"📅 {timestamp}\n"
    msg += f"📊 Evaluated {stats.get('total_evaluated', 0)} symbols\n\n"
    
    msg += "🎯 **TOP PICKS**\n"
    
    for i, pick in enumerate(picks["picks"], 1):
        is_last = i == len(picks["picks"])
        prefix = "└─" if is_last else "├─"
        
        signal_emoji = "🚀" if pick['signal'] == "UP" else "📉"
        
        msg += f"{prefix} **{pick['symbol']}** ({pick['asset_type'].upper()}) {signal_emoji}\n"
        msg += f"{'   ' if is_last else '│  '}├─ Signal: {pick['signal']}\n"
        msg += f"{'   ' if is_last else '│  '}├─ Confidence: {pick['confidence']:.1f}%\n"
        
        if pick.get('current_price'):
            msg += f"{'   ' if is_last else '│  '}├─ Current: ${pick['current_price']:.2f}\n"
        if pick.get('target_price'):
            msg += f"{'   ' if is_last else '│  '}├─ Target: ${pick['target_price']:.2f}\n"
        if pick.get('stop_loss'):
            msg += f"{'   ' if is_last else '│  '}├─ Stop: ${pick['stop_loss']:.2f}\n"
        
        msg += f"{'   ' if is_last else '│  '}├─ Expected: {pick['expected_gain']:+.1f}%\n"
        msg += f"{'   ' if is_last else '│  '}└─ Features: {pick.get('feature_count', 0)} indicators\n"
        
        if not is_last:
            msg += "│\n"
    
    msg += f"\n📈 Avg Confidence: {stats.get('avg_confidence', 0):.1f}%\n"
    msg += "⚡ Live updates every 5 minutes\n"
    
    return msg


# ============================================================================
# SCHEDULER INTEGRATION
# ============================================================================

async def daily_briefing_task():
    """
    Background task that runs at 6:00 AM CT daily
    Integrates with Ghost's auto_prediction_loop
    """
    while True:
        try:
            now = datetime.now(CHICAGO_TZ)
            
            # Check if it's 6:00 AM CT
            if now.hour == 6 and now.minute == 0:
                LOGGER.info("🌅 Daily briefing time! Generating picks...")
                
                picks = await generate_daily_picks()
                
                # Format briefing
                briefing = await format_daily_briefing(picks)
                
                # Send via existing Telegram alerts
                from core.telegram_alerts import send_alert
                await send_alert(briefing, priority="HIGH")
                
                # Sleep until next day (avoid duplicate runs in same minute)
                await asyncio.sleep(120)
            else:
                # Check every 30 seconds
                await asyncio.sleep(30)
        
        except Exception as e:
            LOGGER.error(f"❌ Daily briefing task error: {e}", exc_info=True)
            await asyncio.sleep(60)


def inject_dependencies(run_prediction_func, stock_symbols, crypto_symbols):
    """
    Inject Ghost's actual functions (called by orchestrator)
    """
    global RUN_PREDICTION_FUNC_ASYNC, HUNTER_STOCK_SYMBOLS, HUNTER_CRYPTO_SYMBOLS
    RUN_PREDICTION_FUNC_ASYNC = run_prediction_func
    HUNTER_STOCK_SYMBOLS = stock_symbols
    HUNTER_CRYPTO_SYMBOLS = crypto_symbols
    LOGGER.info("✅ Daily predictions engine initialized with Ghost infrastructure")


# ============================================================================
# MANUAL TEST ENDPOINT (for development)
# ============================================================================

async def test_daily_picks():
    """
    Manual test function for development
    """
    import sys
    sys.path.insert(0, "/workspaces/ghost-protocol")
    
    # Import actual Ghost functions
    from wolf_app import RUN_PREDICTION_FUNC_ASYNC
    from core.beast_scheduler import HUNTER_STOCK_SYMBOLS, HUNTER_CRYPTO_SYMBOLS
    
    # Inject dependencies
    inject_dependencies(RUN_PREDICTION_FUNC_ASYNC, HUNTER_STOCK_SYMBOLS, HUNTER_CRYPTO_SYMBOLS)
    
    # Generate picks
    picks = await generate_daily_picks()
    
    # Format briefing
    briefing = await format_daily_briefing(picks)
    
    print(briefing)
    
    return picks


if __name__ == "__main__":
    asyncio.run(test_daily_picks())
