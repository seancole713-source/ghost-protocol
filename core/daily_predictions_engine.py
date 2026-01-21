"""
🌅 DAILY PREDICTIONS ENGINE V2 - GHOST INFRASTRUCTURE
Generates autonomous daily briefing at 6:00 AM CT with 5 top picks
Uses actual Ghost prediction system + data pillars
"""

import asyncio
import fnmatch
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOGGER = logging.getLogger(__name__)
CHICAGO_TZ = ZoneInfo("America/Chicago")

# =============================================================================
# Prediction Controls (from environment)
# =============================================================================
# Master switch to enable/disable stock predictions
PREDICT_STOCKS_ENABLED = os.getenv("PREDICT_STOCKS_ENABLED", "1").lower() in ("1", "true", "yes")

# Comma-separated list of allowed stocks (supports wildcards)
# e.g., "AAPL,MSFT,*" means AAPL, MSFT, and all others
# e.g., "WOLF,AAPL" means only WOLF and AAPL
PREDICT_STOCKS_ALLOW = os.getenv("PREDICT_STOCKS_ALLOW", "*")  # Default: allow all


def _is_stock_allowed(symbol: str) -> bool:
    """Check if a stock symbol is allowed for prediction based on PREDICT_STOCKS_ALLOW"""
    if not PREDICT_STOCKS_ENABLED:
        return False
    
    allow_list = [s.strip().upper() for s in PREDICT_STOCKS_ALLOW.split(",") if s.strip()]
    symbol_upper = symbol.upper()
    
    for pattern in allow_list:
        # Support wildcard matching
        if pattern == "*":
            return True
        if fnmatch.fnmatch(symbol_upper, pattern):
            return True
    
    return False


# Configuration
DAILY_PICKS_COUNT = int(os.getenv("DAILY_PICKS_COUNT", "5"))

# Confidence gating: Ghost core returns confidence as 0.0-1.0.
# Allow env var to be specified either as 0.70 or 70.
_min_conf_raw = float(os.getenv("MIN_CONFIDENCE_PCT", os.getenv("MIN_CONFIDENCE", "70")))
MIN_CONFIDENCE = _min_conf_raw / 100.0 if _min_conf_raw > 1.0 else _min_conf_raw
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
    
    # Filter symbols based on PREDICT_STOCKS_ENABLED and PREDICT_STOCKS_ALLOW
    if asset_type == "stock":
        allowed_symbols = [s for s in symbols if _is_stock_allowed(s)]
        if len(allowed_symbols) < len(symbols):
            LOGGER.info(f"📋 Filtered {len(symbols) - len(allowed_symbols)} stocks by PREDICT_STOCKS_ALLOW")
        symbols = allowed_symbols
    
    if not symbols:
        LOGGER.info(f"No {asset_type} symbols to predict (empty or all filtered)")
        return []
    
    # Run predictions with concurrency limit (match Ghost's auto_prediction_loop: 2 concurrent)
    semaphore = asyncio.Semaphore(2)
    
    async def _predict(symbol: str):
        async with semaphore:
            try:
                result = await RUN_PREDICTION_FUNC_ASYNC(symbol)
                
                if result.get("ok") and float(result.get("confidence", 0) or 0) >= MIN_CONFIDENCE:
                    # Extract core fields from Ghost's prediction result
                    confidence = float(result.get("confidence", 0) or 0)
                    expected_move_pct = result.get("expected_move_pct")
                    try:
                        expected_move_pct_f = None if expected_move_pct is None else float(expected_move_pct)
                    except Exception:
                        expected_move_pct_f = None

                    predictions.append({
                        "symbol": symbol,
                        "asset_type": asset_type,
                        "confidence": confidence,
                        "signal": result.get("direction", "UNKNOWN"),  # "UP" or "DOWN"
                        "current_price": result.get("current_price"),
                        "prediction_id": result.get("prediction_id"),
                        "feature_count": result.get("feature_count", 0),
                        "duration_ms": result.get("duration_ms", 0),
                        # Prefer model-derived expected move when available
                        "expected_gain": expected_move_pct_f if expected_move_pct_f is not None else _calculate_expected_gain(result),
                        "target_price": result.get("target_price") if result.get("target_price") is not None else _calculate_target(result),
                        "stop_loss": result.get("stop_loss") if result.get("stop_loss") is not None else _calculate_stop(result),
                        "stage5_ok": bool(result.get("stage5_ok", False)),
                        "stage6_ok": bool(result.get("stage6_ok", False)),
                        "gate": result.get("gate", "MONITOR"),
                        "timestamp": datetime.now(CHICAGO_TZ).isoformat()
                    })
                    LOGGER.info(f"✅ [{symbol}] Confidence: {confidence:.0%} Signal: {result.get('direction')}")
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
    confidence = float(result.get("confidence", 0) or 0)
    direction = result.get("direction", "")
    
    # Expected gain scales with confidence (0.0-1.0)
    # 0.70 -> ~4%, 0.85 -> ~7%, 0.95 -> ~9%
    base_gain = max(0.0, (confidence - 0.60)) * 30.0
    
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
    # Enforce the same 70% gate used for Telegram alerts.
    filtered = [p for p in predictions if p["confidence"] >= MIN_CONFIDENCE and bool(p.get("stage5_ok", True))]
    
    # Rank by combined score (confidence * 0.6 + expected_gain * 0.4)
    for p in filtered:
        # Score: prioritize gated confidence, then expected move magnitude.
        p["score"] = (p["confidence"] * 100.0) * 0.75 + abs(float(p.get("expected_gain") or 0.0)) * 0.25
    
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
    
    msg = "🌅 DAILY MARKET BRIEFING\n"
    msg += f"📅 {timestamp}\n"
    msg += f"📊 Evaluated {stats.get('total_evaluated', 0)} symbols\n\n"
    
    msg += "🎯 TOP PICKS (>=70%)\n"
    
    for i, pick in enumerate(picks["picks"], 1):
        is_last = i == len(picks["picks"])
        prefix = "└─" if is_last else "├─"
        
        signal_emoji = "🚀" if pick['signal'] == "UP" else "📉"
        
        msg += f"{prefix} {pick['symbol']} ({pick['asset_type'].upper()}) {signal_emoji}\n"
        msg += f"{'   ' if is_last else '│  '}├─ Signal: {pick['signal']}\n"
        msg += f"{'   ' if is_last else '│  '}├─ Confidence: {pick['confidence']:.0%}\n"
        msg += f"{'   ' if is_last else '│  '}├─ Gate: {pick.get('gate','MONITOR')}\n"
        
        if pick.get('current_price'):
            msg += f"{'   ' if is_last else '│  '}├─ Current: ${pick['current_price']:.2f}\n"
        if pick.get('target_price'):
            msg += f"{'   ' if is_last else '│  '}├─ Target: ${pick['target_price']:.2f}\n"
        if pick.get('stop_loss'):
            msg += f"{'   ' if is_last else '│  '}├─ Stop: ${pick['stop_loss']:.2f}\n"
        
        msg += f"{'   ' if is_last else '│  '}├─ Expected: {float(pick.get('expected_gain') or 0.0):+.1f}%\n"
        msg += f"{'   ' if is_last else '│  '}└─ Features: {pick.get('feature_count', 0)} indicators\n"
        
        if not is_last:
            msg += "│\n"
    
    msg += f"\n📈 Avg Confidence: {float(stats.get('avg_confidence', 0) or 0):.0%}\n"
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
                
                # Send via configured Telegram sender
                try:
                    from core import telegram_alerts

                    if telegram_alerts.TELEGRAM_SEND_FUNC and telegram_alerts.TELEGRAM_CHAT_ID:
                        telegram_alerts.TELEGRAM_SEND_FUNC(telegram_alerts.TELEGRAM_CHAT_ID, briefing)
                    else:
                        LOGGER.warning("Telegram not configured for daily briefing")
                except Exception as e:
                    LOGGER.warning(f"Daily briefing send failed: {e}")
                
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
