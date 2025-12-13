"""
Pre-Market Predictor
Priority 3: Generate predictions 24-48h BEFORE market opens

Strategy:
1. Run at 7:00 AM CT (before 9:30 AM market open)
2. Analyze overnight news, futures, crypto markets
3. Generate predictions for top movers BEFORE trading starts
4. Send Cash-App style alerts with "EARLY SIGNAL" tag
5. Track which predictions were made 24-48h before events

This gives Ghost users a 2.5 hour head start before market opens.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

LOGGER = logging.getLogger(__name__)

# Configuration
PREMARKET_RUN_HOUR_CT = int(os.getenv("PREMARKET_RUN_HOUR_CT", "7"))  # 7:00 AM CT
PREMARKET_ENABLED = os.getenv("PREMARKET_ENABLED", "1") == "1"
PREMARKET_SYMBOLS = os.getenv("PREMARKET_SYMBOLS", "WOLF,NVDA,TSLA,AAPL,SPY").split(",")

# State tracking
_LAST_PREMARKET_RUN = 0
_PREMARKET_PREDICTIONS: list[dict[str, Any]] = []


def should_run_premarket() -> tuple[bool, str]:
    """
    Check if it's time to run pre-market predictions
    
    Returns:
        (should_run, reason)
    """
    if not PREMARKET_ENABLED:
        return False, "Pre-market predictor disabled"
    
    # Get current time in CT
    from pytz import timezone
    ct_tz = timezone("America/Chicago")
    now_ct = datetime.now(ct_tz)
    
    # Only run on weekdays (Mon-Fri)
    if now_ct.weekday() >= 5:
        return False, f"Weekend (day {now_ct.weekday()})"
    
    # Check if it's the right hour (7:00 AM CT)
    current_hour = now_ct.hour
    if current_hour != PREMARKET_RUN_HOUR_CT:
        return False, f"Wrong hour (current: {current_hour}, target: {PREMARKET_RUN_HOUR_CT})"
    
    # Check if we already ran today
    global _LAST_PREMARKET_RUN
    last_run_date = datetime.fromtimestamp(_LAST_PREMARKET_RUN, tz=ct_tz).date() if _LAST_PREMARKET_RUN > 0 else None
    today = now_ct.date()
    
    if last_run_date == today:
        return False, f"Already ran today ({today})"
    
    return True, "Ready to run"


async def run_premarket_predictions() -> dict[str, Any]:
    """
    Generate pre-market predictions for top symbols
    
    Returns:
        {
            'run_at': timestamp,
            'symbols': ['WOLF', 'NVDA', ...],
            'predictions': [...],
            'alerts_sent': 3,
            'early_signals': ['WOLF +5% UP', ...]
        }
    """
    global _LAST_PREMARKET_RUN, _PREMARKET_PREDICTIONS
    
    from pytz import timezone
    ct_tz = timezone("America/Chicago")
    now_ct = datetime.now(ct_tz)
    run_at = time.time()
    
    LOGGER.info(f"🌅 Pre-market predictor starting at {now_ct.strftime('%I:%M %p %Z')}")
    
    predictions = []
    alerts_sent = 0
    early_signals = []
    
    for symbol in PREMARKET_SYMBOLS:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        
        try:
            # Generate prediction (will be imported from wolf_app.py)
            prediction = await _generate_premarket_prediction(symbol)
            
            if prediction:
                predictions.append(prediction)
                
                # Tag as early signal
                hours_before_open = (9.5 - now_ct.hour) + (30 / 60)  # Hours until 9:30 AM
                prediction['early_signal'] = True
                prediction['hours_before_open'] = hours_before_open
                
                # Send alert if significant
                if prediction.get('confidence', 0) >= 0.70:
                    direction = prediction.get('direction', 'FLAT')
                    confidence = int(prediction.get('confidence', 0) * 100)
                    
                    early_signal_text = f"{symbol} {direction} ({confidence}% confidence) — {hours_before_open:.1f}h before market open"
                    early_signals.append(early_signal_text)
                    
                    # Send Cash-App style alert with EARLY SIGNAL tag
                    await _send_premarket_alert(symbol, prediction, hours_before_open)
                    alerts_sent += 1
                    
                    LOGGER.info(f"🎯 Early signal: {early_signal_text}")
        
        except Exception as e:
            LOGGER.error(f"Pre-market prediction failed for {symbol}: {e}")
    
    _LAST_PREMARKET_RUN = run_at
    _PREMARKET_PREDICTIONS = predictions
    
    result = {
        'run_at': run_at,
        'run_at_ct': now_ct.strftime('%I:%M %p %Z'),
        'symbols': PREMARKET_SYMBOLS,
        'predictions': predictions,
        'alerts_sent': alerts_sent,
        'early_signals': early_signals
    }
    
    LOGGER.info(
        f"🌅 Pre-market complete: {len(predictions)} predictions, "
        f"{alerts_sent} alerts, {len(early_signals)} early signals"
    )
    
    return result


async def _generate_premarket_prediction(symbol: str) -> dict[str, Any] | None:
    """
    Generate prediction for a single symbol
    Uses existing prediction engine but adds pre-market context
    """
    try:
        # Import prediction function and asyncio
        import asyncio
        from wolf_app import run_prediction
        
        # FIXED: Run blocking prediction function in thread pool
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, run_prediction, symbol)
        
        if prediction and prediction.get('ok'):
            # Add pre-market metadata
            prediction['premarket'] = True
            prediction['premarket_factors'] = await _analyze_premarket_factors(symbol)
            
            return prediction
        
        return None
    
    except Exception as e:
        LOGGER.error(f"Prediction generation failed for {symbol}: {e}")
        return None


async def _analyze_premarket_factors(symbol: str) -> dict[str, Any]:
    """
    Analyze pre-market specific factors
    
    Returns:
        {
            'overnight_news': {...},
            'futures_trend': 'up',
            'crypto_sentiment': 0.65,
            'earnings_today': False
        }
    """
    factors = {
        'overnight_news': None,
        'futures_trend': 'neutral',
        'crypto_sentiment': 0.5,
        'earnings_today': False
    }
    
    try:
        # Check overnight news (from wolf_app get_wolf_news)
        from wolf_app import get_wolf_news
        news = get_wolf_news(limit=5)
        if news:
            factors['overnight_news'] = news.get('news_signal', {})
    
    except Exception as e:
        LOGGER.debug(f"Overnight news check failed: {e}")
    
    # Future enhancements (optional):
    # - Futures analysis: SPY/QQQ pre-market futures (requires futures data feed)
    # - Crypto sentiment: BTC/ETH overnight moves (available via crypto module)
    # - Earnings calendar: Pre-earnings prediction adjustments (now available via economic_calendar.py)
    
    return factors


async def _send_premarket_alert(symbol: str, prediction: dict[str, Any], hours_before_open: float):
    """
    Send Cash-App style alert for pre-market prediction
    
    Format:
        🌅 EARLY SIGNAL
        WOLF predicted UP
        Confidence: 78%
        2.5h before market open
    """
    try:
        from core.telegram_alerts import send_mover_alert
        
        direction = prediction.get('direction', 'FLAT')
        confidence = prediction.get('confidence', 0)
        
        # Send alert with special pre-market formatting
        send_mover_alert(
            symbol=symbol,
            market="stock",
            current_price=prediction.get('current_price', 0),
            change_pct=0,  # No change yet (market closed)
            volume=0,
            volume_avg=0,
            tier="PREMARKET",
            provider="ghost-predictor"
        )
        
        LOGGER.info(f"Pre-market alert sent for {symbol}: {direction} ({confidence:.0%})")
    
    except Exception as e:
        LOGGER.error(f"Failed to send pre-market alert for {symbol}: {e}")


def get_premarket_status() -> dict[str, Any]:
    """
    Get pre-market predictor status
    
    Returns:
        {
            'enabled': True,
            'last_run': 1731654000,
            'predictions_count': 5,
            'next_run_ct': '7:00 AM CT tomorrow'
        }
    """
    from pytz import timezone
    ct_tz = timezone("America/Chicago")
    now_ct = datetime.now(ct_tz)
    
    # Calculate next run time
    next_run = now_ct.replace(hour=PREMARKET_RUN_HOUR_CT, minute=0, second=0, microsecond=0)
    if now_ct.hour >= PREMARKET_RUN_HOUR_CT:
        next_run += timedelta(days=1)
    
    # Skip weekends
    while next_run.weekday() >= 5:
        next_run += timedelta(days=1)
    
    return {
        'enabled': PREMARKET_ENABLED,
        'run_hour_ct': PREMARKET_RUN_HOUR_CT,
        'last_run': _LAST_PREMARKET_RUN,
        'last_run_ct': datetime.fromtimestamp(_LAST_PREMARKET_RUN, tz=ct_tz).strftime('%I:%M %p %Z %Y-%m-%d') if _LAST_PREMARKET_RUN > 0 else 'Never',
        'predictions_count': len(_PREMARKET_PREDICTIONS),
        'recent_predictions': _PREMARKET_PREDICTIONS[-5:] if _PREMARKET_PREDICTIONS else [],
        'next_run_ct': next_run.strftime('%I:%M %p %Z %Y-%m-%d'),
        'symbols': PREMARKET_SYMBOLS
    }
