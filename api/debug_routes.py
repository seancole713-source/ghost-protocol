"""
Debug API routes.

These endpoints are for development and debugging purposes.
In production, consider adding authentication or disabling.
"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter

from config.settings import settings
from config.symbols import (
    V3_VALIDATED_STRATEGIES, 
    V3_REMOVED_SYMBOLS,
    V3_BLACKLIST,
)
from core.models import Prediction, Direction
from core.v3_filter import V3Filter

router = APIRouter()


@router.get("/v3-filter-test")
async def debug_v3_filter_test(
    eth_conf: float = 0.65,
    xrp_conf: float = 0.52,
    link_conf: float = 0.45,
    sol_conf: float = 0.65,
    btc_conf: float = 0.65,
):
    """
    Test V3 filter with configurable confidence levels.
    
    Query params:
        eth_conf: ETH confidence (default 0.65)
        xrp_conf: XRP confidence (default 0.52)
        link_conf: LINK confidence (default 0.45)
        sol_conf: SOL confidence (default 0.65)
        btc_conf: BTC confidence (default 0.65)
    """
    # Build test predictions
    test_preds = [
        Prediction(
            symbol='ETH',
            direction=Direction.DOWN,
            confidence=eth_conf,
            current_price=2300.0,
            target_price=2200.0,
            stop_loss=2350.0,
            timestamp=datetime.now(),
        ),
        Prediction(
            symbol='XRP',
            direction=Direction.DOWN,
            confidence=xrp_conf,
            current_price=1.65,
            target_price=1.55,
            stop_loss=1.72,
            timestamp=datetime.now(),
        ),
        Prediction(
            symbol='LINK',
            direction=Direction.DOWN,
            confidence=link_conf,
            current_price=12.50,
            target_price=11.75,
            stop_loss=13.00,
            timestamp=datetime.now(),
        ),
        Prediction(
            symbol='SOL',
            direction=Direction.DOWN,
            confidence=sol_conf,
            current_price=120.0,
            target_price=110.0,
            stop_loss=125.0,
            timestamp=datetime.now(),
        ),
        Prediction(
            symbol='BTC',
            direction=Direction.DOWN,
            confidence=btc_conf,
            current_price=75000.0,
            target_price=72000.0,
            stop_loss=76500.0,
            timestamp=datetime.now(),
        ),
    ]
    
    # Run V3 filter
    v3_filter = V3Filter()
    filtered = v3_filter.filter_and_score(test_preds)
    
    # Build detailed report
    filter_report = {}
    for pred in test_preds:
        result = v3_filter.filter_single(pred)
        filter_report[pred.symbol] = {
            "raw_direction": pred.direction.value,
            "raw_confidence": pred.confidence,
            "in_validated": pred.symbol in V3_VALIDATED_STRATEGIES,
            "in_removed": pred.symbol in V3_REMOVED_SYMBOLS,
            "passed_filter": result.passed,
            "reason": result.reason,
        }
    
    return {
        "ok": True,
        "v3_enabled": settings.V3_ENABLED,
        "min_confidence": settings.V3_MIN_CONFIDENCE,
        "test_predictions_count": len(test_preds),
        "filtered_count": len(filtered),
        "filter_report": filter_report,
        "filtered_symbols": [p.symbol for p in filtered],
        "stats": v3_filter.stats,
        "explanation": {
            "ETH": f"Only if Ghost predicts DOWN AND conf >= {settings.V3_MIN_CONFIDENCE:.0%} (inverse to UP)",
            "XRP": f"Any direction (mean_reversion) if conf >= {settings.V3_MIN_CONFIDENCE:.0%}",
            "LINK": f"Any direction (mean_reversion) if conf >= {settings.V3_MIN_CONFIDENCE:.0%}",
            "SOL": "REMOVED: Inverse 50.2% over 4962 trades - not significant",
            "BTC": "REMOVED: Inverse 52% over large sample - not significant",
        }
    }


@router.get("/v3-validation")
async def debug_v3_validation():
    """
    V3 backtest validation status.
    
    Shows performance of backtest-validated strategies and what to expect.
    """
    validation_report = {}
    
    for symbol, strategy in V3_VALIDATED_STRATEGIES.items():
        validation_report[symbol] = {
            "strategy": strategy.strategy,
            "hold_hours": strategy.hold_hours,
            "hold_days": strategy.hold_hours // 24,
            "backtest_win_rate": strategy.backtest_win_rate,
            "backtest_p_value": strategy.p_value,
            "backtest_sample_size": strategy.backtest_trades,
            "live_win_rate": None,  # Will be filled when tracking works
            "live_resolved": 0,
            "live_pending": 0,
            "tracking": "⏳ NO DATA YET",
            "validation": "🔬 VALIDATING",
        }
    
    return {
        "ok": True,
        "v3_mode": "BACKTEST-VALIDATED",
        "min_confidence": settings.V3_MIN_CONFIDENCE,
        "default_hold_hours": settings.V3_DEFAULT_HOLD_HOURS,
        "validated_symbols": list(V3_VALIDATED_STRATEGIES.keys()),
        "removed_symbols": list(V3_REMOVED_SYMBOLS.keys()),
        "validation_report": validation_report,
        "backtest_summary": {
            "total_trades_analyzed": 52433,
            "statistically_significant_results": 8,
            "significance_threshold": "p < 0.05",
            "overall_market_efficiency": "50.0% (random walk confirmed)",
        },
        "notes": [
            "ETH ghost_inverse: Only symbol where inverting Ghost beats 50%",
            "XRP/LINK mean_reversion: Price bounces beat trend following",
            "RSI strategies: 45-46% win rate - consistently LOSE money",
            "SOL/BTC/AVAX: Removed - no statistical significance",
        ]
    }


@router.get("/config-dump")
async def debug_config_dump():
    """
    Dump all non-sensitive configuration values.
    """
    return {
        "settings": {
            "APP_NAME": settings.APP_NAME,
            "VERSION": settings.VERSION,
            "DEBUG": settings.DEBUG,
            "V3_MIN_CONFIDENCE": settings.V3_MIN_CONFIDENCE,
            "V3_DEFAULT_HOLD_HOURS": settings.V3_DEFAULT_HOLD_HOURS,
            "V3_ENABLED": settings.V3_ENABLED,
            "DEFAULT_TARGET_PCT": settings.DEFAULT_TARGET_PCT,
            "DEFAULT_STOP_PCT": settings.DEFAULT_STOP_PCT,
            "DEFAULT_RR_RATIO": settings.DEFAULT_RR_RATIO,
            "TOP10_HOUR": settings.TOP10_HOUR,
            "TOP10_MINUTE": settings.TOP10_MINUTE,
            "TIMEZONE": settings.TIMEZONE,
        },
        "price_ranges": {
            "BTC": (settings.MIN_PRICE_BTC, settings.MAX_PRICE_BTC),
            "ETH": (settings.MIN_PRICE_ETH, settings.MAX_PRICE_ETH),
            "XRP": (settings.MIN_PRICE_XRP, settings.MAX_PRICE_XRP),
            "LINK": (settings.MIN_PRICE_LINK, settings.MAX_PRICE_LINK),
            "SOL": (settings.MIN_PRICE_SOL, settings.MAX_PRICE_SOL),
        },
        "symbols": {
            "validated_count": len(V3_VALIDATED_STRATEGIES),
            "removed_count": len(V3_REMOVED_SYMBOLS),
            "blacklist_count": len(V3_BLACKLIST),
        }
    }
