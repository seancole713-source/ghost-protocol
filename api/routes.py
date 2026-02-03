"""
Main API routes.
"""
import subprocess
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException

from config.settings import settings
from config.symbols import V3_VALIDATED_STRATEGIES, V3_REMOVED_SYMBOLS

router = APIRouter()


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "operational",
    }


@router.get("/health")
async def health():
    """
    Health check endpoint.
    
    Used by Railway/Kubernetes for liveness and readiness probes.
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "git_sha": get_git_sha(),
        "timestamp": datetime.utcnow().isoformat(),
        "v3_enabled": settings.V3_ENABLED,
    }


@router.get("/config")
async def get_config():
    """
    Get current configuration (non-sensitive values only).
    
    Useful for debugging and verifying deployment configuration.
    """
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "v3_min_confidence": settings.V3_MIN_CONFIDENCE,
        "v3_default_hold_hours": settings.V3_DEFAULT_HOLD_HOURS,
        "v3_enabled": settings.V3_ENABLED,
        "default_target_pct": settings.DEFAULT_TARGET_PCT,
        "default_stop_pct": settings.DEFAULT_STOP_PCT,
        "default_rr_ratio": settings.DEFAULT_RR_RATIO,
        "timezone": settings.TIMEZONE,
        "top10_hour": settings.TOP10_HOUR,
        "validated_symbols": list(V3_VALIDATED_STRATEGIES.keys()),
        "removed_symbols_count": len(V3_REMOVED_SYMBOLS),
    }


@router.get("/v3/strategies")
async def get_v3_strategies():
    """
    Get V3 validated strategies.
    
    Returns the complete configuration for each validated symbol.
    """
    strategies = {}
    for symbol, strategy in V3_VALIDATED_STRATEGIES.items():
        strategies[symbol] = {
            "strategy": strategy.strategy,
            "direction_override": strategy.direction_override,
            "hold_hours": strategy.hold_hours,
            "hold_days": strategy.hold_hours // 24,
            "backtest_win_rate": strategy.backtest_win_rate,
            "backtest_trades": strategy.backtest_trades,
            "p_value": strategy.p_value,
        }
    
    return {
        "count": len(strategies),
        "strategies": strategies,
        "notes": {
            "ETH": "ghost_inverse: Only trade when Ghost says DOWN, flip to UP",
            "XRP": "mean_reversion: Trade any direction with 168h hold",
            "LINK": "mean_reversion: Trade any direction with 72h hold",
        }
    }


@router.get("/v3/removed")
async def get_v3_removed():
    """
    Get symbols removed from V3 validation.
    
    These symbols were analyzed but did not show statistical significance (p >= 0.05).
    """
    return {
        "count": len(V3_REMOVED_SYMBOLS),
        "symbols": V3_REMOVED_SYMBOLS,
        "note": "These symbols were analyzed in 52K trade backtest but did not meet p < 0.05 threshold"
    }
