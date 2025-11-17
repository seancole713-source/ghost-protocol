"""
Multi-timeframe forecast generation.
Provides 1h, 4h, daily, and weekly forecasts with adjusted volatility.
"""

import math
import time
from typing import Any

from wolf_app import (
    WOLF,
    PRED_SIGMA_DAILY,
    get_wolf_price,
    _build_price_providers,
    get_price_quorum,
    _is_market_open_now,
    _get_portfolio_qty_and_avg,
    _calculate_entry_exit_targets,
    _detect_market_condition,
    _evaluate_signal,
    _store_forecast_48h,
    get_wolf_news,
    _get_filings_signal,
    LOGGER,
)


def _generate_timeframe_forecast(
    symbol: str,
    timeframe: str,
    hours: float
) -> dict[str, Any]:
    """
    Generate forecast for any timeframe (1h, 4h, daily, weekly).
    
    Args:
        symbol: Stock/crypto ticker
        timeframe: Human label ("1h", "4h", "1d", "1w")
        hours: Forecast horizon in hours (1, 4, 24, 168)
    
    Returns:
        Forecast data with entry/exit targets, stop-loss, position sizing
    """
    try:
        # Get current price
        normalized_symbol = symbol.upper()
        
        if symbol == WOLF:
            price, _, provider = get_wolf_price()
        else:
            try:
                is_market_open, _ = _is_market_open_now()
            except Exception:
                is_market_open = False
            
            providers = _build_price_providers(normalized_symbol, is_market_open=is_market_open)
            if providers:
                decision = get_price_quorum().get_price(
                    symbol=normalized_symbol,
                    providers=providers,
                    prev_close=None,
                    is_market_open=is_market_open,
                    timeout=120.0,
                )
                price = decision.price
                provider = decision.provider_label
            else:
                price = None
                provider = "unavailable"

        if price is None or price <= 0:
            provider_label = provider if 'provider' in locals() else "unknown"
            return {
                "ok": False,
                "error": f"live price unavailable (provider: {provider_label})",
                "symbol": symbol,
                "timeframe": timeframe,
            }

        # Adjust volatility based on timeframe
        sigma_daily = float(PRED_SIGMA_DAILY)
        days = hours / 24.0
        vol = sigma_daily * math.sqrt(days)

        # Shorter timeframes have smaller expected moves
        if timeframe == "1h":
            price_pred_mid = price * (1.0 + (vol * 0.05))  # Small bias
        elif timeframe == "4h":
            price_pred_mid = price * (1.0 + (vol * 0.08))
        elif timeframe == "1d":
            price_pred_mid = price * (1.0 + (vol * 0.10))
        else:  # Weekly
            price_pred_mid = price * (1.0 + (vol * 0.15))  # Larger moves over week

        price_pred_lo = price * (1.0 - vol)
        price_pred_hi = price * (1.0 + vol)

        # PnL prediction
        qty, avg_cost = _get_portfolio_qty_and_avg()
        if qty > 0:
            pred_value = qty * price_pred_mid
            pnl_pred_mid = pred_value - (qty * avg_cost)
        else:
            pnl_pred_mid = None

        # Confidence adjustments by timeframe
        base_confidence = 0.70
        if timeframe == "1h":
            confidence = base_confidence * 0.85  # Less confident short-term
        elif timeframe == "4h":
            confidence = base_confidence * 0.90
        elif timeframe == "1d":
            confidence = base_confidence * 1.00
        else:  # Weekly
            confidence = base_confidence * 0.95  # Slightly less confident long-term

        # Store forecast (reuse 48h storage for now)
        model = f"simple-vol-{timeframe}"
        forecast_id = _store_forecast_48h(
            symbol=symbol,
            price_now=price,
            price_pred_mid=price_pred_mid,
            price_pred_lo=price_pred_lo,
            price_pred_hi=price_pred_hi,
            pnl_pred_mid=pnl_pred_mid,
            confidence=confidence,
            model=model,
            features={
                "provider": provider,
                "vol_daily": sigma_daily,
                "vol_period": vol,
                "timeframe": timeframe,
                "hours": hours,
            },
        )

        # Calculate entry/exit targets
        targets = _calculate_entry_exit_targets(price, price_pred_mid, confidence)
        
        # Market condition
        market_condition = _detect_market_condition(symbol)
        
        # Gain potential
        gain_pct = ((price_pred_mid - price) / price * 100) if price else 0
        
        # Trade signal
        try:
            signal = _evaluate_signal(symbol)
            trade_signal = signal.get("action") if signal else None
        except Exception:
            trade_signal = None

        return {
            "ok": True,
            "forecast_id": forecast_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "hours": hours,
            "ts_issued": int(time.time()),
            "price_now": round(price, 2),
            "price_pred_mid": round(price_pred_mid, 2),
            "price_pred_lo": round(price_pred_lo, 2),
            "price_pred_hi": round(price_pred_hi, 2),
            "pnl_pred_mid": round(pnl_pred_mid, 2) if pnl_pred_mid else None,
            "confidence": round(confidence, 3),
            "model": model,
            "trade_signal": trade_signal,
            "gain_potential_pct": round(gain_pct, 2),
            "market_condition": market_condition,
            "entry_target": targets.get("entry_target"),
            "exit_targets": {
                "take_profit_1": targets.get("exit_target_1"),
                "take_profit_2": targets.get("exit_target_2")
            },
            "stop_loss": targets.get("stop_loss"),
            "risk_reward_ratio": targets.get("risk_reward_ratio"),
            "position_size_pct": targets.get("position_size_pct"),
        }

    except Exception as e:
        LOGGER.exception(f"Multi-timeframe forecast exception for {symbol} ({timeframe})")
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol,
            "timeframe": timeframe,
        }


def generate_1h_forecast(symbol: str) -> dict[str, Any]:
    """Generate 1-hour forecast."""
    return _generate_timeframe_forecast(symbol, "1h", 1.0)


def generate_4h_forecast(symbol: str) -> dict[str, Any]:
    """Generate 4-hour forecast."""
    return _generate_timeframe_forecast(symbol, "4h", 4.0)


def generate_daily_forecast(symbol: str) -> dict[str, Any]:
    """Generate daily (24h) forecast."""
    return _generate_timeframe_forecast(symbol, "1d", 24.0)


def generate_weekly_forecast(symbol: str) -> dict[str, Any]:
    """Generate weekly (7d) forecast."""
    return _generate_timeframe_forecast(symbol, "1w", 168.0)
