"""Routes: broker — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse

try:
    import httpx
except ImportError:
    httpx = None

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

try:
    from wolf_helpers import (
        AUTH_DEP, SECURITY_SCHEME, WOLF, WOLF_SQLITE_PATH,
        _is_truthy, _json500, with_cap,
        AlertTemplateBody, AlertToggle, AlertConfigBody,
        RuntimeConfigBody, ControlBody, ModeBody, TrainBody,
        AgentControlBody, CashBody, PositionAddBody, PositionsImportBody,
        WatchlistImportBody, TradeRequest, PredFeedbackBody,
        AddPositionBody, OrderPlaceBody,
        _PredictRunBody, _RecordPriceBody, _ScoreBody, _BacktestBody,
        ChatRequest, AiDecision, TelegramUpdate,
    )
    from fastapi.security import HTTPAuthorizationCredentials
except Exception as _wh_e:
    import logging as _l
    _l.getLogger("ghost").warning(f"wolf_helpers import partial: {_wh_e}")
    AUTH_DEP = None
    WOLF = "WOLF"
    WOLF_SQLITE_PATH = "data/wolf.db"


# ── Also inject wolf_helpers globals (private helper functions + shared state) ─
import wolf_helpers as _wh
globals().update({k: v for k, v in vars(_wh).items() if not k.startswith("__")})
del _wh

# ── Inject all app-config globals into this route module ─────────────────────
# Mirrors wolf_app.py's pattern: provides all module-level constants that route
# handlers reference directly, without needing per-name imports.
import engines.app_config as _ac
globals().update({k: v for k, v in vars(_ac).items() if not k.startswith("__")})
del _ac

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- 55 endpoints ---

@router.get("/api/v3/trade/dashboard")
async def api_v3_trade_dashboard():
    """
    Phase 6: Get real-time trade monitoring dashboard.
    """
    try:
        from core.trade_monitor import get_dashboard_summary
        return get_dashboard_summary()
    except Exception as e:
        LOGGER.error(f"Trade dashboard error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/trade/history")
async def api_v3_trade_history(limit: int = 100):
    """
    Phase 6: Get recent trade history.
    """
    try:
        from core.trade_monitor import get_trade_history
        return {
            "ok": True,
            "trades": get_trade_history(limit),
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/stage3/ensemble/forecast")
async def api_stage3_ensemble_forecast(
    symbol: str,
    current_price: float,
    horizon_hours: int = 24,
    historical_prices: list[float] | None = None,
    sentiment_score: float = 0.0,
):
    """Generate ensemble forecast combining multiple models."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        ensemble = get_ensemble_forecaster()
        forecast = ensemble.forecast(
            symbol=symbol,
            current_price=current_price,
            horizon_hours=horizon_hours,
            historical_prices=historical_prices,
            sentiment_score=sentiment_score,
        )
        return forecast
    except Exception as e:
        LOGGER.error(f"stage3_ensemble_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage3/ensemble/performance")
async def api_stage3_ensemble_performance():
    """Get ensemble performance report."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        ensemble = get_ensemble_forecaster()
        report = ensemble.get_performance_report()
        return report
    except Exception as e:
        LOGGER.error(f"stage3_ensemble_perf_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage3/regime/detect")
async def api_stage3_regime_detect(
    prices: list[float], spy_price: float | None = None, vix_level: float | None = None
):
    """Detect current market regime."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        regime = get_regime_detector()
        result = regime.detect_regime(prices=prices, spy_price=spy_price, vix_level=vix_level)
        return result
    except Exception as e:
        LOGGER.error(f"stage3_regime_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage3/regime/current")
async def api_stage3_regime_current():
    """Get current market regime."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        regime = get_regime_detector()
        return {
            "regime": regime.current_regime,
            "confidence": regime.confidence,
            "strategy_adjustments": regime._get_strategy_adjustments(regime.current_regime),
        }
    except Exception as e:
        LOGGER.error(f"stage3_regime_current_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage3/regime/history")
async def api_stage3_regime_history(limit: int = 50):
    """Get regime history."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        regime = get_regime_detector()
        history = regime.get_regime_history(limit=limit)
        distribution = regime.get_regime_distribution(days=30)
        return {"history": history, "distribution_30d": distribution}
    except Exception as e:
        LOGGER.error(f"stage3_regime_history_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage3/risk/check")
async def api_stage3_risk_check(symbol: str, position_size_usd: float, regime: str = "SIDEWAYS"):
    """Check if proposed position passes risk limits."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        risk = get_risk_engine()
        result = risk.check_position_limits(
            symbol=symbol, position_size_usd=position_size_usd, regime=regime
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage3_risk_check_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage3/risk/update")
async def api_stage3_risk_update(portfolio_value: float):
    """Update portfolio value and calculate drawdown."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        risk = get_risk_engine()
        risk.update_portfolio_value(portfolio_value)
        return {
            "portfolio_value": risk.portfolio_value,
            "drawdown_pct": risk.current_drawdown_pct,
        }
    except Exception as e:
        LOGGER.error(f"stage3_risk_update_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage3/risk/dashboard")
async def api_stage3_risk_dashboard():
    """Get comprehensive risk metrics dashboard."""
    if not STAGE3_ENABLED:
        return {"error": "Stage 3 not enabled"}
    try:
        risk = get_risk_engine()
        dashboard = risk.get_risk_dashboard()
        return dashboard
    except Exception as e:
        LOGGER.error(f"stage3_risk_dashboard_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/portfolio/optimize")
async def api_stage4_portfolio_optimize(
    assets: list[str],
    returns: dict[str, list[float]],
    target_return: float | None = None,
    risk_free_rate: float = 0.02,
):
    """Optimize portfolio allocation using MPT."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        portfolio_mgr = get_portfolio_manager()
        result = portfolio_mgr.optimize_portfolio(
            assets=assets,
            returns=returns,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage4_portfolio_optimize_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/portfolio/risk-parity")
async def api_stage4_portfolio_risk_parity(assets: list[str], returns: dict[str, list[float]]):
    """Calculate risk parity allocation."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        portfolio_mgr = get_portfolio_manager()
        result = portfolio_mgr.calculate_risk_parity(assets, returns)
        return result
    except Exception as e:
        LOGGER.error(f"stage4_portfolio_risk_parity_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/portfolio/rebalance-check")
async def api_stage4_portfolio_rebalance_check(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    threshold: float = 0.05,
):
    """Check if portfolio needs rebalancing."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        portfolio_mgr = get_portfolio_manager()
        result = portfolio_mgr.check_rebalance_needed(current_weights, target_weights, threshold)
        return result
    except Exception as e:
        LOGGER.error(f"stage4_portfolio_rebalance_check_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/hedging/beta-hedge")
async def api_stage4_hedging_beta_hedge(
    portfolio_symbol: str,
    portfolio_returns: list[float],
    market_returns: list[float],
    hedge_symbol: str = "SPY",
):
    """Calculate beta-neutral hedge ratio."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        hedging = get_hedging_engine()
        result = hedging.calculate_beta_hedge(
            portfolio_symbol, portfolio_returns, market_returns, hedge_symbol
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage4_hedging_beta_hedge_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/hedging/pairs-trade")
async def api_stage4_hedging_pairs_trade(
    symbol_a: str,
    returns_a: list[float],
    symbol_b: str,
    returns_b: list[float],
    entry_z_threshold: float = 2.0,
):
    """Find pairs trading opportunity."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        hedging = get_hedging_engine()
        result = hedging.find_pairs_trade(
            symbol_a, returns_a, symbol_b, returns_b, entry_z_threshold
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage4_hedging_pairs_trade_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/backtest/run")
async def api_stage4_backtest_run(
    strategy_name: str,
    returns: list[float],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
):
    """Run historical backtest on strategy."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        backtester = get_backtester()
        result = backtester.run_backtest(
            strategy_name, returns, start_date, end_date, initial_capital
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage4_backtest_run_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/backtest/monte-carlo")
async def api_stage4_backtest_monte_carlo(
    returns: list[float], num_simulations: int = 1000, simulation_length: int = 252
):
    """Run Monte Carlo simulation."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        backtester = get_backtester()
        result = backtester.monte_carlo_simulation(returns, num_simulations, simulation_length)
        return result
    except Exception as e:
        LOGGER.error(f"stage4_backtest_monte_carlo_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/backtest/walk-forward")
async def api_stage4_backtest_walk_forward(
    returns: list[float],
    in_sample_window: int = 120,
    out_sample_window: int = 30,
    step_size: int = 30,
):
    """Run walk-forward analysis."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        backtester = get_backtester()
        result = backtester.walk_forward_analysis(
            returns, in_sample_window, out_sample_window, step_size
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage4_backtest_walk_forward_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/strategy/register")
async def api_stage4_strategy_register(strategy_id: str, strategy_name: str, description: str = ""):
    """Register a new strategy for A/B testing."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        strategy_tester = get_strategy_tester()
        result = strategy_tester.register_strategy(strategy_id, strategy_name, description)
        return result
    except Exception as e:
        LOGGER.error(f"stage4_strategy_register_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage4/strategy/ab-test")
async def api_stage4_strategy_ab_test(
    strategy_a: str,
    strategy_b: str,
    market_data: dict[str, list[float]],
    start_date: str,
    end_date: str,
):
    """Run A/B test between two strategies."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        strategy_tester = get_strategy_tester()
        result = strategy_tester.run_ab_test(
            strategy_a, strategy_b, market_data, start_date, end_date
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage4_strategy_ab_test_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage4/strategy/champion")
async def api_stage4_strategy_champion():
    """Get current champion strategy."""
    if not STAGE4_ENABLED:
        return {"error": "Stage 4 not enabled"}
    try:
        strategy_tester = get_strategy_tester()
        result = strategy_tester.get_champion()
        return result
    except Exception as e:
        LOGGER.error(f"stage4_strategy_champion_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/order/create")
async def api_stage5_order_create(
    symbol: str,
    order_type: str,
    side: str,
    quantity: float,
    price: float | None = None,
    stop_price: float | None = None,
    time_in_force: str = "DAY",
    strategy: str | None = None,
):
    """Create a new order."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        order_mgr = get_order_manager()
        result = order_mgr.create_order(
            symbol=symbol,
            order_type=OrderType[order_type],
            side=OrderSide[side],
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=TimeInForce[time_in_force],
            strategy=strategy,
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage5_order_create_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/order/submit/{order_id}")
async def api_stage5_order_submit(order_id: str):
    """Submit order for execution."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        order_mgr = get_order_manager()
        result = order_mgr.submit_order(order_id)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_order_submit_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/order/cancel/{order_id}")
async def api_stage5_order_cancel(order_id: str):
    """Cancel an order."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        order_mgr = get_order_manager()
        result = order_mgr.cancel_order(order_id)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_order_cancel_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage5/order/{order_id}")
async def api_stage5_order_get(order_id: str):
    """Get order details."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        order_mgr = get_order_manager()
        result = order_mgr.get_order(order_id)
        return result if result else {"error": "Order not found"}
    except Exception as e:
        LOGGER.error(f"stage5_order_get_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage5/orders/active")
async def api_stage5_orders_active(symbol: str | None = None):
    """Get all active orders."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        order_mgr = get_order_manager()
        orders = order_mgr.get_active_orders(symbol)
        return {"orders": orders, "count": len(orders)}
    except Exception as e:
        LOGGER.error(f"stage5_orders_active_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage5/positions")
async def api_stage5_positions():
    """Get all positions."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        order_mgr = get_order_manager()
        positions = order_mgr.get_all_positions()
        return {"positions": positions, "count": len(positions)}
    except Exception as e:
        LOGGER.error(f"stage5_positions_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/router/vwap")
async def api_stage5_router_vwap(
    symbol: str,
    total_quantity: float,
    duration_minutes: int = 30,
    participation_rate: float = 0.10,
):
    """Create VWAP execution plan."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        smart_router = get_smart_router()
        result = smart_router.create_vwap_plan(
            symbol, total_quantity, duration_minutes, participation_rate
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage5_router_vwap_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/router/twap")
async def api_stage5_router_twap(symbol: str, total_quantity: float, duration_minutes: int = 30):
    """Create TWAP execution plan."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        smart_router = get_smart_router()
        result = smart_router.create_twap_plan(symbol, total_quantity, duration_minutes)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_router_twap_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/router/adaptive")
async def api_stage5_router_adaptive(
    symbol: str,
    total_quantity: float,
    duration_minutes: int = 30,
    urgency: str = "medium",
):
    """Create adaptive execution plan."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        smart_router = get_smart_router()
        result = smart_router.create_adaptive_plan(
            symbol, total_quantity, duration_minutes, urgency
        )
        return result
    except Exception as e:
        LOGGER.error(f"stage5_router_adaptive_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage5/analytics/dashboard")
async def api_stage5_analytics_dashboard(lookback_days: int = 7):
    """Get execution analytics dashboard."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        exec_analytics = get_execution_analytics()
        result = exec_analytics.get_execution_dashboard(lookback_days)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_analytics_dashboard_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage5/analytics/latency")
async def api_stage5_analytics_latency(lookback_days: int = 7):
    """Get latency distribution."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        exec_analytics = get_execution_analytics()
        result = exec_analytics.get_latency_distribution(lookback_days)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_analytics_latency_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/risk/check")
async def api_stage5_risk_check(
    order_id: str, symbol: str, side: str, quantity: float, price: float | None = None
):
    """Run pre-trade risk check."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        exec_risk = get_execution_risk()
        result = exec_risk.pre_trade_check(order_id, symbol, side, quantity, price)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_risk_check_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/risk/kill-switch/activate")
async def api_stage5_kill_switch_activate(reason: str, triggered_by: str = "system"):
    """Activate kill switch."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        exec_risk = get_execution_risk()
        result = exec_risk.activate_kill_switch(reason, triggered_by)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_kill_switch_activate_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage5/risk/kill-switch/deactivate")
async def api_stage5_kill_switch_deactivate(authorized_by: str = "admin"):
    """Deactivate kill switch."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        exec_risk = get_execution_risk()
        result = exec_risk.deactivate_kill_switch(authorized_by)
        return result
    except Exception as e:
        LOGGER.error(f"stage5_kill_switch_deactivate_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage5/risk/kill-switch/status")
async def api_stage5_kill_switch_status():
    """Get kill switch status."""
    if not STAGE5_ENABLED:
        return {"error": "Stage 5 not enabled"}
    try:
        exec_risk = get_execution_risk()
        result = exec_risk.get_kill_switch_status()
        return result
    except Exception as e:
        LOGGER.error(f"stage5_kill_switch_status_error: {e}")
        return {"error": str(e)}


@router.get("/api/risk/status")
async def api_risk_status(symbol: str = "WOLF"):
    """
    APEX Risk Shell 2.0 - Get current risk status

    Returns:
        Risk status with can_trade flag, risk_level, reasons, and metrics
    """
    import yfinance as yf

    from core.enhanced_risk_shell import get_enhanced_risk_manager

    if symbol.upper() != WOLF:
        return {"error": f"Symbol {symbol} not supported"}, 404

    try:
        # Get portfolio data (use defaults if cockpit not available)
        try:
            cockpit_resp = await api_cockpit_snapshot()
            if hasattr(cockpit_resp, "body"):
                import json

                cockpit_data = json.loads(cockpit_resp.body.decode())
            else:
                cockpit_data = cockpit_resp if isinstance(cockpit_resp, dict) else {}
            portfolio_data = {
                "daily_pnl": (
                    cockpit_data.get("pnl", {}).get("total", 0.0)
                    if isinstance(cockpit_data.get("pnl"), dict)
                    else 0.0
                ),
                "daily_drawdown_pct": (
                    abs(cockpit_data.get("pnl", {}).get("total_pct", 0.0))
                    if isinstance(cockpit_data.get("pnl"), dict)
                    and cockpit_data.get("pnl", {}).get("total_pct", 0.0) < 0
                    else 0.0
                ),
                "var_95": (
                    cockpit_data.get("risk", {}).get("var_95", 0.0)
                    if isinstance(cockpit_data.get("risk"), dict)
                    else 0.0
                ),
                "max_concentration": 0.0,
            }
        except Exception:
            portfolio_data = {
                "daily_pnl": 0.0,
                "daily_drawdown_pct": 0.0,
                "var_95": 0.0,
                "max_concentration": 0.0,
            }

        # Get market volatility data from real sources
        try:
            ticker = yf.Ticker(WOLF)
            hist = ticker.history(period="90d")

            # Safety check: ensure we have enough data
            if hist.empty or len(hist) < 20:
                LOGGER.warning(
                    f"Insufficient yfinance data for {WOLF}, using fallback volatility"
                )
                market_data = {
                    "volatility": 0.25,
                    "volatility_mean": 0.22,
                    "volatility_std": 0.04,
                    "model_drift_pct": 0.0,
                    "model_mape": 0.0,
                }
            else:
                returns = hist["Close"].pct_change().dropna()
                current_vol = returns.tail(20).std() * (252**0.5)
                historical_vol_mean = returns.std() * (252**0.5)
                historical_vol_std = returns.rolling(20).std().std() * (252**0.5)
                market_data = {
                    "volatility": current_vol,
                    "volatility_mean": historical_vol_mean,
                    "volatility_std": historical_vol_std,
                    "model_drift_pct": 0.0,
                    "model_mape": 0.0,
                }
        except Exception as e:
            LOGGER.warning(f"yfinance error for {WOLF}: {e}, using fallback")
            market_data = {
                "volatility": 0.25,
                "volatility_mean": 0.22,
                "volatility_std": 0.04,
                "model_drift_pct": 0.0,
                "model_mape": 0.0,
            }

        risk_mgr = get_enhanced_risk_manager()
        result = risk_mgr.check_risk_status(portfolio_data, market_data)
        # Always return a JSON object, never a tuple
        if isinstance(result, dict):
            result.setdefault("error", None)
            return result
        else:
            return {"error": "Risk manager returned non-dict result"}
    except Exception as e:
        LOGGER.error(f"Risk status check failed: {e}", exc_info=True)
        # Always return a JSON object with error field
        return {
            "error": f"Risk check failed: {str(e)}",
            "can_trade": False,
            "risk_level": "CRITICAL",
            "reasons": [str(e)],
        }


@router.post("/api/risk/kill_switch")
async def api_risk_kill_switch(action: str = "status", auth_token: str = ""):
    """
    APEX Risk Shell 2.0 - Control kill-switch

    Args:
        action: "activate", "deactivate", or "status"
        auth_token: Authorization token (required for activate/deactivate)

    Returns:
        Kill-switch status
    """
    from core.enhanced_risk_shell import get_enhanced_risk_manager

    risk_mgr = get_enhanced_risk_manager()

    # Status check doesn't require auth
    if action == "status":
        return {
            "kill_switch_active": risk_mgr.kill_switch_active,
            "circuit_breaker_active": risk_mgr.circuit_breaker_until is not None,
            "circuit_breaker_until": (
                risk_mgr.circuit_breaker_until.isoformat()
                if risk_mgr.circuit_breaker_until
                else None
            ),
            "cooldown_reason": risk_mgr.cooldown_reason,
        }

    # Auth required for control actions
    expected_token = os.getenv("GHOST_API_TOKEN", "")
    if not expected_token or auth_token != expected_token:
        return {"error": "Unauthorized - valid auth_token required"}, 403

    try:
        if action == "activate":
            risk_mgr.activate_kill_switch(reason="Manual activation via API")
            return {
                "success": True,
                "message": "Kill-switch activated - all trading halted",
                "kill_switch_active": True,
            }

        elif action == "deactivate":
            risk_mgr.deactivate_kill_switch()
            return {
                "success": True,
                "message": "Kill-switch deactivated - trading resumed",
                "kill_switch_active": False,
            }

        else:
            return {
                "error": f"Invalid action: {action}. Use 'activate', 'deactivate', or 'status'"
            }, 400

    except Exception as e:
        LOGGER.error(f"Kill-switch control failed: {e}", exc_info=True)
        return {"error": f"Kill-switch control failed: {str(e)}"}, 500


@router.post("/api/risk/circuit_breaker")
async def api_risk_circuit_breaker(action: str = "status", auth_token: str = ""):
    """
    APEX Risk Shell 2.0 - Control circuit breaker

    Args:
        action: "reset" or "status"
        auth_token: Authorization token (required for reset)

    Returns:
        Circuit breaker status
    """
    from core.enhanced_risk_shell import get_enhanced_risk_manager

    risk_mgr = get_enhanced_risk_manager()

    # Status check doesn't require auth
    if action == "status":
        return {
            "circuit_breaker_active": risk_mgr.circuit_breaker_until is not None,
            "circuit_breaker_until": (
                risk_mgr.circuit_breaker_until.isoformat()
                if risk_mgr.circuit_breaker_until
                else None
            ),
            "cooldown_reason": risk_mgr.cooldown_reason,
        }

    # Auth required for control actions
    expected_token = os.getenv("GHOST_API_TOKEN", "")
    if not expected_token or auth_token != expected_token:
        return {"error": "Unauthorized - valid auth_token required"}, 403

    try:
        if action == "reset":
            risk_mgr.reset_circuit_breaker()
            return {
                "success": True,
                "message": "Circuit breaker manually reset",
                "circuit_breaker_active": False,
            }

        else:
            return {"error": f"Invalid action: {action}. Use 'reset' or 'status'"}, 400

    except Exception as e:
        LOGGER.error(f"Circuit breaker control failed: {e}", exc_info=True)
        return {"error": f"Circuit breaker control failed: {str(e)}"}, 500


@router.get("/api/risk/dashboard")
async def api_risk_dashboard():
    """
    APEX Risk Shell 2.0 - Comprehensive risk dashboard

    Returns:
        Recent events, anomalies, model drift, limits
    """
    from core.enhanced_risk_shell import get_enhanced_risk_manager

    try:
        risk_mgr = get_enhanced_risk_manager()
        dashboard = risk_mgr.get_risk_dashboard()

        return dashboard

    except Exception as e:
        LOGGER.error(f"Risk dashboard failed: {e}", exc_info=True)
        return {"error": f"Risk dashboard failed: {str(e)}"}, 500


@router.get("/api/broker/health")
async def broker_health(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """
    Check broker connectivity and account status.
    Returns account info, buying power, positions count.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {
                "ok": False,
                "enabled": False,
                "message": "Broker not enabled (set BROKER=alpaca)",
            }

        health = broker.health_check()
        return health
    except Exception as e:
        LOGGER.error(f"Broker health check failed: {e}")
        return {
            "ok": False,
            "enabled": False,
            "error": str(e),
        }


@router.get("/api/broker/metrics")
async def broker_metrics(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    from core.alpaca_broker import get_broker

    broker = get_broker()
    snapshot: dict[str, Any]
    try:
        snapshot = broker.metrics_snapshot()
    except Exception:
        snapshot = {}

    return {
        "enabled": broker.enabled,
        "paper": getattr(broker, "paper", True),
        "metrics": snapshot,
    }


@router.get("/api/broker/positions")
async def broker_get_positions(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """
    Get all open positions from broker.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "positions": [], "message": "Broker not enabled"}

        positions = broker.get_positions()
        return {
            "ok": True,
            "count": len(positions),
            "positions": positions,
        }
    except Exception as e:
        LOGGER.error(f"Failed to get broker positions: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/broker/account")
async def broker_get_account(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """
    Get broker account information.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "message": "Broker not enabled"}

        account = broker.get_account()
        return {
            "ok": True,
            "account": account,
        }
    except Exception as e:
        LOGGER.error(f"Failed to get broker account: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/broker/clock")
async def broker_get_clock(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """
    Get market clock (is market open, next open/close times).
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "message": "Broker not enabled"}

        clock = broker.get_clock()
        return {
            "ok": True,
            "clock": clock,
        }
    except Exception as e:
        LOGGER.error(f"Failed to get market clock: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/api/trade/submit")
async def trade_submit(
    request: TradeRequest, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """
    Submit a trade order with full risk management checks.

    Example:
        POST /api/trade/submit
        {
            "symbol": "WOLF",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "dry_run": false
        }
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker
        from core.risk_engine import get_risk_engine

        broker = get_broker()
        risk_engine = get_risk_engine()

        if not broker.enabled:
            return {
                "ok": False,
                "submitted": False,
                "error": "Broker not enabled (set BROKER=alpaca)",
            }

        # Get current portfolio state for risk checks
        try:
            account = broker.get_account()
            portfolio_value = float(account.get("portfolio_value", 0))
            current_nav = portfolio_value
            positions = broker.get_positions()

            # Convert positions to dict for risk engine
            existing_positions = {}
            for pos in positions:
                sym = pos.get("symbol", "")
                existing_positions[sym] = {
                    "qty": float(pos.get("qty", 0)),
                    "price": float(pos.get("current_price", 0)),
                    "value": float(pos.get("market_value", 0)),
                }
        except Exception as e:
            LOGGER.error(f"Failed to get account state for risk check: {e}")
            return {
                "ok": False,
                "submitted": False,
                "error": f"Failed to get account state: {e}",
            }

        # Get current price for the symbol
        symbol = request.symbol.upper()
        try:
            current_price = get_current_price(symbol)
            if not current_price or current_price <= 0:
                return {
                    "ok": False,
                    "submitted": False,
                    "error": f"Could not get valid price for {symbol}"
                }
        except Exception as e:
            LOGGER.error(f"Failed to get price for {symbol}: {e}")
            return {
                "ok": False,
                "submitted": False,
                "error": f"Price lookup failed: {e}"
            }

        # === RISK GUARD CHECK (Ghost 2.x) ===
        # Apply risk budget enforcement for paper trading
        try:
            from core.risk.risk_guard import get_risk_guard
            risk_guard = get_risk_guard()

            if risk_guard.is_enabled():
                # Determine quantity
                trade_qty = request.qty if request.qty else 0
                if not trade_qty and request.notional:
                    trade_qty = request.notional / current_price

                # Get current equity and P&L
                current_equity = portfolio_value
                daily_pnl = 0.0  # Requires intraday trade log (not tracked yet)
                total_pnl = current_equity - float(account.get("last_equity", current_equity))

                # Check risk limits
                allowed, reason = risk_guard.check_order(
                    symbol=symbol,
                    side=request.side,
                    quantity=trade_qty,
                    price=current_price,
                    current_equity=current_equity,
                    current_positions=existing_positions,
                    daily_pnl=daily_pnl,
                    total_pnl=total_pnl
                )

                if not allowed:
                    LOGGER.warning(f"Risk guard blocked order: {symbol} {request.side} - {reason}")
                    return {
                        "ok": False,
                        "submitted": False,
                        "blocked_by_risk_guard": True,
                        "error": f"Risk limit exceeded: {reason}",
                        "risk_guard_reason": reason
                    }

                LOGGER.info(f"Risk guard approved order: {symbol} {request.side} {trade_qty}@${current_price:.2f}")
        except Exception as e:
            LOGGER.error(f"Risk guard check failed: {e}")
            # Continue without risk guard if it fails (fail-open for availability)
        # === END RISK GUARD CHECK ===
        try:
            if request.type == "market":
                # For market orders, get current price for risk calculation
                price_info = get_wolf_price(symbol=symbol)
                current_price = price_info[0] if price_info else 0
            elif request.limit_price:
                current_price = request.limit_price
            elif request.stop_price:
                current_price = request.stop_price
            else:
                current_price = 0
        except Exception:
            current_price = 0

        # Build order object for risk check
        order = {
            "symbol": symbol,
            "qty": request.qty or 0,
            "notional": request.notional or 0,
            "side": request.side.lower(),
            "type": request.type,
            "price": current_price,
        }

        # RISK CHECK
        allowed, risk_reason = risk_engine.risk_check_order(
            order=order,
            portfolio_value=portfolio_value,
            current_nav=current_nav,
            existing_positions=existing_positions,
        )

        if not allowed:
            _add_event(
                "trade.blocked",
                "Order blocked by risk engine",
                {
                    "symbol": symbol,
                    "side": request.side,
                    "qty": request.qty,
                    "reason": risk_reason,
                },
            )
            return {
                "ok": False,
                "submitted": False,
                "blocked": True,
                "reason": risk_reason,
                "order": order,
            }

        # If dry run, stop here
        if request.dry_run:
            return {
                "ok": True,
                "submitted": False,
                "dry_run": True,
                "risk_check": "PASSED",
                "reason": risk_reason,
                "order": order,
            }

        # SUBMIT ORDER TO BROKER
        result = broker.submit_order(
            symbol=symbol,
            qty=request.qty,
            notional=request.notional,
            side=request.side,
            type=request.type,
            time_in_force=request.time_in_force,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            trail_price=request.trail_price,
            trail_percent=request.trail_percent,
            extended_hours=request.extended_hours,
            client_order_id=request.client_order_id,
        )

        # Log successful submission
        _add_event(
            "trade.submitted",
            f"{request.side.upper()} {symbol}",
            {
                "symbol": symbol,
                "side": request.side,
                "qty": request.qty,
                "type": request.type,
                "order_id": result.get("id"),
                "status": result.get("status"),
            },
        )

        # Store in local orders table
        try:
            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT INTO {ORDERS_TABLE}
                (id, ts, symbol, side, qty, type, status, broker_id, broker, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.get("client_order_id", str(uuid.uuid4())),
                    time.time(),
                    symbol,
                    request.side,
                    request.qty or request.notional,
                    request.type,
                    result.get("status", "submitted"),
                    result.get("id"),
                    "alpaca",
                    f"Submitted via API at {datetime.now().isoformat()}",
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.warning(f"Failed to store order in local DB: {e}")

        return {
            "ok": True,
            "submitted": True,
            "risk_check": "PASSED",
            "order": result,
        }

    except Exception as e:
        LOGGER.error(f"Trade submission failed: {e}", exc_info=True)
        return {
            "ok": False,
            "submitted": False,
            "error": str(e),
        }


@router.get("/api/trade/orders")
async def trade_get_orders(
    status: str | None = None,
    limit: int = 50,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get orders from broker.

    Query params:
        status: "open", "closed", "all" (default: open)
        limit: max number of orders (default: 50)
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "orders": [], "message": "Broker not enabled"}

        orders = broker.get_orders(status=status or "open", limit=limit)
        return {
            "ok": True,
            "count": len(orders),
            "orders": orders,
        }
    except Exception as e:
        LOGGER.error(f"Failed to get orders: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/trade/order/{order_id}")
async def trade_get_order(
    order_id: str, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """
    Get specific order by ID.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "message": "Broker not enabled"}

        order = broker.get_order(order_id)
        return {
            "ok": True,
            "order": order,
        }
    except Exception as e:
        LOGGER.error(f"Failed to get order {order_id}: {e}")
        return {"ok": False, "error": str(e)}


@router.delete("/api/trade/order/{order_id}")
async def trade_cancel_order(
    order_id: str, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """
    Cancel an order by ID.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "message": "Broker not enabled"}

        result = broker.cancel_order(order_id)

        _add_event(
            "trade.cancelled",
            f"Order {order_id} cancelled",
            {
                "order_id": order_id,
            },
        )

        return {
            "ok": True,
            "cancelled": True,
            "order": result,
        }
    except Exception as e:
        LOGGER.error(f"Failed to cancel order {order_id}: {e}")
        return {"ok": False, "error": str(e)}


@router.delete("/api/trade/orders/cancel_all")
async def trade_cancel_all_orders(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """
    Cancel ALL open orders.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "message": "Broker not enabled"}

        result = broker.cancel_all_orders()

        _add_event(
            "trade.cancel_all",
            "All orders cancelled",
            {
                "count": len(result),
            },
        )

        return {
            "ok": True,
            "cancelled": len(result),
            "orders": result,
        }
    except Exception as e:
        LOGGER.error(f"Failed to cancel all orders: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/api/trade/position/close/{symbol}")
async def trade_close_position(
    symbol: str, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """
    Close entire position for a symbol (sell all shares).
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()

        if not broker.enabled:
            return {"ok": False, "message": "Broker not enabled"}

        result = broker.close_position(symbol.upper())

        _add_event(
            "trade.position_closed",
            f"Position closed: {symbol}",
            {
                "symbol": symbol,
            },
        )

        return {
            "ok": True,
            "closed": True,
            "order": result,
        }
    except Exception as e:
        LOGGER.error(f"Failed to close position {symbol}: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/risk/scan_exits")
async def risk_scan_exits(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """
    Scan all positions for stop-loss and take-profit triggers.
    Returns list of positions that should be exited.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.alpaca_broker import get_broker
        from core.risk_engine import get_risk_engine

        broker = get_broker()
        risk_engine = get_risk_engine()

        if not broker.enabled:
            return {"ok": False, "message": "Broker not enabled"}

        # Get current positions
        positions = broker.get_positions()

        # Convert to format risk engine expects
        position_list = []
        for pos in positions:
            position_list.append(
                {
                    "symbol": pos.get("symbol", ""),
                    "qty": float(pos.get("qty", 0)),
                    "avg_cost": float(pos.get("avg_entry_price", 0)),
                    "entry_price": float(pos.get("avg_entry_price", 0)),
                    "current_price": float(pos.get("current_price", 0)),
                }
            )

        # Scan for exit signals
        exit_signals = risk_engine.scan_positions_for_exits(position_list)

        return {
            "ok": True,
            "positions_scanned": len(position_list),
            "exit_signals": exit_signals,
            "count": len(exit_signals),
        }
    except Exception as e:
        LOGGER.error(f"Failed to scan exits: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/risk/position-size")
async def api_calculate_position_size(
    portfolio_value: float,
    win_rate: float = 0.6,
    avg_win_pct: float = 5.0,
    avg_loss_pct: float = 3.0,
    confidence: float = 0.7
):
    """Calculate optimal position size using Kelly Criterion"""
    try:
        from core.position_sizing import calculate_position_size
        
        result = calculate_position_size(
            portfolio_value, win_rate, avg_win_pct, avg_loss_pct, confidence
        )
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/risk/report")
async def api_get_risk_report(
    portfolio_value: float = 10000,
    positions: str = ""  # JSON string of positions
):
    """Get comprehensive risk report"""
    try:
        from core.position_sizing import get_risk_report
        import json
        
        positions_dict = json.loads(positions) if positions else {}
        
        report = get_risk_report(portfolio_value, positions_dict)
        return {"ok": True, **report}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/v3/risk/check-limits")
async def api_check_risk_limits(request: Request):
    """Check if a proposed position violates risk limits"""
    try:
        from core.position_sizing import check_risk_limits
        
        data = await request.json()
        
        result = check_risk_limits(
            portfolio_value=data["portfolio_value"],
            proposed_position=data["proposed_position"],
            symbol=data["symbol"],
            current_exposure=data.get("current_exposure", {})
        )
        
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


