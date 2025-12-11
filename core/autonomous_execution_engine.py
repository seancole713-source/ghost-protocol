#!/usr/bin/env python3
"""
GHOST PROTOCOL - PHASE 5: AUTONOMOUS EXECUTION ENGINE
======================================================
Master Control: Fully autonomous trading without human intervention

This engine runs every 5 minutes and:
1. Evaluates latest predictions from Phase 4 self-improvement loop
2. Filters by confidence, liquidity, market hours
3. Calculates position size using Kelly Criterion
4. Checks risk limits (drawdown, max positions, correlation)
5. Executes trades via Alpaca broker
6. Monitors existing positions for exit signals
7. Sends Telegram notifications for all actions

SAFETY FEATURES:
- Default: Paper trading mode (fake money)
- Circuit breakers on losses
- Position limits (max 5 positions, 10% per position)
- Emergency kill switch via Telegram
- Risk engine validation before every trade

CONSTRAINTS:
- Zero cost (free tier APIs only)
- Private deployment (single user)
- No public endpoints
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Master switch
AUTO_EXECUTION_ENABLED = os.getenv("AUTO_EXECUTION_ENABLED", "0") == "1"

# Decision thresholds
AUTO_EXECUTION_MIN_CONFIDENCE = float(os.getenv("AUTO_EXECUTION_MIN_CONFIDENCE", "70"))
AUTO_EXECUTION_MAX_POSITIONS = int(os.getenv("AUTO_EXECUTION_MAX_POSITIONS", "5"))
AUTO_EXECUTION_INTERVAL_S = int(os.getenv("AUTO_EXECUTION_INTERVAL_S", "300"))  # 5 min
AUTO_EXECUTION_MARKET_HOURS_ONLY = os.getenv("AUTO_EXECUTION_MARKET_HOURS_ONLY", "1") == "1"

# Position sizing
AUTO_EXECUTION_DEFAULT_KELLY_FRACTION = float(os.getenv("AUTO_EXECUTION_KELLY_FRACTION", "0.25"))
AUTO_EXECUTION_MAX_POSITION_PCT = float(os.getenv("AUTO_EXECUTION_MAX_POSITION_PCT", "10"))  # % of portfolio

# Liquidity filters
AUTO_EXECUTION_MIN_VOLUME_STOCKS = int(os.getenv("AUTO_EXECUTION_MIN_VOLUME_STOCKS", "100000"))  # 100k shares
AUTO_EXECUTION_MIN_PRICE = float(os.getenv("AUTO_EXECUTION_MIN_PRICE", "5.0"))  # $5 minimum

# Risk limits
AUTO_EXECUTION_MAX_DAILY_LOSS_PCT = float(os.getenv("AUTO_EXECUTION_MAX_DAILY_LOSS_PCT", "5.0"))  # 5% of portfolio
AUTO_EXECUTION_MAX_DRAWDOWN_PCT = float(os.getenv("AUTO_EXECUTION_MAX_DRAWDOWN_PCT", "15.0"))  # 15% from peak

# Aliases for backward compatibility / testing
MIN_CONFIDENCE = AUTO_EXECUTION_MIN_CONFIDENCE
MAX_POSITIONS = AUTO_EXECUTION_MAX_POSITIONS
INTERVAL_S = AUTO_EXECUTION_INTERVAL_S
MIN_VOLUME_STOCKS = AUTO_EXECUTION_MIN_VOLUME_STOCKS
MIN_PRICE = AUTO_EXECUTION_MIN_PRICE
MAX_DAILY_LOSS_PCT = AUTO_EXECUTION_MAX_DAILY_LOSS_PCT
MAX_DRAWDOWN_PCT = AUTO_EXECUTION_MAX_DRAWDOWN_PCT


# ============================================================================
# STATE TRACKING
# ============================================================================

_execution_state = {
    "enabled": AUTO_EXECUTION_ENABLED,
    "last_cycle_time": 0,
    "total_cycles": 0,
    "trades_today": 0,
    "last_trade_time": 0,
    "circuit_breaker_active": False,
    "circuit_breaker_reason": "",
    "portfolio_start_value": 0,
    "portfolio_peak_value": 0,
    "daily_pnl_dollar": 0.0,
}


# ============================================================================
# AUTONOMOUS EXECUTION ENGINE
# ============================================================================

class AutonomousExecutionEngine:
    """
    Autonomous trading engine that executes trades based on predictions.
    
    Runs every 5 minutes, evaluates predictions, and places trades when
    confidence and risk criteria are met.
    """
    
    def __init__(self):
        self.enabled = AUTO_EXECUTION_ENABLED
        self.min_confidence = AUTO_EXECUTION_MIN_CONFIDENCE
        self.max_positions = AUTO_EXECUTION_MAX_POSITIONS
        self.kelly_fraction = AUTO_EXECUTION_DEFAULT_KELLY_FRACTION
        self.broker = None
        self.risk_engine = None
        self.position_sizer = None
        
        LOGGER.info(f"🤖 Autonomous Execution Engine initialized (enabled={self.enabled})")
    
    def _init_dependencies(self):
        """Lazy-load dependencies to avoid circular imports"""
        if self.broker is None:
            try:
                from core.alpaca_broker import get_broker
                self.broker = get_broker()
                LOGGER.info("✅ Alpaca broker loaded")
            except Exception as e:
                LOGGER.error(f"❌ Failed to load broker: {e}")
                self.enabled = False
        
        if self.risk_engine is None:
            try:
                from core.risk_engine import get_risk_engine
                self.risk_engine = get_risk_engine()
                LOGGER.info("✅ Risk engine loaded")
            except Exception as e:
                LOGGER.warning(f"⚠️  Risk engine not available: {e}")
        
        if self.position_sizer is None:
            try:
                from core.position_sizer import get_position_sizer
                self.position_sizer = get_position_sizer()
                LOGGER.info("✅ Position sizer loaded")
            except Exception as e:
                LOGGER.warning(f"⚠️  Position sizer not available: {e}")
    
    def run_execution_cycle(self) -> dict[str, Any]:
        """
        Main execution cycle: evaluate predictions and execute trades
        
        Returns:
            Dict with cycle summary
        """
        global _execution_state
        
        cycle_start = time.time()
        _execution_state["last_cycle_time"] = cycle_start
        _execution_state["total_cycles"] += 1
        
        result = {
            "ok": True,
            "cycle_number": _execution_state["total_cycles"],
            "timestamp": cycle_start,
            "trades_executed": 0,
            "predictions_evaluated": 0,
            "predictions_skipped": 0,
            "errors": [],
            "circuit_breaker_active": _execution_state["circuit_breaker_active"],
        }
        
        try:
            # Check if enabled
            if not self.enabled:
                result["ok"] = False
                result["errors"].append("Execution engine disabled")
                LOGGER.debug("[AUTO-EXEC] Engine disabled via config")
                return result
            
            # Initialize dependencies
            self._init_dependencies()
            
            if not self.broker or not self.broker.enabled:
                result["ok"] = False
                result["errors"].append("Broker not enabled")
                LOGGER.error("[AUTO-EXEC] Broker not available")
                return result
            
            # Check circuit breaker
            if _execution_state["circuit_breaker_active"]:
                result["ok"] = False
                result["errors"].append(f"Circuit breaker active: {_execution_state['circuit_breaker_reason']}")
                LOGGER.warning(f"[AUTO-EXEC] ⚠️  Circuit breaker active: {_execution_state['circuit_breaker_reason']}")
                return result
            
            # Get account state
            account = self._get_account_state()
            if not account:
                result["ok"] = False
                result["errors"].append("Failed to get account state")
                return result
            
            # Check risk limits
            risk_check = self._check_risk_limits(account)
            if not risk_check["ok"]:
                result["ok"] = False
                result["errors"].extend(risk_check["errors"])
                return result
            
            # Get predictions to evaluate
            predictions = self._get_predictions()
            result["predictions_evaluated"] = len(predictions)
            
            if not predictions:
                LOGGER.debug("[AUTO-EXEC] No predictions to evaluate")
                return result
            
            # Get current positions
            positions = self._get_current_positions()
            position_symbols = {p.get("symbol") for p in positions}
            
            # Check if we can add more positions
            if len(positions) >= self.max_positions:
                LOGGER.info(f"[AUTO-EXEC] Max positions reached ({len(positions)}/{self.max_positions})")
                result["predictions_skipped"] = len(predictions)
                return result
            
            # Evaluate each prediction
            trades_executed = 0
            for prediction in predictions:
                try:
                    # Check if already in position
                    symbol = prediction.get("symbol")
                    if symbol in position_symbols:
                        result["predictions_skipped"] += 1
                        continue
                    
                    # Check if we can add more positions
                    if len(positions) + trades_executed >= self.max_positions:
                        result["predictions_skipped"] += 1
                        continue
                    
                    # Evaluate trade opportunity
                    decision = self._evaluate_trade(prediction, account, positions)
                    
                    if decision["action"] == "EXECUTE":
                        # Execute trade
                        trade_result = self._execute_trade(decision, account)
                        
                        if trade_result.get("status") == "success":
                            trades_executed += 1
                            _execution_state["trades_today"] += 1
                            _execution_state["last_trade_time"] = time.time()
                            
                            # Send notification
                            self._send_trade_notification(trade_result)
                            
                            LOGGER.info(f"✅ [AUTO-EXEC] Trade executed: {trade_result['summary']}")
                        else:
                            result["errors"].append(f"{symbol}: {trade_result['error']}")
                    else:
                        result["predictions_skipped"] += 1
                        LOGGER.debug(f"[AUTO-EXEC] Skipped {symbol}: {decision['reason']}")
                
                except Exception as e:
                    LOGGER.error(f"[AUTO-EXEC] Error evaluating {prediction.get('symbol', 'unknown')}: {e}", exc_info=True)
                    result["errors"].append(str(e))
            
            result["trades_executed"] = trades_executed
            
            # Monitor existing positions
            if positions:
                exit_actions = self._monitor_positions(positions)
                result["exit_actions"] = len(exit_actions)
            
            cycle_duration = time.time() - cycle_start
            LOGGER.info(f"🤖 [AUTO-EXEC] Cycle complete: {trades_executed} trades, {result['predictions_skipped']} skipped, {cycle_duration:.2f}s")
            
        except Exception as e:
            LOGGER.error(f"[AUTO-EXEC] Cycle error: {e}", exc_info=True)
            result["ok"] = False
            result["errors"].append(f"Cycle error: {str(e)}")
        
        return result
    
    def _get_account_state(self) -> dict[str, Any] | None:
        """Get account info from broker"""
        try:
            account = self.broker.get_account()
            
            if not account:
                return None
            
            portfolio_value = float(account.get("portfolio_value", 0))
            
            # Track portfolio peak for drawdown calculation
            global _execution_state
            if _execution_state["portfolio_start_value"] == 0:
                _execution_state["portfolio_start_value"] = portfolio_value
                _execution_state["portfolio_peak_value"] = portfolio_value
            else:
                _execution_state["portfolio_peak_value"] = max(
                    _execution_state["portfolio_peak_value"],
                    portfolio_value
                )
            
            return {
                "portfolio_value": portfolio_value,
                "cash": float(account.get("cash", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "positions_value": float(account.get("long_market_value", 0)),
            }
        
        except Exception as e:
            LOGGER.error(f"[AUTO-EXEC] Failed to get account: {e}")
            return None
    
    def _check_risk_limits(self, account: dict) -> dict[str, Any]:
        """Check if risk limits allow trading"""
        
        # Check daily loss limit
        portfolio_value = account["portfolio_value"]
        start_value = _execution_state["portfolio_start_value"]
        
        if start_value > 0:
            daily_loss_pct = ((portfolio_value - start_value) / start_value) * 100
            
            if daily_loss_pct <= -AUTO_EXECUTION_MAX_DAILY_LOSS_PCT:
                self._activate_circuit_breaker(f"Daily loss {daily_loss_pct:.1f}% exceeds {AUTO_EXECUTION_MAX_DAILY_LOSS_PCT}%")
                return {"status": "circuit_breaker", "reason": f"Daily loss {abs(daily_loss_pct):.1f}% exceeds limit {AUTO_EXECUTION_MAX_DAILY_LOSS_PCT}%"}
        
        # Check drawdown limit
        peak_value = _execution_state["portfolio_peak_value"]
        if peak_value > 0:
            drawdown_pct = ((peak_value - portfolio_value) / peak_value) * 100
            
            if drawdown_pct > AUTO_EXECUTION_MAX_DRAWDOWN_PCT:
                self._activate_circuit_breaker(f"Drawdown {drawdown_pct:.1f}% exceeds {AUTO_EXECUTION_MAX_DRAWDOWN_PCT}%")
                return {"status": "circuit_breaker", "reason": f"Drawdown {drawdown_pct:.1f}% exceeds limit {AUTO_EXECUTION_MAX_DRAWDOWN_PCT}%"}
        
        return {"status": "ok", "reason": "Risk limits OK"}
    
    def _get_predictions(self) -> list[dict]:
        """Get latest predictions from cache"""
        try:
            # Import here to avoid circular dependency
            from wolf_app import _LATEST_PREDICTIONS
            
            if not _LATEST_PREDICTIONS:
                return []
            
            # Convert to list and filter by confidence
            predictions = []
            for symbol, pred in _LATEST_PREDICTIONS.items():
                if isinstance(pred, dict):
                    confidence = pred.get("confidence", 0)
                    if confidence >= self.min_confidence:
                        pred["symbol"] = symbol
                        predictions.append(pred)
            
            # Sort by confidence (highest first)
            predictions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            return predictions[:20]  # Limit to top 20
        
        except Exception as e:
            LOGGER.error(f"[AUTO-EXEC] Failed to get predictions: {e}")
            return []
    
    def _get_current_positions(self) -> list[dict]:
        """Get current open positions"""
        try:
            positions = self.broker.get_positions()
            return positions if positions else []
        except Exception as e:
            LOGGER.error(f"[AUTO-EXEC] Failed to get positions: {e}")
            return []
    
    def _evaluate_trade(
        self,
        prediction: dict,
        account: dict,
        positions: list[dict]
    ) -> dict[str, Any]:
        """
        Evaluate if we should trade this prediction
        
        Returns:
            {"action": "EXECUTE" | "SKIP", "reason": str, ...}
        """
        from core.trade_decision_engine import evaluate_trade_opportunity
        
        return evaluate_trade_opportunity(
            prediction=prediction,
            portfolio=account,
            current_positions=positions,
            risk_engine=self.risk_engine
        )
    
    def _execute_trade(self, decision: dict, account: dict) -> dict[str, Any]:
        """Execute trade based on decision"""
        try:
            symbol = decision["symbol"]
            side = decision["side"]  # "buy" or "sell"
            shares = decision["shares"]
            
            # Submit order
            order = self.broker.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                type="market",
                time_in_force="day"
            )
            
            if not order:
                return {
                    "ok": False,
                    "error": "Order submission failed"
                }
            
            return {
                "ok": True,
                "order_id": order.get("id"),
                "symbol": symbol,
                "side": side,
                "shares": shares,
                "confidence": decision.get("confidence", 0),
                "reasoning": decision.get("reason", ""),
                "stop_loss": decision.get("stop_loss_price"),
                "take_profit": decision.get("take_profit_price"),
                "summary": f"{side.upper()} {shares} {symbol} @ {decision.get('confidence', 0):.0f}% confidence"
            }
        
        except Exception as e:
            LOGGER.error(f"[AUTO-EXEC] Trade execution error: {e}", exc_info=True)
            return {
                "ok": False,
                "error": str(e)
            }
    
    def _monitor_positions(self, positions: list[dict]) -> list[dict]:
        """Monitor existing positions for exit signals"""
        exit_actions = []
        
        # TODO: Implement position monitoring (SL/TP checks)
        # This will be enhanced in Task 1.3
        
        return exit_actions
    
    def _send_trade_notification(self, trade_result: dict):
        """Send Telegram notification for trade"""
        try:
            from core.telegram_hunter import send_trade_notification
            
            send_trade_notification(trade_result)
        
        except Exception as e:
            LOGGER.warning(f"[AUTO-EXEC] Failed to send notification: {e}")
    
    def _activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker to halt trading"""
        global _execution_state
        _execution_state["circuit_breaker_active"] = True
        _execution_state["circuit_breaker_reason"] = reason
        
        LOGGER.error(f"🚨 [AUTO-EXEC] CIRCUIT BREAKER ACTIVATED: {reason}")
        
        # Send alert
        try:
            from core.telegram_hunter import send_telegram_message
            
            message = f"""
🚨 CIRCUIT BREAKER ACTIVATED
━━━━━━━━━━━━━━━━━━━━━━━━━
Reason: {reason}
Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

All autonomous trading halted.
Use /resume_trading to restart.
━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            send_telegram_message(message)
        except Exception as e:
            LOGGER.error(f"Failed to send circuit breaker alert: {e}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_execution_engine_instance = None


def get_execution_engine() -> AutonomousExecutionEngine:
    """Get singleton execution engine instance"""
    global _execution_engine_instance
    
    if _execution_engine_instance is None:
        _execution_engine_instance = AutonomousExecutionEngine()
    
    return _execution_engine_instance


def run_execution_cycle() -> dict[str, Any]:
    """Convenience function to run execution cycle"""
    engine = get_execution_engine()
    return engine.run_execution_cycle()


def get_execution_status() -> dict[str, Any]:
    """Get current execution engine status"""
    global _execution_state
    
    return {
        "ok": True,
        "enabled": _execution_state["enabled"],
        "circuit_breaker_active": _execution_state["circuit_breaker_active"],
        "circuit_breaker_reason": _execution_state["circuit_breaker_reason"],
        "total_cycles": _execution_state["total_cycles"],
        "trades_today": _execution_state["trades_today"],
        "last_cycle_time": _execution_state["last_cycle_time"],
        "last_trade_time": _execution_state["last_trade_time"],
        "portfolio_peak_value": _execution_state["portfolio_peak_value"],
        "config": {
            "min_confidence": AUTO_EXECUTION_MIN_CONFIDENCE,
            "max_positions": AUTO_EXECUTION_MAX_POSITIONS,
            "interval_s": AUTO_EXECUTION_INTERVAL_S,
            "kelly_fraction": AUTO_EXECUTION_DEFAULT_KELLY_FRACTION,
        }
    }


def pause_trading(reason: str = "Manual pause"):
    """Pause autonomous trading"""
    global _execution_state
    _execution_state["circuit_breaker_active"] = True
    _execution_state["circuit_breaker_reason"] = reason
    LOGGER.warning(f"⏸️  [AUTO-EXEC] Trading paused: {reason}")


def resume_trading():
    """Resume autonomous trading"""
    global _execution_state
    _execution_state["circuit_breaker_active"] = False
    _execution_state["circuit_breaker_reason"] = ""
    LOGGER.info("▶️  [AUTO-EXEC] Trading resumed")


# ============================================================================
# MANUAL TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    print("🤖 Testing Autonomous Execution Engine")
    print("=" * 60)
    
    engine = get_execution_engine()
    
    print(f"\nEngine enabled: {engine.enabled}")
    print(f"Min confidence: {engine.min_confidence}%")
    print(f"Max positions: {engine.max_positions}")
    
    print("\nRunning test cycle...")
    result = engine.run_execution_cycle()
    
    print(f"\nResult: {result}")
    
    print("\nStatus:")
    status = get_execution_status()
    print(f"  Total cycles: {status['total_cycles']}")
    print(f"  Trades today: {status['trades_today']}")
    print(f"  Circuit breaker: {status['circuit_breaker_active']}")
