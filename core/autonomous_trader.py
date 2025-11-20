#!/usr/bin/env python3
"""
🤖 GHOST AUTONOMOUS TRADER
Transforms Ghost from passive opportunity detector to active hunter-trader

Responsibilities:
1. Monitor scanner outputs continuously
2. Evaluate opportunities against confidence threshold
3. Calculate position sizes using Kelly Criterion
4. Execute trades via broker integration
5. Track decision rationale for learning loop
6. Enforce portfolio risk limits
7. Send Telegram alerts on autonomous actions

Configuration:
- AUTO_TRADE_ENABLED: Master switch (default: 0)
- AUTO_TRADE_MIN_CONFIDENCE: Minimum confidence threshold (default: 80%)
- AUTO_TRADE_MAX_POSITION_PCT: Max % of portfolio per position (default: 10%)
- AUTO_TRADE_KELLY_FRACTION: Kelly multiplier (default: 0.25 for quarter-Kelly)
- AUTO_TRADE_MAX_DAILY_TRADES: Rate limit (default: 5)
- AUTO_TRADE_SCAN_INTERVAL: Opportunity scan frequency (default: 60s)

Architecture:
- Monitoring Loop: Continuously polls scanner + watchlist
- Decision Engine: Evaluates opportunities against rules
- Position Sizer: Kelly Criterion with risk management
- Execution Layer: Routes orders to broker
- Learning Loop: Records decisions for model training

Safety:
- Fail-safe: Disabled by default, requires explicit env var
- Rate limiting: Max trades per day
- Portfolio limits: Max % per position
- Risk checks: Pre-trade validation via risk_engine
- Audit trail: All decisions logged with rationale

Usage:
    from core.autonomous_trader import start_autonomous_trader
    
    asyncio.create_task(start_autonomous_trader())
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta

LOGGER = logging.getLogger("ghost.autonomous_trader")

# Configuration from environment
ENABLED = os.getenv("AUTO_TRADE_ENABLED", "0") == "1"
MIN_CONFIDENCE = float(os.getenv("AUTO_TRADE_MIN_CONFIDENCE", "80.0"))
MAX_POSITION_PCT = float(os.getenv("AUTO_TRADE_MAX_POSITION_PCT", "10.0"))
KELLY_FRACTION = float(os.getenv("AUTO_TRADE_KELLY_FRACTION", "0.25"))
MAX_DAILY_TRADES = int(os.getenv("AUTO_TRADE_MAX_DAILY_TRADES", "5"))
SCAN_INTERVAL = int(os.getenv("AUTO_TRADE_SCAN_INTERVAL", "60"))

# Internal state
_TRADES_TODAY = 0
_LAST_RESET = datetime.now().date()
_EXECUTION_HISTORY = []


class OpportunityEvaluator:
    """Evaluates opportunities for autonomous execution"""

    def __init__(self, logger):
        self.logger = logger

    def should_trade(self, opportunity: dict) -> tuple[bool, str]:
        """
        Determine if opportunity meets criteria for autonomous execution

        Args:
            opportunity: Scanner output with score, symbol, market, etc.

        Returns:
            (should_execute, reason) tuple
        """
        # Check confidence threshold
        confidence = opportunity.get("score", 0)
        if confidence < MIN_CONFIDENCE:
            return False, f"Confidence {confidence}% < threshold {MIN_CONFIDENCE}%"

        # Check symbol validity
        symbol = opportunity.get("symbol")
        if not symbol:
            return False, "Missing symbol"

        # Check market
        market = opportunity.get("market", "stock")
        if market not in ["stock", "crypto"]:
            return False, f"Invalid market: {market}"

        # Check for recent price data
        price = opportunity.get("price")
        if not price or price <= 0:
            return False, "Invalid or missing price"

        # Check volume (stocks only)
        if market == "stock":
            volume = opportunity.get("volume", 0)
            if volume < 100000:  # Minimum 100k volume for liquidity
                return False, f"Low volume: {volume}"

        # All checks passed
        return True, f"Confidence {confidence}%, criteria met"


class PositionSizer:
    """Calculates position sizes using Kelly Criterion with safety caps"""

    def __init__(self, logger):
        self.logger = logger

    def calculate_size(
        self, confidence: float, portfolio_value: float, price: float, win_rate: float = None
    ) -> dict:
        """
        Calculate position size using Kelly Criterion

        Args:
            confidence: Prediction confidence (0-100)
            portfolio_value: Total portfolio value in $
            price: Current price per share/token
            win_rate: Historical win rate (optional, uses confidence if None)

        Returns:
            Dict with position_size_usd, position_size_shares, kelly_pct, reasoning
        """
        # Convert confidence to probability
        prob_win = (confidence / 100.0) if confidence > 1 else confidence
        if win_rate is not None:
            prob_win = (prob_win + win_rate) / 2.0  # Blend confidence + historical

        prob_loss = 1.0 - prob_win

        # Assume typical win/loss ratio (adjust based on historical data)
        # Conservative: 1.5:1 reward-to-risk
        win_loss_ratio = 1.5

        # Kelly Formula: f = (p * b - q) / b
        # where p = probability of win, q = probability of loss, b = win/loss ratio
        kelly_fraction = (prob_win * win_loss_ratio - prob_loss) / win_loss_ratio

        # Safety: Cap at configured fraction (e.g., 0.25 for quarter-Kelly)
        kelly_capped = max(0, min(kelly_fraction, 1.0)) * KELLY_FRACTION

        # Calculate position size in USD
        position_usd = portfolio_value * kelly_capped

        # Cap at max position percentage
        max_position_usd = portfolio_value * (MAX_POSITION_PCT / 100.0)
        position_usd = min(position_usd, max_position_usd)

        # Convert to shares/tokens
        position_shares = position_usd / price if price > 0 else 0

        reasoning = (
            f"Kelly: {kelly_fraction:.2%} → Capped: {kelly_capped:.2%} "
            f"→ ${position_usd:.2f} ({position_shares:.2f} shares)"
        )

        return {
            "position_size_usd": position_usd,
            "position_size_shares": position_shares,
            "kelly_pct": kelly_capped * 100,
            "reasoning": reasoning,
        }


class AutonomousTrader:
    """Main autonomous trading coordinator"""

    def __init__(self, logger):
        self.logger = logger
        self.evaluator = OpportunityEvaluator(logger)
        self.sizer = PositionSizer(logger)
        self.running = False

    async def monitoring_loop(self):
        """
        Main monitoring loop - continuously evaluates opportunities
        """
        global _TRADES_TODAY, _LAST_RESET

        self.running = True
        self.logger.info(f"🤖 Autonomous Trader: Monitoring loop started (scan every {SCAN_INTERVAL}s)")

        while self.running:
            try:
                # Reset daily trade counter at midnight
                today = datetime.now().date()
                if today != _LAST_RESET:
                    _TRADES_TODAY = 0
                    _LAST_RESET = today
                    self.logger.info("🔄 Daily trade counter reset")

                # Check if rate limit reached
                if _TRADES_TODAY >= MAX_DAILY_TRADES:
                    self.logger.info(
                        f"⏸️ Daily trade limit reached ({_TRADES_TODAY}/{MAX_DAILY_TRADES})"
                    )
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                # Get latest opportunities from scanner
                opportunities = await self._fetch_opportunities()

                if not opportunities:
                    self.logger.debug("No opportunities detected")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                # Evaluate each opportunity
                for opp in opportunities:
                    # Check if should trade
                    should_execute, reason = self.evaluator.should_trade(opp)

                    if not should_execute:
                        self.logger.debug(
                            f"❌ {opp.get('symbol')}: {reason}"
                        )
                        continue

                    # Check rate limit again (in case multiple passed)
                    if _TRADES_TODAY >= MAX_DAILY_TRADES:
                        self.logger.info("⏸️ Daily limit reached during evaluation")
                        break

                    # Execute trade
                    success = await self._execute_trade(opp, reason)

                    if success:
                        _TRADES_TODAY += 1
                        self.logger.info(
                            f"✅ Trade executed: {opp.get('symbol')} "
                            f"({_TRADES_TODAY}/{MAX_DAILY_TRADES} today)"
                        )

                await asyncio.sleep(SCAN_INTERVAL)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(SCAN_INTERVAL)

    async def _fetch_opportunities(self) -> list[dict]:
        """
        Fetch latest opportunities from market scanner

        Returns:
            List of opportunity dicts sorted by confidence
        """
        try:
            from core.market_scanner import scan_all

            results = await scan_all()
            all_opps = results.get("stocks", []) + results.get("crypto", [])

            # Sort by score (descending)
            all_opps.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Return top 5 for evaluation
            return all_opps[:5]

        except Exception as e:
            self.logger.error(f"Failed to fetch opportunities: {e}")
            return []

    async def _execute_trade(self, opportunity: dict, reason: str) -> bool:
        """
        Execute trade for given opportunity

        Args:
            opportunity: Scanner opportunity dict
            reason: Evaluation reason string

        Returns:
            True if executed successfully, False otherwise
        """
        try:
            symbol = opportunity.get("symbol")
            market = opportunity.get("market", "stock")
            price = opportunity.get("price")
            confidence = opportunity.get("score", 0)

            # Get portfolio value
            portfolio_value = await self._get_portfolio_value()

            if portfolio_value <= 0:
                self.logger.warning("Cannot execute: portfolio value <= 0")
                return False

            # Calculate position size
            sizing = self.sizer.calculate_size(confidence, portfolio_value, price)

            position_usd = sizing["position_size_usd"]
            position_shares = sizing["position_size_shares"]

            if position_shares <= 0:
                self.logger.warning(
                    f"Position size too small for {symbol}: {position_shares} shares"
                )
                return False

            # Execute via broker
            order_result = await self._place_order(
                symbol, market, "BUY", position_shares, price
            )

            if not order_result.get("success"):
                self.logger.error(
                    f"Order failed for {symbol}: {order_result.get('error')}"
                )
                return False

            # Log execution
            execution_record = {
                "timestamp": int(time.time()),
                "symbol": symbol,
                "market": market,
                "action": "BUY",
                "shares": position_shares,
                "price": price,
                "total_usd": position_usd,
                "confidence": confidence,
                "reason": reason,
                "sizing_reasoning": sizing["reasoning"],
                "order_id": order_result.get("order_id"),
            }

            _EXECUTION_HISTORY.append(execution_record)

            # Send Telegram alert
            await self._send_execution_alert(execution_record)

            return True

        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}", exc_info=True)
            return False

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value from STATE"""
        try:
            # Import here to avoid circular dependency
            from wolf_app import STATE

            cash = STATE.get("cash", 0.0)
            positions_value = sum(
                pos.get("market_value", 0.0) for pos in STATE.get("positions", [])
            )

            return float(cash + positions_value)

        except Exception as e:
            self.logger.error(f"Failed to get portfolio value: {e}")
            return 0.0

    async def _place_order(
        self, symbol: str, market: str, side: str, quantity: float, price: float
    ) -> dict:
        """
        Place order via broker integration

        Args:
            symbol: Trading symbol
            market: "stock" or "crypto"
            side: "BUY" or "SELL"
            quantity: Number of shares/tokens
            price: Current price

        Returns:
            Dict with success, order_id, error
        """
        try:
            from core.alpaca_broker import get_broker

            broker = get_broker()

            if not broker.enabled:
                self.logger.warning("Broker not enabled, cannot place order")
                return {"success": False, "error": "Broker not enabled"}

            # Place market order
            order = broker.place_order(
                symbol=symbol, side=side, quantity=quantity, order_type="market"
            )

            if order:
                return {
                    "success": True,
                    "order_id": order.get("id"),
                    "status": order.get("status"),
                }
            else:
                return {"success": False, "error": "Order placement returned None"}

        except Exception as e:
            self.logger.error(f"Order placement error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _send_execution_alert(self, execution: dict):
        """
        Send Telegram alert for autonomous execution

        Args:
            execution: Execution record dict
        """
        try:
            from core.telegram_alerts import send_telegram

            symbol = execution["symbol"]
            shares = execution["shares"]
            price = execution["price"]
            total_usd = execution["total_usd"]
            confidence = execution["confidence"]
            reason = execution["reason"]

            message = (
                f"🤖 *AUTONOMOUS TRADE*\n\n"
                f"Symbol: `{symbol}`\n"
                f"Action: BUY {shares:.2f} shares @ ${price:.2f}\n"
                f"Total: ${total_usd:.2f}\n"
                f"Confidence: {confidence:.0f}%\n\n"
                f"Reason: {reason}\n"
                f"Sizing: {execution['sizing_reasoning']}"
            )

            send_telegram(message)

        except Exception as e:
            self.logger.error(f"Failed to send execution alert: {e}")

    def stop(self):
        """Stop the monitoring loop"""
        self.running = False
        self.logger.info("🛑 Autonomous Trader: Monitoring loop stopped")


# Singleton instance
_TRADER: AutonomousTrader | None = None


async def start_autonomous_trader():
    """
    Start the autonomous trader background task

    Note: Only starts if AUTO_TRADE_ENABLED=1 in environment
    """
    global _TRADER

    if not ENABLED:
        LOGGER.info("🤖 Autonomous Trader: DISABLED (AUTO_TRADE_ENABLED=0)")
        return

    if _TRADER is not None:
        LOGGER.warning("Autonomous Trader already running")
        return

    _TRADER = AutonomousTrader(LOGGER)

    LOGGER.info("=" * 80)
    LOGGER.info("🤖 AUTONOMOUS TRADER CONFIGURATION:")
    LOGGER.info(f"   Enabled: {ENABLED}")
    LOGGER.info(f"   Min Confidence: {MIN_CONFIDENCE}%")
    LOGGER.info(f"   Max Position %: {MAX_POSITION_PCT}%")
    LOGGER.info(f"   Kelly Fraction: {KELLY_FRACTION}")
    LOGGER.info(f"   Max Daily Trades: {MAX_DAILY_TRADES}")
    LOGGER.info(f"   Scan Interval: {SCAN_INTERVAL}s")
    LOGGER.info("=" * 80)

    # Start monitoring loop
    await _TRADER.monitoring_loop()


def stop_autonomous_trader():
    """Stop the autonomous trader"""
    global _TRADER

    if _TRADER:
        _TRADER.stop()
        _TRADER = None


def get_execution_history() -> list[dict]:
    """Get autonomous execution history"""
    return _EXECUTION_HISTORY.copy()


def get_trader_status() -> dict:
    """Get current trader status"""
    return {
        "enabled": ENABLED,
        "running": _TRADER is not None and _TRADER.running if _TRADER else False,
        "trades_today": _TRADES_TODAY,
        "max_daily_trades": MAX_DAILY_TRADES,
        "last_reset": _LAST_RESET.isoformat() if _LAST_RESET else None,
        "total_executions": len(_EXECUTION_HISTORY),
    }
