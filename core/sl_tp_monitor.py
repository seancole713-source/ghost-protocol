#!/usr/bin/env python3
"""
🔄 GHOST AUTO SL/TP MONITOR - Background Loop

Automatically exits positions when stop-loss or take-profit levels are hit.
Enhanced with:
- Trailing stops
- Prediction expiry (exit after 6 hours)
- Adverse move protection (exit if -2% within 1 hour)

Usage:
    python3 core/sl_tp_monitor.py

Or run as background task in wolf_app.py:
    asyncio.create_task(start_sl_tp_monitor())
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta

LOGGER = logging.getLogger("ghost.sl_tp_monitor")

# Configuration from environment
CHECK_INTERVAL_SECONDS = int(os.getenv("SL_TP_CHECK_INTERVAL", "60"))  # Check every 60 seconds
STOP_LOSS_PCT = float(os.getenv("RISK_SL_PCT", "3.0"))  # -3% stop loss
TAKE_PROFIT_PCT = float(os.getenv("RISK_TP_PCT", "6.0"))  # +6% take profit
ENABLED = os.getenv("SL_TP_MONITOR_ENABLED", "1") == "1"

# Phase 5 enhancements
TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "1") == "1"
TRAILING_STOP_ACTIVATION_PCT = float(os.getenv("TRAILING_STOP_ACTIVATION_PCT", "3.0"))  # Activate after +3%
TRAILING_STOP_DISTANCE_PCT = float(os.getenv("TRAILING_STOP_DISTANCE_PCT", "2.0"))  # Trail by 2%
PREDICTION_EXPIRY_HOURS = float(os.getenv("PREDICTION_EXPIRY_HOURS", "6.0"))  # Exit after 6 hours
ADVERSE_MOVE_PCT = float(os.getenv("ADVERSE_MOVE_PCT", "2.0"))  # Exit if -2% within 1 hour
ADVERSE_MOVE_TIME_MINUTES = float(os.getenv("ADVERSE_MOVE_TIME_MINUTES", "60"))  # 1 hour

# Track position entry times and peak prices for trailing stops
_position_tracker: dict[str, dict] = {}


async def check_positions_for_exits() -> list[dict]:
    """
    Check all open positions for SL/TP triggers.
    Enhanced with trailing stops, prediction expiry, adverse move detection.

    Returns:
        List of positions that need to be exited
    """
    exit_signals = []

    try:
        # Import here to avoid circular dependencies
        from core.alpaca_broker import get_broker
        from core.risk_engine import get_risk_engine

        broker = get_broker()
        risk_engine = get_risk_engine()

        if not broker.enabled:
            LOGGER.debug("Broker not enabled, skipping SL/TP check")
            return []

        # Get all open positions from broker
        positions = broker.get_positions()

        if not positions:
            LOGGER.debug("No open positions to monitor")
            return []

        now = datetime.now()

        # Convert to format expected by risk engine
        risk_positions = []
        for pos in positions:
            symbol = pos.get("symbol")
            entry_price = float(pos.get("avg_entry_price", 0))
            current_price = float(pos.get("current_price", 0))
            unrealized_plpc = float(pos.get("unrealized_plpc", 0)) * 100

            # Initialize position tracking if first time seeing it
            if symbol not in _position_tracker:
                _position_tracker[symbol] = {
                    "entry_time": now,
                    "entry_price": entry_price,
                    "peak_price": current_price,
                    "trailing_stop_active": False,
                    "trailing_stop_price": 0.0
                }
                LOGGER.info(f"Started tracking {symbol} (entry: ${entry_price:.2f})")

            tracker = _position_tracker[symbol]

            # Update peak price for trailing stop
            if current_price > tracker["peak_price"]:
                tracker["peak_price"] = current_price

                # Activate trailing stop if profit exceeds activation threshold
                if TRAILING_STOP_ENABLED and unrealized_plpc >= TRAILING_STOP_ACTIVATION_PCT:
                    if not tracker["trailing_stop_active"]:
                        tracker["trailing_stop_active"] = True
                        LOGGER.info(
                            f"Trailing stop ACTIVATED for {symbol} (profit: {unrealized_plpc:.2f}%)"
                        )

                    # Update trailing stop price
                    tracker["trailing_stop_price"] = tracker["peak_price"] * (
                        1 - TRAILING_STOP_DISTANCE_PCT / 100.0
                    )

            # CHECK 1: Trailing stop
            if tracker["trailing_stop_active"] and current_price <= tracker["trailing_stop_price"]:
                exit_signals.append({
                    "symbol": symbol,
                    "type": "trailing_stop",
                    "reason": f"Price ${current_price:.2f} fell below trailing stop ${tracker['trailing_stop_price']:.2f}",
                    "pnl_pct": unrealized_plpc,
                    "entry_price": entry_price,
                    "exit_price": current_price
                })
                continue

            # CHECK 2: Prediction expiry (exit after 6 hours)
            time_in_position = (now - tracker["entry_time"]).total_seconds() / 3600.0  # hours
            if time_in_position >= PREDICTION_EXPIRY_HOURS:
                exit_signals.append({
                    "symbol": symbol,
                    "type": "prediction_expiry",
                    "reason": f"Position held for {time_in_position:.1f}h (expiry: {PREDICTION_EXPIRY_HOURS}h)",
                    "pnl_pct": unrealized_plpc,
                    "entry_price": entry_price,
                    "exit_price": current_price
                })
                continue

            # CHECK 3: Adverse move (exit if -2% within 1 hour)
            time_in_position_minutes = (now - tracker["entry_time"]).total_seconds() / 60.0
            if (
                time_in_position_minutes <= ADVERSE_MOVE_TIME_MINUTES
                and unrealized_plpc <= -ADVERSE_MOVE_PCT
            ):
                exit_signals.append({
                    "symbol": symbol,
                    "type": "adverse_move",
                    "reason": f"Quick loss {unrealized_plpc:.2f}% within {time_in_position_minutes:.0f} minutes",
                    "pnl_pct": unrealized_plpc,
                    "entry_price": entry_price,
                    "exit_price": current_price
                })
                continue

            risk_positions.append(
                {
                    "symbol": symbol,
                    "qty": float(pos.get("qty", 0)),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "market_value": float(pos.get("market_value", 0)),
                    "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                    "unrealized_plpc": unrealized_plpc,
                }
            )

        # CHECK 4: Standard SL/TP triggers from risk engine
        standard_exits = risk_engine.scan_positions_for_exits(risk_positions)
        exit_signals.extend(standard_exits)

        if exit_signals:
            LOGGER.info(
                f"Found {len(exit_signals)} positions to exit: {[s['symbol'] for s in exit_signals]}"
            )

        return exit_signals

    except Exception as e:
        LOGGER.error(f"Error checking positions for exits: {e}", exc_info=True)
        return []


async def execute_exit(signal: dict) -> bool:
    """
    Execute an exit order for a position that hit SL/TP.

    Args:
        signal: Exit signal with symbol, type (stop_loss/take_profit/trailing_stop/prediction_expiry/adverse_move), reason

    Returns:
        True if order submitted successfully
    """
    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()
        symbol = signal.get("symbol")
        exit_type = signal.get("type")  # stop_loss, take_profit, trailing_stop, prediction_expiry, adverse_move
        reason = signal.get("reason", "")
        pnl_pct = signal.get("pnl_pct", 0)

        LOGGER.info(f"Executing {exit_type} exit for {symbol}: {reason}")

        # Close entire position (market order)
        result = broker.close_position(symbol)

        if result:
            LOGGER.info(
                f"✅ AUTO-EXIT SUCCESS: {symbol} closed via {exit_type} (P&L: {pnl_pct:+.2f}%)"
            )

            # Remove from position tracker
            if symbol in _position_tracker:
                del _position_tracker[symbol]

            # Log to database/events
            try:
                from wolf_app import _add_event

                _add_event(
                    "auto_exit",
                    f"{exit_type.upper()} triggered for {symbol}",
                    {
                        "symbol": symbol,
                        "type": exit_type,
                        "reason": reason,
                        "pnl_pct": pnl_pct,
                        "timestamp": int(time.time()),
                    },
                )
            except Exception:
                pass

            return True
        else:
            LOGGER.error(f"Failed to close position {symbol}")
            return False

    except Exception as e:
        LOGGER.error(f"Error executing exit for {signal.get('symbol')}: {e}", exc_info=True)
        return False


async def sl_tp_monitor_loop():
    """
    Main monitoring loop - runs continuously in background.
    Checks positions every CHECK_INTERVAL_SECONDS and auto-exits if SL/TP hit.
    Enhanced with trailing stops, prediction expiry, adverse move detection.
    """
    LOGGER.info(
        f"🔄 SL/TP Monitor started (interval={CHECK_INTERVAL_SECONDS}s, "
        f"SL={STOP_LOSS_PCT}%, TP={TAKE_PROFIT_PCT}%, "
        f"TrailingStop={TRAILING_STOP_ENABLED}, PredictionExpiry={PREDICTION_EXPIRY_HOURS}h)"
    )

    while True:
        try:
            # Check if monitor is enabled
            if not ENABLED:
                LOGGER.debug("SL/TP monitor disabled via env var")
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # Check for positions that need to be exited
            exit_signals = await check_positions_for_exits()

            # Execute exits
            for signal in exit_signals:
                try:
                    await execute_exit(signal)
                    # Small delay between orders to avoid rate limits
                    await asyncio.sleep(1)
                except Exception as e:
                    LOGGER.error(f"Error processing exit signal: {e}")

            # Wait before next check
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

        except Exception as e:
            LOGGER.error(f"Error in SL/TP monitor loop: {e}", exc_info=True)
            # Wait a bit longer after error
            await asyncio.sleep(CHECK_INTERVAL_SECONDS * 2)


async def start_sl_tp_monitor():
    """Start the SL/TP monitor as a background task."""
    if not ENABLED:
        LOGGER.info("SL/TP monitor is disabled (set SL_TP_MONITOR_ENABLED=1 to enable)")
        return

    # Run the monitoring loop
    await sl_tp_monitor_loop()


if __name__ == "__main__":
    # Run standalone
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("🔄 Starting Ghost SL/TP Monitor (standalone mode)")
    print(f"   Check interval: {CHECK_INTERVAL_SECONDS}s")
    print(f"   Stop Loss: -{STOP_LOSS_PCT}%")
    print(f"   Take Profit: +{TAKE_PROFIT_PCT}%")
    print(f"   Enabled: {ENABLED}")
    print()

    asyncio.run(sl_tp_monitor_loop())
