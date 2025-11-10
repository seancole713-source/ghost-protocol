"""
Prediction-to-Order Automation for Ghost Trading System
Converts prediction signals into executable Alpaca orders with risk management.
"""

import logging
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class SignalAction(str, Enum):
    """Trading signal actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class PositionSizingMethod(str, Enum):
    """Position sizing calculation methods."""

    FIXED_DOLLAR = "fixed_dollar"  # Fixed dollar amount per trade
    FIXED_SHARES = "fixed_shares"  # Fixed number of shares
    PERCENT_PORTFOLIO = "percent_portfolio"  # Percentage of portfolio value
    KELLY_CRITERION = "kelly"  # Kelly Criterion (requires win rate + odds)
    VOLATILITY_ADJUSTED = "volatility"  # Adjust size based on volatility


def interpret_prediction_signal(
    prediction: float,
    confidence: float = 0.5,
    threshold_buy: float = 0.05,
    threshold_sell: float = -0.05,
) -> tuple[SignalAction, float]:
    """
    Convert prediction percentage into trading signal.

    Args:
        prediction: Predicted price change (e.g., 0.03 = +3%)
        confidence: Prediction confidence (0-1)
        threshold_buy: Minimum prediction for buy signal (default +5%)
        threshold_sell: Maximum prediction for sell signal (default -5%)

    Returns:
        Tuple of (action, strength) where strength is 0-1
    """
    # Adjust prediction by confidence
    adjusted_prediction = prediction * confidence

    if adjusted_prediction >= threshold_buy:
        strength = min(adjusted_prediction / threshold_buy, 1.0)
        return SignalAction.BUY, strength
    elif adjusted_prediction <= threshold_sell:
        strength = min(abs(adjusted_prediction / threshold_sell), 1.0)
        return SignalAction.SELL, strength
    else:
        return SignalAction.HOLD, 0.0


def calculate_position_size(
    method: PositionSizingMethod,
    portfolio_value: float,
    current_price: float,
    signal_strength: float = 1.0,
    max_position_value: float | None = None,
    fixed_dollar: float = 1000.0,
    fixed_shares: float = 10.0,
    percent_portfolio: float = 0.02,
    volatility: float | None = None,
    win_rate: float | None = None,
    avg_win_loss_ratio: float | None = None,
) -> tuple[float, str]:
    """
    Calculate position size based on method and parameters.

    Args:
        method: Position sizing method
        portfolio_value: Total portfolio value
        current_price: Current price of asset
        signal_strength: Signal strength (0-1), scales position size
        max_position_value: Maximum position value (caps position)
        fixed_dollar: Dollar amount for FIXED_DOLLAR method
        fixed_shares: Share count for FIXED_SHARES method
        percent_portfolio: Portfolio percentage for PERCENT_PORTFOLIO method
        volatility: Annualized volatility (for VOLATILITY_ADJUSTED)
        win_rate: Historical win rate (for KELLY_CRITERION)
        avg_win_loss_ratio: Average win/loss ratio (for KELLY_CRITERION)

    Returns:
        Tuple of (shares, reason_string)
    """
    if current_price <= 0:
        return 0.0, "Invalid price"

    shares = 0.0
    reason = ""

    if method == PositionSizingMethod.FIXED_DOLLAR:
        dollar_amount = fixed_dollar * signal_strength
        shares = dollar_amount / current_price
        reason = f"Fixed ${dollar_amount:.2f} at ${current_price:.2f}/share"

    elif method == PositionSizingMethod.FIXED_SHARES:
        shares = fixed_shares * signal_strength
        reason = f"Fixed {fixed_shares} shares, scaled by signal {signal_strength:.2f}"

    elif method == PositionSizingMethod.PERCENT_PORTFOLIO:
        dollar_amount = portfolio_value * percent_portfolio * signal_strength
        shares = dollar_amount / current_price
        reason = f"{percent_portfolio * 100:.1f}% of ${portfolio_value:,.2f} portfolio"

    elif method == PositionSizingMethod.KELLY_CRITERION:
        if win_rate is None or avg_win_loss_ratio is None:
            return 0.0, "Kelly requires win_rate and avg_win_loss_ratio"

        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = 1-p, b = win/loss ratio
        kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% (fractional Kelly)

        dollar_amount = portfolio_value * kelly_fraction * signal_strength
        shares = dollar_amount / current_price
        reason = f"Kelly {kelly_fraction * 100:.1f}% of portfolio (win_rate={win_rate:.2f})"

    elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
        if volatility is None:
            return 0.0, "Volatility-adjusted sizing requires volatility parameter"

        # Inverse volatility: lower vol = larger position
        base_percent = percent_portfolio
        vol_adjustment = 0.20 / max(volatility, 0.05)  # Normalize to 20% vol
        adjusted_percent = base_percent * vol_adjustment * signal_strength
        adjusted_percent = min(adjusted_percent, 0.10)  # Cap at 10% of portfolio

        dollar_amount = portfolio_value * adjusted_percent
        shares = dollar_amount / current_price
        reason = f"Vol-adjusted {adjusted_percent * 100:.1f}% (vol={volatility * 100:.1f}%)"

    # Apply maximum position value cap
    if max_position_value and shares * current_price > max_position_value:
        original_shares = shares
        shares = max_position_value / current_price
        reason += f" (capped from {original_shares:.2f} shares)"

    return shares, reason


def build_order_from_prediction(
    symbol: str,
    prediction_pct: float,
    confidence: float,
    current_price: float,
    portfolio_value: float,
    existing_position_qty: float = 0.0,
    sizing_method: PositionSizingMethod = PositionSizingMethod.PERCENT_PORTFOLIO,
    **sizing_kwargs,
) -> dict[str, Any] | None:
    """
    Build complete order dict from prediction signal.

    Args:
        symbol: Stock symbol
        prediction_pct: Predicted price change as decimal (0.05 = +5%)
        confidence: Prediction confidence (0-1)
        current_price: Current market price
        portfolio_value: Total portfolio value
        existing_position_qty: Current position in shares (positive=long, negative=short)
        sizing_method: Position sizing method
        **sizing_kwargs: Additional kwargs for position sizing

    Returns:
        Order dict ready for broker submission, or None if HOLD signal
    """
    # Interpret signal
    action, strength = interpret_prediction_signal(prediction_pct, confidence)

    if action == SignalAction.HOLD:
        LOGGER.info(
            f"{symbol}: HOLD signal (prediction={prediction_pct:.2%}, confidence={confidence:.2f})"
        )
        return None

    # Calculate position size
    shares, reason = calculate_position_size(
        method=sizing_method,
        portfolio_value=portfolio_value,
        current_price=current_price,
        signal_strength=strength,
        **sizing_kwargs,
    )

    if shares < 0.01:  # Minimum viable position
        LOGGER.info(f"{symbol}: Position too small ({shares:.4f} shares), skipping")
        return None

    # Determine side based on action and existing position
    if action == SignalAction.BUY:
        side = "buy"
        # If already long, this adds to position
        # If short, this reduces/closes short
    elif action == SignalAction.SELL:
        side = "sell"
        # If already long, this reduces/closes long
        # If short, this adds to short position
    else:
        return None

    # Build order
    order = {
        "symbol": symbol.upper(),
        "qty": round(shares, 2),  # Round to 2 decimals for fractional shares
        "side": side,
        "type": "market",  # Default to market orders for execution certainty
        "time_in_force": "day",
        "prediction_pct": prediction_pct,
        "confidence": confidence,
        "signal_strength": strength,
        "sizing_reason": reason,
    }

    LOGGER.info(
        f"{symbol}: {action.upper()} signal - "
        f"qty={shares:.2f}, strength={strength:.2f}, "
        f"reason='{reason}'"
    )

    return order


def should_close_position(
    symbol: str,
    current_qty: float,
    current_price: float,
    entry_price: float,
    unrealized_pl_pct: float,
    prediction_pct: float,
    confidence: float,
    stop_loss_pct: float = -0.10,
    take_profit_pct: float = 0.20,
    reversal_threshold: float = -0.03,
) -> tuple[bool, str]:
    """
    Determine if existing position should be closed.

    Args:
        symbol: Stock symbol
        current_qty: Current position size (positive=long, negative=short)
        current_price: Current market price
        entry_price: Entry price for position
        unrealized_pl_pct: Unrealized P&L as percentage
        prediction_pct: New prediction for future price change
        confidence: Prediction confidence
        stop_loss_pct: Stop loss threshold (e.g., -0.10 = -10%)
        take_profit_pct: Take profit threshold (e.g., 0.20 = +20%)
        reversal_threshold: Prediction reversal threshold (e.g., -0.03 = -3%)

    Returns:
        Tuple of (should_close, reason)
    """
    if current_qty == 0:
        return False, "No position to close"

    is_long = current_qty > 0

    # Stop loss check
    if unrealized_pl_pct <= stop_loss_pct:
        return (
            True,
            f"Stop loss triggered at {unrealized_pl_pct:.2%} (threshold {stop_loss_pct:.2%})",
        )

    # Take profit check
    if unrealized_pl_pct >= take_profit_pct:
        return (
            True,
            f"Take profit triggered at {unrealized_pl_pct:.2%} (threshold {take_profit_pct:.2%})",
        )

    # Prediction reversal check
    adjusted_prediction = prediction_pct * confidence
    if is_long and adjusted_prediction <= reversal_threshold:
        return (
            True,
            f"Prediction reversed: {adjusted_prediction:.2%} (threshold {reversal_threshold:.2%})",
        )
    elif not is_long and adjusted_prediction >= -reversal_threshold:
        return (
            True,
            f"Prediction reversed: {adjusted_prediction:.2%} (threshold {-reversal_threshold:.2%})",
        )

    return False, "Hold position"


def create_close_order(
    symbol: str,
    current_qty: float,
    reason: str,
) -> dict[str, Any]:
    """
    Create order to close existing position.

    Args:
        symbol: Stock symbol
        current_qty: Current position size (will be closed completely)
        reason: Reason for closing position

    Returns:
        Order dict to close position
    """
    # To close: sell if long, buy if short
    close_side = "sell" if current_qty > 0 else "buy"
    close_qty = abs(current_qty)

    order = {
        "symbol": symbol.upper(),
        "qty": close_qty,
        "side": close_side,
        "type": "market",
        "time_in_force": "day",
        "close_reason": reason,
    }

    LOGGER.info(f"{symbol}: Closing position - qty={close_qty}, reason='{reason}'")

    return order


# Example usage:
"""
# Get prediction from Ghost
prediction = get_prediction("AAPL")  # Returns {"prediction_pct": 0.08, "confidence": 0.75}

# Get current state
broker = get_broker()
account = broker.get_account()
portfolio_value = float(account["portfolio_value"])
positions = broker.get_positions()
current_position = next((p for p in positions if p["symbol"] == "AAPL"), None)
current_qty = float(current_position["qty"]) if current_position else 0.0

# Get current price
price_data = fetch_price_live("AAPL")
current_price = price_data["price"]

# Check if should close existing position
if current_position:
    should_close, reason = should_close_position(
        symbol="AAPL",
        current_qty=current_qty,
        current_price=current_price,
        entry_price=float(current_position["avg_entry_price"]),
        unrealized_pl_pct=float(current_position["unrealized_plpc"]),
        prediction_pct=prediction["prediction_pct"],
        confidence=prediction["confidence"],
    )

    if should_close:
        close_order = create_close_order("AAPL", current_qty, reason)
        broker.submit_order(**close_order)

# Otherwise, check for new entry
else:
    order = build_order_from_prediction(
        symbol="AAPL",
        prediction_pct=prediction["prediction_pct"],
        confidence=prediction["confidence"],
        current_price=current_price,
        portfolio_value=portfolio_value,
        sizing_method=PositionSizingMethod.PERCENT_PORTFOLIO,
        percent_portfolio=0.05,  # 5% of portfolio per trade
    )

    if order:
        # Submit to risk engine and broker
        broker.submit_order(**order)
"""
