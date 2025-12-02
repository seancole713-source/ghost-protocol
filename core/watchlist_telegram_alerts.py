#!/usr/bin/env python3
"""
Ghost Protocol Watchlist Telegram Alerts
=========================================

Extends telegram_hunter pipeline with watchlist-specific alerts.

Alert Types:
- market_open: Fresh prediction at NYSE/Nasdaq open
- market_close: Fresh prediction at market close
- big_move: Symbol price moved ±threshold%
- target_hit: Prediction target reached (future enhancement)

Features:
- Watchlist event formatting (distinct from hunter alerts)
- Cooldown enforcement (4h per symbol per alert type)
- Rate limiting (max 5 alerts/hour global)
- Respects WATCHLIST_ALERTS_ENABLED flag
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Environment configuration
WATCHLIST_ALERTS_ENABLED = os.getenv("WATCHLIST_ALERTS_ENABLED", "1") == "1"
WATCHLIST_ALERTS_INCLUDE_OPEN_CLOSE = os.getenv("WATCHLIST_ALERTS_INCLUDE_OPEN_CLOSE", "1") == "1"
WATCHLIST_ALERTS_INCLUDE_BIG_MOVES = os.getenv("WATCHLIST_ALERTS_INCLUDE_BIG_MOVES", "1") == "1"
WATCHLIST_ALERT_COOLDOWN_HOURS = int(os.getenv("WATCHLIST_ALERT_COOLDOWN_HOURS", "4"))
WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR = int(os.getenv("WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR", "5"))

# Global rate limiter state
_ALERT_TIMESTAMPS: List[float] = []


class WatchlistTelegramAlerter:
    """
    Manages Telegram alerts for personal watchlist events.
    """

    def __init__(self):
        if not WATCHLIST_ALERTS_ENABLED:
            LOGGER.info("🔕 Watchlist Telegram alerts disabled (WATCHLIST_ALERTS_ENABLED=0)")

    def send_market_open_alert(self, symbol: str, asset_type: str, prediction: Dict[str, Any], price: float, owns_position: bool):
        """
        Send market open alert for a watchlist symbol.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            prediction: Dict with direction, confidence, expected_move
            price: Current price
            owns_position: TRUE if user owns this asset
        """
        if not WATCHLIST_ALERTS_ENABLED or not WATCHLIST_ALERTS_INCLUDE_OPEN_CLOSE:
            return

        alert_type = "open"

        # Check cooldown
        if not self._check_cooldown(symbol, alert_type):
            LOGGER.debug(f"⏳ Alert cooldown active for {symbol} (type: {alert_type})")
            return

        # Check global rate limit
        if not self._check_global_rate_limit():
            LOGGER.warning(f"⚠️  Global rate limit reached, skipping {symbol} alert")
            return

        # Format message
        message = self._format_market_event_message(symbol, asset_type, "MARKET OPEN", prediction, price, owns_position)

        # Send to Telegram
        success = self._send_telegram_message(message)

        # Log alert
        if success:
            self._log_alert(symbol, asset_type, alert_type, prediction, price, None, message)
            self._record_alert_sent()

    def send_market_close_alert(self, symbol: str, asset_type: str, prediction: Dict[str, Any], price: float, owns_position: bool):
        """
        Send market close alert for a watchlist symbol.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            prediction: Dict with direction, confidence, expected_move
            price: Current price
            owns_position: TRUE if user owns this asset
        """
        if not WATCHLIST_ALERTS_ENABLED or not WATCHLIST_ALERTS_INCLUDE_OPEN_CLOSE:
            return

        alert_type = "close"

        # Check cooldown
        if not self._check_cooldown(symbol, alert_type):
            LOGGER.debug(f"⏳ Alert cooldown active for {symbol} (type: {alert_type})")
            return

        # Check global rate limit
        if not self._check_global_rate_limit():
            LOGGER.warning(f"⚠️  Global rate limit reached, skipping {symbol} alert")
            return

        # Format message
        message = self._format_market_event_message(symbol, asset_type, "MARKET CLOSE", prediction, price, owns_position)

        # Send to Telegram
        success = self._send_telegram_message(message)

        # Log alert
        if success:
            self._log_alert(symbol, asset_type, alert_type, prediction, price, None, message)
            self._record_alert_sent()

    def send_big_move_alert(
        self, symbol: str, asset_type: str, prediction: Dict[str, Any], price: float, move_pct: float, owns_position: bool
    ):
        """
        Send big move alert for a watchlist symbol.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            prediction: Dict with direction, confidence, expected_move
            price: Current price
            move_pct: Price move % that triggered alert
            owns_position: TRUE if user owns this asset
        """
        if not WATCHLIST_ALERTS_ENABLED or not WATCHLIST_ALERTS_INCLUDE_BIG_MOVES:
            return

        alert_type = "big_move"

        # Check cooldown
        if not self._check_cooldown(symbol, alert_type):
            LOGGER.debug(f"⏳ Alert cooldown active for {symbol} (type: {alert_type})")
            return

        # Check global rate limit
        if not self._check_global_rate_limit():
            LOGGER.warning(f"⚠️  Global rate limit reached, skipping {symbol} alert")
            return

        # Format message
        message = self._format_big_move_message(symbol, asset_type, prediction, price, move_pct, owns_position)

        # Send to Telegram
        success = self._send_telegram_message(message)

        # Log alert
        if success:
            self._log_alert(symbol, asset_type, alert_type, prediction, price, move_pct, message)
            self._record_alert_sent()

    def _format_market_event_message(
        self, symbol: str, asset_type: str, event: str, prediction: Dict[str, Any], price: float, owns_position: bool
    ) -> str:
        """
        Format market open/close alert message.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            event: 'MARKET OPEN' or 'MARKET CLOSE'
            prediction: Dict with direction, confidence, expected_move
            price: Current price
            owns_position: TRUE if user owns this asset

        Returns:
            Formatted Telegram message
        """
        direction = prediction.get("direction", "FLAT")
        confidence = prediction.get("confidence", 0.0)
        expected_move = prediction.get("expected_move", 0.0)
        horizon_h = prediction.get("horizon_h", 48)

        # Direction emoji
        dir_emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "⚪"

        # Ownership status
        ownership_line = "✅ **You OWN this**" if owns_position else "⚠️ You DO NOT own this yet"

        message = f"""📌 **WATCHLIST** – {event}

🎯 **{symbol}** ({asset_type.upper()})
{dir_emoji} **{horizon_h}h Prediction:** {direction}
📊 **Confidence:** {confidence:.0%}
📈 **Expected Move:** {expected_move:+.1f}%
💰 **Current Price:** ${price:,.2f}

{ownership_line}

⏰ Ghost AI – {event} Signal
"""
        return message.strip()

    def _format_big_move_message(
        self, symbol: str, asset_type: str, prediction: Dict[str, Any], price: float, move_pct: float, owns_position: bool
    ) -> str:
        """
        Format big move alert message.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            prediction: Dict with direction, confidence, expected_move
            price: Current price
            move_pct: Price move % that triggered alert
            owns_position: TRUE if user owns this asset

        Returns:
            Formatted Telegram message
        """
        direction = prediction.get("direction", "FLAT")
        confidence = prediction.get("confidence", 0.0)
        expected_move = prediction.get("expected_move", 0.0)
        horizon_h = prediction.get("horizon_h", 48)

        # Direction emoji
        dir_emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "⚪"

        # Move emoji
        move_emoji = "🚀" if move_pct > 0 else "📉"

        # Ownership status
        ownership_line = "✅ **You OWN this**" if owns_position else "⚠️ You DO NOT own this yet"

        message = f"""📌 **WATCHLIST** – BIG MOVE DETECTED

🎯 **{symbol}** ({asset_type.upper()})
{move_emoji} **Price Move:** {move_pct:+.1f}% (last 15-60 min)
💰 **Current Price:** ${price:,.2f}

{dir_emoji} **{horizon_h}h Ghost Prediction:** {direction}
📊 **Confidence:** {confidence:.0%}
📈 **Expected Move:** {expected_move:+.1f}%

{ownership_line}

⚡ Ghost AI – Intraday Alert
"""
        return message.strip()

    def _check_cooldown(self, symbol: str, alert_type: str) -> bool:
        """
        Check if alert can be sent (cooldown enforcement).

        Args:
            symbol: Ticker symbol
            alert_type: 'open', 'close', 'big_move'

        Returns:
            TRUE if alert can be sent
        """
        try:
            from core.personal_watchlist import get_personal_watchlist_manager

            pwm = get_personal_watchlist_manager()
            return pwm.check_alert_cooldown(symbol, alert_type, cooldown_hours=WATCHLIST_ALERT_COOLDOWN_HOURS)
        except Exception as e:
            LOGGER.error(f"❌ Cooldown check failed for {symbol}: {e}")
            return True  # Fail open

    def _check_global_rate_limit(self) -> bool:
        """
        Check global rate limit (max N alerts per hour).

        Returns:
            TRUE if under rate limit
        """
        global _ALERT_TIMESTAMPS

        now = time.time()
        one_hour_ago = now - 3600

        # Clean old timestamps
        _ALERT_TIMESTAMPS = [ts for ts in _ALERT_TIMESTAMPS if ts > one_hour_ago]

        # Check limit
        if len(_ALERT_TIMESTAMPS) >= WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR:
            return False

        return True

    def _record_alert_sent(self):
        """Record that an alert was sent (for rate limiting)."""
        global _ALERT_TIMESTAMPS
        _ALERT_TIMESTAMPS.append(time.time())

    def _send_telegram_message(self, message: str) -> bool:
        """
        Send message to Telegram using existing hunter pipeline.

        Args:
            message: Formatted message text

        Returns:
            TRUE if sent successfully
        """
        try:
            # Import telegram hunter
            from services.telegram_hunter import send_telegram_alert

            result = send_telegram_alert(message, alert_type="watchlist")
            return result.get("ok", False)
        except ImportError:
            LOGGER.warning("⚠️  telegram_hunter not available, message not sent")
            return False
        except Exception as e:
            LOGGER.error(f"❌ Failed to send Telegram message: {e}")
            return False

    def _log_alert(
        self,
        symbol: str,
        asset_type: str,
        alert_type: str,
        prediction: Dict[str, Any],
        price: float,
        move_pct: Optional[float],
        message: str,
    ):
        """
        Log alert to database.

        Args:
            symbol: Ticker symbol
            asset_type: 'crypto' or 'stock'
            alert_type: 'open', 'close', 'big_move'
            prediction: Dict with direction, confidence, expected_move
            price: Current price
            move_pct: Price move % (for big_move alerts)
            message: Alert message text
        """
        try:
            from core.personal_watchlist import get_personal_watchlist_manager

            pwm = get_personal_watchlist_manager()
            watchlist_items = pwm.get_watchlist(active_only=True)

            # Find matching watchlist item
            watchlist_item_id = None
            for item in watchlist_items:
                if item["symbol"] == symbol and item["asset_type"] == asset_type:
                    watchlist_item_id = item["id"]
                    break

            if watchlist_item_id:
                pwm.log_alert(
                    watchlist_item_id=watchlist_item_id,
                    symbol=symbol,
                    alert_type=alert_type,
                    direction=prediction.get("direction"),
                    confidence=prediction.get("confidence"),
                    expected_move_pct=prediction.get("expected_move"),
                    current_price=price,
                    change_pct=move_pct,
                    message=message,
                    telegram_sent=True,
                    telegram_chat_id=None,  # TODO: Get from config
                )
            else:
                LOGGER.warning(f"⚠️  Could not log alert: watchlist item not found for {symbol}")

        except Exception as e:
            LOGGER.error(f"❌ Failed to log alert for {symbol}: {e}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_WATCHLIST_TELEGRAM_ALERTER = None


def get_watchlist_telegram_alerter() -> WatchlistTelegramAlerter:
    """Get singleton instance of WatchlistTelegramAlerter."""
    global _WATCHLIST_TELEGRAM_ALERTER
    if _WATCHLIST_TELEGRAM_ALERTER is None:
        _WATCHLIST_TELEGRAM_ALERTER = WatchlistTelegramAlerter()
    return _WATCHLIST_TELEGRAM_ALERTER


def send_watchlist_alert_if_needed(
    symbol: str, asset_type: str, alert_type: str, prediction: Dict[str, Any], price: float, owns_position: bool, **kwargs
):
    """
    Convenience function to send watchlist alert based on type.

    Args:
        symbol: Ticker symbol
        asset_type: 'crypto' or 'stock'
        alert_type: 'open', 'close', 'big_move'
        prediction: Dict with direction, confidence, expected_move
        price: Current price
        owns_position: TRUE if user owns this asset
        **kwargs: Additional args (e.g., move_pct for big_move)
    """
    alerter = get_watchlist_telegram_alerter()

    if alert_type == "open":
        alerter.send_market_open_alert(symbol, asset_type, prediction, price, owns_position)
    elif alert_type == "close":
        alerter.send_market_close_alert(symbol, asset_type, prediction, price, owns_position)
    elif alert_type == "big_move":
        move_pct = kwargs.get("move_pct", 0.0)
        alerter.send_big_move_alert(symbol, asset_type, prediction, price, move_pct, owns_position)
    else:
        LOGGER.warning(f"⚠️  Unknown alert type: {alert_type}")
