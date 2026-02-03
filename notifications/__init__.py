"""
Ghost Protocol - Notifications Package
======================================
Clean, tested notification components.

Modules:
- telegram: Telegram bot integration with rate limiting
- formatters: V3-aware message formatters for TOP 10
"""

from notifications.telegram import (
    TelegramNotifier,
    TelegramConfig,
    CooldownTracker,
    DailyCapTracker,
    send_telegram_message,
    get_notifier,
)

from notifications.formatters import (
    format_top10_message,
    format_v3_alert,
    format_price_alert,
    format_single_pick,
    TradePick,
)

__all__ = [
    # Telegram
    "TelegramNotifier",
    "TelegramConfig",
    "CooldownTracker",
    "DailyCapTracker",
    "send_telegram_message",
    "get_notifier",
    # Formatters
    "format_top10_message",
    "format_v3_alert",
    "format_price_alert",
    "format_single_pick",
    "TradePick",
]
