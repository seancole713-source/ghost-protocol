"""
Phase 8: Alert System
Sends notifications for trades, circuit breakers, and performance milestones.
Supports Slack, Discord, and Email.
Also includes original price alert functionality.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, UTC
from typing import Any
import aiohttp

LOGGER = logging.getLogger(__name__)

# Alert configuration from environment
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")

# Alert storage (original price alerts)
_ALERTS: dict[str, list[dict[str, Any]]] = {}
_ALERT_HISTORY: list[dict[str, Any]] = []


class TradingAlertSystem:
    """Manage trading alerts and notifications."""
    
    def __init__(self):
        self.enabled = bool(SLACK_WEBHOOK_URL or DISCORD_WEBHOOK_URL or ALERT_EMAIL)
        self.daily_pnl_threshold = float(os.getenv("ALERT_DAILY_PNL_THRESHOLD", "1000"))
        self.circuit_breaker_alert_sent = False
    
    async def send_trade_alert(self, trade_data: dict[str, Any]) -> None:
        """Send alert for trade execution."""
        if not self.enabled:
            return
        
        symbol = trade_data.get("symbol", "UNKNOWN")
        side = trade_data.get("side", "").upper()
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        pnl = trade_data.get("pnl", 0)
        
        emoji = "📈" if side == "BUY" else "📉"
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        
        message = f"""{emoji} **Trade Executed**
Symbol: {symbol}
Side: {side}
Quantity: {quantity}
Price: ${price:.2f}
{f'P&L: {pnl_emoji} ${pnl:.2f}' if pnl != 0 else ''}
Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"""
        
        await self._send_alert(message, "trade")
    
    async def send_circuit_breaker_alert(self, reason: str, metrics: dict) -> None:
        """Send alert when circuit breaker is triggered."""
        if not self.enabled or self.circuit_breaker_alert_sent:
            return
        
        self.circuit_breaker_alert_sent = True
        
        message = f"""🚨 **CIRCUIT BREAKER ACTIVATED** 🚨
Reason: {reason}
Daily P&L: ${metrics.get('daily_pnl', 0):.2f}
Total Trades: {metrics.get('total_trades', 0)}
Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%
Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}

⚠️ Autonomous trading PAUSED. Manual review required."""
        
        await self._send_alert(message, "critical")
    
    async def send_daily_summary(self, metrics: dict[str, Any]) -> None:
        """Send daily P&L summary."""
        if not self.enabled:
            return
        
        daily_pnl = metrics.get("daily_pnl", 0)
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        
        emoji = "🎉" if daily_pnl > 0 else "📊" if daily_pnl == 0 else "⚠️"
        
        message = f"""{emoji} **Daily Trading Summary**
Date: {datetime.now(UTC).strftime('%Y-%m-%d')}
Daily P&L: ${daily_pnl:.2f}
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Total P&L: ${metrics.get('total_pnl', 0):.2f}
Current Drawdown: {metrics.get('current_drawdown', 0) * 100:.2f}%"""
        
        await self._send_alert(message, "summary")
    
    async def send_milestone_alert(self, milestone: str, value: float) -> None:
        """Send alert for performance milestones."""
        if not self.enabled:
            return
        
        messages = {
            "profit_target": f"🎯 **Profit Target Reached!**\nTotal P&L: ${value:.2f}",
            "trade_count": f"📊 **Milestone: {int(value)} Trades Executed**",
            "win_streak": f"🔥 **Win Streak: {int(value)} consecutive wins!**",
            "sharpe_ratio": f"📈 **High Sharpe Ratio: {value:.2f}**"
        }
        
        message = messages.get(milestone, f"🎉 Milestone: {milestone} = {value}")
        await self._send_alert(message, "milestone")
    
    async def _send_alert(self, message: str, alert_type: str) -> None:
        """Send alert to all configured channels."""
        tasks = []
        
        if SLACK_WEBHOOK_URL:
            tasks.append(self._send_slack(message, alert_type))
        
        if DISCORD_WEBHOOK_URL:
            tasks.append(self._send_discord(message, alert_type))
        
        if ALERT_EMAIL and SENDGRID_API_KEY:
            tasks.append(self._send_email(message, alert_type))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_slack(self, message: str, alert_type: str) -> None:
        """Send alert to Slack."""
        try:
            color = {
                "trade": "#36a64f",
                "critical": "#ff0000",
                "summary": "#0066cc",
                "milestone": "#ffcc00"
            }.get(alert_type, "#808080")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "text": message,
                    "mrkdwn_in": ["text"]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(SLACK_WEBHOOK_URL, json=payload) as resp:
                    if resp.status == 200:
                        LOGGER.info(f"[ALERTS] Slack alert sent: {alert_type}")
                    else:
                        LOGGER.warning(f"[ALERTS] Slack failed: {resp.status}")
        
        except Exception as e:
            LOGGER.error(f"[ALERTS] Slack error: {e}")
    
    async def _send_discord(self, message: str, alert_type: str) -> None:
        """Send alert to Discord."""
        try:
            color = {
                "trade": 0x36a64f,
                "critical": 0xff0000,
                "summary": 0x0066cc,
                "milestone": 0xffcc00
            }.get(alert_type, 0x808080)
            
            payload = {
                "embeds": [{
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now(UTC).isoformat()
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(DISCORD_WEBHOOK_URL, json=payload) as resp:
                    if resp.status in (200, 204):
                        LOGGER.info(f"[ALERTS] Discord alert sent: {alert_type}")
                    else:
                        LOGGER.warning(f"[ALERTS] Discord failed: {resp.status}")
        
        except Exception as e:
            LOGGER.error(f"[ALERTS] Discord error: {e}")
    
    async def _send_email(self, message: str, alert_type: str) -> None:
        """Send alert via SendGrid email."""
        try:
            subject_map = {
                "trade": "🔔 Trade Executed",
                "critical": "🚨 CIRCUIT BREAKER ALERT",
                "summary": "📊 Daily Trading Summary",
                "milestone": "🎯 Trading Milestone"
            }
            subject = subject_map.get(alert_type, "Ghost Protocol Alert")
            
            payload = {
                "personalizations": [{
                    "to": [{"email": ALERT_EMAIL}],
                    "subject": subject
                }],
                "from": {"email": "alerts@ghost-protocol.com", "name": "Ghost Protocol"},
                "content": [{
                    "type": "text/plain",
                    "value": message
                }]
            }
            
            headers = {
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status == 202:
                        LOGGER.info(f"[ALERTS] Email alert sent: {alert_type}")
                    else:
                        LOGGER.warning(f"[ALERTS] Email failed: {resp.status}")
        
        except Exception as e:
            LOGGER.error(f"[ALERTS] Email error: {e}")


# Global alert system instance
_trading_alerts = TradingAlertSystem()


async def send_trade_alert(trade_data: dict) -> None:
    """Send trade execution alert."""
    await _trading_alerts.send_trade_alert(trade_data)


async def send_circuit_breaker_alert(reason: str, metrics: dict) -> None:
    """Send circuit breaker alert."""
    await _trading_alerts.send_circuit_breaker_alert(reason, metrics)


async def send_daily_summary(metrics: dict) -> None:
    """Send daily summary alert."""
    await _trading_alerts.send_daily_summary(metrics)


async def send_milestone_alert(milestone: str, value: float) -> None:
    """Send milestone alert."""
    await _trading_alerts.send_milestone_alert(milestone, value)


def create_alert(
    symbol: str,
    alert_type: str,
    target_price: float | None = None,
    trigger_condition: str | None = None,
    message: str | None = None
) -> dict[str, Any]:
    """
    Create a price alert.

    Alert types:
    - "price_above": Trigger when price goes above target
    - "price_below": Trigger when price goes below target
    - "gain_pct": Trigger when gain % threshold hit
    - "confidence_spike": Trigger when confidence jumps significantly
    - "momentum_shift": Trigger when momentum changes dramatically

    Args:
        symbol: Stock/crypto ticker
        alert_type: Type of alert
        target_price: Price threshold (for price alerts)
        trigger_condition: Custom condition string
        message: Custom alert message

    Returns:
        Alert configuration
    """
    if symbol not in _ALERTS:
        _ALERTS[symbol] = []

    alert = {
        "id": f"{symbol}_{alert_type}_{int(time.time())}",
        "symbol": symbol,
        "alert_type": alert_type,
        "target_price": target_price,
        "trigger_condition": trigger_condition,
        "message": message or f"{symbol} {alert_type} alert triggered",
        "created_at": time.time(),
        "triggered": False,
        "active": True
    }

    _ALERTS[symbol].append(alert)
    return alert


def check_alerts(symbol: str, current_price: float, forecast_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Check if any alerts should trigger for a symbol.

    Args:
        symbol: Stock/crypto ticker
        current_price: Current price
        forecast_data: Latest forecast with confidence, momentum, etc.

    Returns:
        List of triggered alerts
    """
    if symbol not in _ALERTS:
        return []

    triggered_alerts = []

    for alert in _ALERTS[symbol]:
        if not alert["active"] or alert["triggered"]:
            continue

        should_trigger = False

        if alert["alert_type"] == "price_above":
            if current_price >= alert["target_price"]:
                should_trigger = True

        elif alert["alert_type"] == "price_below":
            if current_price <= alert["target_price"]:
                should_trigger = True

        elif alert["alert_type"] == "confidence_spike":
            confidence = forecast_data.get("confidence", 0)
            if confidence >= 0.85:  # High confidence threshold
                should_trigger = True

        elif alert["alert_type"] == "momentum_shift":
            gain_pct = forecast_data.get("gain_potential_pct", 0)
            if abs(gain_pct) > 5.0:  # >5% move
                should_trigger = True

        if should_trigger:
            alert["triggered"] = True
            alert["triggered_at"] = time.time()
            alert["trigger_price"] = current_price
            triggered_alerts.append(alert)
            _ALERT_HISTORY.append(alert.copy())

    return triggered_alerts


def get_active_alerts(symbol: str | None = None) -> list[dict[str, Any]]:
    """Get all active alerts (optionally filtered by symbol)."""
    if symbol:
        return [a for a in _ALERTS.get(symbol, []) if a["active"] and not a["triggered"]]

    all_alerts = []
    for alerts_list in _ALERTS.values():
        all_alerts.extend([a for a in alerts_list if a["active"] and not a["triggered"]])
    return all_alerts


def get_alert_history(limit: int = 50) -> list[dict[str, Any]]:
    """Get recent triggered alerts."""
    return _ALERT_HISTORY[-limit:]


def delete_alert(alert_id: str) -> dict[str, Any]:
    """Delete/deactivate an alert."""
    for symbol, alerts in _ALERTS.items():
        for alert in alerts:
            if alert["id"] == alert_id:
                alert["active"] = False
                return {"ok": True, "message": f"Alert {alert_id} deleted"}

    return {"ok": False, "error": "Alert not found"}
