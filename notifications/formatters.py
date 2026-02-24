"""
Ghost Protocol - Message Formatters
===================================
V3-aware message formatting for Telegram alerts.

This module handles:
- TOP 10 trade plan formatting
- V3 inverse signal display
- Market hours awareness
- Mobile-friendly output

Usage:
    from notifications.formatters import format_top10_message
    
    messages = format_top10_message(stocks=[], crypto=[...])
    for msg in messages:
        send_telegram(msg)
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo


# ========================================
# CONSTANTS
# ========================================

# V3 Configuration
V3_MIN_SAMPLE_SIZE = 50  # Minimum trades for reliable win rate

# US Market Holidays (NYSE/NASDAQ) 2025-2027
US_MARKET_HOLIDAYS = {
    # 2025
    (2025, 1, 1), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18), 
    (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1),
    (2025, 11, 27), (2025, 12, 25),
    # 2026
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3),
    (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25),
    # 2027
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26),
    (2027, 5, 31), (2027, 6, 18), (2027, 7, 5), (2027, 9, 6),
    (2027, 11, 25), (2027, 12, 24),
}

HOLIDAY_NAMES = {
    (1, 1): "New Year's Day", (1, 18): "MLK Day", (1, 19): "MLK Day", (1, 20): "MLK Day",
    (2, 15): "Presidents Day", (2, 16): "Presidents Day", (2, 17): "Presidents Day",
    (3, 26): "Good Friday", (4, 3): "Good Friday", (4, 18): "Good Friday",
    (5, 25): "Memorial Day", (5, 26): "Memorial Day", (5, 31): "Memorial Day",
    (6, 18): "Juneteenth", (6, 19): "Juneteenth",
    (7, 3): "Independence Day", (7, 4): "Independence Day", (7, 5): "Independence Day",
    (9, 1): "Labor Day", (9, 6): "Labor Day", (9, 7): "Labor Day",
    (11, 25): "Thanksgiving", (11, 26): "Thanksgiving", (11, 27): "Thanksgiving",
    (12, 24): "Christmas", (12, 25): "Christmas",
}


# ========================================
# HELPER FUNCTIONS
# ========================================

def get_central_time() -> datetime:
    """Get current time in Central timezone"""
    try:
        return datetime.now(ZoneInfo("America/Chicago"))
    except Exception:
        return datetime.now()


def is_us_holiday(dt: datetime) -> bool:
    """Check if date is a US market holiday"""
    return (dt.year, dt.month, dt.day) in US_MARKET_HOLIDAYS


def get_holiday_name(dt: datetime) -> str:
    """Get the name of the holiday for a date"""
    return HOLIDAY_NAMES.get((dt.month, dt.day), "Market Holiday")


def is_trading_day(dt: datetime) -> bool:
    """Check if date is a trading day (weekday + not holiday)"""
    if dt.weekday() >= 5:  # Weekend
        return False
    if is_us_holiday(dt):
        return False
    return True


def get_next_trading_day(dt: datetime) -> datetime:
    """Get next trading day, skipping weekends and holidays"""
    while not is_trading_day(dt):
        dt = dt + timedelta(days=1)
    return dt


def format_price(price: float) -> str:
    """Format price with appropriate precision"""
    if price == 0:
        return "$0.00"
    if abs(price) >= 100:
        return f"${price:,.2f}"
    if abs(price) >= 1:
        return f"${price:.2f}"
    if abs(price) >= 0.01:
        return f"${price:.4f}"
    return f"${price:.6f}"


def calibrate_display_confidence(raw_conf: float, symbol: str = "") -> float:
    """
    Calibrate confidence for display.
    Delegates to the real calibration curve in core.ghost_notifications.
    """
    try:
        from core.ghost_notifications import calibrate_display_confidence as _real_calibrate
        return _real_calibrate(raw_conf, symbol)
    except ImportError:
        # Fallback: apply simple scaling if ghost_notifications unavailable
        return raw_conf * 0.7


def get_risk_level(conf: float) -> str:
    """Get risk level based on confidence"""
    if conf >= 0.60:
        return "Low"
    elif conf >= 0.40:
        return "Moderate"
    return "High"


def get_hold_reason(conf: float, hold_days: int) -> str:
    """Get intelligent hold reason based on confidence and days"""
    if hold_days == 1:
        return "RSI extreme" if conf >= 0.35 else "low conviction scalp"
    elif hold_days == 2:
        return "volatility swing"
    elif hold_days <= 4:
        return "swing trade"
    return "position trade"


def calculate_rr(entry: float, target: float, stop: float, direction: str) -> float:
    """Calculate risk/reward ratio"""
    if direction in ('UP', 'BUY'):
        reward = abs(target - entry)
        risk = abs(entry - stop)
    else:
        reward = abs(entry - target)
        risk = abs(stop - entry)
    return reward / risk if risk > 0 else 1.0


# ========================================
# DATA CLASSES
# ========================================

@dataclass
class TradePick:
    """
    A single trade pick with V3 metadata.
    Extracted from prediction dictionary for type safety.
    """
    symbol: str
    direction: str  # 'UP' or 'DOWN'
    current_price: float
    target_price: float
    stop_price: float
    confidence: float
    hold_days: int = 3
    news_influenced: bool = False
    volatility: float = 0.02
    expected_move_pct: float = 0.03
    
    # V3 fields
    v3_is_inverse: bool = False
    v3_original_direction: str = ""
    v3_historical_win_rate: float = 0.50
    v3_sample_size: int = 0
    v3_score: float = 0.0
    v3_is_whitelisted: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradePick":
        """Create TradePick from prediction dictionary"""
        current = data.get('current', data.get('current_price', 0))
        direction = data.get('direction', 'UP')
        is_buy = direction in ('UP', 'BUY')
        
        return cls(
            symbol=data.get('symbol', ''),
            direction=direction,
            current_price=current,
            target_price=data.get('prediction_48h', data.get('target_price', current)),
            stop_price=data.get('stop', current * 0.97 if is_buy else current * 1.03),
            confidence=data.get('confidence', 0.5),
            hold_days=data.get('hold_days', 3),
            news_influenced=data.get('news_influenced', False),
            volatility=data.get('volatility', 0.02),
            expected_move_pct=data.get('expected_move_pct', 0.03),
            v3_is_inverse=data.get('v3_is_inverse', False),
            v3_original_direction=data.get('v3_original_direction', ''),
            v3_historical_win_rate=data.get('v3_historical_win_rate', 0.50),
            v3_sample_size=data.get('v3_sample_size', 0),
            v3_score=data.get('v3_score', 0.0),
            v3_is_whitelisted=data.get('v3_is_whitelisted', False),
        )


# ========================================
# PICK FORMATTER
# ========================================

def format_single_pick(
    pick: TradePick,
    idx: int,
    is_stock: bool,
    ct: datetime,
    stock_entry_date: datetime
) -> str:
    """
    Format a single trade pick in detailed format.
    
    Example output:
    1) 🟢 ETH — BUY 🔄 INVERSE ✅
    🔄 Ghost predicted DOWN → FLIPPED to UP
    • BUY NOW (24/7): Entry Zone: $3,450.00 – $3,520.00
    • SELL: Mon Feb 03 | Target: $3,750.00 (+8.7%)
    • STOP LOSS: $3,346.00
    • Hold: 3 days (swing trade)
    • Confidence: 72% | Risk: Moderate | R/R: ~2.1 : 1 | Win Rate: 61%
    💰 $100 → $108.70
    ✅ News influence confirmed
    """
    symbol = pick.symbol
    direction = pick.direction
    is_buy = direction in ('UP', 'BUY')
    
    current = pick.current_price
    target = pick.target_price
    stop = pick.stop_price
    conf = calibrate_display_confidence(pick.confidence, symbol)
    hold_days = pick.hold_days
    
    # V3 data
    v3_inverse = pick.v3_is_inverse
    v3_original = pick.v3_original_direction
    v3_win_rate = pick.v3_historical_win_rate
    v3_sample = pick.v3_sample_size
    v3_whitelisted = pick.v3_is_whitelisted
    
    # Entry zone calculation
    volatility = pick.volatility
    daily_vol = volatility / 16 if volatility > 0.01 else abs(pick.expected_move_pct) / 3
    entry_range_pct = max(0.005, min(0.03, daily_vol / 2))
    entry_high = current * (1 + entry_range_pct)
    
    # Gain percentage
    if is_buy:
        gain_pct = ((target - current) / current * 100) if current > 0 else 0
    else:
        gain_pct = ((current - target) / current * 100) if current > 0 else 0
    
    # Calculated fields
    rr = calculate_rr(current, target, stop, direction)
    risk = get_risk_level(conf)
    hold_reason = get_hold_reason(conf, hold_days)
    return_val = 100 + abs(gain_pct)
    
    # Exit date calculation
    base_date = stock_entry_date if is_stock else ct
    exit_dt = base_date + timedelta(days=hold_days)
    if is_stock and exit_dt.weekday() >= 5:
        days_to_monday = 7 - exit_dt.weekday()
        exit_dt = exit_dt + timedelta(days=days_to_monday)
    exit_date = exit_dt.strftime("%a %b %d")
    
    # Entry date
    entry_date = stock_entry_date if is_stock else ct
    entry_day_name = entry_date.strftime("%a")
    
    # Emojis and badges
    emoji = "🟢" if is_buy else "🔴"
    action = "BUY" if is_buy else "SELL"
    inverse_badge = " 🔄 INVERSE" if v3_inverse else ""
    whitelist_badge = " ⭐" if v3_whitelisted else ""
    news_badge = " ✅" if pick.news_influenced else ""
    
    # Build lines
    lines = []
    lines.append(f"{idx}) {emoji} {symbol} — {action}{inverse_badge}{whitelist_badge}{news_badge}")
    
    # V3 inverse explanation
    if v3_inverse:
        lines.append(f"🔄 Ghost predicted {v3_original} → FLIPPED to {direction}")
    
    # Market closed warning for stocks
    is_market_closed = not is_trading_day(ct)
    if is_stock and is_market_closed:
        if is_us_holiday(ct):
            holiday = get_holiday_name(ct)
            lines.append(f"⚠️ MARKET CLOSED ({holiday}) - Execute {entry_day_name}")
        else:
            lines.append(f"⚠️ MARKET CLOSED (Weekend) - Execute {entry_day_name}")
    
    # Buy/Sell lines
    if is_buy:
        if is_stock:
            lines.append(f"• BUY (CT): {entry_day_name} {entry_date.strftime('%b %d')} @ 9:30 AM | Entry Zone: {format_price(current)} – {format_price(entry_high)}")
            lines.append(f"• SELL (CT): {exit_date} @ Open | Target: {format_price(target)} ({gain_pct:+.1f}%)")
        else:
            lines.append(f"• BUY NOW (24/7): Entry Zone: {format_price(current)} – {format_price(entry_high)}")
            lines.append(f"• SELL: {exit_date} | Target: {format_price(target)} ({gain_pct:+.1f}%)")
    else:
        if is_stock:
            lines.append(f"• SELL (CT): {entry_day_name} {entry_date.strftime('%b %d')} @ 9:30 AM | Entry Zone: {format_price(current)} – {format_price(entry_high)}")
            lines.append(f"• BUY-BACK (CT): {exit_date} @ Open | Target: {format_price(target)} (+{gain_pct:.1f}%)")
        else:
            lines.append(f"• SELL NOW (24/7): Entry Zone: {format_price(current)} – {format_price(entry_high)}")
            lines.append(f"• BUY-BACK: {exit_date} | Target: {format_price(target)} (+{gain_pct:.1f}%)")
    
    lines.append(f"• STOP LOSS: {format_price(stop)}")
    lines.append(f"• Hold: {hold_days} day{'s' if hold_days > 1 else ''} ({hold_reason})")
    
    # Win rate line with sample size awareness
    base_metrics = f"• Confidence: {conf:.0%} | Risk: {risk} | R/R: ~{rr:.1f} : 1"
    if v3_win_rate > 0 and v3_win_rate != 0.50:
        if v3_sample >= V3_MIN_SAMPLE_SIZE:
            lines.append(f"{base_metrics} | Win Rate: {v3_win_rate:.0%}")
        elif v3_sample > 0:
            lines.append(f"{base_metrics} | Win Rate: {v3_win_rate:.0%}* ({v3_sample} trades)")
        else:
            lines.append(base_metrics)
    else:
        lines.append(base_metrics)
    
    lines.append(f"💰 $100 → ${return_val:.2f}")
    
    if pick.news_influenced:
        lines.append("✅ News influence confirmed")
    
    return "\n".join(lines)


# ========================================
# TOP 10 MESSAGE FORMATTER
# ========================================

def format_top10_message(
    stocks: List[Dict[str, Any]],
    crypto: List[Dict[str, Any]],
    inverse_mode: bool = None
) -> List[str]:
    """
    Format TOP 10 message (up to 5 stocks + 5 crypto) - DETAILED TRADE PLAN format.
    
    Returns LIST of messages to fit Telegram 4096 char limit.
    
    Args:
        stocks: List of stock picks as dictionaries
        crypto: List of crypto picks as dictionaries
        inverse_mode: Override for inverse mode (default: from env)
        
    Returns:
        List of formatted messages (usually 1-2 messages)
    """
    if inverse_mode is None:
        inverse_mode = os.getenv("INVERSE_GHOST", "0") == "1"
    
    ct = get_central_time()
    date_str = ct.strftime("%b %d, %Y")
    
    # Calculate stock entry date
    is_market_closed = not is_trading_day(ct)
    stock_entry_date = get_next_trading_day(ct)
    stock_day_name = stock_entry_date.strftime("%a")
    
    # Build header
    all_lines = [
        "🎯 GHOST TOP 10 — TRADE PLAN (V3 VALIDATED)",
        f"📅 {date_str} | ⏰ 8:00 AM CT",
    ]
    
    # Market closed warning
    if is_market_closed:
        if is_us_holiday(ct):
            holiday = get_holiday_name(ct)
            all_lines.extend([
                "",
                f"⚠️ **{holiday.upper()} — STOCK MARKETS CLOSED**",
                f"📆 Next trading day: {stock_day_name} {stock_entry_date.strftime('%b %d')}",
                "🪙 Crypto trades 24/7 — execute anytime",
            ])
        else:
            all_lines.extend([
                "",
                "⚠️ **WEEKEND — STOCK MARKETS CLOSED**",
                f"📆 Next trading day: {stock_day_name} {stock_entry_date.strftime('%b %d')}",
                "🪙 Crypto trades 24/7 — execute anytime",
            ])
    
    # Add stocks section
    if stocks:
        all_lines.extend(["", "📈 **TOP STOCKS**", ""])
        for i, stock_data in enumerate(stocks[:5], 1):
            pick = TradePick.from_dict(stock_data)
            formatted = format_single_pick(
                pick, i, is_stock=True, 
                ct=ct, stock_entry_date=stock_entry_date
            )
            all_lines.append(formatted)
            all_lines.append("")  # Blank line between picks
    
    # Add crypto section
    if crypto:
        all_lines.extend(["", "🪙 **TOP CRYPTO**", ""])
        for i, crypto_data in enumerate(crypto[:5], 1):
            pick = TradePick.from_dict(crypto_data)
            formatted = format_single_pick(
                pick, i, is_stock=False,
                ct=ct, stock_entry_date=stock_entry_date
            )
            all_lines.append(formatted)
            all_lines.append("")
    
    # Add footer
    all_lines.extend([
        "━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ DYOR — Not financial advice",
        "📊 V3 Validated: Only p<0.05 strategies",
    ])
    
    # Build final message
    full_message = "\n".join(all_lines)
    
    # Split if too long for Telegram (4096 chars)
    if len(full_message) <= 4096:
        return [full_message]
    
    # Split at section boundary
    messages = []
    current = []
    current_len = 0
    
    for line in all_lines:
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > 4000:  # Leave margin
            messages.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    
    if current:
        messages.append("\n".join(current))
    
    return messages


# ========================================
# SIMPLE ALERT FORMATTERS
# ========================================

def format_v3_alert(
    symbol: str,
    direction: str,
    confidence: float,
    strategy: str,
    is_inverse: bool = False,
    hold_hours: int = 72,
    win_rate: float = 0.0
) -> str:
    """
    Format a simple V3 alert for individual signals.
    
    Example:
    🎯 V3 SIGNAL: ETH
    
    📈 Direction: UP (🔄 INVERSE)
    🎲 Strategy: ghost_inverse
    ⏱️ Hold: 72h
    📊 Confidence: 75%
    ✅ Win Rate: 61.5%
    """
    emoji = "📈" if direction == "UP" else "📉"
    inverse_badge = " (🔄 INVERSE)" if is_inverse else ""
    
    lines = [
        f"🎯 V3 SIGNAL: {symbol}",
        "",
        f"{emoji} Direction: {direction}{inverse_badge}",
        f"🎲 Strategy: {strategy}",
        f"⏱️ Hold: {hold_hours}h",
        f"📊 Confidence: {confidence:.0%}",
    ]
    
    if win_rate > 0:
        lines.append(f"✅ Win Rate: {win_rate:.1%}")
    
    return "\n".join(lines)


def format_price_alert(
    symbol: str,
    alert_type: str,  # 'target_hit' or 'stop_hit'
    entry_price: float,
    exit_price: float,
    target_price: float,
    stop_price: float
) -> str:
    """
    Format alert when target or stop is hit.
    
    Example:
    ✅ TARGET HIT: ETH
    
    📈 Entry: $3,450.00
    🎯 Target: $3,750.00
    💰 Exit: $3,755.00
    📊 Gain: +8.8%
    """
    if alert_type == 'target_hit':
        emoji = "✅"
        title = "TARGET HIT"
        gain = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        result_line = f"📊 Gain: {gain:+.1f}%"
    else:
        emoji = "❌"
        title = "STOP HIT"
        loss = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        result_line = f"📊 Loss: {loss:+.1f}%"
    
    lines = [
        f"{emoji} {title}: {symbol}",
        "",
        f"📈 Entry: {format_price(entry_price)}",
        f"🎯 Target: {format_price(target_price)}",
        f"🛑 Stop: {format_price(stop_price)}",
        f"💰 Exit: {format_price(exit_price)}",
        result_line,
    ]
    
    return "\n".join(lines)
