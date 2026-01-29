#!/usr/bin/env python3
"""
🎯 GHOST ADVISOR - SIMPLE TRADE MANAGEMENT

What the user ACTUALLY needs:

1. 📈 BUY ALERT: "Buy SPOT at $513, target $529, stop $500"
2. 📊 TRACKING: "SPOT is now $520 (+1.4%), 60% to target"
3. 🎯 SELL ALERT: "SELL SPOT NOW - Target hit at $529!" 
4. 🛑 STOP ALERT: "SELL SPOT NOW - Stop triggered at $500!"
5. ⏰ TIME UPDATE: "24h on SPOT: +1.2%, hold or take profit"

NO MORE:
- "Hold 24h" then silence
- Meaningless ✅ indicators  
- Predictions without follow-up

THIS ADVISOR:
- Tracks every position
- Sends alerts when action needed
- Tells you EXACTLY what to do
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

LOGGER = logging.getLogger("ghost.advisor")


class PositionStatus(Enum):
    OPEN = "open"           # Active position, watching
    TARGET_HIT = "target"   # Hit target - SELL NOW
    STOP_HIT = "stop"       # Hit stop - SELL NOW  
    EXPIRED = "expired"     # Time's up - decide
    CLOSED = "closed"       # Position closed


class AlertType(Enum):
    BUY = "buy"             # Entry signal
    UPDATE = "update"       # Progress update
    APPROACHING_TARGET = "approaching_target"
    APPROACHING_STOP = "approaching_stop"
    TARGET_HIT = "target_hit"   # SELL - you won!
    STOP_HIT = "stop_hit"       # SELL - cut loss
    TIME_CHECK = "time_check"   # Hold/sell decision
    EXPIRED = "expired"     # Time's up


@dataclass
class Position:
    """A tracked position"""
    symbol: str
    asset_type: str         # "stock" or "crypto"
    direction: str          # "BUY" or "SELL"
    entry_price: float
    target_price: float
    stop_price: float
    confidence: float
    hold_hours: int
    opened_at: datetime
    expires_at: datetime
    current_price: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    last_alert_at: Optional[datetime] = None
    last_alert_type: Optional[AlertType] = None
    peak_price: float = 0.0     # Best price seen
    worst_price: float = 0.0    # Worst price seen
    
    @property
    def pnl_pct(self) -> float:
        """Current P&L percentage"""
        if self.entry_price <= 0:
            return 0
        pct = (self.current_price - self.entry_price) / self.entry_price * 100
        # For SELL positions, profit is inverted
        if self.direction == "SELL":
            pct = -pct
        return pct
    
    @property
    def target_progress_pct(self) -> float:
        """How close to target (0% = at entry, 100% = at target)"""
        if self.direction in ("BUY", "UP"):
            total_move = self.target_price - self.entry_price
            current_move = self.current_price - self.entry_price
        else:
            total_move = self.entry_price - self.target_price
            current_move = self.entry_price - self.current_price
        
        if total_move == 0:
            return 0
        return (current_move / total_move) * 100
    
    @property
    def stop_distance_pct(self) -> float:
        """Distance from stop (positive = safe, negative = hit)"""
        if self.direction in ("BUY", "UP"):
            return (self.current_price - self.stop_price) / self.entry_price * 100
        else:
            return (self.stop_price - self.current_price) / self.entry_price * 100
    
    @property
    def hours_remaining(self) -> float:
        """Hours until expiration"""
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.total_seconds() / 3600)
    
    @property
    def is_target_hit(self) -> bool:
        """Check if target price reached"""
        if self.direction in ("BUY", "UP"):
            return self.current_price >= self.target_price
        else:
            return self.current_price <= self.target_price
    
    @property
    def is_stop_hit(self) -> bool:
        """Check if stop price triggered"""
        if self.direction in ("BUY", "UP"):
            return self.current_price <= self.stop_price
        else:
            return self.current_price >= self.stop_price


class GhostAdvisor:
    """
    🎯 THE ADVISOR - Tells you what to do
    
    - Tracks all positions
    - Checks prices regularly
    - Sends ACTIONABLE alerts
    """
    
    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._send_telegram = None
        self._get_price = None
        
    def set_telegram_sender(self, func):
        """Set the Telegram send function"""
        self._send_telegram = func
        
    def set_price_getter(self, func):
        """Set the price fetch function"""
        self._get_price = func
    
    def open_position(self, prediction: Dict) -> Position:
        """
        Open a new tracked position from a prediction.
        
        This is called when the TOP 10 is sent.
        """
        symbol = prediction.get("symbol", "")
        
        pos = Position(
            symbol=symbol,
            asset_type=prediction.get("asset_type", "stock"),
            direction=prediction.get("direction", "BUY"),
            entry_price=prediction.get("current", prediction.get("entry_price", 0)),
            target_price=prediction.get("prediction_48h", prediction.get("target_price", 0)),
            stop_price=prediction.get("stop", prediction.get("sell", 0)),
            confidence=prediction.get("confidence", 0.5),
            hold_hours=prediction.get("hold_hours", 48),
            opened_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=prediction.get("hold_hours", 48)),
            current_price=prediction.get("current", prediction.get("entry_price", 0)),
            peak_price=prediction.get("current", prediction.get("entry_price", 0)),
            worst_price=prediction.get("current", prediction.get("entry_price", 0)),
        )
        
        self._positions[symbol] = pos
        LOGGER.info(f"📈 [ADVISOR] Opened position: {symbol} {pos.direction} @ ${pos.entry_price:.2f}")
        return pos
    
    def update_price(self, symbol: str, new_price: float) -> Optional[Tuple[AlertType, Position]]:
        """
        Update price and check if alert needed.
        
        Returns (AlertType, Position) if alert should be sent, None otherwise.
        """
        if symbol not in self._positions:
            return None
            
        pos = self._positions[symbol]
        if pos.status != PositionStatus.OPEN:
            return None
        
        old_price = pos.current_price
        pos.current_price = new_price
        
        # Track peak/worst
        pos.peak_price = max(pos.peak_price, new_price)
        pos.worst_price = min(pos.worst_price, new_price)
        
        # Check for TARGET HIT - IMMEDIATE ALERT!
        if pos.is_target_hit:
            pos.status = PositionStatus.TARGET_HIT
            pos.last_alert_type = AlertType.TARGET_HIT
            pos.last_alert_at = datetime.utcnow()
            LOGGER.info(f"🎯 [ADVISOR] TARGET HIT: {symbol} at ${new_price:.2f}")
            return (AlertType.TARGET_HIT, pos)
        
        # Check for STOP HIT - IMMEDIATE ALERT!
        if pos.is_stop_hit:
            pos.status = PositionStatus.STOP_HIT
            pos.last_alert_type = AlertType.STOP_HIT
            pos.last_alert_at = datetime.utcnow()
            LOGGER.info(f"🛑 [ADVISOR] STOP HIT: {symbol} at ${new_price:.2f}")
            return (AlertType.STOP_HIT, pos)
        
        # Check for EXPIRATION
        if pos.hours_remaining <= 0:
            pos.status = PositionStatus.EXPIRED
            pos.last_alert_type = AlertType.EXPIRED
            pos.last_alert_at = datetime.utcnow()
            LOGGER.info(f"⏰ [ADVISOR] EXPIRED: {symbol} at ${new_price:.2f}")
            return (AlertType.EXPIRED, pos)
        
        # Check for APPROACHING TARGET (within 1%)
        if pos.target_progress_pct >= 80 and pos.last_alert_type != AlertType.APPROACHING_TARGET:
            pos.last_alert_type = AlertType.APPROACHING_TARGET
            pos.last_alert_at = datetime.utcnow()
            return (AlertType.APPROACHING_TARGET, pos)
        
        # Check for APPROACHING STOP (within 1%)
        if pos.stop_distance_pct < 1 and pos.last_alert_type != AlertType.APPROACHING_STOP:
            pos.last_alert_type = AlertType.APPROACHING_STOP
            pos.last_alert_at = datetime.utcnow()
            return (AlertType.APPROACHING_STOP, pos)
        
        # Periodic time check (every 8 hours if holding)
        if pos.last_alert_at:
            hours_since_alert = (datetime.utcnow() - pos.last_alert_at).total_seconds() / 3600
            if hours_since_alert >= 8:
                pos.last_alert_type = AlertType.TIME_CHECK
                pos.last_alert_at = datetime.utcnow()
                return (AlertType.TIME_CHECK, pos)
        
        return None
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position"""
        return self._positions.get(symbol)
    
    def close_position(self, symbol: str):
        """Mark position as closed"""
        if symbol in self._positions:
            self._positions[symbol].status = PositionStatus.CLOSED
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        all_pos = list(self._positions.values())
        closed = [p for p in all_pos if p.status in (
            PositionStatus.TARGET_HIT, 
            PositionStatus.STOP_HIT, 
            PositionStatus.EXPIRED,
            PositionStatus.CLOSED
        )]
        
        wins = len([p for p in closed if p.status == PositionStatus.TARGET_HIT])
        losses = len([p for p in closed if p.status == PositionStatus.STOP_HIT])
        expired = len([p for p in closed if p.status == PositionStatus.EXPIRED])
        
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return {
            "open_positions": len(self.get_open_positions()),
            "total_closed": len(closed),
            "wins": wins,
            "losses": losses,
            "expired": expired,
            "win_rate": win_rate
        }


# Global advisor instance
_ADVISOR: Optional[GhostAdvisor] = None

def get_advisor() -> GhostAdvisor:
    """Get the global advisor instance"""
    global _ADVISOR
    if _ADVISOR is None:
        _ADVISOR = GhostAdvisor()
    return _ADVISOR


# ============================================================================
# ALERT MESSAGE FORMATTING
# ============================================================================

def format_advisor_alert(alert_type: AlertType, position: Position) -> str:
    """
    Format a clear, actionable alert message.
    
    These are the messages users ACTUALLY need!
    """
    symbol = position.symbol
    price = position.current_price
    entry = position.entry_price
    target = position.target_price
    stop = position.stop_price
    pnl = position.pnl_pct
    direction = position.direction
    hours_left = position.hours_remaining
    
    emoji_dir = "🟢" if direction in ("BUY", "UP") else "🔴"
    pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
    
    if alert_type == AlertType.TARGET_HIT:
        return f"""🎯 TARGET HIT - SELL NOW!

{emoji_dir} {symbol} — {direction}
━━━━━━━━━━━━━━━━━━━━━
✅ TARGET REACHED!

💵 Entry: ${entry:.2f}
📍 Now: ${price:.2f}
🎯 Target: ${target:.2f}

{pnl_emoji} Profit: {pnl:+.1f}%
💰 $100 → ${100 * (1 + pnl/100):.2f}

⚡ ACTION: SELL NOW to lock in profit!
━━━━━━━━━━━━━━━━━━━━━
Ghost made you money. 💰"""

    elif alert_type == AlertType.STOP_HIT:
        return f"""🛑 STOP TRIGGERED - SELL NOW!

{emoji_dir} {symbol} — {direction}
━━━━━━━━━━━━━━━━━━━━━
⚠️ STOP LOSS HIT!

💵 Entry: ${entry:.2f}
📍 Now: ${price:.2f}
🛑 Stop: ${stop:.2f}

{pnl_emoji} Loss: {pnl:+.1f}%
💰 $100 → ${100 * (1 + pnl/100):.2f}

⚡ ACTION: SELL NOW to limit loss!
━━━━━━━━━━━━━━━━━━━━━
Cut your losses. Move on."""

    elif alert_type == AlertType.APPROACHING_TARGET:
        progress = position.target_progress_pct
        return f"""📊 APPROACHING TARGET

{emoji_dir} {symbol} — {direction}
━━━━━━━━━━━━━━━━━━━━━
🔥 {progress:.0f}% to target!

💵 Entry: ${entry:.2f}
📍 Now: ${price:.2f} ({pnl:+.1f}%)
🎯 Target: ${target:.2f}

⏰ {hours_left:.0f}h remaining

💡 TIP: Consider setting a trailing stop
━━━━━━━━━━━━━━━━━━━━━
Almost there! 🎯"""

    elif alert_type == AlertType.APPROACHING_STOP:
        return f"""⚠️ APPROACHING STOP

{emoji_dir} {symbol} — {direction}
━━━━━━━━━━━━━━━━━━━━━
🚨 DANGER ZONE!

💵 Entry: ${entry:.2f}
📍 Now: ${price:.2f} ({pnl:+.1f}%)
🛑 Stop: ${stop:.2f}

Distance to stop: {position.stop_distance_pct:.1f}%

💡 TIP: Consider exiting early
━━━━━━━━━━━━━━━━━━━━━
Protect your capital! 🛡️"""

    elif alert_type == AlertType.EXPIRED:
        action = "TAKE PROFIT" if pnl > 1 else "CUT LOSS" if pnl < -1 else "CLOSE POSITION"
        return f"""⏰ TIME'S UP

{emoji_dir} {symbol} — {direction}
━━━━━━━━━━━━━━━━━━━━━
Hold period complete!

💵 Entry: ${entry:.2f}
📍 Now: ${price:.2f}

{pnl_emoji} Result: {pnl:+.1f}%
💰 $100 → ${100 * (1 + pnl/100):.2f}

⚡ RECOMMENDED: {action}
━━━━━━━━━━━━━━━━━━━━━
Position expired. Make your decision."""

    elif alert_type == AlertType.TIME_CHECK:
        progress = position.target_progress_pct
        hours_passed = position.hold_hours - hours_left
        
        if pnl > 2:
            advice = "Consider taking partial profit"
        elif pnl < -2:
            advice = "Consider cutting losses"
        else:
            advice = "Hold for now, watching..."
        
        return f"""📊 POSITION UPDATE

{emoji_dir} {symbol} — {direction}
━━━━━━━━━━━━━━━━━━━━━
⏱️ {hours_passed:.0f}h in / {hours_left:.0f}h left

💵 Entry: ${entry:.2f}
📍 Now: ${price:.2f}
🎯 Target: ${target:.2f} ({progress:.0f}% there)
🛑 Stop: ${stop:.2f}

{pnl_emoji} P&L: {pnl:+.1f}%

💡 {advice}
━━━━━━━━━━━━━━━━━━━━━
Ghost is watching. 👁️"""

    else:
        return f"📊 {symbol}: ${price:.2f} ({pnl:+.1f}%)"


# ============================================================================
# PRICE CHECK LOOP
# ============================================================================

async def check_all_positions(get_price_func, send_telegram_func):
    """
    Check all open positions and send alerts.
    
    This should run every 5-15 minutes.
    """
    advisor = get_advisor()
    open_positions = advisor.get_open_positions()
    
    if not open_positions:
        return {"checked": 0, "alerts_sent": 0}
    
    alerts_sent = 0
    
    for pos in open_positions:
        try:
            # Get current price
            new_price = await get_price_func(pos.symbol, pos.asset_type)
            if not new_price:
                continue
            
            # Update and check for alert
            result = advisor.update_price(pos.symbol, new_price)
            
            if result:
                alert_type, updated_pos = result
                message = format_advisor_alert(alert_type, updated_pos)
                
                # Send alert
                success = send_telegram_func(message)
                if success:
                    alerts_sent += 1
                    LOGGER.info(f"📤 [ADVISOR] Sent {alert_type.value} alert for {pos.symbol}")
        
        except Exception as e:
            LOGGER.error(f"❌ [ADVISOR] Error checking {pos.symbol}: {e}")
    
    return {
        "checked": len(open_positions),
        "alerts_sent": alerts_sent
    }


# ============================================================================
# INTEGRATION WITH TOP 10
# ============================================================================

def register_top10_positions(stocks: List[Dict], crypto: List[Dict]):
    """
    Register positions from TOP 10 for tracking.
    
    Call this when TOP 10 is sent to Telegram.
    """
    advisor = get_advisor()
    
    for s in stocks:
        s["asset_type"] = "stock"
        advisor.open_position(s)
    
    for c in crypto:
        c["asset_type"] = "crypto"
        advisor.open_position(c)
    
    LOGGER.info(f"📊 [ADVISOR] Registered {len(stocks)} stocks + {len(crypto)} crypto for tracking")
    return len(stocks) + len(crypto)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    advisor = get_advisor()
    
    # Simulate a position
    test_pred = {
        "symbol": "SPOT",
        "direction": "BUY",
        "current": 513.0,
        "target_price": 529.0,
        "stop": 500.0,
        "confidence": 0.7,
        "hold_hours": 72,
        "asset_type": "stock"
    }
    
    pos = advisor.open_position(test_pred)
    print(f"Opened: {pos.symbol} @ ${pos.entry_price}")
    
    # Simulate price movement to target
    result = advisor.update_price("SPOT", 529.50)
    if result:
        alert_type, pos = result
        print(f"\n{format_advisor_alert(alert_type, pos)}")
