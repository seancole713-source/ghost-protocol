#!/usr/bin/env python3
"""
🎯 GHOST TOP 10 AGGREGATOR - Combines predictions into ONE message

THE FIX: Instead of sending 10 separate Telegram notifications,
this module collects predictions and sends ONE consolidated "TOP 10" message.

FLOW:
1. Predictions come in throughout the scan cycle
2. This module queues them (up to 10: 5 crypto + 5 stocks)
3. Once queue is full OR scan cycle ends → send ONE combined message
4. Register picks with Active Tracking System for 48h monitoring

NO MORE SPAM. ONE notification with all picks.
"""

import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

LOGGER = logging.getLogger("ghost.top10_aggregator")

# ============================================================================
# CONFIGURATION
# ============================================================================

# How many picks per category
MAX_CRYPTO_PICKS = int(os.getenv("GHOST_TOP_CRYPTO", "5"))
MAX_STOCK_PICKS = int(os.getenv("GHOST_TOP_STOCKS", "5"))

# How long to wait for picks before sending (allows batch collection)
AGGREGATION_WINDOW_SECONDS = int(os.getenv("GHOST_AGGREGATION_WINDOW_S", "120"))  # 2 minutes

# Minimum confidence to include in TOP 10
MIN_TOP_10_CONFIDENCE = float(os.getenv("GHOST_TOP_10_MIN_CONF", "0.85"))

# Whether to send individual alerts (if False, only TOP 10)
INDIVIDUAL_ALERTS_ENABLED = os.getenv("INDIVIDUAL_ALERTS_ENABLED", "0") == "1"


@dataclass
class QueuedPick:
    """A prediction queued for the TOP 10"""
    symbol: str
    asset_type: str  # 'crypto' or 'stock'
    direction: str   # 'UP' or 'DOWN'
    entry_price: float
    target_price: float
    stop_price: float
    confidence: float
    queued_at: float
    prediction_id: int = 0
    reasons: str = ""
    
    @property
    def target_pct(self) -> float:
        """Target % change from entry"""
        if self.entry_price <= 0:
            return 0.0
        return ((self.target_price - self.entry_price) / self.entry_price) * 100


class Top10Aggregator:
    """
    Collects predictions and sends them as ONE combined TOP 10 message.
    
    Usage:
        aggregator = get_top10_aggregator()
        
        # Instead of sending individual alerts:
        aggregator.add_prediction(symbol, prediction_dict)
        
        # When ready (auto-triggered or manual):
        aggregator.send_top_10()
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._crypto_picks: List[QueuedPick] = []
        self._stock_picks: List[QueuedPick] = []
        self._first_pick_time: Optional[float] = None
        self._last_sent_date: Optional[str] = None
        self._send_telegram_func: Optional[Callable] = None
        self._timer: Optional[threading.Timer] = None
        
        LOGGER.info("[TOP 10] Aggregator initialized")
    
    def set_telegram_func(self, func: Callable[[str], bool]):
        """Set the function used to send Telegram messages"""
        self._send_telegram_func = func
    
    def add_prediction(
        self,
        symbol: str,
        asset_type: str,
        direction: str,
        entry_price: float,
        target_price: float,
        stop_price: float,
        confidence: float,
        prediction_id: int = 0,
        reasons: str = ""
    ) -> bool:
        """
        Add a prediction to the TOP 10 queue.
        
        Returns True if added, False if rejected (below confidence, full queue, etc.)
        """
        with self._lock:
            # Check confidence threshold
            if confidence < MIN_TOP_10_CONFIDENCE:
                LOGGER.debug(f"[TOP 10] Rejected {symbol}: confidence {confidence:.0%} < {MIN_TOP_10_CONFIDENCE:.0%}")
                return False
            
            # Check if we've already sent today
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if self._last_sent_date == today:
                LOGGER.debug(f"[TOP 10] Rejected {symbol}: already sent TOP 10 today")
                return False
            
            # Determine which queue
            is_crypto = asset_type.lower() == "crypto"
            queue = self._crypto_picks if is_crypto else self._stock_picks
            max_picks = MAX_CRYPTO_PICKS if is_crypto else MAX_STOCK_PICKS
            
            # Check if queue is full
            if len(queue) >= max_picks:
                # Replace if this one has higher confidence
                lowest = min(queue, key=lambda p: p.confidence)
                if confidence > lowest.confidence:
                    queue.remove(lowest)
                    LOGGER.info(f"[TOP 10] Replaced {lowest.symbol} ({lowest.confidence:.0%}) with {symbol} ({confidence:.0%})")
                else:
                    LOGGER.debug(f"[TOP 10] Queue full, {symbol} not better than existing picks")
                    return False
            
            # Check for duplicate
            if any(p.symbol == symbol for p in queue):
                LOGGER.debug(f"[TOP 10] Duplicate {symbol} ignored")
                return False
            
            # Add to queue
            pick = QueuedPick(
                symbol=symbol,
                asset_type=asset_type,
                direction=direction,
                entry_price=entry_price,
                target_price=target_price,
                stop_price=stop_price,
                confidence=confidence,
                prediction_id=prediction_id,
                reasons=reasons,
                queued_at=time.time()
            )
            queue.append(pick)
            
            # Sort by confidence (highest first)
            queue.sort(key=lambda p: p.confidence, reverse=True)
            
            # Track first pick time for aggregation window
            if self._first_pick_time is None:
                self._first_pick_time = time.time()
                self._start_aggregation_timer()
            
            total = len(self._crypto_picks) + len(self._stock_picks)
            LOGGER.info(f"[TOP 10] Queued {symbol} ({asset_type}, {confidence:.0%}) - Total: {total}/10")
            
            # Auto-send if we have full complement
            if len(self._crypto_picks) >= MAX_CRYPTO_PICKS and len(self._stock_picks) >= MAX_STOCK_PICKS:
                LOGGER.info("[TOP 10] Queue full (10 picks), sending combined message...")
                self._send_combined_message()
            
            return True
    
    def _start_aggregation_timer(self):
        """Start a timer to send TOP 10 after aggregation window"""
        if self._timer:
            self._timer.cancel()
        
        self._timer = threading.Timer(AGGREGATION_WINDOW_SECONDS, self._on_aggregation_timeout)
        self._timer.daemon = True
        self._timer.start()
        LOGGER.info(f"[TOP 10] Aggregation timer started ({AGGREGATION_WINDOW_SECONDS}s)")
    
    def _on_aggregation_timeout(self):
        """Called when aggregation window expires"""
        with self._lock:
            total = len(self._crypto_picks) + len(self._stock_picks)
            if total > 0:
                LOGGER.info(f"[TOP 10] Aggregation window expired with {total} picks, sending...")
                self._send_combined_message()
            else:
                LOGGER.debug("[TOP 10] Aggregation window expired with no picks")
    
    def _send_combined_message(self):
        """
        DISABLED - Use ghost_notifications.py instead.
        
        This function had wrong color logic:
            emoji = "🔴" if p.direction == "DOWN" else "🟢"
        
        Should be:
            emoji = "🔴" if target_price < entry_price else "🟢"
        """
        LOGGER.warning("[TOP 10 AGGREGATOR] _send_combined_message DISABLED - use ghost_notifications.py")
        return  # Never send from here
    
    def _send_combined_message_ORIGINAL_DISABLED(self):
        """OLD CODE - Send the combined TOP 10 message"""
        if not self._send_telegram_func:
            LOGGER.warning("[TOP 10] No Telegram function configured")
            return
        
        if not self._crypto_picks and not self._stock_picks:
            LOGGER.info("[TOP 10] No picks to send")
            return
        
        # Build the message
        today_str = datetime.utcnow().strftime("%B %d, %Y")
        inverse_mode = os.getenv("INVERSE_GHOST_MODE", "1") == "1"
        direction_label = "INVERSE GHOST" if inverse_mode else "GHOST"
        
        lines = [
            f"🔮 **{direction_label} TOP 10 — {today_str}**",
            "",
        ]
        
        # Crypto picks
        if self._crypto_picks:
            lines.append("🪙 **CRYPTO:**")
            for i, p in enumerate(self._crypto_picks[:MAX_CRYPTO_PICKS], 1):
                emoji = "🔴" if p.direction == "DOWN" else "🟢"
                
                # Smart price formatting
                if p.entry_price >= 1000:
                    entry_fmt = f"${p.entry_price:,.0f}"
                    target_fmt = f"${p.target_price:,.0f}"
                elif p.entry_price >= 1:
                    entry_fmt = f"${p.entry_price:,.2f}"
                    target_fmt = f"${p.target_price:,.2f}"
                elif p.entry_price >= 0.01:
                    entry_fmt = f"${p.entry_price:.4f}"
                    target_fmt = f"${p.target_price:.4f}"
                else:
                    entry_fmt = f"${p.entry_price:.8f}"
                    target_fmt = f"${p.target_price:.8f}"
                
                lines.append(
                    f"{i}. **{p.symbol}** {emoji} {p.direction} | "
                    f"{entry_fmt} → {target_fmt} ({p.target_pct:+.1f}%) | "
                    f"{p.confidence*100:.0f}%"
                )
            lines.append("")
        
        # Stock picks
        if self._stock_picks:
            lines.append("📈 **STOCKS:**")
            for i, p in enumerate(self._stock_picks[:MAX_STOCK_PICKS], 1):
                emoji = "🔴" if p.direction == "DOWN" else "🟢"
                lines.append(
                    f"{i}. **{p.symbol}** {emoji} {p.direction} | "
                    f"${p.entry_price:,.2f} → ${p.target_price:,.2f} ({p.target_pct:+.1f}%) | "
                    f"{p.confidence*100:.0f}%"
                )
            lines.append("")
        
        lines.extend([
            "⏱️ **48-Hour Tracking Started**",
            "📊 Updates ONLY on significant changes (>3%)",
            "🎯 Instant alerts on target/stop hit",
            "",
            "_Ghost is watching. You'll hear from us._"
        ])
        
        message = "\n".join(lines)
        
        # Send the combined message
        try:
            success = self._send_telegram_func(message)
            if success:
                LOGGER.info(f"[TOP 10] ✅ Sent combined message with {len(self._crypto_picks)} crypto + {len(self._stock_picks)} stocks")
                
                # Register picks with Active Tracking System
                self._register_with_tracker()
                
                # Mark as sent today
                self._last_sent_date = datetime.utcnow().strftime("%Y-%m-%d")
            else:
                LOGGER.error("[TOP 10] ❌ Failed to send combined message")
        except Exception as e:
            LOGGER.error(f"[TOP 10] ❌ Error sending: {e}")
        
        # Clear queues
        self._crypto_picks = []
        self._stock_picks = []
        self._first_pick_time = None
        if self._timer:
            self._timer.cancel()
            self._timer = None
    
    def _register_with_tracker(self):
        """Register all picks with the Active Tracking System for 48h monitoring"""
        try:
            from core.active_tracking import get_active_tracker, ActivePick, TrackingStatus, TrackingOutcome
            
            tracker = get_active_tracker()
            batch_date = datetime.utcnow().strftime("%Y-%m-%d")
            now = datetime.utcnow()
            expires = now + timedelta(hours=48)
            
            all_picks = self._crypto_picks + self._stock_picks
            registered = 0
            
            for pick in all_picks:
                active_pick = ActivePick(
                    pick_id=0,  # Will be assigned by DB
                    symbol=pick.symbol,
                    asset_type=pick.asset_type,
                    direction=pick.direction,
                    entry_price=pick.entry_price,
                    target_price=pick.target_price,
                    stop_price=pick.stop_price,
                    confidence=pick.confidence,
                    created_at=now,
                    expires_at=expires,
                    batch_date=batch_date,
                    status=TrackingStatus.ACTIVE,
                    outcome=TrackingOutcome.PENDING,
                    current_price=pick.entry_price,
                    last_notified_price=pick.entry_price,
                    reasons=pick.reasons,
                )
                if tracker.add_pick(active_pick):
                    registered += 1
            
            LOGGER.info(f"[TOP 10] Registered {registered} picks with Active Tracker (48h monitoring)")
            
            # Mark TOP 10 as sent in tracker
            tracker.mark_top_10_sent(batch_date, len(self._stock_picks), len(self._crypto_picks))
            
        except Exception as e:
            LOGGER.error(f"[TOP 10] Failed to register with tracker: {e}")
    
    def force_send(self) -> bool:
        """Force send whatever picks are queued (for testing/manual trigger)"""
        with self._lock:
            total = len(self._crypto_picks) + len(self._stock_picks)
            if total == 0:
                LOGGER.info("[TOP 10] No picks to force send")
                return False
            
            LOGGER.info(f"[TOP 10] Force sending {total} queued picks...")
            self._send_combined_message()
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current aggregator status"""
        with self._lock:
            return {
                "crypto_queued": len(self._crypto_picks),
                "stock_queued": len(self._stock_picks),
                "total_queued": len(self._crypto_picks) + len(self._stock_picks),
                "last_sent_date": self._last_sent_date,
                "aggregation_window_s": AGGREGATION_WINDOW_SECONDS,
                "waiting_since": self._first_pick_time,
                "crypto_picks": [{"symbol": p.symbol, "confidence": p.confidence} for p in self._crypto_picks],
                "stock_picks": [{"symbol": p.symbol, "confidence": p.confidence} for p in self._stock_picks],
            }
    
    def reset(self):
        """Reset the aggregator (clear queues, allow new TOP 10)"""
        with self._lock:
            self._crypto_picks = []
            self._stock_picks = []
            self._first_pick_time = None
            self._last_sent_date = None
            if self._timer:
                self._timer.cancel()
                self._timer = None
            LOGGER.info("[TOP 10] Aggregator reset")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_aggregator: Optional[Top10Aggregator] = None
_aggregator_lock = threading.Lock()


def get_top10_aggregator() -> Top10Aggregator:
    """Get or create the singleton aggregator instance"""
    global _aggregator
    with _aggregator_lock:
        if _aggregator is None:
            _aggregator = Top10Aggregator()
        return _aggregator


# ============================================================================
# INTEGRATION FUNCTION - Hook into existing alert flow
# ============================================================================

def intercept_prediction_for_top10(
    symbol: str,
    prediction: Dict[str, Any],
    price_meta: Dict[str, Any],
    send_telegram_func: Callable[[str], bool]
) -> bool:
    """
    Intercept a prediction and add it to the TOP 10 queue.
    
    Call this INSTEAD of sending individual alerts.
    
    Args:
        symbol: The symbol (ETH, BTC, TSLA, etc.)
        prediction: The prediction dict with confidence, direction, etc.
        price_meta: Price metadata with current price
        send_telegram_func: Function to send Telegram messages
    
    Returns:
        True if added to queue (or sent), False if rejected
    """
    try:
        from core.asset_classifier import get_asset_type
        
        # Get aggregator
        aggregator = get_top10_aggregator()
        aggregator.set_telegram_func(send_telegram_func)
        
        # Classify asset - get_asset_type returns 'crypto', 'stock_large', 'stock_volatile', or 'stock_mid'
        try:
            asset_class = get_asset_type(symbol)
            asset_type = "crypto" if asset_class == "crypto" else "stock"
        except Exception:
            # Fallback classification
            asset_type = "crypto" if any(c in symbol.upper() for c in ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "USDT", "BNB"]) else "stock"
        
        # Get prediction details
        confidence = prediction.get("confidence", 0)
        direction = prediction.get("direction", "DOWN")
        
        # Get prices
        entry_price = price_meta.get("price", 0) or prediction.get("entry_price", 0) or prediction.get("price", 0)
        if entry_price <= 0:
            LOGGER.warning(f"[TOP 10] No entry price for {symbol}")
            return False
        
        # Get targets (or calculate defaults)
        target_price = prediction.get("target_price", 0)
        stop_price = prediction.get("stop_loss", 0) or prediction.get("stop_price", 0)
        
        if target_price <= 0:
            # Calculate default target
            if asset_type == "crypto":
                target_pct = 0.06  # 6% for crypto
            else:
                target_pct = 0.03  # 3% for stocks
            
            if direction == "DOWN":
                target_price = entry_price * (1 - target_pct)
            else:
                target_price = entry_price * (1 + target_pct)
        
        if stop_price <= 0:
            # Calculate default stop
            if asset_type == "crypto":
                stop_pct = 0.045  # 4.5% for crypto
            else:
                stop_pct = 0.025  # 2.5% for stocks
            
            if direction == "DOWN":
                stop_price = entry_price * (1 + stop_pct)
            else:
                stop_price = entry_price * (1 - stop_pct)
        
        # Add to queue
        return aggregator.add_prediction(
            symbol=symbol,
            asset_type=asset_type,
            direction=direction,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            confidence=confidence,
            prediction_id=prediction.get("prediction_id", prediction.get("id", 0)),
            reasons=str(prediction.get("signals", prediction.get("reasons", "")))
        )
        
    except Exception as e:
        LOGGER.error(f"[TOP 10] Error intercepting prediction for {symbol}: {e}")
        return False


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("🎯 Ghost TOP 10 Aggregator - Test Mode")
    print("=" * 50)
    
    aggregator = get_top10_aggregator()
    
    # Mock send function
    def mock_send(msg: str) -> bool:
        print("\n" + "=" * 50)
        print("📤 WOULD SEND TO TELEGRAM:")
        print("=" * 50)
        print(msg)
        print("=" * 50 + "\n")
        return True
    
    aggregator.set_telegram_func(mock_send)
    
    # Add test predictions
    test_picks = [
        ("ETH", "crypto", "DOWN", 2995.0, 0.95),
        ("BTC", "crypto", "DOWN", 88255.0, 0.92),
        ("SOL", "crypto", "DOWN", 125.0, 0.92),
        ("ADA", "crypto", "DOWN", 0.37, 0.95),
        ("AVAX", "crypto", "DOWN", 12.40, 0.95),
        ("TSLA", "stock", "DOWN", 402.50, 0.91),
        ("NVDA", "stock", "DOWN", 137.50, 0.89),
        ("AAPL", "stock", "DOWN", 248.50, 0.88),
        ("MSFT", "stock", "DOWN", 425.00, 0.87),
        ("GOOGL", "stock", "DOWN", 198.00, 0.86),
    ]
    
    for symbol, asset_type, direction, price, confidence in test_picks:
        target = price * 0.94 if asset_type == "crypto" else price * 0.97
        stop = price * 1.045 if asset_type == "crypto" else price * 1.025
        
        aggregator.add_prediction(
            symbol=symbol,
            asset_type=asset_type,
            direction=direction,
            entry_price=price,
            target_price=target,
            stop_price=stop,
            confidence=confidence
        )
    
    print("\nStatus:", aggregator.get_status())
