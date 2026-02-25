"""
Ghost Protocol - Telegram Integration
=====================================
Clean, tested Telegram bot integration for V3 notifications.

This module handles:
- Sending messages to Telegram bot
- Rate limiting and cooldowns
- Daily alert caps with Redis persistence
- V3 metadata awareness

Usage:
    from notifications.telegram import TelegramNotifier
    
    notifier = TelegramNotifier()
    notifier.send("Hello!")
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Any, Protocol
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)


# ========================================
# PROTOCOLS (for dependency injection)
# ========================================

class RedisClient(Protocol):
    """Protocol for Redis client (allows testing with mocks)"""
    def get(self, key: str) -> Optional[bytes]: ...
    def incr(self, key: str) -> int: ...
    def expire(self, key: str, seconds: int) -> bool: ...
    def rpush(self, key: str, value: str) -> int: ...
    def lrange(self, key: str, start: int, end: int) -> List[bytes]: ...


# ========================================
# CONFIGURATION
# ========================================

@dataclass
class TelegramConfig:
    """Telegram configuration from environment"""
    bot_token: str
    chat_id: str
    
    # Rate limits
    symbol_cooldown_hours: int = 4
    max_alerts_per_hour: int = 5
    daily_alert_cap: int = 10
    min_alert_confidence: float = 0.70  # V3 requirement
    
    # Timezone
    timezone: str = "America/Chicago"
    
    @classmethod
    def from_env(cls) -> "TelegramConfig":
        """Load configuration from environment variables"""
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            symbol_cooldown_hours=int(os.getenv("SYMBOL_COOLDOWN_HOURS", "4")),
            max_alerts_per_hour=int(os.getenv("MAX_ALERTS_PER_HOUR", "5")),
            daily_alert_cap=int(os.getenv("DAILY_ALERT_CAP", "10")),
            min_alert_confidence=float(os.getenv("V3_MIN_CONFIDENCE", "0.78")),
            timezone=os.getenv("TZ", "America/Chicago"),
        )


# ========================================
# COOLDOWN TRACKER
# ========================================

class CooldownTracker:
    """
    Track cooldowns for symbols and hourly limits.
    Prevents spam by limiting alerts per symbol and per hour.
    """
    
    def __init__(self, cooldown_hours: int = 4, max_per_hour: int = 5):
        self.cooldown_hours = cooldown_hours
        self.max_per_hour = max_per_hour
        self._recent_alerts: Dict[str, float] = {}  # {symbol: timestamp}
        self._hourly_count: int = 0
        self._hourly_reset_time: float = time.time() + 3600
    
    def can_alert(self, symbol: str) -> tuple[bool, str]:
        """
        Check if we can send alert for this symbol.
        
        Returns:
            (allowed, reason) tuple
        """
        now = time.time()
        
        # Reset hourly counter if needed
        if now >= self._hourly_reset_time:
            self._hourly_count = 0
            self._hourly_reset_time = now + 3600
        
        # Check hourly limit
        if self._hourly_count >= self.max_per_hour:
            return False, f"hourly_limit ({self._hourly_count}/{self.max_per_hour})"
        
        # Check symbol cooldown
        if symbol in self._recent_alerts:
            last_time = self._recent_alerts[symbol]
            cooldown_end = last_time + (self.cooldown_hours * 3600)
            if now < cooldown_end:
                remaining_mins = int((cooldown_end - now) / 60)
                return False, f"cooldown ({remaining_mins} min left)"
        
        return True, "allowed"
    
    def mark_sent(self, symbol: str) -> None:
        """Mark symbol as alerted (track for cooldown)"""
        now = time.time()
        self._recent_alerts[symbol] = now
        self._hourly_count += 1
        
        # Cleanup old entries (older than cooldown period)
        cutoff = now - (self.cooldown_hours * 3600)
        self._recent_alerts = {
            s: t for s, t in self._recent_alerts.items() 
            if t > cutoff
        }
    
    def reset(self) -> None:
        """Reset all tracking (for testing)"""
        self._recent_alerts.clear()
        self._hourly_count = 0
        self._hourly_reset_time = time.time() + 3600


# ========================================
# DAILY CAP TRACKER
# ========================================

class DailyCapTracker:
    """
    Track daily alert caps with optional Redis persistence.
    Survives restarts when Redis is configured.
    """
    
    REDIS_COUNT_KEY = "ghost:alerts:daily_count"
    REDIS_LOG_KEY = "ghost:alerts:daily_log"
    
    def __init__(
        self, 
        daily_cap: int = 10, 
        timezone: str = "America/Chicago",
        redis_client: Optional[RedisClient] = None
    ):
        self.daily_cap = daily_cap
        self.timezone = timezone
        self.redis = redis_client
        
        # Memory fallback
        self._daily_count: Dict[str, int] = {}  # {date_str: count}
        self._daily_log: Dict[str, List[Dict]] = {}
    
    def _get_today(self) -> str:
        """Get today's date key in configured timezone"""
        try:
            tz = ZoneInfo(self.timezone)
            return datetime.now(tz).strftime("%Y-%m-%d")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")
    
    def get_count(self) -> int:
        """Get how many alerts sent today"""
        today = self._get_today()
        
        # Try Redis first
        if self.redis:
            try:
                key = f"{self.REDIS_COUNT_KEY}:{today}"
                count = self.redis.get(key)
                if count is not None:
                    return int(count)
            except Exception as e:
                logger.warning(f"Redis read failed: {e}")
        
        # Fallback to memory
        return self._daily_count.get(today, 0)
    
    def can_send(self, symbol: str, confidence: float) -> tuple[bool, str]:
        """
        Check if this alert should be sent based on daily cap.
        
        Implements tiered confidence requirements:
        - Last 3 slots: require 85%+ confidence
        - Last slot: require 90%+ confidence
        
        Returns:
            (allowed, reason) tuple
        """
        current = self.get_count()
        remaining = self.daily_cap - current
        
        if remaining <= 0:
            return False, f"daily_cap ({current}/{self.daily_cap})"
        
        # V3 minimum confidence
        if confidence < 0.70:
            return False, f"below_v3_min ({confidence:.0%} < 70%)"
        
        # Last 3 slots need 85%+
        if remaining <= 3 and confidence < 0.85:
            return False, f"reserved_high_conf (need 85%, got {confidence:.0%})"
        
        # Last slot needs 90%+
        if remaining == 1 and confidence < 0.90:
            return False, f"last_slot_reserved (need 90%, got {confidence:.0%})"
        
        return True, f"allowed ({current + 1}/{self.daily_cap})"
    
    def increment(self, symbol: str, confidence: float) -> bool:
        """
        Increment daily count after sending alert.
        
        Returns:
            True if successfully incremented
        """
        today = self._get_today()
        
        # Try Redis first
        if self.redis:
            try:
                import json
                
                # Atomic increment
                key = f"{self.REDIS_COUNT_KEY}:{today}"
                count = self.redis.incr(key)
                self.redis.expire(key, 172800)  # 48h expiry
                
                # Log entry
                log_key = f"{self.REDIS_LOG_KEY}:{today}"
                entry = json.dumps({
                    "symbol": symbol,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "count": count
                })
                self.redis.rpush(log_key, entry)
                self.redis.expire(log_key, 172800)
                
                logger.info(f"📊 Daily alert: {count}/{self.daily_cap} (Redis)")
                return True
                
            except Exception as e:
                logger.warning(f"Redis increment failed: {e}")
        
        # Memory fallback
        current = self._daily_count.get(today, 0)
        self._daily_count[today] = current + 1
        
        if today not in self._daily_log:
            self._daily_log[today] = []
        self._daily_log[today].append({
            "symbol": symbol,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "count": current + 1
        })
        
        logger.info(f"📊 Daily alert: {current + 1}/{self.daily_cap} (memory)")
        return True
    
    def reset(self) -> None:
        """Reset counters (for testing)"""
        self._daily_count.clear()
        self._daily_log.clear()


# ========================================
# TELEGRAM NOTIFIER
# ========================================

class TelegramNotifier:
    """
    Main Telegram notification service.
    
    Features:
    - HTTP API integration
    - Rate limiting (cooldowns + daily cap)
    - V3 confidence enforcement
    - Redis persistence (optional)
    
    Example:
        notifier = TelegramNotifier()
        
        # Simple message
        notifier.send("Hello!")
        
        # V3 alert with validation
        success, reason = notifier.send_alert(
            symbol="ETH",
            message="🚀 ETH Alert",
            confidence=0.75
        )
    """
    
    def __init__(
        self,
        config: Optional[TelegramConfig] = None,
        redis_client: Optional[RedisClient] = None
    ):
        self.config = config or TelegramConfig.from_env()
        
        self.cooldown = CooldownTracker(
            cooldown_hours=self.config.symbol_cooldown_hours,
            max_per_hour=self.config.max_alerts_per_hour
        )
        
        self.daily_cap = DailyCapTracker(
            daily_cap=self.config.daily_alert_cap,
            timezone=self.config.timezone,
            redis_client=redis_client
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.config.bot_token and self.config.chat_id)
    
    def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send message to Telegram (no rate limiting).
        
        Args:
            message: Text to send
            parse_mode: "Markdown" or "HTML"
            
        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            logger.warning("❌ Telegram not configured (missing token/chat_id)")
            return False
        
        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("📲 Telegram message sent")
                return True
            else:
                logger.error(f"❌ Telegram failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Telegram exception: {e}")
            return False
    
    def send_alert(
        self, 
        symbol: str, 
        message: str, 
        confidence: float,
        skip_rate_limit: bool = False,
        parse_mode: str = "Markdown"
    ) -> tuple[bool, str]:
        """
        Send alert with V3 validation and rate limiting.
        
        Args:
            symbol: Asset symbol (ETH, XRP, etc.)
            message: Alert message
            confidence: Confidence score (0.0-1.0)
            skip_rate_limit: Bypass rate limits (for testing)
            parse_mode: "Markdown" or "HTML"
            
        Returns:
            (success, reason) tuple
        """
        if not self.is_configured:
            return False, "not_configured"
        
        if not skip_rate_limit:
            # Check cooldown
            can_cool, reason = self.cooldown.can_alert(symbol)
            if not can_cool:
                return False, reason
            
            # Check daily cap
            can_cap, reason = self.daily_cap.can_send(symbol, confidence)
            if not can_cap:
                return False, reason
        
        # Send message
        success = self.send(message, parse_mode)
        
        if success and not skip_rate_limit:
            self.cooldown.mark_sent(symbol)
            self.daily_cap.increment(symbol, confidence)
        
        return success, "sent" if success else "send_failed"
    
    def send_batch(
        self,
        messages: List[str],
        parse_mode: str = "Markdown"
    ) -> tuple[int, int]:
        """
        Send multiple messages (for TOP 10 split messages).
        
        Returns:
            (sent_count, failed_count)
        """
        sent = 0
        failed = 0
        
        for msg in messages:
            if self.send(msg, parse_mode):
                sent += 1
            else:
                failed += 1
        
        return sent, failed


# ========================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ========================================

# Singleton instance (lazy loaded)
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the singleton notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def send_telegram_message(message: str, parse_mode: str = "Markdown") -> bool:
    """
    Convenience function for sending messages.
    Compatible with existing code that uses this function.
    """
    return get_notifier().send(message, parse_mode)
