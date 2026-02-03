"""
Tests for notifications.telegram module
======================================
Tests for Telegram integration with rate limiting.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from notifications.telegram import (
    TelegramConfig,
    CooldownTracker,
    DailyCapTracker,
    TelegramNotifier,
    send_telegram_message,
    get_notifier,
)


# ========================================
# TelegramConfig Tests
# ========================================

class TestTelegramConfig:
    """Tests for TelegramConfig"""
    
    def test_from_env_defaults(self, monkeypatch):
        """Test loading from environment with defaults"""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        
        config = TelegramConfig.from_env()
        
        assert config.bot_token == ""
        assert config.chat_id == ""
        assert config.symbol_cooldown_hours == 4
        assert config.max_alerts_per_hour == 5
        assert config.daily_alert_cap == 10
        assert config.min_alert_confidence == 0.70
    
    def test_from_env_with_values(self, monkeypatch):
        """Test loading from environment with custom values"""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
        monkeypatch.setenv("SYMBOL_COOLDOWN_HOURS", "2")
        monkeypatch.setenv("MAX_ALERTS_PER_HOUR", "10")
        monkeypatch.setenv("DAILY_ALERT_CAP", "20")
        monkeypatch.setenv("V3_MIN_CONFIDENCE", "0.80")
        
        config = TelegramConfig.from_env()
        
        assert config.bot_token == "test-token"
        assert config.chat_id == "123456"
        assert config.symbol_cooldown_hours == 2
        assert config.max_alerts_per_hour == 10
        assert config.daily_alert_cap == 20
        assert config.min_alert_confidence == 0.80


# ========================================
# CooldownTracker Tests
# ========================================

class TestCooldownTracker:
    """Tests for CooldownTracker"""
    
    def test_can_alert_first_time(self):
        """First alert for symbol should be allowed"""
        tracker = CooldownTracker()
        
        allowed, reason = tracker.can_alert("ETH")
        
        assert allowed is True
        assert reason == "allowed"
    
    def test_cooldown_after_alert(self):
        """Symbol should be on cooldown after alert"""
        tracker = CooldownTracker(cooldown_hours=4)
        tracker.mark_sent("ETH")
        
        allowed, reason = tracker.can_alert("ETH")
        
        assert allowed is False
        assert "cooldown" in reason
    
    def test_different_symbol_allowed(self):
        """Different symbol should not be affected by cooldown"""
        tracker = CooldownTracker()
        tracker.mark_sent("ETH")
        
        allowed, reason = tracker.can_alert("XRP")
        
        assert allowed is True
    
    def test_hourly_limit(self):
        """Should hit hourly limit after max alerts"""
        tracker = CooldownTracker(max_per_hour=3)
        
        for i in range(3):
            tracker.mark_sent(f"SYM{i}")
        
        allowed, reason = tracker.can_alert("NEW")
        
        assert allowed is False
        assert "hourly_limit" in reason
    
    def test_reset_clears_tracking(self):
        """Reset should clear all tracking"""
        tracker = CooldownTracker()
        tracker.mark_sent("ETH")
        tracker.reset()
        
        allowed, reason = tracker.can_alert("ETH")
        
        assert allowed is True


# ========================================
# DailyCapTracker Tests
# ========================================

class TestDailyCapTracker:
    """Tests for DailyCapTracker"""
    
    def test_initial_count_zero(self):
        """Initial count should be zero"""
        tracker = DailyCapTracker()
        
        assert tracker.get_count() == 0
    
    def test_increment_increases_count(self):
        """Increment should increase count"""
        tracker = DailyCapTracker()
        tracker.increment("ETH", 0.75)
        
        assert tracker.get_count() == 1
    
    def test_can_send_below_cap(self):
        """Should allow sending when below cap"""
        tracker = DailyCapTracker(daily_cap=10)
        
        allowed, reason = tracker.can_send("ETH", 0.75)
        
        assert allowed is True
        assert "allowed" in reason
    
    def test_can_send_at_cap(self):
        """Should deny sending when at cap"""
        tracker = DailyCapTracker(daily_cap=3)
        for i in range(3):
            tracker.increment(f"SYM{i}", 0.90)
        
        allowed, reason = tracker.can_send("NEW", 0.90)
        
        assert allowed is False
        assert "daily_cap" in reason
    
    def test_rejects_low_confidence(self):
        """Should reject below V3 minimum confidence"""
        tracker = DailyCapTracker()
        
        allowed, reason = tracker.can_send("ETH", 0.65)
        
        assert allowed is False
        assert "below_v3_min" in reason
    
    def test_last_slots_need_high_confidence(self):
        """Last 3 slots should require 85%+ confidence"""
        tracker = DailyCapTracker(daily_cap=10)
        # Fill 7 slots
        for i in range(7):
            tracker.increment(f"SYM{i}", 0.90)
        
        # 3 slots left - 80% should fail
        allowed, reason = tracker.can_send("NEW", 0.80)
        
        assert allowed is False
        assert "reserved_high_conf" in reason
    
    def test_last_slot_needs_90_percent(self):
        """Last slot should require 90%+ confidence"""
        tracker = DailyCapTracker(daily_cap=10)
        # Fill 9 slots
        for i in range(9):
            tracker.increment(f"SYM{i}", 0.95)
        
        # Last slot - 85% should fail
        allowed, reason = tracker.can_send("NEW", 0.85)
        
        assert allowed is False
        assert "last_slot_reserved" in reason
    
    def test_last_slot_allows_90_percent(self):
        """Last slot should allow 90%+ confidence"""
        tracker = DailyCapTracker(daily_cap=10)
        for i in range(9):
            tracker.increment(f"SYM{i}", 0.95)
        
        allowed, reason = tracker.can_send("NEW", 0.92)
        
        assert allowed is True
    
    def test_reset_clears_counts(self):
        """Reset should clear all counts"""
        tracker = DailyCapTracker()
        tracker.increment("ETH", 0.80)
        tracker.reset()
        
        assert tracker.get_count() == 0


# ========================================
# DailyCapTracker Redis Tests
# ========================================

class TestDailyCapTrackerRedis:
    """Tests for DailyCapTracker with Redis"""
    
    def test_get_count_from_redis(self):
        """Should read count from Redis when available"""
        mock_redis = Mock()
        mock_redis.get.return_value = b"5"
        
        tracker = DailyCapTracker(redis_client=mock_redis)
        count = tracker.get_count()
        
        assert count == 5
        mock_redis.get.assert_called_once()
    
    def test_increment_uses_redis(self):
        """Should increment via Redis when available"""
        mock_redis = Mock()
        mock_redis.incr.return_value = 3
        
        tracker = DailyCapTracker(redis_client=mock_redis)
        tracker.increment("ETH", 0.80)
        
        mock_redis.incr.assert_called_once()
        mock_redis.expire.assert_called()
        mock_redis.rpush.assert_called_once()
    
    def test_fallback_on_redis_error(self):
        """Should fallback to memory on Redis error"""
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis error")
        
        tracker = DailyCapTracker(redis_client=mock_redis)
        count = tracker.get_count()
        
        assert count == 0  # Memory fallback


# ========================================
# TelegramNotifier Tests
# ========================================

class TestTelegramNotifier:
    """Tests for TelegramNotifier"""
    
    def test_not_configured_without_token(self):
        """Should report not configured without token"""
        config = TelegramConfig(bot_token="", chat_id="123")
        notifier = TelegramNotifier(config=config)
        
        assert notifier.is_configured is False
    
    def test_not_configured_without_chat_id(self):
        """Should report not configured without chat_id"""
        config = TelegramConfig(bot_token="token", chat_id="")
        notifier = TelegramNotifier(config=config)
        
        assert notifier.is_configured is False
    
    def test_is_configured_with_both(self):
        """Should be configured with token and chat_id"""
        config = TelegramConfig(bot_token="token", chat_id="123")
        notifier = TelegramNotifier(config=config)
        
        assert notifier.is_configured is True
    
    @patch('notifications.telegram.requests.post')
    def test_send_success(self, mock_post):
        """Should return True on successful send"""
        mock_post.return_value.status_code = 200
        
        config = TelegramConfig(bot_token="token", chat_id="123")
        notifier = TelegramNotifier(config=config)
        
        result = notifier.send("Test message")
        
        assert result is True
        mock_post.assert_called_once()
    
    @patch('notifications.telegram.requests.post')
    def test_send_failure(self, mock_post):
        """Should return False on failed send"""
        mock_post.return_value.status_code = 400
        
        config = TelegramConfig(bot_token="token", chat_id="123")
        notifier = TelegramNotifier(config=config)
        
        result = notifier.send("Test message")
        
        assert result is False
    
    @patch('notifications.telegram.requests.post')
    def test_send_exception(self, mock_post):
        """Should return False on exception"""
        mock_post.side_effect = Exception("Network error")
        
        config = TelegramConfig(bot_token="token", chat_id="123")
        notifier = TelegramNotifier(config=config)
        
        result = notifier.send("Test message")
        
        assert result is False
    
    def test_send_alert_not_configured(self):
        """Should fail alert when not configured"""
        config = TelegramConfig(bot_token="", chat_id="")
        notifier = TelegramNotifier(config=config)
        
        success, reason = notifier.send_alert("ETH", "Test", 0.80)
        
        assert success is False
        assert reason == "not_configured"
    
    @patch('notifications.telegram.requests.post')
    def test_send_alert_with_rate_limiting(self, mock_post):
        """Should apply rate limiting to alerts"""
        mock_post.return_value.status_code = 200
        
        config = TelegramConfig(
            bot_token="token", chat_id="123",
            symbol_cooldown_hours=1
        )
        notifier = TelegramNotifier(config=config)
        
        # First alert should succeed
        success1, reason1 = notifier.send_alert("ETH", "Test", 0.80)
        assert success1 is True
        
        # Second alert for same symbol should fail (cooldown)
        success2, reason2 = notifier.send_alert("ETH", "Test", 0.80)
        assert success2 is False
        assert "cooldown" in reason2
    
    @patch('notifications.telegram.requests.post')
    def test_send_alert_skip_rate_limit(self, mock_post):
        """Should skip rate limiting when requested"""
        mock_post.return_value.status_code = 200
        
        config = TelegramConfig(bot_token="token", chat_id="123")
        notifier = TelegramNotifier(config=config)
        
        # First alert
        notifier.send_alert("ETH", "Test", 0.80)
        
        # Second alert with skip_rate_limit
        success, reason = notifier.send_alert(
            "ETH", "Test", 0.80, 
            skip_rate_limit=True
        )
        
        assert success is True
    
    @patch('notifications.telegram.requests.post')
    def test_send_batch(self, mock_post):
        """Should send multiple messages in batch"""
        mock_post.return_value.status_code = 200
        
        config = TelegramConfig(bot_token="token", chat_id="123")
        notifier = TelegramNotifier(config=config)
        
        messages = ["Message 1", "Message 2", "Message 3"]
        sent, failed = notifier.send_batch(messages)
        
        assert sent == 3
        assert failed == 0
        assert mock_post.call_count == 3


# ========================================
# Module-level Function Tests
# ========================================

class TestModuleFunctions:
    """Tests for module-level convenience functions"""
    
    @patch('notifications.telegram.requests.post')
    def test_send_telegram_message(self, mock_post, monkeypatch):
        """Test module-level send function"""
        mock_post.return_value.status_code = 200
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
        
        # Reset singleton
        import notifications.telegram as tg
        tg._notifier = None
        
        result = send_telegram_message("Test")
        
        assert result is True
    
    def test_get_notifier_singleton(self, monkeypatch):
        """Test that get_notifier returns singleton"""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
        
        # Reset singleton
        import notifications.telegram as tg
        tg._notifier = None
        
        notifier1 = get_notifier()
        notifier2 = get_notifier()
        
        assert notifier1 is notifier2
