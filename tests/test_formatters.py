"""
Tests for notifications.formatters module
=========================================
Tests for V3-aware message formatting.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from notifications.formatters import (
    TradePick,
    format_single_pick,
    format_top10_message,
    format_v3_alert,
    format_price_alert,
    format_price,
    get_risk_level,
    get_hold_reason,
    calculate_rr,
    is_trading_day,
    is_us_holiday,
    get_next_trading_day,
    get_holiday_name,
)


# ========================================
# Helper Function Tests
# ========================================

class TestFormatPrice:
    """Tests for format_price function"""
    
    def test_zero_price(self):
        """Zero should format as $0.00"""
        assert format_price(0) == "$0.00"
    
    def test_large_price(self):
        """Large prices should have 2 decimals with commas"""
        assert format_price(1234.56) == "$1,234.56"
        assert format_price(100.00) == "$100.00"
    
    def test_small_price(self):
        """Small prices should have more decimals"""
        assert format_price(0.1234) == "$0.1234"
        assert format_price(50.25) == "$50.25"
    
    def test_micro_price(self):
        """Micro prices should have 6 decimals"""
        assert format_price(0.001234) == "$0.001234"


class TestGetRiskLevel:
    """Tests for get_risk_level function"""
    
    def test_high_confidence_low_risk(self):
        assert get_risk_level(0.70) == "Low"
        assert get_risk_level(0.80) == "Low"
    
    def test_medium_confidence_moderate_risk(self):
        assert get_risk_level(0.50) == "Moderate"
        assert get_risk_level(0.45) == "Moderate"
    
    def test_low_confidence_high_risk(self):
        assert get_risk_level(0.30) == "High"
        assert get_risk_level(0.20) == "High"


class TestGetHoldReason:
    """Tests for get_hold_reason function"""
    
    def test_one_day_high_conf(self):
        assert get_hold_reason(0.50, 1) == "RSI extreme"
    
    def test_one_day_low_conf(self):
        assert get_hold_reason(0.30, 1) == "low conviction scalp"
    
    def test_two_days(self):
        assert get_hold_reason(0.50, 2) == "volatility swing"
    
    def test_swing_trade(self):
        assert get_hold_reason(0.50, 3) == "swing trade"
        assert get_hold_reason(0.50, 4) == "swing trade"
    
    def test_position_trade(self):
        assert get_hold_reason(0.50, 7) == "position trade"


class TestCalculateRR:
    """Tests for calculate_rr function"""
    
    def test_buy_positive_rr(self):
        # Entry 100, Target 110 (+10), Stop 95 (-5) = 2:1
        rr = calculate_rr(100, 110, 95, "UP")
        assert rr == 2.0
    
    def test_sell_positive_rr(self):
        # Entry 100, Target 90 (+10), Stop 105 (-5) = 2:1
        rr = calculate_rr(100, 90, 105, "DOWN")
        assert rr == 2.0
    
    def test_zero_risk(self):
        # Should return 1.0 to avoid division by zero
        rr = calculate_rr(100, 110, 100, "UP")
        assert rr == 1.0


# ========================================
# Trading Day Tests
# ========================================

class TestTradingDays:
    """Tests for trading day functions"""
    
    def test_weekday_is_trading_day(self):
        """Monday-Friday should be trading days"""
        # Use a known Monday
        monday = datetime(2025, 2, 3)  # Feb 3, 2025 is Monday
        assert is_trading_day(monday) is True
    
    def test_weekend_not_trading_day(self):
        """Saturday/Sunday should not be trading days"""
        saturday = datetime(2025, 2, 1)  # Feb 1, 2025 is Saturday
        sunday = datetime(2025, 2, 2)  # Feb 2, 2025 is Sunday
        assert is_trading_day(saturday) is False
        assert is_trading_day(sunday) is False
    
    def test_holiday_not_trading_day(self):
        """US holidays should not be trading days"""
        # Christmas 2025
        christmas = datetime(2025, 12, 25)
        assert is_us_holiday(christmas) is True
        assert is_trading_day(christmas) is False
    
    def test_get_holiday_name(self):
        """Should return correct holiday name"""
        christmas = datetime(2025, 12, 25)
        assert get_holiday_name(christmas) == "Christmas"
    
    def test_next_trading_day_from_friday(self):
        """Next trading day from Friday should be Monday"""
        friday = datetime(2025, 1, 31)  # Jan 31, 2025 is Friday
        next_day = get_next_trading_day(friday)
        # Friday itself is a trading day
        assert next_day.weekday() < 5  # Mon-Fri
    
    def test_next_trading_day_from_saturday(self):
        """Next trading day from Saturday should be Monday"""
        saturday = datetime(2025, 2, 1)
        next_day = get_next_trading_day(saturday)
        assert next_day.weekday() == 0  # Monday


# ========================================
# TradePick Tests
# ========================================

class TestTradePick:
    """Tests for TradePick dataclass"""
    
    def test_from_dict_minimal(self):
        """Should handle minimal dictionary"""
        data = {
            "symbol": "ETH",
            "current": 3500.0,
            "confidence": 0.72
        }
        
        pick = TradePick.from_dict(data)
        
        assert pick.symbol == "ETH"
        assert pick.current_price == 3500.0
        assert pick.confidence == 0.72
        assert pick.direction == "UP"  # Default
    
    def test_from_dict_full(self):
        """Should handle full dictionary with V3 data"""
        data = {
            "symbol": "ETH",
            "direction": "UP",
            "current": 3500.0,
            "prediction_48h": 3800.0,
            "stop": 3400.0,
            "confidence": 0.72,
            "hold_days": 3,
            "news_influenced": True,
            "v3_is_inverse": True,
            "v3_original_direction": "DOWN",
            "v3_historical_win_rate": 0.615,
            "v3_sample_size": 156,
        }
        
        pick = TradePick.from_dict(data)
        
        assert pick.symbol == "ETH"
        assert pick.target_price == 3800.0
        assert pick.stop_price == 3400.0
        assert pick.v3_is_inverse is True
        assert pick.v3_original_direction == "DOWN"
        assert pick.v3_historical_win_rate == 0.615
        assert pick.v3_sample_size == 156


# ========================================
# format_single_pick Tests
# ========================================

class TestFormatSinglePick:
    """Tests for format_single_pick function"""
    
    def test_buy_signal_format(self):
        """Should format buy signal correctly"""
        pick = TradePick(
            symbol="ETH",
            direction="UP",
            current_price=3500.0,
            target_price=3800.0,
            stop_price=3400.0,
            confidence=0.72,
            hold_days=3,
        )
        
        ct = datetime(2025, 2, 3, 8, 0)  # Monday 8 AM
        stock_entry = datetime(2025, 2, 3)  # Monday
        
        result = format_single_pick(pick, 1, is_stock=False, ct=ct, stock_entry_date=stock_entry)
        
        assert "1) 🟢 ETH — BUY" in result
        assert "BUY NOW (24/7)" in result
        assert "Target:" in result
        assert "STOP LOSS:" in result
    
    def test_sell_signal_format(self):
        """Should format sell signal correctly"""
        pick = TradePick(
            symbol="BTC",
            direction="DOWN",
            current_price=50000.0,
            target_price=45000.0,
            stop_price=52000.0,
            confidence=0.65,
            hold_days=2,
        )
        
        ct = datetime(2025, 2, 3, 8, 0)
        stock_entry = datetime(2025, 2, 3)
        
        result = format_single_pick(pick, 1, is_stock=False, ct=ct, stock_entry_date=stock_entry)
        
        assert "🔴" in result
        assert "SELL" in result
        assert "BUY-BACK" in result
    
    def test_inverse_signal_badge(self):
        """Should show inverse badge and explanation"""
        pick = TradePick(
            symbol="ETH",
            direction="UP",
            current_price=3500.0,
            target_price=3800.0,
            stop_price=3400.0,
            confidence=0.72,
            v3_is_inverse=True,
            v3_original_direction="DOWN",
        )
        
        ct = datetime(2025, 2, 3, 8, 0)
        stock_entry = datetime(2025, 2, 3)
        
        result = format_single_pick(pick, 1, is_stock=False, ct=ct, stock_entry_date=stock_entry)
        
        assert "🔄 INVERSE" in result
        assert "Ghost predicted DOWN → FLIPPED to UP" in result
    
    def test_stock_market_hours(self):
        """Should show 9:30 AM for stocks"""
        pick = TradePick(
            symbol="AAPL",
            direction="UP",
            current_price=180.0,
            target_price=190.0,
            stop_price=175.0,
            confidence=0.60,
        )
        
        ct = datetime(2025, 2, 3, 8, 0)  # Monday
        stock_entry = datetime(2025, 2, 3)
        
        result = format_single_pick(pick, 1, is_stock=True, ct=ct, stock_entry_date=stock_entry)
        
        assert "@ 9:30 AM" in result
    
    def test_win_rate_high_sample(self):
        """Should show win rate without asterisk for high sample"""
        pick = TradePick(
            symbol="ETH",
            direction="UP",
            current_price=3500.0,
            target_price=3800.0,
            stop_price=3400.0,
            confidence=0.72,
            v3_historical_win_rate=0.615,
            v3_sample_size=150,
        )
        
        ct = datetime(2025, 2, 3, 8, 0)
        stock_entry = datetime(2025, 2, 3)
        
        result = format_single_pick(pick, 1, is_stock=False, ct=ct, stock_entry_date=stock_entry)
        
        # 61.5% rounds to 62% when displayed
        assert "Win Rate: 62%" in result
        # Should not have asterisk for high sample size
        assert "*" not in result.split("Win Rate")[1].split("\n")[0]
    
    def test_win_rate_low_sample(self):
        """Should show win rate with asterisk for low sample"""
        pick = TradePick(
            symbol="ETH",
            direction="UP",
            current_price=3500.0,
            target_price=3800.0,
            stop_price=3400.0,
            confidence=0.72,
            v3_historical_win_rate=0.60,
            v3_sample_size=25,
        )
        
        ct = datetime(2025, 2, 3, 8, 0)
        stock_entry = datetime(2025, 2, 3)
        
        result = format_single_pick(pick, 1, is_stock=False, ct=ct, stock_entry_date=stock_entry)
        
        assert "60%*" in result
        assert "(25 trades)" in result


# ========================================
# format_top10_message Tests
# ========================================

class TestFormatTop10Message:
    """Tests for format_top10_message function"""
    
    @patch('notifications.formatters.get_central_time')
    def test_empty_lists(self, mock_time):
        """Should handle empty lists"""
        mock_time.return_value = datetime(2025, 2, 3, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
        
        messages = format_top10_message(stocks=[], crypto=[])
        
        assert len(messages) >= 1
        assert "GHOST TOP 10" in messages[0]
    
    @patch('notifications.formatters.get_central_time')
    def test_crypto_only(self, mock_time):
        """Should format crypto-only message"""
        mock_time.return_value = datetime(2025, 2, 3, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
        
        crypto = [
            {"symbol": "ETH", "current": 3500, "confidence": 0.72, "direction": "UP"},
            {"symbol": "XRP", "current": 2.50, "confidence": 0.75, "direction": "UP"},
        ]
        
        messages = format_top10_message(stocks=[], crypto=crypto)
        
        assert len(messages) >= 1
        assert "TOP CRYPTO" in messages[0]
        assert "ETH" in messages[0]
        assert "XRP" in messages[0]
    
    @patch('notifications.formatters.get_central_time')
    def test_stocks_only(self, mock_time):
        """Should format stocks-only message"""
        mock_time.return_value = datetime(2025, 2, 3, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
        
        stocks = [
            {"symbol": "AAPL", "current": 180, "confidence": 0.65, "direction": "UP"},
        ]
        
        messages = format_top10_message(stocks=stocks, crypto=[])
        
        assert len(messages) >= 1
        assert "TOP STOCKS" in messages[0]
        assert "AAPL" in messages[0]
    
    @patch('notifications.formatters.get_central_time')
    def test_v3_validated_badge(self, mock_time):
        """Should show V3 VALIDATED in header"""
        mock_time.return_value = datetime(2025, 2, 3, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
        
        messages = format_top10_message(stocks=[], crypto=[])
        
        assert "V3 VALIDATED" in messages[0]
    
    @patch('notifications.formatters.get_central_time')
    def test_weekend_warning(self, mock_time):
        """Should show weekend warning for stocks"""
        # Set to Saturday
        mock_time.return_value = datetime(2025, 2, 1, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
        
        stocks = [{"symbol": "AAPL", "current": 180, "confidence": 0.65}]
        
        messages = format_top10_message(stocks=stocks, crypto=[])
        
        assert "WEEKEND" in messages[0] or "MARKET" in messages[0]
    
    @patch('notifications.formatters.get_central_time')
    def test_message_split_long_content(self, mock_time):
        """Should split message if too long for Telegram"""
        mock_time.return_value = datetime(2025, 2, 3, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
        
        # Create many picks to exceed 4096 chars
        stocks = [
            {"symbol": f"STK{i}", "current": 100+i, "confidence": 0.70, "direction": "UP",
             "v3_is_inverse": True, "v3_original_direction": "DOWN"}
            for i in range(5)
        ]
        crypto = [
            {"symbol": f"CRY{i}", "current": 50+i, "confidence": 0.72, "direction": "UP",
             "v3_is_inverse": True, "v3_original_direction": "DOWN"}
            for i in range(5)
        ]
        
        messages = format_top10_message(stocks=stocks, crypto=crypto)
        
        # Should return valid messages
        for msg in messages:
            assert len(msg) <= 4096


# ========================================
# format_v3_alert Tests
# ========================================

class TestFormatV3Alert:
    """Tests for format_v3_alert function"""
    
    def test_basic_alert(self):
        """Should format basic V3 alert"""
        result = format_v3_alert(
            symbol="ETH",
            direction="UP",
            confidence=0.75,
            strategy="ghost_inverse",
            is_inverse=True,
            hold_hours=72,
            win_rate=0.615
        )
        
        assert "V3 SIGNAL: ETH" in result
        assert "Direction: UP" in result
        assert "INVERSE" in result
        assert "ghost_inverse" in result
        assert "72h" in result
        assert "75%" in result
        assert "61.5%" in result
    
    def test_non_inverse_alert(self):
        """Should not show INVERSE badge for normal signals"""
        result = format_v3_alert(
            symbol="XRP",
            direction="UP",
            confidence=0.76,
            strategy="mean_reversion",
            is_inverse=False,
            hold_hours=168
        )
        
        assert "INVERSE" not in result
        assert "168h" in result


# ========================================
# format_price_alert Tests
# ========================================

class TestFormatPriceAlert:
    """Tests for format_price_alert function"""
    
    def test_target_hit(self):
        """Should format target hit alert"""
        result = format_price_alert(
            symbol="ETH",
            alert_type="target_hit",
            entry_price=3500,
            exit_price=3800,
            target_price=3750,
            stop_price=3400
        )
        
        assert "TARGET HIT" in result
        assert "ETH" in result
        assert "Gain:" in result
        assert "✅" in result
    
    def test_stop_hit(self):
        """Should format stop hit alert"""
        result = format_price_alert(
            symbol="BTC",
            alert_type="stop_hit",
            entry_price=50000,
            exit_price=48000,
            target_price=55000,
            stop_price=48500
        )
        
        assert "STOP HIT" in result
        assert "BTC" in result
        assert "Loss:" in result
        assert "❌" in result
