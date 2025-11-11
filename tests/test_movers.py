"""
Tests for Ghost Movers Scanner
"""
import os
import pytest
from app.core import movers_scanner


class TestTierLogic:
    """Test tier threshold logic"""
    
    def test_tier_20_plus(self):
        """Test 20%+ moves"""
        result = movers_scanner.tier(20.0)
        assert result["tier"] == "🔥20+"
        assert result["emoji"] == "🔥"
        
        result = movers_scanner.tier(-25.0)
        assert result["tier"] == "🔥20+"
        assert result["emoji"] == "🔥"
    
    def test_tier_15_plus(self):
        """Test 15%+ moves"""
        result = movers_scanner.tier(15.0)
        assert result["tier"] == "⚡15+"
        assert result["emoji"] == "⚡"
        
        result = movers_scanner.tier(-17.5)
        assert result["tier"] == "⚡15+"
        assert result["emoji"] == "⚡"
    
    def test_tier_10_plus(self):
        """Test 10%+ moves"""
        result = movers_scanner.tier(10.0)
        assert result["tier"] == "📈10+"
        assert result["emoji"] == "📈"
        
        result = movers_scanner.tier(-12.0)
        assert result["tier"] == "📈10+"
        assert result["emoji"] == "📈"
    
    def test_tier_6_plus(self):
        """Test 6%+ moves"""
        result = movers_scanner.tier(6.0)
        assert result["tier"] == "📊6+"
        assert result["emoji"] == "📊"
        
        result = movers_scanner.tier(-8.0)
        assert result["tier"] == "📊6+"
        assert result["emoji"] == "📊"
    
    def test_tier_below_6(self):
        """Test moves below 6%"""
        result = movers_scanner.tier(3.0)
        assert result["tier"] == "📉<6"
        assert result["emoji"] == "📉"


class TestPayloadSchema:
    """Test payload structure"""
    
    def test_build_payload_structure(self):
        """Test payload has required fields"""
        crypto_movers = [
            {
                "symbol": "BTC",
                "price": 89000.0,
                "pct_1h": 1.5,
                "pct_24h": 8.0,
                "vol_mult": 2.0,
                "age_s": 30,
                "provider": "coingecko",
                "tier": "📊6+",
                "emoji": "📊",
                "is_watch": False
            }
        ]
        stock_movers = [
            {
                "symbol": "AAPL",
                "price": 234.50,
                "pct_1h": 0.5,
                "pct_24h": 6.5,
                "vol_mult": 1.5,
                "age_s": 45,
                "provider": "polygon",
                "tier": "📊6+",
                "emoji": "📊",
                "is_watch": False
            }
        ]
        
        payload = movers_scanner.build_payload(crypto_movers, stock_movers)
        
        # Check required fields
        assert "crypto" in payload
        assert "stocks" in payload
        assert "ts" in payload
        assert "crypto_count" in payload
        assert "stocks_count" in payload
        
        # Check counts
        assert payload["crypto_count"] == 1
        assert payload["stocks_count"] == 1
        
        # Check arrays
        assert len(payload["crypto"]) == 1
        assert len(payload["stocks"]) == 1
        
        # Check mover structure
        crypto_mover = payload["crypto"][0]
        assert "symbol" in crypto_mover
        assert "price" in crypto_mover
        assert "pct_24h" in crypto_mover
        assert "vol_mult" in crypto_mover
        assert "provider" in crypto_mover
        assert "tier" in crypto_mover
    
    def test_build_payload_empty(self):
        """Test empty payload"""
        payload = movers_scanner.build_payload([], [])
        
        assert payload["crypto_count"] == 0
        assert payload["stocks_count"] == 0
        assert len(payload["crypto"]) == 0
        assert len(payload["stocks"]) == 0


class TestUniverseLoading:
    """Test universe loading logic"""
    
    def test_load_universe_includes_vip_coins(self):
        """Test that VIP coins are always included"""
        crypto_symbols, _ = movers_scanner.load_universe()
        
        # Check VIP coins present
        for vip_coin in movers_scanner.VIP_COINS:
            assert vip_coin in crypto_symbols
    
    def test_load_universe_includes_top_stocks(self):
        """Test that top stocks are included"""
        _, stock_symbols = movers_scanner.load_universe()
        
        # Check some major stocks
        assert len(stock_symbols) > 0
        assert "AAPL" in stock_symbols or "MSFT" in stock_symbols


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("POLYGON_API_KEY") or not os.getenv("CRYPTO_ENABLED") == "1",
    reason="Requires API keys and live data"
)
class TestLiveScanning:
    """Integration tests with live data"""
    
    @pytest.mark.asyncio
    async def test_scan_crypto_live(self):
        """Test crypto scanning with live data"""
        # Mock price fetcher (would use real in production)
        async def mock_fetch_price(symbol: str, is_crypto: bool = False):
            return {
                "price": 89000.0,
                "ts": int(os.time.time() * 1000),
                "provider": "coingecko",
                "age_s": 10
            }
        
        movers = await movers_scanner.scan_crypto(mock_fetch_price, None, None)
        
        # Should return a list (may be empty if no movers)
        assert isinstance(movers, list)
        
        # If movers found, check structure
        if movers:
            mover = movers[0]
            assert "symbol" in mover
            assert "price" in mover
            assert "tier" in mover
            assert mover["age_s"] <= 60  # Freshness check
    
    @pytest.mark.asyncio
    async def test_scan_stocks_live(self):
        """Test stock scanning with live data"""
        async def mock_fetch_price(symbol: str, is_crypto: bool = False):
            return {
                "price": 234.50,
                "ts": int(os.time.time() * 1000),
                "provider": "polygon",
                "age_s": 15
            }
        
        movers = await movers_scanner.scan_stocks(mock_fetch_price, None, None)
        
        assert isinstance(movers, list)
        
        if movers:
            mover = movers[0]
            assert "symbol" in mover
            assert "price" in mover
            assert mover["age_s"] <= 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
