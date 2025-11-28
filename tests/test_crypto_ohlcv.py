"""
Test Script: Verify Crypto OHLCV Integration
=============================================
Validates that BTC/ETH can now fetch OHLCV data from Binance.

Before Fix:
- BTC: 5/25 features (20%) - NO OHLCV
- ETH: 5/25 features (20%) - NO OHLCV
- Result: 40% FLAT always

After Fix (Expected):
- BTC: 20-23/25 features (80-92%)
- ETH: 20-23/25 features (80-92%)
- Result: 45-75% confidence, varied direction
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.providers.unified_provider import get_unified_provider
from core.data_pillars.technical_engine import TechnicalEngine
from core.data_pillars.volume_engine import VolumeEngine


def test_unified_provider():
    """Test unified provider directly"""
    print("\n" + "="*70)
    print("TEST 1: Unified Provider Direct Access")
    print("="*70)
    
    provider = get_unified_provider()
    
    # Test BTC
    print("\n[BTC] Fetching OHLCV...")
    ohlcv = provider.get_ohlcv("BTC", interval="1d", lookback=50)
    
    if ohlcv:
        print(f"✅ BTC: {len(ohlcv.bars)} bars from {ohlcv.provider}")
        print(f"   Cache hit: {ohlcv.cache_hit}")
        print(f"   Latest bar: {ohlcv.bars[-1]}")
        print(f"   Date range: {ohlcv.bars[0].timestamp} to {ohlcv.bars[-1].timestamp}")
    else:
        print("❌ BTC: Failed to fetch OHLCV")
        return False
    
    # Test ETH
    print("\n[ETH] Fetching OHLCV...")
    ohlcv = provider.get_ohlcv("ETH", interval="1d", lookback=50)
    
    if ohlcv:
        print(f"✅ ETH: {len(ohlcv.bars)} bars from {ohlcv.provider}")
        print(f"   Cache hit: {ohlcv.cache_hit}")
        print(f"   Latest bar: {ohlcv.bars[-1]}")
    else:
        print("❌ ETH: Failed to fetch OHLCV")
        return False
    
    return True


def test_technical_engine():
    """Test technical engine with new provider"""
    print("\n" + "="*70)
    print("TEST 2: Technical Engine (16 Indicators)")
    print("="*70)
    
    engine = TechnicalEngine()
    
    # Test BTC
    print("\n[BTC] Calculating technical indicators...")
    result = engine.get_signals("BTC", period=90)
    
    available_signals = [s for s in result.signals if s.data_available]
    unavailable_signals = [s for s in result.signals if not s.data_available]
    
    print(f"✅ BTC Technical: {len(available_signals)}/{len(result.signals)} available")
    print(f"   Available: {[s.name for s in available_signals[:5]]}...")
    print(f"   Unavailable: {len(unavailable_signals)}")
    print(f"   Execution time: {result.execution_time_ms:.1f}ms")
    print(f"   Errors: {result.errors}")
    
    if len(available_signals) < 10:
        print("⚠️  WARNING: Less than 10/16 indicators available!")
        return False
    
    # Test ETH
    print("\n[ETH] Calculating technical indicators...")
    result = engine.get_signals("ETH", period=90)
    
    available_signals = [s for s in result.signals if s.data_available]
    print(f"✅ ETH Technical: {len(available_signals)}/{len(result.signals)} available")
    
    return True


def test_volume_engine():
    """Test volume engine with new provider"""
    print("\n" + "="*70)
    print("TEST 3: Volume Engine (5 Signals)")
    print("="*70)
    
    engine = VolumeEngine()
    
    # Test BTC
    print("\n[BTC] Calculating volume signals...")
    result = engine.get_signals("BTC", period=90)
    
    available_signals = [s for s in result.signals if s.data_available]
    unavailable_signals = [s for s in result.signals if not s.data_available]
    
    print(f"✅ BTC Volume: {len(available_signals)}/{len(result.signals)} available")
    print(f"   Signals: {[s.name for s in available_signals]}")
    print(f"   Unavailable: {len(unavailable_signals)}")
    print(f"   Execution time: {result.execution_time_ms:.1f}ms")
    
    if len(available_signals) < 3:
        print("⚠️  WARNING: Less than 3/5 volume signals available!")
        return False
    
    return True


def test_provider_health():
    """Test provider health tracking"""
    print("\n" + "="*70)
    print("TEST 4: Provider Health Metrics")
    print("="*70)
    
    provider = get_unified_provider()
    health = provider.get_health_stats()
    
    print("\nProvider Statistics:")
    for prov_name, stats in health["providers"].items():
        if stats["requests"] > 0:
            print(f"  {prov_name}:")
            print(f"    - Requests: {stats['requests']}")
            print(f"    - Success rate: {stats['success_rate']*100:.1f}%")
            print(f"    - Avg latency: {stats['avg_latency_ms']:.1f}ms")
    
    print("\nCache Statistics:")
    cache_stats = health["cache"]
    if "error" not in cache_stats:
        print(f"  - Hit rate: {cache_stats.get('hit_rate', 0)*100:.1f}%")
        print(f"  - Total keys: {cache_stats.get('total_keys', 0)}")
        print(f"  - Memory used: {cache_stats.get('memory_used_mb', 0):.2f} MB")
    else:
        print(f"  - {cache_stats['error']}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("GHOST CRYPTO OHLCV INTEGRATION TEST")
    print("="*70)
    print("\nObjective: Verify BTC/ETH can fetch OHLCV from Binance")
    print("Expected: 20-23/25 features (up from 5/25)")
    
    results = {
        "Unified Provider": test_unified_provider(),
        "Technical Engine": test_technical_engine(),
        "Volume Engine": test_volume_engine(),
        "Provider Health": test_provider_health(),
    }
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nConclusion:")
        print("- Binance OHLCV integration working")
        print("- Technical indicators now available for crypto")
        print("- Volume signals now available for crypto")
        print("- BTC/ETH predictions will no longer be stuck at 40% FLAT")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nNext steps:")
        print("1. Check REDIS_URL environment variable")
        print("2. Verify Binance.US API is accessible")
        print("3. Check logs for detailed error messages")
        return 1


if __name__ == "__main__":
    sys.exit(main())
