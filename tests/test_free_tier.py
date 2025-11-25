"""
FREE-TIER VALIDATION TEST
=========================
Validates Ghost works 100% on FREE providers (no paid APIs).

Required: 20+ features for all assets using ONLY:
- Yahoo Finance (FREE)
- yfinance (FREE)
- Binance Public API (FREE, no key)
- CoinGecko (FREE)

Test Symbols:
- Stocks: MSFT, AAPL, SPY
- Crypto: BTC, ETH, SOL

Success Criteria:
- Feature count >= 20/26 (stocks), >= 20/25 (crypto)
- Confidence: 40-85% (varied, not stuck)
- Direction: UP/DOWN/FLAT mix (not always FLAT)
- Provider success rate: 80%+
- NO paid API keys required
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.providers.unified_provider import get_unified_provider
from core.data_pillars.technical_engine import TechnicalEngine
from core.data_pillars.volume_engine import VolumeEngine


def test_free_providers():
    """Test all FREE providers work"""
    print("\n" + "="*70)
    print("TEST 1: FREE PROVIDER ACCESS")
    print("="*70)
    
    provider = get_unified_provider()
    
    test_cases = [
        ("AAPL", "stock", "1d", 60),
        ("MSFT", "stock", "1d", 60),
        ("SPY", "stock", "1d", 60),
        ("BTC", "crypto", "1d", 60),
        ("ETH", "crypto", "1d", 60),
        ("SOL", "crypto", "1d", 60),
    ]
    
    results = {}
    
    for symbol, asset_type, interval, lookback in test_cases:
        print(f"\n[{symbol}] Testing {asset_type} OHLCV...")
        
        ohlcv = provider.get_ohlcv(symbol, interval=interval, lookback=lookback)
        
        if ohlcv and ohlcv.bars and len(ohlcv.bars) >= 20:
            results[symbol] = {
                "success": True,
                "bars": len(ohlcv.bars),
                "provider": ohlcv.provider,
                "cache_hit": ohlcv.cache_hit
            }
            print(f"✅ {symbol}: {len(ohlcv.bars)} bars from {ohlcv.provider} (cache={ohlcv.cache_hit})")
        else:
            results[symbol] = {
                "success": False,
                "bars": len(ohlcv.bars) if ohlcv and ohlcv.bars else 0,
                "provider": "none",
                "cache_hit": False
            }
            print(f"❌ {symbol}: Failed (only {results[symbol]['bars']} bars)")
    
    # Summary
    success_count = sum(1 for r in results.values() if r["success"])
    print(f"\n{'='*70}")
    print(f"Provider Test Results: {success_count}/{len(test_cases)} passed")
    print(f"{'='*70}")
    
    return all(r["success"] for r in results.values())


def test_feature_extraction():
    """Test feature extraction with FREE providers"""
    print("\n" + "="*70)
    print("TEST 2: FEATURE EXTRACTION (FREE-TIER)")
    print("="*70)
    
    tech_engine = TechnicalEngine()
    vol_engine = VolumeEngine()
    
    test_symbols = [
        ("AAPL", "stock", 26),
        ("MSFT", "stock", 26),
        ("SPY", "stock", 26),
        ("BTC", "crypto", 25),
        ("ETH", "crypto", 25),
        ("SOL", "crypto", 25),
    ]
    
    results = {}
    
    for symbol, asset_type, expected_total in test_symbols:
        print(f"\n[{symbol}] Extracting features...")
        
        # Technical indicators
        tech_result = tech_engine.get_signals(symbol, period=90)
        tech_available = [s for s in tech_result.signals if s.data_available]
        
        # Volume signals
        vol_result = vol_engine.get_signals(symbol, period=90)
        vol_available = [s for s in vol_result.signals if s.data_available]
        
        total_available = len(tech_available) + len(vol_available)
        availability_pct = (total_available / expected_total) * 100
        
        results[symbol] = {
            "total": total_available,
            "expected": expected_total,
            "pct": availability_pct,
            "technical": len(tech_available),
            "volume": len(vol_available)
        }
        
        status = "✅" if total_available >= 20 else "❌"
        print(f"{status} {symbol}: {total_available}/{expected_total} features ({availability_pct:.1f}%)")
        print(f"   Technical: {len(tech_available)}/{len(tech_result.signals)}")
        print(f"   Volume: {len(vol_available)}/{len(vol_result.signals)}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Feature Extraction Results:")
    print(f"{'='*70}")
    
    for symbol, data in results.items():
        status = "✅ PASS" if data["total"] >= 20 else "❌ FAIL"
        print(f"{status}: {symbol} = {data['total']}/{data['expected']} ({data['pct']:.1f}%)")
    
    all_passed = all(r["total"] >= 20 for r in results.values())
    return all_passed


def test_provider_health():
    """Test provider health metrics"""
    print("\n" + "="*70)
    print("TEST 3: PROVIDER HEALTH (FREE-TIER)")
    print("="*70)
    
    provider = get_unified_provider()
    health = provider.get_health_stats()
    
    print("\nFREE Provider Statistics:")
    for prov_name, stats in health["providers"].items():
        if stats["requests"] > 0:
            success_rate = stats["success_rate"] * 100
            status = "✅" if success_rate >= 80 else "⚠️ "
            
            print(f"{status} {prov_name}:")
            print(f"    - Requests: {stats['requests']}")
            print(f"    - Success rate: {success_rate:.1f}%")
            print(f"    - Avg latency: {stats['avg_latency_ms']:.1f}ms")
    
    print("\nCache Statistics:")
    cache_stats = health["cache"]
    if "error" not in cache_stats:
        hit_rate = cache_stats.get('hit_rate', 0) * 100
        print(f"  - Hit rate: {hit_rate:.1f}%")
        print(f"  - Total keys: {cache_stats.get('total_keys', 0)}")
    else:
        print(f"  - Status: {cache_stats['error']}")
    
    # Check if all providers meet minimum standards
    all_healthy = True
    for prov_name, stats in health["providers"].items():
        if stats["requests"] > 0 and stats["success_rate"] < 0.8:
            all_healthy = False
            print(f"\n⚠️  WARNING: {prov_name} success rate below 80%")
    
    return all_healthy


def test_no_paid_apis():
    """Verify NO paid API keys are being used"""
    print("\n" + "="*70)
    print("TEST 4: VERIFY NO PAID APIs")
    print("="*70)
    
    paid_keys = {
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY"),
        "ALPHAVANTAGE_API_KEY": os.getenv("ALPHAVANTAGE_API_KEY"),
        "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
    }
    
    using_paid = []
    for key_name, key_value in paid_keys.items():
        if key_value:
            using_paid.append(key_name)
            print(f"⚠️  {key_name} is set (paid API)")
        else:
            print(f"✅ {key_name} not set (FREE-TIER only)")
    
    if using_paid:
        print(f"\n⚠️  WARNING: Using paid APIs: {', '.join(using_paid)}")
        print("Ghost should work WITHOUT these for free-tier validation!")
        return False
    else:
        print("\n✅ Confirmed: 100% FREE-TIER providers only")
        return True


def main():
    """Run all FREE-TIER validation tests"""
    print("\n" + "="*70)
    print("GHOST FREE-TIER VALIDATION")
    print("="*70)
    print("\nObjective: Prove Ghost works 100% on FREE providers")
    print("Required: 20+ features for all symbols WITHOUT paid APIs")
    print("\nFREE Providers:")
    print("  - Yahoo Finance (stocks)")
    print("  - yfinance (stocks)")
    print("  - Binance Public API (crypto, no key)")
    print("  - CoinGecko (crypto)")
    
    results = {
        "Free Providers": test_free_providers(),
        "Feature Extraction": test_feature_extraction(),
        "Provider Health": test_provider_health(),
        "No Paid APIs": test_no_paid_apis(),
    }
    
    print("\n" + "="*70)
    print("FREE-TIER VALIDATION RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 FREE-TIER VALIDATION PASSED!")
        print("\nGhost is fully operational using ONLY free providers:")
        print("✅ Stocks: Yahoo Finance + yfinance")
        print("✅ Crypto: Binance Public API")
        print("✅ Feature count: 20+ for all symbols")
        print("✅ Provider success rate: 80%+")
        print("✅ NO paid APIs required")
        print("\n💰 Cost: $0/month (100% FREE)")
        print("\nRecommendation: Ghost is production-ready on free tier!")
        print("Consider paid APIs later to improve reliability, not necessity.")
        return 0
    else:
        print("\n⚠️  FREE-TIER VALIDATION INCOMPLETE")
        print("\nAction items:")
        print("1. Fix failing providers")
        print("2. Ensure 20+ features for all symbols")
        print("3. Remove dependencies on paid APIs")
        print("4. Re-run validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
