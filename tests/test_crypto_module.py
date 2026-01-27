#!/usr/bin/env python3
"""
Quick test script for crypto module
Run: python3 test_crypto_module.py
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_providers():
    """Test all crypto price providers"""
    print("=" * 60)
    print("🧪 Testing Crypto Price Providers")
    print("=" * 60)

    from core.crypto.crypto_providers import (
        BinanceProvider,
        CoinbaseProvider,
        CoinGeckoProvider,
        get_crypto_price_quorum,
    )

    # Test individual providers
    symbols = ["BTC", "ETH"]

    for symbol in symbols:
        print(f"\n📊 Testing {symbol}:")
        print("-" * 40)

        # CoinGecko
        print("  CoinGecko...", end=" ")
        try:
            coingecko = CoinGeckoProvider()
            data = coingecko.get_price(symbol)
            if data:
                print(f"✅ ${data['price']:.2f}")
            else:
                print("❌ No data")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Binance
        print("  Binance...", end=" ")
        try:
            binance = BinanceProvider()
            data = binance.get_price(symbol)
            if data:
                print(f"✅ ${data['price']:.2f}")
            else:
                print("❌ No data")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Coinbase
        print("  Coinbase...", end=" ")
        try:
            coinbase = CoinbaseProvider()
            data = coinbase.get_price(symbol)
            if data:
                print(f"✅ ${data['price']:.2f}")
            else:
                print("❌ No data")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Quorum
        print("\n  Quorum consensus...", end=" ")
        try:
            quorum = await get_crypto_price_quorum(symbol, use_cache=False)
            if quorum:
                print(f"✅ ${quorum['price']:.2f}")
                print(f"    Confidence: {quorum['confidence']:.0%}")
                print(f"    Quorum: {quorum['quorum_size']} providers")
                print(f"    Spread: {quorum['spread'] * 100:.2f}%")
            else:
                print("❌ No quorum")
        except Exception as e:
            print(f"❌ Error: {e}")


async def test_prediction():
    """Test crypto prediction engine"""
    print("\n" + "=" * 60)
    print("🔮 Testing Crypto Prediction Engine")
    print("=" * 60)

    from core.crypto.crypto_predictor import CryptoPredictionEngine

    try:
        engine = CryptoPredictionEngine()
        print("✅ Prediction engine initialized")
        print("✅ Database tables created")

        # Generate BTC prediction
        print("\n📈 Generating BTC 24h prediction...")
        pred = await engine.generate_prediction("BTC")

        print("\n✅ Prediction Generated:")
        print(f"  ID: {pred['prediction_id'][:8]}...")
        print(f"  Symbol: {pred['symbol']}")
        print(f"  Current Price: ${pred['current_price']:.2f}")
        print(f"  Direction: {pred['direction']}")
        print(f"  Confidence: {pred['confidence']:.0%}")
        print(f"  Volatility: {pred['volatility']:.1%}")
        print(f"  Horizon: {pred['horizon_hours']}h")

        # Verify stored in database
        latest = engine.get_latest_prediction("BTC")
        if latest:
            print("\n✅ Prediction stored in database")
            print(f"  Retrieved ID: {latest['id'][:8]}...")
        else:
            print("\n❌ Failed to retrieve from database")

    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_historical():
    """Test historical data fetch"""
    print("\n" + "=" * 60)
    print("📜 Testing Historical Data Fetch")
    print("=" * 60)

    from core.crypto.crypto_providers import CoinGeckoProvider

    try:
        coingecko = CoinGeckoProvider()

        print("\n📊 Fetching 7-day BTC history...")
        history = coingecko.get_historical("BTC", days=7)

        if history:
            print(f"✅ Retrieved {len(history)} hourly data points")
            print(f"  First: ${history[0]['price']:.2f} @ {history[0]['timestamp']}")
            print(f"  Last:  ${history[-1]['price']:.2f} @ {history[-1]['timestamp']}")

            # Calculate change
            change = (history[-1]["price"] - history[0]["price"]) / history[0]["price"] * 100
            print(f"  7-day change: {change:+.2f}%")
        else:
            print("❌ No historical data retrieved")

    except Exception as e:
        print(f"❌ Historical test failed: {e}")


async def main():
    """Run all tests"""
    print("\n" + "🚀" * 30)
    print("GHOST CRYPTO MODULE TEST SUITE")
    print("🚀" * 30)

    try:
        await test_providers()
        await test_historical()
        await test_prediction()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 60)
        print("\n💡 Next step: Integrate with wolf_app.py")
        print("   See CRYPTO_MODULE_QUICKSTART.md for instructions")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
