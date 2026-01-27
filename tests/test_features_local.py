#!/usr/bin/env python3
"""
Test feature extraction locally to verify fixes work.
"""

import sys
import asyncio

sys.path.insert(0, "/Users/studio713/ghost-protocol")

from core.data_pillars.feature_orchestrator import get_feature_orchestrator


async def test_features():
    """Test feature extraction for multiple symbols"""
    
    orchestrator = get_feature_orchestrator()
    
    test_symbols = ["MSFT", "AAPL", "SPY", "BTC", "ETH"]
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Testing: {symbol}")
        print('='*60)
        
        try:
            result = orchestrator.get_all_features(symbol, period=90)
            
            print(f"✅ Symbol: {result['symbol']}")
            print(f"📊 Features: {result['available_count']}/{result['feature_count']} ({result['available_count']/result['feature_count']*100:.1f}%)")
            print(f"⏱️  Time: {result['execution_time_ms']:.0f}ms")
            
            if result.get('errors'):
                print(f"\n⚠️  Errors ({len(result['errors'])}):")
                for err in result['errors'][:5]:  # Show first 5 errors
                    print(f"   - {err}")
            
            print(f"\n🟢 Available Features ({result['available_count']}):")
            features = result.get('features', {})
            available = {k: v for k, v in features.items() if v is not None}
            
            # Group by pillar
            price_features = {k: v for k, v in available.items() if k in ['PRICE', 'PREV_CLOSE', 'MARKET_CAP', 'VOLUME_24H']}
            technical_features = {k: v for k, v in available.items() if k in ['RSI_14', 'MACD_HISTOGRAM', 'SMA_20', 'SMA_50', 'SMA_200', 'BB_UPPER', 'BB_LOWER', 'BOLLINGER_POSITION', 'ATR_14', 'STOCH_K']}
            volume_features = {k: v for k, v in available.items() if k in ['VOLUME_SPIKE', 'VOLATILITY_20D', 'VOLATILITY_60D', 'VOLUME_MA_20', 'VOLUME_ROC']}
            sentiment_features = {k: v for k, v in available.items() if k in ['NEWS_SENTIMENT_SCORE', 'NEWS_COUNT_24H', 'BULLISH_RATIO']}
            world_features = {k: v for k, v in available.items() if k in ['SPY_PRICE', 'SPY_CHANGE', 'VIX_LEVEL', 'MARKET_REGIME']}
            
            if price_features:
                print(f"   Price Engine ({len(price_features)}):")
                for k, v in list(price_features.items())[:3]:
                    print(f"      {k}: {v}")
            
            if technical_features:
                print(f"   Technical Engine ({len(technical_features)}):")
                for k, v in list(technical_features.items())[:5]:
                    print(f"      {k}: {v}")
            
            if volume_features:
                print(f"   Volume Engine ({len(volume_features)}):")
                for k, v in list(volume_features.items())[:3]:
                    print(f"      {k}: {v}")
            
            if sentiment_features:
                print(f"   Sentiment Engine ({len(sentiment_features)}):")
                for k, v in sentiment_features.items():
                    print(f"      {k}: {v}")
            
            if world_features:
                print(f"   World Context ({len(world_features)}):")
                for k, v in world_features.items():
                    print(f"      {k}: {v}")
            
            print(f"\n🔴 Missing Features ({result['unavailable_count']}):")
            missing = [k for k, v in features.items() if v is None]
            for feat in missing[:10]:  # Show first 10
                print(f"   - {feat}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_features())
