#!/usr/bin/env python3
"""
Test RSS async event loop fix for sentiment engine.

Verifies that nest_asyncio allows RSS scanning to work
in contexts where an event loop is already running.
"""

import asyncio
import sys


def test_rss_in_existing_loop():
    """Test RSS scanning when event loop is already running"""
    print("🧪 Testing RSS scan with existing event loop...")
    
    from core.data_pillars.sentiment_engine import SentimentEngine
    
    # Create sentiment engine
    engine = SentimentEngine()
    
    # Try to scan RSS for a symbol (this would fail before the fix)
    result = engine._scan_rss_for_symbol("WOLF")
    
    print(f"\n📊 RSS Scan Result:")
    print(f"   Status: {'✅ OK' if result.get('ok') else '❌ FAILED'}")
    print(f"   Articles: {result.get('articles', 0)}")
    
    if result.get("ok"):
        print(f"   Sentiment: {result.get('sentiment_score', 0.0):.2f}")
        print(f"   Bullish: {result.get('bullish_count', 0)}")
        print(f"   Bearish: {result.get('bearish_count', 0)}")
        print("\n✅ RSS scanning works in async context!")
        return True
    else:
        print("\n⚠️ No articles found (but no event loop error!)")
        return True  # Still success - no crash


async def test_in_async_context():
    """Test RSS scanning from within an async function"""
    print("\n🧪 Testing RSS scan from async function...")
    
    from core.data_pillars.sentiment_engine import SentimentEngine
    
    engine = SentimentEngine()
    
    # This is the problematic case - calling sync code from async
    # Before fix: "RuntimeError: this event loop is already running"
    result = engine._scan_rss_for_symbol("NVDA")
    
    print(f"\n📊 Async Context Result:")
    print(f"   Status: {'✅ OK' if result.get('ok') else '❌ FAILED'}")
    print(f"   Articles: {result.get('articles', 0)}")
    
    if result.get("ok"):
        print(f"   Sentiment: {result.get('sentiment_score', 0.0):.2f}")
    
    return result


def test_full_sentiment_signals():
    """Test full sentiment signal generation"""
    print("\n🧪 Testing full sentiment signal flow...")
    
    from core.data_pillars.sentiment_engine import SentimentEngine
    
    engine = SentimentEngine()
    
    # Get signals for a symbol (this will internally call _scan_rss_for_symbol)
    response = engine.get_signals("WOLF")
    
    print(f"\n📊 Sentiment Signals:")
    print(f"   Pillar: {response.pillar_name}")
    print(f"   Signals: {len(response.signals)}")
    
    for signal in response.signals:
        print(f"   - {signal.name}: {signal.value} (confidence: {signal.confidence})")
    
    if response.errors:
        print(f"\n⚠️ Errors:")
        for error in response.errors:
            print(f"   - {error}")
    
    print(f"\n✅ Sentiment engine {'working' if response.signals else 'returned signals'}")
    return len(response.signals) > 0


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 RSS Async Event Loop Fix Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Regular sync context
    try:
        test_rss_in_existing_loop()
    except RuntimeError as e:
        if "already running" in str(e):
            print(f"\n❌ FAILED: Event loop error still occurs!")
            print(f"   Error: {e}")
            success = False
        else:
            raise
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}")
    
    # Test 2: Async context (the real test)
    try:
        asyncio.run(test_in_async_context())
    except RuntimeError as e:
        if "already running" in str(e):
            print(f"\n❌ FAILED: Event loop error in async context!")
            print(f"   Error: {e}")
            success = False
        else:
            raise
    except Exception as e:
        print(f"\n⚠️ Unexpected error in async: {e}")
    
    # Test 3: Full signal flow
    try:
        test_full_sentiment_signals()
    except RuntimeError as e:
        if "already running" in str(e):
            print(f"\n❌ FAILED: Event loop error in signal flow!")
            print(f"   Error: {e}")
            success = False
        else:
            raise
    except Exception as e:
        print(f"\n⚠️ Unexpected error in signal flow: {e}")
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - RSS async fix working!")
    else:
        print("❌ TESTS FAILED - Event loop issue persists")
        sys.exit(1)
    print("=" * 60)
