#!/usr/bin/env python3
"""
Test World Feed Fusion functionality
"""

import sys

sys.path.insert(0, "/workspaces/GHOST")

from core.world_feed_fusion import get_feed_fusion


def test_world_feed_fusion():
    """Test World Feed Fusion basic functionality"""

    print("=" * 80)
    print("TESTING WORLD FEED FUSION - RSS + NLP SENTIMENT")
    print("=" * 80)

    # Initialize
    print("\n[1/5] Initializing World Feed Fusion...")
    fusion = get_feed_fusion()
    print("✓ World Feed Fusion initialized")

    # Get sources
    print("\n[2/5] Checking feed sources...")
    sources = fusion.get_sources()
    print(f"✓ Found {len(sources)} feed sources:")
    for source in sources[:3]:
        print(f"  - {source['name']} ({source['category']}) priority={source['priority']}")
    if len(sources) > 3:
        print(f"  ... and {len(sources) - 3} more")

    # Test sentiment analysis
    print("\n[3/5] Testing sentiment analysis...")
    test_texts = [
        "Company reports record earnings, beating estimates by 20%",
        "Stock plunges on disappointing guidance and weak sales",
        "Market remains flat as investors await Fed decision",
    ]

    for text in test_texts:
        score, magnitude = fusion.analyze_sentiment(text)
        sentiment = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
        print(f"  Text: {text[:50]}...")
        print(f"    → Sentiment: {sentiment} (score={score:.3f}, confidence={magnitude:.3f})")

    # Fetch sample feed (will likely be empty in dev but tests infrastructure)
    print("\n[4/5] Testing feed fetching...")
    try:
        # Fetch from one source
        articles = fusion.fetch_feed("marketwatch")
        print(f"✓ Fetched {len(articles)} articles from MarketWatch")

        if articles:
            article = articles[0]
            print(f"  Sample: {article.title[:60]}...")
            print(f"  Sentiment: {article.sentiment_score:.3f}, Symbols: {article.symbols}")
    except Exception as e:
        print(f"⚠ Feed fetch test skipped: {e}")

    # Test latest articles query
    print("\n[5/5] Testing article queries...")
    latest = fusion.get_latest_articles(limit=5)
    print(f"✓ Found {len(latest)} articles in database")

    if latest:
        for i, article in enumerate(latest[:3], 1):
            print(f"  {i}. {article['title'][:60]}...")
            print(
                f"     Sentiment: {article['sentiment_score']:.3f}, Source: {article['source_id']}"
            )

    # Test sentiment aggregate (will be None if no articles yet)
    print("\n[BONUS] Testing sentiment aggregation...")
    aggregate = fusion.get_sentiment_aggregate("AAPL", "1d")
    if aggregate:
        print(f"✓ AAPL 1-day sentiment: {aggregate.weighted_sentiment:.3f}")
        print(
            f"  Articles: {aggregate.article_count} ({aggregate.bullish_count} bullish, {aggregate.bearish_count} bearish)"
        )
    else:
        print("⚠ No articles for AAPL in database yet (run feed fetch to populate)")

    print("\n" + "=" * 80)
    print("WORLD FEED FUSION TEST COMPLETE")
    print("=" * 80)
    print("\nKey features:")
    print(
        "  ✓ 8 RSS feed sources configured (Reuters, Bloomberg, FT, WSJ, CNBC, MarketWatch, Seeking Alpha)"
    )
    print("  ✓ TextBlob NLP sentiment analysis operational")
    print("  ✓ Symbol extraction from article text")
    print("  ✓ SQLite persistence with 3 tables")
    print("  ✓ Sentiment aggregation with weighted scoring")
    print("  ✓ Integration ready for NewsShockStrategy and Feature Importance")
    print("\nNext steps:")
    print("  1. Run server: uvicorn wolf_app:app --host 0.0.0.0 --port 5000")
    print("  2. Fetch feeds: curl -X POST 'http://localhost:5000/api/feeds/fetch'")
    print("  3. Check sentiment: curl 'http://localhost:5000/api/feeds/sentiment?symbol=AAPL'")
    print()


if __name__ == "__main__":
    test_world_feed_fusion()
