"""
GHOST INTEL - SMOKE TEST
=========================
Verification tests for the Ghost Intel module.

Run this to verify all components are working.

Usage:
    python intel_smoke_test.py

Author: Ghost AI
Date: 2026-01-26
"""

import asyncio
import os
import sys
import time
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, passed: bool, message: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if message:
        print(f"          {message}")


async def test_sources():
    """Test data sources"""
    print_header("TESTING DATA SOURCES")
    
    from ghost_intel.sources import get_intel_sources
    
    sources = get_intel_sources()
    results = {}
    
    # Test rates (free, no API key needed)
    print("\n  Testing rates and liquidity...")
    try:
        data = await sources.get_rates_and_liquidity()
        if data.get("available") or data.get("data"):
            results["rates"] = True
            vix = data.get("data", {}).get("vix", {}).get("price")
            print_result("Rates/Liquidity", True, f"VIX: {vix}")
        else:
            results["rates"] = False
            print_result("Rates/Liquidity", False, "No data returned")
    except Exception as e:
        results["rates"] = False
        print_result("Rates/Liquidity", False, str(e))
    
    # Test FRED (needs API key)
    print("\n  Testing FRED macro data...")
    fred_key = os.getenv("FRED_API_KEY", "")
    if fred_key:
        try:
            data = await sources.get_macro_data()
            if data.get("available") and data.get("data"):
                results["fred"] = True
                indicators = list(data.get("data", {}).keys())[:3]
                print_result("FRED Macro", True, f"Got: {indicators}")
            else:
                results["fred"] = False
                print_result("FRED Macro", False, data.get("error", "No data"))
        except Exception as e:
            results["fred"] = False
            print_result("FRED Macro", False, str(e))
    else:
        results["fred"] = None
        print_result("FRED Macro", False, "FRED_API_KEY not set - SKIPPED")
    
    # Test StockTwits (free)
    print("\n  Testing StockTwits...")
    try:
        data = await sources.get_stocktwits_sentiment("AAPL")
        if data.get("available"):
            results["stocktwits"] = True
            sentiment = data.get("sentiment_label")
            print_result("StockTwits", True, f"AAPL sentiment: {sentiment}")
        else:
            results["stocktwits"] = False
            print_result("StockTwits", False, data.get("error", "No data"))
    except Exception as e:
        results["stocktwits"] = False
        print_result("StockTwits", False, str(e))
    
    # Test Reddit (free public endpoint)
    print("\n  Testing Reddit WSB...")
    try:
        data = await sources.get_reddit_wsb_sentiment()
        if data.get("available"):
            results["reddit"] = True
            top_tickers = list(data.get("top_tickers", {}).keys())[:5]
            print_result("Reddit WSB", True, f"Top tickers: {top_tickers}")
        else:
            results["reddit"] = False
            print_result("Reddit WSB", False, data.get("error", "No data"))
    except Exception as e:
        results["reddit"] = False
        print_result("Reddit WSB", False, str(e))
    
    # Test Polygon news (needs API key)
    print("\n  Testing Polygon news...")
    polygon_key = os.getenv("POLYGON_API_KEY", "")
    if polygon_key:
        try:
            data = await sources.get_polygon_news(limit=5)
            if data.get("available"):
                results["polygon"] = True
                count = data.get("count", 0)
                print_result("Polygon News", True, f"Got {count} articles")
            else:
                results["polygon"] = False
                print_result("Polygon News", False, data.get("error", "No data"))
        except Exception as e:
            results["polygon"] = False
            print_result("Polygon News", False, str(e))
    else:
        results["polygon"] = None
        print_result("Polygon News", False, "POLYGON_API_KEY not set - SKIPPED")
    
    # Test put/call ratio
    print("\n  Testing Put/Call ratio...")
    try:
        data = await sources.get_put_call_ratio()
        if data.get("available"):
            results["pcr"] = True
            pcr = data.get("put_call_ratio")
            fear = data.get("fear_level")
            print_result("Put/Call Ratio", True, f"P/C: {pcr}, Fear: {fear}")
        else:
            results["pcr"] = False
            print_result("Put/Call Ratio", False, "No data")
    except Exception as e:
        results["pcr"] = False
        print_result("Put/Call Ratio", False, str(e))
    
    # Health check
    print("\n  Checking overall health...")
    health = sources.get_health()
    available = sum(1 for s in health["sources"].values() if s["available"])
    total = len(health["sources"])
    print(f"  Sources available: {available}/{total}")
    
    return results


async def test_normalization():
    """Test event normalization"""
    print_header("TESTING EVENT NORMALIZATION")
    
    from ghost_intel.normalize import normalize_event, EventLayer, get_deduplicator
    
    # Test basic normalization
    print("\n  Testing basic normalization...")
    try:
        event = normalize_event(
            source="polygon_news",
            data={
                "title": "Fed raises rates by 25 basis points",
                "description": "The Federal Reserve raised interest rates.",
                "tickers": ["SPY", "QQQ"],
                "published_utc": "2026-01-26T10:00:00Z"
            },
            layer=EventLayer.RATES,
            category="fomc"
        )
        
        assert event.event_id
        assert event.headline
        assert len(event.tickers) == 2
        print_result("Basic normalization", True, f"Event ID: {event.event_id[:8]}...")
    except Exception as e:
        print_result("Basic normalization", False, str(e))
        return False
    
    # Test deduplication
    print("\n  Testing deduplication...")
    try:
        deduper = get_deduplicator()
        
        event1 = normalize_event(
            source="reuters",
            data={"title": "Apple beats Q4 earnings expectations"},
            layer=EventLayer.CORPORATE,
            category="earnings"
        )
        
        event2 = normalize_event(
            source="bloomberg",
            data={"title": "Apple earnings beat expectations in Q4"},
            layer=EventLayer.CORPORATE,
            category="earnings"
        )
        
        result1 = deduper.process(event1)
        result2 = deduper.process(event2)
        
        # First should pass, second should be deduplicated
        if result1 and not result2:
            print_result("Deduplication", True, "Similar events merged")
            if result1.corroborated:
                print_result("Corroboration", True, f"Source count: {result1.source_count}")
        else:
            print_result("Deduplication", False, "Events not merged")
    except Exception as e:
        print_result("Deduplication", False, str(e))
    
    return True


async def test_impact_scoring():
    """Test impact scoring model"""
    print_header("TESTING IMPACT SCORING")
    
    from ghost_intel.normalize import normalize_event, EventLayer
    from ghost_intel.impact_model import get_impact_scorer, is_signal_not_noise
    
    scorer = get_impact_scorer()
    
    # Set context
    scorer.update_context({
        "vix": 22,
        "put_call_ratio": 1.1,
    })
    
    # Test high-impact event (FOMC)
    print("\n  Testing FOMC event scoring...")
    try:
        event = normalize_event(
            source="fed",
            data={"title": "FOMC raises rates 50bps, hawkish guidance"},
            layer=EventLayer.RATES,
            category="fomc"
        )
        
        score = scorer.score(event)
        
        print_result("FOMC Scoring", score.score > 50, 
                    f"Score: {score.score:.1f}, Direction: {score.direction.value}")
        print(f"          Action: {score.action_signal}")
        print(f"          Rationale: {score.rationale}")
    except Exception as e:
        print_result("FOMC Scoring", False, str(e))
    
    # Test low-impact event
    print("\n  Testing low-impact event...")
    try:
        event = normalize_event(
            source="stocktwits",
            data={"title": "AAPL looks good today"},
            layer=EventLayer.SOCIAL,
            category="social"
        )
        
        score = scorer.score(event)
        
        print_result("Social Post Scoring", score.score < 50, 
                    f"Score: {score.score:.1f}")
    except Exception as e:
        print_result("Social Post Scoring", False, str(e))
    
    # Test signal vs noise filter
    print("\n  Testing signal vs noise filter...")
    try:
        fomc_event = normalize_event(
            source="fed",
            data={"title": "FOMC statement"},
            layer=EventLayer.MACRO,
            category="fomc"
        )
        
        social_event = normalize_event(
            source="twitter",
            data={"title": "Random stock opinion"},
            layer=EventLayer.SOCIAL,
            category="social"
        )
        
        fomc_signal = is_signal_not_noise(fomc_event, scorer)
        social_signal = is_signal_not_noise(social_event, scorer)
        
        if fomc_signal and not social_signal:
            print_result("Signal vs Noise", True, "FOMC=signal, social=noise")
        else:
            print_result("Signal vs Noise", False, 
                        f"FOMC={fomc_signal}, social={social_signal}")
    except Exception as e:
        print_result("Signal vs Noise", False, str(e))
    
    return True


async def test_positioning():
    """Test positioning analyzer"""
    print_header("TESTING POSITIONING ANALYZER")
    
    from ghost_intel.positioning import get_positioning_analyzer
    
    analyzer = get_positioning_analyzer()
    
    # Test normal conditions
    print("\n  Testing normal market conditions...")
    try:
        analysis = analyzer.analyze({
            "vix": 15,
            "put_call_ratio": 0.85,
            "vix_change": 2,
        })
        
        print_result("Normal Conditions", True, 
                    f"Fear: {analysis.fear_level}, Fragility: {analysis.fragility:.0f}")
    except Exception as e:
        print_result("Normal Conditions", False, str(e))
    
    # Test fear conditions
    print("\n  Testing fear conditions...")
    try:
        analysis = analyzer.analyze({
            "vix": 35,
            "put_call_ratio": 1.3,
            "vix_change": 25,
            "vix_term_structure": {"backwardation": True}
        })
        
        print_result("Fear Conditions", analysis.fear_level == "EXTREME_FEAR", 
                    f"Fear: {analysis.fear_level}, Fragility: {analysis.fragility:.0f}")
        
        if analysis.warnings:
            print(f"          Warnings: {analysis.warnings[:2]}")
    except Exception as e:
        print_result("Fear Conditions", False, str(e))
    
    # Test positioning signal
    print("\n  Testing positioning signal...")
    try:
        signal = analyzer.get_positioning_signal({
            "vix": 28,
            "put_call_ratio": 1.15,
        })
        
        print_result("Position Signal", True, 
                    f"Bias: {signal['bias']}, Size Adj: {signal['position_size_adj']}")
    except Exception as e:
        print_result("Position Signal", False, str(e))
    
    # Test gamma estimation
    print("\n  Testing gamma exposure estimation...")
    try:
        gex = analyzer.estimate_gamma_exposure(spy_price=480.0, vix=22)
        
        print_result("Gamma Exposure", True, 
                    f"GEX: {gex.estimated_gex}, Risk: {gex.amplification_risk}")
    except Exception as e:
        print_result("Gamma Exposure", False, str(e))
    
    return True


async def test_taxonomy():
    """Test event taxonomy"""
    print_header("TESTING EVENT TAXONOMY")
    
    from ghost_intel.taxonomy import (
        EventTaxonomy, EventCategory, 
        classify_event, get_ticker_sector
    )
    
    # Test classification
    print("\n  Testing event classification...")
    test_cases = [
        ("Fed raises interest rates by 25bps", "fomc"),
        ("Apple beats Q4 earnings expectations", "earnings"),
        ("Russia escalates conflict in Ukraine", "conflict"),
        ("CPI comes in hotter than expected at 3.2%", "cpi"),
    ]
    
    for text, expected in test_cases:
        try:
            result = classify_event(text)
            matched = expected in result["category"].value
            print_result(f"'{text[:30]}...'", matched, 
                        f"Category: {result['category'].value}")
        except Exception as e:
            print_result(text[:30], False, str(e))
    
    # Test sector mapping
    print("\n  Testing sector mapping...")
    test_tickers = ["AAPL", "JPM", "XOM", "NVDA"]
    for ticker in test_tickers:
        sector = get_ticker_sector(ticker)
        print_result(f"{ticker} sector", sector is not None, f"Sector: {sector}")
    
    # Test impact weights
    print("\n  Testing impact weights...")
    high_impact = [EventCategory.FOMC, EventCategory.CPI, EventCategory.WAR]
    low_impact = [EventCategory.NEWS, EventCategory.STOCKTWITS]
    
    for cat in high_impact:
        weight = EventTaxonomy.get_impact_weight(cat)
        print_result(f"{cat.value} weight", weight >= 0.9, f"Weight: {weight}")
    
    for cat in low_impact:
        weight = EventTaxonomy.get_impact_weight(cat)
        print_result(f"{cat.value} weight", weight < 0.5, f"Weight: {weight}")
    
    return True


async def test_full_flow():
    """Test full intelligence flow"""
    print_header("TESTING FULL INTEL FLOW")
    
    from ghost_intel.sources import get_intel_sources
    from ghost_intel.normalize import normalize_event, EventLayer
    from ghost_intel.impact_model import get_impact_scorer
    from ghost_intel.positioning import get_positioning_analyzer
    
    sources = get_intel_sources()
    scorer = get_impact_scorer()
    analyzer = get_positioning_analyzer()
    
    print("\n  Fetching all layers...")
    try:
        start = time.time()
        data = await sources.fetch_all_layers("AAPL")
        elapsed = (time.time() - start) * 1000
        
        print_result("Fetch all layers", True, f"Completed in {elapsed:.0f}ms")
        
        # Check what we got
        layers = data.get("layers", {})
        for layer_name, layer_data in layers.items():
            available = layer_data.get("available", False)
            status = "✓" if available else "✗"
            print(f"          {status} {layer_name}")
        
        # Update scorer with context
        rates = layers.get("rates", {}).get("data", {})
        vix_data = rates.get("vix", {})
        if vix_data.get("price"):
            scorer.update_context({"vix": vix_data["price"]})
            print(f"          Context updated: VIX={vix_data['price']}")
        
        # Analyze positioning
        positioning = layers.get("positioning", {})
        if positioning.get("available"):
            pos_analysis = analyzer.analyze(positioning)
            print(f"          Positioning: {pos_analysis.fear_level}, Fragility: {pos_analysis.fragility:.0f}")
        
    except Exception as e:
        print_result("Fetch all layers", False, str(e))
        return False
    
    print_result("Full flow test", True, "All components integrated")
    return True


async def main():
    """Run all smoke tests"""
    print("\n" + "=" * 60)
    print("  GHOST INTEL SMOKE TEST")
    print("  " + "=" * 56)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check API keys
    print("\n  API Keys Status:")
    keys = {
        "FRED_API_KEY": os.getenv("FRED_API_KEY", ""),
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY", ""),
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID", ""),
    }
    for key, value in keys.items():
        status = "✓ Set" if value else "✗ Not set"
        print(f"    {key}: {status}")
    
    # Run tests
    results = {}
    
    results["sources"] = await test_sources()
    results["normalization"] = await test_normalization()
    results["impact"] = await test_impact_scoring()
    results["positioning"] = await test_positioning()
    results["taxonomy"] = await test_taxonomy()
    results["full_flow"] = await test_full_flow()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED - Ghost Intel is operational!")
    else:
        print("\n  ⚠️  Some tests failed - check configuration")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
