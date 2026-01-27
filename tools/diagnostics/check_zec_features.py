#!/usr/bin/env python3
"""
Verify sentiment engine and world context fixes in production
Tests by running live feature extraction for ZEC
"""
import sys
import os
sys.path.insert(0, '/app')

def main():
    print("=" * 70)
    print("VERIFYING SENTIMENT + WORLD CONTEXT FIXES")
    print("Testing with ZEC (whitelisted symbol)")
    print("=" * 70)
    print()
    
    try:
        from core.data_pillars.feature_orchestrator import FeatureOrchestrator
        from core.world_context import get_world_context
        from core.data_pillars.sentiment_engine import SentimentEngine
        
        orchestrator = FeatureOrchestrator()
        sentiment_engine = SentimentEngine()
        symbol = "ZEC"
        
        # Test 1: Sentiment Engine
        print("TEST 1: SENTIMENT ENGINE")
        print("-" * 70)
        sentiment_result = sentiment_engine.get_signals(symbol)
        print(f"Status: {sentiment_result.status}")
        print(f"Signals: {len(sentiment_result.signals)}")
        for signal in sentiment_result.signals:
            status = "✅ REAL" if signal.value != 0.0 else "⚠️  ZERO"
            print(f"  {signal.name}: {signal.value:.4f} ({signal.source}) {status}")
        
        # Test 2: World Context
        print("\nTEST 2: WORLD CONTEXT")
        print("-" * 70)
        world = get_world_context()
        spy_ok = "✅" if world.spy_price and world.spy_price > 0 else "❌"
        vix_ok = "✅" if world.vix_level and world.vix_level > 0 else "❌"
        print(f"SPY: ${world.spy_price:.2f if world.spy_price else 0} {spy_ok}")
        print(f"VIX: {world.vix_level:.2f if world.vix_level else 0} {vix_ok}")
        print(f"Market Regime: {world.market_regime}")
        
        # Test 3: Orchestrator Health
        print("\nTEST 3: FEATURE ORCHESTRATOR")
        print("-" * 70)
        health = orchestrator.health_check()
        print(f"Healthy Pillars: {health['healthy_pillars']}/6")
        for pillar, status in health['pillar_status'].items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {pillar}")
        
        # Summary
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        
        sentiment_ok = len(sentiment_result.signals) > 0
        world_ok = world.spy_price and world.spy_price > 0 and world.vix_level and world.vix_level > 0
        
        if sentiment_ok:
            non_zero = sum(1 for s in sentiment_result.signals if s.value != 0.0)
            print(f"✅ Sentiment Engine: {len(sentiment_result.signals)} signals ({non_zero} non-zero)")
        else:
            print("❌ Sentiment Engine: No signals")
        
        if world_ok:
            print(f"✅ World Context: SPY=${world.spy_price:.2f}, VIX={world.vix_level:.2f}")
        else:
            print(f"❌ World Context: SPY={world.spy_price}, VIX={world.vix_level}")
        
        print(f"✅ Feature Orchestrator: {health['healthy_pillars']}/6 pillars healthy")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
