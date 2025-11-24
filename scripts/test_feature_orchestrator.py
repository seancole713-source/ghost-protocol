#!/usr/bin/env python3
"""
Test Feature Orchestrator - Diagnose Feature Extraction
========================================================

Runs feature orchestrator for test symbols and shows results.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_pillars.feature_orchestrator import get_feature_orchestrator


def test_orchestrator():
    """Test feature orchestrator with multiple symbols"""
    
    orchestrator = get_feature_orchestrator()
    
    # Test symbols: stock, crypto, VIP
    test_symbols = ["AAPL", "BTC", "WOLF"]
    
    print("=" * 80)
    print("FEATURE ORCHESTRATOR DIAGNOSTIC")
    print("=" * 80)
    print()
    
    for symbol in test_symbols:
        print(f"\n{'=' * 80}")
        print(f"SYMBOL: {symbol}")
        print(f"{'=' * 80}")
        
        try:
            result = orchestrator.get_all_features(symbol, period=90)
            
            print(f"\n✅ SUCCESS")
            print(f"   Available Features: {result['available_count']}/{result['feature_count']}")
            print(f"   Execution Time: {result['execution_time_ms']:.1f}ms")
            print(f"   Errors: {len(result['errors'])}")
            
            # Show pillar status
            print(f"\n📊 PILLAR AVAILABILITY:")
            for pillar, status in result['feature_availability'].items():
                print(f"   {pillar:25s} {status}")
            
            # Show available features
            features = result.get('features', {})
            available_features = {k: v for k, v in features.items() if v is not None}
            
            print(f"\n✨ AVAILABLE FEATURES ({len(available_features)}):")
            for fname, fval in sorted(available_features.items()):
                if isinstance(fval, float):
                    print(f"   {fname:30s} {fval:10.4f}")
                else:
                    print(f"   {fname:30s} {fval}")
            
            # Show unavailable features
            unavailable = {k: v for k, v in features.items() if v is None}
            if unavailable:
                print(f"\n❌ UNAVAILABLE FEATURES ({len(unavailable)}):")
                for fname in sorted(unavailable.keys()):
                    print(f"   {fname}")
            
            # Show errors
            if result['errors']:
                print(f"\n⚠️  ERRORS:")
                for err in result['errors']:
                    print(f"   {err}")
        
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    test_orchestrator()
