#!/usr/bin/env python3
"""
Quick test to verify parallel multi-run predictions.
Tests that predictions complete within reasonable time (< 5 minutes).
"""
import sys
import time
import os

# Add wolf_app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
from wolf_app import _generate_multi_symbol_predictions

def test_parallel_multi_run():
    """Test parallel multi-run prediction generation with timing."""
    print("═══ TESTING PARALLEL MULTI-RUN ═══\n")
    
    start_time = time.time()
    
    print("Calling _generate_multi_symbol_predictions()...")
    result = _generate_multi_symbol_predictions()
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Completed in {elapsed:.1f}s")
    print(f"   OK: {result.get('ok')}")
    print(f"   Cached: {result.get('cached', False)}")
    print(f"   Counts: {result.get('counts')}")
    print(f"   Total predictions: {result.get('total')}")
    
    if result.get("generation_time_seconds"):
        print(f"   Generation time: {result['generation_time_seconds']:.1f}s")
    
    # Show sample predictions
    predictions = result.get("predictions", {})
    if predictions.get("stocks"):
        print(f"\n   Sample stocks: {predictions['stocks'][:3]}")
    if predictions.get("crypto"):
        print(f"   Sample crypto: {predictions['crypto'][:3]}")
    
    # Show failed symbols
    failed = result.get("failed_symbols")
    if failed and (failed.get("stocks") or failed.get("crypto")):
        stock_fails = len(failed.get("stocks", []))
        crypto_fails = len(failed.get("crypto", []))
        print(f"\n   Failed: {stock_fails} stocks, {crypto_fails} cryptos")
        if stock_fails > 0:
            print(f"      Stock failures: {failed['stocks'][:3]}")
        if crypto_fails > 0:
            print(f"      Crypto failures: {failed['crypto'][:3]}")
    
    # Assertions
    assert result.get("ok") is True, "Result should be ok=True"
    assert result.get("total", 0) > 0, "Should have at least 1 prediction"
    assert elapsed < 300, f"Should complete in < 5min (took {elapsed:.1f}s)"
    
    print(f"\n✅ All assertions passed!")
    print(f"   Speed improvement: {1362/elapsed:.1f}x faster than sequential")
    
    return result

if __name__ == "__main__":
    try:
        result = test_parallel_multi_run()
        print("\n═══ TEST PASSED ═══")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
