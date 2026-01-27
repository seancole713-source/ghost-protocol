#!/usr/bin/env python3
"""
PHASE 3: Deep Test for PACS & BTC Predictions
==============================================

Tests the /api/predict/run endpoint for both stock and crypto symbols.
"""

import json
import sys
import time
import requests

# Configuration
BASE_URL = "https://ghost-protocol-production.up.railway.app"
# BASE_URL = "http://localhost:8000"  # Uncomment for local testing

SYMBOLS = ["PACS", "BTC"]
TIMEOUT_S = 10  # Max wait per prediction


def test_prediction(symbol: str) -> dict:
    """
    Test prediction endpoint for a single symbol.
    
    Returns:
        {
            "symbol": str,
            "ok": bool,
            "duration_s": float,
            "response": dict or None,
            "error": str or None
        }
    """
    url = f"{BASE_URL}/api/predict/run"
    start = time.monotonic()
    
    try:
        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=TIMEOUT_S
        )
        duration = time.monotonic() - start
        
        if response.status_code == 200:
            data = response.json()
            return {
                "symbol": symbol,
                "ok": True,
                "duration_s": duration,
                "response": data,
                "error": None
            }
        else:
            return {
                "symbol": symbol,
                "ok": False,
                "duration_s": duration,
                "response": None,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
            
    except requests.Timeout:
        duration = time.monotonic() - start
        return {
            "symbol": symbol,
            "ok": False,
            "duration_s": duration,
            "response": None,
            "error": f"Timeout after {TIMEOUT_S}s"
        }
    except Exception as e:
        duration = time.monotonic() - start
        return {
            "symbol": symbol,
            "ok": False,
            "duration_s": duration,
            "response": None,
            "error": str(e)
        }


def format_result(result: dict) -> str:
    """Format test result for display."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Symbol: {result['symbol']}")
    lines.append(f"Status: {'✅ PASS' if result['ok'] else '❌ FAIL'}")
    lines.append(f"Duration: {result['duration_s']:.2f}s")
    
    if result['error']:
        lines.append(f"Error: {result['error']}")
    
    if result['response']:
        resp = result['response']
        lines.append(f"\nPrediction Details:")
        lines.append(f"  Direction: {resp.get('direction', 'N/A')}")
        lines.append(f"  Confidence: {resp.get('confidence', 0)*100:.1f}%")
        lines.append(f"  Current Price: ${resp.get('current_price', 0):.2f}")
        lines.append(f"  Features: {resp.get('available_count', 0)}/{resp.get('feature_count', 0)}")
        lines.append(f"  Response Time: {resp.get('duration_ms', 0)}ms")
    
    lines.append('='*60)
    return '\n'.join(lines)


def main():
    """Run prediction tests for all symbols."""
    print("\n🔍 GHOST PREDICTION ENDPOINT TEST")
    print(f"Base URL: {BASE_URL}")
    print(f"Symbols: {', '.join(SYMBOLS)}\n")
    
    results = []
    for symbol in SYMBOLS:
        print(f"Testing {symbol}...", end=" ", flush=True)
        result = test_prediction(symbol)
        results.append(result)
        print(f"{'✅' if result['ok'] else '❌'} ({result['duration_s']:.2f}s)")
    
    # Print detailed results
    for result in results:
        print(format_result(result))
    
    # Summary
    passed = sum(1 for r in results if r['ok'])
    failed = len(results) - passed
    
    print(f"\n📊 SUMMARY")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
