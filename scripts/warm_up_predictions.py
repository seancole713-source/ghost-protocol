#!/usr/bin/env python3
"""
Warm Up Predictions - Trigger predictions for all watchlist symbols
=====================================================================

Runs predictions for all symbols to populate ghost_score and system state.
"""

import requests
import time
import sys

BASE_URL = "https://ghost-protocol-production.up.railway.app"

def main():
    print("=" * 80)
    print("🟣 GHOST PROTOCOL - PREDICTION WARM-UP")
    print("=" * 80)
    print()
    
    # Get watchlist
    print("📋 Fetching watchlist...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v3/watchlist", timeout=10)
        data = resp.json()
        
        stocks = data.get("stocks", [])
        crypto = data.get("crypto", [])
        vip = data.get("vip", [])
        
        all_symbols = stocks + crypto + vip
        print(f"   Found {len(all_symbols)} symbols")
        print()
        
    except Exception as e:
        print(f"❌ Failed to fetch watchlist: {e}")
        return 1
    
    # Run predictions
    print("🔮 Running predictions...")
    print()
    
    success = 0
    failed = 0
    
    for i, symbol in enumerate(all_symbols, 1):
        try:
            print(f"   [{i:2d}/{len(all_symbols)}] {symbol:8s} ", end="", flush=True)
            
            resp = requests.post(
                f"{BASE_URL}/api/predict/run",
                json={"symbol": symbol},
                timeout=60
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    conf = data.get("confidence", 0) * 100
                    direction = data.get("direction", "FLAT")
                    print(f"✅ {direction:5s} {conf:5.1f}%")
                    success += 1
                else:
                    print(f"❌ FAILED: {data.get('error', 'Unknown')}")
                    failed += 1
            else:
                print(f"❌ HTTP {resp.status_code}")
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)[:40]}")
            failed += 1
        
        # Rate limiting
        if i < len(all_symbols):
            time.sleep(2)  # 2 second delay between predictions
    
    print()
    print("=" * 80)
    print(f"✅ Success: {success}/{len(all_symbols)} ({success/len(all_symbols)*100:.0f}%)")
    print(f"❌ Failed:  {failed}/{len(all_symbols)}")
    print("=" * 80)
    print()
    
    # Check ghost score after predictions
    print("🎯 Checking Ghost Score...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v3/goals/snapshot", timeout=10)
        data = resp.json()
        score = data.get("ghost_score", 0)
        details = data.get("ghost_score_details", {})
        grade = details.get("grade", "?")
        
        components = details.get("components", {})
        data_quality = components.get("data_quality", 0)
        prediction_coverage = components.get("prediction_coverage", 0)
        risk_behavior = components.get("risk_behavior", 0)
        
        print(f"   Score: {score:.1f} (Grade: {grade})")
        print(f"   Data Quality: {data_quality:.1f}")
        print(f"   Prediction Coverage: {prediction_coverage:.1f}")
        print(f"   Risk Behavior: {risk_behavior:.1f}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to fetch ghost score: {e}")
    
    print("=" * 80)
    print("🎉 WARM-UP COMPLETE")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
