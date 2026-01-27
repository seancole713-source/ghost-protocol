#!/usr/bin/env python3
"""
Test script to validate prediction-to-cockpit wiring fix.
Tests the data flow: /api/predict/run → in-memory store → /api/cockpit endpoints
"""

import requests
import json
import time
from typing import Any

# Production URL
BASE_URL = "https://ghost-sniper-bot-seancole713-production.up.railway.app"

def test_predict_run() -> dict[str, Any]:
    """Test /api/predict/run endpoint - should create prediction and store in memory"""
    print("=" * 60)
    print("TEST 1: /api/predict/run")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/predict/run"
    payload = {"symbol": "WOLF"}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"   Prediction ID: {data.get('prediction_id')}")
            print(f"   Symbol: {data.get('symbol')}")
            print(f"   Confidence: {data.get('confidence')}")
            print(f"   Direction: {data.get('direction')}")
            print(f"   Run At: {data.get('run_at')}")
            return data
        else:
            print(f"❌ Failed: {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def test_cockpit_snapshot() -> dict[str, Any]:
    """Test /api/cockpit/snapshot - should show predictions.stocks populated"""
    print("\n" + "=" * 60)
    print("TEST 2: /api/cockpit/snapshot")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/cockpit/snapshot"
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            timestamp = data.get('timestamp')
            predictions = data.get('predictions', {})
            stocks = predictions.get('stocks', [])
            
            print(f"   Timestamp: {timestamp} ({'NULL' if timestamp is None else 'OK'})")
            print(f"   Predictions.stocks: {len(stocks)} items")
            
            if timestamp is None:
                print("❌ FAIL: timestamp is null")
            elif not stocks:
                print("❌ FAIL: predictions.stocks is empty")
            else:
                print("✅ SUCCESS: Predictions populated!")
                for pred in stocks[:3]:  # Show first 3
                    print(f"     - {pred.get('symbol')}: {pred.get('direction')} "
                          f"({pred.get('confidence')}% conf)")
            
            return data
        else:
            print(f"❌ Failed: {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def test_cockpit() -> dict[str, Any]:
    """Test /api/cockpit - should expose predictions field"""
    print("\n" + "=" * 60)
    print("TEST 3: /api/cockpit")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/cockpit"
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions')
            
            print(f"   Predictions field: {'NULL' if predictions is None else f'{len(predictions)} symbols'}")
            
            if predictions is None:
                print("❌ FAIL: predictions field is null")
            elif not predictions:
                print("⚠️  WARN: predictions field exists but empty")
            else:
                print("✅ SUCCESS: Predictions exposed!")
                for symbol, pred in list(predictions.items())[:3]:  # Show first 3
                    print(f"     - {symbol}: {pred.get('direction')} "
                          f"({pred.get('confidence')} conf)")
            
            return data
        else:
            print(f"❌ Failed: {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def main():
    print("\n🧪 GHOST PREDICTION WIRING TEST")
    print(f"Target: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Create a prediction
    predict_result = test_predict_run()
    
    # Wait for in-memory store to update
    time.sleep(2)
    
    # Step 2: Check cockpit/snapshot
    snapshot_result = test_cockpit_snapshot()
    
    # Step 3: Check cockpit
    cockpit_result = test_cockpit()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    
    if predict_result.get('ok'):
        print("✅ Prediction creation: PASS")
    else:
        print("❌ Prediction creation: FAIL")
        all_passed = False
    
    snapshot_preds = snapshot_result.get('predictions', {}).get('stocks', [])
    snapshot_ts = snapshot_result.get('timestamp')
    if snapshot_ts is not None and snapshot_preds:
        print("✅ Cockpit snapshot: PASS (predictions populated, timestamp non-null)")
    else:
        print("❌ Cockpit snapshot: FAIL")
        all_passed = False
    
    cockpit_preds = cockpit_result.get('predictions')
    if cockpit_preds is not None:
        print("✅ Cockpit predictions: PASS (field exposed)")
    else:
        print("❌ Cockpit predictions: FAIL")
        all_passed = False
    
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "❌ SOME TESTS FAILED"))
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
