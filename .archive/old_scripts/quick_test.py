import requests
import json
import time

BASE_URL = "https://ghost-sniper-bot-seancole713-production.up.railway.app"

def test_deployment():
    print("🔍 Testing Ghost Prediction Wiring Fix")
    print("=" * 50)
    print()
    
    # Test 1: Health check
    print("1️⃣ Testing /api/health...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   ✅ {json.dumps(r.json(), indent=2)}")
        else:
            print(f"   ❌ Failed: {r.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 2: Debug endpoint (before)
    print("2️⃣ Testing /api/debug/predictions (before)...")
    try:
        r = requests.get(f"{BASE_URL}/api/debug/predictions", timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Count: {data.get('count')}")
            print(f"   Keys: {data.get('keys')}")
        else:
            print(f"   ❌ Failed: {r.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 3: Create prediction
    print("3️⃣ Creating prediction for WOLF...")
    try:
        r = requests.post(
            f"{BASE_URL}/api/predict/run",
            json={"symbol": "WOLF"},
            timeout=30
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   ✅ Prediction ID: {data.get('prediction_id')}")
            print(f"   ✅ Direction: {data.get('direction')}")
            print(f"   ✅ Confidence: {data.get('confidence')}")
        else:
            print(f"   ❌ Failed: {r.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 4: Debug endpoint (after)
    print("4️⃣ Testing /api/debug/predictions (after)...")
    try:
        r = requests.get(f"{BASE_URL}/api/debug/predictions", timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Count: {data.get('count')}")
            print(f"   Keys: {data.get('keys')}")
            if data.get('store'):
                print(f"   ✅ Store has data!")
                for sym, pred in data['store'].items():
                    print(f"      {sym}: {pred.get('direction')} @ {pred.get('confidence')}")
            else:
                print(f"   ❌ Store is empty!")
        else:
            print(f"   ❌ Failed: {r.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 5: Cockpit snapshot
    print("5️⃣ Testing /api/cockpit/snapshot...")
    try:
        r = requests.get(f"{BASE_URL}/api/cockpit/snapshot", timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            timestamp = data.get('timestamp')
            predictions = data.get('predictions', {})
            stocks = predictions.get('stocks', [])
            print(f"   Timestamp: {timestamp} ({'NULL' if timestamp is None else 'OK'})")
            print(f"   Stocks: {len(stocks)} predictions")
            if stocks:
                print(f"   ✅ Predictions populated!")
                for pred in stocks[:3]:
                    print(f"      {pred.get('symbol')}: {pred.get('direction')} @ {pred.get('confidence')}%")
            else:
                print(f"   ❌ No predictions!")
        else:
            print(f"   ❌ Failed: {r.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test 6: Cockpit
    print("6️⃣ Testing /api/cockpit...")
    try:
        r = requests.get(f"{BASE_URL}/api/cockpit", timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            predictions = data.get('predictions')
            print(f"   Predictions: {type(predictions).__name__}")
            if predictions:
                print(f"   ✅ {len(predictions)} symbols in store")
                for sym, pred in list(predictions.items())[:3]:
                    print(f"      {sym}: {pred.get('direction')} @ {pred.get('confidence')}")
            else:
                print(f"   ❌ Predictions is {predictions}")
        else:
            print(f"   ❌ Failed: {r.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    print("=" * 50)
    print("✅ Test suite complete!")

if __name__ == "__main__":
    # Wait a bit for deployment
    print("⏳ Waiting 60 seconds for Railway deployment...")
    time.sleep(60)
    print()
    test_deployment()
