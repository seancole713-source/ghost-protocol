#!/usr/bin/env python3
"""
Test production feature extraction by directly hitting Railway deployment.
Compares local vs production feature availability.
"""

import requests
import json
import time

PRODUCTION_URL = "https://ghost-protocol-production.up.railway.app"

def test_prediction(symbol: str):
    """Test /api/predict/run endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {symbol}")
    print('='*60)
    
    url = f"{PRODUCTION_URL}/api/predict/run"
    payload = {"symbol": symbol}
    
    start = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start
    
    if response.status_code != 200:
        print(f"❌ Error: HTTP {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    
    # Extract prediction details
    confidence = data.get("confidence", 0)
    direction = data.get("direction", "UNKNOWN")
    prediction_id = data.get("prediction_id")
    
    print(f"✅ Prediction ID: {prediction_id}")
    print(f"📊 Confidence: {confidence*100:.1f}%")
    print(f"📈 Direction: {direction}")
    print(f"⏱️  Response Time: {elapsed*1000:.0f}ms")
    
    # Get prediction details to see feature availability
    details_url = f"{PRODUCTION_URL}/api/v3/predictions/{prediction_id}"
    details_response = requests.get(details_url)
    
    if details_response.status_code == 200:
        details = details_response.json()
        metadata = details.get("metadata", {})
        features_used = metadata.get("features_used", [])
        feature_availability = metadata.get("feature_availability_pct", 0)
        
        print(f"\n📋 Features Used: {len(features_used)}")
        print(f"📈 Feature Availability: {feature_availability:.1f}%")
        
        if features_used:
            print(f"\n🟢 Available Features ({len(features_used)}):")
            for feature in features_used[:10]:  # Show first 10
                print(f"   - {feature}")
            if len(features_used) > 10:
                print(f"   ... and {len(features_used) - 10} more")

def main():
    symbols = ["MSFT", "AAPL", "NVDA", "SPY", "TSLA"]
    
    print("🚀 Testing Production Feature Extraction Pipeline")
    print(f"🌐 Production URL: {PRODUCTION_URL}")
    print(f"📅 Testing {len(symbols)} symbols")
    
    for symbol in symbols:
        test_prediction(symbol)
        time.sleep(2)  # Rate limit protection
    
    print("\n" + "="*60)
    print("✅ Production Testing Complete")
    print("="*60)

if __name__ == "__main__":
    main()
