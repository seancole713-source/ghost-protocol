#!/usr/bin/env python3
"""
Ghost Provider Diagnostic Tool
Tests all price providers for a given symbol to diagnose failures.
"""

import os
import sys
import time
import requests

# Import Ghost's environment loading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY", "")

def test_polygon(symbol: str):
    """Test Polygon API"""
    print(f"\n{'='*80}")
    print(f"TESTING POLYGON API")
    print(f"{'='*80}")
    
    if not POLYGON_KEY:
        print("❌ POLYGON_KEY not configured")
        return False
    
    print(f"✅ POLYGON_KEY configured (len={len(POLYGON_KEY)})")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/prev?adjusted=true&limit=1&apiKey={POLYGON_KEY}"
    print(f"📡 Testing URL: {url[:80]}...")
    
    try:
        start = time.time()
        response = requests.get(url, timeout=10)
        elapsed = time.time() - start
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                price = results[0].get("c", 0)
                print(f"✅ SUCCESS: Price = ${price}")
                return True
            else:
                print(f"❌ FAIL: No results in response")
                print(f"📄 Response: {data}")
                return False
        elif response.status_code == 429:
            print(f"❌ FAIL: Rate limited (429)")
            return False
        elif response.status_code == 403:
            print(f"❌ FAIL: Forbidden (403) - API key invalid or no permissions")
            return False
        else:
            print(f"❌ FAIL: HTTP {response.status_code}")
            print(f"📄 Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_alphavantage(symbol: str):
    """Test Alpha Vantage API"""
    print(f"\n{'='*80}")
    print(f"TESTING ALPHA VANTAGE API")
    print(f"{'='*80}")
    
    if not ALPHAVANTAGE_KEY:
        print("❌ ALPHAVANTAGE_KEY not configured")
        return False
    
    print(f"✅ ALPHAVANTAGE_KEY configured (len={len(ALPHAVANTAGE_KEY)})")
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol.upper()}&apikey={ALPHAVANTAGE_KEY}"
    print(f"📡 Testing URL: {url[:80]}...")
    
    try:
        start = time.time()
        response = requests.get(url, timeout=10)
        elapsed = time.time() - start
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            global_quote = data.get("Global Quote", {})
            price = global_quote.get("05. price")
            
            if price:
                print(f"✅ SUCCESS: Price = ${price}")
                return True
            else:
                print(f"❌ FAIL: No price in Global Quote")
                print(f"📄 Response: {data}")
                return False
        elif response.status_code == 429:
            print(f"❌ FAIL: Rate limited (429)")
            return False
        else:
            print(f"❌ FAIL: HTTP {response.status_code}")
            print(f"📄 Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_yfinance(symbol: str):
    """Test yfinance library"""
    print(f"\n{'='*80}")
    print(f"TESTING YFINANCE LIBRARY")
    print(f"{'='*80}")
    
    try:
        import yfinance as yf
        print(f"✅ yfinance library imported")
        
        start = time.time()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        elapsed = time.time() - start
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price:
            print(f"✅ SUCCESS: Price = ${price}")
            return True
        else:
            print(f"❌ FAIL: No price in ticker.info")
            print(f"📄 Available keys: {list(info.keys())[:10]}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_ghost_api(symbol: str, base_url: str):
    """Test Ghost's prediction endpoint"""
    print(f"\n{'='*80}")
    print(f"TESTING GHOST PREDICTION API")
    print(f"{'='*80}")
    
    url = f"{base_url}/api/predict/run"
    print(f"📡 Testing URL: {url}")
    print(f"📦 Payload: {{'symbol': '{symbol}'}}")
    
    try:
        start = time.time()
        response = requests.post(
            url,
            json={"symbol": symbol},
            timeout=60
        )
        elapsed = time.time() - start
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                confidence = data.get("confidence", 0) * 100
                direction = data.get("direction", "UNKNOWN")
                print(f"✅ SUCCESS: {direction} at {confidence:.1f}% confidence")
                return True
            else:
                print(f"❌ FAIL: {data.get('error', 'Unknown error')}")
                return False
        elif response.status_code == 404:
            print(f"❌ FAIL: 404 Not Found - Unable to fetch price")
            print(f"📄 Response: {response.text[:200]}")
            return False
        elif response.status_code == 503:
            print(f"❌ FAIL: 503 Service Unavailable - Configuration issues")
            print(f"📄 Response: {response.text[:200]}")
            return False
        else:
            print(f"❌ FAIL: HTTP {response.status_code}")
            print(f"📄 Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_providers.py <SYMBOL> [production|localhost]")
        print("Example: python test_providers.py AAPL production")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    env = sys.argv[2] if len(sys.argv) > 2 else "production"
    
    base_url = "https://ghost-protocol-production.up.railway.app" if env == "production" else "http://localhost:8080"
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    GHOST PROVIDER DIAGNOSTIC TOOL                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Symbol: {symbol}
Environment: {env}
Base URL: {base_url}

Environment Variables:
- POLYGON_API_KEY: {'✅ SET' if POLYGON_KEY else '❌ MISSING'}
- ALPHAVANTAGE_API_KEY: {'✅ SET' if ALPHAVANTAGE_KEY else '❌ MISSING'}
""")
    
    results = {
        "polygon": test_polygon(symbol),
        "alphavantage": test_alphavantage(symbol),
        "yfinance": test_yfinance(symbol),
        "ghost_api": test_ghost_api(symbol, base_url)
    }
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    for provider, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{provider.upper():20s} {status}")
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\n{'='*80}")
    print(f"RESULT: {success_count}/{total_count} providers working ({success_count/total_count*100:.0f}%)")
    print(f"{'='*80}")
    
    if results["ghost_api"]:
        print("\n🎉 Ghost prediction endpoint is working!")
    else:
        print("\n⚠️  Ghost prediction endpoint is failing!")
        print("\nDiagnostic steps:")
        if not results["polygon"] and not results["alphavantage"]:
            print("1. Both paid providers failed - check API keys in Railway")
        if not results["yfinance"]:
            print("2. Free fallback (yfinance) also failed - network issue?")
        if not results["ghost_api"]:
            print("3. Ghost API returning 404 - price fetch is failing")
    
    sys.exit(0 if results["ghost_api"] else 1)

if __name__ == "__main__":
    main()
